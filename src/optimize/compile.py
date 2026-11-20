"""DSPy prompt compilation pipeline using BootstrapFewShot teleprompter."""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from rich.console import Console
from rich.table import Table

from src.eval.llm_judge import LLMJudge
from src.eval.runner import _get_git_sha
from src.optimize.dspy_program import (
    RAGModule,
    configure_dspy_lm,
    load_compiled_program,
    resolve_dspy_provider,
    sum_history_tokens,
)
from src.pipeline.ingestion import DocumentChunk, chunk_documents, load_raw_documents
from src.utils.config import ChunkingConfig


def build_chunk_map(corpus_dir: str | Path) -> dict[str, DocumentChunk]:
    """Chunks corpus documents and builds a lookup mapping of chunk_id -> DocumentChunk."""
    raw_docs = load_raw_documents(corpus_dir, "*.md")
    chunk_cfg = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=50)
    chunks = chunk_documents(raw_docs, chunk_cfg)
    return {c.chunk_id: c for c in chunks}


def resolve_oracle_chunk_ids(
    gt_spans: list[dict[str, Any]],
    gt_context_ids: list[str],
    chunk_map: dict[str, DocumentChunk] | dict[str, str],
) -> list[str]:
    """Resolves current chunk IDs covering ground_truth_spans (falls back to gt_context_ids)."""
    if not gt_spans:
        return list(gt_context_ids)

    sample_val = next(iter(chunk_map.values()), None)
    if sample_val is None or not hasattr(sample_val, "metadata"):
        return list(gt_context_ids)

    matched_ids: list[str] = []
    seen: set[str] = set()
    for sp in gt_spans:
        target_doc = sp.get("doc")
        s_start = int(sp.get("char_start", 0))
        s_end = int(sp.get("char_end", 0))
        for cid, chunk in chunk_map.items():
            if not isinstance(chunk, DocumentChunk):
                continue
            c_doc = chunk.metadata.get("doc", cid.split("_c")[0])
            if c_doc != target_doc:
                continue
            if s_end == -1:
                if cid not in seen:
                    seen.add(cid)
                    matched_ids.append(cid)
                continue
            c_start = int(chunk.metadata.get("char_start", 0))
            c_end = int(chunk.metadata.get("char_end", len(chunk.content)))
            if min(c_end, s_end) - max(c_start, s_start) > 0 and cid not in seen:
                seen.add(cid)
                matched_ids.append(cid)

    return matched_ids if matched_ids else list(gt_context_ids)


def format_context_for_example(
    chunk_ids: list[str], chunk_map: dict[str, DocumentChunk] | dict[str, str]
) -> str:
    """Formats context passages with chunk ID headers."""
    if not chunk_ids:
        return "No supporting context available."
    pieces = []
    for cid in chunk_ids:
        entry = chunk_map.get(cid)
        content = entry.content if isinstance(entry, DocumentChunk) else str(entry or "")
        if content:
            pieces.append(f"[{cid}]\n{content}")
    return "\n\n".join(pieces) if pieces else "No supporting context available."


def load_dspy_splits(
    golden_qa_path: str | Path,
    chunk_map: dict[str, DocumentChunk] | dict[str, str],
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Loads golden dataset and creates dspy.Example objects for train, dev, and quarantined test splits."""
    path = Path(golden_qa_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden QA dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    train_examples: list[dspy.Example] = []
    dev_examples: list[dspy.Example] = []
    test_examples: list[dspy.Example] = []

    for item in records:
        q_id = item["id"]
        query = item["query"]
        gt_answer = item["ground_truth_answer"]
        gt_context_ids = item.get("ground_truth_context_ids", [])
        gt_spans = item.get("ground_truth_spans", [])
        split = item.get("split", "train")

        oracle_ids = resolve_oracle_chunk_ids(gt_spans, gt_context_ids, chunk_map)
        context_str = format_context_for_example(oracle_ids, chunk_map)
        citations_str = ", ".join(oracle_ids) if oracle_ids else "none"

        ex = dspy.Example(
            id=q_id,
            question=query,
            context=context_str,
            answer=gt_answer,
            citations=citations_str,
            query_type=item.get("query_type", ""),
        ).with_inputs("question", "context")

        if split == "train":
            train_examples.append(ex)
        elif split == "dev":
            dev_examples.append(ex)
        elif split == "test":
            test_examples.append(ex)

    return train_examples, dev_examples, test_examples


def rag_optimization_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Optimization teleprompter metric balancing correctness, citations, and refusal adherence."""
    gold_ans = gold.answer.strip().lower()
    pred_ans = getattr(pred, "answer", "").strip().lower()

    if not pred_ans:
        return 0.0

    # Handle refusal alignment
    is_refusal_expected = "insufficient information" in gold_ans
    is_refusal_predicted = "insufficient information" in pred_ans

    if is_refusal_expected:
        return 1.0 if is_refusal_predicted else 0.0

    if is_refusal_predicted and not is_refusal_expected:
        # Penalize false refusal when answer exists
        return 0.0

    # Token overlap for factual alignment
    gold_tokens = set(re.findall(r"\w+", gold_ans))
    pred_tokens = set(re.findall(r"\w+", pred_ans))

    stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "for", "with", "of", "on"}
    core_gold = {t for t in gold_tokens if t not in stopwords and len(t) > 2}
    if not core_gold:
        return 1.0

    overlap = len(core_gold.intersection(pred_tokens)) / len(core_gold)

    # Citation bonus (conditional on meaningful overlap)
    pred_cits = getattr(pred, "citations", "").strip().lower()
    citation_score = 0.2 if (pred_cits not in ("", "none") and overlap >= 0.5) else 0.0

    total_score = min(1.0, (overlap * 0.8) + citation_score)
    return round(total_score, 3)


def demo_answer_text(demo: Any) -> str:
    """Reads the answer field from a DSPy demo example or a saved dict."""
    if isinstance(demo, dict):
        return str(demo.get("answer", ""))
    return str(getattr(demo, "answer", ""))


def demo_is_broken(demo: Any) -> bool:
    """True when a demo answer is empty, starts mid-sentence, or is a markdown heading."""
    answer = demo_answer_text(demo).lstrip()
    if not answer:
        return True
    return answer[:1].islower() or answer.startswith("#")


def evaluate_program_on_split(
    program: dspy.Module,
    examples: list[dspy.Example],
    judge: LLMJudge,
    retriever: Any | None = None,
) -> dict[str, float]:
    """Evaluates a DSPy module on answerable examples (correctness/faithfulness) and traps (refusal)."""
    correctness_scores = []
    faithfulness_scores = []
    latencies = []
    multi_hop_scores = []
    negative_total = 0
    negative_refused = 0
    parse_failures = 0

    for ex in examples:
        is_negative = "insufficient information" in ex.answer.lower()
        if retriever is not None:
            ret_chunks = retriever.retrieve(ex.question).chunks
            ctx_str = "\n\n".join([f"[{c.chunk_id}]\n{c.content}" for c in ret_chunks])
        else:
            ctx_str = ex.context

        start_t = time.perf_counter()
        try:
            pred = program(question=ex.question, context=ctx_str)
            answer = getattr(pred, "answer", "") or ""
        except Exception as exc:
            from dspy.utils.exceptions import AdapterParseError

            if not isinstance(exc, AdapterParseError):
                raise
            answer = ""
            parse_failures += 1
            latencies.append((time.perf_counter() - start_t) * 1000.0)
            if is_negative:
                negative_total += 1
            else:
                correctness_scores.append(0.0)
                faithfulness_scores.append(0.0)
                if getattr(ex, "query_type", "") == "multi_hop":
                    multi_hop_scores.append(0.0)
            continue
        latencies.append((time.perf_counter() - start_t) * 1000.0)

        if is_negative:
            negative_total += 1
            if "insufficient information" in answer.lower():
                negative_refused += 1
            continue

        if judge.mode == "live" and judge.is_decision_model:
            faith_eval, corr_eval, _usage = judge.evaluate_sample(
                ex.question, ctx_str, answer, ex.answer
            )
        else:
            corr_eval = judge.evaluate_correctness(ex.question, answer, ex.answer)
            faith_eval = judge.evaluate_faithfulness(ex.question, ctx_str, answer)

        correctness_scores.append(corr_eval.score)
        faithfulness_scores.append(faith_eval.score)
        if getattr(ex, "query_type", "") == "multi_hop":
            multi_hop_scores.append(corr_eval.score)

    return {
        "n_answerable": float(len(correctness_scores)),
        "answer_correctness": (
            sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        ),
        "faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        ),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "multi_hop_correctness": (
            sum(multi_hop_scores) / len(multi_hop_scores) if multi_hop_scores else 0.0
        ),
        "n_multi_hop": float(len(multi_hop_scores)),
        "negative_refusal_rate": (negative_refused / negative_total if negative_total else 0.0),
        "n_negative": float(negative_total),
        "parse_failures": float(parse_failures),
    }


DEFAULT_OPENROUTER_JUDGE = "perplexity/pplx-decider-flash-v1.2"
OPENROUTER_COMPILED_PROGRAM = "src/optimize/compiled_dspy_rag.openrouter.json"


def _demo_field(demo: Any, name: str) -> str:
    if isinstance(demo, dict):
        return str(demo.get(name, ""))
    return str(getattr(demo, name, ""))


def _write_live_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def compile_dspy_pipeline(
    golden_qa_path: str | Path = "data/golden_qa.json",
    corpus_dir: str | Path = "data/raw",
    output_path: str | Path = "src/optimize/compiled_dspy_rag.json",
    mode: str = "offline",
    model_name: str = "gpt-4o-mini",
    provider: str | None = None,
    judge_model: str | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_candidate_programs: int = 4,
    max_tokens: int = 4000,
    summary_path: str | Path | None = None,
    eval_only: bool = False,
) -> RAGModule:
    """Compiles RAG prompt using BootstrapFewShot on train/dev splits strictly quarantining test split."""
    console = Console()
    resolved_provider = resolve_dspy_provider(provider, model_name) if mode == "live" else "offline"
    console.print(
        f"\n[bold green]Starting DSPy Prompt Compilation ({mode} mode, "
        f"provider: {resolved_provider}, model: {model_name})[/bold green]"
    )

    # 1. Build corpus chunk lookup
    chunk_map = build_chunk_map(corpus_dir)
    console.print(f"Loaded {len(chunk_map)} corpus chunks from {corpus_dir}")

    # 2. Load splits
    train_exs, dev_exs, test_exs = load_dspy_splits(golden_qa_path, chunk_map)
    console.print(
        f"Splits loaded: [cyan]{len(train_exs)} train[/cyan], [cyan]{len(dev_exs)} dev[/cyan], "
        f"[yellow]{len(test_exs)} test (QUARANTINED)[/yellow]"
    )

    # 3. Configure DSPy Language Model
    lm = configure_dspy_lm(
        mode=mode,
        model_name=model_name,
        provider=provider,
        max_tokens=max_tokens,
    )

    # 4. Instantiate Baseline Program. compile() copies the student; keep a fresh
    # zero-shot module so the holdout comparison is not the optimized program twice.
    baseline_module = RAGModule()

    out_file = Path(output_path)
    if eval_only:
        compiled_module = load_compiled_program(out_file)
        kept = list(getattr(compiled_module.generate, "demos", []) or [])
        dropped = 0
        candidate_scores = []
        console.print(
            f"[bold]Loaded compiled program from {out_file} ({len(kept)} demos). "
            "Skipping teleprompter.[/bold]"
        )
    else:
        # Seeds include zero-shot (-3), labeled few-shot (-2), and bootstrapped programs.
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=rag_optimization_metric,
            metric_threshold=0.6,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_candidate_programs=num_candidate_programs,
            num_threads=1,
        )
        console.print(
            "\n[bold]Running BootstrapFewShotWithRandomSearch teleprompter optimization...[/bold]"
        )
        console.print(
            "[dim]The search metric is token overlap on train/dev. "
            "Published scores, in live mode, come from the holdout judge.[/dim]"
        )
        compiled_module = teleprompter.compile(
            student=RAGModule(),
            trainset=train_exs,
            valset=dev_exs,
        )
        candidate_scores = [
            {"seed": item.get("seed"), "dev_lexical_score": item.get("score")}
            for item in (getattr(compiled_module, "candidate_programs", None) or [])
            if isinstance(item, dict)
        ]
        if hasattr(compiled_module, "candidate_programs"):
            delattr(compiled_module, "candidate_programs")

        demos = list(getattr(compiled_module.generate, "demos", []) or [])
        kept = [demo for demo in demos if not demo_is_broken(demo)]
        dropped = len(demos) - len(kept)
        compiled_module.generate.demos = kept
        if dropped:
            console.print(
                f"[yellow]Dropped {dropped} demo(s) that were empty, mid-sentence, or a heading.[/yellow]"
            )
        if not kept:
            console.print(
                "[yellow]Selected program has no usable demos. The holdout comparison is "
                "zero-shot versus zero-shot.[/yellow]"
            )

        out_file.parent.mkdir(parents=True, exist_ok=True)
        compiled_module.save(str(out_file))
        saved_text = out_file.read_text(encoding="utf-8")
        if any(marker in saved_text for marker in ("sk-or-", "sk-proj-", "Bearer ")):
            out_file.unlink(missing_ok=True)
            raise RuntimeError(f"Refusing to keep {out_file}: saved program contains a secret.")
        console.print(
            f"[bold green]Successfully saved compiled DSPy program to {out_file}[/bold green]"
        )

    # 7. Evaluate on Quarantined Test Split (Live mode only)
    if mode == "live":
        from src.pipeline.indexing import VectorIndexer
        from src.pipeline.retriever import RAGRetriever
        from src.utils.config import PipelineConfig

        config = PipelineConfig.from_yaml("configs/hybrid.yaml")
        indexer = VectorIndexer(config.embedding, config.vector_store)
        retriever = RAGRetriever(indexer, config.retriever)

        resolved_judge = judge_model
        if not resolved_judge:
            resolved_judge = (
                DEFAULT_OPENROUTER_JUDGE if resolved_provider == "openrouter" else model_name
            )
        judge_provider = "openrouter" if resolved_provider == "openrouter" else "openai"
        judge = LLMJudge(model_name=resolved_judge, mode=mode, provider=judge_provider)
        history_before_eval = len(getattr(lm, "history", None) or [])
        console.print(
            f"\n[bold]Evaluating quarantined test split ({len(test_exs)} items) "
            f"with judge {resolved_judge}. Train and dev stay out of this table.[/bold]"
        )

        base_oracle = evaluate_program_on_split(baseline_module, test_exs, judge)
        comp_oracle = evaluate_program_on_split(compiled_module, test_exs, judge)

        base_retrieved = evaluate_program_on_split(
            baseline_module, test_exs, judge, retriever=retriever
        )
        comp_retrieved = evaluate_program_on_split(
            compiled_module, test_exs, judge, retriever=retriever
        )

        table = Table(
            title="DSPy Generation Prompt Optimization Benchmark (Test Split Delta)",
            header_style="bold magenta",
        )
        table.add_column("Condition / Metric", style="cyan")
        table.add_column("Baseline (Zero-Shot)", justify="right", style="dim")
        table.add_column("Compiled (DSPy Few-Shot)", justify="right", style="green")
        table.add_column("Delta", justify="right", style="bold")
        table.add_column("Status", justify="center")

        def add_delta_row(label: str, b_val: float, c_val: float, is_pct: bool = True) -> None:
            delta = c_val - b_val
            fmt = ".1%" if is_pct else ".1f"
            if not is_pct:
                status = "INFO"
            elif delta > 0.005:
                status = "[green]IMPROVED[/green]"
            elif delta < -0.005:
                status = "[red]REGRESSED[/red]"
            else:
                status = "[yellow]NEUTRAL[/yellow]"
            table.add_row(
                label,
                f"{b_val:{fmt}}",
                f"{c_val:{fmt}}",
                f"{'+' if delta >= 0 else ''}{delta:{fmt}}",
                status,
            )

        add_delta_row(
            "Correctness (Oracle Context)",
            base_oracle["answer_correctness"],
            comp_oracle["answer_correctness"],
        )
        add_delta_row(
            "Correctness (Retrieved Context)",
            base_retrieved["answer_correctness"],
            comp_retrieved["answer_correctness"],
        )
        if base_retrieved["n_multi_hop"]:
            add_delta_row(
                f"Multi-Hop Correctness (Retrieved, n={int(base_retrieved['n_multi_hop'])})",
                base_retrieved["multi_hop_correctness"],
                comp_retrieved["multi_hop_correctness"],
            )
        if base_retrieved["n_negative"]:
            add_delta_row(
                f"Negative Refusal (Retrieved, n={int(base_retrieved['n_negative'])})",
                base_retrieved["negative_refusal_rate"],
                comp_retrieved["negative_refusal_rate"],
            )
        add_delta_row(
            "Faithfulness (Oracle Context)",
            base_oracle["faithfulness"],
            comp_oracle["faithfulness"],
        )
        add_delta_row(
            "Faithfulness (Retrieved Context)",
            base_retrieved["faithfulness"],
            comp_retrieved["faithfulness"],
        )
        add_delta_row(
            "Mean Latency (ms)",
            base_retrieved["mean_latency_ms"],
            comp_retrieved["mean_latency_ms"],
            is_pct=False,
        )

        console.print(table)

        compile_prompt, compile_completion = sum_history_tokens(lm, 0)
        eval_prompt, eval_completion = sum_history_tokens(lm, history_before_eval)
        compile_prompt -= eval_prompt
        compile_completion -= eval_completion
        from src.llm.pricing import estimate_cost_eur

        summary = {
            "mode": "live",
            "git_sha": _get_git_sha(),
            "provider": resolved_provider,
            "model": model_name,
            "judge_model": resolved_judge,
            "n_train": len(train_exs),
            "n_dev": len(dev_exs),
            "n_test": len(test_exs),
            "n_demos": len(kept),
            "n_demos_dropped": dropped,
            "candidate_dev_lexical_scores": candidate_scores,
            "test_split_quarantined": True,
            "oracle_context_source": "ground_truth_spans",
            "optimizer_metric": "lexical_overlap_plus_citation",
            "published_judge": resolved_judge,
            "baseline_oracle": base_oracle,
            "compiled_oracle": comp_oracle,
            "baseline_retrieved": base_retrieved,
            "compiled_retrieved": comp_retrieved,
            "compile_prompt_tokens": compile_prompt,
            "compile_completion_tokens": compile_completion,
            "eval_prompt_tokens": eval_prompt,
            "eval_completion_tokens": eval_completion,
            "generator_cost_eur": round(
                estimate_cost_eur(model_name, compile_prompt, compile_completion)
                + estimate_cost_eur(model_name, eval_prompt, eval_completion),
                6,
            ),
            "compiled_program": str(out_file),
            "reasoning": "enabled",
            "parse_failures": {
                "baseline_oracle": base_oracle["parse_failures"],
                "compiled_oracle": comp_oracle["parse_failures"],
                "baseline_retrieved": base_retrieved["parse_failures"],
                "compiled_retrieved": comp_retrieved["parse_failures"],
            },
        }
        summary_file = Path(summary_path or "reports/live/dspy-compiled-rag/compile_summary.json")
        _write_live_summary(summary_file, summary)
        _write_live_summary(Path("results/dspy_openrouter_summary.json"), summary)
        console.print(f"[bold green]Wrote live DSPy summary to {summary_file}[/bold green]")
    else:
        console.print(
            "Offline mode: compilation smoke test only. No generation metrics are reported because the offline LM is a deterministic mock."
        )

    # Log Prompt Diff / Demonstration Exemplars
    demos = getattr(compiled_module.generate, "demos", [])
    console.print(f"\n[bold]Compiled Exemplars ({len(demos)} few-shot demos):[/bold]")
    for i, demo in enumerate(demos, start=1):
        question = _demo_field(demo, "question")
        answer = _demo_field(demo, "answer")
        citations = _demo_field(demo, "citations")
        console.print(f"[cyan]Demo {i}:[/cyan] [dim]Q: {question[:60]}...[/dim]")
        console.print(f"        [green]Ans: {answer[:70]}... | Citations: {citations}[/green]")

    return compiled_module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile DSPy RAG Generation Prompts using BootstrapFewShot"
    )
    parser.add_argument(
        "--golden-qa", default="data/golden_qa.json", help="Path to golden QA dataset"
    )
    parser.add_argument(
        "--corpus-dir", default="data/raw", help="Path to raw markdown corpus directory"
    )
    parser.add_argument(
        "--output",
        default="src/optimize/compiled_dspy_rag.json",
        help="Output path for compiled program",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Execution mode (offline or live)",
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v4.1-flash",
        help="Generator model. Live OpenRouter examples: deepseek/deepseek-v4.1-flash, qwen/qwen3.7-flash.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Live provider. Inferred from the model slug when omitted (deepseek/qwen -> openrouter).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Holdout judge. Defaults to perplexity/pplx-decider-flash-v1.2 for OpenRouter.",
    )
    parser.add_argument(
        "--max-demos", type=int, default=4, help="Maximum few-shot bootstrapped demos"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=4,
        help="Random-search candidates, in addition to the zero-shot and labeled seeds.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4000, help="Max completion tokens per DSPy call."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Score an already compiled program on the quarantined test split.",
    )

    args = parser.parse_args()
    output = args.output
    resolved = resolve_dspy_provider(args.provider, args.model)
    if (
        args.mode == "live"
        and resolved == "openrouter"
        and output == "src/optimize/compiled_dspy_rag.json"
    ):
        output = OPENROUTER_COMPILED_PROGRAM

    compile_dspy_pipeline(
        golden_qa_path=args.golden_qa,
        corpus_dir=args.corpus_dir,
        output_path=output,
        mode=args.mode,
        model_name=args.model,
        provider=args.provider,
        judge_model=args.judge_model,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
        num_candidate_programs=args.candidates,
        max_tokens=args.max_tokens,
        eval_only=args.eval_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

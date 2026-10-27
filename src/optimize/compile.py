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
from src.optimize.dspy_program import (
    RAGModule,
    configure_dspy_lm,
)
from src.pipeline.ingestion import chunk_documents, load_raw_documents
from src.utils.config import ChunkingConfig


def build_chunk_map(corpus_dir: str | Path) -> dict[str, str]:
    """Chunks corpus documents and builds a lookup mapping of chunk_id -> content."""
    raw_docs = load_raw_documents(corpus_dir, "*.md")
    chunk_cfg = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=50)
    chunks = chunk_documents(raw_docs, chunk_cfg)
    return {c.chunk_id: c.content for c in chunks}


def format_context_for_example(chunk_ids: list[str], chunk_map: dict[str, str]) -> str:
    """Formats context passages with chunk ID headers."""
    if not chunk_ids:
        return "No supporting context available."
    pieces = []
    for cid in chunk_ids:
        content = chunk_map.get(cid, "")
        if content:
            pieces.append(f"[{cid}]\n{content}")
    return "\n\n".join(pieces) if pieces else "No supporting context available."


def load_dspy_splits(
    golden_qa_path: str | Path,
    chunk_map: dict[str, str],
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
        split = item.get("split", "train")

        context_str = format_context_for_example(gt_context_ids, chunk_map)
        citations_str = ", ".join(gt_context_ids) if gt_context_ids else "none"

        ex = dspy.Example(
            id=q_id,
            question=query,
            context=context_str,
            answer=gt_answer,
            citations=citations_str,
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


def evaluate_program_on_split(
    program: dspy.Module,
    examples: list[dspy.Example],
    judge: LLMJudge,
    retriever: Any | None = None,
) -> dict[str, float]:
    """Evaluates a DSPy module against examples and calculates correctness, faithfulness, and latency."""
    correctness_scores = []
    faithfulness_scores = []
    latencies = []

    for ex in examples:
        if retriever is not None:
            ret_chunks = retriever.retrieve(ex.question).chunks
            ctx_str = "\n\n".join([f"[{c.chunk_id}]\n{c.content}" for c in ret_chunks])
        else:
            ctx_str = ex.context

        start_t = time.perf_counter()
        pred = program(question=ex.question, context=ctx_str)
        latencies.append((time.perf_counter() - start_t) * 1000.0)

        answer = getattr(pred, "answer", "")

        corr_eval = judge.evaluate_correctness(ex.question, answer, ex.answer)
        faith_eval = judge.evaluate_faithfulness(ex.question, ctx_str, answer)

        correctness_scores.append(corr_eval.score)
        faithfulness_scores.append(faith_eval.score)

    return {
        "answer_correctness": (
            sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        ),
        "faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        ),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def compile_dspy_pipeline(
    golden_qa_path: str | Path = "data/golden_qa.json",
    corpus_dir: str | Path = "data/raw",
    output_path: str | Path = "src/optimize/compiled_dspy_rag.json",
    mode: str = "offline",
    model_name: str = "gpt-4o-mini",
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
) -> RAGModule:
    """Compiles RAG prompt using BootstrapFewShot on train/dev splits strictly quarantining test split."""
    console = Console()
    console.print(
        f"\n[bold green]Starting DSPy Prompt Compilation ({mode} mode, model: {model_name})[/bold green]"
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
    configure_dspy_lm(mode=mode, model_name=model_name)

    # 4. Instantiate Baseline Program
    baseline_module = RAGModule()

    # 5. Compile with BootstrapFewShotWithRandomSearch using train and dev splits
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=rag_optimization_metric,
        metric_threshold=0.6,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_candidate_programs=4,
        num_threads=1,
    )

    console.print(
        "\n[bold]Running BootstrapFewShotWithRandomSearch teleprompter optimization...[/bold]"
    )
    compiled_module = teleprompter.compile(
        student=baseline_module,
        trainset=train_exs,
        valset=dev_exs,
    )

    # Demo sanity gate: prevent mid-sentence or markdown header fragments
    demos = getattr(compiled_module.generate, "demos", [])
    bad = [
        d
        for d in demos
        if str(getattr(d, "answer", d.get("answer", "") if isinstance(d, dict) else ""))[
            :1
        ].islower()
        or str(getattr(d, "answer", d.get("answer", "") if isinstance(d, dict) else "")).startswith(
            "#"
        )
    ]
    if bad:
        raise RuntimeError(
            f"{len(bad)} bootstrapped demos start mid-sentence or heading; refusing to save a broken artifact"
        )

    # 6. Save compiled artifact
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    compiled_module.save(str(out_file))
    console.print(
        f"[bold green]Successfully saved compiled DSPy program to {out_file}[/bold green]"
    )

    # 7. Evaluate on Quarantined Test Split (Live mode only)
    if mode == "live":
        from src.pipeline.indexing import VectorIndexer
        from src.pipeline.retriever import RAGRetriever
        from src.utils.config import load_pipeline_config

        config = load_pipeline_config("configs/hybrid.yaml")
        indexer = VectorIndexer(config.embedding, config.vector_store)
        retriever = RAGRetriever(indexer, config.retriever)

        judge = LLMJudge(model_name=model_name, mode=mode)
        console.print(
            "\n[bold]Evaluating holdout test split (32 items) before and after optimization in live mode...[/bold]"
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
            fmt = ".1%" if is_pct else ".1f ms"
            status = "[green]IMPROVED[/green]" if delta > 0 else "[yellow]NEUTRAL[/yellow]"
            table.add_row(
                label,
                f"{b_val:{fmt}}",
                f"{c_val:{fmt}}",
                f"{'+' if delta >= 0 else ''}{delta:{fmt}}",
                status if is_pct else "INFO",
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
            "Mean Latency",
            base_retrieved["mean_latency_ms"],
            comp_retrieved["mean_latency_ms"],
            is_pct=False,
        )

        console.print(table)
    else:
        console.print(
            "Offline mode: compilation smoke test only. No generation metrics are reported because the offline LM is a deterministic mock."
        )

    # Log Prompt Diff / Demonstration Exemplars
    demos = getattr(compiled_module.generate, "demos", [])
    console.print(f"\n[bold]Compiled Exemplars ({len(demos)} bootstrapped few-shot demos):[/bold]")
    for i, d in enumerate(demos, start=1):
        q = getattr(d, "question", "")
        ans = getattr(d, "answer", "")
        cit = getattr(d, "citations", "")
        console.print(f"[cyan]Demo {i}:[/cyan] [dim]Q: {q[:60]}...[/dim]")
        console.print(f"        [green]Ans: {ans[:70]}... | Citations: {cit}[/green]")

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
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument(
        "--max-demos", type=int, default=4, help="Maximum few-shot bootstrapped demos"
    )

    args = parser.parse_args()

    compile_dspy_pipeline(
        golden_qa_path=args.golden_qa,
        corpus_dir=args.corpus_dir,
        output_path=args.output,
        mode=args.mode,
        model_name=args.model,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

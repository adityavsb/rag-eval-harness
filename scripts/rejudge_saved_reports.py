#!/usr/bin/env python3
"""Re-evaluate saved RAG benchmark generator outputs using Perplexity Decider Flash over OpenRouter /api/alpha/decisions."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# Ensure rag-eval-harness root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.llm_judge import LLMJudge  # noqa: E402
from src.utils.config import load_dotenv  # noqa: E402

load_dotenv()
console = Console()

MODEL_REPORT_MAP = {
    "deepseek": "reports/live/openrouter-tri-model-eval/20261029T210935_99f16f5.json",
    "qwen": "reports/live/openrouter-tri-model-eval/20261029T211925_99f16f5.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved benchmark generator outputs using Perplexity Decider."
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to saved report JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek", "qwen"],
        default=None,
        help="Convenience selector for saved models: 'deepseek' or 'qwen'.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="perplexity/pplx-decider-flash-v1.2",
        help="OpenRouter decision model slug.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of cases to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for updated report JSON.",
    )
    parser.add_argument(
        "--pacing-delay",
        type=float,
        default=0.4,
        help="Pacing delay in seconds between requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.report:
        report_path = Path(args.report)
    elif args.model:
        report_path = REPO_ROOT / MODEL_REPORT_MAP[args.model]
    else:
        console.print("[bold red]Error: Either --report or --model must be provided.[/bold red]")
        sys.exit(1)

    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    if not report_path.exists():
        console.print(f"[bold red]Error: Report file not found at {report_path}[/bold red]")
        sys.exit(1)

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict[str, Any]] = data.get("records", [])
    model_id = data.get("model_id", "unknown")
    total_in_report = len(records)
    console.print(
        f"[bold cyan]Loaded {total_in_report} preserved cases for generator '[yellow]{model_id}[/yellow]' from {report_path.name}[/bold cyan]"
    )

    if args.limit and args.limit > 0:
        records = records[: args.limit]
        console.print(f"[yellow]Limiting re-evaluation to first {len(records)} test cases.[/yellow]")

    # Connect to ChromaDB to reconstruct retrieved context
    import chromadb

    chroma_path = REPO_ROOT / "chroma_db/hybrid"
    client = chromadb.PersistentClient(str(chroma_path))
    collection = client.get_collection("hybrid_collection")
    console.print(
        f"[green]Connected to ChromaDB at {chroma_path} ({collection.count()} chunks available for context reconstruction).[/green]"
    )

    judge = LLMJudge(
        model_name=args.judge_model,
        mode="live",
        provider="openrouter",
    )

    console.print(
        "[bold magenta]Starting live Perplexity Decider evaluation via OpenRouter /api/alpha/decisions...[/bold magenta]"
    )

    total_judge_cost = 0.0
    updated_records: list[dict[str, Any]] = []

    progress_table = Table(title=f"Perplexity Re-Judging Progress: {model_id}")
    progress_table.add_column("#", justify="right", style="cyan", no_wrap=True)
    progress_table.add_column("Query ID", style="bold")
    progress_table.add_column("Type", style="dim")
    progress_table.add_column("Faithfulness", justify="right")
    progress_table.add_column("Correctness", justify="right")
    progress_table.add_column("Grade", justify="right", style="green")
    progress_table.add_column("Cost (USD)", justify="right")

    for idx, rec in enumerate(records):
        q_id = rec["query_id"]
        q_type = rec.get("query_type", "factoid")
        query = rec["query"]
        gen_ans = rec["generated_answer"]
        gt_ans = rec["ground_truth_answer"]
        retrieved_ids = rec.get("retrieved_chunk_ids", [])

        # Reconstruct context string from ChromaDB
        if retrieved_ids:
            res = collection.get(ids=retrieved_ids)
            id_to_doc = dict(zip(res["ids"], res["documents"]))
            context_str = "\n".join(
                [f"[{cid}] {id_to_doc.get(cid, '')}" for cid in retrieved_ids]
            )
        else:
            context_str = ""

        # Call Perplexity Decider
        faith_eval, corr_eval, usage = judge.evaluate_sample(
            query=query,
            context=context_str,
            answer=gen_ans,
            ground_truth=gt_ans,
        )

        cost_item = usage.get("cost", 0.0)
        total_judge_cost += cost_item

        updated_rec = dict(rec)
        updated_rec["faithfulness_score"] = faith_eval.score
        updated_rec["correctness_score"] = corr_eval.score
        updated_rec["rubric_grade"] = corr_eval.rubric_grade
        updated_rec["faithfulness_reasoning"] = faith_eval.reasoning
        updated_rec["correctness_reasoning"] = corr_eval.reasoning
        updated_rec["judge_model"] = args.judge_model
        updated_rec["cost_eur"] = round(rec.get("cost_eur", 0.0) + cost_item, 6)
        updated_records.append(updated_rec)

        console.print(
            f"  [{idx + 1}/{len(records)}] {q_id} ({q_type}) -> "
            f"Faith: {faith_eval.score:.4f} | Corr: {corr_eval.score:.4f} (Grade {corr_eval.rubric_grade}/5) | "
            f"Cost: ${cost_item:.6f}"
        )

        if args.pacing_delay > 0:
            time.sleep(args.pacing_delay)

    # Recompute aggregate metrics
    faith_scores = [r["faithfulness_score"] for r in updated_records]
    corr_scores = [r["correctness_score"] for r in updated_records]
    grades = [r["rubric_grade"] for r in updated_records]

    # Calculate refusal rate on adversarial queries
    neg_queries = [r for r in updated_records if "insufficient information" in r["ground_truth_answer"].lower()]
    neg_refused = [r for r in neg_queries if "insufficient information" in r["generated_answer"].lower()]
    neg_refusal_rate = (len(neg_refused) / len(neg_queries)) if neg_queries else 1.0

    mean_faith = float(sum(faith_scores) / len(faith_scores)) if faith_scores else 0.0
    mean_corr = float(sum(corr_scores) / len(corr_scores)) if corr_scores else 0.0
    mean_grade = float(sum(grades) / len(grades)) if grades else 0.0
    total_cost = sum(r["cost_eur"] for r in updated_records)

    metrics = dict(data.get("metrics", {}))
    metrics["mean_faithfulness"] = round(mean_faith, 4)
    metrics["mean_correctness"] = round(mean_corr, 4)
    metrics["mean_rubric_grade"] = round(mean_grade, 2)
    metrics["negative_refusal_rate"] = round(neg_refusal_rate, 4)
    metrics["judge_model"] = args.judge_model
    metrics["judge_evaluation_mode"] = "openrouter_alpha_decisions"

    updated_data = dict(data)
    updated_data["metrics"] = metrics
    updated_data["total_samples"] = len(updated_records)
    updated_data["cost_eur"] = round(total_cost, 6)
    updated_data["records"] = updated_records
    updated_data["judge_model"] = args.judge_model

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = report_path.stem
        out_dir = REPO_ROOT / "reports/live/openrouter-tri-model-eval"
        out_path = out_dir / f"{stem}_perplexity_judged.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2)

    console.print(f"\n[bold green]✔ Successfully saved updated judged report to {out_path}[/bold green]")

    # Print summary table
    summary_table = Table(title=f"Evaluation Summary: {model_id} (Judged by {args.judge_model})")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold green", justify="right")
    summary_table.add_row("Evaluated Test Cases", str(len(updated_records)))
    summary_table.add_row("Mean Faithfulness (noul)", f"{mean_faith:.4f} ({mean_faith * 100:.1f}%)")
    summary_table.add_row("Mean Correctness (noul)", f"{mean_corr:.4f} ({mean_corr * 100:.1f}%)")
    summary_table.add_row("Mean Rubric Grade (1-5)", f"{mean_grade:.2f} / 5.0")
    summary_table.add_row("Negative Refusal Rate", f"{neg_refusal_rate * 100:.1f}% ({len(neg_refused)}/{len(neg_queries)})")
    summary_table.add_row("Judge Tokens / Cost Added", f"${total_judge_cost:.6f}")
    summary_table.add_row("Total Run Cost (Gen + Judge)", f"€{total_cost:.6f}")

    console.print(summary_table)


if __name__ == "__main__":
    main()

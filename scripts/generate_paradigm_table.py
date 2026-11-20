#!/usr/bin/env python3
"""Generate paradigm comparison tables and canonical aggregate JSON summaries under results/."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure rag-eval-harness root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.retrieval_metrics import evaluate_retrieval  # noqa: E402

console = Console()

OFFLINE_RUNS = [
    ("baseline-naive-rag", "reports/offline/baseline-naive-rag/latest.json"),
    ("chunk-optimized-rag", "reports/offline/chunk-optimized-rag/latest.json"),
    ("rerank-two-stage-rag", "reports/offline/rerank-two-stage-rag/latest.json"),
    ("hybrid-rrf-rag", "reports/offline/hybrid-rrf-rag/latest.json"),
    ("hybrid-rerank-rag", "reports/offline/hybrid-rerank-rag/latest.json"),
]


def load_report_metrics(
    report_path: Path, require_live_decider: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads a local report JSON and computes standardized n=61 answerable / n=19 negative metrics."""
    if not report_path.exists():
        return {}, {}

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    stored_metrics = data.get("metrics", {})
    if require_live_decider:
        jmode = stored_metrics.get("judge_evaluation_mode")
        if jmode != "openrouter_alpha_decisions":
            raise ValueError(
                f"Report '{report_path}' has judge_evaluation_mode={jmode!r}, expected "
                "'openrouter_alpha_decisions'. Refusing to mix incompatible judges."
            )

    records = data.get("records", [])
    if not records:
        return stored_metrics, data

    answerable = [r for r in records if r.get("ground_truth_context_ids")]
    negatives = [r for r in records if not r.get("ground_truth_context_ids")]
    factoids = [r for r in answerable if r.get("query_type") == "factoid"]
    multi_hop = [r for r in answerable if r.get("query_type") == "multi_hop"]

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    hit_rate = mean([r.get("retrieval_hit_rate", 0.0) for r in answerable])
    context_precision = mean([r.get("retrieval_precision", 0.0) for r in answerable])
    context_recall = mean([r.get("retrieval_recall", 0.0) for r in answerable])
    mrr = mean([r.get("retrieval_mrr", 0.0) for r in answerable])

    # Full accumulated span retrieval (captures Hop-2 chunks in agentic CRAG)
    all_hops_hits: list[float] = []
    all_hops_recalls: list[float] = []
    all_hops_mh_recalls: list[float] = []
    for r in answerable:
        spans = r.get("retrieved_spans", [])
        gt_spans = r.get("ground_truth_spans", [])
        k_all = max(4, len(spans))
        rm = evaluate_retrieval(spans, gt_spans, k=k_all)
        all_hops_hits.append(rm.hit_rate)
        all_hops_recalls.append(rm.context_recall)
        if r.get("query_type") == "multi_hop":
            all_hops_mh_recalls.append(rm.context_recall)

    mh_recall = mean([r.get("retrieval_recall", 0.0) for r in multi_hop])
    mh_correctness = mean([r.get("correctness_score", 0.0) for r in multi_hop])
    mh_grade = mean([r.get("rubric_grade", 1.0) for r in multi_hop])
    factoid_correctness = mean([r.get("correctness_score", 0.0) for r in factoids])

    neg_refused = [
        r
        for r in negatives
        if "insufficient information" in str(r.get("generated_answer", "")).lower()
    ]
    neg_refusal_rate = (len(neg_refused) / len(negatives)) if negatives else 1.0

    # Generation quality is averaged strictly over n=61 answerable queries (matching src/eval/runner.py)
    faithfulness = mean([r.get("faithfulness_score", 0.0) for r in answerable])
    correctness = mean([r.get("correctness_score", 0.0) for r in answerable])
    rubric_grade = mean([r.get("rubric_grade", 1.0) for r in answerable])
    empty_answers = sum(1 for r in records if not str(r.get("generated_answer", "")).strip())

    retrieval_latencies = [r.get("retrieval_latency_ms", 0.0) for r in records]
    gen_latencies = [r.get("generation_latency_ms", 0.0) for r in records]

    p95_ret = float(np.percentile(retrieval_latencies, 95)) if retrieval_latencies else 0.0
    p95_gen = float(np.percentile(gen_latencies, 95)) if gen_latencies else 0.0
    total_cost = sum(r.get("cost_eur", 0.0) for r in records)

    computed = {
        "k": stored_metrics.get("k", 4),
        "n_answerable": len(answerable),
        "n_negative": len(negatives),
        "total_samples": len(records),
        "hit_rate_at_k": round(hit_rate, 4),
        "mrr_at_k": round(mrr, 4),
        "context_precision_at_k": round(context_precision, 4),
        "context_recall_at_k": round(context_recall, 4),
        "all_hops_hit_rate": round(mean(all_hops_hits), 4),
        "all_hops_context_recall": round(mean(all_hops_recalls), 4),
        "all_hops_multi_hop_recall": round(mean(all_hops_mh_recalls), 4),
        "multi_hop_recall_at_k": round(mh_recall, 4),
        "factoid_correctness": round(factoid_correctness, 4),
        "multi_hop_correctness": round(mh_correctness, 4),
        "multi_hop_grade": round(mh_grade, 2),
        "negative_refusal_rate": round(neg_refusal_rate, 4),
        "mean_faithfulness": round(faithfulness, 4),
        "mean_correctness": round(correctness, 4),
        "mean_rubric_grade": round(rubric_grade, 2),
        "empty_answer_count": empty_answers,
        "p95_retrieval_latency_ms": round(p95_ret, 2),
        "p95_generation_latency_ms": round(p95_gen, 2),
        "total_cost_eur": round(total_cost, 6),
        "ci95": stored_metrics.get("ci95", {}),
        "judge_model": stored_metrics.get("judge_model"),
        "judge_evaluation_mode": stored_metrics.get("judge_evaluation_mode"),
    }
    return computed, data


def export_offline_summary() -> Path:
    """Exports canonical post-fix offline retrieval summary JSON to results/offline_retrieval_summary.json."""
    runs: list[dict[str, Any]] = []
    env_sample: dict[str, Any] = {}
    git_sha = "unknown"

    for run_name, rel_path in OFFLINE_RUNS:
        rep_path = REPO_ROOT / rel_path
        if not rep_path.exists():
            raise FileNotFoundError(f"Offline report missing: {rep_path}. Run `make eval-all`.")
        with open(rep_path, encoding="utf-8") as f:
            data = json.load(f)
        if not env_sample and data.get("environment"):
            env_sample = data["environment"]
        git_sha = str(data.get("git_sha", "unknown"))[:7]
        m = data.get("metrics", {})
        runs.append(
            {
                "run_name": run_name,
                "git_sha": data.get("git_sha", "unknown"),
                "index_fingerprint": data.get("index_fingerprint", "unknown"),
                "metrics": {
                    "k": m.get("k", 4),
                    "n_answerable": m.get("n_answerable"),
                    "n_negative": m.get("n_negative"),
                    "hit_rate_at_k": m.get("hit_rate_at_k"),
                    "mrr_at_k": m.get("mrr_at_k"),
                    "context_precision_at_k": m.get("context_precision_at_k"),
                    "context_recall_at_k": m.get("context_recall_at_k"),
                    "ci95": m.get("ci95", {}),
                    "p95_retrieval_latency_ms": m.get("p95_retrieval_latency_ms"),
                },
            }
        )

    out_path = REPO_ROOT / "results/offline_retrieval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Aggregate offline retrieval metrics only (n=61 answerable, span-overlap >= 50%, "
            "B=2,000 bootstrap CIs). Full per-query traces stay local under reports/ (gitignored)."
        ),
        "git_sha": git_sha,
        "environment": env_sample,
        "runs": runs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    console.print(f"[bold green]✔ Exported offline summary to {out_path}[/bold green]")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate live RAG paradigm comparison tables (Static Hybrid RRF vs LangGraph CRAG) "
            "and export canonical aggregate summaries under results/."
        )
    )
    parser.add_argument(
        "--static-report",
        type=str,
        default="reports/live/openrouter-tri-model-eval/deepseek_latest.json",
        help="Path to live Static Hybrid RRF report JSON (judged by Perplexity Decider).",
    )
    parser.add_argument(
        "--agentic-report",
        type=str,
        default="reports/live/agentic-langgraph-crag/latest.json",
        help="Path to live LangGraph Agentic CRAG report JSON (judged by Perplexity Decider).",
    )
    parser.add_argument(
        "--export-offline",
        action="store_true",
        help="Export results/offline_retrieval_summary.json from local reports/offline/*/latest.json.",
    )
    args = parser.parse_args()

    if args.export_offline:
        export_offline_summary()
        return

    arms = [
        (
            "Static Hybrid RRF (Live)",
            "Recursive 400/50, BM25 + Dense RRF Top-4 (DeepSeek V4.1 Flash)",
            REPO_ROOT / args.static_report,
        ),
        (
            "LangGraph Agentic CRAG (Live)",
            "Decompose -> Retrieve -> Grade -> Multi-Hop Synthesis (DeepSeek V4.1 Flash)",
            REPO_ROOT / args.agentic_report,
        ),
    ]

    table = Table(
        title="Live RAG Paradigm Comparison (n=61 Answerable, n=19 Negative, Perplexity Decider)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Paradigm", style="bold")
    table.add_column("Factoid Corr.", justify="right")
    table.add_column("Multi-Hop Corr. (17 Qs)", justify="right")
    table.add_column("All-Hops MH Recall", justify="right")
    table.add_column("Neg. Refusal (19 Qs)", justify="right")
    table.add_column("Mean Faith. (n=61)", justify="right")
    table.add_column("Mean Corr. (n=61)", justify="right")
    table.add_column("P95 Gen Latency", justify="right")

    md_rows: list[str] = []
    for label, desc, path in arms:
        m, _ = load_report_metrics(path, require_live_decider=True)
        if not m:
            raise FileNotFoundError(f"Required live report not found: {path}")

        row = [
            label,
            f"{m['factoid_correctness'] * 100:.1f}%",
            f"{m['multi_hop_correctness'] * 100:.1f}% ({m['multi_hop_grade']:.2f}/5)",
            f"{m['all_hops_multi_hop_recall'] * 100:.1f}%",
            f"{m['negative_refusal_rate'] * 100:.1f}%",
            f"{m['mean_faithfulness'] * 100:.1f}%",
            f"{m['mean_correctness'] * 100:.1f}% ({m['mean_rubric_grade']:.2f}/5)",
            f"{m['p95_generation_latency_ms']:.0f} ms",
        ]
        table.add_row(*row)
        md_rows.append(
            f"| **{label}** | {desc} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |"
        )

    console.print(table)

    out_path = REPO_ROOT / "results/paradigm_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Live RAG Paradigm Comparison: Static Hybrid RRF vs. LangGraph Agentic CRAG\n\n"
        "All metrics use identical generator (`deepseek/deepseek-v4.1-flash`, `max_tokens=4000`) "
        "and judge (`perplexity/pplx-decider-flash-v1.2` via `/api/alpha/decisions`). "
        "Generation quality is averaged over `n = 61` answerable queries; negative refusal over `n = 19`.\n\n"
        "| Paradigm | Architecture | Factoid Corr. | Multi-Hop Corr. (17 Qs) | All-Hops MH Recall | Negative Refusal (19 Qs) | Mean Faithfulness | Mean Correctness | P95 Gen Latency |\n"
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    )
    out_path.write_text(header + "\n".join(md_rows) + "\n", encoding="utf-8")
    console.print(f"\n[bold green]✔ Saved comparison table to {out_path}[/bold green]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a 4-tier paradigm comparison table across Baseline, Optimized, Compiled, and Agentic RAG."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# Ensure rag-eval-harness root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

console = Console()

PARADIGM_MAP = [
    {
        "tier": "Tier 1: Baseline",
        "name": "baseline-naive-rag",
        "description": "Naive Single-Shot (Fixed 1000/0, Dense Top-4)",
        "report": "reports/offline/baseline-naive-rag/latest.json",
    },
    {
        "tier": "Tier 2: Optimized",
        "name": "hybrid-rerank-rag",
        "description": "Two-Stage Hybrid (Recursive 400/50, RRF Top-15 -> Rerank Top-4)",
        "report": "reports/offline/hybrid-rerank-rag/latest.json",
    },
    {
        "tier": "Tier 3: Compiled",
        "name": "dspy-compiled-rag",
        "description": "DSPy Teleprompter Program (Signatures & FewShot Demos)",
        "report": "reports/offline/dspy-compiled-rag/latest.json",
    },
    {
        "tier": "Tier 4: Agentic",
        "name": "agentic-langgraph-crag",
        "description": "LangGraph CRAG (Decompose -> Retrieve -> Grade -> Generate/Refuse)",
        "report": "reports/offline/agentic-langgraph-crag/latest.json",
    },
]


def load_report_metrics(report_path: Path) -> dict[str, Any]:
    """Loads and aggregates detailed metrics including multi-hop and negative breakdowns."""
    if not report_path.exists():
        return {}

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        return data.get("metrics", {})

    answerable = [r for r in records if r.get("ground_truth_context_ids")]
    negatives = [r for r in records if not r.get("ground_truth_context_ids")]
    multi_hop = [r for r in records if r.get("query_type") == "multi_hop"]

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    hit_rate = mean([r.get("retrieval_hit_rate", 0.0) for r in answerable])
    context_recall = mean([r.get("retrieval_recall", 0.0) for r in answerable])
    mrr = mean([r.get("retrieval_mrr", 0.0) for r in answerable])

    mh_recall = mean([r.get("retrieval_recall", 0.0) for r in multi_hop])
    mh_correctness = mean([r.get("correctness_score", 0.0) for r in multi_hop])
    mh_grade = mean([r.get("rubric_grade", 1.0) for r in multi_hop])

    neg_refused = [
        r for r in negatives if "insufficient information" in str(r.get("generated_answer", "")).lower()
    ]
    neg_refusal_rate = (len(neg_refused) / len(negatives)) if negatives else 1.0

    faithfulness = mean([r.get("faithfulness_score", 0.0) for r in records])
    correctness = mean([r.get("correctness_score", 0.0) for r in records])
    rubric_grade = mean([r.get("rubric_grade", 1.0) for r in records])

    retrieval_latencies = [r.get("retrieval_latency_ms", 0.0) for r in records]
    gen_latencies = [r.get("generation_latency_ms", 0.0) for r in records]

    import numpy as np

    p95_ret = float(np.percentile(retrieval_latencies, 95)) if retrieval_latencies else 0.0
    p95_gen = float(np.percentile(gen_latencies, 95)) if gen_latencies else 0.0
    total_cost = sum(r.get("cost_eur", 0.0) for r in records)

    return {
        "total_samples": len(records),
        "hit_rate": hit_rate,
        "context_recall": context_recall,
        "mrr": mrr,
        "multi_hop_recall": mh_recall,
        "multi_hop_correctness": mh_correctness,
        "multi_hop_grade": mh_grade,
        "neg_refusal_rate": neg_refusal_rate,
        "faithfulness": faithfulness,
        "correctness": correctness,
        "rubric_grade": rubric_grade,
        "p95_retrieval_ms": p95_ret,
        "p95_gen_ms": p95_gen,
        "total_cost_eur": total_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 4-tier paradigm comparison table.")
    parser.add_argument(
        "--live-deepseek",
        action="store_true",
        help="Use live DeepSeek / Perplexity evaluation outputs for the comparison table.",
    )
    args = parser.parse_args()

    paradigm_configs = list(PARADIGM_MAP)
    if args.live_deepseek:
        paradigm_configs[1] = {
            "tier": "Tier 2: Optimized (Live)",
            "name": "hybrid-rerank-live",
            "description": "Two-Stage Hybrid (Live DeepSeek V4.1 Flash + Perplexity Decider)",
            "report": "reports/live/openrouter-tri-model-eval/deepseek_v4.1_perplexity_judged.json",
        }
        paradigm_configs[3] = {
            "tier": "Tier 4: Agentic (Live)",
            "name": "agentic-langgraph-live",
            "description": "LangGraph CRAG (Live DeepSeek V4.1 Flash + Perplexity Decider)",
            "report": "reports/live/agentic-langgraph-crag/latest.json",
        }

    table = Table(
        title="RAG Paradigm Comparison Matrix: Baseline vs. Optimized vs. Compiled vs. Agentic",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Paradigm", style="bold")
    table.add_column("Architecture Summary", style="dim")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Multi-Hop Recall (17 Qs)", justify="right")
    table.add_column("Multi-Hop Correctness", justify="right")
    table.add_column("Negative Refusal (19 Qs)", justify="right")
    table.add_column("Mean Faithfulness", justify="right")
    table.add_column("Mean Correctness", justify="right")
    table.add_column("P95 Latency (Gen)", justify="right")
    table.add_column("Cost (€ / 80 Qs)", justify="right")

    markdown_rows = []

    for item in paradigm_configs:
        rep_path = REPO_ROOT / item["report"]
        m = load_report_metrics(rep_path)
        if not m:
            continue

        hit_str = f"{m.get('hit_rate', 0.0) * 100:.1f}%"
        mh_rec_str = f"{m.get('multi_hop_recall', 0.0) * 100:.1f}%"
        mh_corr_str = f"{m.get('multi_hop_correctness', 0.0) * 100:.1f}% ({m.get('multi_hop_grade', 1.0):.1f}/5)"
        neg_str = f"{m.get('neg_refusal_rate', 0.0) * 100:.1f}%"
        faith_str = f"{m.get('faithfulness', 0.0) * 100:.1f}%"
        corr_str = f"{m.get('correctness', 0.0) * 100:.1f}% ({m.get('rubric_grade', 1.0):.2f}/5)"
        lat_str = f"{m.get('p95_gen_ms', 0.0):.0f} ms"
        cost_str = f"€{m.get('total_cost_eur', 0.0):.4f}"

        table.add_row(
            item["tier"],
            item["description"],
            hit_str,
            mh_rec_str,
            mh_corr_str,
            neg_str,
            faith_str,
            corr_str,
            lat_str,
            cost_str,
        )

        markdown_rows.append(
            f"| **{item['tier']}** | {item['description']} | {hit_str} | {mh_rec_str} | {mh_corr_str} | {neg_str} | {faith_str} | {corr_str} | {lat_str} | {cost_str} |"
        )

    console.print(table)

    # Save to reports/paradigm_comparison_table.md
    out_path = REPO_ROOT / "reports/paradigm_comparison_table.md"
    md_header = """# 4-Tier RAG Paradigm Comparison: Baseline vs. Optimized vs. Compiled vs. Agentic

| Paradigm | Architecture Summary | Hit Rate@K | Multi-Hop Recall (17 Qs) | Multi-Hop Correctness | Negative Refusal (19 Qs) | Mean Faithfulness | Mean Correctness | P95 Gen Latency | Cost (80 Qs) |
|---|---|---|---|---|---|---|---|---|---|
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_header + "\n".join(markdown_rows) + "\n")

    console.print(f"\n[bold green]✔ Saved markdown comparison table to {out_path}[/bold green]")


if __name__ == "__main__":
    main()

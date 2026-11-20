#!/usr/bin/env python3
"""Offline retrieval sweep across K cutoffs (4, 6, 8, 12) for Hybrid RRF."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.retrieval_metrics import evaluate_retrieval  # noqa: E402
from src.eval.runner import _collect_environment_telemetry, _get_git_sha  # noqa: E402
from src.pipeline.indexing import VectorIndexer  # noqa: E402
from src.pipeline.ingestion import chunk_documents, load_raw_documents  # noqa: E402
from src.pipeline.retriever import RAGRetriever  # noqa: E402
from src.utils.config import PipelineConfig  # noqa: E402

console = Console()


def _bootstrap_ci(vals: list[float], n_resamples: int = 2000, seed: int = 0) -> list[float]:
    arr = np.array(vals, dtype=float)
    if len(arr) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(n_resamples)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def run_k_sweep(
    ks: list[int],
    config_path: str | Path = "configs/hybrid.yaml",
    golden_qa_path: str | Path = "data/golden_qa.json",
) -> dict[str, Any]:
    """Sweeps Hybrid RRF retrieval across multiple K values without any LLM API calls."""
    config = PipelineConfig.from_yaml(config_path)
    indexer = VectorIndexer(config.embedding, config.vector_store)

    fp = VectorIndexer.corpus_fingerprint(
        config.corpus.data_dir, config.corpus.file_pattern, config.chunking
    )
    if indexer.count() == 0 or indexer.stored_fingerprint() != fp:
        indexer.reset_collection()
        raw_docs = load_raw_documents(config.corpus.data_dir, config.corpus.file_pattern)
        chunks = chunk_documents(raw_docs, config.chunking)
        indexer.index_chunks(chunks, fingerprint=fp)

    total_chunks = indexer.count()
    golden = json.loads(Path(golden_qa_path).read_text(encoding="utf-8"))
    answerable = [item for item in golden if item.get("ground_truth_spans")]

    # Retrieve once at max(ks) per query to measure latency cleanly per K
    rows: list[dict[str, Any]] = []
    for k in ks:
        cfg_k = config.retriever.model_copy(update={"top_k": k})
        retriever = RAGRetriever(indexer, cfg_k)

        hits, mrrs, precs, recs, mh_recs, latencies = [], [], [], [], [], []
        for item in answerable:
            res = retriever.retrieve(item["query"])
            latencies.append(res.retrieval_latency_ms)
            spans = [
                {
                    "doc": c.metadata.get("doc", c.chunk_id.split("_c")[0]),
                    "char_start": c.metadata.get("char_start", 0),
                    "char_end": c.metadata.get("char_end", len(c.content)),
                }
                for c in res.chunks
            ]
            rm = evaluate_retrieval(spans, item["ground_truth_spans"], k=k)
            hits.append(rm.hit_rate)
            mrrs.append(rm.mrr)
            precs.append(rm.context_precision)
            recs.append(rm.context_recall)
            if item.get("query_type") == "multi_hop":
                mh_recs.append(rm.context_recall)

        rows.append(
            {
                "k": k,
                "corpus_pct_sent": round(k / total_chunks, 4) if total_chunks else 0.0,
                "approx_chars": k * config.chunking.chunk_size,
                "n_answerable": len(answerable),
                "n_multi_hop": len(mh_recs),
                "hit_rate_at_k": round(float(np.mean(hits)), 4),
                "mrr_at_k": round(float(np.mean(mrrs)), 4),
                "context_precision_at_k": round(float(np.mean(precs)), 4),
                "context_recall_at_k": round(float(np.mean(recs)), 4),
                "multi_hop_recall_at_k": round(float(np.mean(mh_recs)), 4),
                "ci95": {
                    "hit_rate_at_k": _bootstrap_ci(hits),
                    "mrr_at_k": _bootstrap_ci(mrrs),
                    "context_precision_at_k": _bootstrap_ci(precs),
                    "context_recall_at_k": _bootstrap_ci(recs),
                    "multi_hop_recall_at_k": _bootstrap_ci(mh_recs),
                },
                "p95_retrieval_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            }
        )

    return {
        "note": (
            "Offline Hybrid RRF (BM25 + dense, recursive 400/50) retrieval sweep across K cutoffs "
            "(n=61 answerable, n=17 multi-hop, B=2,000 bootstrap CIs). Zero LLM API calls."
        ),
        "git_sha": _get_git_sha(),
        "index_fingerprint": fp,
        "total_chunks": total_chunks,
        "environment": _collect_environment_telemetry(),
        "sweep": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline Hybrid RRF retrieval sweep across K cutoffs."
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[4, 6, 8, 12],
        help="List of K values to evaluate (default: 4 6 8 12).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/k_sweep_retrieval_summary.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = run_k_sweep(args.ks)
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table(
        title="Offline Hybrid RRF K-Sweep (n=61 Answerable, n=17 Multi-Hop, 65 Total Chunks)",
        header_style="bold cyan",
    )
    table.add_column("K", justify="right", style="bold")
    table.add_column("Corpus %", justify="right")
    table.add_column("Hit@K", justify="right")
    table.add_column("MRR@K", justify="right")
    table.add_column("Prec@K", justify="right")
    table.add_column("Recall@K", justify="right")
    table.add_column("MH Recall@K", justify="right")
    table.add_column("P95 Latency", justify="right")

    for r in payload["sweep"]:
        table.add_row(
            str(r["k"]),
            f"{r['corpus_pct_sent'] * 100:.1f}%",
            f"{r['hit_rate_at_k'] * 100:.2f}%",
            f"{r['mrr_at_k']:.3f}",
            f"{r['context_precision_at_k'] * 100:.2f}%",
            f"{r['context_recall_at_k'] * 100:.2f}%",
            f"{r['multi_hop_recall_at_k'] * 100:.2f}%",
            f"{r['p95_retrieval_latency_ms']:.1f} ms",
        )

    console.print(table)
    console.print(f"\n[bold green]✔ Exported K-sweep summary to {out_path}[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

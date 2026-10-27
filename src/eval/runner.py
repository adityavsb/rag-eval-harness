"""End-to-end evaluation harness runner."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from src.eval.llm_judge import LLMJudge
from src.eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)
from src.pipeline.generator import RAGGenerator
from src.pipeline.indexing import VectorIndexer
from src.pipeline.ingestion import chunk_documents, load_raw_documents
from src.pipeline.retriever import RAGRetriever
from src.utils.config import PipelineConfig
from src.utils.tracking import MLflowTracker


class EvaluationRecord(BaseModel):
    """Detailed evaluation log for an individual test case."""

    query_id: str
    query: str
    query_type: str
    ground_truth_context_ids: list[str]
    ground_truth_spans: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunk_ids: list[str]
    retrieved_spans: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_hit_rate: float
    retrieval_mrr: float
    retrieval_precision: float
    retrieval_recall: float
    generated_answer: str
    ground_truth_answer: str
    faithfulness_score: float
    correctness_score: float
    rubric_grade: int
    retrieval_latency_ms: float
    generation_latency_ms: float


def _get_git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class EvaluationReport(BaseModel):
    """Aggregate benchmark results and run summary with execution provenance."""

    run_name: str
    total_samples: int
    mode: str = "offline"
    provider: str = "mock"
    model_id: str = "mock-generator"
    index_fingerprint: str = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_sha: str = "unknown"
    cost_eur: float = 0.0
    metrics: dict[str, Any]
    records: list[EvaluationRecord]


class EvaluationRunner:
    """Executes evaluation benchmarks across retrieval and generation stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.console = Console()
        self.tracker = MLflowTracker(config)

        # Apply top-level mode override if present
        if config.mode:
            config.generation.mode = config.mode
            config.evaluation.mode = config.mode

        # Initialize pipeline components
        self.indexer = VectorIndexer(config.embedding, config.vector_store)
        self.retriever = RAGRetriever(self.indexer, config.retriever)
        self.generator = RAGGenerator(config.generation)
        self.judge = LLMJudge(
            model_name=config.evaluation.judge_model,
            mode=config.evaluation.mode,
            provider=config.evaluation.judge_provider,
        )

        # Effective K based on retriever / reranker configuration
        self.effective_k = (
            config.retriever.reranker_top_n
            if config.retriever.use_reranker
            else config.retriever.top_k
        )

    def prepare_index(self, force_reindex: bool = False) -> int:
        """Ensures vector database is populated with chunks from corpus, checking fingerprint."""
        fp = VectorIndexer.corpus_fingerprint(
            self.config.corpus.data_dir, self.config.corpus.file_pattern, self.config.chunking
        )
        existing_count = self.indexer.count()
        stored_fp = self.indexer.stored_fingerprint()
        if existing_count > 0 and not force_reindex and stored_fp == fp:
            self.console.print(
                f"[green]Using existing index with {existing_count} chunks (fingerprint {fp}).[/green]"
            )
            return existing_count

        if existing_count > 0:
            self.console.print(
                f"[yellow]Index fingerprint mismatch (stored={stored_fp}, expected={fp}) or force_reindex={force_reindex}; rebuilding collection.[/yellow]"
            )
            self.indexer.reset_collection()

        self.console.print(f"[cyan]Indexing corpus from {self.config.corpus.data_dir}...[/cyan]")
        raw_docs = load_raw_documents(self.config.corpus.data_dir, self.config.corpus.file_pattern)
        chunks = chunk_documents(raw_docs, self.config.chunking)
        count = self.indexer.index_chunks(chunks, fingerprint=fp)
        self.console.print(
            f"[green]Successfully indexed {count} chunks into ChromaDB (fingerprint {fp}).[/green]"
        )
        return count

    def load_golden_set(self) -> list[dict[str, Any]]:
        """Loads benchmark Q&A evaluation dataset."""
        path = Path(self.config.evaluation.golden_qa_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden QA dataset not found at {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self, force_reindex: bool = False) -> EvaluationReport:
        """Executes full evaluation benchmark, outputs summary table, and logs to MLflow."""
        self.prepare_index(force_reindex=force_reindex)
        golden_set = self.load_golden_set()
        records: list[EvaluationRecord] = []

        self.console.print(
            f"\n[bold]Evaluating run: [yellow]{self.config.run_name}[/yellow] ({len(golden_set)} queries)[/bold]"
        )

        with self.tracker.start_run():
            for item in golden_set:
                q_id = item["id"]
                query = item["query"]
                q_type = item.get("query_type", "factoid")
                gt_ids = item.get("ground_truth_context_ids", [])
                gt_spans = item.get("ground_truth_spans", [])
                gt_ans = item["ground_truth_answer"]

                # 1. Retrieval
                retrieval_res = self.retriever.retrieve(query)
                retrieved_ids = [c.chunk_id for c in retrieval_res.chunks]
                retrieved_spans = [
                    {
                        "doc": c.metadata.get("doc", c.chunk_id.split("_c")[0]),
                        "char_start": c.metadata.get("char_start", 0),
                        "char_end": c.metadata.get("char_end", len(c.content)),
                    }
                    for c in retrieval_res.chunks
                ]
                ret_metrics = evaluate_retrieval(retrieved_spans, gt_spans, k=self.effective_k)

                # 2. Generation
                gen_res = self.generator.generate(query, retrieval_res.chunks)

                # 3. LLM-as-a-Judge Evaluation
                context_str = "\n".join(
                    [f"[{c.chunk_id}] {c.content}" for c in retrieval_res.chunks]
                )
                faith_res = self.judge.evaluate_faithfulness(query, context_str, gen_res.answer)
                corr_res = self.judge.evaluate_correctness(query, gen_res.answer, gt_ans)

                record = EvaluationRecord(
                    query_id=q_id,
                    query=query,
                    query_type=q_type,
                    ground_truth_context_ids=gt_ids,
                    ground_truth_spans=gt_spans,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_spans=retrieved_spans,
                    retrieval_hit_rate=ret_metrics.hit_rate,
                    retrieval_mrr=ret_metrics.mrr,
                    retrieval_precision=ret_metrics.context_precision,
                    retrieval_recall=ret_metrics.context_recall,
                    generated_answer=gen_res.answer,
                    ground_truth_answer=gt_ans,
                    faithfulness_score=faith_res.score,
                    correctness_score=corr_res.score,
                    rubric_grade=corr_res.rubric_grade,
                    retrieval_latency_ms=retrieval_res.retrieval_latency_ms,
                    generation_latency_ms=gen_res.generation_latency_ms,
                )
                records.append(record)

            # Compute Aggregate Metrics: segregate answerable vs negative queries
            answerable = [r for r in records if r.ground_truth_context_ids]
            negatives = [r for r in records if not r.ground_truth_context_ids]

            agg_retrieval = aggregate_retrieval_metrics(
                [
                    evaluate_retrieval(
                        r.retrieved_spans,
                        r.ground_truth_spans,
                        k=self.effective_k,
                    )
                    for r in answerable
                ]
            )

            ret_latencies = [r.retrieval_latency_ms for r in records]
            gen_latencies = [r.generation_latency_ms for r in records]

            refusal_rate = (
                round(
                    float(
                        np.mean(
                            [
                                ("insufficient information" in r.generated_answer.lower())
                                for r in negatives
                            ]
                        )
                    ),
                    4,
                )
                if negatives
                else 0.0
            )

            metrics_summary: dict[str, Any] = {
                "k": self.effective_k,
                "n_answerable": len(answerable),
                "n_negative": len(negatives),
                "hit_rate_at_k": round(agg_retrieval.mean_hit_rate, 4),
                "mrr_at_k": round(agg_retrieval.mean_mrr, 4),
                "context_precision_at_k": round(agg_retrieval.mean_context_precision, 4),
                "context_recall_at_k": round(agg_retrieval.mean_context_recall, 4),
                "negative_refusal_rate": refusal_rate,
                "p95_retrieval_latency_ms": round(float(np.percentile(ret_latencies, 95)), 2),
                "p95_generation_latency_ms": round(float(np.percentile(gen_latencies, 95)), 2),
            }

            is_live = self.config.generation.mode == "live"
            if is_live:
                metrics_summary.update(
                    {
                        "mean_faithfulness": round(
                            float(np.mean([r.faithfulness_score for r in answerable])), 4
                        ),
                        "mean_correctness": round(
                            float(np.mean([r.correctness_score for r in answerable])), 4
                        ),
                        "mean_rubric_grade": round(
                            float(np.mean([r.rubric_grade for r in answerable])), 2
                        ),
                    }
                )

            # Log metrics and artifact to MLflow
            self.tracker.log_metrics(metrics_summary)
            stored_fp = self.indexer.stored_fingerprint() or "unknown"
            report = EvaluationReport(
                run_name=self.config.run_name,
                total_samples=len(records),
                mode=self.config.generation.mode,
                provider=self.config.generation.provider if is_live else "mock",
                model_id=self.config.generation.model_name if is_live else "mock-generator",
                index_fingerprint=stored_fp,
                timestamp=datetime.now(UTC).isoformat(),
                git_sha=_get_git_sha(),
                cost_eur=0.0,
                metrics=metrics_summary,
                records=records,
            )
            self.tracker.log_artifact_dict(
                report.model_dump(), f"eval_report_{self.config.run_name}.json"
            )

            # Also persist report to reports/ on disk with immutable timestamped file and latest.json
            now = datetime.now(UTC)
            git_sha = _get_git_sha()
            base_report_dir = Path("reports/live" if is_live else "reports/offline")
            report_run_dir = base_report_dir / self.config.run_name
            report_run_dir.mkdir(parents=True, exist_ok=True)

            report_filename = f"{now.strftime('%Y%m%dT%H%M%S')}_{git_sha[:7]}.json"
            timestamped_file = report_run_dir / report_filename
            with open(timestamped_file, "w", encoding="utf-8") as f:
                json.dump(report.model_dump(), f, indent=2)

            latest_file = report_run_dir / "latest.json"
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(report.model_dump(), f, indent=2)

            self._print_summary_table(metrics_summary)
            return report

    def _print_summary_table(self, metrics: dict[str, Any]) -> None:
        """Renders benchmark summary table to console."""
        table = Table(title=f"Evaluation Benchmark: {self.config.run_name}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Target Guidance", style="green")

        k_val = metrics.get("k", self.effective_k)
        n_ans = metrics.get("n_answerable", "N/A")
        n_neg = metrics.get("n_negative", "N/A")

        table.add_row("Evaluation Cutoff (K)", str(k_val), "-")
        table.add_row("Answerable Queries", str(n_ans), "-")
        table.add_row("Negative Queries", str(n_neg), "-")
        table.add_row(f"Hit Rate@{k_val}", f"{metrics['hit_rate_at_k']:.2%}", ">= 85.0%")
        table.add_row(f"MRR@{k_val}", f"{metrics['mrr_at_k']:.3f}", ">= 0.700")
        table.add_row(
            f"Context Precision@{k_val}", f"{metrics['context_precision_at_k']:.2%}", ">= 75.0%"
        )
        table.add_row(
            f"Context Recall@{k_val}", f"{metrics['context_recall_at_k']:.2%}", ">= 80.0%"
        )
        table.add_row(
            "Negative Refusal Rate", f"{metrics['negative_refusal_rate']:.2%}", ">= 90.0%"
        )

        if "mean_faithfulness" in metrics:
            table.add_row(
                "Faithfulness (Grounded)", f"{metrics['mean_faithfulness']:.2%}", ">= 90.0%"
            )
        if "mean_correctness" in metrics:
            table.add_row("Answer Correctness", f"{metrics['mean_correctness']:.2%}", ">= 80.0%")
        if "mean_rubric_grade" in metrics:
            table.add_row(
                "Average Rubric Grade", f"{metrics['mean_rubric_grade']:.2f} / 5.0", ">= 4.0 / 5"
            )

        table.add_row(
            "P95 Retrieval Latency", f"{metrics['p95_retrieval_latency_ms']:.1f} ms", "< 150 ms"
        )

        self.console.print("\n", table, "\n")

"""End-to-end evaluation harness runner."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
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
    retrieved_chunk_ids: list[str]
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


class EvaluationReport(BaseModel):
    """Aggregate benchmark results and run summary."""

    run_name: str
    total_samples: int
    metrics: dict[str, float]
    records: list[EvaluationRecord]


class EvaluationRunner:
    """Executes evaluation benchmarks across retrieval and generation stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.console = Console()
        self.tracker = MLflowTracker(config)

        # Initialize pipeline components
        self.indexer = VectorIndexer(config.embedding, config.vector_store)
        self.retriever = RAGRetriever(self.indexer, config.retriever)
        self.generator = RAGGenerator(config.generation)
        self.judge = LLMJudge(config.evaluation.judge_model)

    def prepare_index(self, force_reindex: bool = False) -> int:
        """Ensures vector database is populated with chunks from corpus."""
        existing_count = self.indexer.count()
        if existing_count > 0 and not force_reindex:
            self.console.print(f"[green]Using existing index with {existing_count} chunks.[/green]")
            return existing_count

        self.console.print(f"[cyan]Indexing corpus from {self.config.corpus.data_dir}...[/cyan]")
        raw_docs = load_raw_documents(self.config.corpus.data_dir, self.config.corpus.file_pattern)
        chunks = chunk_documents(raw_docs, self.config.chunking)
        count = self.indexer.index_chunks(chunks)
        self.console.print(f"[green]Successfully indexed {count} chunks into ChromaDB.[/green]")
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
                gt_ans = item["ground_truth_answer"]

                # 1. Retrieval
                retrieval_res = self.retriever.retrieve(query)
                retrieved_ids = [c.chunk_id for c in retrieval_res.chunks]
                ret_metrics = evaluate_retrieval(
                    retrieved_ids, gt_ids, k=self.config.retriever.top_k
                )

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
                    retrieved_chunk_ids=retrieved_ids,
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

            # Compute Aggregate Metrics
            agg_retrieval = aggregate_retrieval_metrics(
                [
                    evaluate_retrieval(
                        r.retrieved_chunk_ids,
                        r.ground_truth_context_ids,
                        k=self.config.retriever.top_k,
                    )
                    for r in records
                ]
            )

            ret_latencies = [r.retrieval_latency_ms for r in records]
            gen_latencies = [r.generation_latency_ms for r in records]

            metrics_summary = {
                "hit_rate_at_k": round(agg_retrieval.mean_hit_rate, 4),
                "mrr_at_k": round(agg_retrieval.mean_mrr, 4),
                "context_precision_at_k": round(agg_retrieval.mean_context_precision, 4),
                "context_recall_at_k": round(agg_retrieval.mean_context_recall, 4),
                "mean_faithfulness": round(
                    float(np.mean([r.faithfulness_score for r in records])), 4
                ),
                "mean_correctness": round(
                    float(np.mean([r.correctness_score for r in records])), 4
                ),
                "mean_rubric_grade": round(float(np.mean([r.rubric_grade for r in records])), 2),
                "p95_retrieval_latency_ms": round(float(np.percentile(ret_latencies, 95)), 2),
                "p95_generation_latency_ms": round(float(np.percentile(gen_latencies, 95)), 2),
            }

            # Log metrics and artifact to MLflow
            self.tracker.log_metrics(metrics_summary)
            report = EvaluationReport(
                run_name=self.config.run_name,
                total_samples=len(records),
                metrics=metrics_summary,
                records=records,
            )
            self.tracker.log_artifact_dict(
                report.model_dump(), f"eval_report_{self.config.run_name}.json"
            )

            self._print_summary_table(metrics_summary)
            return report

    def _print_summary_table(self, metrics: dict[str, float]) -> None:
        """Renders benchmark summary table to console."""
        table = Table(title=f"Evaluation Benchmark: {self.config.run_name}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Target Guidance", style="green")

        table.add_row("Hit Rate@K", f"{metrics['hit_rate_at_k']:.2%}", ">= 85.0%")
        table.add_row("MRR@K", f"{metrics['mrr_at_k']:.3f}", ">= 0.700")
        table.add_row("Context Precision@K", f"{metrics['context_precision_at_k']:.2%}", ">= 75.0%")
        table.add_row("Context Recall@K", f"{metrics['context_recall_at_k']:.2%}", ">= 80.0%")
        table.add_row("Faithfulness (Grounded)", f"{metrics['mean_faithfulness']:.2%}", ">= 90.0%")
        table.add_row("Answer Correctness", f"{metrics['mean_correctness']:.2%}", ">= 80.0%")
        table.add_row(
            "Average Rubric Grade", f"{metrics['mean_rubric_grade']:.2f} / 5.0", ">= 4.0 / 5"
        )
        table.add_row(
            "P95 Retrieval Latency", f"{metrics['p95_retrieval_latency_ms']:.1f} ms", "< 150 ms"
        )

        self.console.print("\n", table, "\n")

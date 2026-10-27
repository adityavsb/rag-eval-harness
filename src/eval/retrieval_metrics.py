"""Deterministic retrieval evaluation metrics for RAG pipelines.

Implements standard IR metrics:
- Hit Rate@K
- Mean Reciprocal Rank (MRR@K)
- Context Precision@K
- Context Recall@K
"""

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    """Container for single-query retrieval evaluation scores."""

    hit_rate: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)


class AggregateRetrievalMetrics(BaseModel):
    """Aggregate averages over an entire evaluation dataset."""

    mean_hit_rate: float = Field(ge=0.0, le=1.0)
    mean_mrr: float = Field(ge=0.0, le=1.0)
    mean_context_precision: float = Field(ge=0.0, le=1.0)
    mean_context_recall: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)


def is_chunk_relevant(
    retrieved: dict[str, Any], ground_truth: Iterable[dict[str, Any]], min_overlap: float = 0.5
) -> bool:
    """Checks whether a retrieved chunk's character span overlaps significantly with ground truth spans."""
    for gt in ground_truth:
        if retrieved.get("doc") != gt.get("doc"):
            continue
        if gt.get("char_end") == -1:  # whole-document relevance span
            return True
        gt_start = gt.get("char_start", 0)
        gt_end = gt.get("char_end", 0)
        r_start = retrieved.get("char_start", 0)
        r_end = retrieved.get("char_end", 0)
        inter = min(r_end, gt_end) - max(r_start, gt_start)
        gt_len = max(1, gt_end - gt_start)
        if inter / gt_len >= min_overlap:
            return True
    return False


def compute_hit_rate_at_k(
    retrieved: Sequence[dict[str, Any]], ground_truth: Iterable[dict[str, Any]], k: int
) -> float:
    """Returns 1.0 if at least one ground-truth span is covered in the top-K retrieved chunks."""
    gt_list = list(ground_truth)
    if not gt_list:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(is_chunk_relevant(chunk, gt_list) for chunk in top_k) else 0.0


def compute_mrr_at_k(
    retrieved: Sequence[dict[str, Any]], ground_truth: Iterable[dict[str, Any]], k: int
) -> float:
    """Computes Mean Reciprocal Rank (MRR) at cutoff K.

    MRR = 1 / rank of the first relevant document chunk in top-K (1-indexed).
    Returns 0.0 if no relevant document appears in top-K.
    """
    gt_list = list(ground_truth)
    if not gt_list:
        return 0.0
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if is_chunk_relevant(chunk, gt_list):
            return 1.0 / rank
    return 0.0


def compute_context_precision_at_k(
    retrieved: Sequence[dict[str, Any]], ground_truth: Iterable[dict[str, Any]], k: int
) -> float:
    """Computes Context Precision at K.

    Evaluates whether relevant chunks are ranked higher than irrelevant ones.
    Calculated as Average Precision: sum(Precision@i * relevance_indicator_i) / num_relevant_in_top_k
    """
    gt_list = list(ground_truth)
    if not gt_list:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    relevant_hits = 0
    precision_sum = 0.0

    for i, chunk in enumerate(top_k, start=1):
        if is_chunk_relevant(chunk, gt_list):
            relevant_hits += 1
            precision_sum += relevant_hits / i

    if relevant_hits == 0:
        return 0.0
    return precision_sum / relevant_hits


def compute_context_recall_at_k(
    retrieved: Sequence[dict[str, Any]], ground_truth: Iterable[dict[str, Any]], k: int
) -> float:
    """Computes Context Recall at K.

    Proportion of ground-truth relevant spans that were successfully retrieved in top-K.
    """
    gt_list = list(ground_truth)
    if not gt_list:
        return 0.0

    top_k = retrieved[:k]
    matched_gt = sum(
        1 for gt_item in gt_list if any(is_chunk_relevant(chunk, [gt_item]) for chunk in top_k)
    )
    return matched_gt / len(gt_list)


def evaluate_retrieval(
    retrieved: Sequence[dict[str, Any]], ground_truth: Iterable[dict[str, Any]], k: int
) -> RetrievalMetrics:
    """Calculates all 4 retrieval metrics for a single query."""
    gt_list = list(ground_truth)
    return RetrievalMetrics(
        hit_rate=compute_hit_rate_at_k(retrieved, gt_list, k),
        mrr=compute_mrr_at_k(retrieved, gt_list, k),
        context_precision=compute_context_precision_at_k(retrieved, gt_list, k),
        context_recall=compute_context_recall_at_k(retrieved, gt_list, k),
    )


def aggregate_retrieval_metrics(metrics_list: list[RetrievalMetrics]) -> AggregateRetrievalMetrics:
    """Aggregates a list of query-level metrics into dataset-level averages."""
    n = len(metrics_list)
    if n == 0:
        return AggregateRetrievalMetrics(
            mean_hit_rate=0.0,
            mean_mrr=0.0,
            mean_context_precision=0.0,
            mean_context_recall=0.0,
            sample_count=0,
        )

    return AggregateRetrievalMetrics(
        mean_hit_rate=sum(m.hit_rate for m in metrics_list) / n,
        mean_mrr=sum(m.mrr for m in metrics_list) / n,
        mean_context_precision=sum(m.context_precision for m in metrics_list) / n,
        mean_context_recall=sum(m.context_recall for m in metrics_list) / n,
        sample_count=n,
    )

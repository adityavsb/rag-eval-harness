"""Deterministic retrieval evaluation metrics for RAG pipelines.

Implements standard IR metrics:
- Hit Rate@K
- Mean Reciprocal Rank (MRR@K)
- Context Precision@K
- Context Recall@K
"""

from collections.abc import Iterable, Sequence

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


def is_chunk_relevant(retrieved_id: str, ground_truth_ids: Iterable[str]) -> bool:
    """Checks whether a retrieved chunk ID satisfies any ground truth ID reference.

    Matches exact chunk ID, prefix stem (e.g. matching '02_dependency_injection' with
    '02_dependency_injection_c002'), or vice-versa.
    """
    for gt_id in ground_truth_ids:
        if (
            retrieved_id == gt_id
            or retrieved_id.startswith(gt_id)
            or gt_id.startswith(retrieved_id)
        ):
            return True
    return False


def compute_hit_rate_at_k(
    retrieved_ids: Sequence[str], ground_truth_ids: Iterable[str], k: int
) -> float:
    """Returns 1.0 if at least one ground-truth ID appears in the top-K retrieved IDs."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(is_chunk_relevant(doc_id, ground_truth_ids) for doc_id in top_k) else 0.0


def compute_mrr_at_k(
    retrieved_ids: Sequence[str], ground_truth_ids: Iterable[str], k: int
) -> float:
    """Computes Mean Reciprocal Rank (MRR) at cutoff K.

    MRR = 1 / rank of the first relevant document in top-K (1-indexed).
    Returns 0.0 if no relevant document appears in top-K.
    """
    if not ground_truth_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if is_chunk_relevant(doc_id, ground_truth_ids):
            return 1.0 / rank
    return 0.0


def compute_context_precision_at_k(
    retrieved_ids: Sequence[str], ground_truth_ids: Iterable[str], k: int
) -> float:
    """Computes Context Precision at K.

    Evaluates whether relevant chunks are ranked higher than irrelevant ones.
    Calculated as Average Precision: sum(Precision@i * relevance_indicator_i) / num_relevant_in_top_k
    """
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_hits = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(top_k, start=1):
        if is_chunk_relevant(doc_id, ground_truth_ids):
            relevant_hits += 1
            precision_sum += relevant_hits / i

    if relevant_hits == 0:
        return 0.0
    return precision_sum / relevant_hits


def compute_context_recall_at_k(
    retrieved_ids: Sequence[str], ground_truth_ids: Iterable[str], k: int
) -> float:
    """Computes Context Recall at K.

    Proportion of ground-truth relevant chunks that were successfully retrieved in top-K.
    """
    gt_list = list(ground_truth_ids)
    if not gt_list:
        return 0.0

    top_k = retrieved_ids[:k]
    matched_gt = sum(
        1 for gt_id in gt_list if any(is_chunk_relevant(doc_id, [gt_id]) for doc_id in top_k)
    )
    return matched_gt / len(gt_list)


def evaluate_retrieval(
    retrieved_ids: Sequence[str], ground_truth_ids: Iterable[str], k: int
) -> RetrievalMetrics:
    """Calculates all 4 retrieval metrics for a single query."""
    return RetrievalMetrics(
        hit_rate=compute_hit_rate_at_k(retrieved_ids, ground_truth_ids, k),
        mrr=compute_mrr_at_k(retrieved_ids, ground_truth_ids, k),
        context_precision=compute_context_precision_at_k(retrieved_ids, ground_truth_ids, k),
        context_recall=compute_context_recall_at_k(retrieved_ids, ground_truth_ids, k),
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

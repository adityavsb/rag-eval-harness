"""Unit tests for retrieval evaluation metrics."""

import pytest

from src.eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_context_precision_at_k,
    compute_context_recall_at_k,
    compute_hit_rate_at_k,
    compute_mrr_at_k,
    evaluate_retrieval,
    is_chunk_relevant,
)


def _span(doc: str, start: int = 0, end: int = 100) -> dict:
    return {"doc": doc, "char_start": start, "char_end": end}


def test_span_overlap_relevance():
    # Target span: 100 to 200 (length 100)
    gt_spans = [_span("doc_1", 100, 200)]

    # 1. Exact match
    assert is_chunk_relevant(_span("doc_1", 100, 200), gt_spans) is True

    # 2. 60% overlap (100 to 160) -> overlap 60 >= 50 -> True
    assert is_chunk_relevant(_span("doc_1", 100, 160), gt_spans) is True

    # 3. 40% overlap (100 to 140) -> overlap 40 < 50 -> False
    assert is_chunk_relevant(_span("doc_1", 100, 140), gt_spans) is False

    # 4. Whole-document span (-1)
    whole_doc_gt = [_span("doc_1", 0, -1)]
    assert is_chunk_relevant(_span("doc_1", 500, 800), whole_doc_gt) is True

    # 5. Different document
    assert is_chunk_relevant(_span("doc_2", 100, 200), gt_spans) is False


def test_hit_rate_at_k():
    retrieved = [_span("doc_a"), _span("doc_b"), _span("doc_c"), _span("doc_d")]
    ground_truth = [_span("doc_c"), _span("doc_x")]

    # In top 2: doc_c is not in ["doc_a", "doc_b"]
    assert compute_hit_rate_at_k(retrieved, ground_truth, k=2) == 0.0

    # In top 3: doc_c is present
    assert compute_hit_rate_at_k(retrieved, ground_truth, k=3) == 1.0


def test_mrr_at_k():
    retrieved = [_span("doc_a"), _span("doc_b"), _span("doc_c"), _span("doc_d")]

    # First relevant is at rank 3 -> MRR = 1/3
    assert compute_mrr_at_k(retrieved, [_span("doc_c")], k=4) == pytest.approx(1.0 / 3.0)

    # First relevant at rank 1 -> MRR = 1.0
    assert compute_mrr_at_k(retrieved, [_span("doc_a")], k=4) == 1.0

    # Not found in top 2 -> MRR = 0.0
    assert compute_mrr_at_k(retrieved, [_span("doc_c")], k=2) == 0.0


def test_context_recall_at_k():
    retrieved = [_span("doc_a"), _span("doc_b"), _span("doc_c"), _span("doc_d")]
    ground_truth = [_span("doc_a"), _span("doc_c"), _span("doc_z")]

    # Top 2 has ["doc_a"] -> 1 out of 3 = 1/3
    assert compute_context_recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1.0 / 3.0)

    # Top 4 has ["doc_a", "doc_c"] -> 2 out of 3 = 2/3
    assert compute_context_recall_at_k(retrieved, ground_truth, k=4) == pytest.approx(2.0 / 3.0)


def test_context_precision_at_k():
    # Hit at rank 1 (1/1) and rank 3 (2/3) -> average precision = (1.0 + 2/3) / 2 = 5/6
    retrieved = [_span("doc_a"), _span("doc_b"), _span("doc_c")]
    ground_truth = [_span("doc_a"), _span("doc_c")]

    expected_ap = (1.0 + (2.0 / 3.0)) / 2.0
    assert compute_context_precision_at_k(retrieved, ground_truth, k=3) == pytest.approx(
        expected_ap
    )

    # No hits in top-k
    assert compute_context_precision_at_k(retrieved, [_span("doc_x")], k=3) == 0.0


def test_aggregate_retrieval_metrics():
    retrieved_1 = [_span("doc_a"), _span("doc_b")]
    retrieved_2 = [_span("doc_b"), _span("doc_c")]

    m1 = evaluate_retrieval(retrieved_1, [_span("doc_a")], k=2)
    m2 = evaluate_retrieval(retrieved_2, [_span("doc_z")], k=2)

    agg = aggregate_retrieval_metrics([m1, m2])
    assert agg.sample_count == 2
    assert agg.mean_hit_rate == 0.5  # (1.0 + 0.0) / 2
    assert agg.mean_mrr == 0.5  # (1.0 + 0.0) / 2

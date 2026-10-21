"""Unit tests for retrieval evaluation metrics."""

import pytest

from src.eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_context_precision_at_k,
    compute_context_recall_at_k,
    compute_hit_rate_at_k,
    compute_mrr_at_k,
    evaluate_retrieval,
)


def test_hit_rate_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    ground_truth = ["doc_c", "doc_x"]

    # In top 2: doc_c is not in ["doc_a", "doc_b"]
    assert compute_hit_rate_at_k(retrieved, ground_truth, k=2) == 0.0

    # In top 3: doc_c is present
    assert compute_hit_rate_at_k(retrieved, ground_truth, k=3) == 1.0


def test_mrr_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]

    # First relevant is at rank 3 -> MRR = 1/3
    assert compute_mrr_at_k(retrieved, ["doc_c"], k=4) == pytest.approx(1.0 / 3.0)

    # First relevant at rank 1 -> MRR = 1.0
    assert compute_mrr_at_k(retrieved, ["doc_a"], k=4) == 1.0

    # Not found in top 2 -> MRR = 0.0
    assert compute_mrr_at_k(retrieved, ["doc_c"], k=2) == 0.0


def test_context_recall_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    ground_truth = ["doc_a", "doc_c", "doc_z"]

    # Top 2 has ["doc_a"] -> 1 out of 3 = 1/3
    assert compute_context_recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1.0 / 3.0)

    # Top 4 has ["doc_a", "doc_c"] -> 2 out of 3 = 2/3
    assert compute_context_recall_at_k(retrieved, ground_truth, k=4) == pytest.approx(2.0 / 3.0)


def test_context_precision_at_k():
    # Hit at rank 1 (1/1) and rank 3 (2/3) -> average precision = (1.0 + 2/3) / 2 = 5/6
    retrieved = ["doc_a", "doc_b", "doc_c"]
    ground_truth = ["doc_a", "doc_c"]

    expected_ap = (1.0 + (2.0 / 3.0)) / 2.0
    assert compute_context_precision_at_k(retrieved, ground_truth, k=3) == pytest.approx(
        expected_ap
    )

    # No hits in top-k
    assert compute_context_precision_at_k(retrieved, ["doc_x"], k=3) == 0.0


def test_aggregate_retrieval_metrics():
    retrieved_1 = ["doc_a", "doc_b"]
    retrieved_2 = ["doc_b", "doc_c"]

    m1 = evaluate_retrieval(retrieved_1, ["doc_a"], k=2)
    m2 = evaluate_retrieval(retrieved_2, ["doc_z"], k=2)

    agg = aggregate_retrieval_metrics([m1, m2])
    assert agg.sample_count == 2
    assert agg.mean_hit_rate == 0.5  # (1.0 + 0.0) / 2
    assert agg.mean_mrr == 0.5  # (1.0 + 0.0) / 2

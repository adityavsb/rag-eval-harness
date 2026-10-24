"""Unit tests for Milestone M4: Judge calibration against human ground-truth labels."""

from pathlib import Path

from src.eval.judge_agreement import (
    calculate_cohen_kappa,
    calculate_spearman_rho,
    evaluate_judge_agreement,
)


def test_cohen_kappa_computation():
    """Validates Cohen's kappa computation with known rater alignments."""
    # Perfect agreement
    assert calculate_cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0

    # Complete disagreement
    assert calculate_cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0.0

    # Partial agreement
    kappa = calculate_cohen_kappa([1, 0, 1, 1], [1, 0, 0, 1])
    assert abs(kappa - 0.5) < 1e-3


def test_spearman_rho_computation():
    """Validates Spearman rho rank correlation computation."""
    # Perfect positive correlation
    rho, pval = calculate_spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert rho == 1.0
    assert pval < 0.01

    # Perfect inverse correlation
    rho_inv, _ = calculate_spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert rho_inv == -1.0


def test_judge_agreement_end_to_end():
    """Validates judge agreement evaluation on the 30 human labeled samples."""
    labels_path = Path("data/human_labels.jsonl")
    assert labels_path.exists(), "data/human_labels.jsonl does not exist"

    metrics, discrepancies = evaluate_judge_agreement(
        labels_path=labels_path,
        mode="offline",
    )

    assert metrics.total_samples == 30
    assert metrics.faithfulness_agreement_pct >= 80.0
    assert metrics.cohen_kappa >= 0.60
    assert metrics.spearman_rho >= 0.70
    assert metrics.spearman_pvalue < 0.05

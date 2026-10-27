"""Unit tests for Milestone M0 hygiene: explicit mode enforcement, provenance stamping, and fail-loud credentials."""

import os
import subprocess
import sys

import pytest

from src.eval.llm_judge import LLMJudge
from src.eval.runner import EvaluationReport
from src.pipeline.generator import RAGGenerator
from src.pipeline.retriever import RetrievedChunk
from src.utils.config import GenerationConfig


def test_generator_live_mode_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen_config = GenerationConfig(mode="live", provider="openai", model_name="gpt-4o-mini")
    generator = RAGGenerator(gen_config)

    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        doc_id="doc_01",
        content="Test content",
        score=0.9,
        rank=1,
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is missing"):
        generator.generate("Test query", [chunk])


def test_judge_live_mode_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = LLMJudge(mode="live", provider="openai", model_name="gpt-4o-mini")

    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is missing"):
        judge.evaluate_faithfulness("Query", "Context", "Answer")

    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is missing"):
        judge.evaluate_correctness("Query", "Answer", "Ground truth")


def test_evaluation_report_provenance_fields():
    report = EvaluationReport(
        run_name="test-run",
        total_samples=10,
        mode="offline",
        provider="mock",
        model_id="mock-generator",
        metrics={"hit_rate_at_k": 0.8},
        records=[],
    )
    dumped = report.model_dump()
    assert dumped["mode"] == "offline"
    assert dumped["provider"] == "mock"
    assert dumped["model_id"] == "mock-generator"
    assert "timestamp" in dumped
    assert "git_sha" in dumped
    assert "cost_eur" in dumped
    assert dumped["cost_eur"] == 0.0


def test_cli_live_mode_without_key_exits_nonzero(monkeypatch):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_eval.py",
            "--config",
            "configs/baseline.yaml",
            "--mode",
            "live",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert (
        "OPENAI_API_KEY environment variable is missing" in result.stdout
        or "OPENAI_API_KEY environment variable is missing" in result.stderr
    )


def test_readme_discloses_offline_mode():
    text = open("README.md", encoding="utf-8").read()
    assert "deterministic mock" in text and "lexical-overlap heuristic" in text

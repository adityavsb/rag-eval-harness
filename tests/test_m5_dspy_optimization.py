"""Unit tests for Milestone M5: DSPy generation prompt optimization and teleprompter compilation."""

import dspy
from pydantic import BaseModel

from src.optimize.compile import (
    build_chunk_map,
    load_dspy_splits,
    rag_optimization_metric,
)
from src.optimize.dspy_program import (
    RAGModule,
    configure_dspy_lm,
    load_compiled_program,
)
from src.pipeline.generator import RAGGenerator
from src.pipeline.retriever import RetrievedChunk
from src.utils.config import GenerationConfig


def test_dspy_signature_and_rag_module():
    """Validates DSPy signature input/output fields and RAGModule forward execution."""
    configure_dspy_lm(mode="offline")
    module = RAGModule()
    assert isinstance(module.generate, dspy.Predict)

    # Test forward pass with offline model
    pred = module(
        question="What is the token expiration period for short-lived access tokens?",
        context="[03_authentication_c001]\nShort-lived JWT access tokens expire after 15 minutes.",
    )

    assert hasattr(pred, "answer")
    assert hasattr(pred, "citations")
    assert len(pred.answer) > 0


def test_dspy_refusal_on_adversarial_queries():
    """Validates that OfflineRAGLM reliably produces refusal on ungrounded queries."""
    configure_dspy_lm(mode="offline")
    module = RAGModule()
    pred = module(
        question="What quantum computing hardware is deployed?",
        context="Infrastructure runs on EC2 c6i instances.",
    )

    assert "insufficient information" in pred.answer.lower()
    assert pred.citations == "none"


def test_rag_optimization_metric_behavior():
    """Tests teleprompter evaluation metric across refusal, correctness, and citations."""

    class MockPred(BaseModel):
        answer: str
        citations: str = ""

    # 1. Correct refusal on out-of-scope query
    gold_refusal = dspy.Example(answer="I have insufficient information to answer this question.")
    pred_refusal = MockPred(
        answer="Based on the provided documentation, there is insufficient information to answer this question.",
        citations="none",
    )
    assert rag_optimization_metric(gold_refusal, pred_refusal) == 1.0

    # 2. Failed refusal (hallucinated answer when refusal was expected)
    pred_bad_answer = MockPred(answer="The system uses IBM quantum processors.", citations="c1")
    assert rag_optimization_metric(gold_refusal, pred_bad_answer) == 0.0

    # 3. False refusal (refused when context had the answer)
    gold_fact = dspy.Example(answer="Tokens expire after 15 minutes.")
    assert rag_optimization_metric(gold_fact, pred_refusal) == 0.0

    # 4. Correct answer with citations
    pred_fact = MockPred(
        answer="Short-lived JWT tokens expire after 15 minutes.", citations="chunk_001"
    )
    score = rag_optimization_metric(gold_fact, pred_fact)
    assert score >= 0.70


def test_split_quarantine_integrity():
    """Ensures test split examples are strictly quarantined and never present in train or dev."""
    chunk_map = build_chunk_map("data/raw")
    train_exs, dev_exs, test_exs = load_dspy_splits("data/golden_qa.json", chunk_map)

    assert len(train_exs) == 32
    assert len(dev_exs) == 16
    assert len(test_exs) == 32

    train_ids = {ex.id for ex in train_exs}
    dev_ids = {ex.id for ex in dev_exs}
    test_ids = {ex.id for ex in test_exs}

    assert len(train_ids.intersection(test_ids)) == 0
    assert len(dev_ids.intersection(test_ids)) == 0
    assert len(train_ids.intersection(dev_ids)) == 0


def test_load_compiled_program_artifact(tmp_path):
    """Validates saving and reloading a compiled DSPy program."""
    configure_dspy_lm(mode="offline")
    module = RAGModule()
    # Add dummy demo
    demo = dspy.Example(question="Q?", context="C", answer="A", citations="c1").with_inputs(
        "question", "context"
    )
    module.generate.demos = [demo]

    save_path = tmp_path / "test_compiled_dspy.json"
    module.save(str(save_path))
    assert save_path.exists()

    reloaded = load_compiled_program(save_path)
    assert len(reloaded.generate.demos) == 1
    demo_first = reloaded.generate.demos[0]
    demo_q = demo_first["question"] if isinstance(demo_first, dict) else demo_first.question
    assert demo_q == "Q?"


def test_rag_generator_with_dspy_prompt_style():
    """Validates that RAGGenerator executes correctly with prompt_style='dspy'."""
    cfg = GenerationConfig(
        prompt_style="dspy",
        mode="offline",
        compiled_dspy_program_path="src/optimize/compiled_dspy_rag.json",
    )
    generator = RAGGenerator(cfg)
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            content="Short-lived JWT access tokens expire after 15 minutes.",
            score=0.95,
            rank=1,
        )
    ]
    res = generator.generate("What is the token expiration period?", chunks)

    assert len(res.answer) > 0
    assert res.generation_latency_ms >= 0.0
    assert not res.is_refusal

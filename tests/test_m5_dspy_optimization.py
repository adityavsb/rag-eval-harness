"""Unit tests for Milestone M5: DSPy generation prompt optimization and teleprompter compilation."""

import dspy
import pytest
from pydantic import BaseModel

from src.eval.llm_judge import LLMJudge
from src.optimize.compile import (
    build_chunk_map,
    demo_is_broken,
    evaluate_program_on_split,
    load_dspy_splits,
    rag_optimization_metric,
)
from src.optimize.dspy_program import (
    GenerateAnswerWithCitations,
    LenientChatAdapter,
    RAGModule,
    configure_dspy_lm,
    load_compiled_program,
    normalize_dspy_field_headers,
    openrouter_model_slug,
    resolve_dspy_provider,
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
        question="How does Snowflake configure micro-partition pruning for clustering keys?",
        context="[07_observability_and_telemetry_c002]\nTraces are exported via OpenTelemetry collectors with batch processors sending telemetry to external endpoints.",
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


def test_oracle_context_resolves_from_ground_truth_spans():
    """Every answerable DSPy example must receive chunks overlapping all its ground_truth_spans."""
    import json

    from src.eval.retrieval_metrics import is_chunk_relevant

    chunk_map = build_chunk_map("data/raw")
    train_exs, dev_exs, test_exs = load_dspy_splits("data/golden_qa.json", chunk_map)
    ex_by_id = {ex.id: ex for ex in (*train_exs, *dev_exs, *test_exs)}
    records = json.load(open("data/golden_qa.json", encoding="utf-8"))

    for item in records:
        ex = ex_by_id[item["id"]]
        gt_spans = item.get("ground_truth_spans", [])
        if not gt_spans:
            assert ex.context == "No supporting context available."
            assert ex.citations == "none"
            continue

        cited_ids = [cid.strip() for cid in ex.citations.split(",") if cid.strip()]
        assert cited_ids, f"{item['id']} has no oracle chunk IDs"
        assert ex.context != "No supporting context available."

        # Every ground-truth span must be covered by at least one oracle chunk
        for sp in gt_spans:
            covered = any(
                is_chunk_relevant(
                    {
                        "doc": chunk_map[cid].metadata.get("doc", cid.split("_c")[0]),
                        "char_start": chunk_map[cid].metadata["char_start"],
                        "char_end": chunk_map[cid].metadata["char_end"],
                    },
                    [sp],
                    min_overlap=0.25,
                )
                for cid in cited_ids
            )
            assert covered, f"{item['id']} oracle chunks {cited_ids} missed span {sp}"

    # Spot-check previously broken test items where old chunk_ids pointed to wrong passages
    assert (
        "PostgreSQL 16" in ex_by_id["qa_004"].context
        or "patroni" in ex_by_id["qa_004"].context.lower()
    )
    assert "403" in ex_by_id["qa_014"].context or "RBAC" in ex_by_id["qa_014"].context


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


def test_openrouter_slug_and_provider_resolution():
    """DeepSeek and Qwen route through LiteLLM's openrouter/ prefix."""
    assert openrouter_model_slug("deepseek/deepseek-v4.1-flash") == (
        "openrouter/deepseek/deepseek-v4.1-flash"
    )
    assert openrouter_model_slug("qwen/qwen3.7-flash") == "openrouter/qwen/qwen3.7-flash"
    assert openrouter_model_slug("openrouter/qwen/qwen3.7-flash") == (
        "openrouter/qwen/qwen3.7-flash"
    )
    assert resolve_dspy_provider("openrouter", "deepseek/deepseek-v4.1-flash") == "openrouter"
    assert resolve_dspy_provider(None, "qwen/qwen3.7-flash") == "openrouter"
    assert resolve_dspy_provider("openai", "gpt-4o-mini") == "openai"


def test_dspy_live_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is missing"):
        configure_dspy_lm(
            mode="live",
            model_name="deepseek/deepseek-v4.1-flash",
            provider="openrouter",
        )


def test_dspy_live_openrouter_builds_litellm_client(monkeypatch):
    captured = {}

    class FakeLM:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setattr("src.optimize.dspy_program.dspy.LM", FakeLM)
    monkeypatch.setattr("src.optimize.dspy_program.dspy.settings.configure", lambda **kwargs: None)
    configure_dspy_lm(
        mode="live",
        model_name="qwen/qwen3.7-flash",
        provider="openrouter",
    )
    assert captured["model"] == "openrouter/qwen/qwen3.7-flash"
    assert captured["kwargs"]["api_base"] == "https://openrouter.ai/api/v1"
    assert captured["kwargs"]["engine"] == "litellm"
    assert captured["kwargs"]["api_key"] == "sk-or-test-key"
    assert captured["kwargs"]["max_tokens"] == 4000


def test_configure_dspy_lm_default_max_tokens(monkeypatch):
    captured = {}

    class FakeLM:
        def __init__(self, model, **kwargs):
            captured["max_tokens"] = kwargs.get("max_tokens")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setattr("src.optimize.dspy_program.dspy.LM", FakeLM)
    monkeypatch.setattr("src.optimize.dspy_program.dspy.settings.configure", lambda **kwargs: None)
    configure_dspy_lm(mode="live", model_name="deepseek/deepseek-v4.1-flash", provider="openrouter")
    assert captured["max_tokens"] == 4000


def test_normalize_dspy_field_headers_fixes_missing_space():
    raw = "[[ ## citations ##]]"
    assert normalize_dspy_field_headers(raw) == "[[ ## citations ## ]]"


def test_lenient_chat_adapter_parses_export_job_style_reply():
    adapter = LenientChatAdapter()
    completion = """[[ ## citations ##]]
08_asynchronous_worker_fleet_c002, 06_database_migrations_and_schema_c006

[[ ## answer ## ]]
The context states that bulk export jobs such as PDF exports are assigned to the bulk priority tier with an SLA of under 15 minutes.

[[ ## completed ## ]]"""
    fields = adapter.parse(GenerateAnswerWithCitations, completion)
    assert "citations" in fields
    assert "answer" in fields
    assert "08_asynchronous_worker_fleet_c002" in fields["citations"]


def test_evaluate_program_on_split_continues_on_parse_failure():
    from dspy.utils.exceptions import AdapterParseError

    class BrokenModule:
        def __call__(self, question: str, context: str):
            raise AdapterParseError(
                adapter_name="LenientChatAdapter",
                signature=GenerateAnswerWithCitations,
                lm_response="",
            )

    judge = LLMJudge(mode="offline")
    ex = dspy.Example(
        question="Q?",
        context="C",
        answer="A",
        query_type="factoid",
    ).with_inputs("question", "context")
    metrics = evaluate_program_on_split(BrokenModule(), [ex], judge)
    assert metrics["parse_failures"] == 1.0
    assert metrics["answer_correctness"] == 0.0
    assert metrics["n_answerable"] == 1.0


def test_evaluate_program_on_split_excludes_negatives_from_correctness():
    """Correctness and faithfulness must average over answerable items only, not unanswerable traps."""

    class FixedModule:
        def __call__(self, question: str, context: str):
            if "Snowflake" in question:
                return dspy.Prediction(
                    answer="Based on the provided documentation, there is insufficient information to answer this question.",
                    citations="none",
                )
            return dspy.Prediction(answer="Totally wrong answer.", citations="c1")

    judge = LLMJudge(mode="offline")
    examples = [
        dspy.Example(
            question="What is JWT TTL?",
            context="[c1]\nShort-lived JWT access tokens expire after 15 minutes.",
            answer="Short-lived JWT access tokens expire after 15 minutes.",
            query_type="factoid",
        ).with_inputs("question", "context"),
        dspy.Example(
            question="How does Snowflake cluster?",
            context="No supporting context available.",
            answer="Based on the provided documentation, there is insufficient information to answer this question.",
            query_type="adversarial_negative",
        ).with_inputs("question", "context"),
    ]

    metrics = evaluate_program_on_split(FixedModule(), examples, judge)
    assert metrics["n_answerable"] == 1.0
    assert metrics["n_negative"] == 1.0
    assert metrics["negative_refusal_rate"] == 1.0
    # If the negative refusal (1.0) were mixed in, correctness would be 0.5 instead of 0.0
    assert metrics["answer_correctness"] == 0.0


def test_demo_sanity_gate_rejects_fragments():
    assert demo_is_broken({"answer": "mid sentence fragment."})
    assert demo_is_broken({"answer": "# Heading"})
    assert demo_is_broken({"answer": ""})
    assert not demo_is_broken({"answer": "Tokens expire after 15 minutes."})


def test_offline_lm_is_demo_agnostic():
    """The mock must not change behaviour based on the presence of demonstrations."""
    from src.optimize.dspy_program import OfflineRAGLM

    lm = OfflineRAGLM()
    msg = {
        "role": "user",
        "content": "[c1]\nShort-lived JWT access tokens expire after 15 minutes.\n\nQuestion: What is the token expiration period?",
    }
    no_demo = lm(messages=[{"role": "system", "content": "x"}, msg])
    with_demo = lm(
        messages=[
            {"role": "system", "content": "x"},
            {"role": "user", "content": "demo"},
            {"role": "assistant", "content": "demo"},
            msg,
        ]
    )
    assert no_demo == with_demo

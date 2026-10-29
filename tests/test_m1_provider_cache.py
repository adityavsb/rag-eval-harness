"""Unit tests for Milestone M1: LiteLLM provider, disk response caching, and EUR pricing."""

from pathlib import Path

import pytest

from src.llm.cache import DiskResponseCache
from src.llm.pricing import estimate_batch_cost_eur, estimate_cost_eur
from src.llm.provider import call_llm


def test_pricing_calculations():
    # gpt-4o-mini: $0.15 / $0.60 per 1M tokens converted at USD_EUR_RATE (0.94) -> ~0.705 EUR
    cost = estimate_cost_eur("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - 0.705) < 0.02

    # Ollama / local is always 0.0 EUR
    ollama_cost = estimate_cost_eur("ollama/qwen2.5:7b", 500_000, 500_000)
    assert ollama_cost == 0.0

    batch_cost = estimate_batch_cost_eur("gpt-4o-mini", num_queries=30)
    assert batch_cost > 0.0
    assert batch_cost < 0.10


def test_pricing_unknown_model_raises_key_error():
    with pytest.raises(KeyError, match="No price for 'unknown-unpriced-model'"):
        estimate_cost_eur("unknown-unpriced-model", 1000, 1000)


def test_disk_cache_operations(tmp_path: Path):
    cache = DiskResponseCache(cache_dir=str(tmp_path))

    model = "test-model"
    prompt = "What is the architecture of FastAPI?"
    sample_payload = {
        "content": "FastAPI is a modern web framework.",
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    }

    # Initial get is None
    assert cache.get(model, prompt) is None

    # Set and get
    cache.set(model, prompt, sample_payload)
    retrieved = cache.get(model, prompt)
    assert retrieved is not None
    assert retrieved["content"] == sample_payload["content"]

    # Clear
    removed = cache.clear()
    assert removed == 1
    assert cache.get(model, prompt) is None


def test_call_llm_cache_hit_avoids_api_call(tmp_path: Path):
    cache = DiskResponseCache(cache_dir=str(tmp_path))
    model = "mock-model"
    prompt = "Explain token bucket rate limiting"

    # Pre-seed cache
    cache.set(
        model,
        prompt,
        {
            "content": "Token bucket allows bursts up to bucket capacity.",
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25,
        },
    )

    # Call LLM with use_cache=True pointing to tmp_path
    response = call_llm(
        model=model,
        prompt=prompt,
        use_cache=True,
        cache_dir=str(tmp_path),
    )

    assert response.cached is True
    assert response.cost_eur == 0.0
    assert response.content == "Token bucket allows bursts up to bucket capacity."


def test_call_openrouter_decisions_missing_key_raises(monkeypatch):
    from src.llm.provider import call_openrouter_decisions

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is missing"):
        call_openrouter_decisions(
            model="perplexity/pplx-decider-flash-v1.2",
            state={"query": "test"},
            questions={"q": {"type": "noul"}},
        )


def test_judge_is_decision_model_property():
    from src.eval.llm_judge import LLMJudge

    judge_decider = LLMJudge(model_name="perplexity/pplx-decider-flash-v1.2", mode="live")
    assert judge_decider.is_decision_model is True

    judge_chat = LLMJudge(model_name="gpt-4o-mini", mode="live")
    assert judge_chat.is_decision_model is False


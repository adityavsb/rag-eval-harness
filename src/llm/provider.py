"""Unified LLM provider adapter using LiteLLM with disk caching and cost tracking."""

from pydantic import BaseModel, Field

from src.llm.cache import DiskResponseCache
from src.llm.pricing import estimate_cost_eur


class LLMResponse(BaseModel):
    """Encapsulates response text, token usage, cost, and cache provenance."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_eur: float = 0.0
    cached: bool = False
    model: str
    raw_response: dict = Field(default_factory=dict)


def call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    use_cache: bool = True,
    cache_dir: str = ".cache/llm_responses",
) -> LLMResponse:
    """Invokes LLM via LiteLLM with automated on-disk response caching and cost computation."""
    cache = DiskResponseCache(cache_dir=cache_dir) if use_cache else None

    # Check disk cache first
    if cache:
        cached_data = cache.get(model=model, prompt=prompt)
        if cached_data:
            return LLMResponse(
                content=cached_data["content"],
                prompt_tokens=cached_data.get("prompt_tokens", 0),
                completion_tokens=cached_data.get("completion_tokens", 0),
                total_tokens=cached_data.get("total_tokens", 0),
                cost_eur=0.0,  # Cached execution has zero additional incremental cost
                cached=True,
                model=model,
                raw_response=cached_data.get("raw_response", {}),
            )

    import litellm

    # Suppress verbose litellm logging in tests/evals
    litellm.suppress_debug_info = True

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"LiteLLM call failed for model '{model}': {e}") from e

    choice = response.choices[0]
    content = choice.message.content or ""

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else len(prompt.split())
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else len(content.split())
    total_tokens = prompt_tokens + completion_tokens

    cost = estimate_cost_eur(model, prompt_tokens, completion_tokens)

    payload = {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_eur": cost,
        "model": model,
    }

    if cache:
        cache.set(model=model, prompt=prompt, response=payload)

    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_eur=cost,
        cached=False,
        model=model,
    )

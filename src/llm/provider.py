"""Unified LLM provider adapter using LiteLLM with disk caching, OpenRouter routing, and cost tracking."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from src.llm.cache import DiskResponseCache
from src.llm.pricing import estimate_cost_eur
from src.utils.config import load_dotenv

# Ensure local environment variables are loaded
load_dotenv()


def mask_secret(secret: str | None) -> str:
    """Return a masked representation of an API secret (e.g. sk-or-...cdef)."""
    if not secret:
        return "[NOT SET]"
    if len(secret) <= 8:
        return "***MASKED***"
    return f"{secret[:6]}...{secret[-4:]}"


def _provider_debug_errors() -> bool:
    """When true, exception messages may include capped, redacted provider detail."""
    return os.environ.get("RAG_EVAL_DEBUG_PROVIDER_ERRORS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _sanitize_error_detail(text: str, max_len: int = 200) -> str:
    import re

    redacted = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    redacted = re.sub(r"sk-[a-zA-Z0-9_-]{8,}", "sk-[REDACTED]", redacted)
    if len(redacted) > max_len:
        return redacted[:max_len] + "..."
    return redacted


def _format_litellm_failure(model_slug: str, max_retries: int, last_error: Exception | None) -> str:
    err_cls = type(last_error).__name__ if last_error else "UnknownError"
    msg = f"LiteLLM call failed for model '{model_slug}' after {max_retries} attempts ({err_cls})"
    if _provider_debug_errors() and last_error:
        msg += f": {_sanitize_error_detail(str(last_error))}"
    return msg


def _format_openrouter_http_error(http_code: int, body: str) -> str:
    if _provider_debug_errors():
        return f"OpenRouter decisions HTTP {http_code}: {_sanitize_error_detail(body)}"
    return f"OpenRouter decisions HTTP {http_code} (enable RAG_EVAL_DEBUG_PROVIDER_ERRORS=1 for detail)"


def _format_openrouter_failure(max_retries: int, last_error: Exception | None) -> str:
    err_cls = type(last_error).__name__ if last_error else "UnknownError"
    msg = f"OpenRouter decisions call failed after {max_retries} attempts ({err_cls})"
    if _provider_debug_errors() and last_error:
        msg += f": {_sanitize_error_detail(str(last_error))}"
    return msg

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
    timeout: float = 120.0,
    max_retries: int = 3,
    use_cache: bool = True,
    cache_dir: str = ".cache/llm_responses",
) -> LLMResponse:
    """Invokes LLM via LiteLLM with automated on-disk response caching, OpenRouter headers, and cost computation."""
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

    # Normalize OpenRouter model slug
    model_slug = model
    extra_headers: dict[str, str] = {}
    api_key = None

    is_openrouter = (
        model_slug.startswith("openrouter/")
        or "deepseek" in model_slug
        or "qwen" in model_slug
        or "perplexity" in model_slug
        or "pplx-decider" in model_slug
        or "nemotron" in model_slug
        or "free" in model_slug
    )

    if is_openrouter:
        if not model_slug.startswith("openrouter/"):
            model_slug = f"openrouter/{model_slug}"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        extra_headers = {
            "HTTP-Referer": "https://github.com/adityavsb/rag-eval-harness",
            "X-Title": "rag-eval-harness",
        }

    response: Any = None
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model_slug,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if api_key:
                kwargs["api_key"] = api_key
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            response = litellm.completion(**kwargs)
            break
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if attempt < max_retries - 1 and (
                "429" in err_str or "503" in err_str or "rate" in err_str or "overloaded" in err_str
            ):
                time.sleep(2.0 * (attempt + 1))
            else:
                break

    if response is None:
        raise RuntimeError(
            _format_litellm_failure(model_slug, max_retries, last_error)
        ) from last_error

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


def call_openrouter_decisions(
    model: str,
    state: dict[str, Any],
    questions: dict[str, Any],
    timeout: float = 60.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Invokes OpenRouter /api/alpha/decisions for deterministic decision models (e.g. perplexity/pplx-decider-flash-v1.2)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is missing for OpenRouter decisions call."
        )

    url = "https://openrouter.ai/api/alpha/decisions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/adityavsb/rag-eval-harness",
        "X-Title": "rag-eval-harness",
    }
    payload = {
        "model": model,
        "state": state,
        "questions": questions,
    }
    data = json.dumps(payload).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_error = e
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(_format_openrouter_http_error(e.code, err_body)) from e
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(_format_openrouter_failure(max_retries, last_error)) from e

    raise RuntimeError(_format_openrouter_failure(max_retries, last_error))

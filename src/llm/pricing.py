"""Token pricing and cost estimation in EUR for multi-model RAG evaluations."""

PRICE_TABLE_DATE = "2026-10-24"  # update when any row changes
USD_EUR_RATE = 0.94  # ECB reference rate on PRICE_TABLE_DATE
PRICE_SOURCES = {
    "openai": "https://openai.com/api/pricing/",
    "anthropic": "https://www.anthropic.com/pricing",
    "google": "https://ai.google.dev/pricing",
}

# USD list prices per 1M tokens (input, output); EUR derived at USD_EUR_RATE.
MODEL_PRICE_TABLE_USD: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "ollama": (0.0, 0.0),
    "mock": (0.0, 0.0),
}

MODEL_PRICE_TABLE: dict[str, tuple[float, float]] = {
    k: (round(i * USD_EUR_RATE, 4), round(o * USD_EUR_RATE, 4))
    for k, (i, o) in MODEL_PRICE_TABLE_USD.items()
}


def get_model_rates(model_name: str) -> tuple[float, float]:
    """Retrieves (prompt_eur_per_million, completion_eur_per_million) for model."""
    normalized = (
        model_name.lower()
        .replace("openai/", "")
        .replace("anthropic/", "")
        .replace("vertex_ai/", "")
    )

    if (
        "ollama" in normalized
        or "qwen" in normalized
        or "llama" in normalized
        or "local" in normalized
        or "mock" in normalized
    ):
        return 0.0, 0.0

    for key, rates in MODEL_PRICE_TABLE.items():
        if key in normalized:
            return rates

    raise KeyError(
        f"No price for '{model_name}' in MODEL_PRICE_TABLE (dated {PRICE_TABLE_DATE}); add a row."
    )


def estimate_cost_eur(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates actual cost in EUR based on token consumption."""
    input_rate, output_rate = get_model_rates(model_name)
    cost = (prompt_tokens / 1_000_000.0) * input_rate + (
        completion_tokens / 1_000_000.0
    ) * output_rate
    return round(cost, 6)


def estimate_batch_cost_eur(
    model_name: str,
    num_queries: int,
    avg_prompt_tokens: int = 1200,
    avg_completion_tokens: int = 250,
) -> float:
    """Projected cost in EUR for a batch evaluation run."""
    single_cost = estimate_cost_eur(model_name, avg_prompt_tokens, avg_completion_tokens)
    return round(single_cost * num_queries, 4)

"""Token pricing and cost estimation in EUR for multi-model RAG evaluations."""

# Price per 1,000,000 tokens (input_eur_per_m, output_eur_per_m)
MODEL_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.14, 0.56),
    "gpt-4o": (2.35, 9.40),
    # Anthropic
    "claude-3-5-haiku": (0.75, 3.75),
    "claude-3-5-haiku-20241022": (0.75, 3.75),
    "claude-3-5-sonnet": (2.80, 14.00),
    # Google Vertex AI
    "gemini-1.5-flash": (0.07, 0.28),
    "gemini-2.0-flash": (0.09, 0.36),
    "gemini-1.5-pro": (1.18, 4.70),
    # Ollama / Local / Mock
    "ollama": (0.0, 0.0),
    "mock": (0.0, 0.0),
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
    ):
        return 0.0, 0.0

    for key, rates in MODEL_PRICE_TABLE.items():
        if key in normalized:
            return rates

    # Default fallback: conservative small model pricing
    return 0.15, 0.60


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

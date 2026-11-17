"""DSPy prompt compilation and optimization package."""

from src.optimize.dspy_program import (
    GenerateAnswerWithCitations,
    LenientChatAdapter,
    OfflineRAGLM,
    RAGModule,
    configure_dspy_lm,
    load_compiled_program,
    normalize_dspy_field_headers,
    openrouter_model_slug,
    resolve_dspy_provider,
)

__all__ = [
    "GenerateAnswerWithCitations",
    "RAGModule",
    "OfflineRAGLM",
    "configure_dspy_lm",
    "LenientChatAdapter",
    "load_compiled_program",
    "normalize_dspy_field_headers",
    "openrouter_model_slug",
    "resolve_dspy_provider",
]

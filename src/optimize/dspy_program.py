"""DSPy signature, module, and model configuration for RAG prompt optimization."""

import json
import os
import re
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature

# Accept headers with or without the space before the closing brackets.
_LOOSE_FIELD_HEADER = re.compile(r"\[\[\s*##\s*(\w+)\s*##\s*\]\]")


def normalize_dspy_field_headers(completion: str) -> str:
    """Rewrites near-miss DSPy field headers to the canonical ``[[ ## name ## ]]`` form."""
    normalized_lines = []
    for line in completion.splitlines():
        stripped = line.strip()
        if _LOOSE_FIELD_HEADER.fullmatch(stripped):
            name = _LOOSE_FIELD_HEADER.fullmatch(stripped).group(1)
            normalized_lines.append(f"[[ ## {name} ## ]]")
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


class LenientChatAdapter(ChatAdapter):
    """ChatAdapter that tolerates minor header formatting drift and never JSON-retries."""

    def __init__(self, **kwargs):
        super().__init__(use_json_adapter_fallback=False, **kwargs)

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        return super().parse(signature, normalize_dspy_field_headers(completion))


class GenerateAnswerWithCitations(dspy.Signature):
    """Answer the user's question factually and concisely using ONLY the provided context passages.

    If the context does not contain sufficient facts to answer the question with certainty,
    output 'Based on the provided documentation, there is insufficient information to answer this question.'
    and set citations to 'none'.
    """

    context: str = dspy.InputField(
        desc="Retrieved reference context passages formatted with bracketed chunk IDs"
    )
    question: str = dspy.InputField(desc="The user question to be answered")
    citations: str = dspy.OutputField(
        desc="Comma-separated list of chunk IDs directly supporting the answer, or 'none'"
    )
    answer: str = dspy.OutputField(
        desc="Factual, grounded answer without fabrication or outside knowledge"
    )


class RAGModule(dspy.Module):
    """DSPy RAG generation module with Predict layer over GenerateAnswerWithCitations."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateAnswerWithCitations)

    def forward(self, question: str, context: str) -> dspy.Prediction:
        return self.generate(question=question, context=context)


class OfflineRAGLM(dspy.LM):
    """Deterministic offline language model for DSPy compilation and local execution."""

    def __init__(self, model: str = "offline-rag-generator"):
        super().__init__(model=model)

    def __call__(self, prompt=None, messages=None, **kwargs) -> list[str]:
        if messages and isinstance(messages, list) and len(messages) > 0:
            target_msg = messages[-1].get("content", "")
        else:
            target_msg = str(prompt or "")

        # Extract chunk ID and snippet from context of current message
        chunk_matches = list(re.finditer(r"\[([\w_]+)\]\s*([^\[]+)", target_msg))

        # Extract question text
        q_match = re.search(
            r"(?:Question|\[\[ ## question ## \]\]):?\s*([^\n]+)",
            target_msg,
            re.IGNORECASE,
        )
        q_text = q_match.group(1).lower() if q_match else ""
        q_words = {w for w in re.findall(r"\w+", q_text) if len(w) > 3}

        # Content-based refusal: check word and prefix overlap between question and all context words
        all_ctx_words = {
            w for cm in chunk_matches for w in re.findall(r"\w+", cm.group(2).lower()) if len(w) > 3
        }
        overlap = (q_words & all_ctx_words) | {
            qw for qw in q_words if any(qw[:4] == cw[:4] for cw in all_ctx_words)
        }
        if not chunk_matches or len(overlap) < 2:
            res = {
                "answer": "Based on the provided documentation, there is insufficient information to answer this question.",
                "citations": "none",
            }
            return [json.dumps(res)]

        best_chunk_id = chunk_matches[0].group(1)
        best_sentence = ""
        best_score = -1

        for cm in chunk_matches:
            cid = cm.group(1)
            body = cm.group(2).strip()
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[a-zA-Z0-9\)])\.\s+(?=[A-Z])", body)
                if len(s.strip()) > 10
            ]
            for s in sentences:
                s_words = {w for w in re.findall(r"\w+", s.lower()) if len(w) > 3}
                overlap = len(s_words.intersection(q_words))
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = s
                    best_chunk_id = cid

        if not best_sentence:
            best_sentence = chunk_matches[0].group(2).strip().split("\n")[0]

        clean_sentence = best_sentence.strip()
        if not clean_sentence.endswith("."):
            clean_sentence = f"{clean_sentence}."

        answer_str = clean_sentence
        cit_str = best_chunk_id

        res = {
            "answer": answer_str,
            "citations": cit_str,
        }
        return [json.dumps(res)]


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_MODEL_HINTS = (
    "deepseek",
    "qwen",
    "perplexity",
    "pplx-decider",
    ":free",
    "nemotron",
)


def resolve_dspy_provider(provider: str | None, model_name: str) -> str:
    """Picks the live DSPy backend. OpenRouter slugs win over a stale OpenAI default."""
    explicit = (provider or "").strip().lower()
    model = model_name.lower()
    looks_openrouter = model.startswith("openrouter/") or any(
        hint in model for hint in _OPENROUTER_MODEL_HINTS
    )
    if explicit == "openrouter" or looks_openrouter:
        return "openrouter"
    if explicit in ("", "mock", "offline"):
        return "openai"
    return explicit


def openrouter_model_slug(model_name: str) -> str:
    """LiteLLM OpenRouter id, e.g. deepseek/deepseek-v4.1-flash -> openrouter/deepseek/deepseek-v4.1-flash."""
    if model_name.startswith("openrouter/"):
        return model_name
    return f"openrouter/{model_name}"


def sum_history_tokens(lm: dspy.LM, start: int = 0) -> tuple[int, int]:
    """Sums non-cached prompt and completion tokens from a DSPy LM history."""
    history = getattr(lm, "history", None) or []
    prompt_tokens = 0
    completion_tokens = 0
    for entry in history[start:]:
        response = entry.get("response") if isinstance(entry, dict) else None
        if getattr(response, "cache_hit", False):
            continue
        usage = entry.get("usage") if isinstance(entry, dict) else None
        if not isinstance(usage, dict):
            continue
        prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt_tokens, completion_tokens


def _use_project_dspy_cache() -> None:
    """Keep DSPy's disk cache inside the repo so a missing home directory does not disable it."""
    cache_dir = Path(".cache/dspy")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DSPY_CACHEDIR", str(cache_dir.resolve()))


def configure_dspy_lm(
    mode: str = "offline",
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
    provider: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> dspy.LM:
    """Configures DSPy global language model settings for offline or live execution."""
    _use_project_dspy_cache()
    if mode == "live":
        resolved = resolve_dspy_provider(provider, model_name)
        if resolved == "openrouter":
            key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ValueError(
                    f"Live mode requested for DSPy with model '{model_name}', "
                    "but OPENROUTER_API_KEY environment variable is missing."
                )
            import litellm

            litellm.suppress_debug_info = True
            lm = dspy.LM(
                openrouter_model_slug(model_name),
                api_key=key,
                api_base=OPENROUTER_API_BASE,
                temperature=temperature,
                max_tokens=max_tokens,
                engine="litellm",
                extra_headers={
                    "HTTP-Referer": "https://github.com/adityavsb/rag-eval-harness",
                    "X-Title": "rag-eval-harness",
                },
            )
        elif resolved == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    f"Live mode requested for DSPy with model '{model_name}', "
                    "but OPENAI_API_KEY environment variable is missing."
                )
            lm = dspy.LM(
                f"openai/{model_name}",
                api_key=key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(
                f"Live DSPy provider '{resolved}' is not supported. Use 'openrouter' "
                "(deepseek/deepseek-v4.1-flash or qwen/qwen3.7-flash) or 'openai'."
            )
    else:
        lm = OfflineRAGLM(model=f"offline-{model_name}")

    if mode == "live":
        dspy.settings.configure(lm=lm, adapter=LenientChatAdapter())
    else:
        dspy.settings.configure(lm=lm)
    return lm


def load_compiled_program(path: str | Path) -> RAGModule:
    """Loads a compiled DSPy program from JSON artifact."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Compiled DSPy program not found at: {target_path}")

    module = RAGModule()
    module.load(str(target_path))
    return module

"""Grounded LLM response generation with context binding and refusal handling."""

import os
import time

from pydantic import BaseModel, Field

from src.pipeline.retriever import RetrievedChunk
from src.utils.config import GenerationConfig


class GenerationResult(BaseModel):
    """Encapsulates generated answer, citations, and execution telemetry."""

    query: str
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    generation_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_refusal: bool = False


NAIVE_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user question based on the following context:

Context:
{context}

Question: {query}
Answer:"""

GROUNDED_PROMPT_TEMPLATE = """You are an enterprise AI assistant answering technical questions strictly based on the provided context.

Context:
{context}

Instructions:
1. Rely ONLY on the clear facts directly mentioned in the context.
2. If the context does not contain enough information to answer the question with complete confidence, explicitly state: "Based on the provided documentation, there is insufficient information to answer this question."
3. Mention the relevant source chunk ID (e.g. [chunk_id]) whenever referencing facts.
4. Do NOT fabricate or extrapolate information.

Question: {query}
Answer:"""


class RAGGenerator:
    """Generates answers conditioned on retrieved contexts using configured prompt strategies."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._llm = None

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formats retrieved chunks with structured ID tags for citation attribution."""
        if not chunks:
            return "No relevant context found."

        formatted_pieces = []
        for chunk in chunks:
            formatted_pieces.append(f"[{chunk.chunk_id}]\n{chunk.content}")
        return "\n\n".join(formatted_pieces)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        """Generates a grounded answer from query and retrieved chunks."""
        context_str = self._format_context(chunks)

        if self.config.prompt_style == "naive":
            prompt = NAIVE_PROMPT_TEMPLATE.format(context=context_str, query=query)
        else:
            prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context_str, query=query)

        start_time = time.perf_counter()

        if self.config.mode == "live":
            # Live model generation requires valid credentials
            provider = self.config.provider.lower()
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"Live mode generation requested with model '{self.config.model_name}', "
                        "but OPENAI_API_KEY environment variable is missing."
                    )
                try:
                    from openai import OpenAI

                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    answer_text = response.choices[0].message.content or ""
                    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                    completion_tokens = response.usage.completion_tokens if response.usage else 0
                except (RuntimeError, ValueError, KeyError, TypeError, OSError) as e:
                    raise RuntimeError(f"Error during live LLM generation: {e}") from e
            elif provider in ("ollama", "local"):
                # Ollama local inference (free, no API key needed)
                try:
                    import json
                    import urllib.request

                    req_data = json.dumps(
                        {
                            "model": self.config.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": self.config.temperature},
                        }
                    ).encode("utf-8")
                    req = urllib.request.Request(
                        "http://localhost:11434/api/generate",
                        data=req_data,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        answer_text = res.get("response", "")
                        prompt_tokens = res.get("prompt_eval_count", len(prompt.split()))
                        completion_tokens = res.get("eval_count", len(answer_text.split()))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to connect to local Ollama server at http://localhost:11434: {e}"
                    ) from e
            else:
                raise ValueError(f"Unsupported provider '{self.config.provider}' in live mode.")
        else:
            # Deterministic offline mock generator for testing & evaluation without API costs
            if not chunks:
                answer_text = "Based on the provided documentation, there is insufficient information to answer this question."
            else:
                top_chunk = chunks[0]
                answer_text = f"According to [{top_chunk.chunk_id}], {top_chunk.content[:200]}..."
            prompt_tokens = len(prompt.split())
            completion_tokens = len(answer_text.split())

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        is_refusal = "insufficient information" in answer_text.lower()
        cited_ids = [c.chunk_id for c in chunks if f"[{c.chunk_id}]" in answer_text]

        return GenerationResult(
            query=query,
            answer=answer_text,
            cited_chunk_ids=cited_ids,
            generation_latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_refusal=is_refusal,
        )

"""Grounded LLM response generation with context binding and refusal handling."""

import os
import time
from pathlib import Path

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

        if self.config.prompt_style == "dspy":
            from src.optimize.dspy_program import (
                RAGModule,
                configure_dspy_lm,
                load_compiled_program,
            )

            start_time = time.perf_counter()
            configure_dspy_lm(mode=self.config.mode, model_name=self.config.model_name)
            dspy_path = Path(self.config.compiled_dspy_program_path)
            if dspy_path.exists():
                prog = load_compiled_program(dspy_path)
            else:
                prog = RAGModule()
            pred = prog(question=query, context=context_str)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            answer_text = getattr(pred, "answer", "")
            citations_raw = getattr(pred, "citations", "")
            citations = [
                c.strip()
                for c in citations_raw.split(",")
                if c.strip() and c.strip().lower() != "none"
            ]
            is_ref = "insufficient information" in answer_text.lower()
            return GenerationResult(
                query=query,
                answer=answer_text,
                cited_chunk_ids=citations,
                generation_latency_ms=latency_ms,
                prompt_tokens=len(context_str.split()) + len(query.split()),
                completion_tokens=len(answer_text.split()),
                is_refusal=is_ref,
            )

        if self.config.prompt_style == "naive":
            prompt = NAIVE_PROMPT_TEMPLATE.format(context=context_str, query=query)
        else:
            prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context_str, query=query)

        start_time = time.perf_counter()

        if self.config.mode == "live":
            # Live model generation requires valid credentials
            provider = self.config.provider.lower()
            model_target = self.config.model_name
            if provider in ("ollama", "local") and not model_target.startswith("ollama/"):
                model_target = f"ollama/{model_target}"

            if provider == "openai" or "gpt" in model_target:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"Live mode generation requested with model '{model_target}', "
                        "but OPENAI_API_KEY environment variable is missing."
                    )
            elif provider == "anthropic" or "claude" in model_target:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"Live mode generation requested with model '{model_target}', "
                        "but ANTHROPIC_API_KEY environment variable is missing."
                    )

            from src.llm.provider import call_llm

            llm_res = call_llm(
                model=model_target,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            answer_text = llm_res.content
            prompt_tokens = llm_res.prompt_tokens
            completion_tokens = llm_res.completion_tokens
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

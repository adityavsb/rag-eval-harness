"""Agentic Corrective RAG (CRAG) pipeline using LangGraph with iterative query decomposition, document grading, and multi-hop retrieval."""

import json
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.pipeline.generator import GROUNDED_PROMPT_TEMPLATE, GenerationResult
from src.pipeline.retriever import RAGRetriever, RetrievedChunk
from src.utils.config import GenerationConfig


class AgenticRAGState(TypedDict):
    """Execution state for the LangGraph Agentic RAG workflow."""

    query: str
    sub_queries: list[str]
    current_sub_idx: int
    retrieved_chunks: list[RetrievedChunk]
    retrieved_ids: list[str]
    answer: str
    cited_chunk_ids: list[str]
    hop_count: int
    max_hops: int
    is_refusal: bool
    prompt_tokens: int
    completion_tokens: int
    route: str


def _decompose_query_offline(query: str) -> list[str]:
    """Deterministic query decomposition for offline / testing mode."""
    split_patterns = [
        r"\s+and\s+how\s+",
        r"\s+and\s+what\s+",
        r"\s+coordinate\s+with\s+",
        r"\s+interact\s+with\s+",
        r"\s+across\s+",
        r",\s+and\s+",
    ]
    for pattern in split_patterns:
        parts = re.split(pattern, query, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and len(parts[0].strip()) > 10 and len(parts[1].strip()) > 10:
            part0 = parts[0].strip()
            part1 = parts[1].strip()
            if not part1.lower().startswith(("how", "what", "which", "where", "why", "describe")):
                part1 = f"How does the system handle {part1}"
            return [part0, part1]

    # If question contains 'between X and Y', extract both concepts
    between_match = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
    if between_match:
        c1, c2 = between_match.group(1).strip(), between_match.group(2).strip()
        if len(c1) > 5 and len(c2) > 5:
            return [f"What is {c1}?", f"What is {c2}?"]

    return [query]


def _decompose_query_live(query: str, model_name: str, timeout: float = 60.0) -> tuple[list[str], int, int]:
    """Decompose complex queries into 1 to 2 targeted sub-queries via LLM."""
    from src.llm.provider import call_llm

    prompt = f"""You are a technical query analysis agent for an enterprise documentation search system.
Analyze the following user query. If it asks about relationships, coordination, or comparisons between multiple distinct system components, decompose it into 1 or 2 focused, standalone sub-queries.
If it is already a single focused question, return just the original question in the list.

Respond ONLY with a JSON array of strings, e.g. ["sub-query 1", "sub-query 2"].

Question: {query}
"""
    res = call_llm(model=model_name, prompt=prompt, temperature=0.0, timeout=timeout)
    content = res.content.strip()
    # Extract json array
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(content[start : end + 1])
            if isinstance(arr, list) and arr and all(isinstance(x, str) for x in arr):
                return arr[:2], res.prompt_tokens, res.completion_tokens
        except json.JSONDecodeError:
            pass
    return [query], res.prompt_tokens, res.completion_tokens



def _grade_chunks_offline(query: str, chunks: list[RetrievedChunk]) -> bool:
    """Deterministic document relevance grading for offline execution."""
    if not chunks:
        return False

    stopwords = {
        "what", "how", "why", "which", "where", "when", "does", "the", "system",
        "and", "with", "from", "for", "are", "is", "can", "occur", "handle",
    }
    q_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3 and w not in stopwords]
    if not q_words:
        return True

    all_content = " ".join([c.content.lower() for c in chunks])
    overlap = sum(1 for w in q_words if w in all_content)
    # Require at least 1 key technical word match for relevance
    return overlap >= 1


def _grade_chunks_live(
    query: str, chunks: list[RetrievedChunk], model_name: str, timeout: float = 60.0
) -> tuple[bool, int, int]:
    """Live document grading assessing relevance against the question."""
    if not chunks:
        return False, 0, 0

    from src.llm.provider import call_llm

    context_preview = "\n".join([f"[{c.chunk_id}] {c.content}" for c in chunks[:4]])
    prompt = f"""You are an objective document grading agent evaluating context relevance.
Determine whether the provided context contains facts, concepts, or technical terms directly related to the question.
If the context mentions relevant system components, configuration values, or operations related to the question, mark it as relevant.
Only mark as NOT relevant if the context is entirely unrelated or unhelpful (e.g. out-of-domain or adversarial distractor).

Question: {query}

Context:
{context_preview}

Respond ONLY with a JSON object:
{{"is_relevant": true}} or {{"is_relevant": false}}
"""
    res = call_llm(model=model_name, prompt=prompt, temperature=0.0, timeout=timeout)
    content = res.content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(content[start : end + 1])
            is_rel = bool(data.get("is_relevant", True))
            return is_rel, res.prompt_tokens, res.completion_tokens
        except json.JSONDecodeError:
            pass

    # Heuristic fallback if LLM parse fails
    return _grade_chunks_offline(query, chunks), res.prompt_tokens, res.completion_tokens


class AgenticRAGGraph:
    """Compiled LangGraph Agentic Corrective RAG pipeline."""

    def __init__(self, retriever: RAGRetriever, config: GenerationConfig):
        self.retriever = retriever
        self.config = config
        self.max_hops = getattr(config, "max_agent_hops", 2)
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgenticRAGState)

        # Add Nodes
        workflow.add_node("decompose", self._decompose_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("refuse", self._refuse_node)

        # Add Edges & Conditional Routing
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self._route_decision,
            {
                "refuse": "refuse",
                "retrieve": "retrieve",
                "generate": "generate",
            },
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("refuse", END)

        return workflow.compile()

    def _decompose_node(self, state: AgenticRAGState) -> dict[str, Any]:
        query = state["query"]
        p_tokens = 0
        c_tokens = 0

        if self.config.mode == "live":
            sub_queries, p_tokens, c_tokens = _decompose_query_live(
                query=query,
                model_name=self.config.model_name,
                timeout=getattr(self.config, "timeout_sec", 60.0),
            )
        else:
            sub_queries = _decompose_query_offline(query)

        return {
            "sub_queries": sub_queries,
            "current_sub_idx": 0,
            "hop_count": 0,
            "max_hops": self.max_hops,
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "retrieved_ids": [c.chunk_id for c in state.get("retrieved_chunks", [])],
            "prompt_tokens": state.get("prompt_tokens", 0) + p_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + c_tokens,
            "route": "retrieve",
        }

    def _retrieve_node(self, state: AgenticRAGState) -> dict[str, Any]:
        sub_queries = state.get("sub_queries", [state["query"]])
        curr_idx = state.get("current_sub_idx", 0)

        # Select target sub-query for this hop
        if curr_idx < len(sub_queries):
            target_q = sub_queries[curr_idx]
        else:
            target_q = state["query"]

        ret_res = self.retriever.retrieve(target_q)
        existing_chunks = list(state.get("retrieved_chunks", []))
        existing_ids = set(state.get("retrieved_ids", []))

        # Merge and deduplicate
        for chunk in ret_res.chunks:
            if chunk.chunk_id not in existing_ids:
                existing_chunks.append(chunk)
                existing_ids.add(chunk.chunk_id)

        return {
            "retrieved_chunks": existing_chunks,
            "retrieved_ids": list(existing_ids),
            "hop_count": state.get("hop_count", 0) + 1,
            "current_sub_idx": curr_idx + 1,
            "route": "grade",
        }

    def _grade_node(self, state: AgenticRAGState) -> dict[str, Any]:
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        p_tokens = 0
        c_tokens = 0

        if self.config.mode == "live":
            is_relevant, p_tokens, c_tokens = _grade_chunks_live(
                query=query,
                chunks=chunks,
                model_name=self.config.model_name,
                timeout=getattr(self.config, "timeout_sec", 60.0),
            )
        else:
            is_relevant = _grade_chunks_offline(query, chunks)

        hop_count = state.get("hop_count", 1)
        sub_queries = state.get("sub_queries", [])
        curr_idx = state.get("current_sub_idx", 1)

        # Corrective Routing Logic
        if not is_relevant:
            # If context is not yet deemed relevant, and we have remaining hop capacity, attempt corrective retrieval
            if hop_count < self.max_hops:
                route = "retrieve"
            else:
                # Exhausted hops and still irrelevant context (adversarial negative / hallucination trigger)
                route = "refuse"
        elif curr_idx < len(sub_queries) and hop_count < self.max_hops:
            # We have another sub-query and capacity for another hop
            route = "retrieve"
        else:
            # We have sufficient relevant evidence
            route = "generate"

        return {
            "route": route,
            "prompt_tokens": state.get("prompt_tokens", 0) + p_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + c_tokens,
        }

    def _route_decision(self, state: AgenticRAGState) -> str:
        return state.get("route", "generate")

    def _generate_node(self, state: AgenticRAGState) -> dict[str, Any]:
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        context_str = "\n".join([f"[{c.chunk_id}] {c.content}" for c in chunks])
        prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context_str, query=query)

        p_tokens = len(prompt.split())
        c_tokens = 0
        answer_text = ""

        if self.config.mode == "live":
            from src.llm.provider import call_llm

            res = call_llm(
                model=self.config.model_name,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=getattr(self.config, "timeout_sec", 120.0),
            )
            raw = res.content or ""
            # Strip reasoning tags
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            cleaned = re.sub(r"\[REASONING\].*?\[/REASONING\]", "", cleaned, flags=re.DOTALL).strip()
            answer_text = cleaned if cleaned else raw
            p_tokens = res.prompt_tokens
            c_tokens = res.completion_tokens
        else:
            # Offline grounded generation citing retrieved chunks
            if not chunks:
                answer_text = "Based on the provided documentation, there is insufficient information to answer this question."
            else:
                top_chunks = chunks[:min(3, len(chunks))]
                citations_text = " ".join([f"[{c.chunk_id}]" for c in top_chunks])
                snippet = top_chunks[0].content[:160].replace("\n", " ").strip()
                answer_text = f"According to {citations_text}, {snippet}."
            c_tokens = len(answer_text.split())

        is_refusal = "insufficient information" in answer_text.lower()
        cited_ids = [c.chunk_id for c in chunks if f"[{c.chunk_id}]" in answer_text]

        return {
            "answer": answer_text,
            "cited_chunk_ids": cited_ids,
            "is_refusal": is_refusal,
            "prompt_tokens": state.get("prompt_tokens", 0) + p_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + c_tokens,
        }

    def _refuse_node(self, state: AgenticRAGState) -> dict[str, Any]:
        refusal_answer = "Based on the provided documentation, there is insufficient information to answer this question."
        return {
            "answer": refusal_answer,
            "cited_chunk_ids": [],
            "is_refusal": True,
            "prompt_tokens": state.get("prompt_tokens", 0) + 10,
            "completion_tokens": state.get("completion_tokens", 0) + 15,
        }

    def execute(self, query: str, initial_chunks: list[RetrievedChunk] | None = None) -> GenerationResult:
        """Executes the full agentic graph and returns structured GenerationResult."""
        start_time = time.perf_counter()

        initial_state: AgenticRAGState = {
            "query": query,
            "sub_queries": [query],
            "current_sub_idx": 0,
            "retrieved_chunks": list(initial_chunks or []),
            "retrieved_ids": [c.chunk_id for c in (initial_chunks or [])],
            "answer": "",
            "cited_chunk_ids": [],
            "hop_count": 0,
            "max_hops": self.max_hops,
            "is_refusal": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "route": "decompose",
        }

        final_state = self.app.invoke(initial_state)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return GenerationResult(
            query=query,
            answer=final_state.get("answer", ""),
            cited_chunk_ids=final_state.get("cited_chunk_ids", []),
            retrieved_chunks=final_state.get("retrieved_chunks", []),
            generation_latency_ms=latency_ms,
            prompt_tokens=final_state.get("prompt_tokens", 0),
            completion_tokens=final_state.get("completion_tokens", 0),
            is_refusal=final_state.get("is_refusal", False),
        )

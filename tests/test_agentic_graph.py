"""Unit tests for LangGraph Agentic Corrective RAG (CRAG) pipeline."""

from src.pipeline.agentic_graph import (
    AgenticRAGGraph,
    _decompose_query_offline,
    _grade_chunks_offline,
)
from src.pipeline.generator import GenerationResult, RAGGenerator
from src.pipeline.retriever import RetrievalResult, RetrievedChunk
from src.utils.config import GenerationConfig


class MockRetriever:
    """Mock retriever providing deterministic chunks based on query content."""

    def __init__(self):
        self.call_history: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.call_history.append(query)
        q_lower = query.lower()

        if "rate limit" in q_lower:
            chunks = [
                RetrievedChunk(
                    chunk_id="01_arch_c002",
                    content="API Gateway layer enforces distributed rate limiting using Redis token bucket algorithms.",
                    metadata={"doc": "01_arch", "char_start": 0, "char_end": 80},
                    score=0.92,
                    rank=1,
                )
            ]
        elif "authentication" in q_lower or "token" in q_lower:
            chunks = [
                RetrievedChunk(
                    chunk_id="03_auth_c006",
                    content="Authentication layer validates short-lived asymmetric RS256 JWT tokens with 15 min TTL.",
                    metadata={"doc": "03_auth", "char_start": 0, "char_end": 85},
                    score=0.89,
                    rank=1,
                )
            ]
        elif "adversarial" in q_lower or "quantum" in q_lower or "snowflake" in q_lower:
            # Irrelevant context to test refusal
            chunks = [
                RetrievedChunk(
                    chunk_id="99_misc_c001",
                    content="Unrelated logging configuration for Kubernetes daemonsets.",
                    metadata={"doc": "99_misc", "char_start": 0, "char_end": 60},
                    score=0.25,
                    rank=1,
                )
            ]
        else:
            chunks = [
                RetrievedChunk(
                    chunk_id="01_arch_c001",
                    content=f"General architecture documentation regarding {query}.",
                    metadata={"doc": "01_arch", "char_start": 0, "char_end": 50},
                    score=0.85,
                    rank=1,
                )
            ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_latency_ms=2.5,
            stage="mock",
        )


def test_decompose_query_offline_single_vs_multi_hop():
    """Validates that offline decomposition splits multi-hop queries but preserves single questions."""
    # Single direct factoid
    single_q = "What is the token expiration period for JWT access tokens?"
    assert _decompose_query_offline(single_q) == [single_q]

    # Multi-hop query with 'coordinate with'
    multi_q = "How does distributed rate limiting at the API Gateway coordinate with authentication token validation?"
    sub_qs = _decompose_query_offline(multi_q)
    assert len(sub_qs) == 2
    assert "rate limiting" in sub_qs[0].lower()
    assert "authentication token" in sub_qs[1].lower()

    # Multi-hop query with 'between X and Y'
    between_q = "What is the difference between synchronous web workers and the background worker fleet?"
    sub_between = _decompose_query_offline(between_q)
    assert len(sub_between) == 2
    assert "synchronous web workers" in sub_between[0].lower()
    assert "background worker fleet" in sub_between[1].lower()


def test_grade_chunks_offline():
    """Validates document grading logic across relevant and adversarial chunks."""
    relevant_chunks = [
        RetrievedChunk(
            chunk_id="01_arch_c002",
            content="API Gateway layer enforces distributed rate limiting using Redis token bucket algorithms.",
            score=0.92,
            rank=1,
        )
    ]
    irrelevant_chunks = [
        RetrievedChunk(
            chunk_id="99_misc_c001",
            content="Unrelated logging configuration for Kubernetes daemonsets.",
            score=0.25,
            rank=1,
        )
    ]

    q_relevant = "What algorithm is used for rate limiting at the Gateway?"
    q_irrelevant = "How does Snowflake configure micro-partition clustering keys?"

    assert _grade_chunks_offline(q_relevant, relevant_chunks) is True
    assert _grade_chunks_offline(q_irrelevant, irrelevant_chunks) is False
    assert _grade_chunks_offline(q_relevant, []) is False


def test_agentic_graph_multi_hop_execution():
    """Tests that AgenticRAGGraph performs multiple hops to retrieve context from distinct components."""
    retriever = MockRetriever()
    config = GenerationConfig(mode="offline", prompt_style="agentic", max_agent_hops=2)
    graph = AgenticRAGGraph(retriever=retriever, config=config)

    query = "How does distributed rate limiting at the API Gateway coordinate with authentication token validation?"
    result = graph.execute(query=query)

    assert isinstance(result, GenerationResult)
    assert result.is_refusal is False
    # Verified that 2 hops were performed by the mock retriever
    assert len(retriever.call_history) == 2
    assert "rate limiting" in retriever.call_history[0].lower()
    assert "authentication token" in retriever.call_history[1].lower()

    # Context contains chunks from both components
    retrieved_chunk_ids = [c.chunk_id for c in result.retrieved_chunks]
    assert "01_arch_c002" in retrieved_chunk_ids
    assert "03_auth_c006" in retrieved_chunk_ids
    assert len(result.answer) > 0


def test_agentic_graph_refusal_on_adversarial_queries():
    """Tests that the document grading node detects ungrounded context and routes to refusal."""
    retriever = MockRetriever()
    config = GenerationConfig(mode="offline", prompt_style="agentic", max_agent_hops=2)
    graph = AgenticRAGGraph(retriever=retriever, config=config)

    query = "How does quantum entanglement accelerate Snowflake clustering keys?"
    result = graph.execute(query=query)

    assert isinstance(result, GenerationResult)
    assert result.is_refusal is True
    assert "insufficient information" in result.answer.lower()
    assert len(result.cited_chunk_ids) == 0


def test_generator_interface_with_agentic_prompt_style():
    """Validates that RAGGenerator cleanly wraps the agentic graph."""
    config = GenerationConfig(mode="offline", prompt_style="agentic", max_agent_hops=2)
    generator = RAGGenerator(config=config)

    retriever = MockRetriever()
    query = "How does distributed rate limiting at the API Gateway coordinate with authentication token validation?"
    result = generator.generate(query=query, chunks=[], retriever=retriever)

    assert isinstance(result, GenerationResult)
    assert len(result.retrieved_chunks) >= 2
    assert result.generation_latency_ms > 0.0
    assert result.prompt_tokens > 0

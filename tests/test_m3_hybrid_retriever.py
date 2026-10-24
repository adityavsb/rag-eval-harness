"""Unit tests for Milestone M3: BM25 and Reciprocal Rank Fusion (RRF) hybrid retriever."""

from unittest.mock import MagicMock

from src.pipeline.retriever import RAGRetriever, RetrievedChunk
from src.utils.config import PipelineConfig, RetrieverConfig


def test_rrf_fuse_mathematics():
    """Validates Reciprocal Rank Fusion calculation: 1/(k + rank)."""
    cfg = RetrieverConfig(top_k=3, retrieval_mode="hybrid", rrf_k=60)
    mock_indexer = MagicMock()
    retriever = RAGRetriever(mock_indexer, cfg)

    # Dense: doc_a (rank 1), doc_b (rank 2)
    dense_chunks = [
        RetrievedChunk(chunk_id="doc_a", content="Doc A content", score=0.9, rank=1),
        RetrievedChunk(chunk_id="doc_b", content="Doc B content", score=0.8, rank=2),
    ]

    # BM25: doc_b (rank 1), doc_c (rank 2)
    bm25_chunks = [
        RetrievedChunk(chunk_id="doc_b", content="Doc B content", score=10.0, rank=1),
        RetrievedChunk(chunk_id="doc_c", content="Doc C content", score=8.0, rank=2),
    ]

    # Expected RRF scores:
    # doc_a: 1/(60+1) = 1/61 = ~0.016393
    # doc_b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = ~0.016129 + ~0.016393 = ~0.032522
    # doc_c: 1/(60+2) = 1/62 = ~0.016129
    fused = retriever._rrf_fuse(dense_chunks, bm25_chunks, top_k=3, k=60)

    assert len(fused) == 3
    # doc_b should be first because it is in both lists
    assert fused[0].chunk_id == "doc_b"
    assert fused[0].rank == 1
    expected_doc_b_score = (1.0 / 62) + (1.0 / 61)
    assert abs(fused[0].score - expected_doc_b_score) < 1e-5

    # doc_a is rank 1 in dense (1/61) > doc_c which is rank 2 in bm25 (1/62)
    assert fused[1].chunk_id == "doc_a"
    assert fused[1].rank == 2
    assert abs(fused[1].score - (1.0 / 61)) < 1e-5

    assert fused[2].chunk_id == "doc_c"
    assert fused[2].rank == 3
    assert abs(fused[2].score - (1.0 / 62)) < 1e-5


def test_hybrid_configs_validation():
    """Ensures configs/hybrid.yaml and configs/hybrid_rerank.yaml parse accurately."""
    hybrid_cfg = PipelineConfig.from_yaml("configs/hybrid.yaml")
    assert hybrid_cfg.retriever.retrieval_mode == "hybrid"
    assert hybrid_cfg.retriever.top_k == 4
    assert hybrid_cfg.retriever.use_reranker is False

    hybrid_rerank_cfg = PipelineConfig.from_yaml("configs/hybrid_rerank.yaml")
    assert hybrid_rerank_cfg.retriever.retrieval_mode == "hybrid"
    assert hybrid_rerank_cfg.retriever.top_k == 15
    assert hybrid_rerank_cfg.retriever.use_reranker is True
    assert hybrid_rerank_cfg.retriever.reranker_top_n == 4


def test_retriever_hybrid_execution():
    """Tests retrieve() routing when mode is hybrid."""
    cfg = RetrieverConfig(top_k=2, retrieval_mode="hybrid", rrf_k=60)
    mock_indexer = MagicMock()
    retriever = RAGRetriever(mock_indexer, cfg)

    # Mock _dense_retrieve and _bm25_retrieve
    retriever._dense_retrieve = MagicMock(
        return_value=[
            RetrievedChunk(chunk_id="c1", content="alpha", score=0.9, rank=1),
            RetrievedChunk(chunk_id="c2", content="beta", score=0.7, rank=2),
        ]
    )
    retriever._bm25_retrieve = MagicMock(
        return_value=[
            RetrievedChunk(chunk_id="c2", content="beta", score=12.0, rank=1),
            RetrievedChunk(chunk_id="c3", content="gamma", score=5.0, rank=2),
        ]
    )

    result = retriever.retrieve("query about beta")
    assert result.stage == "hybrid"
    assert len(result.chunks) == 2
    # c2 was ranked in both, so it should be rank 1 in fused
    assert result.chunks[0].chunk_id == "c2"
    assert result.retrieval_latency_ms >= 0.0

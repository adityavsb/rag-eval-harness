"""Single-stage and two-stage (Cross-Encoder) retrieval pipeline."""

import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from src.pipeline.indexing import VectorIndexer
from src.utils.config import RetrieverConfig


class RetrievedChunk(BaseModel):
    """Represents a retrieved document chunk with relevance scoring and rank."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float
    rank: int


class RetrievalResult(BaseModel):
    """Encapsulates retrieved chunks along with execution profiling metadata."""

    query: str
    chunks: List[RetrievedChunk]
    retrieval_latency_ms: float
    stage: str  # "dense" or "two_stage_rerank"


class RAGRetriever:
    """Configurable retriever supporting dense bi-encoder and cross-encoder reranking."""

    def __init__(self, indexer: VectorIndexer, config: RetrieverConfig):
        self.indexer = indexer
        self.config = config
        self._reranker: Optional[CrossEncoder] = None

    @property
    def reranker(self) -> Optional[CrossEncoder]:
        """Lazy-loaded CrossEncoder reranker model."""
        if self.config.use_reranker and self._reranker is None:
            model_name = self.config.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._reranker = CrossEncoder(model_name)
        return self._reranker

    def _dense_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Executes dense vector similarity search in ChromaDB."""
        collection = self.indexer.get_or_create_collection()
        total_docs = collection.count()
        if total_docs == 0:
            return []

        n_results = min(top_k, total_docs)
        query_embedding = self.indexer.embedder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for rank, (chunk_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            # Chroma returns cosine distance (1 - cosine_similarity); convert to similarity score
            sim_score = max(0.0, 1.0 - float(dist))
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc,
                    metadata=meta,
                    score=sim_score,
                    rank=rank,
                )
            )

        return chunks

    def _rerank(self, query: str, candidates: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        """Scores candidate chunks using a Cross-Encoder and returns the top-N reordered chunks."""
        if not candidates or not self.reranker:
            return candidates[:top_n]

        # Prepare sentence pairs for Cross-Encoder scoring
        pairs = [[query, chunk.content] for chunk in candidates]
        scores = self.reranker.predict(pairs)

        # Pair candidates with cross-encoder scores
        scored_candidates: List[Tuple[RetrievedChunk, float]] = list(zip(candidates, scores))
        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks: List[RetrievedChunk] = []
        for rank, (chunk, ce_score) in enumerate(scored_candidates[:top_n], start=1):
            reranked_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(ce_score),
                    rank=rank,
                )
            )

        return reranked_chunks

    def retrieve(self, query: str) -> RetrievalResult:
        """Performs retrieval according to configuration (single-stage dense or two-stage reranked)."""
        start_time = time.perf_counter()

        if self.config.use_reranker:
            # Stage 1: Candidate retrieval with top_k (e.g. 15)
            candidates = self._dense_retrieve(query, top_k=self.config.top_k)
            # Stage 2: Cross-Encoder rerank to reranker_top_n (e.g. 4)
            final_chunks = self._rerank(query, candidates, top_n=self.config.reranker_top_n)
            stage = "two_stage_rerank"
        else:
            # Single-stage dense retrieval (top_k)
            final_chunks = self._dense_retrieve(query, top_k=self.config.top_k)
            stage = "dense"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            retrieval_latency_ms=latency_ms,
            stage=stage,
        )

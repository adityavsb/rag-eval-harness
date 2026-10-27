"""Single-stage and two-stage (Cross-Encoder) retrieval pipeline with Hybrid RRF support."""

import re
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.pipeline.indexing import VectorIndexer
from src.utils.config import RetrieverConfig


class RetrievedChunk(BaseModel):
    """Represents a retrieved document chunk with relevance scoring and rank."""

    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    rank: int


class RetrievalResult(BaseModel):
    """Encapsulates retrieved chunks along with execution profiling metadata."""

    query: str
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    stage: str  # "dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"


class RAGRetriever:
    """Configurable retriever supporting dense bi-encoder, BM25, hybrid RRF, and cross-encoder reranking."""

    def __init__(self, indexer: VectorIndexer, config: RetrieverConfig):
        self.indexer = indexer
        self.config = config
        self._reranker: CrossEncoder | None = None
        self._bm25_index: BM25Okapi | None = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_documents: list[str] = []
        self._bm25_metadatas: list[dict[str, Any]] = []

    @property
    def reranker(self) -> CrossEncoder | None:
        """Lazy-loaded CrossEncoder reranker model."""
        if self.config.use_reranker and self._reranker is None:
            model_name = self.config.reranker_model or "cross-encoder/ms-marco-MiniLM-L6-v2"
            self._reranker = CrossEncoder(model_name)
        return self._reranker

    def _init_bm25_if_needed(self) -> None:
        """Initializes in-memory BM25 index over vector store chunks if not already built."""
        if self._bm25_index is not None and len(self._bm25_chunk_ids) > 0:
            return

        collection = self.indexer.get_or_create_collection()
        if collection.count() == 0:
            return

        data = collection.get(include=["documents", "metadatas"])
        self._bm25_chunk_ids = data["ids"]
        self._bm25_documents = data["documents"] or [""] * len(data["ids"])
        self._bm25_metadatas = data["metadatas"] or [{}] * len(data["ids"])

        tokenized_corpus = [re.findall(r"\b\w+\b", doc.lower()) for doc in self._bm25_documents]
        self._bm25_index = BM25Okapi(tokenized_corpus)

    def _bm25_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Executes BM25 lexical keyword retrieval."""
        self._init_bm25_if_needed()
        if self._bm25_index is None or not self._bm25_chunk_ids:
            return []

        tokens = re.findall(r"\b\w+\b", query.lower())
        if not tokens:
            return []

        scores = self._bm25_index.get_scores(tokens)
        ranked_indices = np.argsort(scores)[::-1]

        chunks: list[RetrievedChunk] = []
        limit = min(top_k, len(ranked_indices))
        for rank, idx in enumerate(ranked_indices[:limit], start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=self._bm25_chunk_ids[idx],
                    content=self._bm25_documents[idx],
                    metadata=self._bm25_metadatas[idx] if self._bm25_metadatas else {},
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return chunks

    def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
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

        chunks: list[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, distances), start=1
        ):
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

    def _rrf_fuse(
        self,
        dense_chunks: list[RetrievedChunk],
        bm25_chunks: list[RetrievedChunk],
        top_k: int,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Combines dense and lexical ranked results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_chunks, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_chunks, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort descending by fused score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused_chunks: list[RetrievedChunk] = []
        limit = min(top_k, len(sorted_items))
        for rank, (chunk_id, score) in enumerate(sorted_items[:limit], start=1):
            base_chunk = chunk_map[chunk_id]
            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=base_chunk.content,
                    metadata=base_chunk.metadata,
                    score=float(score),
                    rank=rank,
                )
            )
        return fused_chunks

    def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Scores candidate chunks using a Cross-Encoder and returns the top-N reordered chunks."""
        if not candidates or not self.reranker:
            return candidates[:top_n]

        # Prepare sentence pairs for Cross-Encoder scoring
        pairs = [[query, chunk.content] for chunk in candidates]
        scores = self.reranker.predict(pairs)

        # Pair candidates with cross-encoder scores
        scored_candidates: list[tuple[RetrievedChunk, float]] = list(zip(candidates, scores))
        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks: list[RetrievedChunk] = []
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
        """Performs retrieval according to configuration (single-stage dense, bm25, or hybrid RRF)."""
        start_time = time.perf_counter()
        mode = self.config.retrieval_mode

        # Decide candidate pool size for first-stage retrieval
        candidate_k = self.config.top_k
        if self.config.use_reranker:
            candidate_k = max(self.config.top_k, self.config.reranker_top_n * 3, 15)

        if mode == "dense":
            candidates = self._dense_retrieve(query, top_k=candidate_k)
            stage = "dense"
        elif mode == "bm25":
            candidates = self._bm25_retrieve(query, top_k=candidate_k)
            stage = "bm25"
        elif mode == "hybrid":
            dense_cand = self._dense_retrieve(query, top_k=candidate_k)
            bm25_cand = self._bm25_retrieve(query, top_k=candidate_k)
            candidates = self._rrf_fuse(
                dense_cand, bm25_cand, top_k=candidate_k, k=self.config.rrf_k
            )
            stage = "hybrid"
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        if self.config.use_reranker:
            final_chunks = self._rerank(query, candidates, top_n=self.config.reranker_top_n)
            stage = f"{stage}_rerank"
        else:
            final_chunks = candidates[: self.config.top_k]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            retrieval_latency_ms=latency_ms,
            stage=stage,
        )

"""Vector store indexing using local Hugging Face embeddings and ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.pipeline.ingestion import DocumentChunk
from src.utils.config import EmbeddingConfig, VectorStoreConfig


class VectorIndexer:
    """Manages local embedding generation and ChromaDB persistence."""

    def __init__(self, embed_config: EmbeddingConfig, store_config: VectorStoreConfig):
        self.embed_config = embed_config
        self.store_config = store_config
        self._embedder: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-loaded SentenceTransformer model."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self.embed_config.model_name,
                device=self.embed_config.device,
            )
        return self._embedder

    @property
    def client(self) -> chromadb.PersistentClient:
        """Persistent ChromaDB client."""
        if self._client is None:
            persist_path = Path(self.store_config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path))
        return self._client

    @staticmethod
    def corpus_fingerprint(data_dir: str | Path, pattern: str, chunking: Any) -> str:
        """Computes a deterministic SHA-256 fingerprint of the corpus and chunking parameters."""
        import hashlib
        from pathlib import Path

        h = hashlib.sha256()
        base_path = Path(data_dir)
        for p in sorted(base_path.rglob(pattern)):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        h.update(f"{chunking.strategy}|{chunking.chunk_size}|{chunking.chunk_overlap}".encode())
        return h.hexdigest()[:16]

    def stored_fingerprint(self) -> str | None:
        """Retrieves the corpus fingerprint stored in the collection's metadata."""
        try:
            coll = self.get_or_create_collection()
            return (coll.metadata or {}).get("fingerprint")
        except Exception:
            return None

    def reset_collection(self) -> None:
        """Deletes the existing Chroma collection to force a clean re-index."""
        try:
            self.client.delete_collection(self.store_config.collection_name)
        except Exception:
            pass

    def get_or_create_collection(self, fingerprint: str | None = None) -> Collection:
        """Retrieves or creates the vector collection with cosine distance metric."""
        metadata = {"hnsw:space": "cosine"}
        if fingerprint:
            metadata["fingerprint"] = fingerprint
        return self.client.get_or_create_collection(
            name=self.store_config.collection_name,
            metadata=metadata,
        )

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 64,
        fingerprint: str | None = None,
    ) -> int:
        """Encodes and indexes document chunks into the ChromaDB collection."""
        if not chunks:
            return 0

        collection = self.get_or_create_collection(fingerprint=fingerprint)
        if fingerprint and (
            not collection.metadata or collection.metadata.get("fingerprint") != fingerprint
        ):
            try:
                new_meta = dict(collection.metadata or {})
                new_meta["fingerprint"] = fingerprint
                collection.modify(metadata=new_meta)
            except Exception:
                pass

        # Extract IDs, texts, and metadata
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Batch encode with normalization for cosine similarity
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            embeddings = self.embedder.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
            )

        return collection.count()

    def count(self) -> int:
        """Returns total indexed chunk count in the collection."""
        return self.get_or_create_collection().count()

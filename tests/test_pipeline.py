"""Unit tests for the ingestion, indexing, and retrieval pipeline."""

import tempfile

from src.pipeline.generator import RAGGenerator
from src.pipeline.indexing import VectorIndexer
from src.pipeline.ingestion import (
    DocumentChunk,
    _split_fixed,
    _split_recursive,
    chunk_documents,
)
from src.pipeline.retriever import RAGRetriever
from src.utils.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    RetrieverConfig,
    VectorStoreConfig,
)


def test_fixed_chunking():
    text = "ABCDEFGHIJ"
    # Chunk size 4, overlap 1 -> step 3: "ABCD", "DEFG", "GHIJ", "J"
    chunks = _split_fixed(text, chunk_size=4, chunk_overlap=1)
    assert len(chunks) >= 3
    assert chunks[0] == "ABCD"
    assert chunks[1] == "DEFG"


def test_recursive_chunking_preserves_paragraphs():
    text = (
        "Paragraph one with some details.\n\nParagraph two with more details.\n\nParagraph three."
    )
    chunks = _split_recursive(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    assert "Paragraph one" in chunks[0]


def test_recursive_chunks_respect_size_and_word_boundaries():
    from src.pipeline.ingestion import chunk_documents, load_raw_documents
    from src.utils.config import ChunkingConfig

    docs = load_raw_documents("data/raw", "*.md")
    chunks = chunk_documents(
        docs, ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=50)
    )
    assert max(len(c.content) for c in chunks) <= 400
    assert not any(c.content[0].islower() for c in chunks), "chunk starts mid-word"


def test_chunk_documents_id_generation():
    raw_docs = [
        {
            "source": "guide.md",
            "filename": "guide.md",
            "stem": "guide",
            "content": "Short text for testing chunk generation.",
        }
    ]
    config = ChunkingConfig(strategy="fixed", chunk_size=50, chunk_overlap=0)
    chunks = chunk_documents(raw_docs, config)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "guide_c001"
    assert chunks[0].metadata["source"] == "guide.md"
    assert chunks[0].metadata["chunk_index"] == 1


def test_vector_indexing_and_retrieval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        embed_config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        store_config = VectorStoreConfig(
            persist_directory=f"{tmp_dir}/chroma",
            collection_name="test_collection",
        )

        indexer = VectorIndexer(embed_config, store_config)

        sample_chunks = [
            DocumentChunk(
                chunk_id="spark_c001",
                content="Apache Spark uses resilient distributed datasets (RDDs) for fault tolerance.",
                metadata={"topic": "spark"},
            ),
            DocumentChunk(
                chunk_id="k8s_c001",
                content="Kubernetes manages containerized workloads across node clusters.",
                metadata={"topic": "k8s"},
            ),
        ]

        count = indexer.index_chunks(sample_chunks)
        assert count == 2

        retriever_config = RetrieverConfig(top_k=2, use_reranker=False)
        retriever = RAGRetriever(indexer, retriever_config)

        result = retriever.retrieve("How does Spark provide fault tolerance?")
        assert len(result.chunks) == 2
        # Spark chunk should be the top match
        assert result.chunks[0].chunk_id == "spark_c001"
        assert result.chunks[0].score > result.chunks[1].score


def test_generator_offline_fallback():
    gen_config = GenerationConfig(prompt_style="grounded")
    generator = RAGGenerator(gen_config)

    chunks = [
        DocumentChunk(
            chunk_id="doc_c001",
            content="FastAPI has built-in support for OpenAPI and Swagger UI documentation.",
        )
    ]
    # Mock retrieved chunk
    from src.pipeline.retriever import RetrievedChunk

    retrieved = [
        RetrievedChunk(
            chunk_id=chunks[0].chunk_id,
            content=chunks[0].content,
            score=0.95,
            rank=1,
        )
    ]

    res = generator.generate("What documentation does FastAPI support?", retrieved)
    assert res.query == "What documentation does FastAPI support?"
    assert "[doc_c001]" in res.answer
    assert not res.is_refusal

"""Configuration loader and schema validation using Pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CorpusConfig(BaseModel):
    data_dir: str = "data/raw"
    file_pattern: str = "*.md"


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    persist_directory: str = "chroma_db/default"
    collection_name: str = "default_collection"


class RetrieverConfig(BaseModel):
    top_k: int = Field(default=4, gt=0)
    use_reranker: bool = False
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = Field(default=4, gt=0)


class GenerationConfig(BaseModel):
    mode: Literal["offline", "live"] = "offline"
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)
    prompt_style: Literal["naive", "grounded"] = "grounded"


class EvaluationConfig(BaseModel):
    mode: Literal["offline", "live"] = "offline"
    judge_provider: str = "openai"
    golden_qa_path: str = "data/golden_qa.json"
    judge_model: str = "gpt-4o-mini"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "hit_rate_at_k",
            "mrr_at_k",
            "context_recall",
            "faithfulness",
            "answer_correctness",
        ]
    )


class TrackingConfig(BaseModel):
    experiment_name: str = "rag-evaluation-benchmark"
    tracking_uri: str = "sqlite:///mlflow.db"


class PipelineConfig(BaseModel):
    run_name: str
    description: str = ""
    mode: Literal["offline", "live"] | None = None
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

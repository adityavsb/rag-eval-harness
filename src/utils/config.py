"""Configuration loader and schema validation using Pydantic."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


def load_dotenv(env_path: str | Path = ".env") -> dict[str, str]:
    """Lightweight .env parser that loads variables into os.environ without external dependencies."""
    p = Path(env_path)
    if not p.is_file():
        # Check parent and workspace root relative paths
        alt_paths = [
            Path("rag-eval-harness/.env"),
            Path(__file__).resolve().parent.parent.parent / ".env",
        ]
        for alt in alt_paths:
            if alt.is_file():
                p = alt
                break
        else:
            return {}

    loaded = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and v and k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
    except OSError:
        pass
    return loaded


# Automatically load environment variables on import
load_dotenv()


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
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "dense"
    rrf_k: int = Field(default=60, gt=0)
    use_reranker: bool = False
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_top_n: int = Field(default=4, gt=0)


class GenerationConfig(BaseModel):
    mode: Literal["offline", "live"] = "offline"
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    reasoning_model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)
    timeout_sec: float = Field(default=120.0, gt=0.0)
    pacing_delay_sec: float = Field(default=1.5, ge=0.0)
    prompt_style: Literal["naive", "grounded", "dspy"] = "grounded"
    compiled_dspy_program_path: str = "src/optimize/compiled_dspy_rag.json"


class EvaluationConfig(BaseModel):
    mode: Literal["offline", "live"] = "offline"
    judge_provider: str = "openai"
    golden_qa_path: str = "data/golden_qa.json"
    judge_model: str = "gpt-4o-mini"
    stage: Literal["check-one", "check-remaining", "full"] = "full"
    timeout_sec: float = Field(default=120.0, gt=0.0)
    checkpoint_dir: str = "reports/live/checkpoints"
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

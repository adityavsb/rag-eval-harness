"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from src.utils.config import PipelineConfig


def test_load_baseline_config():
    config_path = Path("configs/baseline.yaml")
    config = PipelineConfig.from_yaml(config_path)

    assert config.run_name == "baseline-naive-rag"
    assert config.chunking.strategy == "fixed"
    assert config.chunking.chunk_size == 1000
    assert config.retriever.top_k == 4
    assert config.retriever.use_reranker is False


def test_load_rerank_optimized_config():
    config_path = Path("configs/rerank_optimized.yaml")
    config = PipelineConfig.from_yaml(config_path)

    assert config.run_name == "rerank-two-stage-rag"
    assert config.chunking.strategy == "recursive"
    assert config.retriever.use_reranker is True
    assert config.retriever.reranker_top_n == 4
    assert config.retriever.top_k == 15


def test_missing_config_raises_error():
    with pytest.raises(FileNotFoundError):
        PipelineConfig.from_yaml("configs/non_existent.yaml")

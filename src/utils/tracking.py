"""MLflow experiment tracking and metric logging utilities."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.utils.config import PipelineConfig


class MLflowTracker:
    """Wrapper around MLflow for standardized RAG benchmarking runs."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.experiment_name = config.tracking.experiment_name
        self.tracking_uri = config.tracking.tracking_uri
        self._mlflow = None
        self._active_run = None

    def _get_mlflow(self):
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
            except ImportError:
                self._mlflow = False
        return self._mlflow

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Generator[Any, None, None]:
        """Context manager for an MLflow run with parameter autologging."""
        mlf = self._get_mlflow()
        name = run_name or self.config.run_name

        if mlf:
            with mlf.start_run(run_name=name) as run:
                self._active_run = run
                # Log flattened config parameters
                self._log_config_parameters()
                try:
                    yield self
                finally:
                    self._active_run = None
        else:
            # Fallback when MLflow is not installed
            print(f"[MLflowTracker] Local run started: {name}")
            try:
                yield self
            finally:
                print(f"[MLflowTracker] Local run ended: {name}")

    def _log_config_parameters(self) -> None:
        """Logs structured configuration parameters to MLflow."""
        if not self._mlflow or not self._active_run:
            return

        params = {
            "chunk_strategy": self.config.chunking.strategy,
            "chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap,
            "embedding_model": self.config.embedding.model_name,
            "retriever_top_k": self.config.retriever.top_k,
            "use_reranker": self.config.retriever.use_reranker,
            "reranker_model": str(self.config.retriever.reranker_model),
            "reranker_top_n": self.config.retriever.reranker_top_n,
            "llm_model": self.config.generation.model_name,
            "prompt_style": self.config.generation.prompt_style,
        }
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs a dictionary of scalar evaluation metrics."""
        mlf = self._get_mlflow()
        if mlf and self._active_run:
            mlf.log_metrics(metrics, step=step)
        else:
            print(f"[Metrics] {json.dumps(metrics, indent=2)}")

    def log_artifact_dict(self, data: dict[str, Any], artifact_name: str) -> None:
        """Saves a JSON dictionary as a run artifact."""
        mlf = self._get_mlflow()
        output_dir = Path(".artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        local_path = output_dir / artifact_name

        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if mlf and self._active_run:
            mlf.log_artifact(str(local_path))

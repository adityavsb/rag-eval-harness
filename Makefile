.PHONY: help install lint test eval mlflow clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
MLFLOW = $(VENV)/bin/mlflow

help:
	@echo "Available commands:"
	@echo "  make install     Install production and development dependencies"
	@echo "  make lint        Run ruff linter and style checks"
	@echo "  make test        Execute test suite"
	@echo "  make eval        Run baseline evaluation benchmark"
	@echo "  make mlflow      Launch local MLflow UI dashboard"
	@echo "  make clean       Remove caches, local DBs, and temp files"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check src tests experiments
	$(VENV)/bin/ruff format --check src tests experiments

test:
	$(PYTEST) -v tests/

eval:
	$(PYTHON) experiments/run_eval.py --config configs/baseline.yaml

eval-all:
	$(PYTHON) experiments/run_eval.py --config configs/baseline.yaml
	$(PYTHON) experiments/run_eval.py --config configs/chunk_optimized.yaml
	$(PYTHON) experiments/run_eval.py --config configs/rerank_optimized.yaml

mlflow:
	$(MLFLOW) ui --backend-store-uri sqlite:///mlflow.db --port 5001

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov *.egg-info chroma_db

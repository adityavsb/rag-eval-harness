.PHONY: help install lint test eval eval-all eval-live-openrouter secret-scan mlflow clean

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
	@echo "  make eval-live-openrouter  Reminder + OpenRouter full eval (requires OPENROUTER_API_KEY)"
	@echo "  make secret-scan   Scan git history for secrets (requires gitleaks on PATH)"
	@echo "  make mlflow      Launch local MLflow UI dashboard"
	@echo "  make clean       Remove caches, local DBs, and temp files"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.lock
	$(PIP) install -e . --no-deps

lint:
	$(VENV)/bin/ruff check src tests experiments scripts
	$(VENV)/bin/ruff format --check src tests experiments scripts

test:
	$(PYTEST) -v tests/

eval:
	$(PYTHON) experiments/run_eval.py --config configs/baseline.yaml --mode offline

eval-all:
	$(PYTHON) experiments/run_eval.py --config configs/baseline.yaml --mode offline
	$(PYTHON) experiments/run_eval.py --config configs/chunk_optimized.yaml --mode offline
	$(PYTHON) experiments/run_eval.py --config configs/rerank_optimized.yaml --mode offline
	$(PYTHON) experiments/run_eval.py --config configs/hybrid.yaml --mode offline
	$(PYTHON) experiments/run_eval.py --config configs/hybrid_rerank.yaml --mode offline

eval-live-openrouter:
	@test -n "$$OPENROUTER_API_KEY" || (echo "Set OPENROUTER_API_KEY in .env or environment before live eval."; exit 1)
	@echo "Live OpenRouter eval: full 80 queries, hybrid RRF — see README Quickstart for per-model commands."
	$(PYTHON) experiments/run_eval.py --config configs/openrouter_cohort.yaml --mode live --stage full

secret-scan:
	@command -v gitleaks >/dev/null 2>&1 || (echo "Install gitleaks: https://github.com/gitleaks/gitleaks#installing"; exit 1)
	gitleaks detect --source . --config .gitleaks.toml --verbose --redact

mlflow:
	$(MLFLOW) ui --backend-store-uri sqlite:///mlflow.db --port 5001

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov *.egg-info chroma_db

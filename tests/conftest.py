"""Pytest configuration ensuring working directory is repo root and offline mode."""

import os
import sys
from pathlib import Path

# Use offline HuggingFace mode locally when the embedding model cache exists; allow downloads on CI
_minilm_cache = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
)
if not os.environ.get("CI") and _minilm_cache.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

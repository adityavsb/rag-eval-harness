"""Pytest configuration ensuring working directory is repo root and offline mode."""

import os
import sys
from pathlib import Path

# Enforce offline HuggingFace mode to prevent network timeouts during sandbox test execution
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


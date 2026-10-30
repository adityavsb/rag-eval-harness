"""Single source for package version read from pyproject.toml (no importlib install required)."""

from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def read_project_version() -> str:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version ="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("version not found in pyproject.toml")


def user_agent() -> str:
    return f"rag-eval-harness/{read_project_version()}"

"""Release metadata consistency (package vs citation)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "0.2.1"


def _read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    pytest.fail("version not found in pyproject.toml")


def _read_cff_version() -> str:
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    pytest.fail("version not found in CITATION.cff")


def test_package_and_citation_versions_match():
    pkg = _read_pyproject_version()
    cff = _read_cff_version()
    assert pkg == cff == EXPECTED_VERSION

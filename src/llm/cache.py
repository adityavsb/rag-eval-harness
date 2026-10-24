"""On-disk response cache for LLM inferences keyed by (model, prompt hash)."""

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskResponseCache:
    """Persists model responses to local disk to eliminate redundant API calls."""

    def __init__(self, cache_dir: str = ".cache/llm_responses"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, model: str, prompt: str) -> str:
        """Computes deterministic SHA-256 hash for model and prompt string."""
        raw_key = f"{model.strip().lower()}:{prompt.strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> dict[str, Any] | None:
        """Retrieves cached payload if available."""
        key = self._compute_key(model, prompt)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def set(self, model: str, prompt: str, response: dict[str, Any]) -> None:
        """Saves response payload to disk cache."""
        key = self._compute_key(model, prompt)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)
        except OSError:
            pass

    def clear(self) -> int:
        """Clears all cached entries and returns number of files removed."""
        count = 0
        for item in self.cache_dir.glob("*.json"):
            try:
                item.unlink()
                count += 1
            except OSError:
                pass
        return count

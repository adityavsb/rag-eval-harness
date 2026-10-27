"""Document ingestion and chunking pipeline."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.utils.config import ChunkingConfig


class DocumentChunk(BaseModel):
    """Represents a discrete chunk of text extracted from a source document."""

    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_raw_documents(data_dir: str | Path, pattern: str = "*.md") -> list[dict[str, Any]]:
    """Loads raw documents from a directory matching the specified file glob pattern."""
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    documents = []
    for file_path in sorted(base_path.rglob(pattern)):
        if file_path.is_file():
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            documents.append(
                {
                    "source": str(file_path.relative_to(base_path)),
                    "filename": file_path.name,
                    "stem": file_path.stem,
                    "content": content,
                }
            )
    return documents


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits text into fixed character windows with overlap."""
    if not text.strip():
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """LangChain-style recursive splitter: split on the highest-priority separator present,
    recurse into any piece still longer than chunk_size, then pack pieces with overlap
    measured in whole pieces (never mid-word)."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    sep = next((s for s in separators if s and s in text), "")
    rest = separators[separators.index(sep) + 1 :] if sep else []
    pieces = text.split(sep) if sep else list(text)
    atoms: list[str] = []
    for p in pieces:
        if len(p) > chunk_size and rest:
            atoms.extend(_split_recursive(p, chunk_size, chunk_overlap, rest))
        elif p.strip():
            atoms.append(p.strip())
    chunks: list[str] = []
    cur: list[str] = []

    def joined(parts: list[str]) -> str:
        return sep.join(parts) if sep else "".join(parts)

    for a in atoms:
        if cur and len(joined(cur + [a])) > chunk_size:
            chunks.append(joined(cur))
            keep: list[str] = []
            size = 0
            for prev in reversed(cur):  # overlap = trailing whole atoms up to chunk_overlap chars
                new_size = size + len(prev) + (len(sep) if keep else 0)
                if new_size > chunk_overlap or len(joined([prev] + keep + [a])) > chunk_size:
                    break
                keep.insert(0, prev)
                size = new_size
            cur = keep + [a]
        else:
            cur.append(a)
    if cur:
        chunks.append(joined(cur))
    return [c for c in chunks if c]


def chunk_documents(
    raw_documents: list[dict[str, Any]], config: ChunkingConfig
) -> list[DocumentChunk]:
    """Applies the configured chunking strategy to raw documents and generates chunk IDs."""
    all_chunks: list[DocumentChunk] = []

    for doc in raw_documents:
        content = doc["content"]
        stem = doc["stem"]
        source = doc["source"]

        if config.strategy == "fixed":
            text_chunks = _split_fixed(content, config.chunk_size, config.chunk_overlap)
        else:
            text_chunks = _split_recursive(content, config.chunk_size, config.chunk_overlap)

        cursor = 0
        for idx, text in enumerate(text_chunks, start=1):
            start = content.find(text[:80], max(0, cursor - config.chunk_overlap - 5))
            if start < 0:
                start = content.find(text[:80])
            end = start + len(text) if start >= 0 else -1
            cursor = max(cursor, end)
            chunk_id = f"{stem}_c{idx:03d}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=text,
                    metadata={
                        "source": source,
                        "doc": stem,
                        "chunk_index": idx,
                        "char_start": start,
                        "char_end": end,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    },
                )
            )

    return all_chunks

"""Document ingestion and chunking pipeline."""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.utils.config import ChunkingConfig


class DocumentChunk(BaseModel):
    """Represents a discrete chunk of text extracted from a source document."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def load_raw_documents(data_dir: str | Path, pattern: str = "*.md") -> List[Dict[str, Any]]:
    """Loads raw documents from a directory matching the specified file glob pattern."""
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    documents = []
    for file_path in sorted(base_path.rglob(pattern)):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
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


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Splits text hierarchically using natural boundary separators."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the highest-priority separator present in text
    chosen_sep = separators[-1]
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep != "" else list(text)

    # Recombine splits up to chunk_size
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece) + (len(chosen_sep) if current_chunk else 0)
        if current_length + piece_len > chunk_size and current_chunk:
            combined = chosen_sep.join(current_chunk).strip()
            if combined:
                chunks.append(combined)
            # Retain overlap from end of current chunk
            overlap_text = combined[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = [overlap_text, piece] if overlap_text else [piece]
            current_length = len(overlap_text) + len(piece)
        else:
            current_chunk.append(piece)
            current_length += piece_len

    if current_chunk:
        combined = chosen_sep.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def chunk_documents(
    raw_documents: List[Dict[str, Any]], config: ChunkingConfig
) -> List[DocumentChunk]:
    """Applies the configured chunking strategy to raw documents and generates chunk IDs."""
    all_chunks: List[DocumentChunk] = []

    for doc in raw_documents:
        content = doc["content"]
        stem = doc["stem"]
        source = doc["source"]

        if config.strategy == "fixed":
            text_chunks = _split_fixed(content, config.chunk_size, config.chunk_overlap)
        else:
            text_chunks = _split_recursive(content, config.chunk_size, config.chunk_overlap)

        for idx, text in enumerate(text_chunks, start=1):
            chunk_id = f"{stem}_c{idx:03d}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=text,
                    metadata={
                        "source": source,
                        "chunk_index": idx,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    },
                )
            )

    return all_chunks

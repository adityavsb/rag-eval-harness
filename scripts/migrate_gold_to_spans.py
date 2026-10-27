"""Convert ground_truth_context_ids (400/50 recursive id space) to chunking-independent spans."""

import json
import sys
from pathlib import Path

# Ensure repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.ingestion import chunk_documents, load_raw_documents
from src.utils.config import ChunkingConfig

docs = load_raw_documents("data/raw", "*.md")
ref = {
    c.chunk_id: c.metadata
    for c in chunk_documents(
        docs, ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=50)
    )
}

gold_path = Path("data/golden_qa.json")
with open(gold_path, encoding="utf-8") as f:
    gold = json.load(f)

for item in gold:
    spans = []
    for gid in item.get("ground_truth_context_ids", []):
        if gid in ref:
            m = ref[gid]
            spans.append(
                {"doc": m["doc"], "char_start": m["char_start"], "char_end": m["char_end"]}
            )
        else:
            # Bare document stem (e.g. qa_026)
            spans.append({"doc": gid, "char_start": 0, "char_end": -1})
    item["ground_truth_spans"] = spans

with open(gold_path, "w", encoding="utf-8") as f:
    json.dump(gold, f, indent=2, ensure_ascii=False)

migrated_count = sum(bool(i["ground_truth_spans"]) for i in gold)
print(f"migrated {migrated_count} items with non-empty ground_truth_spans out of {len(gold)}")

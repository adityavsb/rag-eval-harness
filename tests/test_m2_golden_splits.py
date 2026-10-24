"""Unit tests for Golden QA dataset v2 and train/dev/test split integrity (M2)."""

import json
from pathlib import Path


def test_golden_qa_size_and_verification():
    dataset_path = Path("data/golden_qa.json")
    assert dataset_path.exists(), "data/golden_qa.json does not exist"

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 80, f"Expected at least 80 golden QA items, got {len(data)}"

    # All items must be marked verified
    unverified = [item["id"] for item in data if not item.get("verified", False)]
    assert len(unverified) == 0, f"Found unverified items: {unverified}"


def test_split_integrity_and_ratios():
    dataset_path = Path("data/golden_qa.json")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    splits = {}
    ids = set()
    queries_by_split = {"train": set(), "dev": set(), "test": set()}

    for item in data:
        item_id = item["id"]
        assert item_id not in ids, f"Duplicate ID found: {item_id}"
        ids.add(item_id)

        split = item.get("split")
        assert split in ("train", "dev", "test"), f"Invalid split '{split}' for {item_id}"
        splits[split] = splits.get(split, 0) + 1

        query = item.get("query", "").strip().lower()
        assert len(query) > 5, f"Query too short for {item_id}"
        assert len(item.get("ground_truth_answer", "").strip()) > 0, f"Missing answer for {item_id}"

        queries_by_split[split].add(query)

    assert splits["train"] == 32, f"Expected 32 train items, got {splits['train']}"
    assert splits["dev"] == 16, f"Expected 16 dev items, got {splits['dev']}"
    assert splits["test"] == 32, f"Expected 32 test items, got {splits['test']}"

    # Disjoint check (zero leakage between splits)
    train_dev_overlap = queries_by_split["train"].intersection(queries_by_split["dev"])
    assert len(train_dev_overlap) == 0, f"Leakage between train and dev: {train_dev_overlap}"

    train_test_overlap = queries_by_split["train"].intersection(queries_by_split["test"])
    assert len(train_test_overlap) == 0, f"Leakage between train and test: {train_test_overlap}"

    dev_test_overlap = queries_by_split["dev"].intersection(queries_by_split["test"])
    assert len(dev_test_overlap) == 0, f"Leakage between dev and test: {dev_test_overlap}"


def test_corpus_raw_documents_exist():
    raw_dir = Path("data/raw")
    assert raw_dir.is_dir()
    docs = list(raw_dir.glob("*.md"))
    assert len(docs) >= 10, f"Expected at least 10 markdown documents in data/raw, got {len(docs)}"

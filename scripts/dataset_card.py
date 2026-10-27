"""Generate canonical dataset card markdown from data/golden_qa.json."""

import json
from collections import Counter
from pathlib import Path


def generate_dataset_card() -> str:
    gold_path = Path("data/golden_qa.json")
    with open(gold_path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    splits = Counter(d["split"] for d in data)
    query_types = Counter(d["query_type"] for d in data)

    # Cross-tabulation
    split_order = ["train", "dev", "test"]
    cross = {s: Counter() for s in split_order}
    for d in data:
        cross[d["split"]][d["query_type"]] += 1

    lines = []
    lines.append("<!-- dataset-card:start -->")
    lines.append("## Dataset Composition")
    lines.append("")
    lines.append(f"Total verified evaluation items: **{total}**.")
    lines.append("")
    lines.append("### Splits & Purpose")
    lines.append("")
    lines.append("| Split | Count | Ratio | Purpose |")
    lines.append("|-------|-------|-------|---------|")
    lines.append(
        f"| `train` | {splits['train']} | {splits['train'] / total:.0%} | Prompt compilation, DSPy demonstration selection, and optimizer training. |"
    )
    lines.append(
        f"| `dev` | {splits['dev']} | {splits['dev'] / total:.0%} | Validation, hyperparameter tuning, and optimizer teleprompter metric convergence checks. |"
    )
    lines.append(
        f"| `test` | {splits['test']} | {splits['test'] / total:.0%} | **Quarantined holdout**. Strictly isolated from optimization and prompt tuning. |"
    )
    lines.append("")
    lines.append("### Query Type Distribution by Split")
    lines.append("")
    lines.append("| Split | Factoid | Multi-Hop | Adversarial Negative | Total |")
    lines.append("|-------|:-------:|:---------:|:--------------------:|:-----:|")
    for s in split_order:
        row_tot = sum(cross[s].values())
        lines.append(
            f"| `{s}` | {cross[s]['factoid']} | {cross[s]['multi_hop']} | {cross[s]['adversarial_negative']} | {row_tot} |"
        )
    lines.append(
        f"| **Total** | **{query_types['factoid']}** | **{query_types['multi_hop']}** | **{query_types['adversarial_negative']}** | **{total}** |"
    )
    lines.append("")
    lines.append("### Record Schema")
    lines.append("")
    sample = {
        "id": "qa_001",
        "query": data[0]["query"],
        "ground_truth_context_ids": data[0].get("ground_truth_context_ids", []),
        "ground_truth_spans": data[0].get("ground_truth_spans", []),
        "ground_truth_answer": data[0]["ground_truth_answer"],
        "query_type": data[0]["query_type"],
        "verified": data[0]["verified"],
        "split": data[0]["split"],
    }
    lines.append("```json")
    lines.append(json.dumps(sample, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Field Definitions")
    lines.append("")
    lines.append("- `id` (`str`): Unique query identifier (`qa_001` through `qa_080`).")
    lines.append("- `query` (`str`): Natural language question.")
    lines.append(
        "- `ground_truth_context_ids` (`list[str]`): Reference chunk IDs under 400/50 baseline indexing."
    )
    lines.append(
        '- `ground_truth_spans` (`list[dict]`): Chunking-independent character span annotations: `[{"doc": str, "char_start": int, "char_end": int}]` (whole-document relevance indicated by `char_end: -1`).'
    )
    lines.append(
        "- `ground_truth_answer` (`str`): Concise, factual reference answer derived directly from source documentation."
    )
    lines.append(
        "- `query_type` (`str`): One of `factoid` (single-hop), `multi_hop` (multi-document synthesis), or `adversarial_negative` (unanswerable / out-of-scope query)."
    )
    lines.append(
        "- `verified` (`bool`): Ground-truth validation flag (100% verified against raw corpus)."
    )
    lines.append(
        "- `split` (`str`): Stratified partition (`train`: 32 items, `dev`: 16 items, `test`: 32 items)."
    )
    lines.append(
        "- `note` (`str`, optional): Descriptive annotation for special edge cases (e.g. whole-document synthesis)."
    )
    lines.append("<!-- dataset-card:end -->")
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_dataset_card())

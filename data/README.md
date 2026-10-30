# Golden Evaluation Dataset (v2)

This directory contains the curated golden test set and source corpus for `rag-eval-harness`.

## Provenance

All material in this directory was **written by the repository author for this benchmark**. The ten documents in `raw/` describe a fictional "Enterprise API Platform"; they are not derived from any employer's or vendor's documentation. The 80 QA pairs and the 30 probe labels were authored and checked by the same person (single annotator). Everything here is released under the repository's MIT licence. "Verified" means each answer was checked against the fictional corpus text, not against a real system.

## Dataset Structure

- **`raw/`**: 10 engineering documents describing a fictional platform architecture, runbooks, infrastructure configurations, and incident response guides.
- **`golden_qa.json`**: 80 manually verified question-answer pairs with source context annotations and strict train/dev/test splits.
- **`human_labels.jsonl`**: 30 author-written probe QA items used for in-sample regression testing of the offline heuristic judge. Single annotator (`adityavsb`, 2026-10-24); answers are hand-authored probe targets, not live model outputs.

### Live benchmarks vs. split quarantine

The **train / dev / test** quarantine in `golden_qa.json` still applies to **DSPy teleprompter training, demo pools, and prompt tuning**—do not optimize on `test`. Published **live OpenRouter** runs in this repo scored the **full 80-item suite** in one pass per generator for exploratory end-to-end reporting (retrieval + generation + Decider judge). Treat offline retrieval sweeps on answerable-only IR metrics as the controlled retrieval experiment; live tables add generation quality and negative refusal on top of the same hybrid index.

<!-- dataset-card:start -->
## Dataset Composition

Total verified evaluation items: **80**.

### Splits & Purpose

| Split | Count | Ratio | Purpose |
|-------|-------|-------|---------|
| `train` | 32 | 40% | Prompt compilation, DSPy demonstration selection, and optimizer training. |
| `dev` | 16 | 20% | Validation, hyperparameter tuning, and optimizer teleprompter metric convergence checks. |
| `test` | 32 | 40% | **Quarantined holdout**. Strictly isolated from optimization and prompt tuning. |

### Quarantine Governance Rule
To prevent data leakage and evaluate real-world generalization:
> The `test` split must **never** be used in prompt engineering, few-shot demo pools, or DSPy teleprompter training loops. Any benchmark report must evaluate `test` out-of-sample.

### Query Type Distribution by Split

| Split | Factoid | Multi-Hop | Adversarial Negative | Total |
|-------|:-------:|:---------:|:--------------------:|:-----:|
| `train` | 19 | 6 | 7 | 32 |
| `dev` | 9 | 3 | 4 | 16 |
| `test` | 16 | 8 | 8 | 32 |
| **Total** | **44** | **17** | **19** | **80** |

### Record Schema

```json
{
  "id": "qa_001",
  "query": "What algorithm is used for distributed rate limiting at the API Gateway layer?",
  "ground_truth_context_ids": [
    "01_architecture_overview_c002"
  ],
  "ground_truth_spans": [
    {
      "doc": "01_architecture_overview",
      "char_start": 291,
      "char_end": 725
    }
  ],
  "ground_truth_answer": "Distributed rate limiting is enforced via token bucket algorithms backed by Redis.",
  "query_type": "factoid",
  "verified": true,
  "split": "train"
}
```

### Field Definitions

- `id` (`str`): Unique query identifier (`qa_001` through `qa_080`).
- `query` (`str`): Natural language question.
- `ground_truth_context_ids` (`list[str]`): Reference chunk IDs under 400/50 baseline indexing.
- `ground_truth_spans` (`list[dict]`): Chunking-independent character span annotations: `[{"doc": str, "char_start": int, "char_end": int}]` (whole-document relevance indicated by `char_end: -1`).
- `ground_truth_answer` (`str`): Concise, factual reference answer derived directly from source documentation.
- `query_type` (`str`): One of `factoid` (single-hop), `multi_hop` (multi-document synthesis), or `adversarial_negative` (unanswerable / out-of-scope query).
- `verified` (`bool`): Ground-truth validation flag (100% verified against raw corpus).
- `split` (`str`): Stratified partition (`train`: 32 items, `dev`: 16 items, `test`: 32 items).
- `note` (`str`, optional): Descriptive annotation for special edge cases (e.g. whole-document synthesis).
<!-- dataset-card:end -->

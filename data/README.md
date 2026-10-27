# Golden Evaluation Dataset (v2)

This directory contains the curated golden test set and source corpus for `rag-eval-harness`.

## Dataset Structure

- **`raw/`**: 10 engineering documents representing realistic software platform architecture, runbooks, infrastructure configurations, and incident response guides.
- **`golden_qa.json`**: 80 manually verified question-answer pairs with source context annotations and strict train/dev/test splits.

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

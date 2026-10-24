# Golden Evaluation Dataset (v2)

This directory contains the curated golden test set and source corpus for `rag-eval-harness`.

## Dataset Structure

- **`raw/`**: 10 engineering documents representing realistic software platform architecture, runbooks, infrastructure configurations, and incident response guides.
- **`golden_qa.json`**: 80 manually verified question-answer pairs with source context annotations and strict train/dev/test splits.

## Splits

The dataset follows a strict 40 / 20 / 40 stratified split:

| Split | Count | Ratio | Purpose |
|-------|-------|-------|---------|
| `train` | 32 | 40% | Prompt compilation, DSPy `BootstrapFewShot` demonstration selection, and optimizer training. |
| `dev` | 16 | 20% | Validation, hyperparameter tuning, and optimizer teleprompter metric convergence checks. |
| `test` | 32 | 40% | **Quarantined holdout**. Never seen during prompt optimization or parameter tuning. Evaluated for benchmark reports. |

### Quarantine Governance Rule
To prevent data leakage and evaluate real-world generalization:
> The `test` split must **never** be used in prompt engineering, few-shot demo pools, or DSPy teleprompter training loops. Any benchmark report must evaluate `test` out-of-sample.

## Record Schema

```json
{
  "id": "qa_001",
  "query": "What is the token expiration period for short-lived access tokens?",
  "ground_truth_answer": "Short-lived JWT access tokens expire after 15 minutes.",
  "ground_truth_context_ids": ["chunk_001"],
  "split": "train",
  "verified": true,
  "query_type": "factoid",
  "difficulty": "easy"
}
```

- `id` (`str`): Unique identifier (`qa_001` through `qa_080`).
- `query` (`str`): Natural language query.
- `ground_truth_answer` (`str`): Concise, factual answer derived directly from source documentation.
- `ground_truth_context_ids` (`list[str]`): IDs of document chunks providing verifiable evidence.
- `split` (`str`): One of `"train"`, `"dev"`, `"test"`.
- `verified` (`bool`): Verification flag; all active entries are verified against current corpus chunks.
- `query_type` (`str`): Categorization (`factoid`, `multi-hop`, `procedural`, `troubleshooting`).
- `difficulty` (`str`): `easy`, `medium`, or `hard`.

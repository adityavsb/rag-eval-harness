# Evaluation reports (local only)

Full benchmark JSON (per-query `records`, retrieved context, model outputs, judge reasoning) is written here when you run `experiments/run_eval.py`. Those files can include verbose LLM traces and are **not committed** (`reports/**/*.json` is gitignored).

**In git:** aggregate metrics only, under [`results/`](../results/).

Reproduce locally:

```bash
.venv/bin/python experiments/run_eval.py --config configs/hybrid.yaml --mode offline
.venv/bin/python experiments/run_eval.py --config configs/openrouter_cohort.yaml --mode live --stage full --model deepseek/deepseek-v4.1-flash
```

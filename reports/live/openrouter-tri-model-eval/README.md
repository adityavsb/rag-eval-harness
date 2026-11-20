# OpenRouter Live Cohort Evaluation

Config: [`configs/openrouter_cohort.yaml`](../../../configs/openrouter_cohort.yaml) (`max_tokens: 4000`, Hybrid RRF Top-4, Perplexity Decider judge).

Full run JSON files stay **local-only** (`reports/**/*.json` is gitignored). Canonical aggregate metrics are committed in [`results/openrouter_cohort_summary.json`](../../../results/openrouter_cohort_summary.json) and tracked in [`results/live_run_lineage.json`](../../../results/live_run_lineage.json).

Re-score a saved local report with Perplexity Decider without regenerating answers:

```bash
.venv/bin/python scripts/rejudge_saved_reports.py --report reports/live/openrouter-tri-model-eval/YOUR_RUN.json
```

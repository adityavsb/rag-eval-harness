# OpenRouter tri-model live evaluation

Config: [`configs/openrouter_cohort.yaml`](../../../configs/openrouter_cohort.yaml).

Full run JSON (80 queries × generator outputs × **JEV** scores) is **local-only** under this directory after you run eval. Do not commit those files.

Published **aggregate** metrics for the README live table: [`results/openrouter_cohort_summary.json`](../../../results/openrouter_cohort_summary.json).

Re-score a **local** saved generator report with **JEV**:

```bash
.venv/bin/python scripts/rejudge_saved_reports.py --report reports/live/openrouter-tri-model-eval/YOUR_RUN.json
```

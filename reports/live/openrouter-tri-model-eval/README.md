# OpenRouter tri-model live evaluation

Config: [`configs/openrouter_cohort.yaml`](../../../configs/openrouter_cohort.yaml).

Full run JSON is **local-only** after you run eval (`reports/**/*.json` is gitignored).

## Local artifact status

| Local file (if present) | Status | Use for public metrics? |
| :--- | :--- | :---: |
| `20261029T210935_99f16f5.json` | **superseded** | No — pre-JEV judge path |
| `20261029T211925_99f16f5.json` | **superseded** | No — pre-JEV judge path |
| `*_perplexity_judged.json` (local export) | **valid** (if JEV endpoint) | Yes — after verifying judge mode |
| *(none)* | — | Use [`results/openrouter_cohort_summary.json`](../../../results/openrouter_cohort_summary.json) in git |

Lineage detail: [`results/live_run_lineage.json`](../../../results/live_run_lineage.json).

Re-score a local generator report with **JEV**:

```bash
.venv/bin/python scripts/rejudge_saved_reports.py --report reports/live/openrouter-tri-model-eval/YOUR_RUN.json
```

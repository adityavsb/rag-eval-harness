# Evaluation Reports (Local Only)

Full benchmark JSON files (per-query `records`, retrieved chunks, generated answers, and judge reasoning) are written here by `experiments/run_eval.py`. They are **not committed** (`reports/**/*.json` is gitignored).

**Committed aggregates:** [`results/`](../results/) — see [`results/live_run_lineage.json`](../results/live_run_lineage.json) for canonical runs and superseded local artifacts.

| Location | Committed? | Notes |
| :--- | :---: | :--- |
| `reports/**/*.json` | No | Full per-query local eval traces |
| `results/*_summary.json` | Yes | Canonical aggregate metrics backing `README.md` |
| `results/live_run_lineage.json` | Yes | Canonical runs and superseded run notes |

Reproduce locally:

```bash
.venv/bin/python experiments/run_eval.py --config configs/hybrid.yaml --mode offline
.venv/bin/python experiments/run_eval.py --config configs/openrouter_cohort.yaml --mode live --stage full --model deepseek/deepseek-v4.1-flash
.venv/bin/python experiments/run_eval.py --config configs/agentic_crag.yaml --mode live --stage full --model deepseek/deepseek-v4.1-flash
```

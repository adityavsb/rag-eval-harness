# OpenRouter tri-model live evaluation (2026-10-29)

Config: [`configs/openrouter_cohort.yaml`](../../../configs/openrouter_cohort.yaml). Git SHA at run time: `99f16f5`.

## Canonical reports (Decider judge, apples-to-apples)

Use these for README tables and comparisons:

| File | Generator |
|------|-----------|
| [`deepseek_v4.1_perplexity_judged.json`](deepseek_v4.1_perplexity_judged.json) | `deepseek/deepseek-v4.1-flash` |
| [`qwen_3.7_perplexity_judged.json`](qwen_3.7_perplexity_judged.json) | `qwen/qwen3.7-flash` |

Judge: `perplexity/pplx-decider-flash-v1.2` via OpenRouter `/api/alpha/decisions`. Produced with [`scripts/rejudge_saved_reports.py`](../../../scripts/rejudge_saved_reports.py).

## Raw generator exports

Timestamped full runs (judge scores from the inline live path at generation time):

- [`20261029T210935_99f16f5.json`](20261029T210935_99f16f5.json) — DeepSeek
- [`20261029T211925_99f16f5.json`](20261029T211925_99f16f5.json) — Qwen

Prefer `*_perplexity_judged.json` when comparing generators under one judge.

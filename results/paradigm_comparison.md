# Live RAG Paradigm Comparison: Static Hybrid RRF vs. LangGraph Agentic CRAG

All metrics use identical generator (`deepseek/deepseek-v4.1-flash`, `max_tokens=4000`) and judge (`perplexity/pplx-decider-flash-v1.2` via `/api/alpha/decisions`). Generation quality is averaged over `n = 61` answerable queries; negative refusal over `n = 19`.

| Paradigm | Architecture | Factoid Corr. | Multi-Hop Corr. (17 Qs) | All-Hops MH Recall | Negative Refusal (19 Qs) | Mean Faithfulness | Mean Correctness | P95 Gen Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Static Hybrid RRF (Live)** | Recursive 400/50, BM25 + Dense RRF Top-4 (DeepSeek V4.1 Flash) | 90.4% | 66.7% (3.76/5) | 59.8% | 100.0% | 97.3% | 83.8% (4.46/5) | 5995 ms |
| **LangGraph Agentic CRAG (Live)** | Decompose -> Retrieve -> Grade -> Multi-Hop Synthesis (DeepSeek V4.1 Flash) | 92.5% | 71.1% (3.94/5) | 63.7% | 100.0% | 97.6% | 86.5% (4.57/5) | 12475 ms |

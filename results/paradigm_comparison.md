# Live RAG Paradigm Comparison: Static Hybrid RRF vs. LangGraph Agentic CRAG

All metrics use identical generator (`deepseek/deepseek-v4.1-flash`, `max_tokens=4000`) and judge (`perplexity/pplx-decider-flash-v1.2` via `/api/alpha/decisions`). Generation quality is averaged over `n = 61` answerable queries; negative refusal over `n = 19`.

| Paradigm | Architecture | Factoid Corr. | Multi-Hop Corr. (17 Qs) | All-Hops MH Recall | Negative Refusal (19 Qs) | Mean Faithfulness | Mean Correctness | P95 Gen Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Single-shot Hybrid (K=4)** | Recursive 400/50, BM25 + Dense RRF Top-4 (DeepSeek V4.1 Flash) | 90.6% | 66.9% (3.71/5) | 59.8% | 100.0% | 97.5% | 84.0% (4.44/5) | 5844 ms |
| **Single-shot Hybrid (K=8 control)** | Recursive 400/50, BM25 + Dense RRF Top-8 (DeepSeek V4.1 Flash) | 95.7% | 78.8% (4.24/5) | 69.6% | 100.0% | 97.9% | 91.0% (4.74/5) | 6134 ms |
| **LangGraph Agentic CRAG** | Decompose -> Retrieve -> Grade -> Rewrite -> Multi-Hop Synthesis (DeepSeek V4.1 Flash) | 92.3% | 76.3% (4.00/5) | 66.7% | 100.0% | 97.5% | 87.8% (4.57/5) | 11224 ms |

# 4-Tier RAG Paradigm Comparison: Baseline vs. Optimized vs. Compiled vs. Agentic

| Paradigm | Architecture Summary | Hit Rate@K | Multi-Hop Recall (17 Qs) | Multi-Hop Correctness | Negative Refusal (19 Qs) | Mean Faithfulness | Mean Correctness | P95 Gen Latency | Cost (80 Qs) |
|---|---|---|---|---|---|---|---|---|---|
| **Tier 1: Baseline** | Naive Single-Shot (Fixed 1000/0, Dense Top-4) | 98.4% | 85.3% | 7.4% (1.3/5) | 0.0% | 97.1% | 9.4% (1.38/5) | 0 ms | €0.0000 |
| **Tier 2: Optimized (Live)** | Two-Stage Hybrid (Live DeepSeek V4.1 Flash + Perplexity Decider) | 72.1% | 59.8% | 32.4% (2.3/5) | 100.0% | 97.8% | 80.0% (4.20/5) | 3625 ms | €0.0103 |
| **Tier 3: Compiled** | DSPy Teleprompter Program (Signatures & FewShot Demos) | 72.1% | 59.8% | 58.8% (3.4/5) | 15.8% | 96.9% | 64.4% (3.58/5) | 2 ms | €0.0000 |
| **Tier 4: Agentic (Live)** | LangGraph CRAG (Live DeepSeek V4.1 Flash + Perplexity Decider) | 72.1% | 59.8% | 47.9% (2.9/5) | 100.0% | 90.2% | 78.8% (4.17/5) | 9163 ms | €0.0217 |

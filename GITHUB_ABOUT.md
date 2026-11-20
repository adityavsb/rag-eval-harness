# GitHub Repository Profile Copy

Use this for **Settings → General → Description**, repository topics, and portfolio/LinkedIn blurbs.

## One-line description (GitHub About)

```text
Evaluation harness for engineering-runbook RAG: compares hybrid search, reranking, DSPy compilation, and LangGraph CRAG—showing a single-shot K=8 control beats CRAG (91.0% vs 87.8%) at half the latency.
```

## Suggested topics

`retrieval-augmented-generation`, `rag`, `agentic-rag`, `langgraph`, `dspy`, `hybrid-search`, `information-retrieval`, `llm-evaluation`, `mlflow`, `chromadb`, `openrouter`, `python`

## Short blurb (LinkedIn / Resume / Portfolio)

```text
Built rag-eval-harness to evaluate internal engineering-runbook RAG systems before shipping—separating retrieval quality (span-overlap Hit@K, MRR, 2,000-resample bootstrap CIs, exact McNemar), grounded generation, multi-hop synthesis, and unanswerable-query refusal. Benchmarked BM25+dense RRF, cross-encoder reranking, DSPy prompt compilation on a quarantined test split (+3.6 pp retrieved, +9.9 pp multi-hop), and a LangGraph Corrective RAG loop against a context-matched single-shot K=8 control—finding that K=8 single-shot beat CRAG (91.0% vs 87.8%, 100% refusal) at nearly half the latency and cost on a 65-chunk corpus.
```

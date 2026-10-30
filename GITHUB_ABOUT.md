# GitHub repository profile (copy-paste)

Use this for **Settings → General → Description**, topics, and the short blurb on your portfolio or CV. Keep the one-line description under GitHub’s ~350-character limit.

## One-line description (GitHub About)

```text
Hybrid RAG eval harness: offline IR (BM25+RRF, rerank, bootstrap CIs) on an authored corpus; live OpenRouter generation with JEV (Perplexity Decider). MLflow, pytest, v0.2.1.
```

## Suggested topics

`retrieval-augmented-generation`, `rag`, `information-retrieval`, `benchmark`, `mlflow`, `chromadb`, `openrouter`, `llm-evaluation`, `python`

## Longer blurb (README intro, LinkedIn, applications)

```text
RAG Evaluation & Optimization Harness — compare chunking, dense+BM25 hybrid retrieval, and cross-encoder reranking with honest IR denominators and uncertainty (bootstrap CIs, McNemar). Offline mode runs without API keys for regression; live mode calls OpenRouter chat models and scores generation with JEV (Judge Evaluation via Perplexity PPLX Decider over OpenRouter’s decisions API). Published metrics are aggregate JSON under results/; full per-query traces stay local. Fictional enterprise corpus and golden QA authored for this benchmark (MIT).
```

## Update via GitHub CLI (optional)

```bash
gh repo edit adityavsb/rag-eval-harness \
  --description "Hybrid RAG eval harness: offline IR (BM25+RRF, rerank, bootstrap CIs) on an authored corpus; live OpenRouter generation with JEV (Perplexity Decider). MLflow, pytest, v0.2.1." \
  --add-topic retrieval-augmented-generation --add-topic rag --add-topic benchmark \
  --add-topic mlflow --add-topic python
```

Website field (optional): leave blank or point to the README live-results anchor on GitHub.

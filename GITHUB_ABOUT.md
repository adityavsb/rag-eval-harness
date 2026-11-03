# GitHub repository profile (copy-paste)

Use this for **Settings → General → Description**, topics, and the short blurb on your portfolio or CV. Keep the one-line description under GitHub’s ~350-character limit.

## One-line description (GitHub About)

```text
Rigorous RAG harness comparing hybrid search, DSPy compilation, and LangGraph agentic CRAG on an authored corpus. Live OpenRouter generation + Perplexity Decider JEV.
```

## Suggested topics

`retrieval-augmented-generation`, `rag`, `agentic-rag`, `langgraph`, `langchain`, `dspy`, `information-retrieval`, `benchmark`, `mlflow`, `openrouter`, `llm-evaluation`, `python`

## Longer blurb (README intro, LinkedIn, applications)

```text
rag-eval-harness — compare chunking, dense+BM25 hybrid retrieval, cross-encoder reranking, and LangGraph agentic CRAG with honest IR denominators and uncertainty (bootstrap CIs, McNemar). Offline mode runs without API keys for regression; live mode calls OpenRouter models and scores generation with JEV (Perplexity Decider continuous decision judging). Published metrics are aggregate JSON under results/; full per-query traces stay local. Fictional enterprise corpus and golden QA authored for this benchmark (MIT).
```

## Update via GitHub CLI (optional)

```bash
gh repo edit adityavsb/rag-eval-harness \
  --description "Rigorous RAG harness comparing hybrid search, DSPy compilation, and LangGraph agentic CRAG on an authored corpus. Live OpenRouter generation + Perplexity Decider JEV." \
  --add-topic retrieval-augmented-generation --add-topic rag --add-topic agentic-rag \
  --add-topic langgraph --add-topic langchain --add-topic dspy \
  --add-topic benchmark --add-topic mlflow --add-topic python
```

Website field (optional): leave blank or point to the README live-results anchor on GitHub.

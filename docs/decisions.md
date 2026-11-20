# Architecture Decision Records (ADRs)

This document records the core architectural trade-offs and design decisions behind `rag-eval-harness`.

---

## ADR 001: In-Process ChromaDB vs. Hosted Vector Databases & FAISS

* **Status:** Accepted
* **Context:** An evaluation harness must run deterministically in local development and GitHub Actions CI without Docker sidecars or paid cloud vector services.
* **Decision:** Use in-process `chromadb.PersistentClient` (SQLite metadata store + HNSW index) paired with SHA-256 corpus/chunking fingerprinting (`VectorIndexer.corpus_fingerprint`) to force clean collection rebuilds whenever chunking parameters or source docs change.
* **Alternatives Considered:**
  1. *FAISS (`faiss-cpu`):* Fast ANN search, but requires custom metadata storage and collection lifecycle glue.
  2. *Qdrant / Pinecone / Weaviate:* Production-scale choices, but add container or remote-API dependencies that break offline CI determinism.
* **Consequences:** Zero-setup local runs (`make install && make test`) with ~4–6 ms P95 dense retrieval latency on Apple Silicon. Larger corpora (>100k chunks) can swap in a hosted backend behind `VectorIndexer` without changing retrieval or evaluation code.

---

## ADR 002: Span-Based Ground Truth & Recursive Chunking (400/50 vs. 1000/0)

* **Status:** Accepted
* **Context:** Early iterations scored retrieval by matching retrieved chunk IDs against chunk IDs generated under a 400-character index. That made a 1,000-character fixed baseline appear to collapse to 12.5% Hit Rate@4—measuring chunker alignment rather than retrieval quality.
* **Decision:** Migrate `data/golden_qa.json` to chunking-independent character spans (`ground_truth_spans`) with ≥50% overlap matching, separate answerable queries (`n = 61`) from adversarial negatives (`n = 19`), and use recursive character chunking (400 chars, 50 overlap) as the working default.
* **Consequences:** Under honest span matching, fixed 1000/0 chunks mechanically achieve 98.36% Hit@4 because 4 chunks return 4,000 characters (~20% of the entire 10-document corpus). Recursive 400/50 cuts prompt context volume by 2.5× (~1,600 chars / 6% of corpus) into 65 focused semantic units, trading raw span coverage for downstream token economy and signal-to-noise ratio.

---

## ADR 003: Single-Stage Hybrid RRF vs. Cross-Encoder Reranking

* **Status:** Accepted
* **Context:** Dense bi-encoders (`all-MiniLM-L6-v2`, 22.7M params, 384-dim) precompute chunk embeddings offline so query time is a single encode + HNSW lookup. Cross-encoders (`cross-encoder/ms-marco-MiniLM-L6-v2`, same 22.7M-param 6-layer MiniLM backbone) run full joint attention over each `(query, candidate)` pair.
* **Decision:** Adopt BM25 Okapi + dense Reciprocal Rank Fusion (`k = 60`; Cormack et al. 2009) as the default live retriever, keeping cross-encoder reranking configurable (`use_reranker: true`) rather than always-on.
* **Consequences:** On this 65-chunk corpus (`n = 61` answerable), cross-encoder reranking adds ~13× P95 retrieval latency (4.5 ms → 60.3 ms) for a +0.011 MRR shift whose 95% bootstrap CI spans zero. Hybrid RRF preserves ~4.4 ms latency while adding exact lexical matching for identifiers, error codes, and config keys.

---

## ADR 004: Custom Deterministic IR Math & Bootstrap Uncertainty

* **Status:** Accepted
* **Context:** Turnkey RAG evaluation libraries often bundle opaque prompts, silently change rubric behavior across releases, and report point estimates without confidence intervals on small test sets.
* **Decision:** Implement exact `Hit Rate@K`, `MRR@K`, `Context Precision@K`, and `Context Recall@K` directly in `src/eval/retrieval_metrics.py`, compute 2,000-resample bootstrap 95% CIs in `src/eval/runner.py`, and run paired bootstrap deltas + exact McNemar tests in `scripts/compare_runs.py`.
* **Consequences:** Retrieval math is transparent, deterministic, and covered by fast unit tests. Small-n noise is immediately visible rather than disguised as leaderboard progress.

---

## ADR 005: Separate Offline Wiring Guards vs. Live LLM-as-a-Judge

* **Status:** Accepted
* **Context:** An early offline mock LM (`OfflineRAGLM`) rewarded gold fragments in prompts and created a false +53.9% "DSPy gain." Offline heuristics are useful for CI regression testing, not for evaluating language models.
* **Decision:** Enforce explicit `mode: offline | live` semantics with fail-loud missing-key checks. Keep offline mode strictly for retrieval sweeps and pipeline regression (calibrated against 30 author-written probe labels as a regression guard only). For live runs, score all generators under a single external decision model (`perplexity/pplx-decider-flash-v1.2` via OpenRouter `/api/alpha/decisions`) and refuse to mix judge modes across comparison tables.
* **Consequences:** Eliminates self-judging bias and silent offline fallbacks. Saved local reports can be re-scored independently with `scripts/rejudge_saved_reports.py`.

---

## ADR 006: LangGraph Corrective RAG (CRAG) & Reasoning Token Budgets

* **Status:** Accepted
* **Context:** Single-shot top-4 retrieval frequently fills all four slots with chunks from a single primary runbook, starving cross-document multi-hop queries of secondary-runbook context. Additionally, reasoning-capable models (`deepseek/deepseek-v4.1-flash`, `qwen/qwen3.7-flash`) silently returned empty strings when `max_tokens: 500` exhausted their budget inside reasoning traces.
* **Decision:** Implement a bounded (`max_agent_hops: 2`) LangGraph state machine (`decompose → retrieve → grade_documents → generate | refuse`) in `src/pipeline/agentic_graph.py`, inspect all accumulated multi-hop chunks in the grader, track both `@K=4` and full-hop recall, and set `max_tokens: 4000` for live reasoning models across all paradigms.
* **Consequences:** Makes the multi-hop accuracy vs. latency trade-off measurable on an apples-to-apples footing, informing when production systems should route simple lookups to fast single-shot hybrid RAG and reserve multi-hop orchestration for compound queries.
* **Update (2026-11-20):** With a `rewrite_query` node added, CRAG beat single-shot `K = 4` (+3.8 pp, 95% CI [+0.9, +7.6]). The context-matched single-shot `K = 8` control (`configs/hybrid_k8.yaml`) did better still: 91.0% vs. CRAG's 87.8% correctness, at 6.1 s vs. 11.2 s P95. On this 65-chunk corpus the default becomes single-shot hybrid at `K = 8`, and CRAG stays available as a configurable paradigm until a larger corpus re-tests it.

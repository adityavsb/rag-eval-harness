# RAG Evaluation & Optimization Harness

**Hybrid RAG on an authored corpus—local retrieval science, live [OpenRouter](https://openrouter.ai/) LLMs, and JEV (Judge Evaluation via decision models) on generation quality.**

Compare chunking strategies, **BM25 + dense RRF**, and cross-encoder reranking with real IR numbers (Hit Rate@K, MRR, context precision/recall, **2,000-resample bootstrap CIs**, McNemar checks). Switch to **`--mode live`** and the same pipeline calls **OpenRouter chat models** (e.g. DeepSeek V4.1 Flash, Qwen 3.7 Flash) for grounded answers, then runs **JEV**: **LLM-specific metrics** (faithfulness, rubric correctness, mean grade) from **Perplexity PPLX Decider Flash** over OpenRouter’s **`/api/alpha/decisions`** decision API—one judge model for apples-to-apples generator comparison, not a lexical offline mock. Track **negative-query refusal** beside retrieval hit rate. Full per-query traces stay **local** under `reports/` (gitignored); README numbers are backed by aggregate [`results/`](results/). Offline mode runs without API keys for regression; the published OpenRouter cohort is [documented below](#benchmark-results-live-mode).

---

## What I Learned: Systems Reality vs. Theoretical Claims

Building and benchmarking this harness on Apple Silicon forced me to confront several gaps between theoretical Information Retrieval literature, LLM-as-a-judge claims, and physical systems reality:

1. **Honest IR denominators and span-based ground truth:** Chunk-ID labels tied to a 400-character index made a 1000/0 baseline look like 12.5% Hit Rate@4 until `ground_truth_spans` showed 98.36% under overlap matching—you were testing chunker alignment, not retrieval. Mixing 19 adversarial negatives into the same hit-rate denominator capped scores at 76.25%; IR metrics now use only answerable queries (**n = 61**), with **negative refusal rate** tracked separately.

2. **Small-n retrieval comparisons need intervals, not tiers:** Hybrid RRF, dense, and cross-encoder reranking looked like stacked wins on point estimates until bootstrap CIs and McNemar tests showed MRR deltas spanning zero (e.g. +0.007 [-0.014, 0.033]; one discordant pair on Hit Rate). Cross-encoder reranking adds ≈7× P95 latency (4.6 ms → 35.0 ms) for a +0.011 MRR shift that is not statistically distinguishable on this corpus—joint attention has a real compute bill even when leaderboard gaps are noise.

3. **Offline mocks are for wiring, not science:** An early `OfflineRAGLM` heuristic that rewarded gold fragments in prompts produced a fake +53.9% correctness “gain” once mistaken for DSPy progress. Integration doubles should not implement the metric under study; prompt optimization and generation quality require live model weights.

4. **Same retrieval, different generators (live OpenRouter):** With hybrid RRF fixed (72.13% Hit@4, 0.650 MRR), **DeepSeek V4.1 Flash** and **Qwen 3.7 Flash** diverged sharply on generation: mean correctness 0.826 vs 0.481 and **100% vs 31.6%** refusal on 19 adversarial negatives. Retrieval tables alone miss whether the model grounds answers or hallucinates on unanswerable prompts.

5. **JEV (Judge Evaluation via decision models):** Scoring both generators with **Perplexity PPLX Decider Flash** through OpenRouter’s `/api/alpha/decisions` is the harness’s **JEV** path—structured verdicts instead of generator-self or lexical judging. Re-run **JEV** on a local saved report with `scripts/rejudge_saved_reports.py` (see [`reports/README.md`](reports/README.md)).

### Proposed Next Engineering Directions (A, B, C)

* **Direction A: Multi-annotator judge calibration (partially done)**  
  Live OpenRouter generation plus **JEV** (Decider judging) is archived for DeepSeek and Qwen on the full 80-query suite. Still open: human labels on **live** outputs (annotator/date stamped), Cohen's κ and Spearman ρ against that set, and optional `configs/hybrid_live.yaml` on `gpt-4o-mini` for a second provider baseline.
* **Direction B: Corpus Scaling & Complex Multi-Hop Query Expansion (N > 500)**  
  Expand the raw corpus from 10 technical runbooks to 50+ documents (>500 chunks) and construct 50+ multi-hop synthesis queries requiring cross-document retrieval. Benchmark whether Cross-Encoder reranking and BM25+Dense RRF yield statistically significant gains when candidate pool competition is stiff.
* **Direction C: Deep Observability & MLflow Tracing (OpenInference Integration)**  
  Upgrade `src/utils/tracking.py` to instrument `RAGRetriever`, `VectorIndexer`, `BM25Okapi`, and `RAGGenerator` with hierarchical MLflow Tracing / OpenTelemetry spans. Capture intermediate candidate similarity scores, chunk ranks, token counts, and step-by-step latencies for visual inspection in the local MLflow dashboard (`http://localhost:5001`).

---

## Architecture

```
                                 ┌──────────────────────────────────────┐
                                 │       Curated Technical Corpus       │
                                 │     (Markdown / Specs / Guides)      │
                                 └──────────────────┬───────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Ingestion & Chunking Pipeline   │
                                  │ (Fixed vs. Recursive + Overlap)   │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Vector Indexing (HuggingFace)   │
                                  │    Local Chroma Vector Database   │
                                  └─────────────────┬─────────────────┘
                                                    │
            ┌───────────────────────────────────────┴───────────────────────────────────────┐
            │                                                                               │
            ▼                                                                               ▼
  [Lexical & Dense Search]                                                        [Hybrid & Two-Stage]
  BM25 Okapi + Dense Bi-Encoder                                                   Reciprocal Rank Fusion (RRF, k=60)
            │                                                                               │
            │                                                                               ▼
            │                                                                 Cross-Encoder Reranker
            │                                                                       (Top-K=4)
            │                                                                               │
            └───────────────────────────────────────┬───────────────────────────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │      Generation & Grounding       │
                                  │   (Context-Bounded LLM Prompt)    │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                         ┌─────────────────────────────────────────────────────┐
                         │              Evaluation & Metric Suite              │
                         │                                                     │
                         │  • Retrieval: Hit Rate@K, MRR@K, Context Precision  │
                         │  • Generation: Faithfulness & Rubric Judge          │
                         │    (offline: lexical heuristic; live: JEV / Decider) │
                         │  • System: P95 Latency & Token Usage                │
                         └──────────────────────────┬──────────────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │     MLflow Experiment Tracking    │
                                  │    (Runs, Artifacts, Dashboards)  │
                                  └───────────────────────────────────┘
```

---

## Benchmark Results (offline mode)

> [!NOTE]
> **How these numbers were produced.** Every figure on this page comes from `--mode offline`:
> retrieval uses real local models (`all-MiniLM-L6-v2`, BM25, `cross-encoder/ms-marco-MiniLM-L6-v2`) on a
> 10-document authored corpus; **generation is a deterministic mock** that echoes the top chunk;
> **the "judge" is a lexical-overlap heuristic**, not an LLM; the DSPy run uses a scripted mock LM.
> Offline mode exists so the harness can be exercised and regression-tested without live API keys.
> Retrieval metrics below are real; generation and judge figures in this section are not evidence about any
> language model. End-to-end **live** generation and Decider-judge results are in
> [Benchmark results (live mode)](#benchmark-results-live-mode).
>
> **Model selection rationale:**
> - **Bi-encoder (`all-MiniLM-L6-v2`):** 22.7M parameters, 384-dimensional dense embeddings. Selected for parameter efficiency and low CPU / Apple Silicon query latency (<5 ms), enabling local vector retrieval without requiring GPU hardware.
> - **Cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`):** 22.7M parameters, trained on MS MARCO ranking. Shares the identical 6-layer MiniLM backbone as the bi-encoder, which isolates the latency versus precision trade-off of full joint cross-attention against bi-encoder dot products while keeping model capacity constant.

The harness was run once per configuration against the **80-query Golden Evaluation Suite (v2)** across 10 architectural runbooks (44 single-hop factoid, 17 multi-hop, 19 adversarial negative queries):

### 1. Retrieval Pipeline Comparison (80-Item Golden Evaluation Suite)

All retrieval metrics are evaluated at effective **K = 4** over the **n = 61** answerable test cases with 19 adversarial negative queries tracked separately. 95% bootstrap confidence intervals (**B = 2,000** resamples) shown in brackets.

> [!NOTE]
> **Interpreting Run 1 vs. Runs 2–5 (fixed-K context volume bias):**
> * **Context Volume Distortion:** At fixed **K = 4**, Run 1 (1000/0) retrieves **4,000 characters** (~1,000 tokens; nearly **20% of the entire 10-document corpus** across only 21 total chunks). In contrast, Runs 2–5 (400/50) retrieve only **1,600 characters** (~400 tokens; 6% of the corpus across 65 chunks).
> * **The Downstream Trade-Off:** While large 1,000-character chunks mechanically elevate raw retrieval hit rate by returning massive text blocks, they impose a **2.5× prompt token volume penalty** and dilute context with irrelevant distractor text, increasing downstream model distraction. Real-world RAG optimization requires balancing retrieval coverage against prompt token economy and context signal-to-noise ratio.

| Metric | Run 1: Baseline Naive (1000/0) | Run 2: Chunk-Optimized (400/50) | Run 3: Dense + Cross-Encoder | Run 4: Hybrid RRF (BM25 + Dense) | Run 5: Hybrid RRF + Cross-Encoder | Δ vs Chunk-Optimized (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate@K** | 98.36% [95.08%, 100.0%] | 73.77% [62.30%, 83.61%] | 73.77% [62.30%, 83.61%] | 72.13% [60.66%, 83.61%] | 73.77% [62.30%, 83.61%] | -1.64% [-4.92%, 0.00%] (p = 1.00) |
| **MRR@K** | 0.906 [0.848, 0.959] | 0.643 [0.529, 0.750] | 0.654 [0.542, 0.761] | 0.650 [0.536, 0.757] | 0.654 [0.542, 0.761] | +0.007 [-0.014, 0.033] |
| **Context Precision@K** | 90.16% [84.43%, 95.22%] | 63.39% [52.19%, 74.04%] | 64.16% [53.14%, 74.41%] | 63.93% [52.60%, 74.46%] | 64.16% [53.14%, 74.41%] | +0.55% [-1.91%, 3.42%] |
| **Context Recall@K** | 94.26% [89.07%, 98.36%] | 63.11% [52.19%, 73.50%] | 65.57% [54.37%, 75.96%] | 63.39% [52.46%, 74.04%] | 65.57% [54.37%, 75.96%] | +0.27% [-1.64%, 2.46%] |
| **P95 Retrieval Latency** | 4.8 ms | 4.8 ms | 35.0 ms | 4.6 ms | 34.4 ms | -0.2 ms |

---

## Benchmark results (live mode)

> [!NOTE]
> **How these numbers were produced.** Config: [`configs/openrouter_cohort.yaml`](configs/openrouter_cohort.yaml)
> (recursive 400/50 chunking, hybrid BM25 + dense RRF, **k = 4**, no cross-encoder reranker). **Retrieval** uses the
> same local embedding and Chroma index as offline Run 4 (index fingerprint `aab84d8b…`). **Generation** calls
> OpenRouter chat models (`OPENROUTER_API_KEY` in `.env`; see Quickstart). **JEV judging** uses
> `perplexity/pplx-decider-flash-v1.2` via OpenRouter `/api/alpha/decisions` (`judge_evaluation_mode`:
> `openrouter_alpha_decisions`)—not the offline lexical heuristic in §3 below.
>
> **Scope:** All **80** golden queries (train + dev + test) were evaluated in one pass per generator; IR metrics
> use **n = 61** answerable cases, generation quality on **n = 61**, and **negative refusal rate** on **n = 19** adversarial
> negatives. Aggregate metrics (no per-query traces in git): [`results/openrouter_cohort_summary.json`](results/openrouter_cohort_summary.json)
> (from local runs at git `99f16f5`, 2026-10-29).

### OpenRouter cohort (hybrid RRF retrieval, JEV / Decider judge)

| Generator | Hit@4 | MRR@4 | Neg. refusal | Mean faith. | Mean corr. | Rubric (1–5) | P95 gen (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `deepseek/deepseek-v4.1-flash` | 72.13% | 0.650 | 100.0% | 0.890 | 0.826 | 4.36 | 3625 |
| `qwen/qwen3.7-flash` | 72.13% | 0.650 | 31.6% | 0.606 | 0.481 | 2.95 | 5914 |

Retrieval point estimates match offline hybrid RRF; differences between rows are in generation latency,
faithfulness, correctness, and refusal on unanswerable queries. Qwen’s low negative refusal rate is a
guardrail signal (answers when the corpus cannot support one), not captured in Hit Rate@4 alone.

### Live benchmark environment

Environment block shape (from local run; platform packages vary by install):

```json
{
  "platform": "macOS-27.0.1-arm64-arm-64bit-Mach-O",
  "python_version": "3.13.6",
  "machine": "arm64"
}
```

* **Full traces:** Written locally under [`reports/live/`](reports/live/) when you run eval (gitignored).
* **Aggregates in git:** [`results/openrouter_cohort_summary.json`](results/openrouter_cohort_summary.json).

---

### 2. DSPy Generation Prompt Optimization

`src/optimize/compile.py` compiles `RAGModule` with `BootstrapFewShotWithRandomSearch` on the `train` split, selects on `dev`, and evaluates on the quarantined `test` split. **No generation results are published yet**: the only runs so far used the deterministic offline mock LM (`OfflineRAGLM`), which cannot produce evidence about prompt quality. A live run (`--mode live --model gpt-4o-mini`) is required before any number appears here.

---

### 3. Offline heuristic judge — agreement with author-written probe labels

The offline (no-API) judge is a lexical heuristic (`LLMJudge._offline_faithfulness`, `_offline_correctness`). Its thresholds were set by inspecting the 30 probe labels in `data/human_labels.jsonl` (single annotator: repository author), so the figures below are **in-sample** and describe fit, not generalisation. They are retained only as a regression guard for the heuristic. Live faithfulness and correctness in the [OpenRouter cohort table](#openrouter-cohort-hybrid-rrf-retrieval-jev--decider-judge) use **JEV** (Decider decision models), not these rules.

| Calibration Metric | Observed Score | 95 % CI | Reference Threshold | Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Evaluated Samples** | 30 | - | n ≥ 30 | Author-written probe set |
| **Faithfulness Raw Agreement** | **83.3%** | - | ≥ 80.0% | Binary entailment consistency |
| **Faithfulness Cohen's Kappa (κ)** | **0.667** | [0.40, 0.93] | ≥ 0.600 | Substantial agreement beyond chance (Landis & Koch 1977) |
| **Correctness Spearman Rho (ρ)** | **0.872** | [0.77, 0.93] | ≥ 0.700 | Monotonic rank correlation (p = 3.69 × 10⁻¹⁰) |

---

### Benchmark environment
Captured automatically into each local report (`environment` key). Example package pins from an offline hybrid RRF run at commit `9a083af` (see [`results/offline_retrieval_summary.json`](results/offline_retrieval_summary.json)):

```json
{
  "platform": "macOS-27.0.1-arm64-arm-64bit-Mach-O",
  "python_version": "3.13.6",
  "machine": "arm64",
  "packages": {
    "chromadb": "1.5.9",
    "mlflow": "3.17.2",
    "sentence-transformers": "6.1.0",
    "dspy": "3.4.1",
    "rank-bm25": "0.2.2",
    "torch": "2.14.1",
    "scipy": "1.18.1",
    "pydantic": "2.14.1"
  }
}
```

* **Aggregate metrics in git:** [`results/offline_retrieval_summary.json`](results/offline_retrieval_summary.json). **Full JSON traces** from eval runs stay local under [`reports/`](reports/) (gitignored).

---

## Engineering Analysis & Friction

### 1. The Impact of Chunking Boundaries and Corpus Coverage
Large fixed-width chunks (1000 characters with 0 overlap) produce only 21 chunks across the entire 10-document technical corpus. In early iterations of this benchmark, this baseline was reported as 12.5% Hit Rate; that was an evaluation artifact caused by scoring 1000-character chunks against 400-character chunk ID labels rather than true document content spans. Under span-overlap matching, retrieving top **K = 4** returns 4,000 characters (~19% of the total indexed knowledge base), mechanically inflating span-overlap hit rate (98.36%) while flooding context windows with un-chunked document bodies. Switching to recursive character chunking (400 characters with 50 overlap) produces 65 focused semantic units. On this properly chunked corpus, top-4 retrieval covers ~6% of the corpus, achieving 73.77% Hit Rate@4 [62.30%, 83.61%] and 63.39% Context Precision@4 [52.19%, 74.04%].

### 2. The Cross-Encoder Latency vs. Precision Trade-off
Bi-encoders embed corpus chunks offline, so query-time work is one query encoding plus an approximate nearest-neighbour (HNSW) search; cross-encoders require full cross-attention evaluation per (query, chunk) candidate, which is why reranking compute scales with candidate count:
* **Retrieval Gain:** Cross-Encoder reranking marginally lifts MRR from 0.643 to 0.654 (+0.011 [-0.042, 0.066]) and Context Precision from 63.39% to 64.16% (+0.77% [-4.28%, 5.88%]).
* **Friction & Latency Penalty:** Joint self-attention incurs a ≈7× latency penalty (4.8 ms → 35.0 ms P95). While 35.0 ms remains within interactive SLAs (<150 ms), the paired difference confidence interval spans zero, demonstrating that cross-encoder gains are not statistically significant on small evaluation corpora (**n = 61**).

### 3. Lexical-Dense Fusion (BM25 + Dense RRF)
Dense vector search and lexical BM25 Okapi were fused using Reciprocal Rank Fusion (**k = 60**; Cormack et al. 2009):
* **Performance:** Achieved 72.13% Hit Rate@4 [60.66%, 83.61%] and 0.650 MRR@4 [0.536, 0.757]. Paired comparison against pure dense retrieval yields exact McNemar test **p = 1.0000** (1 discordant pair out of 61 queries).
* **Latency:** Preserves bi-encoder retrieval latency (4.6 ms P95), offering hybrid vocabulary matching without the 7× inference overhead of a cross-encoder reranker.

### 4. Generation Prompt Optimization Status
Offline optimization runs execute `OfflineRAGLM`, a deterministic test double designed strictly to validate teleprompter compilation, parameter serialization, and pipeline integration without external network calls. **DSPy compile deltas are still offline-only**; live OpenRouter results above evaluate the fixed grounded prompt, not a compiled DSPy program.

### 5. Offline Heuristic Judge Guardrails
The offline evaluation mode relies on regex entity checking and keyword overlap rules calibrated against 30 in-sample probe labels (`data/human_labels.jsonl`; Landis & Koch 1977 band 0.61–0.80). It serves solely as a regression guard for local offline development. Published live scores use **JEV** with the Perplexity Decider (see live benchmark section).

### 6. Live OpenRouter cohort
Hybrid RRF retrieval is held constant while OpenRouter generators swap under a shared **JEV** Decider judge. DeepSeek achieved full refusal on adversarial negatives and higher mean correctness; Qwen answered many unanswerable prompts despite similar retrieval. P95 generation latency is seconds-scale (API-bound), unlike sub-40 ms offline retrieval.

---

## System Capabilities

* **Deterministic Information Retrieval Scoring:** Formulations for exact `Hit Rate@K`, `MRR@K`, `Context Precision@K`, and `Context Recall@K` with 2,000-resample bootstrap 95% confidence intervals and span-overlap matching (RAGAS-style; Es et al. 2024).
* **Vector Engine & Embeddings:** ChromaDB in-process vector store (`PersistentClient`; SQLite metadata + HNSW index) using normalized 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings (Reimers & Gurevych 2019).
* **Hybrid Retrieval (BM25 + Dense RRF):** Reciprocal Rank Fusion (**k = 60**; Cormack et al. 2009) combining lexical BM25 Okapi (Robertson & Zaragoza 2009) with dense embeddings, followed by optional Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`).
* **OpenRouter provider layer:** Multi-provider LLM abstraction via LiteLLM with OpenRouter routing (`src/llm/provider.py`), disk-backed response caching (`.cache/llm_responses/`), and token usage helpers in `src/llm/pricing.py`.
* **JEV (Judge Evaluation via decision models):** Live generation quality scored through OpenRouter `/api/alpha/decisions` (e.g. `perplexity/pplx-decider-flash-v1.2`) in `src/eval/llm_judge.py`; `scripts/rejudge_saved_reports.py` re-applies **JEV** to archived answers.
* **Live evaluation staging:** `experiments/run_eval.py` supports `--mode live`, `--stage check-one|check-remaining|full`, per-query checkpoint resume under `reports/live/checkpoints/`, and `--model` / `--judge-model` overrides.
* **Curated Golden Evaluation Suite (v2):** 80 manually verified question-answer pairs (author-written, fictional corpus; see `data/README.md`) stratified into 40% train, 20% dev, and a strictly quarantined 40% test split across 10 architectural runbooks.
* **Human-in-the-Loop Judge Validation:** Judge calibration framework computing Raw Agreement %, Cohen's Kappa (κ; Cohen 1960; Landis & Koch 1977 band 0.61–0.80 for substantial agreement) for binary faithfulness, and Spearman Rank Correlation (ρ) for rubric correctness against 30 author-written probe labels.
* **DSPy Prompt Optimization:** Few-shot RAG generation pipelines with `BootstrapFewShotWithRandomSearch` teleprompters (Khattab et al. 2023) trained on `train`/`dev` splits with demo quality gates; offline mode serves as a deterministic compilation smoke test.
* **MLflow Tracking:** Persists run parameters, aggregate metrics, environment telemetry, and full JSON run reports into `sqlite:///mlflow.db`.

---

## Quickstart & Reproducibility

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/adityavsb/rag-eval-harness.git
cd rag-eval-harness

# Create virtual environment and install package in editable mode
make install
```

### 2. Run Test Suite
```bash
make test
```
All 46 unit tests run cleanly; cold run ≈ 40 s (model download + encode), warm ≈ 10-20 s.

### 3. Run Benchmark Evaluations
```bash
# Evaluate baseline configuration
make eval

# Benchmark with hybrid BM25 + dense RRF retrieval
.venv/bin/python experiments/run_eval.py --config configs/hybrid.yaml --mode offline

# Run judge calibration against human ground-truth annotations
.venv/bin/python src/eval/judge_agreement.py --mode offline

# Compile DSPy generation program using BootstrapFewShot teleprompter
.venv/bin/python src/optimize/compile.py --mode offline
```

### 4. Live evaluation (OpenRouter)

Set `OPENROUTER_API_KEY` in a local `.env` file at the repo root (loaded by `src/utils/config.load_dotenv`; never commit secrets).

```bash
# Stage 1: one query, generator only, offline judge
.venv/bin/python experiments/run_eval.py --config configs/openrouter_cohort.yaml --mode live --stage check-one

# Full 80-query sweep — run once per generator (Decider judge in config)
.venv/bin/python experiments/run_eval.py --config configs/openrouter_cohort.yaml --mode live --stage full \
  --model deepseek/deepseek-v4.1-flash

.venv/bin/python experiments/run_eval.py --config configs/openrouter_cohort.yaml --mode live --stage full \
  --model qwen/qwen3.7-flash

# Re-score saved generator JSON with JEV / Decider only (no regeneration)
.venv/bin/python scripts/rejudge_saved_reports.py --model deepseek
.venv/bin/python scripts/rejudge_saved_reports.py --model qwen

# Optional: list active free models on OpenRouter
.venv/bin/python scripts/list_free_models.py
```

OpenAI live baseline (separate config): `configs/hybrid_live.yaml` with `OPENAI_API_KEY`.

### 5. Interactive Ad-Hoc Querying
Test any ad-hoc query and inspect retrieved chunks, similarity scores, latency, and grounded responses:
```bash
# Single query
.venv/bin/python experiments/run_pipeline.py --query "What is the token expiration period for short-lived access tokens?"

# Or launch interactive REPL shell
.venv/bin/python experiments/run_pipeline.py
```

### 6. Inspect Runs in MLflow Dashboard
Launch the local MLflow dashboard to view side-by-side run comparisons and metric curves:
```bash
make mlflow
```
Open **`http://localhost:5001`** in your browser.

---

## Repository Layout

```
rag-eval-harness/
├── configs/                       # Declarative pipeline configs (YAML)
│   ├── baseline.yaml              # Fixed chunking baseline
│   ├── chunk_optimized.yaml       # Recursive chunking
│   ├── rerank_optimized.yaml      # Dense + Cross-Encoder reranking
│   ├── hybrid.yaml                # BM25 + Dense RRF fusion
│   ├── hybrid_rerank.yaml         # Hybrid RRF + Cross-Encoder reranking
│   ├── hybrid_live.yaml           # Live OpenAI gpt-4o-mini hybrid RRF baseline
│   └── openrouter_cohort.yaml     # Live OpenRouter DeepSeek/Qwen + Decider judge
├── data/
│   ├── raw/                       # Curated engineering runbooks (10 docs)
│   ├── golden_qa.json             # 80-case verified golden set (train/dev/test)
│   ├── human_labels.jsonl         # 30 human-labeled QA records for judge calibration
│   └── README.md                  # Dataset card, taxonomy, and split quarantine rules
├── results/                       # Aggregate benchmark metrics (committed)
│   ├── offline_retrieval_summary.json
│   └── openrouter_cohort_summary.json
├── reports/                       # Full eval JSON output (local, gitignored)
│   ├── README.md
│   ├── offline/
│   └── live/
├── src/
│   ├── llm/
│   │   ├── cache.py               # SHA-256 disk response cache
│   │   ├── pricing.py             # Provider token-rate helpers
│   │   └── provider.py            # LiteLLM routing and caching wrapper
│   ├── pipeline/
│   │   ├── ingestion.py           # Fixed & recursive chunking strategies
│   │   ├── indexing.py            # Local HuggingFace embeddings & ChromaDB
│   │   ├── retriever.py           # Dense, BM25, and Hybrid RRF retriever
│   │   └── generator.py           # Grounded response generation & citations
│   ├── eval/
│   │   ├── retrieval_metrics.py   # HitRate, MRR, Precision, Recall IR math
│   │   ├── llm_judge.py           # Structured Faithfulness & Correctness judge
│   │   ├── judge_agreement.py     # Cohen's kappa and Spearman rho calibration
│   │   └── runner.py              # End-to-end evaluation runner
│   ├── optimize/
│   │   ├── dspy_program.py        # Typed DSPy RAGModule and signature
│   │   └── compile.py             # Teleprompter compilation & delta evaluation
│   └── utils/
│       ├── config.py              # Pydantic schema validation & YAML loader
│       └── tracking.py            # MLflow run & artifact logger
├── experiments/
│   ├── run_eval.py                # Benchmark evaluation CLI
│   └── run_pipeline.py            # Interactive ad-hoc query CLI
├── scripts/
│   ├── compare_runs.py            # Paired run comparison & exact McNemar test
│   ├── dataset_card.py            # Dataset split & query type cross-tabulation
│   ├── rejudge_saved_reports.py   # Decider re-judge on archived live reports
│   └── list_free_models.py        # OpenRouter free-tier model discovery
├── tests/                         # Comprehensive pytest test suite (46 tests)
├── Makefile                       # Developer automation shortcuts
├── pyproject.toml                 # Package definition & dependencies
└── README.md
```

---

## References

- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR 2009*. https://doi.org/10.1145/1571941.1572114
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4). https://doi.org/10.1561/1500000019
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019*. https://doi.org/10.18653/v1/D19-1410
- Khattab, O., et al. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv:2310.03714.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAs: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024 System Demonstrations*. https://doi.org/10.18653/v1/2024.eacl-demo.16
- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1). https://doi.org/10.1177/001316446002000104
- Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1). https://doi.org/10.2307/2529310

---

## License
MIT License.

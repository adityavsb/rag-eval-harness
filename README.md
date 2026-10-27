# Production RAG Evaluation & Optimization Harness

An open-source, metric-driven benchmarking harness for Retrieval-Augmented Generation (RAG) systems. Designed to transition LLM retrieval pipelines from intuition-based prototypes into reproducible, testable software.

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

## Empirical Benchmark Results

The harness was comprehensively evaluated against the **80-query Golden Evaluation Suite (v2)** across 10 architectural runbooks (44 single-hop factoid, 17 multi-hop, 19 adversarial negative queries) and physical Apple M4 hardware:

### 1. Retrieval Pipeline Comparison (80-Item Golden Evaluation Suite)

All retrieval metrics are evaluated over the $n = 61$ answerable test cases with 19 adversarial negative queries tracked separately. 95% bootstrap confidence intervals ($B = 2,000$ resamples) shown in brackets.

| Metric | Run 1: Baseline Naive (1000/0) | Run 2: Chunk-Optimized (400/50) | Run 3: Dense + Cross-Encoder | Run 4: Hybrid RRF (BM25 + Dense) | Run 5: Hybrid RRF + Cross-Encoder | Δ vs Chunk-Optimized (95% CI) | Target SLA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate@K** | 98.36% [95.08%, 100.0%] | 73.77% [62.30%, 83.61%] | 73.77% [62.30%, 83.61%] | 72.13% [60.66%, 83.61%] | 73.77% [62.30%, 83.61%] | -1.64% [-4.92%, 0.00%] ($p=1.00$) | $\ge 85.0\%$ |
| **MRR@K** | 0.906 [0.848, 0.959] | 0.643 [0.529, 0.750] | 0.654 [0.542, 0.761] | 0.650 [0.536, 0.757] | 0.654 [0.542, 0.761] | +0.007 [-0.014, 0.033] | $\ge 0.700$ |
| **Context Precision@K** | 90.16% [84.43%, 95.22%] | 63.39% [52.19%, 74.04%] | 64.16% [53.14%, 74.41%] | 63.93% [52.60%, 74.46%] | 64.16% [53.14%, 74.41%] | +0.55% [-1.91%, 3.42%] | $\ge 75.0\%$ |
| **Context Recall@K** | 94.26% [89.07%, 98.36%] | 63.11% [52.19%, 73.50%] | 65.57% [54.37%, 75.96%] | 63.39% [52.46%, 74.04%] | 65.57% [54.37%, 75.96%] | +0.27% [-1.64%, 2.46%] | $\ge 80.0\%$ |
| **P95 Retrieval Latency** | 4.8 ms | 4.8 ms | 35.0 ms | 4.6 ms | 34.4 ms | -0.2 ms | $< 150\text{ ms}$ |

---

### 2. DSPy Generation Prompt Optimization

`src/optimize/compile.py` compiles `RAGModule` with `BootstrapFewShotWithRandomSearch` on the `train` split, selects on `dev`, and evaluates on the quarantined `test` split. **No generation results are published yet**: the only runs so far used the deterministic offline mock LM (`OfflineRAGLM`), which cannot produce evidence about prompt quality. A live run (`--mode live --model gpt-4o-mini`, projected cost < €0.05 by `src/llm/pricing.py`) is required before any number appears here.

---

### 3. Offline heuristic judge — agreement with author-written probe labels

The offline (no-API) judge is a lexical heuristic (`LLMJudge._offline_faithfulness`, `_offline_correctness`). Its thresholds were set by inspecting the 30 probe labels in `data/human_labels.jsonl` (single annotator: repository author), so the figures below are **in-sample** and describe fit, not generalisation. They are retained only as a regression guard for the heuristic. A real model judge calibration requires a live run and a label set the heuristic was not tuned on.

| Calibration Metric | Observed Score | 95 % CI | Acceptance Target | Result | Meaning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Evaluated Samples** | 30 | - | $\ge 30$ | **PASS** | Probe set; see CI column |
| **Faithfulness Raw Agreement** | **83.3%** | - | $\ge 80.0\%$ | **PASS** | High binary entailment consistency |
| **Faithfulness Cohen's Kappa ($\kappa$)** | **0.667** | [0.40, 0.93] | $\ge 0.600$ | **PASS** | Substantial agreement beyond chance |
| **Correctness Spearman Rho ($\rho$)** | **0.872** | [0.77, 0.93] | $\ge 0.700$ | **PASS** | Strong monotonic rank correlation ($p = 3.69 \times 10^{-10}$) |

---

### Benchmark Environment & Physical Hardware Telemetry

Environment is recorded in every report under `environment`; the table below is copied from `reports/offline/<run>/latest.json`:

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

* **Raw Telemetry Logs:** Exported JSON reports are archived in [`reports/offline/`](reports/offline/).

---

## Engineering Analysis & Friction

### 1. The Impact of Chunking Boundaries and Corpus Coverage
Large fixed-width chunks (1000 characters with 0 overlap) produce only 21 chunks across the entire 10-document technical corpus. Retrieving top $K=4$ returns 4,000 characters (~19% of the total indexed knowledge base), mechanically inflating span-overlap hit rate (98.36%) while flooding context windows with un-chunked document bodies. Switching to recursive character chunking (400 characters with 50 overlap) produces 65 focused semantic units. On this properly chunked corpus, top-4 retrieval covers ~6% of the corpus, achieving 73.77% Hit Rate@4 [62.30%, 83.61%] and 63.39% Context Precision@4 [52.19%, 74.04%].

### 2. The Cross-Encoder Latency vs. Precision Trade-off
Bi-encoders embed queries and documents independently into vector space ($O(1)$ dot-product at search time). Cross-encoders perform joint full self-attention over $(query, passage)$ pairs, capturing conditional semantic dependencies:
* **Retrieval Gain:** Cross-Encoder reranking marginally lifts MRR from 0.643 to 0.654 (+0.011 [-0.042, 0.066]) and Context Precision from 63.39% to 64.16% (+0.77% [-4.28%, 5.88%]).
* **Friction & Latency Penalty:** Joint self-attention incurs a $\approx 7\times$ latency penalty ($4.8\text{ ms} \rightarrow 35.0\text{ ms}$ P95). While 35.0 ms remains within interactive SLAs ($<150\text{ ms}$), the paired difference confidence interval spans zero, demonstrating that cross-encoder gains are not statistically significant on small evaluation corpora ($n=61$).

### 3. Lexical-Dense Fusion (BM25 + Dense RRF)
Dense vector search and lexical BM25 Okapi were fused using Reciprocal Rank Fusion ($k=60$):
* **Performance:** Achieved 72.13% Hit Rate@4 [60.66%, 83.61%] and 0.650 MRR@4 [0.536, 0.757]. Paired comparison against pure dense retrieval yields exact McNemar test $p = 1.0000$ (1 discordant pair out of 61 queries).
* **Latency:** Preserves bi-encoder retrieval latency (4.6 ms P95), offering hybrid vocabulary matching without the 7x inference cost of a cross-encoder reranker.

### 4. Generation Prompt Optimization Status
Offline optimization runs execute `OfflineRAGLM`, a deterministic test double designed strictly to validate teleprompter compilation, parameter serialization, and pipeline integration without external network calls or API costs. Generation quality benchmarks on `gpt-4o-mini` require a live run (`--mode live`).

### 5. Offline Heuristic Judge Guardrails
The offline evaluation mode relies on regex entity checking and keyword overlap rules calibrated against 30 in-sample probe labels (`data/human_labels.jsonl`). It serves solely as a regression guard for local offline development. Production grade LLM-as-a-judge scoring is evaluated exclusively in live mode.


---

## System Capabilities

* **Deterministic Information Retrieval Scoring:** Formulations for exact `Hit Rate@K`, `MRR@K`, `Context Precision@K`, and `Context Recall@K` with support for exact chunk ID and section-prefix matching.
* **Hybrid Retrieval (BM25 + Dense RRF):** Reciprocal Rank Fusion ($k=60$) combining BM25 keyword matching with dense embedding vector similarity, followed by optional Cross-Encoder reranking.
* **Provider Layer & Cost Accounting:** Multi-provider LLM abstraction via LiteLLM with disk-backed response caching (`.cache/llm_responses/`) and deterministic token/pricing models in EUR across major foundation providers.
* **Curated Golden Evaluation Suite (v2):** 80 manually verified question-answer pairs stratified into 40% train, 20% dev, and a strictly quarantined 40% test split across 10 architectural runbooks.
* **Human-in-the-Loop Judge Validation:** Judge calibration framework computing Raw Agreement %, Cohen's Kappa ($\kappa \ge 0.60$) for binary faithfulness, and Spearman Rank Correlation ($\rho \ge 0.70$) for rubric correctness against 30 human ground-truth labels.
* **DSPy Prompt Optimization:** Compiled few-shot RAG generation pipelines with `BootstrapFewShot` teleprompters trained on `train`/`dev` splits and benchmarked out-of-sample against the quarantined `test` holdout split.
* **Industrial MLflow Tracking:** Persists all runs, hyperparameter sweeps, metric distributions, and full JSON run reports into `sqlite:///mlflow.db`.

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
All 42 unit tests pass; cold run ≈ 38 s (model download + encode), warm ≈ 10-20 s.

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

### 4. Interactive Ad-Hoc Querying
Test any ad-hoc query and inspect retrieved chunks, similarity scores, latency, and grounded responses:
```bash
# Single query
.venv/bin/python experiments/run_pipeline.py --query "What is the token expiration period for short-lived access tokens?"

# Or launch interactive REPL shell
.venv/bin/python experiments/run_pipeline.py
```

### 5. Inspect Runs in MLflow Dashboard
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
│   └── hybrid_rerank.yaml         # Hybrid RRF + Cross-Encoder reranking
├── data/
│   ├── raw/                       # Curated engineering runbooks (10 docs)
│   ├── golden_qa.json             # 80-case verified golden set (train/dev/test)
│   ├── human_labels.jsonl         # 30 human-labeled QA records for judge calibration
│   └── README.md                  # Dataset card, taxonomy, and split quarantine rules
├── reports/
│   ├── live/                      # Real model benchmark evaluations
│   └── offline/                   # Deterministic offline benchmark evaluations
├── src/
│   ├── llm/
│   │   ├── cache.py               # SHA-256 disk response cache
│   │   ├── pricing.py             # Provider rates and cost estimation (EUR)
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
│   │   ├── compile.py             # BootstrapFewShot compilation & delta evaluation
│   │   └── compiled_dspy_rag.json # Compiled few-shot program artifact
│   └── utils/
│       ├── config.py              # Pydantic schema validation & YAML loader
│       └── tracking.py            # MLflow run & artifact logger
├── experiments/
│   ├── run_eval.py                # Benchmark evaluation CLI
│   └── run_pipeline.py            # Interactive ad-hoc query CLI
├── tests/                         # Comprehensive pytest test suite (39 tests)
├── Makefile                       # Developer automation shortcuts
├── pyproject.toml                 # Package definition & dependencies
└── README.md
```

---

## License
MIT License.

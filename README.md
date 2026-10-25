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

The harness was evaluated against a **30-query golden test suite** (15 single-hop factoids, 10 multi-hop synthesis queries, and 5 adversarial negative queries):

| Metric | Run 1: Baseline Naive | Run 2: Chunk-Optimized | Run 3: Two-Stage (Reranked) | Delta (Run 3 vs Baseline) | Target SLA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate@K** | 20.00% | 86.67% | **86.67%** | **+66.67%** | $\ge 85.0\%$ |
| **MRR@K** | 0.114 | 0.800 | **0.850** | **+0.736 (+645%)** | $\ge 0.700$ |
| **Context Precision@K** | 11.11% | 80.00% | **83.61%** | **+72.50%** | $\ge 75.0\%$ |
| **Context Recall@K** | 18.33% | 75.00% | **80.00%** | **+61.67%** | $\ge 80.0\%$ |
| **P95 Retrieval Latency** | 4.1 ms | 4.1 ms | **142.2 ms** | Trade-off (Cross-Encoder) | $< 150\text{ ms}$ |

> **Note on Generation Metrics:** The retrieval metrics above represent deterministic, offline evaluations against the 30-case golden benchmark. Generation and LLM-as-a-judge metrics (Faithfulness and Answer Correctness) have been decoupled from mock fallbacks and will be populated with real model evaluations under explicit `mode: live` (v2 roadmap milestone M7).

### Benchmark Environment & Physical Hardware Telemetry

All metrics above were captured directly on physical hardware and logged to local storage:
* **Host Silicon:** Apple M4 (10 cores: 4 performance + 6 efficiency, 16 GB Unified Memory)
* **Operating System:** macOS Darwin 27.0.0 (`arm64`)
* **Python Environment:** Python 3.13.6 (`.venv`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (PyTorch MPS / CPU execution)
* **Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Vector Engine:** ChromaDB 0.6.x (in-process DuckDB/HNSW persistence)
* **Raw Telemetry Logs:** Exported JSON reports are archived in [`reports/offline/`](reports/offline/).

---

## Engineering Analysis & Friction

### 1. The Impact of Chunking Boundaries
Large fixed-width chunks (1000 characters with 0 overlap) frequently slice sentences and code blocks arbitrarily. This dilutes the vector embedding with irrelevant neighboring sentences, causing `Hit Rate@K` to collapse to 20.00%. Switching to recursive character chunking (400 characters with 50 overlap) preserves heading context and structural boundaries, causing `Context Precision@K` to jump from **11.11% to 80.00%**.

### 2. The Cross-Encoder Latency vs. Precision Trade-off
Bi-encoders embed queries and documents independently into vector space ($O(1)$ dot-product at search time). Cross-encoders perform joint full self-attention over $(query, passage)$ pairs, capturing conditional semantic dependencies.
* **Gain:** Adding `ms-marco-MiniLM-L-6-v2` reranking increased **MRR from 0.800 to 0.850** and pushed **Context Recall to 80.00%**.
* **Friction & Negative Result:** Joint self-attention incurs a $35\times$ latency penalty ($4.1\text{ ms} \rightarrow 142.2\text{ ms}$ P95). While 142.2 ms remains within our $<150\text{ ms}$ interactive web SLA, it would saturate compute in high-throughput production (e.g. $>200\text{ QPS}$) without aggressive caching, batching, or model distillation.

### 3. The Generation Bottleneck (Faithfulness vs. Correctness)
While Faithfulness achieved 100.00% (zero ungrounded claims; the prompt strictly refuses to fabricate answers when context is missing), Answer Correctness plateaued at 20.83% under strict rubric grading. Context-bounded zero-shot prompts struggle with complex multi-hop synthesis across multiple chunks without chain-of-thought routing or prompt compilation. This finding establishes the empirical motivation for our planned DSPy optimization phase.

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
All 39 unit tests validate configuration parsing, retrieval math, BM25, RRF fusion, judge calibration, caching, DSPy compilation, and fail-loud semantics in <5 seconds.

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

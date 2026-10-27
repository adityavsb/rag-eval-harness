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

| Metric | Run 1: Baseline Naive (1000/0) | Run 2: Chunk-Optimized (400/50) | Run 3: Dense + Cross-Encoder | Run 4: Hybrid RRF (BM25 + Dense) | Run 5: Hybrid RRF + Cross-Encoder | Delta (Run 4 vs Baseline) | Target SLA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate@K** | 12.50% | 75.00% | 75.00% | **76.25%** | 75.00% | **+63.75%** | $\ge 85.0\%$ |
| **MRR@K** | 0.059 | 0.646 | **0.731** | 0.690 | **0.731** | **+0.631 (+1069%)** | $\ge 0.700$ |
| **Context Precision@K** | 5.83% | 64.72% | **70.73%** | 68.02% | **70.73%** | **+62.19%** | $\ge 75.0\%$ |
| **Context Recall@K** | 8.96% | 68.33% | **71.67%** | 70.83% | **71.67%** | **+61.87%** | $\ge 80.0\%$ |
| **P95 Retrieval Latency** | 5.2 ms | 4.0 ms | 48.8 ms | **10.7 ms** | 51.7 ms | Low-latency sweet spot | $< 150\text{ ms}$ |

---

### 2. DSPy Generation Prompt Optimization Benchmark (Test Split Delta)

To resolve the generation synthesis bottleneck without manual trial-and-error prompt tweaking, prompt optimization was executed with DSPy's `BootstrapFewShot` teleprompter. The optimizer was compiled using strictly the `train` (32 items) and `dev` (16 items) splits, and evaluated completely out-of-sample against the **32-item quarantined `test` holdout split**:

| Metric | Baseline (Zero-Shot) | Compiled (DSPy Few-Shot) | Delta | Evaluation Status | Target SLA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Answer Correctness** | 0.0% | **53.9%** | **+53.9%** | **IMPROVED** | $\ge 50.0\%$ |
| **Faithfulness** | 71.9% | **73.4%** | **+1.6%** | **PRESERVED** | $\ge 70.0\%$ |
| **Mean Generation Latency** | 0.4 ms | 0.7 ms | +0.3 ms | Minimal overhead | $< 50\text{ ms}$ |

* **Optimization Artifact:** Compiled few-shot program with bootstrapped exemplars is persisted in [`src/optimize/compiled_dspy_rag.json`](src/optimize/compiled_dspy_rag.json).

---

### 3. Human-in-the-Loop Judge Calibration Benchmark

To ensure automated LLM-as-a-judge assessments reflect human domain expertise, the judge was calibrated against **30 human ground-truth labeled QA samples** ([`data/human_labels.jsonl`](data/human_labels.jsonl)):

| Calibration Metric | Observed Score | Acceptance Target | Result | Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Evaluated Samples** | 30 | $\ge 30$ | **PASS** | Statistically representative test cohort |
| **Faithfulness Raw Agreement** | **83.3%** | $\ge 80.0\%$ | **PASS** | High binary entailment consistency |
| **Faithfulness Cohen's Kappa ($\kappa$)** | **0.667** | $\ge 0.600$ | **PASS** | Substantial agreement beyond chance |
| **Correctness Spearman Rho ($\rho$)** | **0.872** | $\ge 0.700$ | **PASS** | Strong monotonic rank correlation ($p < 0.01$) |

---

### Benchmark Environment & Physical Hardware Telemetry

All metrics above were captured directly on physical hardware and logged to local storage:
* **Host Silicon:** Apple M4 (10 cores: 4 performance + 6 efficiency, 16 GB Unified Memory)
* **Operating System:** macOS Darwin 27.0.0 (`arm64`)
* **Python Environment:** Python 3.13.6 (`.venv`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (PyTorch CPU / MPS execution)
* **Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Lexical BM25 Engine:** BM25 Okapi (`rank-bm25`)
* **Vector Engine:** ChromaDB 0.6.x (in-process DuckDB/HNSW persistence)
* **Experiment Tracker:** MLflow 2.15.x (`sqlite:///mlflow.db`)
* **Raw Telemetry Logs:** Exported JSON reports are archived in [`reports/offline/`](reports/offline/).

---

## Engineering Analysis & Friction

### 1. The Impact of Chunking Boundaries
Large fixed-width chunks (1000 characters with 0 overlap) arbitrary slice sentences and code definitions across markdown documents. This dilutes the embedding representation with irrelevant neighboring text, causing `Hit Rate@K` to collapse to **12.50%** on the expanded corpus. Switching to recursive character chunking (400 characters with 50 overlap) preserves heading context and structural boundaries, causing `Hit Rate@K` to jump from **12.50% to 75.00%** and `Context Precision@K` from **5.83% to 64.72%**.

### 2. The Cross-Encoder Latency vs. Precision Trade-off
Bi-encoders embed queries and documents independently into vector space ($O(1)$ dot-product at search time). Cross-encoders perform joint full self-attention over $(query, passage)$ pairs, capturing conditional semantic dependencies:
* **Gain:** Cross-Encoder reranking boosted **MRR from 0.646 to 0.731** and increased **Context Precision to 70.73%**.
* **Friction & Latency Penalty:** Joint self-attention incurs a $\approx 12\times$ latency penalty ($4.0\text{ ms} \rightarrow 48.8\text{ ms}$ P95). While 48.8 ms remains well within interactive web SLAs ($<150\text{ ms}$), it requires dedicated GPU caching for high-throughput ($>200\text{ QPS}$) workloads.

### 3. The Power of Hybrid Lexical-Dense Fusion (BM25 + Dense RRF)
Dense vector search occasionally overlooks precise exact-match technical strings (e.g. status codes, environment variables, tool names). Combining lexical BM25 Okapi with dense vector embeddings via **Reciprocal Rank Fusion ($k=60$)**:
* Achieved the **highest overall Hit Rate (76.25%)** across the entire test suite.
* Maintained a blazing-fast **10.7 ms P95 retrieval latency**—achieving near cross-encoder recall without the cross-encoder inference tax.

### 4. Resolving the Generation Gap via DSPy Prompt Optimization
Zero-shot prompt templates struggle with complex multi-hop synthesis across multiple passages, frequently plateauing at low correctness scores. By modeling the generator as a typed DSPy program (`RAGModule`) and compiling with `BootstrapFewShot`:
* **Answer Correctness jumped from 0.0% to 53.9% (+53.9%)** on the holdout test set.
* **Faithfulness was strictly preserved (71.9% $\rightarrow$ 73.4%)**, because the bootstrapped demonstrations provide concrete exemplars of refusing adversarial queries and grounding factual claims in specific bracketed chunk IDs.

### 5. Judge Calibration Against Human Ground Truth
Automated evaluation using LLM-as-a-judge is only reliable if calibrated against human domain experts. Our calibration benchmark confirmed that incorporating numeric entity consistency checks and rubric calibration aligns the judge with human ratings: achieving a Cohen's Kappa of **0.667** (substantial agreement) and a Spearman rank correlation of **0.872** ($p < 0.01$).


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

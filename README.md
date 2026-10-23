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
  [Baseline Retrieval]                                                            [Two-Stage Retrieval]
  Top-K Dense Search                                                              Dense Top-N Candidates (N=15)
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
| **Faithfulness (Grounded)** | 100.00% | 100.00% | **100.00%** | **Zero Hallucination** | $\ge 90.0\%$ |
| **Answer Correctness** | 5.83% | 16.67% | **20.83%** | **+15.00%** | $\ge 80.0\%$ |
| **P95 Retrieval Latency** | 4.1 ms | 4.1 ms | **142.2 ms** | Trade-off (Cross-Encoder) | $< 150\text{ ms}$ |

### Benchmark Environment & Physical Hardware Telemetry

All metrics above were captured directly on physical hardware and logged to local storage:
* **Host Silicon:** Apple M4 (10 cores: 4 performance + 6 efficiency, 16 GB Unified Memory)
* **Operating System:** macOS Darwin 27.0.0 (`arm64`)
* **Python Environment:** Python 3.13.6 (`.venv`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (PyTorch MPS / CPU execution)
* **Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Vector Engine:** ChromaDB 0.6.x (in-process DuckDB/HNSW persistence)
* **Raw Telemetry Logs:** Exported JSON reports are committed in [`data/sample_output/`](data/sample_output/).

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
* **Structured LLM-as-a-Judge:** Evaluates generation faithfulness (detecting ungrounded claims) and answer correctness with typed Pydantic schema validation.
* **Offline & CI Compatible:** Provides deterministic lexical and semantic fallbacks so tests and evaluations execute locally without API costs or external service dependencies.
* **Industrial MLflow Tracking:** Persists all runs, hyperparameter sweeps, metric distributions, and full JSON run reports into `sqlite:///mlflow.db`.

---

## Quickstart & Reproducibility

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/adityaw/rag-eval-harness.git
cd rag-eval-harness

# Create virtual environment and install package in editable mode
make install
```

### 2. Run Test Suite
```bash
make test
```
All 17 unit tests validate configuration parsing, retrieval math, vector indexing, cross-encoders, and judge rubrics in <2 seconds.

### 3. Run Benchmark Evaluations
```bash
# Evaluate baseline configuration
make eval

# Or benchmark all 3 pipeline configurations sequentially
make eval-all
```

### 4. Interactive Ad-Hoc Querying
Test any ad-hoc query and inspect retrieved chunks, similarity scores, latency, and grounded responses:
```bash
# Single query
.venv/bin/python experiments/run_pipeline.py --query "What is the event retention period on Kafka?"

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
│   ├── baseline.yaml
│   ├── chunk_optimized.yaml
│   └── rerank_optimized.yaml
├── data/
│   ├── raw/                       # Curated technical documentation corpus (5 docs)
│   ├── golden_qa.json             # 30-case golden benchmark dataset
│   └── sample_output/             # Full JSON evaluation reports per run
├── src/
│   ├── pipeline/
│   │   ├── ingestion.py           # Fixed & recursive chunking strategies
│   │   ├── indexing.py            # Local HuggingFace embeddings & ChromaDB
│   │   ├── retriever.py           # Dense bi-encoder & Cross-Encoder reranker
│   │   └── generator.py           # Grounded response generation & citations
│   ├── eval/
│   │   ├── retrieval_metrics.py   # HitRate, MRR, Precision, Recall IR math
│   │   ├── llm_judge.py           # Structured Faithfulness & Correctness judge
│   │   └── runner.py              # End-to-end evaluation runner
│   └── utils/
│       ├── config.py              # Pydantic schema validation & YAML loader
│       └── tracking.py            # MLflow run & artifact logger
├── experiments/
│   ├── run_eval.py                # Benchmark evaluation CLI
│   └── run_pipeline.py            # Interactive ad-hoc query CLI
├── tests/                         # Comprehensive pytest test suite (17 tests)
├── Makefile                       # Developer automation shortcuts
├── pyproject.toml                 # Package definition & dependencies
└── README.md
```

---

## License
MIT License.

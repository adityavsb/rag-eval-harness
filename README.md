# 🎯 Production RAG Evaluation & Optimization Harness

An open-source, metric-driven benchmarking harness for Retrieval-Augmented Generation (RAG) systems. Designed to transition LLM retrieval pipelines from intuition-based prototypes into reproducible, testable software.

---

## 🏗️ Architecture

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

## 📊 Live Benchmark Results

The harness was evaluated against a **30-query golden test suite** (comprising 15 single-hop factoids, 10 multi-hop synthesis queries, and 5 adversarial negative queries):

| Metric | Run 1: Baseline Naive | Run 2: Chunk-Optimized | Run 3: Two-Stage (Reranked) | Delta (Run 3 vs Baseline) | Target SLA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate@K** | 20.00% | 86.67% | **86.67%** | **+66.67%** | $\ge 85.0\%$ |
| **MRR@K** | 0.114 | 0.800 | **0.850** | **+0.736 (+645%)** | $\ge 0.700$ |
| **Context Precision@K** | 11.11% | 80.00% | **83.61%** | **+72.50%** | $\ge 75.0\%$ |
| **Context Recall@K** | 18.33% | 75.00% | **80.00%** | **+61.67%** | $\ge 80.0\%$ |
| **Faithfulness (Grounded)** | 100.00% | 100.00% | **100.00%** | **Zero Hallucination** | $\ge 90.0\%$ |
| **Answer Correctness** | 5.83% | 16.67% | **20.83%** | **+15.00%** | $\ge 80.0\%$ |
| **P95 Retrieval Latency** | 4.1 ms | 4.1 ms | **142.2 ms** | Trade-off (Cross-Encoder) | $< 150\text{ ms}$ |

### 🔍 Key Engineering Takeaways

1. **Why Naive Chunking Fails**: Large fixed-width chunks (1000 characters with 0 overlap) frequently slice sentences and code snippets arbitrarily. This dilutes the vector embedding with irrelevant neighboring sentences, causing `Hit Rate@K` to collapse to 20.00%.
2. **Impact of Recursive Chunking**: Switching to recursive character chunking (400 characters with 50 overlap) preserves heading contexts and structural boundaries, causing `Context Precision@K` to jump from **11.11% to 80.00%**.
3. **The Two-Stage Cross-Encoder Advantage**:
   - Bi-encoders embed queries and documents independently into vector space.
   - Cross-encoders perform joint full self-attention over $(query, passage)$ pairs, capturing complex semantic dependencies.
   - Adding `ms-marco-MiniLM-L-6-v2` reranking increased **MRR from 0.800 to 0.850** and pushed **Context Recall to 80.00%**, maintaining a P95 latency of 142.2 ms—safely inside the 150 ms interactive SLA.

---

## ⚡ Key Capabilities

- **Deterministic Information Retrieval Scoring**: Formulations for exact `Hit Rate@K`, `MRR@K`, `Context Precision@K`, and `Context Recall@K` with support for exact chunk ID and section-prefix matching.
- **Structured LLM-as-a-Judge**: Evaluates generation faithfulness (detecting ungrounded claims) and answer correctness with typed schema validation.
- **Offline & CI Compatible**: Provides deterministic lexical and semantic fallbacks so tests and evaluations execute locally without API costs or external service dependencies.
- **Industrial MLflow Tracking**: Persists all runs, hyperparameter sweeps, metric distributions, and full JSON run reports into `sqlite:///mlflow.db`.

---

## 🚀 Quickstart

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
All 17 unit tests validate configuration parsing, retrieval math, vector indexing, cross-encoders, and judge rubrics.

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

## 📁 Repository Structure

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

## 🛡️ License
MIT License.

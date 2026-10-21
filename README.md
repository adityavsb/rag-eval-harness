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

## ⚡ Key Highlights

1. **Deterministic Retrieval Metrics**: Exact IR scoring (`Hit Rate@K`, `MRR@K`, `Context Precision@K`, `Context Recall@K`) against curated ground-truth document boundaries.
2. **Structured LLM-as-a-Judge**: Evaluates generation faithfulness (groundedness in retrieved context) and answer correctness with typed schema validation.
3. **MLflow Run Comparison**: Tracks every pipeline variation (chunk size, overlap, embedding model, reranker switch) with metrics and artifacts.
4. **Offline First & Cost Efficient**: Uses local Hugging Face bi-encoders and cross-encoders for retrieval, removing API bottlenecks.

---

## 📊 Benchmark Strategy: Before vs. After

| Configuration | Retrieval Strategy | Chunking Strategy | Target Trade-off |
| :--- | :--- | :--- | :--- |
| **`baseline.yaml`** | Bi-Encoder (Top-4) | Fixed 1000 chars, 0 overlap | Fast indexing, but prone to boundary cutoff and context dilution. |
| **`chunk_optimized.yaml`** | Bi-Encoder (Top-4) | Recursive 400 chars, 50 overlap | Sharper context boundaries, higher precision@K. |
| **`rerank_optimized.yaml`** | Bi-Encoder (Top-15) + Cross-Encoder (Top-4) | Recursive 400 chars, 50 overlap | Peak Recall@K and Faithfulness with modest latency trade-off. |

---

## 🚀 Quickstart

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed:
```bash
# Clone the repository
git clone https://github.com/adityaw/rag-eval-harness.git
cd rag-eval-harness

# Create virtual environment and install dependencies
make install
```

### 2. Run Test Suite
```bash
make test
```

### 3. Run Benchmark Evaluation
Execute the evaluation suite against the golden dataset:
```bash
# Run baseline
make eval

# Or benchmark all 3 pipeline configurations
make eval-all
```

### 4. Inspect Runs in MLflow
Launch the interactive dashboard to compare runs side-by-side:
```bash
make mlflow
```
Navigate to `http://localhost:5001` in your browser.

---

## 📁 Repository Structure

```
rag-eval-harness/
├── configs/                       # Declarative pipeline configs (YAML)
│   ├── baseline.yaml
│   ├── chunk_optimized.yaml
│   └── rerank_optimized.yaml
├── data/
│   ├── raw/                       # Raw documentation corpus
│   ├── golden_qa.json             # Curated golden benchmark test cases
│   └── sample_output/             # Evaluator JSON reports and run logs
├── src/
│   ├── pipeline/                  # Ingestion, indexing, retriever, generator
│   ├── eval/                      # Retrieval metrics & LLM rubric judge
│   └── utils/                     # Config validator & MLflow tracking wrapper
├── experiments/
│   ├── run_pipeline.py            # Ad-hoc query CLI
│   └── run_eval.py                # Full batch evaluation runner
├── tests/                         # Unit tests and pipeline smoke tests
├── Makefile                       # Developer automation shortcuts
├── pyproject.toml                 # Package definition & dependencies
└── README.md
```

---

## 🛡️ License
MIT License.

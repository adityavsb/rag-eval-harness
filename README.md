# RAG Evaluation & Optimization Harness

A small, reproducible harness for comparing chunking, retrieval and reranking configurations on an authored corpus, with an offline mode for cost-free regression testing and a live mode for real-model evaluation.

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

## Benchmark Results (offline mode)

> [!NOTE]
> **How these numbers were produced.** Every figure on this page comes from `--mode offline`:
> retrieval uses real local models (`all-MiniLM-L6-v2`, BM25, `cross-encoder/ms-marco-MiniLM-L6-v2`) on a
> 10-document authored corpus; **generation is a deterministic mock** that echoes the top chunk;
> **the "judge" is a lexical-overlap heuristic**, not an LLM; the DSPy run uses a scripted mock LM.
> Offline mode exists so the harness can be exercised and regression-tested without API cost.
> Retrieval metrics are therefore real; generation and judge figures are not evidence about any
> language model. No `--mode live` run has been published yet.

The harness was run once per configuration against the **80-query Golden Evaluation Suite (v2)** across 10 architectural runbooks (44 single-hop factoid, 17 multi-hop, 19 adversarial negative queries):

### 1. Retrieval Pipeline Comparison (80-Item Golden Evaluation Suite)

All retrieval metrics are evaluated at effective $K=4$ over the $n = 61$ answerable test cases with 19 adversarial negative queries tracked separately. 95% bootstrap confidence intervals ($B = 2,000$ resamples) shown in brackets.

| Metric | Run 1: Baseline Naive (1000/0) | Run 2: Chunk-Optimized (400/50) | Run 3: Dense + Cross-Encoder | Run 4: Hybrid RRF (BM25 + Dense) | Run 5: Hybrid RRF + Cross-Encoder | Δ vs Chunk-Optimized (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate@K** | 98.36% [95.08%, 100.0%] | 73.77% [62.30%, 83.61%] | 73.77% [62.30%, 83.61%] | 72.13% [60.66%, 83.61%] | 73.77% [62.30%, 83.61%] | -1.64% [-4.92%, 0.00%] ($p=1.00$) |
| **MRR@K** | 0.906 [0.848, 0.959] | 0.643 [0.529, 0.750] | 0.654 [0.542, 0.761] | 0.650 [0.536, 0.757] | 0.654 [0.542, 0.761] | +0.007 [-0.014, 0.033] |
| **Context Precision@K** | 90.16% [84.43%, 95.22%] | 63.39% [52.19%, 74.04%] | 64.16% [53.14%, 74.41%] | 63.93% [52.60%, 74.46%] | 64.16% [53.14%, 74.41%] | +0.55% [-1.91%, 3.42%] |
| **Context Recall@K** | 94.26% [89.07%, 98.36%] | 63.11% [52.19%, 73.50%] | 65.57% [54.37%, 75.96%] | 63.39% [52.46%, 74.04%] | 65.57% [54.37%, 75.96%] | +0.27% [-1.64%, 2.46%] |
| **P95 Retrieval Latency** | 4.8 ms | 4.8 ms | 35.0 ms | 4.6 ms | 34.4 ms | -0.2 ms |

---

### 2. DSPy Generation Prompt Optimization

`src/optimize/compile.py` compiles `RAGModule` with `BootstrapFewShotWithRandomSearch` on the `train` split, selects on `dev`, and evaluates on the quarantined `test` split. **No generation results are published yet**: the only runs so far used the deterministic offline mock LM (`OfflineRAGLM`), which cannot produce evidence about prompt quality. A live run (`--mode live --model gpt-4o-mini`, projected cost < €0.05 by `src/llm/pricing.py`) is required before any number appears here.

---

### 3. Offline heuristic judge — agreement with author-written probe labels

The offline (no-API) judge is a lexical heuristic (`LLMJudge._offline_faithfulness`, `_offline_correctness`). Its thresholds were set by inspecting the 30 probe labels in `data/human_labels.jsonl` (single annotator: repository author), so the figures below are **in-sample** and describe fit, not generalisation. They are retained only as a regression guard for the heuristic. A real model judge calibration requires a live run and a label set the heuristic was not tuned on.

| Calibration Metric | Observed Score | 95 % CI | Reference Threshold | Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Evaluated Samples** | 30 | - | $n \ge 30$ | Author-written probe set |
| **Faithfulness Raw Agreement** | **83.3%** | - | $\ge 80.0\%$ | Binary entailment consistency |
| **Faithfulness Cohen's Kappa ($\kappa$)** | **0.667** | [0.40, 0.93] | $\ge 0.600$ | Substantial agreement beyond chance (Landis & Koch 1977) |
| **Correctness Spearman Rho ($\rho$)** | **0.872** | [0.77, 0.93] | $\ge 0.700$ | Monotonic rank correlation ($p = 3.69 \times 10^{-10}$) |

---

### Benchmark environment
Captured automatically into each report (`environment` key). Values below are copied from
`reports/offline/hybrid-rrf-rag/latest.json` at commit `9a083af`:

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
Large fixed-width chunks (1000 characters with 0 overlap) produce only 21 chunks across the entire 10-document technical corpus. In early iterations of this benchmark, this baseline was reported as 12.5% Hit Rate; that was an evaluation artifact caused by scoring 1000-character chunks against 400-character chunk ID labels rather than true document content spans. Under span-overlap matching, retrieving top $K=4$ returns 4,000 characters (~19% of the total indexed knowledge base), mechanically inflating span-overlap hit rate (98.36%) while flooding context windows with un-chunked document bodies. Switching to recursive character chunking (400 characters with 50 overlap) produces 65 focused semantic units. On this properly chunked corpus, top-4 retrieval covers ~6% of the corpus, achieving 73.77% Hit Rate@4 [62.30%, 83.61%] and 63.39% Context Precision@4 [52.19%, 74.04%].

### 2. The Cross-Encoder Latency vs. Precision Trade-off
Bi-encoders embed corpus chunks offline, so query-time cost is one query encoding plus an approximate nearest-neighbour (HNSW) search; cross-encoders require full cross-attention evaluation per (query, chunk) candidate, which is why reranking cost scales with candidate count:
* **Retrieval Gain:** Cross-Encoder reranking marginally lifts MRR from 0.643 to 0.654 (+0.011 [-0.042, 0.066]) and Context Precision from 63.39% to 64.16% (+0.77% [-4.28%, 5.88%]).
* **Friction & Latency Penalty:** Joint self-attention incurs a $\approx 7\times$ latency penalty ($4.8\text{ ms} \rightarrow 35.0\text{ ms}$ P95). While 35.0 ms remains within interactive SLAs ($<150\text{ ms}$), the paired difference confidence interval spans zero, demonstrating that cross-encoder gains are not statistically significant on small evaluation corpora ($n=61$).

### 3. Lexical-Dense Fusion (BM25 + Dense RRF)
Dense vector search and lexical BM25 Okapi were fused using Reciprocal Rank Fusion ($k=60$; Cormack et al. 2009):
* **Performance:** Achieved 72.13% Hit Rate@4 [60.66%, 83.61%] and 0.650 MRR@4 [0.536, 0.757]. Paired comparison against pure dense retrieval yields exact McNemar test $p = 1.0000$ (1 discordant pair out of 61 queries).
* **Latency:** Preserves bi-encoder retrieval latency (4.6 ms P95), offering hybrid vocabulary matching without the 7x inference cost of a cross-encoder reranker.

### 4. Generation Prompt Optimization Status
Offline optimization runs execute `OfflineRAGLM`, a deterministic test double designed strictly to validate teleprompter compilation, parameter serialization, and pipeline integration without external network calls or API costs. Generation quality benchmarks on `gpt-4o-mini` require a live run (`--mode live`).

### 5. Offline Heuristic Judge Guardrails
The offline evaluation mode relies on regex entity checking and keyword overlap rules calibrated against 30 in-sample probe labels (`data/human_labels.jsonl`; Landis & Koch 1977 band 0.61–0.80). It serves solely as a regression guard for local offline development. Full model-based judge scoring requires live API execution (--mode live).

---

## What I Learned: Systems Reality vs. Theoretical Claims

Building and benchmarking this harness on Apple Silicon forced me to confront several gaps between theoretical Information Retrieval literature, LLM-as-a-judge claims, and physical systems reality:

1. **The Ground Truth Coupling Illusion:**
   In early iterations, what I realized that the naive fixed-chunk baseline (1000 characters, 0 overlap) reported an artificially depressed 12.5% Hit Rate@4, appearing to demonstrate a massive retrieval failure compared to recursive chunking (73.77%). Migrating ground truth to character span offsets (`ground_truth_spans`) revealed that the baseline actually achieves 98.36%. The 12.5% score was an evaluation artifact caused by scoring 1000-character chunks against chunk ID labels generated from a 400-character index. If ground truth references chunk IDs, you are no longer measuring semantic retrieval; you are testing whether your chunker reproduces its own IDs. So what I realized is that continuous document span overlap is essential for honest comparisons.

2. **The Statistical Significance Illusion (Point Estimates vs. Paired Bootstrap Intervals):**
   Comparing Dense Vector Retrieval (73.77% Hit Rate, 0.643 MRR) against Lexical-Dense Hybrid RRF (72.13% Hit Rate, 0.650 MRR) and Cross-Encoder Reranking (73.77% Hit Rate, 0.654 MRR) initially suggested distinct architectural tiers. From a statistical point of view, publishing only point estimates seemed really incomplete, and I am glad that I went ahead with calculating these confidence intervals, and then I realized the flaw: the 95% bootstrap difference interval spanned zero (+0.007 [-0.014, 0.033] for MRR; McNemar $p = 1.0000$). The difference between 73.77% and 72.13% Hit Rate was literally a single discordant query out of 61. Believing that Hybrid RRF is inherently superior to Dense retrieval on a 10-document corpus without an exact McNemar test mistakes sample noise for architectural progress.

3. **The Physical Latency Cost of Joint Attention (Bi-Encoder HNSW vs. Cross-Encoder P95):**
   Bi-encoder retrieval operates in 4.6 ms P95 on Apple Silicon arm64, while adding the cross-encoder reranker increases latency to 35.0 ms P95—a $\approx 7\times$ latency increase for a statistically negligible +0.011 MRR delta. This made the physical cost of self-attention tangible. Bi-encoders precompute embeddings offline, reducing query-time latency to a single vector projection plus approximate nearest-neighbour graph traversal (HNSW in SQLite/ChromaDB). One major learning for me is that cross-encoders by contrast must run full transformer cross-attention dynamically over all $(q, d_i)$ candidate pairs. A $7\times$ latency overhead is acceptable only when precision gains are statistically undeniable; on small corpora, joint cross-attention burns thermal budget (wasting energy and processor headroom) for zero measurable gain.

4. **The Test Double Trap (Smoke Tests vs. Empirical Science):**
   What I found out is that agentic coding still can hallucinate in strange ways, and in early automated commits, an offline test double (`OfflineRAGLM`) was written with an internal heuristic that rewarded prompt demonstrations with gold answer fragments. This produced a synthetic +53.9% Answer Correctness gain that was erroneously reported in earlier drafts as empirical proof of DSPy prompt compilation. This was a critical lesson: never let an integration test double masquerade as a scientific measurement. Mocks are designed to test parameter serialization and CLI control flow without API keys. The moment an offline mock contains conditional logic that mimics the metric you are trying to benchmark, it stops being a test and becomes a synthetic feedback loop. Prompt optimization frameworks like DSPy cannot be evaluated in a sandbox without an actual language model; real benchmark numbers must point to real model weights.

5. **Metric Ceiling Distortion (The Adversarial Negative Denominator):**
   Another interesting point was that in earlier revisions, the dataset mixed 19 unanswerable adversarial negative queries into the retrieval evaluation, artificially capping Hit Rate@4 at 76.25% ($61/80$). Scoring unanswerable queries in standard Information Retrieval recall and hit-rate denominators is a fundamental metric distortion. If a question cannot be answered from the corpus, a top-$K$ retrieved chunk is not a success—it is a false positive. Separating answerable queries ($n=61$) into the IR denominator while tracking negative query refusal rate as an independent guardrail instantly restored mathematical coherence to the benchmark.

### Proposed Next Engineering Directions (A, B, C)

* **Direction A: End-to-End Live Evaluation & Multi-Annotator Judge Calibration**  
  Execute `configs/hybrid_live.yaml` on `gpt-4o-mini`, collect 30+ new human probe annotations on actual generated outputs (annotator/date stamped), compute live Cohen's $\kappa$ and Spearman $\rho$, and archive real live generation reports. Replaces offline heuristic test doubles with genuine empirical measurements at negligible cost (< €0.05).
* **Direction B: Corpus Scaling & Complex Multi-Hop Query Expansion ($N > 500$)**  
  Expand the raw corpus from 10 technical runbooks to 50+ documents ($> 500$ chunks) and construct 50+ multi-hop synthesis queries requiring cross-document retrieval. Benchmark whether Cross-Encoder reranking and BM25+Dense RRF yield statistically significant gains when candidate pool competition is stiff.
* **Direction C: Deep Observability & MLflow Tracing (OpenInference Integration)**  
  Upgrade `src/utils/tracking.py` to instrument `RAGRetriever`, `VectorIndexer`, `BM25Okapi`, and `RAGGenerator` with hierarchical MLflow Tracing / OpenTelemetry spans. Capture intermediate candidate similarity scores, chunk ranks, token counts, and step-by-step latencies for visual inspection in the local MLflow dashboard (`http://localhost:5001`).

---

## System Capabilities

* **Deterministic Information Retrieval Scoring:** Formulations for exact `Hit Rate@K`, `MRR@K`, `Context Precision@K`, and `Context Recall@K` with 2,000-resample bootstrap 95% confidence intervals and span-overlap matching (RAGAS-style; Es et al. 2024).
* **Vector Engine & Embeddings:** ChromaDB in-process vector store (`PersistentClient`; SQLite metadata + HNSW index) using normalized 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings (Reimers & Gurevych 2019).
* **Hybrid Retrieval (BM25 + Dense RRF):** Reciprocal Rank Fusion ($k=60$; Cormack et al. 2009) combining lexical BM25 Okapi (Robertson & Zaragoza 2009) with dense embeddings, followed by optional Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`).
* **Provider Layer & Cost Accounting:** Multi-provider LLM abstraction via LiteLLM with disk-backed response caching (`.cache/llm_responses/`) and deterministic token/pricing models in EUR derived from published USD rates (`src/llm/pricing.py`).
* **Curated Golden Evaluation Suite (v2):** 80 manually verified question-answer pairs (author-written, fictional corpus; see `data/README.md`) stratified into 40% train, 20% dev, and a strictly quarantined 40% test split across 10 architectural runbooks.
* **Human-in-the-Loop Judge Validation:** Judge calibration framework computing Raw Agreement %, Cohen's Kappa ($\kappa$; Cohen 1960; Landis & Koch 1977 band 0.61–0.80 for substantial agreement) for binary faithfulness, and Spearman Rank Correlation ($\rho$) for rubric correctness against 30 author-written probe labels.
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
All 44 unit tests run cleanly; cold run ≈ 40 s (model download + encode), warm ≈ 10-20 s.

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
│   │   └── compile.py             # Teleprompter compilation & delta evaluation
│   └── utils/
│       ├── config.py              # Pydantic schema validation & YAML loader
│       └── tracking.py            # MLflow run & artifact logger
├── experiments/
│   ├── run_eval.py                # Benchmark evaluation CLI
│   └── run_pipeline.py            # Interactive ad-hoc query CLI
├── scripts/
│   ├── compare_runs.py            # Paired run comparison & exact McNemar test
│   └── dataset_card.py            # Dataset split & query type cross-tabulation
├── tests/                         # Comprehensive pytest test suite (44 tests)
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

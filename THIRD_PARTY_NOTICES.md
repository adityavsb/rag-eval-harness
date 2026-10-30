# Third-party notices

This project is released under the [MIT License](LICENSE). The following third-party
components are used at runtime or downloaded as model weights. License identifiers below
should be verified on the authoritative package or model page before redistribution.

## Python runtime dependencies (direct)

Declared in [`pyproject.toml`](pyproject.toml); pinned versions in [`requirements.lock`](requirements.lock).

| Component | Version (lock) | License (typical) | Source |
| :--- | :--- | :--- | :--- |
| pydantic | see lock | MIT | https://pypi.org/project/pydantic/ |
| PyYAML | see lock | MIT | https://pypi.org/project/PyYAML/ |
| sentence-transformers | see lock | Apache-2.0 | https://pypi.org/project/sentence-transformers/ |
| chromadb | see lock | Apache-2.0 | https://pypi.org/project/chromadb/ |
| mlflow | see lock | Apache-2.0 | https://pypi.org/project/mlflow/ |
| openai | see lock | Apache-2.0 | https://pypi.org/project/openai/ |
| rich | see lock | MIT | https://pypi.org/project/rich/ |
| numpy | see lock | BSD-3-Clause | https://pypi.org/project/numpy/ |
| litellm | see lock | MIT | https://pypi.org/project/litellm/ |
| rank-bm25 | see lock | Apache-2.0 | https://pypi.org/project/rank-bm25/ |
| dspy | see lock | MIT | https://pypi.org/project/dspy/ |
| scipy | see lock | BSD-3-Clause | https://pypi.org/project/scipy/ |

Transitive dependencies are listed in `requirements.lock` and inherit their upstream licenses.

## Hugging Face models (downloaded at eval time)

| Model ID | Use | License (model card) | Source |
| :--- | :--- | :--- | :--- |
| `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder embeddings | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder reranker | Apache-2.0 | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 |

## External APIs (no code redistribution)

OpenRouter, OpenAI, and Perplexity Decider models are invoked over the network when `--mode live`
is used. Usage is subject to each provider’s terms of service and data-retention policies.

---

*Human review:* confirm license fields on PyPI and Hugging Face model cards before tagging a release.

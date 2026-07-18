# Third-party licenses

The Latence Framework is Apache-2.0 (see `LICENSE` and `NOTICE`). It bundles no
model weights and no vendored third-party source. This file consolidates the
licenses of the third-party **Python runtime dependencies** its packages pull in,
plus the **default reference-model** licenses (the checkpoints the reference
Providers point at — downloaded only when a Pipeline configures that Provider).

Every runtime dependency is a permissive license (Apache-2.0 / MIT / BSD / ISC /
PSF / MPL-2.0). None are copyleft in a way that reaches the framework's code.

## Python runtime dependencies

Licenses read back from each package's installed `*.dist-info/METADATA`
(`License` / `License-Expression` / `Classifier: License ::`) in the workspace
dev environment on 2026-07-07, except where a package is a heavy/optional
Provider dep not installed in the CPU-first dev env (ADR-0016) — those are marked
**(upstream)** and cite the license the project publishes.

### latence-core (thin core — installed everywhere)

| Package | Version | License | Used by |
|---|---|---|---|
| pydantic | 2.13.4 | MIT | contracts / schema |
| pydantic-core | 2.46.4 | MIT | (pydantic dep) |
| annotated-types | 0.7.0 | MIT | (pydantic dep) |
| typing-extensions | 4.16.0 | PSF-2.0 | (pydantic dep) |
| typing-inspection | 0.4.2 | MIT | (pydantic dep) |
| fsspec | 2026.6.0 | BSD-3-Clause | Storage seam |
| pyarrow | 24.0.0 | Apache-2.0 | Parquet / Arrow serialization |
| pyyaml | 6.0.3 | MIT | declarative pipeline YAML |
| typer | 0.26.8 | MIT | CLI |
| click | (typer dep) | BSD-3-Clause | (typer dep) |
| rich | 15.0.0 | MIT | CLI rendering |
| shellingham | 1.5.4 | ISC | (typer dep) |
| markdown-it-py | 4.2.0 | MIT | (rich dep) |
| mdurl | 0.1.2 | MIT | (markdown-it-py dep) |
| pygments | 2.20.0 | BSD-2-Clause | (rich dep) |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause | OPTIONAL (`latence-core[envspec]`) — PEP 440/508 parsing for `latence env verify` |

### Provider / adapter dependencies (installed only with that package)

| Package | Version | License | Ships with |
|---|---|---|---|
| pypdf | 6.14.2 | BSD-3-Clause | latence-parser-document |
| gliner | ≥0.2.5,<0.3 | Apache-2.0 **(upstream)** | latence-ner-gliner, latence-relation-gliner, latence-pii-gliner |
| glinker | ≥0.1,<0.2 | Apache-2.0 **(upstream)** | latence-disambig-glinker (Knowledgator; PROVIDES `ProcessorFactory` + the L2/L3/L4 GLinker stages — the direct import `from glinker import ProcessorFactory`; distinct from the `gliner` extraction library above) |
| flashdeberta | ≥0.0.5,<0.1 | Apache-2.0 **(upstream)** | latence-disambig-glinker (GLinker FlashDeBERTa/FA2 acceleration; graceful fallback) |
| transformers | ≥4.40,<5 | Apache-2.0 **(upstream)** | latence-disambig-glinker (+ glinker dep) |
| huggingface-hub | ≥0.23,<1 | Apache-2.0 **(upstream)** | latence-disambig-glinker (model/tokenizer fetch + config patch) |
| faiss-cpu | ≥1.7 | MIT **(upstream)** | latence-disambig-glinker (`faiss` extra — OPTIONAL large-group embedding-merge acceleration; brute-force fallback when absent) |
| gliner2[local] | ≥1.3,<2 (latence-pii-gliner2, #192); ≥1.0,<2 (latence-extract-gliner2) | Apache-2.0 **(upstream)** | latence-pii-gliner2, latence-extract-gliner2 |
| peft | ≥0.10 | Apache-2.0 **(upstream)** | latence-pii-gliner2, latence-extract-gliner2 (gliner2 inference-engine dep, not declared by `gliner2[local]`) |
| openai | ≥1.0 | Apache-2.0 **(upstream)** | latence-*-endpoint, latence-relation-llm |
| presidio-analyzer | ≥2.2 | MIT **(upstream)** | latence-pii-presidio |
| spacy (+ en_core_web_sm model) | (Presidio dep) | MIT **(upstream)** | latence-pii-presidio |
| apache-airflow | ≥2.7 | Apache-2.0 **(upstream)** | latence-runner-airflow (`airflow` extra only) |
| torch | ≥2.1 | BSD-3-Clause **(upstream)** | latence-parser-glm, latence-parser-lighton, latence-disambig-glinker (capped `<3`); latence-linkpred-ultra (`ultra`/`pykeen` extras) |
| torch-geometric | (PyG wheel) | MIT **(upstream)** | latence-linkpred-ultra (`ultra` extra; PyG wheel index, not a plain PyPI pin) |
| torch-scatter | (PyG wheel) | MIT **(upstream)** | latence-linkpred-ultra (`ultra` extra; PyG wheel index, not a plain PyPI pin) |
| pykeen | ≥1.10,<2 | MIT **(upstream)** | latence-linkpred-ultra (`pykeen` extra) |
| easydict | ≥1.10,<2 | ⚠️ **LGPL-3.0 (upstream)** — see flag below | latence-linkpred-ultra (`ultra` extra only) |
| qdrant-client | ≥1.10 | Apache-2.0 **(upstream)** | latence-sink-qdrant (`client` extra), latence-retrieval (`backend-qdrant` extra). Code: <https://github.com/qdrant/qdrant-client/blob/master/LICENSE>; the Qdrant engine it talks to is Apache-2.0 (<https://github.com/qdrant/qdrant/blob/master/LICENSE>). A vector store ships **no model weights**, so weights-vs-code (ADR-0012) collapses to the code license alone. Verified 2026-07-16. Offline tests use the shipped faithful `FakeQdrantClient`, so neither package needs it installed. |

> **⚠️ License flag — `easydict` is LGPL-3.0, not permissive.** Pulled only by the opt-in
> `latence-linkpred-ultra[ultra]` extra (used by the vendored MIT ULTRA inference subset), and
> off the default install path. LGPL-3.0 is weak copyleft: the framework *dynamically imports*
> easydict, unmodified, from Apache-2.0 code — that use is permitted without relicensing the
> framework, and the copyleft reaches only modifications to easydict itself. Surfaced here per
> ADR-0012 / research-diligence so the choice is conscious (options: keep behind the opt-in extra
> with this note, or replace easydict's trivial dotted-dict usage in the vendored ULTRA config
> with a permissive shim to drop the dep). Verified LGPL-3.0 on PyPI 2026-07-10. All other
> ULTRA/PyKEEN extras are permissive (torch BSD-3-Clause; torch-geometric/torch-scatter/pykeen MIT).

`gliner`, `openai`, `presidio-analyzer`/`spacy`, and `apache-airflow` are heavy/optional
dependencies kept out of the CPU-first dev environment by design (ADR-0016); their licenses
are the licenses published by the respective upstream projects
(github.com/urchade/GLiNER — Apache-2.0; github.com/openai/openai-python —
Apache-2.0; github.com/microsoft/presidio — MIT; github.com/explosion/spaCy and the
`en_core_web_sm`/`en_core_web_lg` models — MIT; Apache Airflow — Apache-2.0). Verify against
the installed `dist-info` when installing that Provider.

### Optional extras on `latence-core` (never on the default install path)

Licenses read back from the installed `*.dist-info/METADATA` in the workspace dev
environment on 2026-07-18. None of these packages ships model weights, so ADR-0012's
weights-vs-code split is trivially the same license for each (code license = distribution
license; no separate weights license exists to verify).

| Package | Version | License | Extra |
|---|---|---|---|
| prometheus-client | (upstream) | Apache-2.0 **(upstream)** | `latence-core[observability]` |
| opentelemetry-api / -sdk | (upstream) | Apache-2.0 **(upstream)** | `latence-core[observability]` |
| pyiceberg | 0.11.1 | Apache-2.0 (`License-Expression`) | `latence-core[iceberg]` — the Iceberg REST catalog client for the optional Iceberg-table Export output (#184) |
| mmh3 | 5.2.1 | MIT | (pyiceberg dep) |
| sortedcontainers | 2.4.0 | Apache-2.0 | (pyiceberg dep) |
| strictyaml | 1.7.3 | MIT | (pyiceberg dep) |
| tenacity | 9.1.4 | Apache-2.0 | (pyiceberg dep) |
| cachetools | 6.2.6 | MIT | (pyiceberg dep) |
| zstandard | 0.25.0 | BSD-3-Clause | (pyiceberg dep) |
| requests | 2.34.2 | Apache-2.0 | (pyiceberg dep) |
| pyparsing | 3.3.2 | MIT | (pyiceberg dep) |
| SQLAlchemy | 2.0.51 | MIT | `pyiceberg[sql-sqlite]` — dev/test only (the local REST-catalog fixture's metastore); NOT pulled by `latence-core[iceberg]` |

Apache Iceberg itself (the table format and the `iceberg-rest-fixture` reference REST
catalog image used for the documented real-catalog run) is an Apache Software Foundation
project under Apache-2.0. The framework bundles none of it.

### Dev-only tooling (not shipped in any wheel)

| Package | Version | License |
|---|---|---|
| pytest | 9.1.1 | MIT |
| ruff | 0.15.20 | MIT |
| mypy | 2.1.0 | MIT |
| pathspec | 1.1.1 | MPL-2.0 (mypy dep; file-level copyleft, not linked into framework code) |
| detect-secrets | (CI secret scan) | Apache-2.0 **(upstream)** |

**All runtime dependencies on the default install path are permissive. No GPL/AGPL
runtime dependency is present.** Two exceptions are called out explicitly and neither
is on a default install path: `easydict` (**LGPL-3.0**, opt-in `latence-linkpred-ultra[ultra]`
extra only — see the flag above), and the MPL-2.0 `pathspec` (a mypy transitive, dev-only,
never shipped in any wheel).

## Default reference-model weights

The framework bundles no weights. These are the DEFAULT checkpoints the reference
Providers point at; each is fetched only when a Pipeline configures that Provider.
Weights and code licenses were verified separately on 2026-07-06 (ADR-0012).

| Model | Capability | Weights license |
|---|---|---|
| lightonai/LightOnOCR-1B-1025 | Parse (OCR, `parser.lighton`) | Apache-2.0 |
| urchade/gliner_multi-v2.1 | Entity Extraction | Apache-2.0 |
| knowledgator/gliner-relex-multi-v1.0 | Relation Extraction (`relation.gliner_relex`, superseded by gliner2) | Apache-2.0 |
| fastino/gliner2-multi-v1 | Fused Entity+Relation Extraction (`fused_entity_relation.gliner2`, W9 unified default) | Apache-2.0 |
| fastino/GLiNER2-Guardrails-PII-Multi | Redaction (`redaction.gliner2`) | Apache-2.0 |
| urchade/gliner_multi_pii-v1 | Redaction | Apache-2.0 |
| knowledgator/gliner-linker-large-v1.0 | Disambiguation (L3 linker; `disambiguation.cascade` reference + `disambiguation.glinker` enterprise, ADR-0046) | Apache-2.0 |
| knowledgator/gliner-linker-rerank-v1.0 | Disambiguation (L4 reranker; `disambiguation.glinker`, ADR-0046) | Apache-2.0 |
| ibm-granite/granite-embedding-311m-multilingual-r2 | Embedding (default, GPU tier; `embedding.sentence_transformers`, wired in `deploy/env/gpu-learned/stack.yaml`) | Apache-2.0 |
| ibm-granite/granite-embedding-97m-multilingual-r2 | Embedding (CPU tier) | Apache-2.0 |
| opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 | Sparse signal (`sparse.splade`, wired in `deploy/env/cpu-text/stack.yaml`) | Apache-2.0 |
| intfloat/multilingual-e5-small | Embedding (fallback) | MIT |
| sentence-transformers/all-MiniLM-L6-v2 | Embedding (fallback) | Apache-2.0 |
| microsoft/llmlingua-2 | Compression (post-v1) | MIT |
| answerdotai/ModernBERT | encoder base | Apache-2.0 |
| google/mt5 | seq2seq base | Apache-2.0 |
| Qwen/Qwen3-* (dense + MoE) | LLM | Apache-2.0 |

### Restricted-license model (OPT-IN, non-default)

| Model | Capability | License | Note |
|---|---|---|---|
| google/embeddinggemma-300m | Embedding | Gemma | **Not a default; no Gemma-backed Provider ships in this repo.** Not OSI open-source; carries Google's Prohibited Use Policy + flow-down (downstream must be bound to the Gemma restrictions, with a Notice). POLICY: the Gemma family is non-default and, were a Gemma Provider offered, it would be an explicitly opt-in Provider carrying the license and its flow-down obligations. Default Embedder is IBM Granite Embedding r2 (Apache-2.0). |

### CUDA runtime wheels — NOT permissive (issue #198)

`deploy/env/gpu-learned/requirements.lock` resolves 15 `nvidia-*` wheels (`nvidia-cuda-runtime`,
`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-cusparselt-cu13`, …) as
transitive dependencies of `torch==2.13.0+cu130`. **They are NOT open-source.** They ship NVIDIA's
proprietary CUDA Toolkit / cuDNN / NCCL end-user licence agreements, not an OSI licence: checked
against the PyPI JSON API on 2026-07-18, these distributions declare **no** `license`,
`license_expression` or `License ::` classifier at all — the terms live in the EULA text inside the
wheel and at <https://docs.nvidia.com/cuda/eula/>.

What that means for this project, stated plainly:

* Nothing changes for the framework's own licence, or for the CPU/text environment — the CUDA
  wheels are pulled in **only** by the optional `gpu-learned` environment, and only because the
  operator chose to install a GPU stack. `latence-core` and every Provider package remain
  permissively licensed and CUDA-free (ADR-0016).
* An operator installing `deploy/env/gpu-learned/` accepts NVIDIA's terms for those runtime
  libraries, exactly as they would installing CUDA from NVIDIA directly. This is normal for GPU
  serving and is called out here so it is a decision, not a surprise.
* `triton==3.7.1` (also transitive via torch, and required by `flashdeberta`) **is** permissive:
  MIT, per its PyPI classifier, verified the same day.
* UNVERIFIED: the precise EULA *version* bound to each pinned wheel was not independently
  confirmed — NVIDIA does not publish it in the package metadata. Treat the linked CUDA EULA as
  the governing terms and re-check if a redistribution question ever arises.

Rejected after verification (recorded so the check is not repeated): **naver/splade-v3** —
`cc-by-nc-sa-4.0` (non-commercial, not OSI open-source) and gated; not wired anywhere.
**BAAI/bge-m3** is MIT-licensed and was cleared on licence grounds, but it is NOT wired into
`sparse.splade`: its HF config declares `architectures: ["XLMRobertaModel"]` (no MLM head), so the
`AutoModelForMaskedLM` load path that Provider uses would randomly initialise `lm_head` and emit a
meaningless sparse vector. All three checked against the Hugging Face model API on 2026-07-18.

Per the project's research-diligence directive, a GLiNER/HF checkpoint's license
is per-weight and cannot be inferred from the library. Any model whose license
could not be verified is flagged UNVERIFIED rather than guessed; none in the
default set are currently unverified.

## Public benchmark datasets

The framework bundles no benchmark data. These are the PUBLIC datasets the
benchmark harness (`packages/latence-benchmark`, #181) can score against; each is
fetched only when a benchmark run names it, always at a pinned commit SHA. A
benchmark ships TWO artifacts under TWO licenses — the corpus and the evaluation
toolkit — so each was verified separately on 2026-07-18, exactly as ADR-0012
requires for weights vs. code. The harness's license gate refuses acquisition of
a non-commercial corpus unless the operator opts in explicitly, and refuses an
UNVERIFIED one unconditionally.

| Dataset | Used by | Corpus license | Toolkit license | Commercial use | Status |
|---|---|---|---|---|---|
| allenai/olmOCR-bench | Parse benchmark suite (default) | ODC-BY-1.0 (dataset-card `license: odc-by`) | Apache-2.0 (github.com/allenai/olmocr) | Yes, with attribution | Default; attribution carried on the published results page |
| opendatalab/OmniDocBench | Parse benchmark suite (opt-in) | **research-only, non-commercial** (project Copyright Statement: "The dataset is for research purposes only and not for commercial use.") | Apache-2.0 | **No** | Registered but GATED behind `--accept-non-commercial`; never a default, never vendored |

The OmniDocBench row is why the two licenses are recorded separately: the
repository's Apache-2.0 badge describes its evaluation toolkit, not its corpus.

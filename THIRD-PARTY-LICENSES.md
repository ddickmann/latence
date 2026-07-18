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
| gliner2[local] | ≥1.0,<2 | Apache-2.0 **(upstream)** | latence-pii-gliner2, latence-extract-gliner2 |
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
| ibm-granite/granite-embedding-311m-multilingual-r2 | Embedding (default, GPU tier; `embedding.sentence_transformers`) | Apache-2.0 |
| ibm-granite/granite-embedding-97m-multilingual-r2 | Embedding (CPU tier) | Apache-2.0 |
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

Per the project's research-diligence directive, a GLiNER/HF checkpoint's license
is per-weight and cannot be inferred from the library. Any model whose license
could not be verified is flagged UNVERIFIED rather than guessed; none in the
default set are currently unverified.

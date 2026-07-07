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
| gliner | ≥0.2.5 | Apache-2.0 **(upstream)** | latence-ner-gliner, latence-relation-gliner, latence-pii-gliner |
| openai | ≥1.0 | Apache-2.0 **(upstream)** | latence-*-endpoint, latence-relation-llm |
| apache-airflow | ≥2.7 | Apache-2.0 **(upstream)** | latence-runner-airflow (`airflow` extra only) |

`gliner`, `openai`, and `apache-airflow` are heavy/optional dependencies kept out
of the CPU-first dev environment by design (ADR-0016); their licenses are the
licenses published by the respective upstream projects
(github.com/urchade/GLiNER — Apache-2.0; github.com/openai/openai-python —
Apache-2.0; Apache Airflow — Apache-2.0). Verify against the installed
`dist-info` when installing that Provider.

### Dev-only tooling (not shipped in any wheel)

| Package | Version | License |
|---|---|---|
| pytest | 9.1.1 | MIT |
| ruff | 0.15.20 | MIT |
| mypy | 2.1.0 | MIT |
| pathspec | 1.1.1 | MPL-2.0 (mypy dep; file-level copyleft, not linked into framework code) |
| detect-secrets | (CI secret scan) | Apache-2.0 **(upstream)** |

**All runtime dependencies are permissive. No GPL/AGPL/LGPL or otherwise
non-permissive runtime dependency is present.** The single MPL-2.0 item
(`pathspec`, a mypy transitive dependency) is dev-only and never shipped.

## Default reference-model weights

The framework bundles no weights. These are the DEFAULT checkpoints the reference
Providers point at; each is fetched only when a Pipeline configures that Provider.
Weights and code licenses were verified separately on 2026-07-06 (ADR-0012).

| Model | Capability | Weights license |
|---|---|---|
| PaddleOCR-VL-1.5 | Parse | Apache-2.0 |
| LightOnOCR-2-1B | Parse | Apache-2.0 |
| urchade/gliner_multi-v2.1 | Entity Extraction | Apache-2.0 |
| knowledgator/gliner-relex-multi-v1.0 | Relation Extraction | Apache-2.0 |
| urchade/gliner_multi_pii-v1 | Redaction | Apache-2.0 |
| knowledgator/gliner-linker-large | Disambiguation | Apache-2.0 |
| intfloat/multilingual-e5-small | Embedding (default) | MIT |
| sentence-transformers/all-MiniLM-L6-v2 | Embedding | Apache-2.0 |
| microsoft/llmlingua-2 | Compression (post-v1) | MIT |
| answerdotai/ModernBERT | encoder base | Apache-2.0 |
| google/mt5 | seq2seq base | Apache-2.0 |
| Qwen/Qwen3-* (dense + MoE) | LLM | Apache-2.0 |

### Restricted-license model (OPT-IN, non-default)

| Model | Capability | License | Note |
|---|---|---|---|
| google/embeddinggemma-300m | Embedding | Gemma | **Not a default.** Not OSI open-source; carries Google's Prohibited Use Policy + flow-down (downstream must be bound to the Gemma restrictions, with a Notice). Available only as an explicitly opt-in Provider that surfaces the license at install time. Default Embedder is multilingual-e5-small (MIT). |

Per the project's research-diligence directive, a GLiNER/HF checkpoint's license
is per-weight and cannot be inferred from the library. Any model whose license
could not be verified is flagged UNVERIFIED rather than guessed; none in the
default set are currently unverified.

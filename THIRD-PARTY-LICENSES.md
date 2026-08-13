# Third-party licenses

> **The framework's own licence changed on 2026-07-29.** It is no longer
> Apache-2.0. It is the **PolyForm Noncommercial License 1.0.0**
> (SPDX: `PolyForm-Noncommercial-1.0.0`) — free for noncommercial use, commercial
> use by separate written agreement (see [`LICENSE`](./LICENSE) and
> [`COMMERCIAL.md`](./COMMERCIAL.md)). The repository was Apache-2.0 internally
> and **never published publicly** — private since creation, nothing on PyPI, no
> release cut, sole copyright holder across every commit — so the relicence is
> clean. The verified provenance record is at the top of `LICENSE`.
>
> **What that changes in this file, and what it does not.** The verified facts
> below are unchanged: every licence, version, and verification date still holds.
> What changed is the *framing*. This file used to argue that every inbound
> licence was permissive and therefore compatible with the project's own
> permissive outbound posture. That argument no longer applies, and the honest
> version runs in the opposite direction of travel:
>
> * **Inbound compatibility got easier, not harder.** A noncommercial outbound
>   licence imposes no obligations that a permissive inbound licence could
>   conflict with. Nothing here is now *in*compatible that was compatible before.
> * **But the outbound licence is not a shield.** Third-party terms bind the
>   *user* of this framework directly, independently of what licence the framework
>   itself carries. The LGPL-3.0, Gemma, NVIDIA-EULA and non-commercial-dataset
>   obligations recorded below apply to a commercial licensee exactly as they did
>   to a user under the old licence. A commercial agreement with the copyright
>   holder settles rights in *this* code and nothing else.
> * **Reciprocal licences are still the thing to watch, in both directions.** A
>   future inbound GPL/AGPL dependency would be a problem; there is none today.
>
> This file records verified facts. It is not legal advice.

The framework bundles no model weights and no vendored third-party source. This
file consolidates the licenses of the third-party **Python runtime dependencies**
its packages pull in, plus the **default reference-model** licenses (the
checkpoints the reference Providers point at — downloaded only when a Pipeline
configures that Provider).

Every runtime dependency on a **default install path** — `latence-core`, every
Provider package, and every extra except the two named below — is permissively
licensed (Apache-2.0 / MIT / BSD / ISC / PSF / MPL-2.0). Two carve-outs sit off
the default path and neither is small:

1. **`easydict` is LGPL-3.0** — weak copyleft, pulled only by the opt-in
   `latence-linkpred-ultra[ultra]` extra. Free to use as we use it; it carries
   real conditions **if a binary or container image is ever shipped**. Read
   [the LGPL-3.0 obligation, stated precisely](#the-lgpl-30-obligation-stated-precisely).
2. **NVIDIA CUDA wheels are proprietary** — pulled only by the two pinned GPU
   environments under `deploy/env/` (`gpu-learned/` and `served-ocr-server/`).
   Read
   [CUDA runtime wheels — NOT permissive](#cuda-runtime-wheels--not-permissive-issues-198-and-196)
   before installing either one. Nothing outside those two environments is
   affected.

## Python runtime dependencies

Licenses read back from each package's installed `*.dist-info/METADATA`
(`License` / `License-Expression` / `Classifier: License ::`) in the workspace
dev environment on 2026-07-18 (the pre-publish refresh, issue #72), except where a package is a heavy/optional
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
| click | 8.4.2 | BSD-3-Clause | (typer dep) |
| rich | 14.3.4 | MIT | CLI rendering |
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
> off the default install path. All other ULTRA/PyKEEN extras are permissive (torch
> BSD-3-Clause; torch-geometric/torch-scatter/pykeen MIT). The obligation this creates is
> stated precisely in the next section — it is **not** "LGPL is fine, move on".

### The LGPL-3.0 obligation, stated precisely

**Verified 2026-07-29** (re-verified under the new licence posture; first verified
2026-07-10). `easydict` 1.13 on PyPI declares `License: LGPL-3.0` and the classifier
`License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)`. Its upstream
repository (github.com/makinacorpus/easydict) ships the verbatim **GNU Lesser General
Public License, Version 3, 29 June 2007**; GitHub's licence API resolves it to
`spdx_id: LGPL-3.0`.

**What we actually do with it.** `latence-linkpred-ultra` imports `easydict` at run time,
unmodified, as a plain Python package that pip installed from PyPI. We do not vendor it,
do not modify it, and **do not redistribute it** — no wheel or sdist in this repository
contains a byte of easydict. In LGPL terms nothing here **conveys** the Library, so no §4
obligation is triggered for this project as it stands today.

**What changes the moment anything is shipped.** If you distribute an artifact that
*contains* easydict — a container image, a frozen/PyInstaller binary, an appliance, an
offline wheelhouse, a VM image, a bundled installer — that artifact is a **Combined
Work** and LGPL-3.0 **§4** applies. §4 permits conveying a Combined Work under terms of
your choice (including proprietary terms) **only if** those terms, taken together, do not
restrict modification of the Library portions or reverse engineering for debugging such
modifications, **and** only if you do *each* of the following:

| LGPL-3.0 §4 | What it requires of a shipped artifact |
|---|---|
| §4(a) | Give **prominent notice** with each copy of the Combined Work that the Library is used in it and that the Library and its use are covered by the LGPL. |
| §4(b) | **Accompany** the Combined Work with a copy of the GNU GPL **and** the LGPL text. |
| §4(c) | If the Combined Work displays copyright notices during execution, include easydict's copyright notice among them, plus a pointer to the GPL/LGPL copies. |
| §4(d) | Do **one** of: **(d)(0)** convey the Minimal Corresponding Source for easydict under the LGPL **and** your Corresponding Application Code in a form that lets the user recombine or relink against a modified easydict; **or** **(d)(1)** use a suitable shared-library mechanism that uses at run time a copy of easydict already present on the user's system and that works with any interface-compatible modified version. |
| §4(e) | Provide **Installation Information** — but only where **GPL-3.0 §6** would otherwise require it (the "User Product" / anti-tivoization case), and only as far as necessary to install and run a Combined Work recombined with a modified easydict. |

Practical reading for a Python distribution: §4(d)(1) is the cheap route, and an ordinary
`pip install` already satisfies its substance — easydict stays a separate,
user-replaceable package on the import path, so a user can swap in a modified copy. A
**frozen binary or a locked-down appliance breaks that**, and then §4(d)(0) — shipping
relinkable source — is what remains. §4(e) can additionally bite for consumer hardware.

**UNVERIFIED:** whether upstream elected the "or any later version" option of LGPL-3.0
**§6** (Revised Versions). The package metadata says `LGPL-3.0`, not `LGPL-3.0-or-later`,
and the repository ships the bare licence text without a stated per-file election. Treat
the terms as LGPL-3.0 exactly, and re-check if a downstream ever needs to rely on a later
version.

**The clean way out, if it matters.** The whole use of easydict is one lazy import and
two `EasyDict({...})` calls that wrap the ULTRA relation/entity config dicts so the
vendored model code can read them with dotted attribute access
(`latence_linkpred_ultra/scorer.py`, the `EasyDict` import and the `rel_cfg`/`ent_cfg`
construction). A small permissive attribute-dict shim replaces it and removes the
dependency and the entire obligation. That remains a real option, recorded here so the
choice stays deliberate (ADR-0012 / research-diligence) rather than defaulted into.

**Not legal advice.** This is a reading of the licence text, not counsel's opinion. See
the legal-review list in `COMMERCIAL.md`.

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
extra only — see [the LGPL-3.0 obligation, stated
precisely](#the-lgpl-30-obligation-stated-precisely)), and the MPL-2.0 `pathspec` (a mypy
transitive, dev-only, never shipped in any wheel).

That statement is about **inbound** licences only. It says nothing about whether *your*
use of the framework is permitted — that is governed by `LICENSE` (PolyForm Noncommercial
1.0.0) and, for commercial use, by a separate agreement (`COMMERCIAL.md`). A commercial
licensee is bound by the third-party terms above exactly as anyone else is.

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

### Served-OCR serving dependencies (issue #196)

`deploy/env/served-ocr-server/requirements.lock` is the pinned vLLM serving environment behind the
OCR endpoint seam (ADR-0030). It runs **outside** any framework process — the framework's own side
of the seam is `deploy/env/served-ocr-client/`, whose only model dependency is an
OpenAI-compatible HTTP client. Its principal distributions, licences read back from the PyPI JSON
API on 2026-07-19:

| Package | Version | License | Note |
|---|---|---|---|
| vllm | 0.24.0 | Apache-2.0 | `license_expression: Apache-2.0`; the serving runtime itself |
| flashinfer-python | 0.6.12 | Apache-2.0 | `license_expression: Apache-2.0`; attention kernels (FM-BLACKWELL-FLASHINFER) |
| flashinfer-cubin | 0.6.12 | Apache-2.0 | precompiled cubins for the above |
| triton | 3.6.0 | MIT | per PyPI classifier (the served env pins 3.6.0; `gpu-learned` pins 3.7.1) |
| tokenizers | 0.22.2 | Apache-2.0 | per PyPI classifier; also pinned by `cpu-text` / `gpu-learned` |
| pillow | 12.3.0 | MIT-CMU | `license_expression: MIT-CMU`; page rasterisation in `served-ocr-client` |

The served **model** (`lightonai/LightOnOCR-2-1B`) has its weights and code licences cited
separately in `deploy/env/compatibility-matrix.yaml`, per ADR-0012.

The CUDA wheels this environment drags in are the subject of the next section.

### CUDA runtime wheels — NOT permissive (issues #198 and #196)

**Two** committed environments pin a CUDA stack, and both are covered here:

| Lock | `nvidia-*` wheels | Pulled in by |
|---|---|---|
| `deploy/env/gpu-learned/requirements.lock` | 15 | `torch==2.13.0+cu130` |
| `deploy/env/served-ocr-server/requirements.lock` | 25 | `torch==2.11.0+cu130`, `vllm==0.24.0`, `flashinfer-python==0.6.12` |

**Most of these are NOT open-source.** They ship NVIDIA's proprietary CUDA Toolkit / cuDNN / NCCL /
cuSPARSELt / CUTLASS end-user licence agreements, not an OSI licence. Checked against the PyPI JSON
API on 2026-07-19, the metadata is inconsistent by design: some declare
`LicenseRef-NVIDIA-Proprietary`, some `License :: Other/Proprietary License`, and many declare
**nothing at all** — the terms live in the EULA text inside the wheel and at
<https://docs.nvidia.com/cuda/eula/>. A prefix cannot tell them apart, so every wheel is named
individually below, including the handful that really are permissive.

`gpu-learned` (15, all also present in the served-OCR lock unless marked):

| Package | Version | Declared license | Verdict |
|---|---|---|---|
| nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cuda-runtime | 13.0.96 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cudnn-cu13 | 9.20.0.48 | *(none declared)* | proprietary (cuDNN EULA) |
| nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software | proprietary |
| nvidia-nccl-cu13 | 2.29.7 | *(none declared)* | proprietary (NCCL EULA) |
| nvidia-nvjitlink | 13.3.33 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-nvshmem-cu13 | 3.4.5 | *(none declared)* | proprietary (NVSHMEM EULA) |
| nvidia-nvtx | 13.0.85 | Apache 2.0 | **permissive** |

`served-ocr-server` (25 — the CUDA-13 set above at the pinned 2.11.0+cu130 versions, plus the
compiler/CUTLASS wheels vLLM and FlashInfer need):

| Package | Version | Declared license | Verdict |
|---|---|---|---|
| nvidia-cublas | 13.1.0.3 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cuda-cccl | 13.3.3.4.1 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cuda-crt | 13.3.73 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cuda-nvcc | 13.2.86 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cuda-runtime | 13.0.96 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cuda-tileiras | 13.2.86 | *(none declared)* | proprietary (CUDA EULA) |
| nvidia-cudnn-cu13 | 9.19.0.56 | *(none declared)* | proprietary (cuDNN EULA) |
| nvidia-cudnn-frontend | 1.26.0 | MIT | **permissive** |
| nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-cusparselt-cu13 | 0.8.0 | NVIDIA Proprietary Software | proprietary |
| nvidia-cutlass-dsl | 4.5.2 | License :: Other/Proprietary License | proprietary |
| nvidia-cutlass-dsl-libs-base | 4.5.2 | License :: Other/Proprietary License | proprietary |
| nvidia-cutlass-dsl-libs-cu13 | 4.5.2 | License :: Other/Proprietary License | proprietary |
| nvidia-ml-py | 13.610.43 | BSD | **permissive** |
| nvidia-nccl-cu13 | 2.28.9 | *(none declared)* | proprietary (NCCL EULA) |
| nvidia-nvjitlink | 13.0.88 | LicenseRef-NVIDIA-Proprietary | proprietary |
| nvidia-nvshmem-cu13 | 3.4.5 | *(none declared)* | proprietary (NVSHMEM EULA) |
| nvidia-nvtx | 13.0.85 | Apache 2.0 | **permissive** |
| nvidia-nvvm | 13.2.86 | *(none declared)* | proprietary (CUDA EULA) |

What that means for this project, stated plainly:

* Nothing changes for the framework's own licence, or for the CPU/text environment, or for the
  OCR **client** environment — the CUDA wheels are pulled in **only** by `gpu-learned/` and
  `served-ocr-server/`, and only because the operator chose to install a GPU stack.
  `latence-core` and every Provider package remain CUDA-free (ADR-0016) and carry the
  framework's own licence (PolyForm Noncommercial 1.0.0), never NVIDIA's terms.
* An operator installing `deploy/env/gpu-learned/` **or** `deploy/env/served-ocr-server/` accepts
  NVIDIA's terms for those runtime libraries, exactly as they would installing CUDA from NVIDIA
  directly. This is normal for GPU serving and is called out here so it is a decision, not a
  surprise.
* `triton` (transitive via torch, and required by `flashdeberta`) **is** permissive: MIT per its
  PyPI classifier — 3.7.1 in `gpu-learned`, 3.6.0 in `served-ocr-server`, both verified.
* This disclosure is machine-gated, not a promise: `latence env matrix` reports
  `E-LICENCE-UNDISCLOSED` if a committed lock pins an `nvidia-*` wheel this file does not name, or
  does not cite that lock's path — see `_check_disclosure` in `latence_core.compat`. That gate
  exists because #196 and #198 each shipped a pinned GPU environment and only one of them was
  disclosed here.
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
| Babelscape/REDFM | Entity-Extraction, Relation-Extraction and Disambiguation suites (#187) | CC BY-SA 4.0 (dataset-card `license: cc-by-sa-4.0` **and** its prose License section) | Apache-2.0 (the repo's own `REDFM.py` loader, per its file header) | Yes, with attribution | Default for the three suites; fetched at run time, never vendored, and only aggregate metrics are published — so ShareAlike, which binds adaptations of the corpus, is not triggered |
| Latence synthetic PII corpus | Redaction suite (#187) | PolyForm-Noncommercial-1.0.0 (this repository) | PolyForm-Noncommercial-1.0.0 (this repository) | Noncommercial free; commercial by agreement (`COMMERCIAL.md`) | SHIPPED in `latence_benchmark/corpora/`. Contains **no real personal data**: RFC 2606 example domains, RFC 5737 TEST-NET addresses, 555 fiction-reserved phone numbers, published test card/IBAN values, AWS's `AKIAIOSFODNN7EXAMPLE` docs key, fictional names. Generated by a committed generator and pinned by content digest |

### E2E-SOTA campaign datasets (retrieval benchmarks)

The end-to-end retrieval campaign (`latence-campaign`, the S1–S5 stages) scores against seven
public multi-hop / document-QA datasets. Their license lanes are **machine-enforced**, not just
documented: each entry below is read verbatim from
`latence_benchmark.campaign.corpus_stage.CORPUS_PLAN`, the acquisition stage refuses a dataset
whose lane it cannot honour, and the repo's `.gitignore` keeps every corpus-derived artifact
local (raw acquisitions, OCR output, exports, per-chunk reports) — only aggregate metrics are
committed. All verified 2026-08-02; "commercial" describes the dataset's own terms, not a grant
from this repository.

| Dataset | Lane | Corpus license | Upstream toolkit | Commercial use |
|---|---|---|---|---|
| MuSiQue | attribution | CC-BY-4.0 | none executed | Yes, with attribution |
| HotpotQA | share-alike-local | CC-BY-SA-4.0 | none executed | Yes; share-alike derived text stays local |
| 2WikiMultihopQA | share-alike-local | Apache-2.0 (card); paragraph text CC-BY-SA Wikipedia-derived | none executed | Yes; derived text stays local |
| MultiHop-RAG | strictly-local | ODC-BY-1.0 (annotations); news text under publisher copyright | none executed | Yes (annotations); article text never redistributed |
| OHR-Bench | research-only-local | **research-only, non-commercial** | Apache-2.0 (not executed) | **No** — benchmark-scoring only, never a product input |
| ViDoSeek | strictly-local | Apache-2.0 (annotations); source-PDF copyright retained upstream | none executed | Yes (annotations); PDFs never redistributed |
| UDA | share-alike-local | CC-BY-SA-4.0 | Apache-2.0 (not executed) | Yes; derived text stays local |

Datasets considered and **excluded** on license grounds (recorded so the decision
is not silently re-litigated):

| Dataset | Considered for | Why excluded |
|---|---|---|
| unimelb-nlp/wikiann (PAN-X) | Multilingual NER suite | Its Hugging Face card carries `license: unknown` (checked 2026-07-18). An unverifiable license is excluded, not guessed — and the harness's gate would refuse it unconditionally anyway |
| Babelscape/rebel (the REDFM authors' model repo) | — | The GitHub repository declares **no** license file (GitHub API, 2026-07-18), so it is UNVERIFIED. Nothing from it is used; the harness reads RED^FM's JSONL directly and never executes its loading script |

The OmniDocBench row is why the two licenses are recorded separately: the
repository's Apache-2.0 badge describes its evaluation toolkit, not its corpus.
The RED^FM row is the same discipline with the opposite outcome — both artifacts
verified, both permissive, and the ShareAlike obligation reasoned about explicitly
rather than assumed away.

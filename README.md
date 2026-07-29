# Latence Framework

**Point it at a folder of messy enterprise documents. Get back a PII-handled, embedded,
provenance-carrying RAG corpus and an evidence-linked knowledge graph — as plain Parquet files
you own.**

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm--Noncommercial--1.0.0-blue)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](./pyproject.toml)
[![tests 4,590 passing](https://img.shields.io/badge/tests-4%2C590%20passing-brightgreen)](./CONTRIBUTING.md)
[![mypy --strict](https://img.shields.io/badge/mypy-strict-brightgreen)](./scripts/typecheck.sh)
[![status pre-1.0](https://img.shields.io/badge/status-pre--1.0-orange)](#status--maturity)

*(No CI or PyPI badge: the repository is private, no distribution has ever been published, and a
badge for a run that has not happened is a lie. See [Status & maturity](#status--maturity).)*

---

## The problem

Enterprise document data is scans, PDFs, office files and text dumps in different systems, with no
shared schema, no resolved entities, PII scattered through the body text, and no explicit structure
connecting one document to another. Naive RAG over that pile chunks everything, embeds it, and hopes
a vector index lands near the answer — which fails exactly when the answer lives in a *different*
document from the one the question names, and fails silently, as a confident wrong answer.

This framework does the unglamorous part: parse the formats that actually exist, extract entities
**and** the cross-references that link documents, resolve them into one canonical graph, mask the PII
under a floor no model choice can bypass, and keep every record traceable to its exact character
offset and source page.

---

## 60-second quickstart

CPU-only. No GPU, no cloud, no credentials, no network calls at run time.

```bash
git clone https://github.com/ddickmann/latence-framework && cd latence-framework
uv sync
uv run latence-demo
open /tmp/latence-demo/_latence/runs/demo/export/index.html
```

That ingests a bundled **deliberately messy** corpus — text/markdown, a real PDF, and four planted
dangerous files — and prints exactly this:

```
  documents: 4 ingested, 3 quarantined, 1 flagged
  graph:     6 nodes, 33 edges, 2 KB-linked, 33 edges with Evidence
  contracts: complete
  screening checks:
    - archive.zip: expected quarantine:zip_bomb, got quarantine:zip_bomb [ok]
    - dump.txt: expected quarantine:oversized, got quarantine:oversized [ok]
    - invoice.pdf: expected quarantine:type_spoof, got quarantine:type_spoof [ok]
    - support-ticket.md: expected flag:prompt_injection, got flag:prompt_injection [ok]
```

The zip bomb, the oversized file and the `.pdf` that is not a PDF are **quarantined before parsing**.
The prompt-injection document is **flagged and carried into the corpus** so a RAG consumer can
exclude it. Every one of the 33 graph edges carries the sentence it came from. `latence-demo` exits
non-zero if any of that stops being true.

**The model-agnostic proof — one line changes, the pipeline does not:**

```bash
uv pip install -e packages/latence-ner-gliner   # a learned NER Provider (pulls torch)
uv run latence-demo --entity-provider entity.gliner
```

Without that install the swap exits non-zero with a message naming the package to install — it never
degrades quietly to the default. `diff examples/demo-pipeline.yaml examples/demo-pipeline.swap.yaml`
is the whole change.

**Then run it on your own corpus:**

```bash
uv run latence setup     # guided wizard: inspects your files, emits a validated stack config
uv run latence process   # runs it
```

Full walkthrough: [`docs/getting-started/index.md`](./docs/getting-started/index.md) and
[`docs/TUTORIAL-fresh-pod-walkthrough.md`](./docs/TUTORIAL-fresh-pod-walkthrough.md).

---

## What it produces

Files on your Storage (`file://`, `s3://`, `gs://`, `abfs://` — same code, one config key). Nothing
is hidden in a proprietary index.

| Artifact | What it is |
|---|---|
| `records.parquet` / `records.jsonl` | the RAG corpus — one row per chunk, with dense embedding, PII-masked text and denormalized KG columns |
| `graph-nodes.parquet` | resolved canonical entities, with PageRank centrality + community |
| `graph-edges.parquet` | extracted + reference relations, **every edge carrying its Evidence** |
| `graph.ttl` / `graph.graphml` | the same graph for RDF and graph-viz tooling |
| `bm25-stats.json` + `bm25-postings.parquet` | index-time BM25 term statistics whose term unit exactly matches the query-side rescorer |
| `multivectors.parquet` | optional late-interaction sidecar, keyed by `record_id` (off by default) |
| `quality-report.json` + `quality-report.md` | the machine + human artifact that substantiates "AI-ready" |
| `manifest.json` | the exact pipeline that produced this run, with credentials redacted |

A real record, copied out of `records.jsonl` from the quickstart run above and abridged only in the
`content` field:

```jsonc
{
  "record_id":      "sha256:981bbeb4…#chunk-0",   // content-addressed, so it is stable across runs
  "schema_version": 19,
  "document_record_id": "sha256:981bbeb4…",
  "index":          0,
  "content":        "Acquisition memo\n\nAlice works at Acme Corporation. …",
  "masked_content": "…the same text with every PII span replaced by its placeholder…",
  "media_type":     "text/markdown",
  "token_count":    56,

  "provenance": {
    "source_uri":  "file:///…/corpus/memo.md",
    "file_name":   "memo.md", "file_type": "md", "file_size": 242,
    "document_id": "sha256:981bbeb4…",
    "page_start":  1, "page_end": 1,
    "char_start":  2, "char_end": 242          // exact half-open span in the parsed document
  },
  "classification": { "language": "en", "category": null, "sensitivity": "internal" },

  "page_slice":  { "pages": [ { "page_number": 1, "char_start": 0, "char_end": 242 } ] },
  "offset_map":  { "length": 240, "segments": [ { "local_start": 0, "doc_start": 2 } ] },
  "pii_spans":   [],        // counts and offsets only — a span never stores the raw PII value
  "risk_markers": [],       // a flagged chunk carries its screening verdict into the corpus
  "induced_labels": null,   // this chunk's own label set, when induction ran
  "context_header": null    // the KG projected back onto this chunk, when enrichment ran
}
```

`records.parquet` is the same data flattened (ADR-0006): `record_id`, `schema_version`, `provenance`,
`classification`, `content`, `media_type` as columns, with the nested contracts as JSON strings —
plus native columns that appear **only** when the producing stage is wired, so the default schema
stays byte-stable: `embedding` (`list<double>`), `sparse_indices`/`sparse_values` (SPLADE),
`fde_embedding` (MUVERA), and `context_header`, `kg_node_ids`, `neighbor_node_ids`,
`kg_node_community`, `kg_node_centrality`. Those `kg_node_ids` are why multi-hop retrieval can start
from a chunk you already found and walk the graph outward.

`provenance` is not decoration. `char_start`/`char_end` are the chunk's exact half-open span in the
parsed document; `page_start`/`page_end` resolve through the chunk's **own** page slice rather than
an inherited range; and the `offset_map` corrects for markup stripped before extraction, so an entity
mention resolves to its **true original offset**, not a lower bound (ADR-0031).

`graph-edges.parquet` carries `edge_id`, `relation_id`, `source_node_id`, `target_node_id`, `label`,
`confidence`, `properties` and `evidence` — the last one being why a claim in the graph can always be
traced back to the sentence that produced it.

---

## The pipeline

Each row is a **Stage**. The model named is the one the enterprise profile
(`latence setup --profile enterprise-sota`) selects; every one of them is a swap, not a dependency.

| # | Stage | Provider | What runs it |
|---|---|---|---|
| 1 | Source | `source.local_folder` | deterministic, zero-dep |
| 2 | Intake screening | `screening.intake_signature` | deterministic — rejects oversized / zip-bomb / type-spoofed files **before** parsing |
| 3 | Parse | `parser.lighton_vllm` / `parser.plaintext` | `lightonai/LightOnOCR-2-1B` over a vLLM endpoint (scans, PDFs, images) |
| 4 | Chunk | `chunk.markdown` | deterministic, offset- and page-preserving |
| 5 | Content screening | `screening.content_keyword` | deterministic prompt-injection / harmful-content guardrail |
| 6 | Label induction | `label_inducer.llm` | `openai/gpt-4.1-mini` via any OpenAI-compatible endpoint — **per chunk** |
| 7 | Extraction (NER + relations, fused) | `fused_entity_relation.gliner2` | `fastino/gliner2-multi-v1` (zero-shot, mdeBERTa-v3) |
| 7b | Reference extraction | `…header_refs`, `…inline_refs` | deterministic, zero-dep — the cross-document edges |
| 8 | Redaction + guardrails | `redaction.gliner2` | `fastino/GLiNER2-Guardrails-PII-Multi`, over a non-bypassable floor |
| 9 | Profiling | `profiling.statistical` | deterministic corpus-level features |
| 10 | Resolution + type consolidation | `disambiguation.glinker` | `knowledgator/gliner-linker-large-v1.0` + its reranker |
| 11 | Graph assembly + centrality | `graph.canonical` + `graph_features.pagerank` | pure-Python, deterministic |
| 12 | Graph completion *(optional)* | `graph_completion.ultra` | ULTRA `mgalkin/ultra_3g` — inferred edges tagged `inferred: true` |
| 13 | Context enrichment | `context.kg_header` | projects the KG back onto each chunk (embedding input only) |
| 14 | Export | `export.knowledge_graph` + `export.jsonl_parquet` | `ibm-granite/granite-embedding-311m-multilingual-r2` (768-dim) + BM25 stats |

**Stages 1–8 are chunk-local. Stages 9–14 are corpus-wide.** That line is the design: the expensive
half is embarrassingly parallel and independent of corpus size, and the one pass that must see
everything runs once, over already-extracted records rather than raw text.

Retrieval is a separate, **strictly stateless** library (`latence-retrieval`, ADR-0048): it holds no
index and runs no first-stage search. You keep Qdrant / Elastic / whatever you already run; it fuses,
expands over the emitted graph, reranks and packs the candidates *your* engine returned.
See [`docs/retrieval-tooling.md`](./docs/retrieval-tooling.md).

---

## Architecture in one paragraph

A **Stage** is a typed transform over records. It depends only on a **Capability** — a narrow
interface such as `Parser`, `EntityExtractor`, `Embedder` — and the core never names a model. A
**Provider** fulfils a Capability: an in-process model, a served endpoint, a cloud service. Providers
live in their own installable packages (33 of them today), so swapping the OCR model, the induction
LLM or the embedder is a config key rather than a refactor, and nothing pulls torch into your
environment unless you ask for it. A **Runner** executes a Pipeline: the built-in local Runner runs
the whole spine on a laptop with checkpoint/resume, and a shipped Airflow adapter
(`latence-runner-airflow`) runs the identical Stages as a DAG. **Storage** is one fsspec seam. Every
record carries Provenance and Classification, validated at every Stage boundary.

The reasoning behind each of those choices is written down: **53 architecture decisions** in
[`docs/adr/`](./docs/adr/), and the glossary that fixes the vocabulary in
[`CONTEXT.md`](./CONTEXT.md).

---

## What makes it different

Five claims, each with the thing that makes it checkable.

**1 · Labels are induced per chunk, not fixed corpus-wide.**
A global schema forces a bad trade: cap it and rare-but-critical types vanish; apply all of it to
every chunk and zero-shot precision collapses under labels that have nothing to do with the text.
GLiNER2 is a uni-encoder — labels and text share one sequence — and its maintainers put the safe
ceiling at 20–30 labels. So each chunk is scored against its own small set, and *consistency* is
earned later, by a resolution stage that can see the whole corpus. The evidence, including the
measured label-count degradation and why the `predefined_labels` seam is the wrong one, is in
[`issues/e2e-extended/DECISIONS.md`](./issues/e2e-extended/DECISIONS.md) §D5.

**2 · A universal PII floor that no Provider can bypass.**
Beneath the learned redactor sits a deterministic recogniser for `ssn` / `credit_card` / `iban`,
enforced at the single `finalize_redacted_chunk` seam that **every** redaction Provider funnels
through. A learned model that misses a card number still cannot leak it. The guardrail twin does the
same for `prompt injection` / `harmful content`: those categories are unioned onto every chunk and
can never be subsetted by induction. Relevance labels are per-chunk; safety floors are universal —
`packages/latence-core/src/latence_core/redaction_policy.py`, ADR-0044.
Stated honestly: the PII floor is a *detection* guarantee for those three patterns; the guardrail
floor is a *label* guarantee (the model is always asked about them), not a promise it always hits.

**3 · Provenance to the character offset and the source page.**
Every record carries a frozen `Provenance` with `source_uri`, a content-addressed `document_id`,
`char_start`/`char_end` and `page_start`/`page_end`. A chunk holds its own page slice rather than
inheriting the document's, and a missing slice raises `PageSliceMissingError` rather than guessing a
plausible wrong page. The G1 gate read this back across three corpora: **49/49, 40/40, 40/40** records
with Provenance *and* Classification *and* aligned offsets, `drift_recovered = 0`.

**4 · Deterministic, byte-identical runs.**
Same input, same config, same bytes out — asserted on the KG deliverables by the demo suite, and
re-verified by the G1 gate's chaos lane: a child process `SIGKILL`ed mid-checkpoint-write, resumed,
produced a final export **byte-identical** to an uninterrupted run. The torn temp file never becomes
live. `PYTHONHASHSEED` is pinned before the interpreter starts and a root conftest fails the suite if
it is not.

**5 · Typed errors, no silent degradation.**
One taxonomy (`LatenceError` → `ContractError` / `ConfigError` / `ProviderError` / `StorageError`,
each also subclassing the builtin it replaces, so pre-existing `except ValueError` call sites keep
working). A Capability the retrieval backend cannot serve is a loud error, never a quietly dropped
leg; a missing Provider exits non-zero naming the package to install; a Stage that skips for lack of
a device says so in the Quality Report. **And the documentation is machine-gated:** doc-truth tests
fail the build when this README's ADR count drifts from `docs/adr/`, when a link or a section anchor
404s, when an install line names a distribution that does not exist, or when a published page calls
the dependency tree clean while the disclosure file says otherwise — see
`packages/latence-core/tests/test_front_door_truthfulness.py`.

### The measured retrieval result

80 IETF RFCs through the enterprise profile on a Blackwell GPU, then retrieval measured against a
**live Qdrant** — no mocks anywhere in the measured path:

| retrieval configuration | recall@10 · all 90 questions | recall@10 · the 7 multi-hop cases |
|---|---|---|
| dense only (Granite → Qdrant) | 0.644 | **0.000** |
| BM25 only | 0.656 | **0.000** |
| **fused — dense + BM25 + KG 2-hop + rerank** | **0.678** | **1.000** |

Read honestly: across the broad set the aggregate lift is **+3 points**, because most questions are
answerable by a single signal. The advantage is concentrated on the 7 of 90 (~8%) questions that
genuinely require chaining evidence across documents — where single signals score zero. The
defensible claim is not "fused is better everywhere"; it is that the graph unlocks a class of query
that single retrievers structurally cannot answer. One detail decided it: the header-only extractor
recovered **14%** of the corpus's real cross-reference edges, and adding the deterministic
inline-citation extractor took that to **~112%** of the gold graph. No edges, no multi-hop win.
Method, the seven worked examples, and the limitations:
[`results/dogfood-rfc/PROOF.md`](./results/dogfood-rfc/PROOF.md) and
[`docs/DEEP-DIVE.md`](./docs/DEEP-DIVE.md).

---

## Status & maturity

Pre-1.0, and specific about it.

**Green, and reproducible on a laptop:**

- **4,590 tests pass, 33 skipped** — `PYTHONHASHSEED=0 uv run pytest -q` (~11 min, CPU, offline).
- **`mypy --strict` over src *and* tests** for every workspace package — `bash scripts/typecheck.sh`.
- **`bash scripts/verify-local.sh` exits 0** — the full merge gate: secret scan, ruff, strict types,
  the suite, provider conformance, end-to-end validation of 9 stack configs, the reproducible
  environment specs, the compatibility matrix, and 7 bake-off matrices.
- The **G1 profound-testing gate is signed off** — end-to-end on 3 messy corpora, security review,
  crash-resume and delta idempotency, a performance baseline, determinism:
  [`issues/gate/G1-RESULTS.md`](./issues/gate/G1-RESULTS.md), residuals included.
- Measured per-Stage throughput with hardware, date and a reproduction command on every row, plus an
  explicit *not yet measured* list: [`PERF-RESULTS.md`](./PERF-RESULTS.md).

**Not yet proven, stated plainly:**

- **The extended GPU end-to-end run is prepared, not proven.** The pipeline itself is built, merged
  and green, and the RFC result above was measured on real hardware — but the large, staged run on a
  24 GB Ampere card is designed down to the ticket level and has **not been executed**. Two defects
  are already documented against it: a TorchScript-JIT kill that fires unconditionally on hardware
  that never needed it, and micro-batch/thread defaults that encode one specific 32 GB / 128-core pod
  instead of reading the machine. [`issues/e2e-extended/`](./issues/e2e-extended/README.md) holds the
  decisions, the tickets and the evidence for each.
- **The retrieval benchmark harness is designed but deliberately unbuilt.** Hop stratification, the
  eight necessity gates (closed-book, single-hop baseline, per-edge necessity, DiRe, bridge leakage,
  adversarial distractors, evidence accessibility, human audit) and the KG ablation are specified in
  [`DECISIONS.md`](./issues/e2e-extended/DECISIONS.md) §D9 and remain **unimplemented** — building a
  measurement harness against a stand-in corpus is a mistake this project has already paid for once.
  The RFC numbers are a real measurement of a narrower thing: one corpus, 90 generated questions,
  single-passage gold.
- **Scale is characterised, not eliminated.** The 100k / 1M-document envelopes are parked in
  `PERF-RESULTS.md` rather than previewed.
- **SPLADE's index-time path is built; native sparse wiring into the fused proof is deferred.** The
  measured stack above is three signals, not four, and says so.
- **Multi-vector / MUVERA ship as real but experimental, off by default,** naming no checkpoint
  (ADR-0050).
- **Nothing has been released.** No public artifact exists and nothing is on PyPI. The signing and
  SBOM machinery is built and tested; it has not yet produced a release.

Bus factor, fork posture, support lifetime and CRA readiness are written down rather than implied:
[`docs/CONTINUITY.md`](./docs/CONTINUITY.md). Start there if you are evaluating this for adoption.

---

## Install

A thin core plus one distribution per Provider family (ADR-0016) — you install only what you use, so
the "runs on a laptop" promise holds. There is no `latence` meta-package; installing is always one or
more of these. Python ≥ 3.11.

```bash
pip install latence-core          # contracts, Capabilities, local Runner, Storage, CLI
pip install latence-demo          # + the CPU demo set and its self-contained viewer
pip install latence-console       # + the read-only, offline localhost run console
pip install latence-retrieval     # the stateless query-time library (fuse / expand / rerank / pack)

pip install latence-extract-gliner2    # opt-in: fused GLiNER2 NER + relations (pulls torch)
pip install latence-pii-gliner2        # opt-in: learned PII + guardrails
pip install latence-disambig-glinker   # opt-in: GLinker entity resolution
pip install latence-embedder-st        # opt-in: sentence-transformers embedder
pip install latence-sink-qdrant        # one-shot idempotent loader: Export → a Qdrant collection
```

`packages/` holds every distribution and each carries its own README; the list above is a tour, not
the index.

---

## Security & release integrity

- **Found a vulnerability? Report it privately.** The coordinated-disclosure policy is in
  [`SECURITY.md`](./SECURITY.md) — acknowledgement within 3 business days, 90-day target embargo.
  Please do not open a public issue for a suspected vulnerability. The trust model and the built-in
  controls are there too, with the attack surface in
  [`docs/THREAT-MODEL.md`](./docs/THREAT-MODEL.md).
- **Releases will be verifiable.** Signed with keyless [Sigstore](https://www.sigstore.dev/) and
  shipping a per-release CycloneDX SBOM; two commands verify a download —
  [`docs/RELEASING.md`](./docs/RELEASING.md) and [`docs/SBOM.md`](./docs/SBOM.md).
- **Third-party terms are disclosed, not summarised away.**
  [`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md) is permissive by default **with two
  disclosed exceptions you must read before relying on them**: the pinned GPU environments under
  `deploy/env/` pull proprietary NVIDIA CUDA wheels (each named, each under its own EULA), and
  EmbeddingGemma is opt-in only, under the Gemma license with flow-down obligations.
  `latence env matrix` machine-checks that disclosure against the tree.

---

## Licence

**PolyForm Noncommercial License 1.0.0** (SPDX: `PolyForm-Noncommercial-1.0.0`).

- **Noncommercial use is free** — personal study, research, teaching, and use by noncommercial
  organisations, including the right to distribute and to make changes within that purpose.
- **Commercial use requires a separate written agreement** with the copyright holder. Terms are by
  agreement; there is no price list. `COMMERCIAL.md` sets out what counts as commercial use, what a
  commercial licence does and does not cover, and how to ask.
- [`LICENSE`](./LICENSE) governs, and carries the provenance record for the relicence: this
  repository was Apache-2.0 internally until 2026-07-29 and **was never published** — private since
  creation, zero forks, nothing on PyPI, no release cut, every commit by the sole copyright holder —
  so no one holds a prior grant.
- A commercial licence covers this project's own code only. Third-party dependencies, model weights
  and datasets keep their own terms, and several restrict commercial use independently
  ([`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md)).
- Not legal advice, and not reviewed by counsel — a caveat written into the licence documents
  themselves rather than left for you to infer.

Copyright 2026 Dennis Dickmann (latence.ai). See [`NOTICE`](./NOTICE).

---

## Where to go next

| | |
|---|---|
| **Understand the whole thing** | [`docs/DEEP-DIVE.md`](./docs/DEEP-DIVE.md) — messy files → knowledge graph → multi-hop retrieval, with the reasoning |
| **Run it on your corpus** | [`docs/getting-started/index.md`](./docs/getting-started/index.md), then [`docs/TUTORIAL-fresh-pod-walkthrough.md`](./docs/TUTORIAL-fresh-pod-walkthrough.md) |
| **Read the results** | [`docs/getting-started/run-console.md`](./docs/getting-started/run-console.md) — the offline, read-only run console |
| **Keep a corpus current** | [incremental add / update / retract / GDPR purge](./docs/TUTORIAL-fresh-pod-walkthrough.md#8-incremental-updates-add--update--retract--purge) |
| **Compare Providers** | [`docs/running-a-bakeoff.md`](./docs/running-a-bakeoff.md) · [`docs/running-a-stack.md`](./docs/running-a-stack.md) |
| **Check our numbers** | [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) · [`docs/running-the-benchmarks.md`](./docs/running-the-benchmarks.md) · [`PERF-RESULTS.md`](./PERF-RESULTS.md) |
| **Read the decisions** | [`docs/adr/`](./docs/adr/) · [`CONTEXT.md`](./CONTEXT.md) — the glossary |
| **Contribute** | [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) — note the contributor position in `COMMERCIAL.md` |
| **Evaluate for adoption** | [`docs/CONTINUITY.md`](./docs/CONTINUITY.md) · [`SECURITY.md`](./SECURITY.md) · [`CHANGELOG.md`](./CHANGELOG.md) |

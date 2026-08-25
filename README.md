# Latence Framework

**Point it at a folder of messy enterprise documents. Get back a PII-handled, embedded,
provenance-carrying RAG corpus and an evidence-linked knowledge graph — as plain Parquet files
you own.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](./pyproject.toml)
[![tests passing](https://img.shields.io/badge/tests-passing-brightgreen)](./scripts/verify-local.sh)
[![mypy --strict](https://img.shields.io/badge/mypy-strict-brightgreen)](./scripts/typecheck.sh)
[![status pre-1.0](https://img.shields.io/badge/status-pre--1.0-orange)](#status--maturity)

*(No CI or PyPI badge: no release has been cut and no distribution has ever been published, so
either badge would report a state that does not exist. The test badge deliberately carries **no
number** — a hardcoded count is stale one merge later; the exact count, the commit it was measured
at and the command that produces it are in [Status & maturity](#status--maturity), which is the
source of truth. See it for what is and is not proven.)*

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
dangerous files — and reports:

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
| `graph-nodes.parquet` | resolved canonical entities — plus PageRank centrality and community when the graph-features computer is wired |
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

The reasoning behind each of those choices is written down: **64 architecture decisions** in
[`docs/adr/`](./docs/adr/), and the glossary that fixes the vocabulary in
[`CONTEXT.md`](./CONTEXT.md).

---

## Extending it — and one place to configure it

**Swapping a component is a class and an entry point.** Capabilities are `typing.Protocol`s, so a
Provider satisfies one *structurally* — your class subclasses nothing from this framework, takes a
`dict` in its constructor, implements the one method the Capability declares, and registers itself
from your own package:

```toml
[project.entry-points."latence.providers"]
"parse.acme_ocr" = "acme_ocr.provider:AcmeParser"
```

Nothing in the framework is edited to accept it. Replace the OCR model with docling or your own,
GLiNER2 with a different extractor, the embedder, the graph store, or a query-time reranker — each is
a Provider under an existing Capability.

**The contract an AI coding agent needs is generated from the code, not written about it:**

```bash
latence contracts --capability parse    # typed signatures, carrier schemas, wiring, conformance cases
```

`docs/reference/stage-contracts.json` carries every Capability's Protocol signature, what the executor
delivers, the record schemas field by field, and the Providers that already fulfil it — and a drift
gate fails CI if it disagrees with the code, so it is safe to hand to a machine that will act on it.
The [conformance suite](./docs/CONFORMANCE.md) then decides admission: it guarantees a replacement
cannot break the end-to-end flow. It says nothing about whether yours is *better* — that is yours to
measure, with the [benchmark harness](./docs/running-the-benchmarks.md).

**One file configures the whole pipeline, and you do not write it:**

```bash
latence setup      # inspects the corpus, emits a validated latence.stack.yaml
latence process    # runs it
```

`setup` validates before writing — DAG structure, phase-boundary invariants, Capability satisfaction,
unknown config keys, sub-Provider slots — and reports a mistyped Provider as a typo with the nearest
real name rather than as an absent package. An installed third-party Provider appears as a choice in
the wizard; it never becomes a YAML edit.

→ **[`docs/extending.md`](./docs/extending.md)** — the full picture, including where a core edit *is*
required (a genuinely new Capability, by design) and why.

---

## What makes it different

Five claims, each with the thing that makes it checkable.

**1 · Labels are induced per chunk, not fixed corpus-wide.**
A global schema forces a bad trade: cap it and rare-but-critical types vanish; apply all of it to
every chunk and zero-shot precision collapses under labels that have nothing to do with the text.
GLiNER2 is a uni-encoder — labels and text share one transformer sequence — and the GLiNER
maintainers put the safe label ceiling at 20 (author) to 30 (docs). So each chunk is scored against
its own small set, and *consistency* is earned later, by a resolution stage that can see the whole
corpus. Stated honestly: the *cost* of more labels is measured (CPU latency +60% from 5 → 50 labels;
throughput −98.7% from 1 → 1,024), while the accuracy claim rests on a maintainer assertion plus a
confirmed label-order sensitivity — no controlled accuracy-vs-label-count ablation exists in the
literature, and this project does not pretend otherwise. The full evidence, including why the
`predefined_labels` seam is the wrong one, is in
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

### Enterprise serving, measured (2026-08)

The production question is not only "does the graph help" but "can you *serve* this — at scale,
on a real engine, with a corpus that changes". Each of those three was closed with a measurement
on real data, not a design note. The retrieval path has **zero LLM calls online**: every
query-time signal is a precomputed export column, a graph traversal, or a deterministic solver.

**Scale.** The end-to-end benchmark campaign pushed five public document-QA corpora through the
full enterprise pipeline on one CUDA pod: **60,766 documents → 297,548 chunks, 3.4M entity
mentions, 900,583 hyperedges, 731,266 PII findings**, per-dataset wall time 0.19–1.29 h; the
only LLM spend was optional schema induction at **$29.26 measured** for 247,944 calls
([`benchmark/sota-campaign/stages/s3-pipeline/metrics.json`](./benchmark/sota-campaign/stages/s3-pipeline/metrics.json)).

**Graph latency at scale — ~100×, zero quality loss.** At the largest export (677,257
hyperedges), a precomputed, transparently-cached hypergraph index plus tiered graph resolution
took every graph lane from seconds to tens of milliseconds per query — hypergraph lane median
3,570 ms → 37 ms (**96×**), traversal 435 ms → 3.9 ms (**111×**), rescue entry 3,733 ms → 33 ms
(**113×**); small corpora honestly gain 2–13×. Quality loss is not "unlikely", it is **counted**:
across **21,215 real gold queries × 3 lanes**, every ranked list, score and solver field compared
equal across old and new code paths — **0 diffs**
([`issues/wayfinder-enterprise-serving/tickets/01-report.md`](./issues/wayfinder-enterprise-serving/tickets/01-report.md)).

**Served on a real ANN engine — same recall.** The benched stack was stood up on a live Qdrant
(pinned compose, shipped loader, shipped backend) and compared against the brute-force reference
over **full query sets**: recall deltas within −0.0004 absolute on 12,576 2Wiki queries and
exactly 0.0 on MultiHop-RAG, at 42–83 ms median per served query through the whole composition
([the serving guide](./docs/guides/serving-qdrant.md) · raw results in
[`benchmark/serving/results/`](./benchmark/serving/results/)).

**Incremental churn — byte-identical, O(changed).** Add/update/remove on a 609-document corpus
through the full enterprise profile produced a final export **byte-identical to a from-scratch
rebuild** (sha256-equal files, bitwise-equal embeddings, 200/200 BM25 and 200/200 dense top-20
rankings identical on real queries), with an
add of 61 docs costing 8.4 min vs 97.4 min for the rebuild (**11.6×**)
([the incremental-ingest guide](./docs/guides/incremental-ingest.md) ·
[`03-report.md`](./issues/wayfinder-enterprise-serving/tickets/03-report.md)).

**In flight, stated as such:** the *recall* feature matrix of the graph-rescue architecture over
the full query sets of all 7 campaign datasets is still landing from the running pod sweep — the
committed [`S5 report`](./benchmark/sota-campaign/stages/s5-retrieval/REPORT.md) currently holds
gold-resolution for all 7 plus pre-rescue 50-query cells for 2, and this README quotes no recall
number from cells that have not landed.

---

## Status & maturity

Pre-1.0, and specific about it.

**Green, and reproducible on a laptop:**

- **5,139 tests pass, 36 skipped** — measured at commit `1101395` with
  `PYTHONHASHSEED=0 uv run pytest -q` (623 s, CPU, offline; a clean clone with the CI workspace
  setup). The count carries the commit it was taken at, the same convention `PERF-RESULTS.md`
  uses, because it grows as work lands — the command is the source of truth, not this line.
- **`mypy --strict` over src *and* tests** for every workspace package — `bash scripts/typecheck.sh`.
- **`bash scripts/verify-local.sh` exits 0** — the full merge gate: secret scan, ruff, strict types,
  the suite, provider conformance, end-to-end validation of 9 stack configs, the reproducible
  environment specs, the compatibility matrix, and 7 bake-off matrices.
- The **G1 profound-testing gate is signed off** — end-to-end on 3 messy corpora, security review,
  crash-resume and delta idempotency, a performance baseline, determinism:
  [`issues/gate/G1-RESULTS.md`](./issues/gate/G1-RESULTS.md), residuals included.
- Measured per-Stage throughput with hardware, date and a reproduction command on every row, plus an
  explicit *not yet measured* list: [`PERF-RESULTS.md`](./PERF-RESULTS.md).

**Not yet proven — and the repository will not go public until it is.**

The unmet claims are not buried in an issue tracker. They are a checklist,
[`RELEASE-GATES.md`](./RELEASE-GATES.md), whose numbers live on one page,
[`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md), where **every value is still a `PENDING:`
token**. Those tokens are load-bearing: `test_public_release_gate.py` inventories them on every
build, and under `LATENCE_RELEASE_CHECK=1` it **fails** while any remain. Publishing with an
unfilled table by forgetting is not possible.

What is outstanding:

- **The `BENCHMARKS-E2E` protocol page is still all `PENDING` tokens.** A large staged GPU run
  *has* now executed — the 60,766-document campaign above — but that page pins a specific
  reproduction protocol (named hardware, named corpus, per-stage VRAM), and its values are filled
  only by a run of that protocol, not by the nearest big number. The historical defects documented
  against the original plan (a TorchScript-JIT kill on hardware that never needed it,
  pod-specific micro-batch defaults) are recorded in
  [`issues/e2e-extended/`](./issues/e2e-extended/README.md); the scaling fixes the campaign
  forced are listed in its own metrics file.
- **SPLADE as a fused signal is implemented but unproven on real data.** The sparse index loads
  into a live Qdrant as a real named sparse vector (verified by the serving live-smoke), but no
  retrieval-quality number exists for it — the S5 matrix deliberately scoped it out. Graph
  traversal, by contrast, has graduated: it now runs as a measured production lane
  ([graph-rescue](./docs/guides/graph-rescue-retrieval.md)) rather than a reorder-only leg,
  with its full-query recall cells landing from the in-flight sweep.
- **The retrieval benchmark harness is built and running; its necessity-gate battery is not.**
  `benchmark/s5/` measures gold-resolution-audited recall/nDCG/chain-coverage with single-hop
  control datasets over full query sets. The eight necessity gates specified in
  [`DECISIONS.md`](./issues/e2e-extended/DECISIONS.md) §D9 (closed-book, per-edge necessity,
  DiRe, bridge leakage, adversarial distractors, …) remain **unimplemented**, so "the graph was
  *necessary*" is still an argued claim, not a gated one. The RFC numbers above are a real
  measurement of a narrower thing: one corpus, 90 generated questions, single-passage gold.
- **Scale is characterised, not eliminated.** The 100k / 1M-document envelopes are parked in
  `PERF-RESULTS.md` rather than previewed.
- **Multi-vector / MUVERA ship as real but experimental, off by default,** naming no checkpoint
  (ADR-0050).
- **Nothing has been released.** No public artifact exists and nothing is on PyPI. The signing and
  SBOM machinery is built and tested; it has not yet produced a release.

Bus factor, fork posture, support lifetime and CRA readiness are written down rather than implied:
[`docs/CONTINUITY.md`](./docs/CONTINUITY.md). Start there if you are evaluating this for adoption.

---

## Install

> **⚠️ Nothing is on PyPI, and nothing will be.** No release has been cut yet, so every
> `pip install latence-*` command below fails today with `ERROR: No matching distribution found` —
> and when releases start, they will be **signed GitHub Releases**
> ([ADR-0062](./docs/adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
> declined PyPI on the strength of the PolyForm licence; that rationale was superseded by
> [ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md) and the decision is reopened — the
> publish job stays disabled until it is retaken). Those names are *used by the workspace, not
> published* — see
> [Status & maturity](#status--maturity). **The install that works right now is from a clone.** The
> distribution map is kept below the fold because the split is the design (ADR-0016) and worth
> seeing before you clone, not because you can install it.

**Today — from a clone.** This is the same path the [quickstart](#60-second-quickstart) uses, and it
is exercised end-to-end on every CI run:

```bash
git clone https://github.com/ddickmann/latence-framework && cd latence-framework
uv sync                                          # CPU-first env: core, demo, console, retrieval
uv run latence --help                            # the CLI
uv pip install -e packages/latence-embedder-st   # add any opt-in Provider package by path
```

**After the first release** — the wheels and sdists are attached to the signed GitHub Release, so
you install a Release URL (`pip install <url-of-the-wheel>`), not a name from an index. The split
is a thin core plus one distribution per Provider family (ADR-0016), so you install only what you
use and the "runs on a laptop" promise holds. There is no `latence` meta-package; installing will
always be one or more of these. Python ≥ 3.11.

```bash
# NOT INSTALLABLE BY NAME — no release exists yet, and these are never published to an index
# (ADR-0062). Reproduced here as the map of the distribution split, not as commands to run.
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
the index. Whether these ever go to PyPI was answered "no" under the noncommercial licence
([ADR-0062](./docs/adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md));
that licence is gone ([ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md)) and the question is
open again. Nothing is published either way until it is decided.

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

**Apache License 2.0** (SPDX: `Apache-2.0`).

- **Any use is free, commercial included** — in a for-profit company, in a product you sell, as a
  hosted service, in consulting work. No agreement, no permission, nothing to buy. `COMMERCIAL.md`
  says so in one page for readers arriving from an older link.
- **You get an express patent licence** (Apache-2.0 §3) from every contributor, which terminates if
  you sue alleging the software infringes. The conditions on redistribution are §4's: ship the
  licence, reproduce [`NOTICE`](./NOTICE), and mark files you changed.
- [`LICENSE`](./LICENSE) governs and is the canonical Apache-2.0 text, byte for byte. The
  provenance record lives in [`NOTICE`](./NOTICE): the repository was Apache-2.0 from its first
  commit, carried PolyForm Noncommercial 1.0.0 from 2026-07-29 to 2026-08-15
  ([ADR-0055](./docs/adr/0055-relicence-to-polyform-noncommercial.md), superseded by
  [ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md)), and **was never published under either**
  — private since creation, zero forks, nothing on PyPI, no release cut, no tag, no CLA signed,
  every commit by the sole copyright holder — so no one holds a grant under a superseded state.
- The licence covers this project's own code only. Third-party dependencies, model weights and
  datasets keep their own terms, and several restrict commercial use independently
  ([`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md)). An outbound licence is never a shield
  against inbound terms.
- Not legal advice, and the project's statements around the licence have not been reviewed by
  counsel — a caveat written into the licence documents themselves rather than left for you to
  infer.

Copyright 2026 Dennis Dickmann (latence.ai). See [`NOTICE`](./NOTICE).

---

## Where to go next

| | |
|---|---|
| **Understand the whole thing** | [`docs/DEEP-DIVE.md`](./docs/DEEP-DIVE.md) — messy files → knowledge graph → multi-hop retrieval, with the reasoning |
| **Every stage, every model, in depth** | [`docs/anatomy/`](./docs/anatomy/index.md) — the 15-stage pipeline that produced the benchmark results, chapter by chapter: [ingest](./docs/anatomy/01-ingest-and-document-understanding.md) · [extraction & privacy](./docs/anatomy/02-extraction-and-privacy.md) · [knowledge graph](./docs/anatomy/03-corpus-to-knowledge-graph.md) · [enrichment & export](./docs/anatomy/04-enrichment-and-export.md) · [retrieval](./docs/anatomy/05-retrieval.md) |
| **Run it on your corpus** | [`docs/getting-started/index.md`](./docs/getting-started/index.md), then [`docs/TUTORIAL-fresh-pod-walkthrough.md`](./docs/TUTORIAL-fresh-pod-walkthrough.md) |
| **Read the results** | [`docs/getting-started/run-console.md`](./docs/getting-started/run-console.md) — the offline, read-only run console |
| **Keep a corpus current** | [`docs/guides/incremental-ingest.md`](./docs/guides/incremental-ingest.md) — churn without rebuild, equivalence proven byte-identical; hands-on: [add / update / retract / GDPR purge](./docs/TUTORIAL-fresh-pod-walkthrough.md#8-incremental-updates-add--update--retract--purge) |
| **Serve it** | [`docs/guides/serving-qdrant.md`](./docs/guides/serving-qdrant.md) — Export → live Qdrant with measured recall parity · [`docs/guides/graph-rescue-retrieval.md`](./docs/guides/graph-rescue-retrieval.md) — the zero-LLM-online query path |
| **Compare Providers** | [`docs/running-a-bakeoff.md`](./docs/running-a-bakeoff.md) · [`docs/running-a-stack.md`](./docs/running-a-stack.md) |
| **Check our numbers** | [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) (measured) · [`PERF-RESULTS.md`](./PERF-RESULTS.md) (measured) · [`docs/running-the-benchmarks.md`](./docs/running-the-benchmarks.md) (reproduce them) |
| **See what is still unproven** | [`RELEASE-GATES.md`](./RELEASE-GATES.md) · [`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md) — every value there is still a placeholder, and the build knows it |
| **Roadmap** | There isn't a dated one, and one will not be invented for the look of it. The plan *is* the unmet set: [`RELEASE-GATES.md`](./RELEASE-GATES.md) worked gate by gate, then the *Not yet proven* items in [Status & maturity](#status--maturity) — the E2E protocol run, SPLADE as a measured signal, the eight necessity gates, the 100k/1M envelopes. Order is gate order. No target dates: this is maintained by one person ([`docs/CONTINUITY.md`](./docs/CONTINUITY.md)), and a date would be a claim about capacity nobody has measured |
| **Read the decisions** | [`docs/adr/`](./docs/adr/) · [`CONTEXT.md`](./CONTEXT.md) — the glossary |
| **Contribute** | [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`CLA.md`](./CLA.md) — every merged contribution needs it signed · [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) — the CLA predates the relicence and may be replaced by a DCO; see `COMMERCIAL.md` § *Contributions* |
| **Evaluate for adoption** | [`docs/CONTINUITY.md`](./docs/CONTINUITY.md) · [`SECURITY.md`](./SECURITY.md) · [`CHANGELOG.md`](./CHANGELOG.md) |

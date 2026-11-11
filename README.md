<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/logo-light.png">
    <img alt="Latence" src="./docs/assets/logo-light.png" width="360">
  </picture>
</p>

<p align="center">
  <strong>Documents in. A corpus that carries its provenance and a knowledge graph that carries<br>
  its evidence out, as plain files you own.</strong>
</p>

<p align="center">
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-00CFD1"></a>
  <a href="./pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-00CFD1"></a>
  <a href="https://github.com/ddickmann/latence/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ddickmann/latence/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#status"><img alt="status pre-1.0" src="https://img.shields.io/badge/status-pre--1.0-orange"></a>
</p>

> ### Using a coding agent?
> Point it at **[`llms.txt`](./llms.txt)**, or hand it
> **[`docs/agent-briefing.md`](./docs/agent-briefing.md)** directly. The briefing is written for
> LLM agents: the mental model, every Provider available at each Stage, the pipeline YAML
> contract, the CLI, the output schema, and the specific things that are easy
> to get wrong.

---

## What it is

Latence is a Python framework that reads a folder of documents and writes two things as plain
files. The first is a corpus in which every chunk carries its source page and its character
offsets. The second is a knowledge graph in which every edge carries the text it was extracted
from. A Redaction Stage masks PII, and the corpus export writes the masked text.

The framework names no model. Each Stage depends on a narrow interface, a **Capability**, and a
**Provider** package supplies the model behind it, so changing a model is a change of one config
key. Every Stage has a deterministic reference Provider that runs on a laptop with no GPU and no
network.

This page describes what the code does today. The project has been re-scoped, and that work is
not built: see [Where this is going](#where-this-is-going).

---

## Quick start

CPU-only. No GPU, no cloud, no credentials, no network calls at run time.

```bash
git clone https://github.com/ddickmann/latence && cd latence
uv sync
uv run latence-demo
open /tmp/latence-demo/_latence/runs/demo/export/index.html
```

The demo ingests a bundled corpus that is messy on purpose: text and markdown files, a real PDF,
and four planted dangerous files. It reports:

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

The zip bomb, the oversized file and the `.pdf` that is not a PDF are quarantined before parsing.
The prompt-injection document is flagged and kept in the corpus with its marker, so a consumer
can exclude it. Each of the 33 graph edges carries the sentence it came from. `latence-demo`
exits non-zero if any of that stops being true.

**Swapping a model is one line.** The pipeline does not change:

```bash
uv pip install -e packages/latence-ner-gliner   # a learned NER Provider (pulls torch)
uv run latence-demo --entity-provider entity.gliner
```

Without that install the swap exits non-zero and names the package to install. It does not fall
back to the default.

**Then run it on your own documents:**

```bash
uv run latence setup      # guided wizard: inspects your files, writes a validated stack config
uv run latence process    # runs it
```

---

## What is built

Each line below is wired end to end and covered by the test suite. "Built" does not mean proven
at scale; [Status](#status) says what is measured. [`ROADMAP.md`](./ROADMAP.md) has the longer
version of this list and the known limitations.

- **Sources.** A local folder, and SharePoint or OneDrive.
- **Screening.** Intake screening quarantines zip bombs, oversized files and type-spoofed files
  before anything parses them. Content screening flags prompt injection and harmful content on
  the chunk and, by default, keeps the chunk in the corpus with the marker.
- **Parsing.** Plain text, PDFs with a text layer, office files and images through a renderer,
  and OCR models that run in-process or behind an HTTP endpoint. Each parser emits a page map,
  so a character offset resolves to a source page.
- **Extraction.** Entity extraction, relation extraction (pattern-based, model-based, fused with
  entity extraction, or LLM-backed), and optional schema induction.
- **Redaction.** Rule-based and learned PII detection. A deterministic check for `ssn`,
  `credit_card` and `iban` runs beneath every redaction Provider, and no choice of Provider
  bypasses it.
- **Resolution and graph.** Consolidation of the type vocabulary, entity disambiguation with a
  merge log, graph assembly with Evidence on every edge, and optional graph completion whose
  predicted edges are marked as inferred.
- **Running it.** A local Runner with checkpoint and resume, an Airflow adapter
  (`latence-runner-airflow`) that runs the same Stages as a DAG, incremental updates with
  retraction and purge, one Storage seam for `file://`, `s3://`, `gs://` and `abfs://`, a guided
  setup, a read-only run console, and a tuner for the performance knobs of the heavy Providers.
- **Checking it.** A conformance suite that runs over every registered Provider, a gold-set
  harness that scores a run against your own annotated documents, and a benchmark harness.

The [Provider catalog](./docs/providers/catalog.md) lists every Provider with its verified
weights and code licence.

### What a run writes

Files on your Storage. Nothing is hidden in a proprietary index.

| Artifact | What it is |
|---|---|
| `corpus.parquet` / `corpus.jsonl` | the processed corpus: one row per chunk, with PII-masked text and, when an Embedder is wired, a dense embedding |
| `graph-nodes.parquet` | resolved canonical entities |
| `graph-edges.parquet` | extracted and reference relations, each edge with its Evidence |
| `graph.ttl` / `graph.graphml` | the same graph for RDF and graph tooling |
| `quality-report.json` | per-Stage metrics, graph statistics and contract-completeness checks. The `latence` commands also write a `quality-report.md` rendering |
| `manifest.json` | the pipeline that produced this run, with credentials redacted |

`char_start` and `char_end` are the chunk's exact half-open span in the parsed document.
`page_start` and `page_end` resolve through the chunk's own page slice, not an inherited range.
An `offset_map` corrects for markup stripped before extraction, so an entity mention resolves to
its original offset. The record schema is in
[the output contract](./docs/agent-briefing.md#7--what-a-run-produces).

### Architecture in one paragraph

A **Stage** is a typed transform over records. It depends only on a **Capability**, a narrow
interface such as `Parser`, `EntityExtractor` or `Embedder`, and the core never names a model. A
**Provider** fulfils a Capability: an in-process model, a served endpoint, a cloud service.
Providers live in their own installable packages, so nothing pulls torch into your environment
unless you ask for it. A **Runner** executes a Pipeline. **Storage** is one fsspec seam. Every
record carries Provenance and Classification, validated at every Stage boundary. The reasoning
behind each choice is written down: **73 architecture decisions** in [`docs/adr/`](./docs/adr/),
and the glossary that fixes the vocabulary in [`CONTEXT.md`](./CONTEXT.md).

---

## Status

Pre-1.0. One maintainer. Nothing has been released.

**Checked on every change, and reproducible on a laptop:**

- The full suite passes with `PYTHONHASHSEED=0`, and `mypy --strict` passes over source and
  tests for every package.
- The merge gate is `bash scripts/verify-local.sh`: secret scan, lint, types, the suite, Provider
  conformance, eight end-to-end stack validations, the reproducible environment specs, the
  compatibility matrix and seven bake-off matrices.
- A run is deterministic. `latence stack validate` runs a stack under a second run id and
  requires a byte-identical export, then resumes the first run and requires the same.

**On record, with its date:** the G1 testing gate was signed off on 2026-07-07, on the tree of
that date: end to end on three messy corpora, a security review, crash-resume with a child
process killed in the middle of a checkpoint write, delta idempotency, a performance baseline
and determinism. The record lists its own gaps:
[`docs/evidence/G1-RESULTS.md`](./docs/evidence/G1-RESULTS.md).

**Measured:** per-Stage throughput, with hardware, date and a reproduction command on every row
and an explicit list of what is not measured, is in [`PERF-RESULTS.md`](./PERF-RESULTS.md).
Quality suites are in [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md).

**Not proven:**

- **The end-to-end benchmark protocol has not been run.**
  [`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md) pins a reproduction (named hardware,
  per-stage wall-clock and VRAM, a device-honesty check, link-prediction precision), and every
  value on it is still a `PENDING:` token. A large staged GPU run has been executed, but it is
  not that protocol, and a number from a different corpus or card would be worse than a missing
  one.
- **Extraction and resolution quality have no current measurement.** The resolvers merge on
  rules and configured thresholds; a merge carries a confidence, not a calibrated probability.
  The gold labels the repository holds are partly generated by a language model.
- **Scale is characterised, not eliminated.** The 100k and 1M-document envelopes are parked in
  `PERF-RESULTS.md`, not previewed.
- **Query-time retrieval is out of scope.** The retrieval tooling was removed
  ([ADR-0068](./docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)).
  The framework writes the resolved graph and the processed corpus, and a consumer brings its
  own retrieval engine.
- **Nothing has been released.** The signing and SBOM machinery is built and tested; it has not
  produced a release. [`RELEASE-GATES.md`](./RELEASE-GATES.md) is the checklist, and it is
  machine-checked: `test_public_release_gate.py` fails under `LATENCE_RELEASE_CHECK=1` while any
  gate is unmet.

Bus factor, fork posture, support lifetime and CRA readiness are written down:
[`docs/CONTINUITY.md`](./docs/CONTINUITY.md). Start there if you are evaluating this for adoption.

---

## Where this is going

**Nothing in this section is built.** It states decisions that are recorded and accepted. The
code described above does not implement them.

The project was re-scoped in November 2026
([ADR-0067](./docs/adr/0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md)).
It is to become an engine that builds a traceable semantic and ontology layer: canonical
concepts, their names and aliases, the relations between them, and the bindings that tie them to
physical data. Turning documents into a corpus and a graph stays, as one input path of that
engine. The name stays Latence.

What the decisions fix:

- **A ledger is the source of truth.** Statements harvested from sources, and the decisions made
  on top of them, are kept append-only. The layer anyone reads is rebuilt by replaying them, so
  a human decision survives every rebuild
  ([ADR-0069](./docs/adr/0069-the-semantic-layer-is-a-replay-of-an-append-only-ledger-of-assertions-and-decisions.md)).
- **Two levels with different authority.** The engine may accept a merge or a link between
  instances by itself only inside a calibrated accept band. Classes, properties and constraints
  are only ever proposals that a human approves (ADR-0069).
- **Warehouse metadata becomes an input.** Unity Catalog comes first. Row values are compared
  inside the customer's own workspace and do not leave it
  ([ADR-0071](./docs/adr/0071-warehouse-metadata-enters-the-pipeline-as-sources-and-row-values-are-matched-inside-the-workspace.md)).
- **In a customer's workspace the engine runs as jobs and keeps its state in Delta tables.** The
  Runner for that is not shipped; the local Runner stays for standalone use
  ([ADR-0072](./docs/adr/0072-on-databricks-the-engine-runs-as-jobs-in-the-workspace-and-keeps-its-state-in-delta.md)).
- **The schema level is exported in W3C standards:** OWL 2 QL, SHACL, SKOS and PROV-O
  ([ADR-0073](./docs/adr/0073-the-schema-level-is-induced-by-voting-signals-validated-on-data-and-exported-in-w3c-standards.md)).
- **One declared seam for a consuming agent harness:** a reviewable proposal file per use case.
  Everything else reaches a consumer through the catalog
  ([ADR-0070](./docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md)).

[`ROADMAP.md`](./ROADMAP.md) lists this work in priority order, next to what the current code
lacks for each item.

---

## Install

**From a clone.** This is the path the quick start uses, and CI exercises it end to end on every
run. Python ≥ 3.11.

```bash
git clone https://github.com/ddickmann/latence && cd latence
uv sync                                          # CPU-first env: core, demo, console, benchmark
uv run latence --help
uv pip install -e packages/latence-embedder-st   # add any opt-in Provider package by path
```

**After the first release**, wheels are attached to a signed GitHub Release, so you install a
release URL rather than a name from an index. Nothing is on PyPI, and `pip install latence-core`
fails today with `ERROR: No matching distribution found`. The split is a thin core plus one
distribution per Provider family, so you install only what you use. There is no `latence`
meta-package:

```bash
pip install latence-core                # contracts, Capabilities, local Runner, Storage, CLI
pip install latence-parser-pdfplumber   # + one package per Provider you actually want
```

The full map is in [Install](./docs/getting-started/index.md).

---

## Licence & security

Apache-2.0 ([`LICENSE`](./LICENSE)). Commercial use is granted by the licence; no separate agreement
is needed ([`COMMERCIAL.md`](./COMMERCIAL.md)). Reference Providers ship **permissive weights only**;
restricted-license models are opt-in, and every Provider declares its verified weights-and-code
licence ([`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md)). The project's licence statements
have not been reviewed by counsel.

Releases are signed GitHub Releases carrying a per-release CycloneDX SBOM and a keyless Sigstore
signature; how a version is cut and how you verify what you downloaded is in
[`docs/RELEASING.md`](./docs/RELEASING.md). To report a vulnerability, read
[`SECURITY.md`](./SECURITY.md) — please do not open a public issue.

---

## Where to go next

| You want | Read |
|---|---|
| To point an AI agent at this repo | [`llms.txt`](./llms.txt) · [`docs/agent-briefing.md`](./docs/agent-briefing.md) |
| A worked path, end to end | [Quickstart](./docs/quickstart.md) · [Tutorials](./docs/tutorials/index.md) |
| The mechanics, Stage by Stage | [Pipeline anatomy](./docs/anatomy/index.md) |
| What is built, what is next, what is weak | [`ROADMAP.md`](./ROADMAP.md) |
| Why it is built this way | [The decision log](./docs/adr/) · [`CONTEXT.md`](./CONTEXT.md) |
| To add your own model | [Authoring a Provider](./docs/providers/README.md) |
| To contribute | [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`AGENTS.md`](./AGENTS.md) |

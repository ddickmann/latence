<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/logo-light.png">
    <img alt="Latence" src="./docs/assets/logo-light.png" width="360">
  </picture>
</p>

<p align="center">
  <strong>Point it at a folder of messy enterprise documents. Get back a PII-handled, embedded,<br>
  provenance-carrying RAG corpus and an evidence-linked knowledge graph — as plain Parquet files you own.</strong>
</p>

<p align="center">
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-00CFD1"></a>
  <a href="./pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-00CFD1"></a>
  <a href="https://github.com/ddickmann/latence/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ddickmann/latence/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#status"><img alt="status pre-1.0" src="https://img.shields.io/badge/status-pre--1.0-orange"></a>
</p>

> ### 🤖 Using a coding agent?
> Point it at **[`llms.txt`](./llms.txt)**, or hand it
> **[`docs/agent-briefing.md`](./docs/agent-briefing.md)** directly. The briefing is written for
> LLM agents: the mental model, every Provider available at each Stage, the pipeline YAML
> contract, the CLI, the output schema, and the specific things that are easy
> to get wrong. It is structured so an agent can go from "never seen this repo" to a working
> pipeline on your documents without reading the source.

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
git clone https://github.com/ddickmann/latence && cd latence
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

Without that install the swap exits non-zero naming the package to install — it never degrades
quietly to the default.

**Then run it on your own corpus:**

```bash
uv run latence setup      # guided wizard: inspects your files, emits a validated stack config
uv run latence process    # runs it
```

---

## What it produces

Files on your Storage (`file://`, `s3://`, `gs://`, `abfs://` — same code, one config key). Nothing
is hidden in a proprietary index.

| Artifact | What it is |
|---|---|
| `corpus.parquet` / `corpus.jsonl` | the processed corpus — one row per chunk, with PII-masked text and, when an Embedder is wired, a dense embedding |
| `graph-nodes.parquet` | resolved canonical entities |
| `graph-edges.parquet` | extracted + reference relations, **every edge carrying its Evidence** |
| `graph.ttl` / `graph.graphml` | the same graph for RDF and graph-viz tooling |
| `quality-report.json` + `quality-report.md` | the machine + human artifact that substantiates "AI-ready" |
| `manifest.json` | the exact pipeline that produced this run, with credentials redacted |

Provenance is not decoration: `char_start`/`char_end` are the chunk's exact half-open span in the
parsed document, `page_start`/`page_end` resolve through the chunk's **own** page slice rather than
an inherited range, and an `offset_map` corrects for markup stripped before extraction — so an
entity mention resolves to its true original offset, not a lower bound. The full record schema is in
[the output contract](./docs/agent-briefing.md#7--what-a-run-produces).

---

## Architecture in one paragraph

A **Stage** is a typed transform over records. It depends only on a **Capability** — a narrow
interface such as `Parser`, `EntityExtractor`, `Embedder` — and the core never names a model. A
**Provider** fulfils a Capability: an in-process model, a served endpoint, a cloud service. Providers
live in their own installable packages, so swapping the OCR model, the induction LLM or the embedder
is a config key rather than a refactor, and nothing pulls torch into your environment unless you ask
for it. A **Runner** executes a Pipeline: the built-in local Runner runs the whole spine on a laptop
with checkpoint/resume, and a shipped Airflow adapter (`latence-runner-airflow`) runs the identical
Stages as a DAG. **Storage** is one fsspec seam. Every record carries Provenance and Classification,
validated at every Stage boundary.

Each Stage ships a deterministic, dependency-light **CPU-first reference Provider**, so the whole
pipeline runs offline before you choose a single model. The
[Provider catalog](./docs/providers/catalog.md) lists every one with its verified weights-and-code
licence. The reasoning behind each choice is written down: **71 architecture decisions** in
[`docs/adr/`](./docs/adr/), and the glossary that fixes the vocabulary in
[`CONTEXT.md`](./CONTEXT.md).

---

## What is measured

Measured per-Stage throughput, with hardware, date and a reproduction command on every row — plus an
explicit *not yet measured* list — is in [`PERF-RESULTS.md`](./PERF-RESULTS.md); measured quality
suites are in [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md).

---

## Status

Pre-1.0, and specific about it.

**Proven, and reproducible on a laptop:** the full suite green with `PYTHONHASHSEED=0`,
`mypy --strict` over src *and* tests for every package, `bash scripts/verify-local.sh` exiting 0
(secret scan, lint, types, suite, Provider conformance, eight end-to-end stack validations, the
reproducible environment specs, the compatibility matrix and seven bake-off matrices), and
byte-identical reruns — including a child process `SIGKILL`ed mid-checkpoint-write, resumed, and
producing an export identical to an uninterrupted run. The **G1 profound-testing gate is signed
off** — end-to-end on three messy corpora, security review, crash-resume and delta idempotency, a
performance baseline and determinism, with residuals included:
[`docs/evidence/G1-RESULTS.md`](./docs/evidence/G1-RESULTS.md).

**Not yet proven, stated plainly:**

- **The end-to-end benchmark protocol has not been run.** [`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md)
  pins a specific reproduction (named hardware, per-stage wall-clock and VRAM, a device-honesty
  check, link-prediction precision) and **every value on it is still a `PENDING:` token**. A large staged GPU run has
  executed, but it is not *that* protocol, and a number from a different corpus or card would be
  worse than a missing one.
- **The graph's *necessity* is argued, not gated.** The eight necessity gates (closed-book, per-edge
  necessity, DiRe, bridge leakage, adversarial distractors, …) are unimplemented.
- **Scale is characterised, not eliminated.** The 100k / 1M-document envelopes are parked in
  `PERF-RESULTS.md` rather than previewed.
- **Query-time retrieval is out of scope.** The retrieval tooling was removed
  ([ADR-0068](./docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md));
  the framework produces the resolved graph and the processed corpus, and a consumer brings its
  own retrieval engine.
- **Nothing has been released.** The signing and SBOM machinery is built and tested; it has not yet
  produced a release. [`RELEASE-GATES.md`](./RELEASE-GATES.md) is the checklist, and it is
  machine-checked — `test_public_release_gate.py` fails under `LATENCE_RELEASE_CHECK=1` while any
  gate is unmet.

Bus factor, fork posture, support lifetime and CRA readiness are written down rather than implied:
[`docs/CONTINUITY.md`](./docs/CONTINUITY.md). Start there if you are evaluating this for adoption.

---

## Install

**From a clone** — the path the quickstart uses, exercised end-to-end on every CI run. Python ≥ 3.11.

```bash
git clone https://github.com/ddickmann/latence && cd latence
uv sync                                          # CPU-first env: core, demo, console, benchmark
uv run latence --help
uv pip install -e packages/latence-embedder-st   # add any opt-in Provider package by path
```

**After the first release**, wheels are attached to a signed GitHub Release, so you install a
release URL rather than a name from an index — nothing is on PyPI, and `pip install latence-core`
fails today with `ERROR: No matching distribution found`. The split is a thin core plus one
distribution per Provider family, so you install only what you use and the "runs on a laptop"
promise holds. There is no `latence` meta-package:

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
| Why it is built this way | [The decision log](./docs/adr/) · [`CONTEXT.md`](./CONTEXT.md) |
| To add your own model | [Authoring a Provider](./docs/providers/README.md) |
| To contribute | [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`AGENTS.md`](./AGENTS.md) |

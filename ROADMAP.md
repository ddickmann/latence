# Roadmap

What exists today, what comes next, and — most usefully — what is known to be weak. No dates and no
version numbers on purpose: this is a pre-1.0 project with one maintainer. Two documents carry what
this page does not restate: [`README.md`](./README.md)'s **Status** section separates what is proven
from what is not, and [`docs/adr/`](./docs/adr/) records *why* each architectural decision was taken.
Planning happens in the open from here: **GitHub Issues** for concrete work, **Discussions** for
direction.

The direction changed with two decisions.
[ADR-0067](./docs/adr/0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md)
re-scopes the project: from a framework that turns documents into AI-ready data, into an engine
that builds a traceable semantic and ontology layer.
[ADR-0068](./docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)
removed the query-time retrieval tooling. [What is built](#what-is-built) describes the tree after
that removal. [What is next](#what-is-next) is the re-scope, and none of it is built.

## What is built

Everything below runs today, is exercised by the test suite, and is reachable from the CLI or the
library API. "Built" means *wired end to end and covered by tests*, not *proven at scale* — see
[Known limitations](#known-limitations).

**Getting documents in.** A Source seam with a local-folder reference implementation and a
SharePoint/OneDrive connector that has no third-party dependency. Intake screening quarantines
dangerous files (zip bombs, oversized files, type spoofing) *before* anything parses them. Content
screening flags prompt-injection and harmful-content hits on the chunk and, by default, keeps the
chunk in the corpus with the marker rather than dropping it. Parsers cover plain text, PDFs
(text-layer and page-aware), a universal renderer free of AGPL code, and VLM-OCR models —
in-process or behind an HTTP endpoint — and each emits a page map, so a character offset resolves
back to a source page.

**Extraction and privacy.** Zero-shot entity extraction; relation extraction that is
pattern-based, model-based, fused entity+relation, or LLM-backed; schema induction, per chunk by
a small LLM or from an operator-declared candidate type space with no network; and PII redaction
with a rule-based floor no model choice can bypass. The corpus export writes the masked text
unless an option named `unsafe_unmasked_corpus` is set.

**From chunks to a knowledge graph.** Offset-preserving chunking, corpus profiling,
canonicalisation of the type vocabulary, entity disambiguation with an audited merge log (a merge
below the confidence policy is recorded and not applied), canonical graph assembly with per-edge
Evidence, and optional graph completion: a learned inductive link-prediction Provider alongside a
deterministic closure reference. A predicted edge is marked as inferred and stays distinct from
the asserted ones.

**What a run leaves behind.** Parquet and JSONL corpus files, graph node and edge tables, RDF
Turtle and GraphML exports, a machine- and human-readable Quality Report, and a manifest of the
pipeline that produced the run with credentials redacted. The corpus Parquet can also be
registered as a table in an Iceberg REST catalog. When an Embedder is wired, each corpus row
carries a dense embedding. Every record carries Provenance and Classification, validated at each
Stage boundary.

**Running it.** A local Runner with checkpoint/resume, an Airflow adapter running the identical
Stages as a DAG, incremental ingest with delta detection, retraction and purge, one fsspec
Storage seam covering local, S3, GCS and Azure, a read-only run console for non-engineers, a
guided setup that writes a validated stack config, a tuner for the heavy Providers' performance
knobs, and reproducible environment specs.

**Measuring it.** A gold-set harness that scores a run's entities and relations against an
adopter's own annotated documents, a labelling tool for question gold sets with supporting-fact
spans over an exported corpus, a benchmark harness with per-Stage suites, and a staged benchmark
campaign that acquires public corpora behind a licence gate.

**Making "model-agnostic" checkable.** Every Provider declares its compute footprint and verified
weights-and-code license, and a conformance suite runs over *every* registered Provider — typed
records, graceful failure on adversarial input, license recorded, determinism or a documented
reason, device honoured, no PII leak. Blessed CPU and GPU stacks are validated end to end, and
per-Stage bake-offs compare Providers on one corpus
([`docs/providers/catalog.md`](./docs/providers/catalog.md)).

## What is next

Rough priority order, not a schedule. Nothing here is committed to a release, and nothing here is
built: each item is an intention recorded in an accepted decision.

1. **Generalise the contracts.** A structured locator next to character offsets, and an actor on
   every decision, so that a statement can come from a catalog object as well as from a span of
   text. This changes the record contracts
   ([ADR-0067](./docs/adr/0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md)).
2. **Make the semantic layer a replay of a ledger.** Statements harvested from sources and the
   decisions made on top of them are kept append-only; the layer anyone reads is rebuilt by
   replaying them, so a human decision survives every rebuild and delta run
   ([ADR-0069](./docs/adr/0069-the-semantic-layer-is-a-replay-of-an-append-only-ledger-of-assertions-and-decisions.md)).
3. **Read warehouse metadata as an input.** Catalog metadata, metric views, lineage and existing
   glossaries enter the same Pipeline as new Source kinds and meet the text path in one
   resolution phase. Databricks Unity Catalog comes first. Row values are compared inside the
   customer's own workspace and do not leave it
   ([ADR-0071](./docs/adr/0071-warehouse-metadata-enters-the-pipeline-as-sources-and-row-values-are-matched-inside-the-workspace.md)).
4. **Calibrate resolution.** The engine accepts an instance-level merge or link by itself only
   inside a calibrated accept band. Everything outside that band is a proposal an expert accepts
   or rejects, and a human decision outranks the engine (ADR-0069). The bars a release must meet
   are Gate 6 of [`RELEASE-GATES.md`](./RELEASE-GATES.md).
5. **Re-select the extraction stack.** The text path stays. Its extraction Providers are chosen
   again by measurement against a recorded baseline (ADR-0067).
6. **Induce the schema level and export it in standards.** Classes, hierarchy, properties and
   constraints are proposed from several explainable signals, a proposed constraint is checked
   against the data before a human sees it, and a human approves every schema-level element. The
   export targets are OWL 2 QL, SHACL, SKOS and PROV-O, with opaque stable identifiers
   ([ADR-0073](./docs/adr/0073-the-schema-level-is-induced-by-voting-signals-validated-on-data-and-exported-in-w3c-standards.md)).
7. **A Databricks Jobs Runner.** The same Pipeline executed as jobs in the customer's workspace,
   with the ledger, the published results and the review state in Delta tables. The local Runner
   stays for standalone use, development and CI
   ([ADR-0072](./docs/adr/0072-on-databricks-the-engine-runs-as-jobs-in-the-workspace-and-keeps-its-state-in-delta.md)).
8. **An expert review surface.** One web app with two modes: deployed inside the workspace in
   production, run locally against files for standalone use. It shows each proposal with the
   evidence behind it (ADR-0072).
9. **One declared seam for a consuming agent harness.** A reviewable proposal file per use case;
   everything else reaches a consumer through the catalog
   ([ADR-0070](./docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md)).
10. **Run the pinned end-to-end benchmark protocol.** [`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md)
    fixes the hardware and the method; every value on it is still a placeholder, and a number
    from a different corpus would be worse than a missing one.

## Known limitations

Read from the code and from the repository's own records. This is where an evaluator should spend
their time.

- **The contracts assume files and character offsets.** Provenance requires a source URI, a file
  name and a file type. Nothing in the tree reads a warehouse: the two Sources are a local folder
  and SharePoint/OneDrive.
- **There is no schema level.** The type vocabulary is a flat list of canonical types with their
  aliases. It has no hierarchy, no properties and no constraints.
- **No human decision is persisted.** The merge log records what the resolver decided. There is
  no record type for an expert's accept or reject.
- **Resolution is not calibrated.** The resolvers merge on rules and configured thresholds. A
  merge decision carries a confidence, not a calibrated probability, and no accept band exists.
- **The default resolver matches by surface form.** `disambiguation.exact_surface` merges on
  exact normalised-surface equality and configured aliases. It will not unify two different
  written forms of one entity without an alias. The cascade adds acronym, substring and embedding
  rungs.
- **The RDF export uses a private vocabulary.** `graph.ttl` is written with an `lg:` prefix under
  a `urn:` base that does not dereference. It is valid RDF; it is not OWL, SHACL, SKOS or PROV-O.
- **Extraction quality has one dated record and no current measurement.** In August 2026 a
  language model judged 250 hyperedges of the earlier pipeline: 54% correct, 32% partial, 14%
  incorrect (`benchmark/official-data/PROVENANCE.md`). That is a record of the earlier pipeline's
  output on one corpus, and the same file lists the labels as pending a human spot-check.
- **Evaluation gold labels are partly LLM-generated.** The enterprise-corpus gold set and that
  judged sample were produced by a language model, with provenance recorded. Any number derived
  from them inherits the judge's errors.
- **The end-to-end benchmark protocol has not been run.** Every value on
  [`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md) is a placeholder.
- **Scale is characterised, not eliminated.** The 100k and 1M-document envelopes are recorded as
  open in [`PERF-RESULTS.md`](./PERF-RESULTS.md) rather than previewed.
- **Nothing has been released yet.** Signing and SBOM machinery is built and tested but has never
  produced a release; [`RELEASE-GATES.md`](./RELEASE-GATES.md) is the machine-checked checklist.

## Explicitly out of scope

So nobody files it, and nobody plans around it arriving:

- **Query-time retrieval.** The retrieval tooling was removed (ADR-0068). The project produces the
  resolved graph and the processed corpus; a consumer brings its own retrieval engine. Embedders
  stay only as a signal the resolution phase uses.
- **A hosted service.** This is a library and a CLI that write files you own.
- **A second source of truth beside the warehouse catalog.** The engine is to bind to what the
  catalog defines and send any change to it upstream as a proposal a human applies (ADR-0069).
- **Accepting schema-level elements without a human.** Decided against, on the measured accuracy
  of ontology learning (ADR-0069).
- **A model that accepts by itself.** A model's verdict is a feature of the score or a suggestion
  shown to the expert (ADR-0069).
- **A graph database or a query language.** The graph is exported as files.
- **Bundled model weights, or a blessed "best" model.** The framework names Capabilities, never
  models. Restricted-license models are opt-in and never a default.
- **Answer generation.** What generates an answer from the layer is the consumer's.

## How to influence it

- **File a GitHub Issue** for a bug, a Provider you need, or a concrete proposal — held to the same
  bar as maintainer work ([`issues/DEFINITION-OF-DONE.md`](./issues/DEFINITION-OF-DONE.md)).
- **Open a GitHub Discussion** for "should this project do X" or "how do I fit this to my stack".
- **Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before opening a pull request** — the quality bar, the
  vocabulary the codebase uses, and how to add a Provider without touching core.
- **The most valuable contribution is a measurement.** Turning one of the limitations above from an
  assertion into a number, on a corpus that is not the maintainer's, moves this project further than
  a feature does.

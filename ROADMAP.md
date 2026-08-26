# Roadmap

What exists today, what comes next, and — most usefully — what is known to be weak. No dates and no
version numbers on purpose: this is a pre-1.0 project with one maintainer. Two documents carry what
this page does not restate: [`README.md`](./README.md)'s **Status** section separates what is proven
from what is not, and [`docs/adr/`](./docs/adr/) records *why* each architectural decision was taken.
Planning happens in the open from here: **GitHub Issues** for concrete work, **Discussions** for
direction.

## What is built

Everything below runs today, is exercised by the test suite, and is reachable from the CLI or the
library API. "Built" means *wired end to end and covered by tests*, not *proven at scale* — see
[Known limitations](#known-limitations).

**Getting documents in.** A Source seam with a local-folder reference implementation and a
dependency-free SharePoint/OneDrive connector. Intake screening quarantines dangerous files (zip
bombs, oversized dumps, type spoofing) *before* anything parses them, and flags prompt-injection
content into the corpus rather than silently dropping it. Parsers cover plain text, PDFs
(text-layer and page-aware), a license-clean universal renderer, and several VLM-OCR models —
in-process or behind an HTTP endpoint — and every one emits a page map, so a character offset
resolves back to a source page.

**Extraction and privacy.** Zero-shot entity extraction; relation extraction that is
pattern-based, model-based, fused entity+relation, or LLM-backed; schema induction; and PII
redaction with a rule-based floor no model choice can bypass. Redaction runs before anything
downstream sees the text, and derived artifacts inherit the redacted variant by construction.

**From chunks to a knowledge graph.** Offset-preserving chunking, corpus profiling, entity
disambiguation, canonical graph assembly with per-edge Evidence, learned graph completion (an
inductive link-prediction Provider alongside a deterministic closure reference), and context
enrichment that projects graph structure back onto chunks.

**What a run leaves behind.** Parquet and JSONL corpus files, graph node/edge tables, RDF and
GraphML exports, BM25 term statistics, a machine- and human-readable Quality Report, and a
manifest of the pipeline that produced the run with credentials redacted. Every record carries
Provenance and Classification, validated at each Stage boundary.

**Retrieval quality without a retrieval engine.** A stateless query-time library that fuses ranked
lists, reranks with a cross-encoder, packs a token-budgeted context window, and expands over the
knowledge graph (1-hop over columns, or multi-hop over the exported Parquet with no graph database).
At index time it emits a structured SPLADE sparse signal and optional multi-vector/FDE encodings. An
MCP server exposes all of it to agents, with agent-facing copy loaded from operator config rather
than baked in. It holds no index and runs no first-stage search — that always stays the adopter's
engine ([`docs/retrieval-tooling.md`](./docs/retrieval-tooling.md)).

**Hypergraph evidence and chain-of-evidence selection.** Sentence-window hyperedges emitted as an
engine-agnostic sidecar, query-aware confidence propagation, and a greedy selector that assembles a
token-budgeted chain of evidence. No LLM call in indexing or the online path.

**Running it.** A local Runner with checkpoint/resume (verified against a process killed
mid-checkpoint-write), an Airflow adapter running the identical Stages as a DAG, incremental ingest
with delta detection and retraction/purge, one fsspec Storage seam covering local, S3, GCS and
Azure, a read-only run console for non-engineers, and a benchmark harness.

**Making "model-agnostic" checkable.** Every Provider declares its compute footprint and verified
weights-and-code license, and a conformance suite runs over *every* registered Provider — typed
records, graceful failure on adversarial input, license recorded, determinism or a documented
reason, device honoured, no PII leak. Blessed CPU and GPU stacks are validated end to end, and
per-Stage bake-offs compare Providers on one corpus
([`docs/providers/catalog.md`](./docs/providers/catalog.md)).

## What is next

Rough priority order, not a schedule. Nothing here is committed to a release.

1. **Close out the extraction-quality work.** The items under [Known limitations](#known-limitations)
   are the active queue, and the highest-value work in the project: every downstream artifact
   inherits extraction quality.
2. **Run the pinned end-to-end benchmark protocol.** [`docs/BENCHMARKS-E2E.md`](./docs/BENCHMARKS-E2E.md)
   already fixes the hardware, corpus, hop stratification and ablations; every number in it is still
   a placeholder, and a number from a different corpus would be worse than a missing one.
3. **Second reference Providers** for Capabilities that still have only one — a seam with a single
   implementation is a claim, not a proof.
4. **Formal ontology induction** — concepts, taxonomy and axioms over the assembled graph, exported
   as SHACL shapes and OWL. Designed and scoped; not built.
5. **More Source connectors** — Confluence, Google Drive, SQL and chat systems, on the seam the
   SharePoint connector already uses.
6. **More retrieval backend adapters.** Qdrant is the reference; the adapter contract is written
   down so Databricks, Azure AI Search and Elastic can follow.
7. **Context compression** as a Capability, for token-budget-constrained RAG.
8. **Groundedness scoring** — scoring a generated answer against the corpus it should have come
   from, with cited evidence spans.
9. **Remote MCP transport.** The MCP server is loopback + bearer token today; remote transport is a
   deliberate follow-on, not an oversight.

## Known limitations

Measured, or deliberately unmeasured — not speculative. This is where an evaluator should spend their
time.

- **Relation extraction fabricates.** On a judged sample of hyperedges from English encyclopedic
  text, a majority of the relation-bearing ones carried at least one triple the source sentence
  does not actually assert, and some relation labels are far worse than others. Queued fixes: gating
  spans that carry a title but no predicate, filtering common-noun and pronoun mentions out of
  entity promotion, and cell-aware segmentation so table rows are not read as prose.
- **Confidence scores are not calibrated out of the box.** The extractor's confidence is informative
  but is not a probability, and the high-confidence tail was the *least* reliable band in the judged
  sample. Calibration ships as tooling you run on your own labeled sample, so an adopter who skips
  it is working with raw scores.
- **Entity resolution merges by surface form.** Distinct entities sharing a written form are not
  reliably separated, and one entity written two ways is not reliably merged.
- **Induced type vocabularies have a junk tail.** There is no minimum-support floor yet, so rare
  spurious types survive into the graph.
- **Chunk size silently gates hyperedge yield.** Sentence windows rarely cross chunk boundaries, so
  a larger chunk size yields markedly fewer hyperedges from the same corpus. The coupling is
  measured; whether it hurts retrieval is not, and no fix is built on suspicion.
- **The chain-of-evidence retrieval gain depends on an opt-in model.** The configuration that
  passed the evaluation gate uses an opt-in query-time NER model; the zero-model default entry
  point does **not** pass that gate on the harder multi-hop dataset. Per-tier curves ship with the
  results rather than being averaged away, so you can see which tier your install is on.
- **Query-side latency for that tier is an assertion, not a measurement.** It is described as "tens
  of milliseconds on CPU". Nobody has measured it. Treat it as unknown.
- **Evaluation gold labels are partly LLM-generated.** The enterprise-corpus gold set and the
  judged calibration sample were produced by a language model with provenance recorded; human
  spot-checking is outstanding, and any number derived from them inherits the judge's errors.
- **The graph does not help general retrieval.** Measured across three datasets, graph-expansion
  lanes added essentially nothing to recall against a strong dense+lexical base, while costing
  hundreds of milliseconds per query. The value is concentrated in cross-document multi-hop
  questions that single-signal retrievers score zero on. Enable graph lanes for that class of
  query, not for a general recall lift.
- **Scale is characterised, not eliminated.** The 100k and 1M-document envelopes are recorded as
  open in [`PERF-RESULTS.md`](./PERF-RESULTS.md) rather than previewed.
- **Nothing has been released yet.** Signing and SBOM machinery is built and tested but has never
  produced a release; [`RELEASE-GATES.md`](./RELEASE-GATES.md) is the machine-checked checklist.

## Explicitly out of scope

So nobody files it, and nobody plans around it arriving:

- **A retrieval engine.** No index, no stored vectors, no standing graph service, no second
  system of record. First-stage search always runs on the adopter's own store.
- **A hosted service.** This is a library and a CLI that write files you own.
- **An LLM in the indexing or online retrieval path.** LLM Providers exist for extraction where an
  adopter picks one, but the default pipeline and every retrieval path are zero-LLM by design —
  a constraint, not a stage of development.
- **A general-purpose graph database or query language.** Exported Parquet plus DuckDB is the
  zero-operations answer; Neo4j is an optional adapter for people who already run one.
- **Bundled model weights, or a blessed "best" model.** The framework names Capabilities, never
  models. Restricted-license models are opt-in and never a default.
- **Answer generation.** This project builds the corpus and retrieves the evidence; what generates
  the answer is yours.

## How to influence it

- **File a GitHub Issue** for a bug, a Provider you need, or a concrete proposal — held to the same
  bar as maintainer work ([`issues/DEFINITION-OF-DONE.md`](./issues/DEFINITION-OF-DONE.md)).
- **Open a GitHub Discussion** for "should this project do X" or "how do I fit this to my stack".
- **Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before opening a pull request** — the quality bar, the
  vocabulary the codebase uses, and how to add a Provider without touching core.
- **The most valuable contribution is a measurement.** Turning one of the limitations above from an
  assertion into a number, on a corpus that is not the maintainer's, moves this project further than
  a feature does.

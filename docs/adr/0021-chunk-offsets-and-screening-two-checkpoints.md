# Chunk preserves offsets over stripped text; Screening is two checkpoints with Quarantine off the DAG

S3 adds two Stages that make the corpus retrieval-shaped and safe: **Chunk**
(split parsed markdown into retrieval-sized pieces) and **Screening** (catch
dangerous data at two checkpoints). Three design questions had non-obvious
answers, recorded here.

## The chunking asset is a core engine; the Chunker is a thin Provider over it

The ported orchestrator asset — the predecessor implementation's 18-pattern markup
stripping + token budgeting, taken as-is with its vendor coupling stripped out, landing in
S3 Chunk — is offset-preserving text processing. A chunk
boundary is an offset, and that offset must resolve back to a source **page** via
`PageOffsetIndex` — a core Provenance concern (ADR-0020) every downstream Stage
(Entity Extraction spans, Disambiguation Evidence offsets) also needs.

**Decision.** The chunking algorithm lives in `latence_core.chunking` (like
`latence_core.pagemap`): the 18-pattern markup stripper, the deterministic token
budgeter, and `chunk_document`. `ChunkRecord` is a versioned core contract
(`SCHEMA_VERSION` → 3) carrying `char_start`/`char_end` and `page_start`/`page_end`
on its Provenance plus a `risk_markers` list. The reference `MarkdownChunker`
Provider (`chunk.markdown`, ships in core, zero extra deps) is a thin wrapper that
stamps the contract and resolves page alignment through the core
`PageOffsetIndex`. Rejected: the whole algorithm inside a Provider package (would
invert the dependency arrow — core must not depend on a plugin — the same argument
ADR-0020 made for `PageOffsetIndex`).

**The subtle invariant: offsets index the ORIGINAL markdown, not the stripped
text.** Markup stripping deletes characters (fences, `**`, link URLs), so the
stripped chunk text is *shorter* than its source span. We track a per-character
offset map during stripping, so a chunk's recorded `[char_start, char_end)` is its
span in the **original** parsed document — a chunk whose text reads `bold` still
points at the source `**bold**`. That is what makes the S3 round-trip
(chunk → offset → original page) exact, and what a naive "chunk the cleaned string"
implementation silently gets wrong. The token estimator is deliberately
dependency-free and deterministic (no tokenizer download) so runs are hash-stable
for the Baseline bar; a real tokenizer is a drop-in Provider config override.

## Screening is two checkpoints, and Quarantine is a disposition — not a Parse error

CONTEXT defines Screening as one Stage with two checkpoints: **Intake** (before
Parse, over raw bytes: malware/zip-bomb/type-spoof/oversized) and **Content**
(after Chunk, over text: prompt injection, harmful, sensitivity). These are
genuinely different Capabilities — one consumes `ParserInput`, the other
`ChunkRecord` — so they are two Capability protocols (`IntakeScreener`,
`ContentScreener`) and two `CapabilityKind`s, wired at their two positions in the
DAG. `Quarantine` (CONTEXT) is distinct from Parse's `PARSE_ERROR` disposition
(ADR-0020): `PARSE_ERROR` is data the Parser *could not decode*; Quarantine is
*dangerous* data Screening *removes*. Intake Quarantine is what stops a malicious
file before it ever reaches the Parser.

**Decision.** Intake Screening partitions raw inputs into passed vs Quarantined;
only the passed inputs proceed to Parse. Content Screening either **flags** a chunk
(attaches a `RiskMarker` that propagates into the corpus so a RAG consumer can
exclude it — the reference Provider marks rather than drops, because a legitimate
document may quote an injection string) or, above a configurable threshold,
Quarantines it. The reference Providers ship in core, permissively (pure-Python
signature/size heuristics + a keyword/regex injection classifier, no model
weights, ADR-0012); a zip bomb is judged from the archive's central directory
**without decompressing any member**.

## Quarantined records leave the DAG but are retained in a sidecar

A Quarantined record must be *removed from every downstream Stage and Export* yet
*retained and inspectable with the reason* (CONTEXT). The Runner threads one output
list per Stage down the DAG; putting Quarantine records there would either leak
them downstream or lose them.

**Decision.** A Screening Stage's DAG output is only its **passed** records (what
flows on). Its `QuarantineRecord`s and per-record `ScreeningFinding`s are persisted
to run-dir **sidecars** (`quarantine/<stage>.jsonl`, `screening/<stage>.jsonl`),
written atomically alongside the checkpoint. The Quality Report is rebuilt by
reading those sidecars, so quarantine dispositions + reasons and flag counts appear
identically on a fresh run and a resumed one (the sidecars persist on Storage like
checkpoints — resume never re-runs Screening). This keeps the DAG's one-list-per-
Stage invariant and the checkpoint round-trip intact, while satisfying the
"retained and inspectable" requirement. Export's pass-through carrier is resolved
from its nearest producing ancestor (a Chunk ancestor ⇒ ChunkRecord), so its
checkpoint re-reads with the right type on resume whether the corpus is documents
(S1/S2) or chunks (S3).

Rejected: a second "quarantine" edge in the DAG (complicates topological ordering
and every Runner adapter for an audit-only side output); dropping Quarantined
records silently (un-auditable — the Quality Report could not report what was held
back or why).

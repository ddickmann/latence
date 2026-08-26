# Parse carries a page map and an error disposition on the record contract

S2 makes Parse a real Stage: a document Parser that turns PDFs (and text) into
markdown and must (a) preserve which source page any character came from, and (b)
survive a corrupt or type-spoofed file without aborting the run. Two design
questions had non-obvious answers, recorded here.

## PageMap is a `latence-core` contract; PageOffsetIndex is a core resolver — not a Provider concern

The ported `PageOffsetIndex` asset — the predecessor implementation's
*provenance-integrity* piece, landing here in **S2/S8** — is offset→page resolution: it is used at Parse to build the
map, and again downstream at Chunk (offset-preserving splits), Entity Extraction
(mention spans), and Disambiguation (Evidence offsets → source page). If the
`PageMap` contract and the resolver lived in the `latence-parser-document`
package, every downstream Stage would depend on a *Provider* package to interpret
Provenance — inverting the dependency arrow (core must not depend on a plugin) and
coupling the corpus Stages to one Parser's package.

**Decision.** `PageSpan` / `PageMap` are versioned contracts in
`latence_core.contracts` (bumping `SCHEMA_VERSION` to 2), and `PageOffsetIndex` +
`PageDriftDiagnostics` live in `latence_core.pagemap`. A `DocumentRecord` gains an
optional `page_map`. The Parser Provider *produces* the map (via
`PageMap.from_page_texts`, which assembles the markdown and the boundaries
together so they are exact by construction); core *owns* the type and the
resolver, so any Stage resolves an offset to a page without importing a Provider.
The Runner audits each parsed record's map through the resolver and folds the
drift diagnostics into the Quality Report's new `parse` section — the "drift
diagnostics in the Quality Report" S2 asks for. The naive language heuristic was
likewise lifted out of the Source into `latence_core.language` so Source and Parse
share it rather than fork it.

Rejected: `page_map` on a Parser-package type (dependency inversion, above); a
separate out-of-band page-map file keyed by document id (breaks the "every Stage
boundary is one validated versioned record" invariant and the checkpoint
round-trip resume depends on).

## A corrupt file yields a PARSE_ERROR record, not an exception

A real Parse Stage meets malformed input: a truncated PDF, an encrypted PDF, a
`.pdf` that is actually an executable (type spoof). The walking skeleton's
passthrough never failed, so nothing forced the question. Letting a parse error
propagate would abort the whole run on one bad document in a folder of thousands —
the opposite of enterprise-grade.

**Decision.** The document Parser catches every recoverable failure and emits a
`DocumentRecord` with `disposition = PARSE_ERROR`, an `error` reason, empty
`content`, and no `page_map` — Provenance intact for audit. The run continues; the
Quality Report counts `parse_errors`. A contract validator enforces the
invariant both ways (a `PARSE_ERROR` must carry a reason; a `PARSED` record must
not), so the disposition can never be set inconsistently. This is distinct from
**Quarantine** (CONTEXT): Quarantine is Screening's disposition for *dangerous*
data removed from the pipeline; `PARSE_ERROR` is Parse's disposition for data it
*could not decode*. Screening (intake/content) is a later slice; this is the
narrow, Parse-local failure path.

The type-spoof guard (reject a `.pdf` lacking the `%PDF-` signature before handing
bytes to the reader) is a defense-in-depth check, not a substitute for the S-slice
Intake Screening that will catch malware / zip bombs / oversized files ahead of
Parse.

Rejected: raising and letting the Runner decide (couples Parser error semantics
into the Runner and loses the per-document audit record); silently dropping the
document (un-auditable — the Quality Report could not report what failed and why).

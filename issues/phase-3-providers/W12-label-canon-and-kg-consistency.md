# W12 — induced-label canonicalization + corpus schema + redact-induced wiring + KG column consistency

**Real-GPU-E2E findings (grounded, file:line-traced).** Extraction quality is good once induction is on,
but induced labels fragment (case + synonyms) because they are never canonicalized to a normal form
corpus-wide. Plus two smaller genuine gaps. Fix all four, test, verify. (Bug "KG type=None" was a
false alarm — types ARE present on `GraphNode.label`; only a cross-format column-name inconsistency remains.)

## Fix 1 — canonical-form induced labels (case fragmentation) [P0]
- **`latence-schema-inducer/.../sanitize.py`** `sanitize_labels`: today it dedupes on `casefold()` (`:112`)
  but APPENDS the original casing (`:121`). Change it to EMIT a single canonical normal form for each
  label: casefold + collapse internal whitespace AND treat `_`/`-` as space-equivalent (so `payment_product`
  == `payment product`, `Date` == `date`, `Website` == `website`). Keep len≤64 / count≤40 / control-strip.
  Because this is a pure per-string function, every casing/format variant maps to the SAME normal form
  corpus-wide with no cross-call coordination.
- **`latence-core/.../induced.py`** `effective_labels` (`:62-82`): dedupe the induced∪config union
  CASE-INSENSITIVELY (defense-in-depth) and FIX the misleading docstring (`:20-22`) which claims the
  inducer already normalized. Config labels keep their given surface form; induced ones arrive canonical.
- Tests: `sanitize_labels(["Date","date","payment_product","payment product"])` → 2 canonical labels;
  `effective_labels(config=["Bank"], induced=["bank"])` collapses to one.

## Fix 2 — corpus-level induced schema (synonym fragmentation) [P0, the structural fix]
- **`latence-schema-inducer/.../provider.py`**: add `granularity: "corpus"` (alongside doc|chunk; the
  guard at `:181-187` currently rejects it). Corpus mode: induce per-doc (reuse the existing async
  fan-out), then MERGE all docs' induced entity/relation/pii label sets into ONE canonicalized schema
  (Fix-1 normal form + dedupe + count-cap), and STAMP that single shared schema onto EVERY chunk's
  `induced_labels`. One coherent vocabulary for the whole corpus → no per-chunk drift; entity types stay
  consistent across the KG. All W4 safety-nets preserved (fail-open per doc, temp 0 determinism, index-order).
  If the stage streams (can't hold the corpus), document the requirement (needs a full-corpus pass; acceptable
  for the induction stage which is small LLM calls) OR implement as a two-phase stage (collect→broadcast).
- Determinism: corpus schema is the sorted-canonical union → byte-stable under seed.
- Tests: two docs inducing `["Bank","payment_product"]` and `["bank","account type"]` → corpus schema is the
  canonical dedup union, identical on both docs' chunks; a doc whose LLM fails still gets the corpus schema.

## Fix 3 — redaction.gliner2 honors induced pii_types [P1]
- **`latence-pii-gliner2/.../provider.py`** (`:275` area): today it uses only the config `pii_labels` floor
  and IGNORES induced `pii_types`. Wire it through the SAME `effective_labels(config, induced)` seam the
  other extractors use (induced ∪ config), so a corpus-induced `IBAN`/`tax id` label is actually redacted.
  Keep the config floor mandatory; keep all masking/offset/counts-only behavior identical.
- Test: a chunk-doc with induced `pii_types=["IBAN"]` + config `["person"]` → the redactor's effective
  label set includes both (equivalence test with the offline fake).

## Fix 4 — KG node type column consistency (cosmetic, DX) [P2]
- **`latence-core/.../stages/graph_export.py`** `_write_nodes_parquet` (`:144` + schema `:159`): the entity
  type is written under column `label` while GraphML uses key `type` (`:291`) and TTL `lg:entityType`
  (`:219`). Align the parquet: RENAME the node type column `label`→`type` (or add a `type` column =
  `n.label`) so all three export formats name it consistently and consumers reading `type` get the value.
  Update any test/consumer that reads the parquet `label` node column. Edges parquet unaffected.

## Test + verify
- `bash scripts/verify-local.sh` GREEN (ruff+mypy-strict+pytest+conformance+stack-validate+bake-off).
- New unit tests per fix (above); keep determinism + no-PII discipline; no contract-breaking changes
  (induced_labels stays SCHEMA_VERSION-compatible; the graph parquet column rename is the only schema shift —
  bump the export note, not the record SCHEMA_VERSION).
- Pod validation (maintainer): re-run the corpus E2E with `granularity: corpus` → confirm KG node types are
  a SMALL consistent canonical set (no Date/date, Website/website, payment_product/payment product splits),
  redact honors induced financial-PII labels, parquet `type` column populated.

## Definition of Done
- Induced labels canonical (case+format) corpus-wide; `granularity: corpus` yields one shared schema;
  redaction.gliner2 honors induced pii_types; KG parquet node type column named `type` consistently.
  verify-local green; determinism preserved; pod E2E shows a clean, unified entity-type vocabulary.

## Reviewer should check
- Canonicalization is a pure normal-form (no cross-call state) so it works under streaming/per-doc.
- Corpus mode preserves fail-open + determinism (sorted canonical union); every chunk gets the SAME schema.
- redaction.gliner2 uses effective_labels (induced∪config), config floor still mandatory.
- Parquet `type` column matches GraphML/TTL; no other consumer of the old `label` node column left broken.

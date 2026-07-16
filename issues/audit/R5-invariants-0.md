# R5 — The Export writes the un-redacted ContextHeader (entity canonical names + rendered relation triples) into inspectable exported corpus cells, re-leaking PII that content redaction removed

- **Severity:** critical
- **Dimension:** invariants / security (REDACTION-BY-DEFAULT, ADR-0053; W13 embedding-input-ONLY, ADR-0039)
- **Branch:** feat/retrieval-tooling
- **File:** `packages/latence-core/src/latence_core/stages/export.py`
- **Status:** fixed

## What is wrong

W13 (ADR-0039) builds each chunk's `ContextHeader` (`entities` = canonical entity names,
`neighbor_triples` = rendered `A —label→ B` relations) from entities the NER extracted over **raw**
text under the default `extract_on: unmasked` posture (`pipeline.py` `ExtractOn` default,
`wizard.py` default). So a `PERSON` entity's `canonical_name` is the **real** PII name — the same
name the PII Redaction Stage masks out of `content` / `masked_content` (→ `[NAME]`).

The header is documented as *embedding-input for the dense vector ONLY* (safe — a vector exposes no
readable term). But the Export **also** writes the raw header into two **inspectable** surfaces with
**no masking applied**:

1. **JSONL `context_header` field** — always, whenever `context_enrichment` ran: `_batched` dumps
   `record.model_dump()` (which carries the raw `context_header` dict) and `export()` writes
   `json.dumps(row, …)` verbatim (export.py line ~492).
2. **Parquet `context_header` column** — when `context_columns: true`:
   `_context_columns_for_row` line 159 emits `json.dumps(header, sort_keys=True)` (and the header's
   raw `entities`).

This is exactly the re-leak the same file deliberately avoids for the sparse / multi-vector signals,
which are fed `_corpus_content` (redacted content, **no header**) precisely because their emitted
columns are inspectable (export.py lines 552-555). The header path skips that masking.

The publish-gate rag-scan `_raw_pii_in_exports` (`stacks/validate.py`) only checks planted
`EMAIL` / `PHONE` tokens (`'alice@acme.example'`, `'555-0142'`), which NER does not extract as
entities — so this NER-name class of re-leak passes the gate. The setup wizard sets
`context_columns: True` by default for any enrichment pipeline, so a wizard-generated PII-redacting
corpus re-materialises redacted person names (and relation triples containing them) into the shipped
RAG corpus.

## Failure scenario

A `ChunkRecord` with `content` redacted to `[NAME] works at Acme.` (`masked_content` set) and a
header `entities=['Jane Confidential', 'Acme']` /
`neighbor_triples=['Jane Confidential —works_for→ Acme']` exports `content` = `[NAME] works at
Acme.` — yet the JSONL `context_header` field **and** the Parquet `context_header` column both
contain the string `Jane Confidential` verbatim, the exact name redaction removed.

## What to build

Apply a **redaction gate** to the **exported** ContextHeader (both the JSONL field and the Parquet
column), mirroring the SPLADE composition's gate (`latence_splade`): a name is kept in the exported
header only if its surface still occurs, **as a whole word/phrase** (not a bare substring), in the
redacted corpus text; a name that no longer occurs was masked and is dropped.

- `entities` (and the aligned `entity_types` / `kg_node_ids` / `kg_node_centrality` /
  `kg_node_community`) — drop the entry together when the entity surface did not survive, so the
  aligned arrays stay consistent.
- `neighbor_triples` — drop a triple unless **both** endpoint names survive; a masked endpoint (or
  an external neighbor whose name the local redacted text cannot vouch for) is redacted by default.
  The pseudonymous `neighbor_node_ids` (sha256 join keys) are retained so the query-time KG-expansion
  seam still works.

The gate must run **after** the dense-vector / SPLADE signals consumed the raw header (the vector is
allowed the full header — non-inspectable), and only when the row was actually redacted
(`masked_content` present); an un-redacted corpus removed nothing to re-leak, so its header is
exported byte-identical (the additive / byte-identical-when-unwired guarantee). The safe,
non-readable fields (sha256 node/community ids, centrality floats, entity-type labels) carry no
canonical name and are not a re-leak surface.

## Acceptance criteria

1. A redacted chunk (`masked_content` set, `content` → `[NAME] works at Acme.`) with a header naming
   `Jane Confidential` exports **no** occurrence of `Jane Confidential` in either the JSONL
   `context_header` field or the Parquet `context_header` column (with `context_columns: true`),
   while `Acme` (still present in the redacted text) survives.
2. The dense embedding input is unchanged (the vector still sees the full header — vector-only is
   safe): existing W13 embedding tests stay green.
3. An **un-redacted** row (`masked_content` None) exports its header byte-identical to today: the
   existing `context_columns` population tests stay green.
4. A regression test asserts the leak on the old code and passes on the fix.
5. No regression: feature test subset + ruff + mypy --strict clean.

# Ticket 21 — the enterprise gold set: DATA BUILT (LLM-generated)

The AFK half of ticket 21 (the harness) was built in `T21-HARNESS-RESULT.md`. This file
records the **data** half: a synthetic enterprise corpus, a run of the real hypergraph
stack over it, and a 100-question gold set that passes the ticket-17 gate.

Everything lives under `benchmark/enterprise-gold/`. Nothing outside that directory was
touched except this file. No commits were made.

## Read this first

The corpus and the gold set are **LLM-GENERATED** (Claude Opus, 2026-08-01) over **invented
entities**; the gold labels are **machine-authored and pending human spot-check**. This is
**NOT** the manually-labeled set ticket 08 specified — Dennis directed the LLM generation on
2026-08-01. Full statement, including what is and is not mechanically guaranteed:
`benchmark/enterprise-gold/PROVENANCE.md`.

## 1. Corpus

`benchmark/enterprise-gold/corpus/` — 40 documents, 61 KB, ~9 200 words, `.md` + `.txt`.
Memos, meeting minutes, supplier/terminal notices, contract excerpts, press releases,
data-sheet extracts. 12 of 40 documents are German (30%); the rest English. Realistic
clutter throughout: reference numbers, dates, distribution lists, tables, signature blocks,
records-office filler sections. 21 invented PII instances (14 `.example` emails, 7 phone
numbers) sit in signature blocks so Redaction has real work to do.

Planted structure: 96 atomic facts across 8 relation kinds (`acquired`, `supplies`,
`merged_with`, `invested_in`, `partner_of`, `works_for`, `manufactures`, `located_in`),
deliberately spread so that **80 chains of depth 2–4 have their links asserted in different
documents**. Every planted fact and every chain is listed in
`benchmark/enterprise-gold/CHAINS.md` (generated from `chains.py`, so it cannot drift from
the gold set) — that is the sheet a reviewer checks the corpus against.

## 2. Pipeline + run

`pipeline.yaml` follows the e2e shape of `packages/latence-core/tests/test_hyperedge.py`:
`source.local_folder → parser.plaintext → chunk.markdown → redaction.hybrid_rule →
entity.gazetteer → relation.pattern → disambiguation.cascade → graph.canonical
(corpus_id: enterprise-gold, hyperedges on) → export.knowledge_graph + export.jsonl_parquet`.
The corpus export basename is `records` (what `latence-label` looks for). `${REPO_ROOT}` is
substituted by the runner script, so the YAML is machine-independent.

```
PYTHONHASHSEED=0 uv run python benchmark/enterprise-gold/run_pipeline.py --fresh
```

```
documents/chunks:  40 / 206
chunks masked:     206
nodes / edges:     83 / 105
hyperedges:        226
  strong (has_relations): 82 (36.3%)
  gamma  min/median/max:  0.450 / 0.450 / 0.793
  support min/median/max: 3 / 3 / 6
hyperedge KV:      present
quality roll-up:   hyperedges_total=226 relations_total=86 token_cost 3..48
```

226 hyperedges clears the ≥ 200 the ticket-09 calibration sample wants. Arity 1..3
(median 2), token_cost 3..48. Edge labels: located_in 29, works_for 20, manufactures 18,
acquired 15, supplies 14, invested_in 4, partner_of 3, merged_with 2 — the extracted graph
matches the planted fact table, with the one spurious edge found on the first run
(`Oakhurst → located_in → Porto`, a 200-char-window artefact) fixed by lengthening the
intervening text in document 09.

**Determinism**: two `--fresh` runs produced identical hyperedge ids, γ and `tau_text`
(sha256 of the triple list: `feace8f0f85deb69` both times).

**Redaction**: wired and working. All 206 exported chunks carry `masked_content`; zero
emails and zero phone numbers survive in the exported corpus text; `[EMAIL]`/`[PHONE]`
placeholders are present and no `tau_text` carries a masked span.

## 3. Gold set

`benchmark/enterprise-gold/gold.jsonl` — 100 questions, 225 supporting-fact spans, citing
38 of the 40 documents. Offsets are into the **exported** chunk text (`masked_content`),
resolved mechanically by `build_gold.py` from the fact table in `chains.py`, which refuses
to write unless: every fact resolves in exactly one document, the fact count equals the hop
count, and every ≥2-hop question's facts sit in **distinct** documents.

| hops | count |
|------|-------|
| 1 | 20 |
| 2 | 45 |
| 3 | 25 |
| 4 | 10 |

1-hop share 20% — inside the 40% guard.

## 4. The gate, verbatim

```
$ uv run latence-label validate \
    --export-dir benchmark/enterprise-gold/store/_latence/runs/gold-v0/export \
    --gold benchmark/enterprise-gold/gold.jsonl
OK: gold valid — every span round-trips against the corpus
questions labeled: 100
  1-hop: 20
  2-hop: 45
  3-hop: 25
  4-hop: 10
1-hop share: 20%
(exit 0)

$ uv run latence-label stats --gold benchmark/enterprise-gold/gold.jsonl
questions labeled: 100
  1-hop: 20
  2-hop: 45
  3-hop: 25
  4-hop: 10
1-hop share: 20%
(exit 0)
```

## 5. Findings worth carrying forward

1. **`support` is inflated ×3 on every hyperedge in this stack shape.** The Runner's
   `_stream_all_records_of_type(outputs, ChunkRecord, ...)` gathers ChunkRecords from
   **every** stage, and three stages in this DAG carry them: `chunk`, `redact` (the
   chunk→chunk transform) and `export_corpus` (the Export passes its records through). The
   Hyperedge dedup therefore sees each chunk three times and counts support 3 where the
   corpus asserts a fact once. It is uniform, so it does not distort ranking, and γ does not
   read `support` — but any downstream use of `support` as an evidence count (ticket 09
   calibration, any confidence story) is wrong by that factor. The e2e test in
   `test_hyperedge.py` cannot see this: it wires neither redaction nor a corpus export.
   Worth a ticket against ADR-0057 / the Runner gather (dedupe the corpus-wide gather by
   `record_id`, or gather from the chunk *source* stage only).
2. **Redaction false-positive on table rows — RESOLVED.** The markdown chunker strips
   table pipes, so `| Industrial drives | 410 | 455 | 470 |` becomes
   `Industrial drives410455470`, which the hybrid-rule detector masked as `[SSN]` — a
   real precision leak on numeric tables. Fixed by a letter-glue guard on the
   `ssn_nodash` recogniser (a 9-digit run fused to a unicode letter on either side is
   rejected; the literal label-glued `SSN123456789` form is kept via a fixed-width
   exception), applied byte-identically to the hybrid_rule built-in and the ADR-0044
   financial floor. A fresh `--fresh` run now exports zero `[SSN]` placeholders on this
   corpus with all 14 `[EMAIL]` / 15 `[PHONE]` chunks unchanged, and
   `latence-label validate` stays exit 0. Regression tests: the #61 recall benchmark
   (glued-row negatives incl. a German umlaut label, `SSN123456789` positive) and the
   floor suite pin both halves.
3. **Hyperedge yield is driven by chunk size, not corpus size.** With `max_tokens: 220` the
   same corpus produced 40 hyperedges; at 90 it produced 226. Below `min_tokens` a span
   merges into its neighbour, so a coarse chunker collapses a whole chunk into one
   hyperedge. Anyone sizing a calibration sample needs to know that the count is a chunker
   knob as much as a corpus property.
4. **Trigger-window artefacts are the main authoring hazard.** `relation.pattern` fires when
   a trigger sits anywhere between a head- and tail-typed mention inside `window_chars`, so
   two unrelated sections within 200 chars can manufacture an edge (that is how the spurious
   `Oakhurst → Porto` appeared). The cross-reference sections added to each document
   deliberately open with a >200-char filler paragraph for exactly this reason.

## 6. What is still open

The human spot-check. `gold.jsonl` is append-only and resumes in the labeling UI:

```
uv run latence-label gold \
  --export-dir benchmark/enterprise-gold/store/_latence/runs/gold-v0/export \
  --out benchmark/enterprise-gold/gold.jsonl
```

Until that pass happens, the set is machine-verified (spans round-trip, chains cross
documents) but not human-verified (questions unambiguous, one correct answer each,
answerable only via the marked facts).

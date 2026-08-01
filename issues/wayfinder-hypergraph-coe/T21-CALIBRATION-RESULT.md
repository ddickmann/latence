# T21 part 2 — Hyperedge γ-calibration judgment set: DONE (LLM-labeled, pending spot-check)

All **226** Hyperedges of the `gold-v0` export are judged in
`benchmark/enterprise-gold/judgments.jsonl`, in the ticket-21 harness format
(`{"hyperedge_id", "judgment", "note"?}`, sorted keys, append-only / last-wins). This is
the ~200-judgment calibration sample ticket 09 asks for (spec 1.5 isotonic fit).

**Provenance caveat up front: these are LLM judgments (Claude Opus, 2026-08-01), not
human labels.** They are recorded as such in `benchmark/enterprise-gold/PROVENANCE.md`
(§ *Calibration judgments*) and are **pending human spot-check** — a curve fitted from
them inherits whatever the model got wrong.

## Distribution

| judgment | count | share |
|---|---|---|
| correct | 222 | 98.2% |
| partial | 3 | 1.3% |
| incorrect | 1 | 0.4% |

That is a much cleaner extractor than the brief anticipated, and it is the honest count —
the corpus is template-planted and the extractor is a pattern matcher over exactly the
sentence forms the generator emits, so near-perfect precision *on this corpus* is the
expected result, not a labeling artifact. It is also a real property of the sample: with
one γ level (0.792857) carrying 82 relation-bearing Hyperedges of which 80 are fully
correct, and the other (0.45) carrying 144 entities-only Hyperedges of which 142 are
correct, **an isotonic fit on this sample has almost nothing to bend** — the informative
signal is the four exceptions, not the bulk. If the calibration A/B (M6) needs a curve
with actual shape, it needs a harder or noisier corpus, not more labels of this one.

## How each row was judged

`benchmark/enterprise-gold/make_judgments.py` (new, in the benchmark dir; no new deps —
`pyarrow` only). Every Hyperedge's `entity_node_ids` are joined to `graph-nodes.parquet`
canonical names and its `relations` JSON to head/label/tail triples, then compared against
what its own `tau_text` span asserts. Four structural checks run over all 226:

1. every extracted member's canonical name occurs literally in the span (no hallucinated
   or mis-linked member) — **0 failures**;
2. no *other* known entity's canonical name occurs in the span (the member set is complete
   for its span) — **0 failures**;
3. every relation endpoint is a member of that Hyperedge's own entity set — **0 failures**;
4. triples are distinct and each head precedes its tail in the span (all eight schema
   relations appear in active voice in this corpus, so tail-before-head would be a
   direction error) — **0 direction failures, 2 duplicate-triple failures**.

Rows that fail a check, or that only reading the span can settle, are listed individually
with reasoning in the script's `EXCEPTIONS` table; the script *raises* on any unclassified
defect rather than guessing a label. No blanket rule was applied on γ, arity or
`has_relations`.

The judgment of the 144 entities-only Hyperedges deserves a note, because "weak ⇒ partial"
would have been wrong: 120 of them are the corpus's deliberate distractor sentences
(40× "The distribution list … also covers X, Y and Z", 40× "Copies were filed with the
offices of X and Y", 40× "X was named in the circulation list on behalf of Y"), 22 are
bare title/letterhead mentions, and 2 are the odd rows in the table below. None of the
distractor or title spans asserts any relation in the schema — in
particular "named in the circulation list on behalf of Y" is *not* a `works_for` claim, and
the extractor correctly declines it. Emitting the right members and no relation is the
**correct** extraction for those spans, and they are labeled `correct`.

## Recall against the planted facts

Cross-checked against `CHAINS.md`: the export contains **86 relation instances** over
84 distinct planted facts (8 `acquired`, 1 `merged_with`, 2 `invested_in`, 12 `supplies`,
13 `manufactures`, 25 `located_in`, 20 `works_for`, 3 `partner_of`) — i.e. **every planted
fact is extracted, with no spurious relation**, and the 2 surplus instances are the
duplicate emissions below. Perfect relation recall and precision on the planted set.

## The four failures (the actual calibration features)

| hyperedge_id | γ | verdict | what is wrong |
|---|---|---|---|
| `7990a42ecd0aba8b` | 0.45 | incorrect | Doc 05 chunk-2, span *"An open session is scheduled … in the Leeds room"*: a `LOCATION` member **Leeds** was extracted from a **meeting-room name** and mis-linked to the same city node that carries Ardent Vale's registered seat. The only genuinely wrong extraction in the export. |
| `2efd61a5e21c7ec1` | 0.792857 | partial | Doc 22 chunk-2, a **two-sentence** span (seat + product): `located_in Duisburg` correct, but `Vantor -[manufactures]-> Kestrel-Profil` is emitted **twice** under two relation ids. |
| `7453ab20e9129c94` | 0.792857 | partial | Doc 14 chunk-3, same shape: `located_in Heidelberg` correct, `Wendlin -[manufactures]-> Ambrel dosing kit` emitted **twice**. |
| `be67ad7050c5f4a0` | 0.45 | partial | Doc 08 chunk-0, minutes header *"Kalthoff AG … Location: Leipzig works, meeting room 2"*: members right, but no `located_in` extracted although the header does associate Kalthoff with Leipzig. Borderline (meeting venue vs corporate seat) and noted as such. |

### Failure patterns worth acting on

1. **Duplicate relation triples on multi-sentence τ spans (2/2 of the arity-3
   relation-bearing Hyperedges).** Both Hyperedges whose span covers two planted sentences
   emitted the second sentence's relation twice. It is *only* the multi-sentence spans, and
   it is 100% of them — this looks like a real dedup gap in the relation assembler (per
   `relation_id`, not per triple), not a coincidence. Cheap to fix and cheap to regression-test.
2. **Room/venue names promoted to `LOCATION` and coreferenced onto the city node.** One
   occurrence here, but the pattern (a city token used as a facility/room label) is common
   in real minutes and notices, and it silently pollutes a real node — worth a probe on any
   adopter corpus.
3. **Header-field facts are not extracted** (`Location: …`, `Ref: …` lines). Arguably
   correct-by-design; flagged once rather than swept under, because a KG built from minutes
   would want it.

Nothing else: no hallucinated entity, no relation the span does not assert, no wrong
direction, no incomplete member set anywhere in the 226.

## `validate` output (verbatim)

```
$ uv run latence-label validate --export-dir benchmark/enterprise-gold/store/_latence/runs/gold-v0/export --gold benchmark/enterprise-gold/gold.jsonl --hyperedges benchmark/enterprise-gold/store/_latence/runs/gold-v0/export/graph-hyperedges.parquet --calibration benchmark/enterprise-gold/judgments.jsonl
OK: gold+calibration valid — every span round-trips against the corpus
questions labeled: 100
  1-hop: 20
  2-hop: 45
  3-hop: 25
  4-hop: 10
1-hop share: 20%
judgments recorded: 226
```

Exit code 0.

## Files touched

- `benchmark/enterprise-gold/judgments.jsonl` (new, 226 lines)
- `benchmark/enterprise-gold/make_judgments.py` (new, the generator + audit)
- `benchmark/enterprise-gold/PROVENANCE.md` (appended: § Calibration judgments)
- `issues/wayfinder-hypergraph-coe/T21-CALIBRATION-RESULT.md` (this file)

Not committed — a concurrent session owns git on `hypergraph-m1`.

## Correction (supervising session, 2026-08-01)

The two duplicate-triple partials were a REAL assembler defect, now fixed at the source
(`_span_relations` dedupes identical triples — highest confidence wins, smallest relation_id
anchors the audit; regression test pins it, including id-stability: a fact stated twice in one
span hashes like a fact stated once). The export was regenerated (`relations_total` 86 → 84;
the two affected hyperedges re-addressed to `477a3729f3dd5d7a` / `617d7f1084959c26`) and their
judgments re-issued as `correct` with notes. Final distribution: **224 correct / 1 partial /
1 incorrect**; `validate` exit 0 over gold + calibration. The γ-two-values caveat stands: this
corpus cannot exercise isotonic calibration meaningfully — the M6 A/B needs a noisier corpus
(recorded on the map).

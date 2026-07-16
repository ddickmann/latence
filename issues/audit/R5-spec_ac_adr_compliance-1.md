# R5 — The SPLADE `confidence_threshold` gate is inert on the index-time emission path: the real per-EntityMention confidence is discarded and every header entity is stamped at a hardcoded `1.0`

- **Severity:** major
- **Dimension:** spec/AC/ADR compliance (ADR-0053 / ticket #162 AC-04)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #162 AC-04 ("entity terms below an extraction-confidence threshold are excluded"), ADR-0053 ("weak NER never pollutes the sparse signal"), user story 6
- **Status:** fixed

## What is wrong

The ADR-0053 SPLADE composition has a documented `confidence_threshold` knob
(`SpladeCompositionEmbedder`, `packages/latence-splade/src/latence_splade/provider.py:230`;
`compose`, `packages/latence-splade/src/latence_splade/composition.py:102` — an entity with
`confidence < confidence_threshold` is dropped so weak NER never pollutes the sparse signal). The
composing seam `encode_sparse_composed` faithfully carries per-entity confidence, and its unit tests
pass with varied confidences.

But on the **actual index-time emission path** the confidence is thrown away.
`_sparse_signals_for_row` (`packages/latence-core/src/latence_core/stages/export.py:325`) builds the
row's `SparseSignals` from `ContextHeader.entities` — a bare `list[str]` of canonical surfaces with
**no confidence** (`contracts.py:680`) — and stamps **every** entity at a hardcoded confidence `1.0`
(export.py:348: `tuple((str(surface), 1.0) for surface in surfaces)`). The composition's gate then
sees `1.0 >= confidence_threshold` for any sane threshold and injects the entity's term
unconditionally. The `confidence_threshold` config is a **silent no-op** on the emitted corpus
sparse vector.

The only surviving upstream gate is `context_enrichment.min_entity_confidence`
(`packages/latence-core/src/latence_core/stages/context_enrichment.py:157`), which **defaults to
0.0** — so a weak NER entity flows into the header by default, is stamped at `1.0`, and pollutes the
emitted `sparse_indices`/`sparse_values` regardless of the SPLADE `confidence_threshold` the
operator set. This contradicts ADR-0053 ("weak NER never pollutes the sparse signal") and #162
AC-04.

The gap is **masked** because the composing seam (`encode_composed` / `encode_sparse_composed`) is
unit-tested with varied confidences and passes, but no test drives real varied confidences through
the Export → SparseEmbedder wiring: `test_retrieval_signal_export.py` only exercises redaction +
metadata/entity injection with all header entities implicitly at `1.0`, so the bypass is untested
and green.

## Failure scenario

Wire Export with `sparse_embedder = sparse.splade` and `confidence_threshold: 0.9`, leaving
`context_enrichment` at its default (`min_entity_confidence = 0.0`). A chunk carries a low-confidence
NER entity (`EntityMention` confidence 0.2) whose surface survives redaction. Per ADR-0053 / #162
AC-04 that weak entity must be excluded from the emitted corpus sparse vector. Actual pre-fix
behaviour: `_sparse_signals_for_row` stamps it at `1.0`, the composition sees `1.0 >= 0.9` and
**injects** the weak entity's term into `sparse_indices`/`sparse_values`. The `confidence_threshold`
knob never excludes it.

## What to build

Carry the real per-entity confidence end-to-end so the SPLADE gate is a faithful continuation of the
enrichment gate (one rule, two knobs):

1. **`ContextHeader.entity_confidences: list[float]`** — a new additive field aligned 1:1 with
   `entities` (same length + order as `entity_types` / `kg_node_ids`), `[]` when absent
   (byte-identical to a header without it). It is the per-entity **admission confidence**:
   `min(max mention.confidence over the chunk's gated mentions resolving to that node,
   node.confidence)` — exactly the value `v` such that the enrichment gate at threshold `t` admits
   the entity iff `v >= t`, so the SPLADE `confidence_threshold` applies the identical rule at the
   emission clock.
2. **`context_enrichment` populates it** (`_build_header`) from the mention/node confidences it
   already reads for the gate.
3. **`_sparse_signals_for_row` reads it** — offering each entity at its real confidence instead of a
   hardcoded `1.0` (graceful `1.0` fallback only for a legacy header that carries no
   `entity_confidences`).
4. **`_redacted_context_header` keeps it aligned** — drop the confidence together with the entity
   when the surface did not survive redaction, so the array stays consistent (ADR-0039/0051
   alignment).

The change is additive (new field with a `[]` default), stays behind the existing interfaces (deep
module), and does not touch the byte-identical-when-unwired guarantee (no header ⇒ no field ⇒ no
change).

## Acceptance criteria

- [x] `ContextHeader` carries `entity_confidences` aligned 1:1 with `entities` (`[]` when the header
      has no entities / a run without the enrichment).
- [x] `context_enrichment` populates `entity_confidences` from the real mention/node confidences,
      full-length and aligned with `entities`.
- [x] `_sparse_signals_for_row` offers each entity at its real header confidence (not a hardcoded
      `1.0`); `_redacted_context_header` drops the confidence in alignment with a dropped entity.
- [x] Regression test drives **real varied confidences** through the Export → composing
      SparseEmbedder wiring: with `confidence_threshold` set, a weak-confidence header entity is
      **excluded** from the emitted `sparse_indices`/`sparse_values` while a strong one is injected —
      FAILS on the pre-fix hardcoded-`1.0` path (weak term present), passes on the fix.
- [x] Redaction, injection, and metadata behaviour unchanged; the byte-identical-when-unwired
      guarantee holds (a header without `entity_confidences` still composes, weak entities gated only
      when the field is present).
- [x] Feature tests green + ruff + mypy --strict clean.

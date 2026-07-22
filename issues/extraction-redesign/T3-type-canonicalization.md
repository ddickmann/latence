# T3 — Corpus-wide TYPE / label canonicalization in the cleaning phase (the lynchpin)

**Stage:** `disambiguate` (`disambiguation.glinker`) — or a new dedicated consolidation step in that
phase · **Depends on:** nothing (can land first) · **Blocks:** T1 flipping the default ·
**Severity:** major (this is what makes the whole inversion viable) · **Needs a short design note first.**

## Why this exists
Per-chunk induction (T1) produces **type drift**: chunk A induces `org`, chunk B `organization`,
chunk C `company` — all the same type. Corpus-mode avoided this by forcing one schema upstream (at the
cost of recall). The correct place to resolve it is the phase already built for resolution.

## Current behavior (verified)
`disambiguation.glinker` (`packages/latence-disambig-glinker/.../provider.py`) resolves entity
**mentions** to canonical **entities** — a self-referencing KB (dominant-label vote) links each
mention, an audited cascade (exact / alias / acronym / substring / fuzzy / optional embedding) fuses
canonical entities. **It does not canonicalize the type/label vocabulary.** A grep across
`packages/*/src` for corpus-wide type consolidation returns nothing — no stage maps
`org`/`organization`/`company` to one canonical type. So today the "cleaning phase" cleans entities,
not types, and per-chunk induction would leave the KG with drifted types.

## Proposed change
Add **type-vocabulary canonicalization** to the cleaning phase (extend `disambiguation.glinker`, or add
a sibling consolidation Provider that runs alongside it — decide in the design note):

1. Collect the full set of induced type labels across the run (entity types, relation types; PII
   handled by its fixed floor, so lower priority here) from the extracted mentions.
2. **Cluster synonymous types** into canonical groups — reuse the machinery already in the resolver:
   surface normalization + alias/acronym/substring + **embedding similarity** over the *type strings*
   (the same Granite STS rung the entity resolver already has, `use_embedding_merge`), plus an optional
   LLM canonical-naming pass to pick the human-facing canonical label per cluster (reuse the induction
   LLM seam; deterministic temp 0).
3. Emit a **canonical type vocabulary + an alias→canonical map**, and **remap every mention's `label`**
   (and every relation type) to its canonical form before graph assembly. Keep the raw induced label
   on the record (provenance) so the mapping is auditable.
4. Surface it on the **Quality Report**: "N raw induced types → M canonical types", with the alias map
   — so an operator SEES the consolidation and can correct it.

This is the deduplication/cleaning the maintainer asked for, in the phase explicitly developed for it:
entities AND their types are resolved corpus-wide, downstream, *after* per-chunk extraction has
captured everything.

## Acceptance criteria
- [ ] A short design note / ADR-0038 amendment records: canonicalization lives in the disambiguation
      phase; entities and types are both resolved there; consistency is earned downstream, not imposed
      at induction.
- [ ] Given mentions with drifted types (`org`, `organization`, `company`), the phase maps them to ONE
      canonical type; the KG's node/edge types are consistent corpus-wide.
- [ ] The raw induced type is preserved on the record for audit; the alias→canonical map is emitted and
      appears on the Quality Report.
- [ ] Determinism: temp-0 LLM naming + stable clustering → identical inputs give identical canonical
      vocabulary.
- [ ] Anti-false-green regression test proving the drift-collapse; `ruff` + `mypy --strict` clean;
      `verify-local` green.

## Affected files
`packages/latence-disambig-glinker/src/latence_disambig_glinker/*` (or a new
`packages/latence-consolidate-types/` Provider), `packages/latence-core/src/latence_core/stages/disambiguation.py`,
the graph-assembly type handling, the Quality-Report schema, `docs/adr/0038-*.md`.

## Design questions for the note
- Extend glinker vs. a separate `type_consolidation` Capability/Provider? (Deep-module bias: a
  separate Capability keeps glinker focused on entities and makes type consolidation swappable.)
- Does type canonicalization run before or after entity resolution? (Likely interleaved: canonical
  types can inform entity clustering and vice-versa.)
- How is the canonical vocabulary persisted for delta runs? → **T4**.

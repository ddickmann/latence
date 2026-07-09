# W1-disambig REWORK — embed surface + CONTEXT (pod-proven enterprise-safe ER)

**Supersedes PR #127.** The pod validation proved the bare-surface embedding disambiguator OVER-MERGES
distinct entities (IBM + Apple collapsed at threshold 0.86) because short entity surfaces aren't
discriminative with e5. **Pod-measured, decisive:**

| pair | bare surface | surface + sentence context |
|---|---|---|
| IBM ~ International Business Machines (**same**) | 0.883 | **0.964** |
| Apple ~ Apple Inc (**same**) | 0.933 | **0.988** |
| IBM ~ Apple (**distinct**) | 0.870 | **0.808** |
| IBM ~ Berlin (distinct) | 0.818 | 0.764 |

Bare surfaces have NO robust threshold (a distinct pair 0.870 outranks a same pair's margin); with
context the separation is huge and clean (same ≈0.96–0.99, distinct ≈0.76–0.81, any ~0.90 threshold
works). So the rework is: **embed each mention's surface + its surrounding context**, which the current
`disambiguate(mentions, relations)` signature can't reach. Everything else about the provider (the
no-silent-over-merge guard, determinism, drop-in contract, AdapterBase reuse, audit log, the
audit_floor clamp) was correct and stays — this changes ONLY where the embedding text comes from.

## Mechanism: thread ChunkRecords to `disambiguate` (no EntityMention contract change)
Chosen for minimal blast radius (no `EntityMention` schema change, no NER-provider changes):
- **Extend the `Disambiguator` protocol** (capability.py) to `disambiguate(mentions, relations,
  chunks=())` — a third parameter carrying the run's `ChunkRecord`s. Additive/defaulted.
- **`CascadeDisambiguator`** accepts `chunks=()` and **ignores it** (its algorithmic path is
  unchanged — no behavior change, its tests stay byte-identical).
- **The Runner's disambiguation dispatch** (runner.py ~1220) gathers the run's `ChunkRecord`s (it
  already has every stage's outputs) and passes them — same streaming/`_require_*` discipline as the
  mention gather; a run with no chunk stage passes an empty tuple (mentions-only still works).
- **`EmbeddingDisambiguator`** builds, per mention, a context string from its chunk: locate the
  mention's `chunk_record_id` → take a **bounded window** of that chunk's `content` around the
  mention's offset (reuse `context_window`, now defaulting to a sensible >0 value, e.g. 160 chars),
  and embed `f"{surface} {context}"` (or the model-appropriate join). No chunk for a mention →
  gracefully fall back to bare surface (never crash). Determinism preserved (fixed window, stable order).

## Enterprise-safety requirements (the whole point)
- **The IBM/Apple over-merge MUST be gone:** the pod re-validation (and an offline test using
  hand-crafted context-bearing vectors) must show IBM + International Business Machines merge while
  IBM and Apple stay SEPARATE at the default threshold. A regression proves it.
- Default threshold set to the context-appropriate value (~0.90, per the pod margins) so the
  out-of-the-box behavior is enterprise-safe (no over-merge) AND useful (catches real cross-surface
  duplicates) — not the fragile bare-surface 0.83.
- The no-silent-over-merge guard, audit_floor clamp, determinism, Evidence/Provenance carry, O(n²)
  block-first + max_pairs cap, typed-error/no-PII discipline — all preserved.
- Still a true drop-in for `disambiguation.cascade` (same `DisambiguationRecord` contract; the swap
  test passes; the cascade's own tests unchanged).

## Definition of Done
- Protocol + cascade + runner dispatch thread `ChunkRecord`s; cascade behavior byte-identical (its
  tests unchanged). `EmbeddingDisambiguator` embeds surface+context from the mention's chunk window,
  falling back to bare surface when no chunk.
- **Enterprise-safety regression:** an offline test (context-bearing vectors) asserts distinct
  same-type entities with different context do NOT merge while genuine duplicates do; the default
  threshold is the context-safe value. Determinism byte-identical.
- Conformance C1–C6 green (roster unchanged at 35); `bash scripts/verify-local.sh` green; the
  `matrix/disambiguation.yaml` bake-off runs with the real ER metric.
- ADR-0036 note updated with the pod finding (bare-surface over-merge → context rework) and the
  cosine evidence; `scripts/validate-disambig-on-gpu.md` runs the REAL IBM/Apple separation check.
- Re-validated on the pod: the exact 5-mention IBM/IBM-full/Apple/Apple-Inc/Berlin case yields the
  CORRECT clusters ({IBM, IBM-full}, {Apple, Apple Inc}, {Berlin}), not the over-merged 2-cluster
  collapse. (I run this over SSH; numbers folded in, never fabricated.)

## Reviewer should check
- The over-merge is genuinely fixed **because of context** (not just a higher threshold masking it) —
  the offline test uses context-bearing vectors and the pod run separates IBM from Apple.
- The protocol/signature change is backward-compatible (cascade + any disambiguator still satisfy it;
  cascade tests unchanged); the Runner passes chunks with the same #69 streaming discipline.
- Determinism holds (fixed context window, stable order); no-silent-over-merge + audit_floor clamp intact.
- Graceful fallback when a mention has no locatable chunk (bare surface, no crash).

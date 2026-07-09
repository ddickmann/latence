# W1-disambig — `latence-disambig-embedding`: learned (embedding-based) entity resolution

**Phase-3 Wave-1 learned-enrichment provider. GitHub issue #94-ish (P3-A enrichment).** The first
**learned** Disambiguation provider — semantic entity resolution beyond the algorithmic
`disambiguation.cascade` (surface/acronym/substring + union-find). It embeds entity mentions and
merges semantically-equivalent entities the surface cascade misses ("IBM" ↔ "International Business
Machines", "J. Smith" ↔ "John Smith" in shared context), which is a core enterprise "AI-ready data"
need (dedup / entity resolution). Explicitly anticipated by the design (the `Disambiguator` docstring:
"a FAISS/GPU blocking Provider is a pluggable option, not required — S8 AC").

**Built through `AdapterBase`, a true drop-in for `disambiguation.cascade`** (same
`DisambiguationRecord` contract), `compute="either"` (CPU small model or GPU), device-gated, and
**pod-validated with a real embedder**. Reuse the pod-validated `latence-embedder-st` seam for the
vectors — do NOT hand-roll embedding.

## The enterprise crux: NO silent over-merge (the contract's hard rule)
Semantic merging is dangerous — a too-eager threshold collapses two distinct real-world entities, which
is worse than under-merging in an enterprise corpus. The provider MUST honor the contract's
**"no silent over-merge — a below-policy merge is logged, not applied"** rule:
- A merge happens ONLY when cosine similarity ≥ a configurable threshold **AND** the type-compatibility
  guard passes (same/compatible entity types — mirror the cascade's per-surface label-set guard) **AND**
  (optionally, config) a shared-document / context guard. A candidate pair that is similar but
  type-incompatible or below threshold is **recorded as a rejected `MergeDecision` with its score +
  reason, never applied** — the audited, confidence-weighted merge log the contract requires.
- Every applied merge carries `Evidence` back to the source mentions + page-accurate Provenance
  (inherited from the canonical mention, reusing the existing seam — not reinvented).
- **Determinism (Baseline bar):** identical input + config + seed → byte-identical output. Embeddings
  must be deterministic (greedy/fixed model, CPU-reproducible), the clustering order fixed
  (canonical-text order), and tie-breaks stable. A non-deterministic merge is a bug.

## What to build
Package **`latence-disambig-embedding`** registering `disambiguation.embedding`, an `AdapterBase`
subclass satisfying the `Disambiguator` Protocol (`disambiguate(mentions, relations) ->
Iterator[DisambiguationRecord]`, capability.py:281):
- **Embed** each mention's surface (+ optional local context) via a configured Embedder provider
  (default `embedding.sentence_transformers`; the in-core `embedding.hashing` as a no-network
  fallback for offline/CI). Reuse the Embedder capability seam so any embedder composes.
- **Cluster/merge**: for each candidate pair (blocked by exact-normalized-surface first for scale,
  then embedding-similar), apply the threshold + type + context guards → union-find into
  `CanonicalEntity` clusters, emitting the audited merge log (applied + rejected decisions).
- Emit ENTITY-scope records (canonical-text order) then RELATION-scope records (endpoints remapped to
  canonical entities), byte-identical to the contract the cascade produces — a stack swapping
  `disambiguation.cascade`→`disambiguation.embedding` needs no other change (prove with a test).
- **Scale note:** exact cosine over the mention set is fine for v1; FAISS is a pluggable optimization
  (document it as future, don't block on it). Guard the O(n²) pairing (block first) so a low-cardinality
  corpus can't blow up (the same class of guard as the relation fan-out cap).
- **AdapterBase for real:** device seam (`either` → cpu/cuda via the shared `select_device`), guard
  (native embedder fault → typed `ProviderError`, no record content/PII in the message), batching.

## Profile + license + conformance
- `ProviderProfile`: `compute="either"`, honest `memory_mb`, `model_id` = the default embedder's model,
  `deterministic=True` (verify — must be, per the Baseline bar), `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1):** inherits the embedder's license (sentence-transformers Apache-2.0
  code + e5-small MIT weights, cited) — record both, verified. Passes conformance **C3**.
- **Passes P3-F3 conformance C1–C6** offline (against the in-core hashing embedder or a shape-matching
  stub — the anti-false-green rule: the offline embedder double must return real-shaped vectors). Roster
  count updates; conformance + stack-validate green.

## Wire-in
- CI/verify-local: `latence-disambig-embedding --no-deps` in the heavy-provider install (ci.yml +
  `scripts/verify-local.sh`) + wheel list; add a `matrix/disambiguation.yaml` bake-off
  (`disambiguation.cascade` vs `disambiguation.embedding`) with a **labeled entity-resolution
  precision/recall metric** (the enterprise ER quality signal — reuse the S10 gold-set harness where a
  Disambiguation gold-set exists; else a stage metric). `disambiguation.embedding` in `stacks/gpu-sota.yaml`.
- `scripts/validate-disambig-on-gpu.md`: turnkey pod steps (real embedder + a real ER run).

## Definition of Done (Gate bar — a correctness-critical enrichment provider)
- `disambiguation.embedding` builds, registers, AdapterBase subclass, `Disambiguator`, drop-in for
  `cascade` (same DisambiguationRecord contract; a swap test passes).
- **No silent over-merge, proven:** a test with two DISTINCT same-type-but-different entities that are
  surface-similar asserts they are NOT merged (below threshold / context guard), and a genuine
  duplicate pair IS merged with an audited applied `MergeDecision`; a rejected candidate is logged with
  score+reason, never applied. A too-eager threshold would FAIL the distinct-entities test (bites).
- Deterministic + seeded (byte-identical over two runs); O(n²) pairing guarded (blocked); Evidence +
  page Provenance carried; native embedder fault → typed error, no PII leak.
- Evidence-bearing license; passes conformance C1–C6; `matrix/disambiguation.yaml` runs with a real ER
  P/R metric where a gold-set exists (never fabricated); `bash scripts/verify-local.sh` green.
- ADR-0036 note (learned enrichment); seam-report; turnkey pod script. Real GPU numbers from the pod run.

## Reviewer should check
- **Over-merge guard is real and bites** — distinct same-type surface-similar entities are NOT merged;
  every applied merge is audited with Evidence; rejected candidates are logged not applied.
- Determinism holds (byte-identical) — semantic clustering with stable order + tie-breaks.
- True drop-in: a stack swap cascade→embedding produces a valid, contract-complete DisambiguationRecord.
- Reuses the Embedder seam + existing Provenance/Evidence/OffsetMap seams (not reinvented); O(n²) guarded.
- License inherited + verified from the embedder; GPU rows skip-with-flag; ER metric real, not fabricated.

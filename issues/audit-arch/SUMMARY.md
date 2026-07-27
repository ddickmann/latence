# Architecture deepening — audit summary

Hostile audit of the 8-candidate architecture deepening on `feat/arch-deepening`, run with
`conduct-audit` after the `improve-codebase-architecture` review. This file was written by the
maintainer: the finalize agent's harness blocked it from writing report files, and it correctly
reported the block rather than routing around it.

## Rounds

| Round | raw crit/major | confirmed | fixed |
|---|---|---|---|
| R1 | 23 | 20 | 14 |
| R2 | 17 | 6 | 6 |
| R3 | 8 | 3 | 3 |
| R4 | 4 | 3 | 3 |

**Converged** — 20 → 6 → 3 → 3. The 4-round bound was reached with R4 at only 4 raw findings, so the
loop was effectively clean rather than truncated mid-stream. Cumulatively **52 raw critical/major,
32 confirmed, 20 refuted** (38% killed by independent refute-verification). 26 tickets filed in this
directory → **18 distinct defects** (8 were `WONTFIX — DUPLICATE`, re-verified independently).

## Final state (verified by the maintainer, not taken from an agent report)

```
make test                  3894 passed, 29 skipped
bash scripts/verify-local.sh   exit 0 — LOCAL VERIFY GREEN
mypy --strict              PASSED for all 32 packages (src + tests)
```
Baseline on `main` before this work: 3,664 passed / 29 skipped. **+230 tests, never lower.**

## Per-candidate verdict — did it actually deepen?

The question the audit was pointed at: did the interface get **smaller** while the behaviour behind
it got **larger**, or did the change just move code around?

| | Verdict | Evidence |
|---|---|---|
| **C2** CapabilityDescriptor | **YES** | 8 tables across 6 modules → 1 frozen record with *computed* derived sets. `_output_type`'s 50-line if-chain → `descriptor_for(kind).carrier`; `_export_carrier`'s second hand-ordered copy → a walk; `_dispatch_stage` isinstance calls **22 → 6**, refusal messages byte-identical. |
| **C7** StackRecipe | **YES** | `wizard.py` **1,618 → 691 lines**; public functions 13 → 10. A new Stage went from editing two ~250/340-line builders plus a header transcription to **one `contribute(answers, ctx)` declaration**. |
| **C5** QualityAggregator | **YES** | A new metric went from **5 edits → 1 class + 1 registry line**. 13 per-kind gathers → one fold; each stream opened once; the mention stream decoded once instead of twice. |
| **C4** perf entry points | **YES**, one overreach | 8 Provider packages' hand-ordered 7-step load sequence → **one call each**. The 829-line AST linter was deleted for behavioural tests. R3 had to restore a glob-*discovered* guard C4 had replaced with a hardcoded 6-provider list — it went one step too far. |
| **C1** page slice | **YES** — see below | Rebuilt after the audit. |
| **C3+C6** StageExecution | **YES, caveat named** | Runner-side choreography → `stage_execution.execute(stage, ctx)`; Airflow's fork now calls the seam verbatim. **Caveat:** `LocalRunner`'s public method count went **7 → 9** — the class surface grew. What shrank to zero is what a *second* Runner must reimplement. |
| **C9** Storage.atomic_writer | **YES caller-side** | The 4-step `makedirs`→`temp_uri`→write→`finalize_atomic` (+ hand-rolled sweep) → `with storage.atomic_writer(uri) as fh:`. Zero production callers of the old pair remain. **Caveat:** `Storage` public methods **16 → 18** — the deprecated pair was kept on the contract. |
| **C8** SubProviderSlot | **PARTIAL — named, not spun** | `_SLOTS` is the single declaration of all 7 nested keys and feeds `envspec.stack_components()`. **But 3 of its own 10-site census still hand-roll the block** (`disambig-glinker`, `disambig-embedding`, `parser.render`'s `ocr`) with divergent error types. Those files are byte-unchanged vs `main` → incomplete consolidation, not a regression. The docstring's "the one place that knows about the block" is **an overclaim today**. |

### Is registry drift now unrepresentable?

**Yes for the eight facts the descriptor owns; not universally — and the boundary is stated, not blurred.**

Proven live rather than asserted: dropping one descriptor and re-indexing raises
`ContractError: no CapabilityDescriptor for type_consolidation …` at `import latence_core`. There is
no path where a new Capability merely *behaves* wrongly.

The measured historical drift cannot recur: `TYPE_CONSOLIDATION` had been corpus-level in the
Runner's and DeltaRunner's tuples but in **neither** `phase_boundary` set, so `check_phase_boundary`
waved a front-half `consolidate_types` through. Both sets are now computed from one declaration.

**Honest limit:** the Quality registry is *deliberately* not descriptor-sourced — the relation is
many-to-many, so a per-Capability factory would be wrong, not merely redundant. It is held by
`check_aggregator_coverage` at import in **one direction only**: a Capability cannot claim a section
nothing fills, but an aggregator filling an unclaimed section is not refused. That is a *checked
invariant*, not an unrepresentable state.

### Are the new seams real (≥2 adapters)?

Real: `PageIndexResolver` (14 sites), `CapabilityDescriptor` (6 consumers), `StageExecution`
(LocalRunner + Airflow), `perf` entry points (gliner ×3, gliner2 ×2, transformers ×3),
`QualityAggregator` (13 implementations), `StackRecipe` (18 contributions, 2 recipes),
`Storage.atomic_writer` (3 callers).

Flagged: **`Storage.atomic_writers`** (the multi-file face) has a **single consumer** (`stages/export`).

## C1 — rebuilt after the audit, because four rounds patched consequences and never the shape

The audit confirmed 10 of its 18 defects against C1 and redesigned its retention policy twice
(R3 added an LRU window; R4 disproved that window's own justification). None of that fixed the root:
v18 stamped `page_map` on a document's **first chunk only**, and the resolver adopted the map from
the first chunk it saw — so document-scoped state rode one record of a stream, and resolution
depended on that record arriving first.

Maintainer probe against the post-audit tree:

```
CASE A — in order (map-carrier observed first):  page_for(c2, 250) = 3     correct
CASE B — c2 observed and queried FIRST:          page_for(c2, 250) = None  unresolved: 1
```

It degraded **silently** (an `unresolved` counter) rather than raising — an ADR-0034 problem on top
of the correctness one. An audit hardens what exists; it does not redesign, and no auditor was told
the correct shape.

**The fix (commits `5722bb2` → `fbfbf78`, schema v18 → v19, BREAKING):** `ChunkRecord.page_map`
became `ChunkRecord.page_slice` — each chunk carries **only the `PageSpan`s overlapping itself**
(typically 1–2, not 519), in original document coordinates. `PageIndexResolver` is now **stateless**:
no `observe()`, no per-document memoisation, no LRU — there is no shared state for ordering to depend
on, so the failure mode is **impossible to express** rather than mitigated. `page_for` returns `int`,
not `int | None`; a missing slice raises `PageSliceMissingError`.

**Mechanism deleted, not kept as a fallback:** `rehome_page_maps()`, `from_documents()` and
`DEFAULT_MAX_LIVE_DOCUMENTS` are all at zero production references. A fix that *deletes* compensating
mechanism is the signature of the right shape. ~99.8% of the per-line payload win C1 existed for is
retained (a 2-span slice vs a 30 KB map).

Verified after the migration: `page_for(c2, 250) = 3` with chunk 0 never having existed and nothing
else observed. No test was deleted to reach green; several were carried into the new shape as a
*stronger* bar (page-straddling chunks fed in reverse stream order — a case v18 lost entirely).

## Residuals — named, not silently downgraded

1. **C8 consolidation is incomplete** — 3 of 10 sub-provider sites still hand-roll the block. Not a
   regression (those files are unchanged vs `main`), but the module docstring overclaims today.
2. **`Storage.atomic_writers`** is a single-consumer seam — hypothetical by the two-adapter rule.
3. **`LocalRunner` (7→9) and `Storage` (16→18) grew their public surface** while deepening what sits
   behind it. Defensible, but it is not an unqualified "interface got smaller".
4. **`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md` still says `schema_version` 15** — stale since before this
   work (v16–v18 never updated it). The v19 field rename is flagged there; the version numbers were
   deliberately not "corrected", since that would assert v16/v17/v18 facts nobody verified.
5. **Not covered by `verify-local` and therefore unproven here:** the MinIO `s3://` e2e lane, the
   Airflow DAG lane, and real GPU provider runs on the CUDA pod. The GPU run is the next milestone.
6. **v18 → v19 is a breaking schema change.** A v18 checkpoint does not load into v19. There is no
   migration path; a corpus processed under v18 must be reprocessed.

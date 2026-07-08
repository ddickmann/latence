# P3-F5 — Bake-off / Provider Matrix harness (Goal B) — the Wave-0 capstone

**Phase-3 Wave-0, final slice. GitHub issue #90. ADR-0036 §4.** Builds on everything: the
`ProviderProfile` (F1), `AdapterBase` (F2), the conformance gate (F3), the real `parser.pdfplumber`
adapter (H-A1), and the stack-validation harness (F4). This is **Goal B — the per-Stage bake-off**:
hold every other Stage at the blessed default and run the SAME corpus + gold-set through each
candidate provider for ONE Stage, producing a comparison the enterprise can act on.

**This slice also retires the last audit thread (THEME A):** `ProviderProfile.memory_mb`,
`cost_per_1k`, and `deterministic` were flagged inert because their consumer didn't exist yet — the
bake-off IS that consumer. F5 reads and surfaces them (declared vs measured), so the descriptor stops
being speculative.

## 1. `matrix/<stage>.yaml` — candidate list for one Stage
A top-level `matrix/` directory: each file names one Stage (by CapabilityKind) and the candidate
providers to compare for it, e.g. `matrix/parse.yaml` → `[parser.plaintext, parser.pdfplumber,
parser.document]`; `matrix/chunk.yaml` → `[chunk.markdown, chunk.sentence_window]`;
`matrix/entity.yaml` → `[entity.gazetteer, entity.gliner]` (GPU/learned candidates included but
device-gated). Ship a couple of real CPU matrices (parse, chunk) so the harness is exercised on true
alternatives, not mocks.

## 2. The bake-off harness — same corpus + gold-set, one stage swapped
A harness (reuse F4's stack-validate machinery + the S10 gold-set scorer in QualityReportBuilder —
do NOT reinvent scoring/completeness/determinism) that, for a given `matrix/<stage>.yaml`:
- holds every OTHER stage at `stacks/default.yaml`, swaps in each candidate for the target stage,
- runs the SAME bundled corpus (+ gold-set where the Stage has one) through each candidate,
- measures + records per candidate:
  - **quality** — the S10 gold-set P/R/F1 where the Stage has a gold set (NER/RE/…), else a
    stage-specific metric (parse: chars-extracted + offset drift; chunk: chunk count + coverage;
    etc.). Reuse the gold-set harness, don't hand-roll.
  - **latency / throughput** — measured wall-clock for the target stage (docs/sec). Deterministic
    seeds; note that wall-clock is machine-relative (report it, don't gate on it).
  - **peak memory** — measured (`tracemalloc` Python allocations, with the H-E1 off-heap-RSS caveat
    stated) **vs the declared `profile.memory_mb`** — surfacing declared-vs-measured is the point.
  - **cost** — `profile.cost_per_1k` (endpoint providers), else n/a.
  - **license** — `profile.license` + whether evidence-bearing/verified (H-B1) + opt-in flags.
  - **determinism** — `profile.deterministic`, cross-checked by running the candidate twice and
    asserting byte-identical output (reuse the H-E1 equality helper); flag a provider that declares
    `deterministic=True` but isn't.
  - **error rate** — from the run's StageMetrics `error_count`.
- **GPU/endpoint candidates:** device-gated via the shared `select_device` seam — run where a rig
  exists, else **skipped-with-flag** and recorded as "needs a rig (#64)", NEVER a fabricated number
  (the honest boundary the whole workstream has held).

## 3. Output — a signed `MATRIX-RESULTS-<stage>.md` + CLI
- `latence bake-off <matrix-file>` (a new `bake-off`/`matrix` CLI sub-command) runs the matrix and
  writes a signed-style `MATRIX-RESULTS-<stage>.md` (a table: candidate × the dimensions above +
  a skipped/why column), like `G1-RESULTS.md`. Machine-readable companion (JSON) alongside so the
  numbers are consumable, not just prose.
- Deterministic + offline for the CPU matrices; a candidate that fails conformance/crashes is
  recorded as a failed row (the bake-off degrades per-candidate, never aborts the whole matrix).

## 4. CI
Run at least one CPU matrix (`matrix/parse.yaml` or `matrix/chunk.yaml`) in CI on the bundled corpus
(offline, `LATENCE_CUDA=0`, deterministic) — proving the bake-off runs end-to-end and its numbers are
reproducible. Keep it fast (tiny corpus). GPU matrices are not run in CI (flagged).

## Out of scope
Real GPU/endpoint quality + throughput numbers (need a rig, #64 — flagged, never fabricated). The
Wave-2 consolidated `PROVIDER-READINESS.md` (#96) is a later roll-up, not this slice.

## Definition of Done (Gate bar)
- `matrix/` with ≥2 real CPU stage matrices; the bake-off harness runs each candidate over the same
  corpus/gold-set holding other stages at default; per-candidate quality/latency/memory(declared-vs-
  measured)/cost/license/determinism/error-rate recorded.
- `ProviderProfile.memory_mb`, `cost_per_1k`, `deterministic` are genuinely READ + surfaced (the
  audit's inert-fields observation is retired — grep shows real readers now).
- `latence bake-off <matrix>` writes `MATRIX-RESULTS-<stage>.md` + a JSON companion; a CPU matrix runs
  green in CI; a declared-deterministic-but-not candidate is flagged (prove the determinism check bites);
  GPU/endpoint candidates are skipped-with-flag with no fabricated numbers.
- Reuses the gold-set scorer + F4 stack machinery + H-E1 equality — no reinvented scoring/determinism.
- `mypy --strict`/`ruff`/pytest green; deterministic seeded; ADR-0036 §4 status note + a docs page
  ("run a bake-off"); the P3-F3 conformance suite + F4 stack-validate stay green.

## Reviewer should check
- The bake-off genuinely holds other stages constant and swaps only the target (a real controlled
  comparison), and degrades per-candidate (one bad candidate doesn't abort the matrix).
- `memory_mb`/`cost_per_1k`/`deterministic` are actually read (not just printed from a mock) — the
  inert-fields finding is really closed.
- Determinism cross-check bites (a lying `deterministic=True` is flagged); GPU/endpoint rows are
  skipped-with-flag, never fabricated.
- Quality reuses the S10 gold-set scorer; no reinvented P/R/F1 or offset logic.

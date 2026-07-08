# Run a bake-off (P3-F5)

A **bake-off** compares many Provider candidates for **one Stage** — a real, controlled comparison
an enterprise can act on. It holds every OTHER Stage at the blessed `stacks/default.yaml` and runs
the SAME bundled corpus (+ gold-set where the Stage has one) through each candidate, recording per
candidate: **quality**, **latency/throughput**, **peak memory** (measured vs declared), **cost**,
**license**, **determinism**, and **error rate**. Architecture: ADR-0036 §4.

This is the consumer that retires the audit's "inert fields" observation: `ProviderProfile.memory_mb`,
`cost_per_1k`, and `deterministic` are genuinely READ and surfaced here.

## The matrix file — `matrix/<stage>.yaml`

A `matrix/<stage>.yaml` names ONE Stage (by `CapabilityKind`) and the candidate Providers to compare:

```yaml
# matrix/parse.yaml
stage: parse
candidates:
  - provider: parser.plaintext
  - provider: parser.pdfplumber
  - provider: parser.document
```

Each candidate may carry a per-candidate `config` override merged onto the base stage's config (so a
GPU candidate can pin `device: cuda` and a chunker candidate its own `max_tokens`), and the matrix
may set an explicit `stage_name` when the base stack has two Stages of the target Capability (e.g.
its two Export nodes):

```yaml
# matrix/entity.yaml
stage: entity_extraction
candidates:
  - provider: entity.gazetteer
  - provider: entity.gliner
    config:
      device: cuda   # honest GPU intent; skipped-with-flag where no CUDA
```

The two shipped CPU matrices — `matrix/parse.yaml` and `matrix/chunk.yaml` — compare real, offline,
pure-Python alternatives, so they run deterministically in CI.

## Running a bake-off — `latence bake-off`

```bash
latence bake-off matrix/parse.yaml
# → a MATRIX-RESULTS-parse.md table + a MATRIX-RESULTS-parse.json companion in matrix/
```

By default the base stack is the repo's `stacks/default.yaml` (resolved relative to the matrix file's
repo root) and the results are written next to the matrix file. Override either:

```bash
latence bake-off matrix/parse.yaml \
    --base-stack stacks/my-client.yaml \
    --out-dir results/ \
    --storage-root file:///tmp/my-bakeoff   # keep the run artifacts to inspect
```

The command always exits 0 when the harness itself ran — a failed or device-skipped candidate is a
**documented row**, not a gate (the enterprise reads the table). The bake-off degrades per candidate:
one crashing candidate is a failed row, never an aborted matrix.

## What each column means

| Column | Source | Notes |
|---|---|---|
| **quality** | The S10 gold-set P/R/F1 (`report.goldset`) where the Stage has a gold set (entity/relation); else a stage-specific metric (parse: chars-extracted + offset-drift; chunk: chunk count + offset-preserving fraction). | Reuses the gold-set scorer — no reinvented P/R/F1. |
| **latency / throughput** | The target Stage's measured `duration_seconds` → docs/sec. | Wall-clock is **machine-relative** — reported, never gated on. |
| **peak memory** | `tracemalloc` peak Python allocation vs the declared `profile.memory_mb`. | `tracemalloc` tracks **Python allocations only** — off-heap/native (GPU) memory is not counted, so GPU rows are measured on a rig (#64), never here. |
| **cost/1k** | `profile.cost_per_1k` (endpoint providers), else n/a. | |
| **license** | `profile.license` + `license_verified` (H-B1) + an opt-in flag for a restricted-but-permitted license. | |
| **deterministic** | `profile.deterministic`, cross-checked by running each candidate twice and comparing the target Stage's checkpoint bytes. | A `deterministic=True` that diverges is **flagged** (`⚠ MISMATCH`). |
| **errors** | The target Stage's `StageMetrics.error_count`. | |
| **skipped/why** | Skipped-with-flag (GPU/endpoint candidate not runnable on this host, #64) or FAILED (crashed / failed conformance). | A skipped row carries **no fabricated numbers**. |

## Device honesty (GPU / endpoint candidates)

A GPU or endpoint candidate that is not installed, or that requests `device: cuda` on a CPU-only
host, is **skipped-with-flag** — recorded as a row whose measured columns are dashes and whose reason
references the rig gap (#64). It is never run, never crashed, and never a fabricated number. On a real
GPU host with the candidate's heavy package installed, it runs (`device: cuda`) and is measured for
real — the same matrix, honest on both hosts.

## CI

`ci.yml` runs `latence bake-off matrix/parse.yaml` and `matrix/chunk.yaml` on every push with
`LATENCE_CUDA=0` — the bake-off proven end-to-end (the real pipeline, not a config-parse smoke) on
the tiny bundled corpus, offline + deterministic + reproducible. GPU matrices are not run in CI
(flagged).

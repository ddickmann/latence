# P3-F4 — Stacks + the stack-validation harness (Goal A)

**Phase-3 Wave-0. GitHub issue #89. ADR-0036 §3.** Resumed after the hostile-audit remediation; the
provider seam is now validated by a real consumer (H-A1, `parser.pdfplumber`), every provider passes
the P3-F3 conformance gate, and licenses are evidence-bearing (H-B1). Now make **Goal A** real:
a **blessed default stack** validated end-to-end by **one command**.

## 1. `stacks/` — full Pipeline configs, not slices
A top-level `stacks/` directory holding complete, runnable Pipeline configs (the whole spine
Source → Intake-Screening → Parse → Chunk → Content-Screening → NER → RE → Redaction → Profiling →
Disambiguation → Graph-Assembly → Export), distinct from the per-stage `examples/*.yaml`:
- **`stacks/default.yaml`** — the blessed **CPU-first** stack (ADR-0007): in-core deterministic
  providers + the real CPU adapter where it adds value (`parser.pdfplumber` for PDFs / `parser.plaintext`
  for text, `chunk.markdown`, `entity.gazetteer`, `relation.pattern`, `redaction.hybrid_rule`,
  `profiling.statistical`, `disambiguation.cascade`, `graph.canonical`, `export.jsonl_parquet`,
  both Screening checkpoints). Comprehensive-by-default redaction (H-C1). This is what an adopter runs
  out of the box; it must be fully offline + permissive + deterministic.
- **`stacks/gpu-sota.yaml`** — the GPU/high-accuracy stack: the learned/endpoint providers
  (`entity.gliner`, `relation.gliner_relex`, `redaction.gliner_pii`, an embedder) that are
  **skipped-with-flag on a CPU-only host** (the P3-F1 device posture) — declared honestly, never
  faking GPU behaviour. Room for `stacks/<client>.yaml` is the documented extension point.

## 2. `latence stack validate <name>` — the one-command gate (generalizes G1)
Add a `stack` sub-command group to the existing `latence` CLI (`latence_core/cli.py`, typer) with a
`validate` command that runs a named stack end-to-end on a (small, bundled) corpus and asserts the
G1-style checks in one place — the G1 gate's E2E/reliability lanes generalized from a one-off gate
into a reusable harness:
- **Contracts/completeness:** every record carries Provenance + Classification; offsets align
  (reuse the QualityReportBuilder completeness check + ADR-0031 zero-drift — do NOT reinvent).
- **KG:** a non-empty, content-addressed knowledge graph with per-edge Evidence is produced + exported.
- **RAG corpus:** the redacted, chunked, embeddable corpus is exported (Parquet/JSONL) and well-formed.
- **Quality Report:** produced, schema-valid, counts-only (no PII), with per-stage metrics + the
  device/license/profile columns (P3-F1).
- **Reliability:** a determinism check (same input+config → byte-identical export, reusing the H-E1
  equality helper) and a resume check (kill-less is fine here: run → resume same run_id → byte-identical),
  and a graceful-failure check (a malformed input is quarantined/PARSE_ERROR'd, never crashes the run).
- **Device honesty:** on a CPU-only host, `validate stacks/gpu-sota.yaml` reports the GPU stages as
  **skipped-with-flag** (not failed, not faked) and still validates the CPU-reachable spine.
Output: a clear pass/fail per check + a signed-style summary (like `G1-RESULTS.md`) written to Storage;
non-zero exit on failure so it can gate CI/publish. Reuse the LocalRunner + QualityReportBuilder +
conformance predicates — this harness is composition, not new pipeline logic.

## 3. Wire the default-stack validation into CI (Goal A = "test the default stack, one command")
Add a CI step running `latence stack validate stacks/default.yaml` on the bundled corpus (offline,
`LATENCE_CUDA=0`, deterministic) so the blessed stack is proven end-to-end on every push — the
generalization of the G1 E2E lane into a standing gate. Keep it fast (tiny corpus).

## Out of scope (P3-F5)
The per-stage **bake-off matrix** (comparing many providers for one Stage) is F5. F4 validates ONE
named stack end-to-end; it does not compare providers. Don't build the matrix here.

## Definition of Done (Gate bar)
- `stacks/default.yaml` (CPU-first, offline, deterministic, comprehensive redaction) + `stacks/gpu-sota.yaml`
  (GPU providers, skip-with-flag on CPU) exist and are valid Pipeline configs.
- `latence stack validate <name>` runs a stack end-to-end and asserts contracts/KG/RAG/report +
  determinism + resume + graceful-failure + device-honesty; writes a summary; non-zero exit on failure.
- `latence stack validate stacks/default.yaml` runs green in CI on the bundled corpus, offline; a
  deliberately-broken stack (wrong wiring / a provider that violates a check) makes it FAIL (prove the
  harness bites, e.g. an xfail-marked test).
- `gpu-sota.yaml` validation on this CPU host reports GPU stages skipped-with-flag (no fabricated numbers).
- `mypy --strict`/`ruff`/pytest green; deterministic seeded; ADR-0036 §3 status note + a short docs page
  ("run/validate a stack").

## Reviewer should check
- The harness genuinely bites (a broken stack fails; not just a happy-path smoke).
- `default.yaml` is truly offline/permissive/deterministic (no learned/endpoint provider sneaks in);
  `gpu-sota.yaml` GPU stages are skipped-with-flag on CPU, never faked.
- Reuses the QualityReportBuilder/conformance/H-E1-equality seams — no reinvented completeness/offset
  or determinism logic.
- The CI default-stack gate is real (runs the pipeline end-to-end, not a config-parse smoke).

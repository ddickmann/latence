# P3-F2 — `AdapterBase`: the thin, optional base every provider adapter reuses

**Phase-3 Wave-0 foundation. GitHub issue #87. ADR-0036 §1.** Builds directly on
P3-F1 (`ProviderProfile` + `select_device`, merged #99) and the ADR-0034 error taxonomy.

A **provider is an adapter** over a native library; the Capability Protocol (ADR-0004)
stays the I/O contract. Every adapter repeats the same cross-cutting glue — pick a device,
batch its inputs, turn a native library exception into a **typed** framework error, and carry
Provenance/Classification + offsets through the native call. P3-F2 factors that glue into one
**optional** `AdapterBase` in `latence-core` so a Wave-1 adapter is small and consistent, and
so those concerns are done **one right way** instead of five subtly-different ways.

## Non-negotiable framing
- **Optional, not a new seam.** `AdapterBase` is a convenience base. A provider may still
  satisfy a Capability Protocol structurally without it (the existing in-core references do —
  do NOT force-migrate them). No change to the Capability protocols. `AdapterBase` must not
  become a required layer (ADR-0036 rejected a second protocol).
- **Dep-light (ADR-0016).** Lives in `latence-core`; pulls in **no** heavy deps (no torch,
  no model libs). It only orchestrates + maps; the adapter subclass owns its native dep.

## Scope (this slice)

### 1. `AdapterBase` in `latence_core/providers/adapter.py`
A small base carrying, as reusable methods/mixins (compose, don't inherit a god-class if a
Protocol-friendly shape is cleaner):

- **Device selection.** From `config.device` (`auto|cpu|cuda`) + the subclass's declared
  `ProviderProfile`, resolve via P3-F1's `select_device()` → expose the chosen device (or the
  skip `DeviceDecision`) to the subclass. **Device placement stays the adapter's job** (it
  moves its own model, ADR-0003); `AdapterBase` only resolves+hands over the decision. A
  `compute="gpu"` adapter on a CPU-only host must see the skip decision, never a fake device.
- **Batching.** A helper that, when `profile.batch` is True, groups an input iterable into
  `config.batch_size` (validated `>= 1`, sensible default) chunks and calls a subclass
  `_run_batch(...)`; when `profile.batch` is False, degrades to one-item calls. Deterministic
  ordering: outputs come back in input order regardless of batching. This is the streaming
  Runner's (ADR-0033) friend — the helper must not buffer the whole corpus.
- **Native-exception → typed taxonomy (ADR-0034).** A context manager / decorator
  (`self._guard("<op>")` or `@adapts`) that catches an exception raised **inside the native
  call** and re-raises it as the right `LatenceError` subtype — a genuine provider/runtime
  failure → `ProviderError`; an operator-supplied bad config (bad device, bad batch_size, a
  malformed model_id) → `ConfigError` — tagged with provider name + op, **carrying no record
  content/PII** (only counts/category, matching the report discipline). A `LatenceError` raised
  *by* the framework passes through unwrapped. `category_of()` must classify the result.
- **Provenance/offset preservation helpers.** Small utilities so an adapter maps a native
  library's output back onto the input's `Provenance`/`Classification` and character offsets
  without hand-rolling it each time (e.g. `carry_provenance(src_record, produced)` and an
  offset-rebase helper that reuses the existing `OffsetMap`/page-resolution seam — do NOT
  reinvent a weaker offset mapper; ADR-0031). These make the "offsets/page-provenance intact"
  invariant the default, not a per-adapter reimplementation.

### 2. Resolve the P3-F1 deferred defect — malformed profile must NOT be silently swallowed
P3-F1's correctness lens found (Sev-3, deferred to here): a provider that declares a **malformed**
`PROFILE`/`profile()` (`profile_of()` raises `TypeError`) is caught by the Runner
(`runner.py` device-routing try/except) and **falls through to normal execution**, which
re-loads the class but never re-reads the profile — so the bad declaration never resurfaces.
**Decision: fail loud.** A malformed profile declaration is an author bug and must surface as a
typed `ConfigError` (a FAIL Quality Report + typed-error span), not run silently as "no profile."
- Change the Runner's profile-routing catch so `profile_of()`'s `TypeError` (malformed
  declaration) is mapped to `ConfigError` and **fails the stage**, while a genuinely *absent*
  profile still means "follow the request" (None), and an *import/load* error still defers to
  the normal execution path that re-raises with full failure handling (do not double-handle).
  Distinguish "no profile declared" (fine) from "profile declared but malformed" (fatal).
- This aligns the Runner with `profile_of()`'s own documented contract ("a typo surfaces
  rather than reading as no profile"). Add the **Runner-path** regression test that was missing
  (a provider whose `PROFILE = "not a profile"` → stage FAILs with a `ConfigError`, not a
  silent normal run). Keep the unit-level `profile_of` malformed test.

### 3. Prove it with a reference adapter (no new heavy dep)
To exercise `AdapterBase` end-to-end without adding a Wave-1 dep, **re-express one existing
in-core reference provider** as an `AdapterBase` subclass **in a test/fixture** (or a tiny
in-core example), showing: device resolution, batching on/off, a native `ValueError` mapped to
`ProviderError`, and provenance/offset carry — running through the LocalRunner. Do **not**
migrate the shipping in-core providers (keep them as the "works without AdapterBase" proof).

## Out of scope (later)
The conformance suite (P3-F3), stacks + validation (P3-F4), the bake-off (P3-F5), and any
Wave-1 adapter package (docling/kreuzberg/paddle/rust-chunk/embedders/GLiNER-variants/PII).
Do not build those here.

## Definition of Done (Baseline bar + these)
- `AdapterBase` (+ the exception-mapping guard, batching helper, device-resolution, provenance/
  offset helpers) in `latence-core`, strict-typed, `mypy --strict` clean over src+tests,
  re-exported from `latence_core` with `__all__`. **No heavy dep in core** (grep-guarded).
- Optionality proven: the existing in-core reference providers are **unchanged** and still pass;
  a test shows a Capability satisfied **without** `AdapterBase` still works.
- Exception mapping: native `RuntimeError`→`ProviderError`, operator bad-config→`ConfigError`,
  framework `LatenceError` passes through; `category_of()` classifies each; **no PII/record
  content** in the raised message (assert on a PII-bearing input).
- Batching: deterministic input-order output for batch and non-batch providers; `batch_size`
  validated; no whole-corpus buffering (streaming-friendly).
- Device: an `AdapterBase` GPU adapter on this CPU host yields the **skip DeviceDecision**
  (never a fabricated device); a CPU/either adapter resolves to cpu; `LATENCE_CUDA=1` path
  exercised at the seam.
- **Malformed-profile fix**: the Runner FAILs a stage whose provider declares a malformed
  profile (typed `ConfigError` + FAIL report), with a Runner-path regression test; absent
  profile still means follow-request; import-error path unchanged. Re-verify determinism/resume
  are intact (core-touching change).
- Reference-adapter E2E through LocalRunner is green, deterministic, seeded; `ruff` clean.
- ADR-0036 §1 status note ("realized, P3-F2") + Phase-3 plan entry; short `docs/` note on
  writing an adapter with `AdapterBase`.

## Reviewer should check
- `AdapterBase` is genuinely optional and did not silently become required (in-core refs untouched).
- No heavy dep leaked into `latence-core`; the guard maps exceptions **without** swallowing a
  real framework `LatenceError` and **without** leaking record content into the message.
- The malformed-profile Runner change fails loud for *malformed* but still treats *absent* as
  "follow request" and does not regress the import-error/load-error path — determinism/resume intact.
- Batching preserves input order and does not buffer the whole stream (ADR-0033 posture).
- Provenance/offset helpers reuse the existing OffsetMap/page seam (ADR-0031), not a weaker copy.

# Passing Provider Conformance (P3-F3) — the adapter author's checklist

The **Provider Conformance suite** is the enterprise-readiness gate: a reusable, pytest suite
parametrized over **every** registered Provider that asserts a fixed contract (C1–C6). "Passes
conformance" *is* the definition of enterprise-ready, and it is the gate every Wave-1 adapter
package must pass to land. Architecture: ADR-0036 §5.

The suite lives in `latence_core.conformance` (dep-light — no torch/model/GPU libs) and runs in
CI on py3.11 + py3.12 with `LATENCE_CUDA=0`. It runs the in-core CPU reference Providers **for
real** and conformance-checks endpoint/GPU Providers against a deterministic **stub** (their real
quality/throughput needs a rig, #64 — flagged, never fabricated).

## What it checks (C1–C6)

| Check | What it asserts |
|---|---|
| **C1** valid typed records | On the canonical input your Provider emits contract-valid records: every record carries `Provenance` + `Classification` and in-range, well-ordered offsets (reusing the S10 completeness + ADR-0031 `PageOffsetIndex` seams). For a **sub-document carrier** (a mention/relation) a cited `page_start`/`page_end` must **be the page the record's own offset resolves to** through the parent document's page map — the machine gate on rule 3 below. The canonical corpus is deliberately multi-page and multi-chunk (its second chunk STRADDLES a page break, so its `page_slice` holds two spans), so a Provider that inherits the chunk's page range instead of resolving **FAILs**. An empty result is allowed **only** where your Capability legitimately produces none. |
| **C2** graceful failure | On an adversarial input (malformed / oversized / injection-bearing / empty) your Provider raises a **typed** `LatenceError` (a `ProviderError`/`ConfigError`), **or** takes your Capability's defined graceful path (a Screening quarantine, a `PARSE_ERROR` disposition, or an empty/no-op result) — **never** a bare crash, hang, or silent-wrong output. The surfaced message / reason carries **no PII or record content** (proven against a PII-bearing payload). |
| **C3** license recorded + permissive-or-opt-in | Your Provider declares a `ProviderProfile` (a missing profile **FAILs**). Its `license` is a real SPDX id or the `UNVERIFIED` sentinel; `UNVERIFIED` must never appear with `license_verified=True`; a **restricted** (non-permissive) license must be `license_verified=True` (flagged opt-in, ADR-0012). |
| **C4** determinism-or-documented | If your profile says `deterministic=True`, running the canonical input **twice** must yield byte-identical output. If not, declare `deterministic=False` — the suite records the declaration, no assertion. Declaring `True` and not being is a **FAIL**. |
| **C5** resource/device honored | A `compute="gpu"` Provider on a CPU-only host is **skipped-with-flag** by `select_device` (recorded, not run, not faked). A `cpu`/`either` Provider runs. Do **not** import a GPU stack at module import (dep-light) — defer it to first use. |
| **C6** no secret/PII leak | A Provider that is **not** the Redactor must not surface raw PII in a record's **non-content** fields (a `record_id`, a `label`, a Provenance/Evidence field) or in logs/spans — the repo's no-PII-in-report discipline. |

## The checklist

1. **Declare a `ProviderProfile`.** A `@classmethod def profile(cls) -> ProviderProfile` (preferred)
   or a `PROFILE` ClassVar. Be honest: `compute` (`cpu`/`gpu`/`either`), an approximate `memory_mb`,
   `model_id` (or `None` for a pure-code / endpoint-client Provider that bundles no weights), the
   verified SPDX `license` (or `UNVERIFIED`/`False` if you could not verify it — **never guess**),
   `deterministic`, `batch`, and `cost_per_1k` (or `None` where deployment-specific).
   - If your Capability is **Profiling**, declare via `PROFILE` ClassVar, **not** a `profile`
     classmethod — the `Profiler` Capability's own I/O method is named `profile`, which would
     shadow the classmethod (see `StatisticalProfiler.PROFILE`).
2. **Fail gracefully, typed.** Map a native-library exception to a `ProviderError` (runtime fault)
   or `ConfigError` (operator bad-config) — `AdapterBase.guard("<op>")` does this for you — and
   **never** put record content / PII in the message. Or take your Capability's defined path (a
   Screening quarantine, a `PARSE_ERROR` record, an empty result).
3. **Keep offsets/Provenance intact.** Every emitted record carries `Provenance` + `Classification`;
   a sub-document carrier (a mention/relation) resolves its char span back to a source page through
   the parent's page spans (reuse `OffsetIndex`/`PageIndexResolver` — `AdapterBase.rebase_offsets`
   / `resolve_pages` — not a weaker copy, and not a hand-rolled page index: since v19 the chunk
   carries its OWN `page_slice`, so one `PageIndexResolver` per call resolves any chunk, in any
   order. A relation's covering span spans two chunks and is answered by
   `pages_for_covering_span(head.provenance, tail.provenance)` instead).
4. **Be deterministic, or say you are not.** Emit records in a stable order; if learned float
   inference makes byte-identical output impossible, declare `deterministic=False`.
5. **Stay dep-light at import.** Defer your heavy `import` to first use so Provider discovery (and
   the CPU-only device seam) never needs torch/openai present.

## Auto-coverage — no silent gaps

A Provider is mapped to its Capability by its entry-point **name prefix** (`entity.gliner` →
Entity), so a new Provider is **auto-covered** the moment its Capability has a `ConformanceCase`.
If you add a Provider for a Capability with no case, the suite **FAILs loudly** (`case_for_provider`
raises) — forcing the case to be written rather than silently skipping. Endpoint/GPU Providers get a
per-Provider stub-scoped override in `latence_core/conformance/cases.py`.

## Running it

```bash
LATENCE_CUDA=0 PYTHONHASHSEED=0 uv run pytest \
  packages/latence-core/tests/test_conformance.py \
  packages/latence-core/tests/test_e2e_conformance.py
```

Or roll the whole gate into a Quality Report programmatically:

```python
from latence_core.capability import ProviderRegistry
from latence_core.conformance import build_conformance_report

report = build_conformance_report(ProviderRegistry())
assert report.all_passed  # every registered provider is enterprise-ready
```

The `ConformanceReport` is counts-only and folds into the run's `QualityReport.conformance`
(`QUALITY_SCHEMA_VERSION` 17) — safe to persist and share.

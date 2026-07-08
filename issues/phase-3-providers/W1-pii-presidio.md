# W1-pii — `latence-pii-presidio`: a real Microsoft Presidio Redaction provider (Wave-1)

**Phase-3 Wave-1. GitHub issue #95 (P3-A-pii).** A real, learned+rule PII Redaction adapter built
through `AdapterBase`, a **drop-in swap** for the in-core `redaction.hybrid_rule` and the
`redaction.gliner_pii` model provider. Presidio (Microsoft, **MIT**) pairs a spaCy NER model with
pattern + context recognizers — a genuinely different detection engine, so it's a real alternative
for the bake-off, and it exercises the part of `AdapterBase` the embedder didn't: **offset/Provenance
mapping** (Presidio returns char spans that must land on the parsed-document coordinate system).

**A working plugin, not a stub** — same honest heavy-ML discipline as the gliner trio + the embedder:
real Presidio `AnalyzerEngine`/`AnonymizerEngine` calls when the dep + spaCy model are present;
offline CI installs the package `--no-deps` and monkeypatches the analyzer with a deterministic fake
for conformance/unit tests; a real-weights test gated on `find_spec("presidio_analyzer")`.

## What to build
Package **`latence-pii-presidio`** registering `redaction.presidio`, an `AdapterBase` subclass
satisfying the `Redactor` Protocol (`redact(documents) -> Iterator[RedactionRecord]`, capability.py:237).
It MUST be a true drop-in — same contract as `redaction.hybrid_rule`:
- **Same `RedactionRecord` output** (contracts.py:809): `masked_content` (the PII-handled RAG text),
  `pii_spans` (list of `PIISpan`: `pii_type`, char offsets, `RedactionAction`), one record per
  `DocumentRecord`. No-detected-PII → masked==source, empty spans (the documented clean case).
- **Metadata-aware policy** — key off the document's `Classification.sensitivity` the way
  `redaction.hybrid_rule` does (mask/replace/hash/tag per policy; `types`/`detectors` narrowing), so
  swapping the provider needs no other config change. **Honor the H-C1 sensitive-doc no-op floor:**
  set `redaction_disabled_for_sensitive` when a sensitive doc's applied policy runs no detectors —
  a disabled control must not read as `documents_with_pii:0` (reuse the core policy/flag machinery;
  don't reinvent it).
- **Offsets are the point:** Presidio's analyzer returns char start/end on the analyzed text; map
  each back onto the **parsed document** coordinate system via AdapterBase's provenance/offset
  helpers (carry_provenance + the ADR-0031 OffsetMap/page seam) — Redaction scans the parsed markdown,
  not the stripped chunk text (capability.py:228). Offsets must round-trip; spans must be
  non-overlapping + stable (the RedactionRecord invariant).
- **AdapterBase for real:** the guard maps a native Presidio/spaCy failure → typed `ProviderError`
  with **NO detected-PII / no source text in the message** (extra care — this provider handles PII);
  batching over documents; `compute="either"` (spaCy runs CPU, GPU-optional). Report any AdapterBase
  misfit (H-A1 discipline).

## Profile + license + conformance
- `ProviderProfile`: `compute="either"`, honest `memory_mb` (spaCy `en_core_web_lg` is ~500MB —
  declare it; pick `en_core_web_sm` as the light default if lg is too heavy for CI, document the
  trade-off), `model_id` = the spaCy model, `deterministic` set truthfully (Presidio analysis is
  deterministic on fixed input+config — verify), `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1):** `license_code` = Presidio SPDX (**MIT**, cited) — and note the
  spaCy library (MIT) + the chosen spaCy model's license in the citation; `license_weights` = the
  spaCy model SPDX (en_core_web_sm/lg are **MIT**, cited HF/spaCy source + date). Verified or honestly
  `UNVERIFIED` — no guessing. Passes conformance **C3**.
- **Passes P3-F3 conformance C1–C6** as a registered provider (offline: against the monkeypatched
  analyzer). The **C2 no-PII-in-error** and **C6 no-leak** checks are especially load-bearing here —
  a PII provider that leaked detected PII into an error/log would be the worst kind of failure.
  Roster count updates; full conformance + stack-validate suites stay green.

## Wire-in
- CI: install `latence-pii-presidio --no-deps` in the heavy-provider install step (with the gliner
  packages); tests monkeypatch `AnalyzerEngine`. Real-weights test gated on the dep.
- Add `redaction.presidio` to `matrix/redaction.yaml` (a bake-off vs `redaction.hybrid_rule` and
  `redaction.gliner_pii`) — dep-gated candidates skip-with-flag offline (#64), the in-core
  `redaction.hybrid_rule` runs real, so the matrix produces a real comparison row. Where a stage has
  a Redaction gold-set, the bake-off reports real P/R/F1; else the stage metric.

## Definition of Done (Gate bar — a security-sensitive PII provider)
- `latence-pii-presidio` builds, registers `redaction.presidio`, is an `AdapterBase` subclass, and is
  a true drop-in for `redaction.hybrid_rule` (same RedactionRecord contract, metadata-aware policy,
  the H-C1 no-op floor) — swapping the provider in a stack needs no other change (prove with a test).
- Offsets round-trip onto the parsed-doc coordinate system; spans non-overlapping + stable; native
  failure → typed `ProviderError` with **no PII/source text** in the message (C2), no leak (C6).
- Evidence-bearing license for Presidio + spaCy + the model (verified or honest UNVERIFIED); passes
  conformance C1–C6; heavy dep isolated (ADR-0016); the conformance + stack-validate suites stay green.
- CI installs `--no-deps` + monkeypatch tests green; a real-weights test gated on the dep;
  `matrix/redaction.yaml` runs (presidio skip-with-flag offline, hybrid_rule real, no fabricated numbers).
- `mypy --strict`/`ruff`/pytest green; deterministic seeded; ADR-0036 + a docs note; the seam-report
  says whether AdapterBase's offset/provenance helpers fit a real redactor.

## Reviewer should check
- True drop-in: a stack with `redaction.presidio` swapped for `redaction.hybrid_rule` produces a
  valid, contract-complete RedactionRecord with correct offsets — and the sensitive-doc no-op floor
  still fires (H-C1 not regressed).
- **No PII leak** anywhere: not in error messages (C2), not in logs, not in the report — proven with a
  PII-bearing input that triggers a native failure.
- Offsets genuinely land on the parsed-doc coordinate system (a real multi-span fixture), spans
  non-overlapping/stable; uses the ADR-0031 seam, not a weaker mapper.
- License really verified for Presidio + spaCy + the model with cited sources, or honest UNVERIFIED.
- The matrix presidio row is skip-with-flag offline (no fabricated quality/throughput); hybrid_rule real.

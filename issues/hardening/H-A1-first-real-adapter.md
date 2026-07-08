# H-A1 — Validate the provider seam with the first REAL Wave-1 adapter

**Source: hostile 360 audit, THEME A (mattpocock: "one adapter = hypothetical seam; two = real").**
The Phase-3 provider seam — `AdapterBase` (P3-F2) + `ProviderProfile` (P3-F1) — is validated only by
tests that fabricate fake subclasses; it has **zero real production consumers**. This slice builds the
**first real Wave-1 adapter THROUGH `AdapterBase`**, turning the hypothetical seam into a real
two-adapter seam and proving (or exposing) that `AdapterBase`'s shape actually fits a real library.
It is also the **template** every subsequent Wave-1 adapter (docling, kreuzberg, paddle, GLiNER
variants, embedders, PII) will copy — so get the ergonomics right here.

**The deliverable is a working, conformant, AdapterBase-based real adapter — the seam validation is
the point, not any one library.** If the chosen library proves unsuitable in CI, substitute a lighter
permissive one and document it; do not flail on a heavy/uninstallable dependency.

## What to build
A new package **`latence-parser-kreuzberg`** — a real second Parser provider (`parser.kreuzberg`)
over the [kreuzberg](https://github.com/Goldziher/kreuzberg) text-extraction library (CPU, permissive
license — verify per below), implemented as an **`AdapterBase` subclass**.
- **Library choice + fallback:** kreuzberg is preferred (CPU, on the Phase-3 plan). **Verify early**
  that it is pip-installable in CI **without system binaries** (pandoc/tesseract). If it requires
  system packages or has an unresolvable/heavy dep, **substitute a pure-Python permissive PDF lib**
  (`pdfplumber`, MIT, or `pypdf`, BSD-3 — both pure-Python, page-aware) and name the package
  accordingly (`latence-parser-pdfplumber`). Record the choice + why. Prefer a library that yields
  **page structure** so the PageMap/offset path is genuinely exercised.
- Heavy/real dep lives ONLY in this package (ADR-0016 thin-core); nothing leaks into `latence-core`.

## It must EXERCISE (and thereby validate) AdapterBase — this is the audit's whole point
Use `AdapterBase`'s facilities for real, not bypass them (if any facility doesn't fit a real parser,
that is a FINDING about AdapterBase — report it, and either fix AdapterBase or document the misfit):
- **native-exception → typed taxonomy:** a malformed/corrupt/password-protected input → the library's
  native exception mapped via AdapterBase's guard to a typed `ProviderError`/PARSE_ERROR, **no raw
  content/PII in the message**. (This is the Parse graceful-failure the conformance C2 check asserts.)
- **offset/Provenance + page preservation:** map the library's output back onto a `DocumentRecord` +
  `PageMap` using AdapterBase's provenance/offset helpers and the existing OffsetMap/page seam
  (ADR-0031) — do NOT hand-roll a weaker mapper (the audit found the in-core GLiNER provider did that;
  don't repeat it). Offsets must round-trip; pages must be correct.
- **device + batching:** a CPU parser declares `compute="cpu"`; exercise AdapterBase's batching path
  over multiple input documents (input-order-preserving, streaming-friendly). Device routing is
  trivially cpu here but must flow through the seam.

## Conformance + profile + license (the gates this template must pass)
- Declare a `ProviderProfile`: `compute="cpu"`, honest `memory_mb`, `deterministic` set truthfully
  (text extraction is typically deterministic — verify), `batch` per the adapter, `model_id=None`
  (pure-code library, no weights).
- **Evidence-bearing license (H-B1):** the profile MUST carry `license_code` (the library's SPDX,
  verified from its repo/PyPI LICENSE with a cited `license_source` URL + `license_verified_on` date),
  `license_weights=None`. If the library's license can't be verified, `UNVERIFIED`/`False` — do not guess.
- **Passes the P3-F3 conformance suite** (C1–C6) as a registered provider — this is the enterprise-
  readiness gate, and the first real adapter passing it end-to-end is the proof the gate + seam compose.
- Registered via the `latence.providers` entry point. **Namespace:** use `parser.kreuzberg` to match
  the existing `parser.plaintext`/`parser.document` (the THEME-G namespace cleanup, #109, will settle
  parse.* vs parser.* repo-wide — follow the existing `parser.*` convention here).

## Report on the seam (the audit's actual question)
In the PR, explicitly answer: **did `AdapterBase` fit a real adapter, or did it need changes?** Which
helpers were used as-is, which were awkward, what (if anything) about `AdapterBase`/`ProviderProfile`
you had to adjust. If a `ProviderProfile` field (`cost_per_1k`, `memory_mb`) was still inert for this
adapter, say so — that informs the trim-vs-keep call for the remaining inert fields.

## Definition of Done (Gate bar)
- `latence-parser-kreuzberg` (or the documented pure-Python substitute) builds, is an `AdapterBase`
  subclass, registers `parser.<name>`, and parses a real PDF/document → `DocumentRecord` + `PageMap`
  with round-tripping offsets and correct pages (a real fixture, not a mock).
- Native parse failure → typed error, no PII in the message; graceful (C2).
- Evidence-bearing `ProviderProfile` (cited, verified license or honest UNVERIFIED); passes P3-F3
  conformance C1–C6; the full conformance suite stays green with the new provider in the roster.
- Heavy dep isolated to the package (ADR-0016); CI installs + tests it (add the lane/wiring if needed,
  the way the MinIO/airflow lanes are isolated). `mypy --strict`/`ruff`/pytest green; deterministic.
- The PR's seam-report answers whether AdapterBase fit, with any AdapterBase changes justified.
- ADR-0036 note: the seam now has a real second consumer (hypothetical → real).

## Reviewer should check
- The adapter genuinely uses AdapterBase (guard + provenance/offset helpers), not bypasses it; a real
  parse failure maps to a typed error with no content leak.
- Offsets round-trip and pages are correct on a real multi-page fixture (not a trivial one).
- License is really verified (cited source substantiates the SPDX for the actual installed version) or
  honestly UNVERIFIED — no rubber-stamp.
- No heavy dep leaked into `latence-core`; the conformance suite still passes for ALL providers.
- The seam-report is honest about AdapterBase's fit (this decides trim-vs-keep for the inert fields).

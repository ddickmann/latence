# H-B1 — Make "verified license" honest-by-construction (not a self-attestation)

**Source: hostile 360 audit, THEME B. Flagged by 3 independent lenses (design, security, test-honesty).**
`ProviderProfile.license_verified: bool` is a hardcoded human claim. The only validator forbids the
single pair `license=="UNVERIFIED" AND verified==True`; nothing else inspects a real license, and the
tests assert the constant back. A provider could ship `license="Apache-2.0", license_verified=True`
on a CC-BY-NC checkpoint and pass every gate. The research-diligence directive (ADR-0012) requires
weights **and** code verified **separately**, with a citation — the profile records none of that.
This lands on my own P3-F1 `entity.gliner` claim. Make the claim un-makeable without its evidence.

Files: `packages/latence-core/src/latence_core/providers/profile.py`, the in-core provider profiles,
`packages/latence-ner-gliner/.../provider.py`, the P3-F3 conformance C3 check, `NOTICE`,
`THIRD-PARTY-LICENSES.md`.

## The fix — evidence-bearing verification

### 1. `ProviderProfile` carries the verification evidence, and `verified=True` requires it
Extend the descriptor so a verification claim is inseparable from its proof (all additive,
backward-compatible defaults; keep the existing `UNVERIFIED` invariant):
- `license_code: str` — SPDX of the **code/library** (a provider's own package code = the repo's
  Apache-2.0).
- `license_weights: str | None` — SPDX of the **weights**, or `None` for a pure-code provider (no
  model). ADR-0012 requires these **separately**; `license` (existing) stays as the headline/effective
  SPDX but must be consistent with these.
- `license_source: str | None` — a citation for the verification (the HF model-card URL, the GitHub
  LICENSE URL, the repo's own LICENSE path) — where a human actually checked.
- `license_verified_on: str | None` — ISO date (`YYYY-MM-DD`) the check was performed.
Add a `model_validator`: **if `license_verified is True` then `license_source` and
`license_verified_on` MUST be non-empty, and `license_code` must be set (plus `license_weights` for a
model-bearing provider, i.e. when `model_id is not None`).** So "verified" cannot be asserted without
naming what was checked, where, and when — honest-by-construction. Keep `UNVERIFIED ⇒ not verified`.
(Optionally expose `license_verified` as a computed property derived from the evidence rather than a
free bool — either shape is fine as long as a bare True without evidence is impossible.)

### 2. The P3-F3 conformance C3 gate enforces it across every provider
Extend C3 so it FAILs a provider that: claims `license_verified=True` without a complete evidence
trail (source + date + code SPDX, + weights SPDX when `model_id`); declares a **restricted**
license (non-OSI / Gemma family) that is not flagged opt-in; or has `license`/`license_code`/
`license_weights` mutually inconsistent. Add the bite-proof (a fake provider claiming verified-without-
evidence FAILs C3). This turns the conformance suite into the real enforcement point the audit wanted.

### 3. Reconcile the existing providers to the evidence standard
- **`entity.gliner`**: it already documents the verification in its module docstring (gliner lib
  Apache-2.0 code + `urchade/gliner_multi-v2.1` Apache-2.0 weights, checked 2026-07-06, with the
  v2.1-relicensed-from-CC-BY-NC note). **Move that into the structured fields** (`license_code`,
  `license_weights`, `license_source` = the HF model-card + GitHub LICENSE URLs, `license_verified_on`
  = 2026-07-06). This is my P3-F1 claim made honest, not retracted — the evidence exists.
- **In-core pure-code providers** (gazetteer, pattern, markdown/sentence chunkers, screeners,
  profilers, graph/disambiguation, exports): `license_code = "Apache-2.0"` (the repo's own code),
  `license_weights = None`, `license_source` = the repo LICENSE path/URL, a fixed `license_verified_on`.
- Any provider whose weights license is **not** actually verifiable in this session → set
  `license = UNVERIFIED`, `license_verified = False` (do NOT guess — the directive). Flag it.

### 4. Fix the NOTICE / THIRD-PARTY EmbeddingGemma overclaim
`NOTICE:38-39` and `THIRD-PARTY-LICENSES.md:95` state EmbeddingGemma "is available **only as an
explicitly opt-in Provider that surfaces its license and flow-down obligations at install time**."
No such provider package exists in the repo — this describes install-time machinery that isn't there.
Rewrite to state the **policy** accurately (the Gemma family is non-default and, *were* it offered,
would be an opt-in provider with flow-down) without asserting a shipped mechanism. Do not claim
machinery that doesn't exist; that's the same overclaim pattern this whole workstream is closing.

## Definition of Done (Gate bar)
- `ProviderProfile` verification evidence fields + the `verified⇒evidence` validator; `UNVERIFIED`
  invariant preserved. Additive/backward-compatible; `mypy --strict` clean over src+tests.
- Every in-core provider + `entity.gliner` reconciled to carry real evidence (or honestly `UNVERIFIED`).
- P3-F3 C3 gate enforces the evidence rule with a bite-proof; the full conformance suite stays green.
- A deterministic test proving `license_verified=True` without source/date/code-SPDX **raises**
  (the theater is now impossible), and that the `UNVERIFIED` invariant still holds.
- NOTICE + THIRD-PARTY-LICENSES corrected re: EmbeddingGemma (policy, not phantom machinery).
- ADR-0012 note describing the evidence-bearing verification contract; `ruff` clean; CI green.

## Reviewer should check
- A bare `license_verified=True` with no evidence is genuinely **rejected** (would fail on a fake
  provider), and C3 bites on it — not just documented.
- `entity.gliner`'s evidence is real (the cited URLs actually state Apache-2.0 for both v2.1 weights
  and the library) — verify, don't rubber-stamp; if any link doesn't substantiate, drop to UNVERIFIED.
- weights vs code are recorded **separately** (ADR-0012), and `license`/`license_code`/`license_weights`
  can't silently disagree.
- The NOTICE rewrite no longer claims a non-existent install-time opt-in mechanism.
- No provider was flipped to `verified` just to satisfy the new required fields — UNVERIFIED where real.

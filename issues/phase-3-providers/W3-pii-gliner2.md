# W3-pii-gliner2 — `redaction.gliner2`: GLiNER2 (fastino) PII Redaction Provider

**A learned PII Redaction Provider over the `gliner2` framework** (`fastino/GLiNER2-Guardrails-PII-Multi`,
**Apache-2.0** weights — verified 2026-07-09 from the HF model card; not gated). A **twin of the existing
`redaction.gliner_pii`** (latence-pii-gliner): it reuses ALL the shared redaction machinery (the
`RedactionPolicy`, masking/placeholder/action logic, `PIISpan`/`RedactionRecord` contracts, offset
handling) and swaps ONLY the detection call — from the `gliner` library to the **`gliner2`** framework,
which is a *different* Python package + API. Registers `redaction.gliner2`, a true drop-in behind the
Redaction Capability (swap `provider: redaction.hybrid_rule` → `provider: redaction.gliner2`).

## The REAL gliner2 API (introspected + run on the pod — pin to this)
```python
from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi")   # mdeberta-v3 encoder
out = model.extract_entities(text, entity_types, threshold=0.5,
                             include_spans=True, include_confidence=True)
# -> {"entities": {"<label>": [{"text": str, "confidence": float, "start": int, "end": int}, ...], ...}}
```
**Verified real output** for `"IBM works with Alice Johnson in Berlin. Contact alice@example.com or
+1-202-555-0143."` with types `["person","email","phone number"]`, threshold 0.4:
`{"entities": {"person":[{"text":"Alice Johnson","confidence":0.997,"start":15,"end":28}],
"email":[{...,"start":48,"end":65}], "phone number":[{...,"start":69,"end":84}]}}`. Offsets are
document-char `start`/`end` (half-open), exact — map directly to `PIISpan(char_start=start,
char_end=end, pii_type=label, confidence=confidence)`.

## What to build (mirror latence-pii-gliner exactly)
Package **`latence-pii-gliner2`** registering `redaction.gliner2`, structured as a near-clone of
`latence-pii-gliner` (`GlinerPIIRedactor`): same `pii_labels` REQUIRED config, same
`RedactionPolicy`/`default_action`/`policies` machinery imported from `latence_core.stages.redaction`
+ `latence_core.redaction_policy`, same `redact(documents) -> Iterator[RedactionRecord]` shape, same
per-document masked variant + audited PIISpans, same PARSE_ERROR pass-through, same metadata-aware
policy keyed on `classification.sensitivity`. **Change only:**
- `_load_model`: `from gliner2 import GLiNER2; GLiNER2.from_pretrained(self._model_id)` (lazy import;
  the heavy dep `gliner2` lives ONLY in this package, ADR-0016). Default `model_id =
  "fastino/GLiNER2-Guardrails-PII-Multi"`.
- `_detect_spans`: call `model.extract_entities(text, self._labels, threshold=self._threshold,
  include_spans=True, include_confidence=True)`, then flatten the `{"entities": {label: [rows]}}` dict
  → one `PIISpan` per row (`pii_type=label`, `char_start=row["start"]`, `char_end=row["end"]`,
  `confidence=row["confidence"]`, `detector="gliner2:<label>"`), resolve overlaps + apply the policy
  allow-list exactly as latence-pii-gliner does (reuse its span-building helper shape). Deterministic
  span order (by start, then label). Slice `text[start:end]` for the surface (bounds-safe).
- `AdapterBase`: `compute="either"` (gliner2/mdeberta runs on CPU or GPU via the device seam,
  skip-with-flag), native fault → typed `ProviderError` with NO record/PII text in the message (C2/C6).

## Anti-false-green (the PaddleOCR/relex lesson — non-negotiable)
The offline stub MUST be faithful to the REAL `gliner2` API above: a fake `GLiNER2` whose
`extract_entities(..., include_spans=True, include_confidence=True)` returns the real
`{"entities": {label: [{text,confidence,start,end}]}}` dict shape. A provider that mis-parses that
shape (e.g. treats it as a flat list, or forgets the per-label dict nesting) MUST fail an anchor test.
The real-weights test is dep-gated (`find_spec('gliner2')`) + skipped offline; I re-validate on the pod
(person/email/phone masked, offsets round-trip).

## Profile + license + conformance
- `ProviderProfile`: `compute="either"`, honest `memory_mb`, `model_id=
  "fastino/GLiNER2-Guardrails-PII-Multi"`, `deterministic` truthfully, `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1):** `license_weights` = **Apache-2.0** (fastino model card, cited +
  dated 2026-07-09). `license_code` = the `gliner2` library license — **VERIFY it during the build**
  (check the fastino/GLiNER2 GitHub LICENSE); record it cited, or mark UNVERIFIED and declare the
  provider's own code license only. Passes conformance C3.
- Passes P3-F3 conformance C1–C6 offline against the faithful stub; roster +1; conformance +
  stack-validate green.

## Wire-in
- CI/verify-local: `latence-pii-gliner2 --no-deps` in the heavy-provider install (ci.yml +
  scripts/verify-local.sh) + wheel list; add `redaction.gliner2` to `matrix/redaction.yaml`.
- `scripts/validate-pii-gliner2-on-gpu.md`: turnkey pod steps (`pip install gliner2` + a real PII run).

## Definition of Done
- `redaction.gliner2` builds, registers, AdapterBase, drop-in for `redaction.hybrid_rule`
  (same RedactionRecord contract); faithful stub; un-faithful dict-parse FAILS an anchor test;
  masked variant + audited PIISpans + policy allow-list + PARSE_ERROR pass-through; deterministic.
- Evidence-bearing Apache-2.0 weights license (cited); gliner2 lib license verified-or-UNVERIFIED;
  passes conformance C1–C6; `bash scripts/verify-local.sh` green; ADR-0036 note; seam-report.
- Native fault → typed error, no PII leak. Pod-validated: real gliner2 masks person/email/phone with
  exact offsets (I run this over SSH; numbers folded in, never fabricated).

## Reviewer should check
- The stub is faithful to the REAL gliner2 `{"entities": {label: [{text,confidence,start,end}]}}`
  TUPLE-of-dicts shape (un-faithful provider FAILS) — the false-green trap.
- Reuses the shared redaction machinery (policy/masking/PIISpan) — NOT a reinvented redactor.
- Offsets map directly (start/end → char_start/char_end); overlaps resolved; policy allow-list applied.
- Weights Apache-2.0 (cited); gliner2 lib license recorded; heavy dep isolated; native fault no-PII-leak.

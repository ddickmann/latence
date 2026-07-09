# Validate `redaction.gliner2` on a real GPU/CPU — turnkey pod steps (W3-pii-gliner2)

`redaction.gliner2` (latence-pii-gliner2) is the **twin** of `redaction.gliner_pii`, over the
fastino **GLiNER2** framework (`fastino/GLiNER2-Guardrails-PII-Multi`, mdeberta-v3 encoder,
**Apache-2.0** weights + code). Offline it is validated against a **faithful** fake `gliner2`
whose `extract_entities(..., include_spans=True, include_confidence=True)` returns the REAL
`{"entities": {label: [{text, confidence, start, end}]}}` per-label-dict shape — a provider that
mis-parses that as a flat list (the `gliner`-library shape) FAILS `test_flat_list_misparse_fails_
anchor`. This script is the turnkey re-validation that proves the REAL weights flow through the
call/parse/mask path end to end: person + email + phone masked, offsets round-trip.

> **The false-green catch (what this run must prove).** The offline slice is validated against a
> faithful per-label-dict fake. But only a real `gliner2` run proves the actual checkpoint exposes
> `extract_entities` with the pinned kwargs and returns the pinned nested shape, and that the
> offsets round-trip on real output. If step 3 raises on `extract_entities` (wrong signature) or
> returns a flat list, the pinned API is wrong — stop and re-introspect. Do NOT fabricate the
> masking/offset numbers; they come from this run only.

`gliner2`/mdeberta-v3 runs on CPU or a modest GPU (compute="either") — this is a compact encoder
model (~1 GB). A CUDA device accelerates it but is not required.

## 0. Prereqs

```bash
python --version               # 3.11+
nvidia-smi                     # optional — a CUDA device accelerates but is not required
```

## 1. Install the framework + this provider WITH its heavy deps

```bash
# from a clone of the repo on the pod:
cd latence-framework
pip install -e packages/latence-core
# The provider declares gliner2[local] (torch/transformers live only in that extra — the
# base gliner2 dist ships neither), so an editable install pulls the full local-inference stack.
pip install -e packages/latence-pii-gliner2   # pulls gliner2[local] (transformers/torch, mdeberta-v3)
python -c "from gliner2 import GLiNER2; print('gliner2 ok')"
```

## 2. Verify the license on the pod (weights + library) — record cited, never guessed

```bash
python - <<'PY'
import importlib.metadata as md
# Library license: the installed gliner2 dist metadata (Apache-2.0 expected).
meta = md.metadata("gliner2")
print("gliner2 version:", meta["Version"])
print("gliner2 License:", meta.get("License"))
print("gliner2 classifiers:", [c for c in meta.get_all("Classifier", []) if "License" in c])
# Weights license: read the HF model card (Apache-2.0, not gated) — done at build time,
# re-confirm here from the downloaded snapshot's README if offline mirroring.
PY
```

## 3. Real PII masking + offset round-trip (the core claim)

Runs the reference Provider (`Gliner2PIIRedactor`) against the real weights over the canonical
sentence and checks: (a) `extract_entities` returns the nested `{"entities": {label: [rows]}}`
dict, (b) person/email/phone are detected + masked, (c) each span's slice round-trips
(`text[char_start:char_end]` == the raw surface), (d) the raw PII never appears in a stored span.

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"          # route the device seam to cuda if present (else cpu)
from latence_core.contracts import Classification, DocumentRecord, PageMap, Provenance
from latence_pii_gliner2.provider import Gliner2PIIRedactor

TEXT = ("IBM works with Alice Johnson in Berlin. Contact alice@example.com "
        "or +1-202-555-0143.")
prov = Provenance(source_uri="file://d1.txt", file_name="d1.txt", file_type="txt",
                  file_size=len(TEXT), document_id="d1", char_start=0, char_end=len(TEXT),
                  page_start=1, page_end=1)
doc = DocumentRecord(record_id="d1", provenance=prov,
                     classification=Classification(language="en", sensitivity="public"),
                     content=TEXT, page_map=PageMap.from_page_texts([TEXT]).page_map)

# Direct API check: the pinned per-label-dict contract.
from gliner2 import GLiNER2
raw = GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi").extract_entities(
    TEXT, ["person", "email", "phone number"], threshold=0.4,
    include_spans=True, include_confidence=True)
assert isinstance(raw, dict) and "entities" in raw, f"expected nested dict, got {type(raw)}"
assert isinstance(raw["entities"], dict), "entities must be a per-label dict, NOT a flat list"
print("extract_entities returned per-label dict with labels:", list(raw["entities"]))

# Through the reference Provider.
red = Gliner2PIIRedactor({"pii_labels": ["person", "email", "phone number"],
                          "threshold": 0.4, "device": "cuda"})
rec = next(iter(red.redact([doc])))
ok = True
for s in rec.pii_spans:
    surf = TEXT[s.char_start:s.char_end]   # the slice is the truth; PIISpan never stores the raw surface (S6)
    print(f"  SPAN {s.pii_type:14s} off[{s.char_start}:{s.char_end}] slice={surf!r} conf={s.confidence:.3f}")
types = {s.pii_type for s in rec.pii_spans}
print("masked:", rec.masked_content)
print("types detected:", types)
# Expected (pod-verified): person, email, phone number all masked; the raw values gone.
assert "Alice Johnson" not in rec.masked_content
assert "alice@example.com" not in rec.masked_content
assert "+1-202-555-0143" not in rec.masked_content
assert all("alice@example.com" not in s.model_dump_json() for s in rec.pii_spans)
print("RESULT:", "PASS" if rec.pii_spans else "FAIL — inspect above")
raise SystemExit(0 if rec.pii_spans else 1)
PY
```

## 4. Conformance on the pod

```bash
python -m pytest packages/latence-core/tests/test_conformance.py \
                 packages/latence-core/tests/test_e2e_conformance.py -q \
                 -k gliner2
```

## 5. Bake-off row over a real corpus (optional)

```bash
LATENCE_CUDA=1 latence bake-off matrix/redaction.yaml
# redaction.gliner2 (device: cuda) is measured for real here; on a CPU-only host it is
# skipped-with-flag with no fabricated numbers.
```

Record the real masked types + offsets + confidences from step 3 back into ADR-0036 (folded in
after the pod run, flagged "pending pod validation" until then — never fabricated).

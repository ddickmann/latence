# Validate `parser.lighton` (LightOn-OCR) on a real GPU — turnkey pod steps (W1-lighton, #91)

The offline slice is validated against a **faithful generate/decode fake** (transformers/pypdfium2/PIL
are monkeypatched; the fake `model.generate` returns the real *full* prompt+continuation token
sequence and `LightOnOcrProcessor.decode` renders the sliced continuation — so the Provider's
prompt-token slice `output_ids[0, inputs["input_ids"].shape[1]:]` is genuinely exercised and an
un-sliced decode would fail). The **real** OCR numbers — extracted text fidelity, offset/PageMap
round-trip on a real scanned page, latency/throughput, and a live conformance verdict — come from a
**CUDA pod run**. Until this runs, the GPU quality is explicitly **"pending pod validation"**, never
fabricated.

> **Text-fidelity check (the false-green catch).** On the pod, the real run in step 3 must extract
> **legible, non-empty markdown** from a real scanned page — NOT prompt/chat scaffolding. If the
> extracted text is prefixed with template tokens (e.g. an `<|...|>` marker or the user turn), the
> prompt-token slice is wrong; the offset round-trip alone will not catch that on a blank synthetic
> image, so eyeball the markdown against the page and record a sample.

These steps are turnkey: run them on a single ≥16 GB CUDA GPU (an H100 slice, an A10, an RTX 4090, a
RunPod pod, …). The model is ~3.5 GB VRAM in bf16.

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device is visible
python --version               # 3.11+
```

## 1. Install the framework + this provider WITH its heavy deps

```bash
# from a clone of the repo on the pod:
cd latence-framework
pip install -e packages/latence-core
pip install -e packages/latence-parser-lighton     # pulls transformers + torch + pillow + pypdfium2

# LightOn-OCR support may need a recent transformers (install-from-source until it lands in a
# release — per the model card). If `from transformers import LightOnOcrForConditionalGeneration`
# fails, install transformers from source:
pip install --upgrade "git+https://github.com/huggingface/transformers.git"
python -c "from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor; print('ok')"
```

## 2. Sanity: the device seam routes to cuda

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"          # or leave unset — a real probe finds the GPU
from latence_parser_lighton.provider import LightOnOcrParser
p = LightOnOcrParser({"device": "cuda"})
print("device:", p.device.device, "skipped:", p.device.skipped)   # expect: cuda False
PY
```

## 3. Real OCR over a real scanned page — extracted text + offset round-trip

Point `LIGHTON_REAL_IMAGE` at a real scanned-document image (a photo/scan of a page with legible
text), then run the dep+CUDA-gated real-OCR test (it setenv-overrides any CPU pin to route to cuda):

```bash
export LIGHTON_REAL_IMAGE=/path/to/a/real/scanned_page.png
PYTHONHASHSEED=0 pytest \
  packages/latence-parser-lighton/tests/test_lighton_e2e_runner.py::test_real_lighton_ocr_on_cuda \
  -q -s
# It renders the real image, runs the REAL model on cuda, and asserts a PARSED record whose page
# map's offsets round-trip. Inspect the extracted markdown with -s.
```

Or drive the provider directly to eyeball the markdown quality:

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"
from pathlib import Path
from latence_core.contracts import Classification, ParserInput, Provenance
from latence_parser_lighton.provider import LightOnOcrParser

data = Path(os.environ["LIGHTON_REAL_IMAGE"]).read_bytes()
inp = ParserInput(
    provenance=Provenance(source_uri="file://x.png", file_name="x.png", file_type="png",
                          file_size=len(data), document_id="sha256:x"),
    classification=Classification(language="und"),
    content=data,
)
(rec,) = list(LightOnOcrParser({"device": "cuda"}).parse([inp]))
print("disposition:", rec.disposition)
print("pages:", len(rec.page_map.pages) if rec.page_map else None)
print("---- markdown ----")
print(rec.content)
PY
```

## 4. Live conformance verdict on the GPU (C1–C6 actually run, not skipped)

On this CUDA host the `compute="gpu"` provider is NOT device-skipped, so C1/C2/C4/C6 execute for
real against the live model:

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
v = run_conformance(ProviderRegistry(), "parser.lighton")
print("passed:", v.passed, "device_skipped:", v.device_skipped)
for c in v.checks:
    print(f"  {c.check}: {c.status.value} — {c.detail}")
PY
```

Note: the default Parse conformance case runs the provider on a small **text** canonical input.
LightOn-OCR is an image model, so for a live C1 you may prefer the real-OCR test in step 3 (a real
scanned image) as the authoritative C1/round-trip evidence, and use step 4 for C3/C5 + the graceful
C2 path. Record whichever you run.

## 5. A real bake-off row (real latency/throughput, no fabricated numbers)

```bash
# On the pod, with a scanned corpus at $CORPUS:
LATENCE_CUDA=1 latence bake-off matrix/parse.yaml
# parser.lighton now RUNS (device: cuda) and is measured — its row carries real quality/latency,
# no longer skipped-with-flag. Fold the numbers into the PR (flagged "pod-measured <date>").
```

## What to fold back into the PR after the pod run

- The extracted-markdown sample + the offset round-trip result from step 3.
- The live conformance verdict from step 4 (C1–C6 statuses on the GPU).
- The real bake-off row (latency/throughput/VRAM) from step 5.
- Confirm on the pod whether greedy decoding is byte-reproducible; if it is on this rig, note it (the
  profile still declares `deterministic=False` conservatively for cross-environment honesty).

Until this pod run is folded in, the GPU quality/latency for `parser.lighton` is **pending pod
validation** — the offline slice proves the adapter code, the seam, the shape-matching fake, the
license, and the graceful-failure discipline, but NOT the real OCR numbers.

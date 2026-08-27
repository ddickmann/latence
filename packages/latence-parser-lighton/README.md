# latence-parser-lighton

A **GPU, image→markdown OCR Parser** Provider (`parser.lighton`) for the
[Latence framework](https://github.com/ddickmann/latence), over
[**LightOn-OCR**](https://huggingface.co/lightonai/LightOnOCR-1B-1025) (`lightonai/LightOnOCR-1B-1025`)
— a compact 1B vision-language OCR model, **Apache-2.0 weights AND code** — built through
`AdapterBase` (H-A1), exactly like `parser.pdfplumber`.

It unlocks what the CPU text-layer parsers cannot read: a **scanned / image-only** page (no text
layer). `parser.pdfplumber` / `parser.document` extract an existing text layer; this Provider
**reads pixels**, emits **markdown per page**, and takes that markdown directly onto the exact-offset
`PageMap.from_page_texts` seam (ADR-0031) — the same page/offset path pdfplumber uses, never a
reinvented mapper. No bbox/line model to port: LightOn-OCR is end-to-end image→markdown.

```python
# In a Pipeline YAML — swap the Parser to the OCR one with one line:
#   parse:
#     capability: parse
#     provider: parser.lighton
#     config: { device: cuda }
```

## Install

```bash
pip install latence-parser-lighton   # pulls transformers + torch + pillow + pypdfium2 (heavy, GPU)
```

Heavy deps ship **here**, never in `latence-core` (ADR-0016), and are **lazy-imported** so Provider
discovery stays torch-free. This package is **not** installed into the CPU-first dev/CI env; its
tests monkeypatch the model with a shape-matching fake, and the real OCR is validated on a CUDA pod
(see `scripts/validate-lighton-on-gpu.md`).

## Device gating (the G1 skip-with-flag posture)

The profile declares `compute="gpu"`. On a CPU-only host the shared device seam returns the
**skip-with-flag** decision — the Runner/conformance record the skip rather than fabricating a CPU
run of a GPU model (no fake numbers, #64). On a CUDA rig (`LATENCE_CUDA=1` / a real GPU) it routes to
`cuda` and the model is placed there (`model.to("cuda")`). CPU inference of a 1B VLM is far too slow
to run as a real Parser, so `either` is not declared — the honest compute requirement is GPU.

## The anti-false-green rule (baked in from the PaddleOCR-VL false green)

The single thing whose divergence made the dropped PaddleOCR-VL attempt a false green was a
**fabricated output shape**. Here the offline test double is **faithful to the exact real** model
API — prompt tokens included:

```text
inputs        = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt")
inputs        = inputs.to(device)
output_ids    = model.generate(**inputs, max_new_tokens=..., do_sample=False)   # FULL: prompt + continuation
generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]                    # SLICE off the prompt tokens
markdown      = processor.decode(generated_ids, skip_special_tokens=True)       # -> str (one page)
```

HF `generate` returns the **full** sequence (prompt + continuation), so the Provider MUST slice the
`inputs["input_ids"].shape[1]` prompt tokens off before decoding — otherwise the chat/prompt
scaffolding is prepended to the OCR markdown on every page, corrupting content, PageMap offsets,
size, and language (verified 2026-07 from the
[HF model card](https://huggingface.co/lightonai/LightOnOCR-1B-1025)). The offline fake is faithful
to *that*: `_FakeModel.generate` returns prompt+continuation tokens and `_FakeProcessor.decode`
renders the sliced continuation — so an un-sliced Provider would **fail**, not pass green. A reviewer
can point at the `test_offline_fake_is_faithful_to_generate_slice_decode_contract` test to confirm
the double matches the real generate → slice → decode contract. A real-OCR test
(`test_real_lighton_ocr_on_cuda`) is present, gated on `transformers` + a genuine CUDA probe, and
**overrides** any `LATENCE_CUDA=0` pin so it truly routes to cuda on a rig — skipped offline.

## Security discipline (OCR output is untrusted, PII-bearing)

* A model/CUDA/transformers fault maps through `AdapterBase.guard` to a typed `ProviderError`
  carrying only the provider + op + native *type name* — **never** image bytes or OCR text (C2/C6).
  That typed error is degraded per-document to a graceful `PARSE_ERROR` record (ADR-0020), so one
  bad page never aborts a run.
* An **offset-0 type-spoof guard** (mirroring `parser.pdfplumber` / #62) rejects a `.pdf`/image whose
  bytes do not match the declared type before the rasteriser/decoder — defense-in-depth.

## Seam-report (H-A1: did AdapterBase fit an OCR parser?)

`AdapterBase` fit cleanly — the *same* four seams the pdfplumber Parser uses carried the OCR path:

* **device** — `compute="gpu"` exercises the seam's **skip-with-flag** branch for real (pdfplumber
  was trivially CPU); the resolved `self.device.device` flows into `model.to(...)` (placement stays
  the adapter's job, ADR-0003).
* **guard** — the native transformers/CUDA fault maps to a typed `ProviderError` with no OCR/PII.
* **batched** — batches over input documents, input-order-preserving, streaming-friendly (ADR-0033).
* **carry_provenance + `PageMap.from_page_texts`** — markdown-per-page → the exact-offset page seam;
  offsets round-trip, pages correct — **no** reinvented mapper.

The one OCR-specific addition outside the base is **rasterisation** (PDF/image bytes → page images
via `pypdfium2`/`pillow`), which is upstream of the seam (it produces the model input), not a
weaker offset mapper. No `AdapterBase` change was needed.

## License (ADR-0012, verified 2026-07-08)

Recorded evidence-bearingly (H-B1), weights and code **separately**:

* **code** — `transformers` loader path + the LightOn-OCR code: **Apache-2.0**.
* **weights** — `lightonai/LightOnOCR-1B-1025`: **Apache-2.0**
  ([model card](https://huggingface.co/lightonai/LightOnOCR-1B-1025)).

Both permissive; the headline `license` is the weights SPDX (`Apache-2.0`). Passes conformance C3.

<!-- GENERATED FILE — do not edit by hand.
     Source of truth: deploy/env/compatibility-matrix.yaml
     Regenerate:      latence env matrix deploy/env/compatibility-matrix.yaml --render
     Verify:          latence env matrix deploy/env/compatibility-matrix.yaml                       -->


# Blackwell / CUDA-13 compatibility matrix (blackwell-cuda13)

Which learned Provider and which serving dependency of the enterprise-sota multilingual stack is known-good on which CUDA / torch / transformers combination on Blackwell (sm_120) — and, with equal weight, which combination nobody has measured yet.

* **Reference hardware:** Linux x86-64, NVIDIA Blackwell (compute capability 12.0, `sm_120`), CUDA-13 driver. The live run this matrix records was on such a pod; the field report does not record the exact SKU, so the SKU is deliberately not stated here rather than guessed. (A related 2026-07-14 run on an RTX PRO 4500 Blackwell / torch 2.8+cu128 is recorded in ADR-0036 §W9-scaling.)
* **Verification date:** 2026-07-17
* **Source of truth:** `deploy/env/compatibility-matrix.yaml` (this page is generated from it)
* **Check it:** `latence env matrix deploy/env/compatibility-matrix.yaml` — every cell is a measurement or `UNVERIFIED`; the verifier refuses a `verified` cell with no evidence and an `UNVERIFIED` cell with no reproduce command.

Two axes per cell. **resolution** — does the combination install from the pinned environment (offline, machine-checked on every commit). **runtime** — does it *run* on Blackwell `sm_120`, which needs the hardware. A green resolution is not a runtime claim.

## `deploy/env/cpu-text/`

| Component | Kind | Version | Resolution | Runtime | Notes |
| --- | --- | --- | --- | --- | --- |
| `torch` | distribution | `2.13.0+cpu` | **verified** | `UNVERIFIED` | CPU build on purpose: this environment runs the learned text Providers without CUDA, and is the leg that keeps working when the GPU legs are unavailable. Measure with `bash deploy/env/cpu-text/smoke.sh`. |
| `transformers` | distribution | `4.57.6` | **verified** | `UNVERIFIED` | The 4.x side of the split (FM-TRANSFORMERS-SPLIT); capped <5 by the GLiNER family. Measure with `bash deploy/env/cpu-text/smoke.sh`. |
| `tokenizers` | distribution | `0.22.2` | **verified** | `UNVERIFIED` | Inside transformers 4.x's >=0.22,<=0.23.0 window; the classic silent-drift dependency. Measure with `bash deploy/env/cpu-text/smoke.sh`. |
| `huggingface-hub` | distribution | `0.36.2` | **verified** | `UNVERIFIED` | Held below 1.x, which transformers 4.x cannot use. Blackwell-specific operational note from the pod walkthrough: the hub's XET transfer backend has hung on fresh Blackwell pods — HF_HUB_DISABLE_XET=1 is the documented workaround (docs/TUTORIAL-fresh-pod-walkthrough.md). Measure with `bash deploy/env/cpu-text/smoke.sh`. |

## `deploy/env/served-ocr-server/`

| Component | Kind | Version | Resolution | Runtime | Notes |
| --- | --- | --- | --- | --- | --- |
| `vllm` | distribution | `0.24.0` | **verified** | `UNVERIFIED` | Pinned exactly. 0.25.1 declares torchcodec>=0.14, which this environment excludes, so a bump fails at resolve rather than landing an ABI-mismatched wheel on a pod (FM-WHEEL-ABI). Serving settings that came from the live run: FM-FLASHINFER-SM120 and FM-MULTIMODAL-OOM. Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |
| `torch` | distribution | `2.11.0+cu130` | **verified** | `UNVERIFIED` | The +cu130 wheel line is the one carrying sm_120 kernels; a default-PyPI cu12 torch is the "no kernel image is available for execution on the device" failure. Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |
| `transformers` | distribution | `5.14.1` | **verified** | `UNVERIFIED` | The 5.x side of the split. It exists ONLY here, behind the endpoint boundary. Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |
| `flashinfer-python` | distribution | `0.6.12` | **verified** | `UNVERIFIED` | The component of FM-FLASHINFER-SM120. Present and pinned; its SAMPLER is disabled by serve.sh on Blackwell, which keeps cudagraphs on (this is not --enforce-eager). Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |
| `flash-attn` | excluded_distribution | — | **verified** | `UNVERIFIED` | Made IMPOSSIBLE to install (constraints.txt pins flash-attn<0). vLLM vendors its own attention backends and never imports the standalone package. See FM-WHEEL-ABI. Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |
| `torchcodec` | excluded_distribution | — | **verified** | `UNVERIFIED` | Excluded for the same reason; the OCR path decodes no audio or video. FM-WHEEL-ABI. Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |
| `lightonai/LightOnOCR-2-1B` | served_model | — | **verified** | `UNVERIFIED` | The served VLM. It never enters a framework environment: the client side holds only the model-id string. Weights: Apache-2.0 (LightOn model card, verified 2026-07-09 per ADR-0012). Code: Apache-2.0 (LightOn model card / vLLM serving path, verified separately). Measure with `bash deploy/env/served-ocr-server/smoke.sh`. |

## `deploy/env/served-ocr-client/`

| Component | Kind | Version | Resolution | Runtime | Notes |
| --- | --- | --- | --- | --- | --- |
| `torch` | distribution | `2.13.0+cpu` | **verified** | `UNVERIFIED` | CPU torch the client never runs a model with — it is installed for the shared page rasteriser. No CUDA on this side of the boundary at all. Measure with `bash deploy/env/served-ocr-client/smoke.sh`. |
| `transformers` | distribution | `4.57.6` | **verified** | `UNVERIFIED` | Capped <5 like the rest of the text stack: the client composes with cpu-text. Measure with `bash deploy/env/served-ocr-client/smoke.sh`. |
| `openai` | distribution | `2.46.0` | **verified** | `UNVERIFIED` | The whole client-side dependency on the OCR model: one OpenAI-compatible HTTP client. Measure with `bash deploy/env/served-ocr-client/smoke.sh`. |
| `pillow` | distribution | `12.3.0` | **verified** | `UNVERIFIED` | Page rasterisation (200 dpi, 1540px longest side) before the POST. Measure with `bash deploy/env/served-ocr-client/smoke.sh`. |
| `flash-attn` | excluded_distribution | — | **verified** | `UNVERIFIED` | Excluded: the client runs no attention kernel at all. FM-WHEEL-ABI. Measure with `bash deploy/env/served-ocr-client/smoke.sh`. |
| `torchcodec` | excluded_distribution | — | **verified** | `UNVERIFIED` | Excluded: sentence-transformers does not need it and nothing here decodes media. Measure with `bash deploy/env/served-ocr-client/smoke.sh`. |

## `deploy/env/gpu-learned/`

| Component | Kind | Version | Resolution | Runtime | Notes |
| --- | --- | --- | --- | --- | --- |
| `torch` | distribution | `2.13.0+cu130` | **verified** | `UNVERIFIED` | CUDA-13 wheel line (sm_120 kernels). Note the deliberate asymmetry with the serving environment's 2.11.0+cu130: the two sides are pinned independently BECAUSE they are separate processes — that is what the endpoint boundary buys. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `transformers` | distribution | `4.57.6` | **verified** | `UNVERIFIED` | The 4.x side, in-process. FM-TRANSFORMERS-SPLIT is why the OCR VLMs are not here. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `tokenizers` | distribution | `0.22.2` | **verified** | `UNVERIFIED` | Inside transformers 4.x's window. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `huggingface-hub` | distribution | `0.36.2` | **verified** | `UNVERIFIED` | Below 1.x. On a fresh Blackwell pod set HF_HUB_DISABLE_XET=1 before the first weight download (docs/TUTORIAL-fresh-pod-walkthrough.md §1.3). Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `fused_entity_relation.gliner2` | learned_provider | — | **verified** | `UNVERIFIED` | mdeberta-v3 backbone — the Provider that produced FM-NVRTC-JIT on the live run, and the one whose CUDA placement `latence env placement` measures in bytes rather than trusting output. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `redaction.gliner_pii` | learned_provider | — | **verified** | `UNVERIFIED` | The #191 Provider: gliner 0.2.x never auto-places, so a configured device: cuda with the model on the CPU is silent and plausible. Runtime here means CUDA bytes allocated, measured. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `redaction.gliner2` | learned_provider | — | **verified** | `UNVERIFIED` | mdeberta-v3 again (inherits the FM-NVRTC-JIT seam); carries the ADR-0044 financial floor. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `disambiguation.glinker` | learned_provider | — | **verified** | `UNVERIFIED` | DeBERTa-backed and compute=gpu; loads through the glinker library, which is why it calls the SAME JIT-neutralising seam rather than a copy of it. Also pulls flashdeberta + triton, which is what makes this environment Linux/x86-64 only. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |
| `embedding.sentence_transformers` | learned_provider | — | **verified** | `UNVERIFIED` | The Granite r2 dense leg (ADR-0045): ibm-granite/granite-embedding-311m-multilingual-r2 — WEIGHTS Apache-2.0 (IBM model card), CODE Apache-2.0 (sentence-transformers / transformers loader), verified separately 2026-07-18 per ADR-0012. Measure with `bash deploy/env/gpu-learned/smoke.sh`. |

## Failure modes from the live run, and their resolutions

Every row below was produced by a real run, not predicted. The resolution column names the file that ENFORCES the fix, so the next operator inherits it without knowing it exists.

### FM-FLASHINFER-SM120 — FlashInfer's sm75 check misparses sm_120 and refuses a capable Blackwell device

* **Where:** `served-ocr-server` / `flashinfer-python`
* **Symptom:** vLLM aborts at startup with "FlashInfer requires sm75+" (equivalently "requires sm75 or higher") on a GPU whose compute capability is 12.0 — far above the stated floor.
* **Cause:** The bundled FlashInfer's architecture check parses compute capability 12.0 as the integer 12 and compares it against 75, so every sm_120 device reads as below an sm75 floor. The device is capable; the check is stale.
* **Resolution:** Set VLLM_USE_FLASHINFER_SAMPLER=0 for the serve. This disables only the SAMPLER, so cudagraphs stay ON — it is deliberately NOT --enforce-eager, which would cost the throughput the served path exists for. Upgrading flashinfer past the fixed check is the other exit, and it is a deliberate version bump, not a silent one.
* **Enforced by:** `deploy/env/served-ocr-server/serve.sh`, `deploy/env/served-ocr-server/environment.yaml`, `docs/serving-lighton-vllm.md`
* **Observed:** **verified**
    * 2026-07-17, Blackwell (sm_120) / CUDA-13 pod (SKU not recorded in the field report) — `vllm serve lightonai/LightOnOCR-2-1B ... (the LightOn production recipe)` observed: startup aborted: FlashInfer requires sm75+ (docs/adr/0036-provider-ecosystem-and-bakeoff.md (W10-lighton-vllm); deploy/env/served-ocr-server/environment.yaml)
* **Resolution confirmed:** `UNVERIFIED`
    * Reproduce: `bash deploy/env/served-ocr-server/smoke.sh`
    * Blocked by: The workaround is encoded in serve.sh but has not been re-measured against THIS pinned set (vllm 0.24.0 / torch 2.11.0+cu130 / flashinfer-python 0.6.12) on an sm_120 device.

### FM-NVRTC-JIT — TorchScript JIT asks nvrtc for CUDA-13 builtins that are not there, crashing mdeberta

* **Where:** `gpu-learned` / `fused_entity_relation.gliner2`
* **Symptom:** A native RuntimeError "... failed to open libnvrtc-builtins.so.13.0" raised on the FIRST forward pass (never at load), from the mdeberta-v3 encoder's @torch.jit.script-ed build_relative_position. It killed a 519-page run mid-extraction.
* **Cause:** Torch has TWO runtime compilers. The earlier guard neutralised torch.compile (dynamo / inductor) and nothing else; the TorchScript JIT fuser still asked nvrtc to build kernels against a builtins library absent for that CUDA-13 / torch pairing. Because the fault is a bare native RuntimeError on first forward, no load-time try/except and no dynamo-fault predicate can see it.
* **Resolution:** One shared function, neutralize_torch_jit(), called by each Provider BEFORE its own backbone import ("from gliner import GLiNER" / "from gliner2 import GLiNER2" / "from glinker import ProcessorFactory") — the ordering IS the fix. That import pulls in transformers' deberta_v2 modeling module, whose @torch.jit.script-decorated relative-position helpers are compiled into ScriptFunctions at module-import time; afterwards neither lever can un-script them (PYTORCH_JIT is read by torch only at torch import, jit._state.disable() affects only future script() calls), so calling it only inside the shared loaders is TOO LATE and does not fix this. The five mdeberta Providers and the glinker backend therefore each call the same shared function ahead of their import; the calls inside load_then_cast / load_gliner2_optimized remain as defence-in-depth, never as the whole fix. Throughput-only degradation — output is byte-identical (issue #190).
* **Enforced by:** `packages/latence-core/src/latence_core/providers/perf.py`, `deploy/env/gpu-learned/stack.yaml`, `docs/adr/0036-provider-ecosystem-and-bakeoff.md`
* **Observed:** **verified**
    * 2026-07-17, Blackwell (sm_120) / CUDA-13 pod (SKU not recorded in the field report) — `a live 519-page German-corpus run through fused_entity_relation.gliner2` observed: RuntimeError: ... failed to open libnvrtc-builtins.so.13.0 (raised on the first forward) (docs/adr/0036-provider-ecosystem-and-bakeoff.md (W9-jit, 2026-07-17, #190))
* **Resolution confirmed:** `UNVERIFIED`
    * Reproduce: `bash deploy/env/gpu-learned/smoke.sh`
    * Blocked by: The seam is proven offline (17 tests in packages/latence-core/tests bite it, including one whose fake torch RAISES the exact nvrtc error when the JIT is left enabled), but it has not been re-run on an sm_120 device against torch 2.13.0+cu130.

### FM-WHEEL-ABI — Prebuilt flash-attn / torchcodec wheels ABI-break under the serving layer's torch churn

* **Where:** `served-ocr-server` / `flash-attn`
* **Symptom:** Import failures and segfaults from prebuilt wheels after the serving torch moved 2.8+cu128 -> 2.11+cu130. Hours lost to a stack that installs cleanly and dies at import.
* **Cause:** flash-attn is compiled against a specific torch C++ ABI and torchcodec binds a specific torch/FFmpeg build. Nothing in the framework's declared dependencies asks for either — they arrive as somebody's transitive extra — so an "uninstall them afterwards" runbook step only runs when somebody remembers.
* **Resolution:** Declare both under excluded_distributions, which renders an UNSATISFIABLE pin (flash-attn<0) into constraints.txt: the resolver REFUSES instead of installing. The serving image bakes the same file in as PIP_CONSTRAINT so a later pip install inside a running container fails too, verify-exclusions.py fails the image BUILD if either module is importable, and `latence env verify` reports E-EXCLUSION-VIOLATED if either reaches a lock.
* **Enforced by:** `deploy/env/served-ocr-server/environment.yaml`, `deploy/env/served-ocr-client/environment.yaml`, `deploy/env/served-ocr-server/verify-exclusions.py`, `deploy/env/served-ocr-server/constraints.txt`
* **Observed:** **verified**
    * 2026-07-17, Blackwell (sm_120) / CUDA-13 pod (SKU not recorded in the field report) — `standing up the vLLM serving environment across the torch 2.8+cu128 -> 2.11+cu130 move` observed: prebuilt flash-attn / torchcodec wheels failed to import against the new torch ABI (deploy/env/README.md (the wheels that must be impossible to install, #196))
* **Resolution confirmed:** **verified**
    * 2026-07-19, any host — the refusal is a RESOLUTION-time fact, checked offline — `uv run pytest packages/latence-core/tests/test_envspec.py -k 'exclusion or conflict'` observed: 8 passed — the committed locks carry neither distribution, and an injected one is reported E-EXCLUSION-VIOLATED / E-EXCLUSION-CONTRADICTED (packages/latence-core/tests/test_envspec.py)

### FM-MULTIMODAL-OOM — vLLM multimodal OOM at the documented gpu-memory-utilization and concurrency

* **Where:** `served-ocr-server` / `vllm`
* **Symptom:** CUDA out-of-memory during OCR serving at the generic recipe's --gpu-memory-utilization 0.90 / --max-num-seqs 32, with 34 client images in flight.
* **Cause:** Per-request decoded page images allocate OUTSIDE the KV-cache fraction that gpu-memory-utilization governs, and every in-flight sequence holds one. The generic text-LLM tuning therefore leaves no room for the pixels.
* **Resolution:** Serve with --gpu-memory-utilization 0.72 and --max-num-seqs 16, and cap the client at max_concurrency 12 (12 under a 16-slot server keeps every slot busy without oversubscribing it). 0.72 was stable across the full 519-page corpus.
* **Enforced by:** `deploy/env/served-ocr-server/serve.sh`, `deploy/env/served-ocr-client/stack.yaml`, `deploy/env/README.md`
* **Observed:** **verified**
    * 2026-07-17, Blackwell (sm_120) / CUDA-13 pod (SKU not recorded in the field report) — `vllm serve ... --gpu-memory-utilization 0.90 --max-num-seqs 32 with 34 concurrent page images` observed: CUDA OOM on the multimodal path; stable after 0.72 / 16 / client concurrency 12 across 519 pages (deploy/env/README.md (Blackwell / CUDA-13 settings that came from the run))
* **Resolution confirmed:** `UNVERIFIED`
    * Reproduce: `LATENCE_OCR_BASE_URL=... bash deploy/env/served-ocr-client/smoke.sh`
    * Blocked by: The settings are encoded in serve.sh and the client stack, but the headroom has not been re-measured against vllm 0.24.0 / torch 2.11.0+cu130 on an sm_120 device, and it is VRAM-size dependent — an operator on a different Blackwell SKU should re-check it.

### FM-TRANSFORMERS-SPLIT — transformers <5 (GLiNER family) vs 5.x (OCR VLMs) — no single interpreter satisfies both

* **Where:** `gpu-learned` / `transformers`
* **Symptom:** Installing the OCR VLM next to the learned text stack produces one site-packages holding one transformers: whichever half installed second works, and the other breaks at from_pretrained — hours after the install "succeeded".
* **Cause:** gliner / gliner2 / glinker / flashdeberta are written against the transformers 4.x API; LightOn-OCR-2 support in vLLM 0.24 rides 5.x. The incompatibility lives in the MODEL code, not in package metadata (latence-parser-lighton declares an uncapped transformers>=4.45), so no resolver would ever object.
* **Resolution:** The ADR-0030 endpoint boundary is the ONE supported resolution: 4.x in-process, 5.x served, one HTTP POST between them. Enforced, not documented — gpu-learned names the in-process OCR packages under excluded_packages, and the check walks the workspace dependency graph so a package that merely DEPENDS on one is refused too (E-PACKAGE-EXCLUDED), at resolution time, before install.sh runs. A two-phase install into one interpreter was explicitly rejected.
* **Enforced by:** `deploy/env/gpu-learned/environment.yaml`, `packages/latence-core/src/latence_core/envspec.py`, `deploy/env/README.md`
* **Observed:** **verified**
    * 2026-07-17, Blackwell (sm_120) / CUDA-13 pod (SKU not recorded in the field report) — `standing up the OCR VLM and the GLiNER learned set in one interpreter` observed: the two transformers majors are irreconcilable in one site-packages; the second install wins and the other half fails at model load (issue #198 pod field evidence; deploy/env/README.md (the two-environment split))
* **Resolution confirmed:** **verified**
    * 2026-07-19, any host — the split is a RESOLUTION-time fact, checked offline — `uv run pytest packages/latence-core/tests/test_envspec.py -k 'exclusion or conflict'` observed: 8 passed, including test_the_ocr_split_is_a_genuine_dependency_conflict, which derives the conflict from the two committed specs rather than asserting a comment (packages/latence-core/tests/test_envspec.py)

## Full-stack end-to-end verification

* **Status:** `UNVERIFIED`
* **Command:** `LATENCE_OCR_BASE_URL=http://<host>:8000 bash deploy/env/full-stack/verify-full-stack.sh --corpus <a small directory of scanned pages> --record`
* **Script:** `deploy/env/full-stack/verify-full-stack.sh`
* **Stack:** `deploy/env/full-stack/stack.yaml`
* **Artifacts the run must produce:**
    * `_latence/runs/full-stack-0001/quality-report.json`
    * `_latence/runs/full-stack-0001/quality-report.md`
    * `_latence/runs/full-stack-0001/export/corpus.jsonl`
    * `_latence/runs/full-stack-0001/export/corpus.parquet`
    * `_latence/runs/full-stack-0001/export/graph-nodes.parquet`
    * `_latence/runs/full-stack-0001/export/graph-edges.parquet`
* **Blocked by:** NOT YET RUN. It needs a Blackwell (sm_120) / CUDA-13 host with both pinned environments installed and the OCR server running — none of which exists on the authoring host. Everything the run needs is committed and one command away: the composed stack, the four gates and the artifact assertions (which FAIL on a skipped Stage, a Stage that resolved to cpu, an empty export or document_count=0, so the run cannot pass by narrowing itself). `--record` prints the evidence block to paste back here, so the measurement is transcribed by the machine that made it. Until then `latence env matrix ... --require-verified` stays red, on purpose.

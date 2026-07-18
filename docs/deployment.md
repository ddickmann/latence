# Deployment: on-prem, VPC, and air-gapped

Production deployment of the Latence pipeline. Everything here lives under `deploy/` and is
**separate from the demo** (`demo/`, which is a CPU-only, offline, one-command showcase). This guide
is grounded in what the pipeline actually needs: a **served vLLM OCR endpoint** (the SOTA scanned-
document path), an optional served LLM endpoint, a corpus in and artifacts out, and — for a learned
in-process Provider or the vLLM server — a Hugging Face model cache.

Contents:
- [What the pipeline depends on](#what-the-pipeline-depends-on)
- [GPU requirements + the Blackwell note](#gpu-requirements--the-blackwell-note)
- [Option A — Docker Compose (single box)](#option-a--docker-compose-single-box)
- [Option B — Kubernetes / Helm (on-prem, VPC)](#option-b--kubernetes--helm-on-prem-vpc)
- [Air-gapped deployment](#air-gapped-deployment)
- [Observability + secrets](#observability--secrets)
- [Reproducible in-process environments](#reproducible-in-process-environments)
- [Blackwell / CUDA-13: the compatibility matrix](#blackwell--cuda-13-the-compatibility-matrix)

## What the pipeline depends on

The pipeline itself (parse → chunk → screen → induce → extract → redact → disambiguate → graph →
export) is a **CPU** process. Its SOTA quality comes from **served** models it calls over an
OpenAI-compatible HTTP endpoint — the heavy models are **not** in the pipeline image:

* **vLLM OCR endpoint (required for the scanned-document path).** `parser.lighton_vllm` POSTs each
  page-image to `{base_url}/v1/chat/completions` on a running vLLM server that serves
  **LightOn-OCR-2** (`lightonai/LightOnOCR-2-1B`, Apache-2.0 weights **and** code). Pod-measured,
  vLLM at concurrency is ~2.7× faster than optimised in-process transformers, which is why
  production OCR is a warm server, not in-process (ADR-0030). The pipeline reaches it by
  `base_url` — one config value.
* **LLM endpoint (optional).** The endpoint LLM Providers (`relation.llm`, `entity.endpoint`, schema
  induction) call any OpenAI-compatible server — a local vLLM, OpenRouter, or an on-prem gateway —
  again by `base_url`. Off by default in the deploy artifacts.
* **A text-layer corpus** needs **no** GPU at all: the in-core CPU parsers (`parser.pdfplumber` /
  `parser.document` / `parser.plaintext`) + the deterministic CPU spine run the whole pipeline with
  no served model. Use the served OCR path only for **scans / image-only pages**.

Device honesty (ADR-0036): the framework never fabricates a GPU number. A GPU-only in-process
Provider on a CPU-only host is **skipped-with-flag** (recorded in the Quality Report), controlled by
the `device: auto|cpu|cuda` stage config and the `LATENCE_CUDA=0|1` override. The **served** OCR
Provider is `compute="either"` — a thin HTTP client that runs on the pipeline's CPU (the GPU is
behind the vLLM server), so it is never skipped; it is *infra-gated* on the endpoint being up.

The pipeline image (`deploy/Dockerfile`) is therefore deliberately **torch-free**: it installs
`latence-core` + the **light** endpoint Provider clients (each pulls only the `openai` client and,
for OCR, a pure-Python rasteriser) + the observability extra. No CUDA stack lands in the pipeline
image — that is the whole point of the served path.

## GPU requirements + the Blackwell note

The **vLLM OCR server** needs a GPU. LightOn-OCR-2-1B is a compact 1B vision-language model; a
single modern data-center GPU (Hopper H100 / Ada L40S / Ampere A100, CUDA ≥ 12.6) serves it
comfortably. The production serve recipe (LightOn's official, cudagraphs ON):

```bash
vllm serve lightonai/LightOnOCR-2-1B --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --max-model-len 10000 --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching --max-num-seqs 32
```

**Blackwell (SM 12.x) caveat.** On bleeding-edge Blackwell, vLLM 0.24's bundled FlashInfer cannot
JIT the cudagraphs (`FlashInfer requires sm75+`). Add `--enforce-eager` there (a ~perf floor) **or**
upgrade flashinfer. Mainstream GPUs (Hopper / Ada / Ampere) run cudagraphs fine without it. In the
compose file and the Helm chart this is one added arg — see the `vllmOcr.extraArgs` value and the
comment in `deploy/docker-compose.yml`.

An **optional LLM server** (if you enable it) has its own GPU need sized to the model you pick
(default `Qwen/Qwen2.5-7B-Instruct` → one ~24 GB GPU at bf16).

## Option A — Docker Compose (single box)

One command brings up the OCR server, waits for it to be healthy, and runs the pipeline:

```bash
cd deploy
export HF_TOKEN=...                 # only to DOWNLOAD the model the first time (skip if air-gapped)
mkdir -p corpus out                 # drop your PDFs / scans into ./corpus
docker compose up --build
```

* `vllm-ocr` serves LightOn-OCR-2 on a GPU (needs the NVIDIA Container Toolkit). Its HF cache is
  persisted to `./hf-cache` so a restart does not re-download.
* `pipeline` runs `deploy/stacks/served-ocr.yaml` over `./corpus`, writing the RAG corpus, the KG
  (Parquet/TTL/GraphML), and the Quality Report to `./out`.
* The optional LLM server: `docker compose --profile llm up --build`, then point the relevant stages
  at `http://vllm-llm:8000/v1` (see the comments in `deploy/stacks/served-ocr.yaml`).

Requires a GPU host with the NVIDIA Container Toolkit for `vllm-ocr`. A **text-only** corpus needs no
GPU: swap the Parse stage for `parser.pdfplumber` and you can drop the `vllm-ocr` service entirely.

## Option B — Kubernetes / Helm (on-prem, VPC)

The chart (`deploy/helm/latence/`) deploys: a pipeline **Job** (one-shot) or **CronJob**
(recurring incremental runs), the vLLM OCR **Deployment + Service** (GPU-scheduled), an optional LLM
Deployment, a **ConfigMap** holding the rendered stack YAML, a **referenced Secret** for endpoint
keys, and **PVCs** for the corpus/output and the HF model cache.

```bash
# Render + inspect first (no cluster needed):
helm template rel deploy/helm/latence

# Install (one-shot Job):
helm install latence deploy/helm/latence -n latence --create-namespace \
  -f my-values.yaml

# Recurring incremental runs instead:
helm install latence deploy/helm/latence -n latence --create-namespace \
  --set pipeline.kind=cronjob --set pipeline.schedule="0 2 * * *"
```

Key values (full list in `deploy/helm/latence/values.yaml`):

| Value | Default | Purpose |
| --- | --- | --- |
| `pipeline.kind` | `job` | `job` (one-shot) or `cronjob` (scheduled incremental re-runs). |
| `pipeline.schedule` | `0 2 * * *` | CronJob cron; `concurrencyPolicy: Forbid` keeps WIP=1 (no overlapping runs). |
| `stack.yaml` | served-OCR stack | The pipeline, rendered into a ConfigMap. Parse `base_url` is force-set to the in-cluster OCR Service. |
| `stack.ocrBaseUrl` | "" (in-cluster) | Override to target an **external** OCR endpoint (then `vllmOcr.enabled=false`). |
| `vllmOcr.enabled` | `true` | The served OCR Deployment+Service. Disable to use an external endpoint. |
| `vllmOcr.gpu.nodeSelector` / `.tolerations` | `nvidia.com/gpu.present` / `nvidia.com/gpu` | Schedule the OCR pod onto GPU nodes only. Adjust to your cluster's labels/taints. |
| `vllmOcr.resources.limits."nvidia.com/gpu"` | `1` | GPU request for the OCR pod. |
| `vllmOcr.extraArgs` | LightOn recipe | Add `--enforce-eager` on Blackwell. |
| `vllmLlm.enabled` | `false` | The optional served LLM Deployment. |
| `persistence.data` | 20Gi RWO | Corpus in (`/data/corpus`) + artifacts out (`/data/out`). Use `existingClaim` for an RWX/NFS claim your corpus is staged into. |
| `persistence.hfCache` | 100Gi RWO | The HF model cache the vLLM servers share; **pre-cache here for air-gapped** (below). |
| `secrets.existingSecret` | "" | A Secret your secret manager projects (see `docs/secrets.md`). |
| `metrics.kind` | `none` | `prometheus`/`otel` turns on the observability seam (`docs/observability.md`). |

The pipeline pod runs as the image's **non-root** uid `10001` (`podSecurityContext.fsGroup=10001`
makes the PVCs writable by it). Stage your corpus into the data PVC at `/data/corpus` (e.g. an
init step, an NFS export, or an `existingClaim` pre-populated out of band).

**GPU scheduling.** The OCR/LLM Deployments carry a `nodeSelector` + `tolerations` for GPU nodes and
request `nvidia.com/gpu`. The defaults match the common `nvidia.com/gpu.present` label and the
`nvidia.com/gpu` taint the NVIDIA GPU Operator applies — change them to your cluster's convention.

## Air-gapped deployment

No component phones home at runtime. The only network the pipeline needs is to the **in-cluster**
vLLM endpoint(s). The one thing that normally reaches the internet is the **first model download**;
in an air-gapped VPC you **pre-cache** it.

1. **Pre-cache the model into the HF cache PVC / directory.** On a machine that *can* reach Hugging
   Face (or from an internal mirror), download the weights into what will become `HF_HOME`:

   ```bash
   pip install 'huggingface_hub[cli]'
   HF_HOME=/mnt/hf-cache huggingface-cli download lightonai/LightOnOCR-2-1B
   # (and your LLM model if you enable the LLM server)
   ```

   Then copy `/mnt/hf-cache` onto the `hf-cache` PVC (or bake it into a PV the cluster mounts). The
   vLLM servers mount it at `/data/hf-cache` (`HF_HOME=/data/hf-cache` is set on those pods).

2. **Turn off phone-home.** For the pipeline image and any in-process learned Provider, the runtime
   already sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` with `HF_HOME=/data/hf-cache`. For
   the vLLM servers, set `HF_HUB_OFFLINE=1` (add it to `vllmOcr`/`vllmLlm` env, or the compose
   service env) so vLLM loads only from the pre-cached snapshot and never resolves online. Leave
   `HF_TOKEN` **unset** in air-gapped mode.

3. **Mirror the container images.** Pull `vllm/vllm-openai:<pinned>` and your built `latence:prod`
   into your internal registry and set `image.repository` / `vllmOcr.image.repository` accordingly
   (`imagePullSecrets` if the registry is private).

4. **Build the pipeline image behind the firewall.** `deploy/Dockerfile` needs network only at
   **build** time (uv resolves the lockfile-pinned deps + the light endpoint clients). Build it once
   on a connected builder or against an internal PyPI mirror (`UV_INDEX_URL`), push to your registry,
   and the runtime pulls only from there.

The corpus and outputs live entirely on the PVCs; the Quality Report, spans, and metrics are all
local. Nothing leaves the VPC.

## Observability + secrets

* **Metrics/traces:** turn on Prometheus or OpenTelemetry export with `metrics.kind` (Helm) or
  `LATENCE_METRICS` (env). Everything emitted is counts-only, never content. Full guide:
  `docs/observability.md`.
* **Secrets:** endpoint keys and `HF_TOKEN` are env-only and **referenced**, never embedded — wire a
  Secret via `secrets.existingSecret` and your secret manager (Vault / External Secrets / cloud CSI).
  Full guide: `docs/secrets.md`.

## Reproducible in-process environments

Everything above is the **served** path: the heavy models live behind an HTTP endpoint and the
pipeline image carries no `torch`. When you instead need the learned Providers **in process**
(GLiNER2 extraction, learned PII redaction, GLinker disambiguation, Granite r2 embeddings, SPLADE
sparse), the environment itself becomes the hard part — `torch` churn ABI-breaks prebuilt wheels,
the GLiNER family and the OCR model disagree on the `transformers` major, and PEP 668 blocks naive
installs.

Those environments are pinned, one-command-installable, and **checked by the build**:

```bash
bash deploy/env/cpu-text/install.sh              # the ONE documented install path (PEP-668 safe)
bash deploy/env/cpu-text/smoke.sh                # the ONE command that proves it stands up
bash deploy/env/cpu-text/smoke.sh --spec-only    # the offline gate — no install, no network

bash deploy/env/gpu-learned/install.sh           # the GPU learned-Provider set on CUDA 13 / Blackwell
bash deploy/env/gpu-learned/smoke.sh             # 4 gates, including REAL GPU-placement proof
```

The GPU environment resolves the `transformers` split the only supported way — the ADR-0030
endpoint boundary: the GLiNER/mdeberta family in-process on `transformers` 4.x, the OCR VLMs
served on 5.x and reached over HTTP through `parser.endpoint`. Naming an in-process OCR package in
that environment fails **gate 1**, i.e. at dependency resolution, before anything installs.

Its extra gate is `latence env placement`: every learned Provider is driven through one real
inference and must allocate CUDA memory doing it. A model that silently stayed on the CPU allocates
zero bytes however correct its output is — the #191 failure (`device: cuda` configured, ~11.5
s/chunk, the GPU idle, no warning) can no longer pass a green build.

The pins are **enforced constraints**, not documentation: `latence env verify` cross-checks the
generated `requirements.in` / `constraints.txt`, the solved `requirements.lock`, the Provider
packages' real dependency declarations and the smoke stack, and `scripts/verify-local.sh` runs it
for every environment on every commit. See **[`deploy/env/README.md`](https://github.com/ddickmann/latence-framework/blob/main/deploy/env/README.md)**
for the full shape and for how to add one.

## Blackwell / CUDA-13: the compatibility matrix

Which learned Provider and which serving dependency is known-good on which CUDA / torch /
`transformers` combination — and which combination **nobody has measured** — is a checked artifact,
not tribal knowledge: **[the compatibility matrix](compatibility-matrix.md)**.

Read it before standing up on Blackwell (`sm_120`). It carries its reference hardware and
verification date, and it records the failure modes the live 519-page CUDA-13 run actually produced
— the FlashInfer `sm75` check that misparses `sm_120`, the `nvrtc`/TorchScript crash inside the
mdeberta encoder, the ABI-broken `flash-attn`/`torchcodec` wheels, the multimodal OOM at the
generic serve settings, and the `transformers` 4.x/5.x split — each with the resolution and the
file in the repo that *enforces* it.

Every cell is a measurement or explicitly `UNVERIFIED`, and that is machine-enforced:

```bash
latence env matrix deploy/env/compatibility-matrix.yaml        # the merge gate runs this
latence env matrix deploy/env/compatibility-matrix.yaml --render   # regenerate the published page
latence env matrix deploy/env/compatibility-matrix.yaml --require-verified  # nothing may be UNVERIFIED
```

A `verified` cell without date/hardware/command/observed output is rejected, an `UNVERIFIED` cell
without a runnable reproduce command is rejected, and coverage is *derived* from the committed
`deploy/env/*/environment.yaml` — so a new enforced boundary or learned Provider fails the gate
until somebody maps it.

The one full-stack end-to-end verification (served OCR **plus** the GPU learned Providers **plus**
the text stack, small corpus in, Quality Report + Exports out) is one command:

```bash
export LATENCE_OCR_BASE_URL=http://<host>:8000        # the RUNNING served-OCR environment
bash deploy/env/full-stack/verify-full-stack.sh --corpus <dir of scanned pages> --record
```

Its artifact gate fails on a skipped Stage, a Stage that silently resolved to `cpu`, an empty
export or `document_count: 0`, so the run cannot pass by narrowing its own scope; `--record` prints
the evidence block to paste back into the matrix.

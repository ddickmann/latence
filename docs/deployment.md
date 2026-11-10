# Deployment: on-prem, VPC, and air-gapped

Production deployment of the Latence pipeline. Everything here lives under `deploy/` and is
**separate from the demo** (`demo/`, which is a CPU-only, offline, one-command showcase). This guide
is grounded in what the pipeline actually needs: a **served vLLM OCR endpoint** (the SOTA scanned-
document path), an optional served LLM endpoint, a corpus in and artifacts out, and — for a learned
in-process Provider or the vLLM server — a Hugging Face model cache.

Contents:
- [What the pipeline depends on](#what-the-pipeline-depends-on)
- [What runs where — the cloud-agnostic compute story](#what-runs-where--the-cloud-agnostic-compute-story)
- [GPU requirements + the Blackwell note](#gpu-requirements--the-blackwell-note)
- [Option A — Docker Compose (single box)](#option-a--docker-compose-single-box)
- [Option B — Kubernetes / Helm (on-prem, VPC)](#option-b--kubernetes--helm-on-prem-vpc)
- [Orchestrators — the Airflow adapter](#orchestrators--the-airflow-adapter)
- [Air-gapped deployment](#air-gapped-deployment)
- [Observability + secrets](#observability--secrets)
- [Reproducible in-process environments](#reproducible-in-process-environments)
- [Blackwell / CUDA-13: the compatibility matrix](#blackwell--cuda-13-the-compatibility-matrix)

## What the pipeline depends on

In these deploy artifacts the pipeline is a **CPU** process. The heavy models are **served**: the
pipeline calls them over an OpenAI-compatible HTTP endpoint, and they are **not** in the pipeline
image:

* **vLLM OCR endpoint (required for the scanned-document path).** `parser.lighton_vllm` POSTs each
  page-image to the chat-completions route of a running vLLM server that serves
  **LightOn-OCR-2** (`lightonai/LightOnOCR-2-1B`, Apache-2.0 weights **and** code). The pipeline
  reaches it by `base_url`, the server's `/v1` root — one config value. Production OCR is a warm
  server rather than an in-process model (ADR-0030); the pod measurement behind that choice
  (vLLM at concurrency about 2.7× faster than optimised in-process transformers) is recorded in
  [Serving LightOn-OCR with vLLM](serving-lighton-vllm.md).
* **LLM endpoint (optional).** The endpoint LLM Providers (`relation.llm`, `entity.endpoint`, schema
  induction) call any OpenAI-compatible server — a local vLLM, OpenRouter, or an on-prem gateway —
  again by `base_url`. Off by default in the deploy artifacts.
* **A text-layer corpus** needs **no** GPU at all: the CPU parsers (`parser.pdfplumber` /
  `parser.document` / `parser.plaintext`) + the deterministic CPU spine run the whole pipeline with
  no served model. Use the served OCR path only for **scans / image-only pages**.

Device honesty (ADR-0036): the framework never fabricates a GPU number. A GPU-only in-process
Provider on a CPU-only host is **skipped-with-flag** (recorded in the Quality Report), controlled by
the `device: auto|cpu|cuda` stage config and the `LATENCE_CUDA=0|1` override. The **served** OCR
Provider is `compute="either"` — a thin HTTP client that runs on the pipeline's CPU (the GPU is
behind the vLLM server), so it is never skipped; it is *infra-gated* on the endpoint being up.

The pipeline image (`deploy/Dockerfile`) is therefore deliberately **torch-free**: it installs
`latence-core` + the **light** endpoint Provider clients (each pulls only the `openai` client and,
for OCR, the `pypdfium2` + `pillow` page rasteriser) + the observability extra. No CUDA stack lands
in the pipeline image — that is the whole point of the served path.

## What runs where — the cloud-agnostic compute story

Nothing in the framework names a cloud. Storage is one fsspec seam (`file://`, `s3://`, `gs://`,
`abfs://` — same code, one config key), and models are reached either in process or over an
OpenAI-compatible `base_url`. "Runs in your environment" therefore decomposes into exactly three
compute shapes, and each stage of the pipeline belongs to one of them:

| Compute shape | Stages | Hardware |
| --- | --- | --- |
| **Deterministic CPU** — always local, no model | Source, intake/content screening, chunking, reference extraction, profiling, type consolidation, exact-surface and cascade disambiguation, graph assembly, exports (the corpus as JSONL/Parquet; the graph as Parquet/TTL/GraphML) | any CPU host |
| **Learned, in-process** — GPU-accelerated | GLiNER2 extraction + redaction and Granite embeddings (`compute="either"`: they also run on CPU, at lower throughput); GLinker disambiguation (`compute="gpu"`: skipped-with-flag on a CPU-only host) | one GPU per worker (measured envelopes: `PERF-RESULTS.md`) |
| **Served over HTTP** — the endpoint seam | OCR (vLLM), optional LLM induction/relation extraction | wherever you host the endpoint; the pipeline host stays CPU |

Embeddings are a resolution signal ([Embedders](embedders.md)): `disambiguation.embedding` and
the opt-in merge rungs of `disambiguation.glinker` and `type_consolidation.cascade` use an
Embedder, and a corpus export attaches a vector per row only when its `embedder` block is set.
Query-time retrieval is not part of the framework
([ADR-0068](adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)):
the exports are files, and serving them is the job of whatever engine you load them into.

**A dated scale record.** The benchmark campaign's pipeline stage ran on a CUDA pod in August 2026
(`benchmark/sota-campaign/stages/s3-pipeline/metrics.json`, completed 2026-08-06, one dataset
re-run 2026-08-09): 60,766 documents → 297,548 chunks / 3.4M entity mentions / 731,266 PII
findings. The four datasets that record a wall time took between 0.19 h (609 news articles) and
1.29 h (1,261 OCR'd documents, LLM induction on). The only LLM spend was optional per-chunk
schema induction on three of five datasets: 247,944 calls at a measured $0.000118 per call, about
$29.26; the text-native datasets ran hand-authored schemas at zero LLM cost. That pipeline
included Context Enrichment, hyperedges, graph features and a BM25 export, all since removed, so
the wall times include work the current pipeline no longer does. Nothing has been re-measured
since the removal.

Incremental churn was measured on 2026-08-13 with the same earlier pipeline, on CPU: adding 61
documents to a 548-document base cost 8.4 min against 97.4 min for the full rebuild of the final
corpus (11.6×), and the incremental export was **byte-identical** to the rebuild. The record is
[the ticket-03 report](evidence/enterprise-serving-03-incremental-ingest.md); the mechanism, and
the test that holds the equivalence today, are in
[Incremental ingest](guides/incremental-ingest.md).

## GPU requirements + the Blackwell note

The **vLLM OCR server** needs a GPU. LightOn-OCR-2-1B is a 1B vision-language model, and the
compose file and the Helm chart request one GPU for it. Their defaults are the serve recipe for
mainstream cards (Hopper / Ada / Ampere), with cudagraphs ON:

```bash
vllm serve lightonai/LightOnOCR-2-1B --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --max-model-len 10000 --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching --max-num-seqs 32
```

**Blackwell (SM 12.x) caveat.** On bleeding-edge Blackwell, vLLM 0.24's bundled FlashInfer refuses a
perfectly capable `sm_120` device at startup (`FlashInfer requires sm75+`) — its capability check
misparses compute capability 12.0 as the integer `12`. The recorded fix is **one environment
variable**:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

That disables only the FlashInfer *sampler*, so **cudagraphs stay ON**. Do **not** reach for
`--enforce-eager`: it also clears the abort, but it turns cudagraphs off and floors the throughput
this served path exists for — the compatibility matrix records it as deliberately *not* the fix
(`FM-FLASHINFER-SM120` in [the compatibility matrix](compatibility-matrix.md)).
Upgrading `flashinfer` past the fixed check is the other exit, and it is a deliberate version bump.
Mainstream GPUs (Hopper / Ada / Ampere) need none of this.

In the compose file this is one uncommented `environment:` line; in the Helm chart it is one
`vllmOcr.extraEnv` entry (below). On a CUDA-13 / Blackwell host prefer the pinned serving
environment `deploy/env/served-ocr-server/` outright. Its `serve.sh` sets the variable itself when
the GPU reports compute capability 12.0 or higher, or when the capability cannot be read, and it
defaults to the `--gpu-memory-utilization 0.72` / `--max-num-seqs 16` the live 519-page run needed.

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
* `pipeline` runs `deploy/stacks/served-ocr.yaml` over `./corpus` and writes under `./out`: the
  chunked corpus (`corpus.jsonl` and `corpus.parquet`) and the Quality Report. The shipped stack
  ends at the corpus export. It has no extraction, redaction or graph Stage, so it writes no
  knowledge graph; add those Stages to the stack file to get one.
* The optional LLM server: `docker compose --profile llm up --build`, then point the relevant stages
  at `http://vllm-llm:8000/v1` (see the comments in `deploy/stacks/served-ocr.yaml`).

Requires a GPU host with the NVIDIA Container Toolkit for `vllm-ocr`. A **text-only** corpus needs no
GPU: swap the Parse stage for `parser.pdfplumber`, remove the `pipeline` service's `depends_on`
entry, and you can drop the `vllm-ocr` service entirely.

## Option B — Kubernetes / Helm (on-prem, VPC)

The chart (`deploy/helm/latence/`) deploys: a pipeline **Job** (one-shot) or **CronJob**
(the same command on a schedule), the vLLM OCR **Deployment + Service** (GPU-scheduled), an
optional LLM Deployment, a **ConfigMap** holding the rendered stack YAML, a **referenced Secret**
for endpoint keys, and **PVCs** for the corpus/output and the HF model cache.

```bash
# Render + inspect first (no cluster needed):
helm template rel deploy/helm/latence

# Install (one-shot Job):
helm install latence deploy/helm/latence -n latence --create-namespace \
  -f my-values.yaml

# The same run on a schedule instead (read the run-id note below first):
helm install latence deploy/helm/latence -n latence --create-namespace \
  --set pipeline.kind=cronjob --set pipeline.schedule="0 2 * * *"
```

Key values (full list in `deploy/helm/latence/values.yaml`):

| Value | Default | Purpose |
| --- | --- | --- |
| `pipeline.kind` | `job` | `job` (one-shot) or `cronjob` (the same `latence run` on a schedule). |
| `pipeline.runId` | `run-0001` | The `--run-id` every Job or CronJob run passes. |
| `pipeline.schedule` | `0 2 * * *` | CronJob cron; `concurrencyPolicy: Forbid` keeps WIP=1 (no overlapping runs). |
| `stack.yaml` | served-OCR stack | The pipeline, rendered into a ConfigMap. Parse `base_url` is force-set to the in-cluster OCR Service. |
| `stack.ocrBaseUrl` | "" (in-cluster) | Override to target an **external** OCR endpoint (then `vllmOcr.enabled=false`). |
| `vllmOcr.enabled` | `true` | The served OCR Deployment+Service. Disable to use an external endpoint. |
| `vllmOcr.gpu.nodeSelector` / `.tolerations` | `nvidia.com/gpu.present` / `nvidia.com/gpu` | Schedule the OCR pod onto GPU nodes only. Adjust to your cluster's labels/taints. |
| `vllmOcr.resources.limits."nvidia.com/gpu"` | `1` | GPU request for the OCR pod. |
| `vllmOcr.extraArgs` | LightOn recipe | The serve flags. On Blackwell lower these to the live-run values (`--gpu-memory-utilization=0.72`, `--max-num-seqs=16`) — **not** `--enforce-eager`, which floors throughput. |
| `vllmOcr.extraEnv` | `[]` | Extra `{name, value}` env for the OCR container. The Blackwell fix goes here: `VLLM_USE_FLASHINFER_SAMPLER=0` (keeps cudagraphs ON). |
| `vllmLlm.enabled` | `false` | The optional served LLM Deployment. |
| `persistence.data` | 20Gi RWO | Corpus in (`/data/corpus`) + artifacts out (`/data/out`). Use `existingClaim` for an RWX/NFS claim your corpus is staged into. |
| `persistence.hfCache` | 100Gi RWO | The HF model cache the vLLM servers share; **pre-cache here for air-gapped** (below). |
| `secrets.existingSecret` | "" | A Secret your secret manager projects (see `docs/secrets.md`). |
| `metrics.kind` | `none` | `prometheus`/`otel` turns on the observability seam (`docs/observability.md`). |

The pipeline pod runs as the image's **non-root** uid `10001` (`podSecurityContext.fsGroup=10001`
makes the PVCs writable by it). Stage your corpus into the data PVC at `/data/corpus` (e.g. an
init step, an NFS export, or an `existingClaim` pre-populated out of band).

**The run id is fixed, so a scheduled run is a resume.** Both the Job and the CronJob run
`latence run /etc/latence/stack.yaml --run-id <pipeline.runId>`. The Runner treats a reused run id
as a resume: a Stage whose checkpoint already exists on the data PVC is not run again. A CronJob
run after the first therefore replays the finished run and does **not** pick up documents added
since. The chart has no template that runs `latence delta`. To ingest new documents on a schedule,
run `latence delta` against the same stack and storage from your own CronJob or orchestrator
([Incremental ingest](guides/incremental-ingest.md)), or give each run a fresh `pipeline.runId`
and accept a full re-run.

**GPU scheduling.** The OCR/LLM Deployments carry a `nodeSelector` + `tolerations` for GPU nodes and
request `nvidia.com/gpu`. The defaults match the common `nvidia.com/gpu.present` label and the
`nvidia.com/gpu` taint the NVIDIA GPU Operator applies — change them to your cluster's convention.

## Orchestrators — the Airflow adapter

`latence-runner-airflow` runs the identical Stages as an Airflow DAG (`build_dag`) — the proof
that the Runner seam (ADR-0003/0030) really is pluggable. Its honest status: **tested, not yet
operated**. The in-process topological driver (an Airflow-shaped executor + XCom stand-in) runs
in every test environment; the real `build_dag` structure tests run in CI's heavy lane against a
constraints-pinned `apache-airflow` install (`.github/workflows/ci-heavy.yml`, `airflow-dag`
job). No production Airflow deployment of the framework has been stood up yet — treat the
adapter as a verified seam you finish operationalising in your own Airflow, not a turnkey
deployment artifact. The Compose/Helm paths above are the deployment shapes this guide stands
behind end-to-end.

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
   the vLLM servers, set `HF_HUB_OFFLINE=1` (add it to `vllmOcr.extraEnv`, or to the compose
   service's `environment:`) so vLLM loads only from the pre-cached snapshot and never resolves
   online. The chart's optional LLM Deployment has no `extraEnv` value; set the variable in its
   template if you enable it. Leave `HF_TOKEN` **unset** in air-gapped mode.

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
(GLiNER2 extraction, learned PII redaction, GLinker disambiguation, Granite r2 embeddings), the
environment itself becomes the hard part — `torch` churn ABI-breaks prebuilt wheels,
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

Its extra gate is `latence env placement`: every model-loading Provider the stack instantiates —
derived from the packages the environment installs, then unioned with the declared
`learned_providers`, so omitting that optional field cannot silence the gate — is driven through a
real inference and must allocate CUDA memory **above its resident footprint** doing it (a warm-up
drive
and a baseline subtraction keep weights parked on the GPU from being counted as work — real
`reset_peak_memory_stats()` resets the peak to the current allocation, not to zero). A forward pass
that silently ran on the CPU allocates zero such bytes however correct its output is — the #191
failure (`device: cuda` configured, ~11.5
s/chunk, the GPU idle, no warning) can no longer pass a green build.

The pins are **enforced constraints**, not documentation: `latence env verify` cross-checks the
generated `requirements.in` / `constraints.txt`, the solved `requirements.lock`, the Provider
packages' real dependency declarations and the smoke stack, and `scripts/verify-local.sh` runs it
for every environment on every commit. See **[`deploy/env/README.md`](https://github.com/ddickmann/latence/blob/main/deploy/env/README.md)**
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

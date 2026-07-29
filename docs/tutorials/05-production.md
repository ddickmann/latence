# Tutorial 5 — Production

**What you will build.** The operational half of the system: a stack you can gate in CI, a run you
can kill and restart without losing work, a corpus you can update incrementally instead of
reprocessing, an erasure path that stands up to an auditor, and served GPU models behind an
endpoint.

**Prerequisites.** [Tutorial 1](01-your-first-pipeline.md) and
[Tutorial 2](02-understanding-the-output.md). Sections 1–5 run on a laptop with no GPU; section 6
needs one and says so.

**Time.** About 45 minutes for the laptop sections.

---

## 1. Gate the stack before you run it

Two commands, deliberately different in cost.

**`latence stack check`** is offline and import-free. It reads the file and inspects Provider
*classes* — no corpus, no models, no network, no weights — and finishes in well under a second:

```bash
uv run latence stack check latence.stack.yaml
```

✓ **look for** — `stack check OK — 0 error(s), 0 warning(s)`, exit code 0. Exit 1 on any error,
exit 2 if the file does not exist.

This is what makes a fleet of committed stack configs lint-gatable. Before it existed, the only way
to check a hand-edited stack was a full run — even when the mistake was a typo a class inspection
catches. A Provider that is simply **not installed on this host is a warning, not an error**, so the
same file passes on a CPU CI box and on a GPU pod. `latence process` runs this check automatically
and refuses to start on an error.

**`latence stack validate`** is the expensive one: it runs a named stack end-to-end on the bundled
corpus and asserts the full battery.

```bash
uv run latence stack validate stacks/default.yaml
```

✓ **look for** — ten checks, all PASS, and a non-zero exit if any fails:

```
# Stack validation — `default.yaml` — **PASS**

- documents ingested: 4

| check | result | detail |
|---|---|---|
| phase_boundary | PASS | front half is chunk/document-level (no corpus-level stage before redaction) |
| static_check | PASS | structure, DAG, capability wiring, sub-Providers and config keys all check out |
| contracts | PASS | completeness 48/48 prov, 48/48 class, offsets_aligned=True, drift=clean |
| kg | PASS | nodes=6, edges=33, evidence_coverage=1.00, exported=True |
| rag | PASS | 4 corpus rows across 1 JSONL export(s), no PII leak |
| report | PASS | schema v20, 13 stage metrics, profile_columns=True, no PII leak |
| graceful_failure | PASS | run completed; 3 quarantined + 0 PARSE_ERROR (dangerous fixtures handled, never crashed) |
| determinism | PASS | 6 export artifact(s) byte-identical across two fresh runs |
| resume | PASS | 6 export artifact(s) byte-identical after same-run_id resume |
| device_honesty | PASS | no GPU stages skipped (all providers CPU-reachable on this host) |
```

Read three of those rows carefully, because they are the ones that catch real regressions:

* **`rag ... no PII leak`** and **`report ... no PII leak`** — the harness scans the exported corpus
  and the Quality Report for raw PII from the bundled fixtures. A redaction regression fails the
  build rather than shipping.
* **`determinism`** and **`resume`** — the export artifacts are compared **byte for byte** across
  two fresh runs, and again after a same-`run_id` resume. Not "similar"; identical.
* **`device_honesty`** — on a CPU-only host, a skipped GPU stage is a **PASS**, not a failure. The
  framework's posture is that a skip is honest and a fabricated GPU number is not.

By default the harness runs in a fresh temp dir so a validation never collides with a real run. Pass
`--storage-root` to keep the artifacts for inspection.

---

## 2. Checkpoint and resume

Run state is files on Storage — no database, no coordinator. Every stage writes a checkpoint, and
the checkpoint's *presence* is what makes resume skip it. Checkpoints are written atomically, so a
reader always sees either the previous checkpoint or the complete new one, never a torn write.

Simulate a run that died three-quarters of the way through:

```bash
cd first-pipeline
rm latence-out/_latence/runs/run-0001/checkpoints/{graph,enrich,export_kg,export_corpus}.jsonl
uv run latence run latence.stack.yaml --run-id run-0001
```

The `--run-id` is the state namespace. Reusing it is what makes this a resume rather than a new run.

```bash
uv run python -c "
import json
r = json.load(open('latence-out/_latence/runs/run-0001/quality-report.json'))
for s in r['stages']:
    print(f\"{s['name']:16} from_checkpoint={s['from_checkpoint']}\")
"
```

✓ **look for** — exactly the four deleted stages re-executed and everything above them restored
from disk:

```
source           from_checkpoint=True
intake_screen    from_checkpoint=True
parse            from_checkpoint=True
chunk            from_checkpoint=True
content_screen   from_checkpoint=True
entities         from_checkpoint=True
redact           from_checkpoint=True
relations        from_checkpoint=True
profiling        from_checkpoint=True
disambiguate     from_checkpoint=True
graph            from_checkpoint=False
enrich           from_checkpoint=False
export_kg        from_checkpoint=False
export_corpus    from_checkpoint=False
```

`from_checkpoint` is the resume witness, per stage, in the report. On a real corpus this is the
difference between re-OCR-ing ten thousand pages and not.

The granularity is worth being precise about: **resume is stage-granular.** A stage that died
halfway re-runs from its start, not from its last batch. `wal.log` records batch-granular progress
for crash forensics — "how far did it get" — but the finalized checkpoint remains the authority for
what to skip. That is a deliberate trade: one authoritative signal beats two that can disagree.

Two consequences for how you stage a real pipeline:

* **Use a fresh `run_id` when the input or config changes**, and reuse it only to continue the same
  work. A stale checkpoint from a different config will be happily reused, because presence is the
  rule.
* **Put the expensive stages behind the cheap ones**, which the DAG already does — parse and extract
  checkpoint before anything corpus-level runs, so a failure in graph assembly never costs you the
  OCR.

---

## 3. Incremental corpus updates

Reprocessing a corpus because three documents changed is the thing that makes pipelines unusable in
production. `latence delta` detects changes by content hash against the last committed **Corpus
Version** and commits the next one.

Start from a committed baseline:

```bash
uv run latence run latence.stack.yaml --run-id run-0001
uv run latence delta latence.stack.yaml --run-id delta-0001
```

✓ **look for** — the first delta establishing v0 over the whole corpus:

```
Corpus Version none -> v0 (reconciled)
  documents: +4 ~0 -0 (=0 unchanged)
  entities:  +5 created, 0 merged, 0 split
  edges:     +9 -0
  records:   0 retracted, 0 purged (retraction)
  affected set: 28/28 corpus-level records recomputed
  extraction: 4 extracted, 0 reused (unchanged documents' doc-level Stages skipped — issue #42 / ADR-0043)
  drift: 1.000
```

A delta needs a **fresh `run_id` every time**; leave `--run-id` off and it takes a UTC-timestamped
one (`delta-20260729T182030Z`). Now add a document and run it again:

```bash
cat > documents/vendor.md <<'EOF'
# Vendor Onboarding

Globex onboarded a new vendor in 2025. Carol Nguyen partnered with Acme Corporation
to run the review.
EOF

uv run latence delta latence.stack.yaml --run-id delta-0002
```

✓ **look for** — an *incremental* commit, not a reconciliation:

```
Corpus Version v0 -> v1 (incremental)
  documents: +1 ~0 -0 (=4 unchanged)
  entities:  +0 created, 3 merged, 0 split
  edges:     +1 -0
  records:   0 retracted, 0 purged (retraction)
  affected set: 21/30 corpus-level records recomputed
  extraction: 5 extracted, 0 reused (unchanged documents' doc-level Stages skipped — issue #42 / ADR-0043)
  drift: 0.200
```

Three lines carry the story:

* **`documents: +1 ~0 -0 (=4 unchanged)`** — change detection by content hash. Touch a file without
  editing it and it stays in the unchanged column.
* **`affected set: 21/30 corpus-level records recomputed`** — this is the performance witness, and
  the number to watch. The corpus-level phase recomputes the *affected set*, not the corpus. If that
  ratio is 1.0 on a small change, something in your stack is forcing a global recompute.
* **`entities: +0 created, 3 merged, 0 split`** — the new document mentioned Carol Nguyen and Acme,
  and resolution merged those mentions into the existing nodes rather than creating duplicates.
  Consistency with the base run comes from the corpus's **canonical type vocabulary**, which a delta
  *extends* rather than recomputes: a new type aliases into an existing canonical entry or appends a
  new one, and existing canonical types never churn.

The `extraction:` line reports doc-level reuse separately. On a five-document corpus of
one-paragraph files it is not where the saving is, and the counters here read `0 reused` — check it
on your own corpus rather than assuming a ratio from this one.

`--reinduce` is the escape hatch: re-induce every chunk and re-canonicalize the type vocabulary from
scratch. It is O(corpus) and it is the only path that may re-elect canonical type labels. Reach for
it when the schema itself has drifted, not when documents have changed.

The `delta` block of `quality-report.json` carries all of this in machine-readable form, so a
scheduled delta can assert on it.

---

## 4. Retraction and erasure

Two different operations, for two different requirements, and the difference is not cosmetic.

**Retraction is the auditable default.** The documents' derived records leave the live corpus but are
**retained in the prior Corpus Version** for audit and rollback, and any entity cluster whose merge
was justified solely by a retracted document is split back apart.

You need a document id — the content-addressed `sha256:<64 hex>` the Source stamps. Read one out of
the export:

```bash
uv run python -c "
import json
for line in open('latence-out/_latence/runs/run-0001/export/records.jsonl'):
    r = json.loads(line)
    if r['provenance']['file_name'] == 'incident.md':
        print(r['provenance']['document_id']); break
"
```

```bash
uv run latence retract latence.stack.yaml --run-id retract-0001 \
  --doc sha256:50b22b3cf46440d1108c64331b327b78c5d0d151e7bd08abd783682d7870c766
```

`--doc` is repeatable, and `--docs-file` takes one id per line (`#` comments allowed).

✓ **look for** — a new Corpus Version, records retracted, and — the part that proves the cluster
logic ran — entities **split**:

```
Corpus Version v1 -> v2 (incremental)
  documents: +0 ~0 -1 (=4 unchanged)
  entities:  +0 created, 0 merged, 3 split
  edges:     +0 -4
  records:   14 retracted, 0 purged (retraction)
  affected set: 22/22 corpus-level records recomputed
  extraction: 4 extracted, 0 reused (unchanged documents' doc-level Stages skipped — issue #42 / ADR-0043)
  drift: 0.250
```

Three entities split. Removing `incident.md` removed the evidence that justified merging some
mentions together, so the resolver un-merged them. A system that only deleted rows would have left
three silently wrong nodes behind.

Pass a malformed id and the CLI refuses before touching anything, and tells you where to find a real
one:

```
These do not look like document ids (expected sha256:<64 hex chars>): incident.md
Find a document_id in the export's records.jsonl (the provenance.document_id field), or hash
the file: printf 'sha256:%s\n' "$(shasum -a 256 FILE | cut -d' ' -f1)".
```

**Purge is GDPR erasure, and it is destructive.** Unlike a retraction, nothing is retained: the
on-disk source file, every derived record across **all** Corpus Versions, and the run artifacts are
physically erased. It prompts for confirmation; `--yes` skips the prompt.

```bash
uv run latence purge latence.stack.yaml --doc sha256:<id> --yes
```

The runbook does not end there. Erasure that is not *verifiable* is not erasure, so the sink ships
the audit step:

```bash
uv run latence-sink-qdrant verify-absent 'sha256:<id>#chunk-0' \
  --collection first-pipeline --url http://localhost:6333
```

It exits non-zero listing the offenders if any named record is still in the collection, and exits
`2` rather than reporting absence if the collection it was pointed at does not exist — a mistyped
`--collection` must never pass for erasure evidence.

The other half of that story is `latence-sink-qdrant load`, which is reconciling by default: it
deletes points absent from the Export, which is precisely what makes a retraction or purge take
effect downstream. Reconciliation only ever sweeps points in the load's own id namespace, so a
co-tenant corpus is never touched. Pass `--no-reconcile` when you are loading an *incremental*
export into a collection that also holds records the increment omits.

---

## 5. Reproducible environments

The expensive production failure is not a bug in the pipeline. It is `torch` version churn
ABI-breaking a prebuilt wheel, or two Provider families disagreeing on the `transformers` major, or
a CUDA build without `sm_120` kernels — all discovered at runtime, on a pod, hours in.

The framework's answer is that **the pins are enforced constraints the build checks**, not a wiki
page. Each environment under `deploy/env/<name>/` is a spec plus generated `requirements.in` /
`constraints.txt` / lock, and the CLI verifies they agree:

```bash
uv run latence env verify deploy/env/cpu-text/environment.yaml
```

✓ **look for** — `environment 'cpu-text': OK (no findings)`. It is offline and import-free: it
compares the generated requirements, the solved lock, the Provider packages' declared dependencies
and entry points, and the smoke stack. Dependency drift fails the build instead of the pod. Add
`--probe` inside an installed environment to additionally load every declared Provider from the
registry.

```bash
uv run latence env render deploy/env/cpu-text/environment.yaml --check
```

✓ **look for** — `0 stale generated file(s)`. Drop `--check` to regenerate.

```bash
uv run latence env matrix deploy/env/compatibility-matrix.yaml
```

✓ **look for** — `compatibility matrix 'blackwell-cuda13': OK (no findings)`. Every cell in that
matrix is either a **measurement** — date, hardware, command, observed output — or explicitly
`UNVERIFIED` with the exact command that would turn it into one. Coverage is derived from the
committed environment specs, so a new learned Provider fails this check until it is mapped. Add
`--require-verified` to fail while anything is still unverified.

The environments that ship: `cpu-text` (CPU/text enterprise-SOTA), `served-ocr-server` (the vLLM
serving environment), `served-ocr-client` (the CPU client that calls it), `gpu-learned` (GPU learned
Providers on CUDA 13 / Blackwell), and `full-stack` (the one documented end-to-end verification).

---

## 6. Served GPU models

*From here on you need a GPU.*

The architectural decision that shapes production deployment: **the pipeline process is CPU.** Its
quality comes from models it calls over an OpenAI-compatible HTTP endpoint, not from models loaded
into the pipeline image. The deploy image is deliberately torch-free.

Concretely:

* **The OCR endpoint** — `parser.lighton_vllm` rasterises each page and POSTs the image to a running
  vLLM server serving `lightonai/LightOnOCR-2-1B` (Apache-2.0 weights *and* code). The pipeline
  reaches it by `base_url`: one config value. Pod-measured, vLLM at concurrency is ~2.7× faster than
  optimised in-process transformers, which is why production OCR is a warm server.
* **The schema-induction LLM** — any OpenAI-compatible server: a local vLLM, an on-prem gateway,
  OpenRouter. Also just a `base_url`.
* **A text-layer corpus needs no GPU at all.** The CPU parsers plus the deterministic spine run the
  whole pipeline with no served model. Use the OCR path for scans and image-only pages, not because
  the corpus happens to be PDFs.

The wizard generates this stack directly:

```bash
uv run latence setup --profile enterprise-sota \
  --source ./documents \
  --ocr-base-url http://localhost:8000/v1 \
  --ocr-model lightonocr
```

`--profile enterprise-sota` pins `device: cuda` for every learned stage and recommends the parse path
from your detected file types: a corpus that is entirely born-digital text routes to the CPU
`parser.plaintext` instead of the OCR VLM, because the OCR path has no text passthrough and forcing
it on a text corpus makes every document a `PARSE_ERROR`. It also detects structured cross-reference
headers in the corpus and wires the deterministic `header_refs` and `inline_refs` extractors when it
finds them — the extractors that recovered the majority of the reference edges in the measured RFC
run ([deep dive](../DEEP-DIVE.md), Act IV).

Full serving recipes, the Compose and Helm paths, the air-gapped path and the Blackwell
`VLLM_USE_FLASHINFER_SAMPLER=0` caveat are in [Deployment](../deployment.md) and
[Serving LightOn-OCR with vLLM](../serving-lighton-vllm.md).

### Prove the GPU is actually being used

A model that silently stayed on the CPU produces perfectly correct output. The only way to know is
to measure the allocator:

```bash
uv run latence env placement deploy/env/gpu-learned/environment.yaml
```

This drives one real inference through every model-loading Provider the environment's stack
instantiates and reads torch's CUDA allocator around the call. A Provider that allocated zero CUDA
bytes fails, however correct its output. A missing CUDA device is a **failure, never a skip** — this
command is meaningful only inside an installed GPU environment, and it refuses to pretend otherwise.

### Tune the device you actually have

```bash
uv run latence tune stacks/gpu-sota.yaml --sample 8 --repeats 3
```

For each heavy Provider in the stack, `tune` sweeps `dtype` / `quantize` / `compile` / `batch_size`
/ `ocr_batch_size` / `max_concurrency` over a small real sample under a **hard correctness gate**:
any setting that crashes — including an inference-time `BackendCompilerFailed`, which only shows up
at inference and not at compile — or that changes output versus the fp32 baseline is **rejected**.
It picks the fastest eligible setting per Provider and writes a device-fingerprinted
`tuned/<device>.yaml` plus a report listing winners **and rejected knobs**.

Every throughput in that report is the median of seeded timed runs on that device. Nothing is
extrapolated.

```bash
uv run latence run stacks/gpu-sota.yaml --tuned tuned/<device>.yaml
```

The overlay applies **only** when the tuned config's `device_fingerprint` matches the host. A
mismatch warns and falls back to stack defaults — a tuning measured on one card is never silently
applied to another.

---

## 7. Observability

Both exporters are opt-in and neither is a dependency of a normal run.

**Metrics** — set `LATENCE_METRICS` to `prometheus` or `otel` (with
`latence-core[observability]` installed) to record the per-stage durations, counts, errors and the
entity-resolution merge audit from the Quality Report. `LATENCE_METRICS_TEXTFILE` dumps the registry
for a node-exporter textfile collector, which is the right shape for batch runs. Unset (or `none`)
is a complete no-op.

**Lineage** — set `LATENCE_OPENLINEAGE_URL` to an OpenLineage endpoint (`http://localhost:5000` for
a local Marquez) and every stage appears in the catalog. `LATENCE_OPENLINEAGE_NAMESPACE` names the
namespace; `LATENCE_OPENLINEAGE_API_KEY` supplies a bearer token for a hosted collector — and that
token may only travel over `https` or to loopback, and is never replayed to a redirect target.

Both follow the same rule, and it is worth stating because it is easy to get wrong: **observability
never breaks a run.** A misconfigured endpoint is logged and disabled, not raised. Telemetry that can
fail your pipeline is a liability, not a feature.

---

## 8. A production checklist

Assemble the pieces into something you would actually operate.

**In CI, on every commit:**

```bash
uv run latence stack check stacks/*.yaml                       # sub-second, offline
uv run latence stack validate stacks/default.yaml              # end-to-end, ten checks
uv run latence env verify deploy/env/<yours>/environment.yaml  # dependency drift
uv run latence env render deploy/env/<yours>/environment.yaml --check
```

**On the pod, before the first real run:**

```bash
uv run latence env placement deploy/env/gpu-learned/environment.yaml   # the GPU is real
uv run latence tune stacks/gpu-sota.yaml                               # measured, correctness-gated
```

**Per run:**

* a fresh `--run-id` for a new input or config; the same one to resume;
* assert `contracts_complete` and `graph.evidence_coverage` from `quality-report.json`;
* check the `stages[]` entries for `skipped` — a stage you expected to run and that was
  device-skipped is the most common silent production surprise.

**Per update:**

* `latence delta` with a fresh run id, and watch the **affected-set ratio**;
* `latence retract` for reversible removal, `latence purge` for erasure;
* `latence-sink-qdrant load` to reconcile the store, then `verify-absent` as the audit step.

---

## Where this leaves you

You have now run every part of the system: a pipeline you configured, output you can read and trace,
a query path over the emitted files, a Provider of your own that passes the enterprise-readiness
gate, and the operational surface that makes it a production system rather than a script.

The remaining depth is reference rather than sequence:
[Deployment](../deployment.md) for Compose, Helm and air-gapped installs;
[Tune a stack per device](../tuning.md) and [Run a bake-off](../running-a-bakeoff.md) for the
measurement tools; [Observability](../observability.md) and [Secrets handling](../secrets.md) for
the operational seams; and [Concepts & Architecture](../concepts/index.md) plus the
[Decision Log](../adr/index.md) for why any of it is shaped the way it is.

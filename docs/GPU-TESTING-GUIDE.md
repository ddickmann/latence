# GPU Testing Guide — run the whole learned system yourself

This is the turnkey guide to exercise the full **learned/GPU** stack end-to-end on a CUDA pod, plus the
CPU path that needs no GPU. Every step here was run on an RTX 2000 Ada (16 GB) on 2026-07-09 and passed
(see `issues/gate/GPU-SOTA-SWEEP-RESULTS.md`).

## 0. CPU path (no GPU needed — sanity first)
```bash
uv sync
bash scripts/verify-local.sh          # the full merge gate: secret-scan·ruff·mypy·pytest·conformance·stack-validate·bake-off
latence stack validate stacks/default.yaml   # blessed CPU stack, end-to-end over the bundled messy corpus
```
Green here means the core spine + all deterministic providers are healthy. The learned/GPU providers
are device-skipped-with-flag on a CPU host (honest, never faked).

## 1. Provision a CUDA pod
Any CUDA box with Python 3.11 + torch 2.4.x (cu124). Then:
```bash
# heavy runtime deps (isolated to the learned-provider packages, ADR-0016)
pip install gliner sentence-transformers torch-geometric easydict
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html   # for ULTRA link-pred only

# install the framework + the learned providers (no-deps: torch/transformers already present)
pip install --no-deps -e packages/latence-core \
  -e packages/latence-ner-gliner -e packages/latence-relation-gliner -e packages/latence-pii-gliner \
  -e packages/latence-embedder-st -e packages/latence-disambig-embedding -e packages/latence-parser-pdfplumber
```

## 2. Run the full learned stack end-to-end
```bash
LATENCE_CUDA=1 latence stack validate stacks/gpu-sota-pod.yaml
```
Expected (PASS): documents=3; **kg nodes=3 edges=18 evidence_coverage=1.00**; rag no-PII-leak;
determinism + resume byte-identical; graceful_failure handles the dangerous fixtures. This exercises,
on real weights: GLiNER **fused NER+RE** → GLiNER-**PII redaction** → e5 **context entity-resolution**
→ graph assembly → KG + RAG export.

- `stacks/gpu-sota-pod.yaml` is self-contained (in-process e5 embedder — no server needed).
- `stacks/gpu-sota.yaml` is the same but uses the *served* `embedding.endpoint` for RAG export (point
  its config at your embedding host).

## 3. Run it on YOUR data
Copy a stack and point `source.path` at your corpus (txt/md/pdf/zip). The stack file is the whole
contract — swap any provider by name (that's the model-agnostic seam):
```bash
cp stacks/gpu-sota-pod.yaml stacks/my.yaml   # edit: source.path, labels, relation_labels, pii_labels, kb
LATENCE_CUDA=1 latence stack validate stacks/my.yaml
# or a real run to an output location:
LATENCE_CUDA=1 latence run stacks/my.yaml --storage file:///path/to/out --corpus file:///path/to/docs
```
Outputs under `<storage>/_latence/runs/<run_id>/`: the KG (`export_kg`: Parquet + TTL + GraphML), the
RAG corpus (`export_corpus`: JSONL + Parquet with embeddings), and the Quality Report.

## 4. Per-provider spot-checks (optional, fastest way to sanity a single model)
The pod probe scripts run a single provider on real weights and check offset round-trip:
```bash
python scripts/_pod_probe_gliner.py           # entity.gliner + relation.gliner_relex + redaction.gliner_pii
# validate-*-on-gpu.md scripts: turnkey pod steps for parser.lighton / parser.glm / disambiguation.embedding / relation.gliner_relex
```

## 5. Bake-off (compare providers on YOUR data)
```bash
latence bake-off matrix/entity.yaml          # swaps each candidate for the entity stage, records quality/latency/memory
latence bake-off matrix/disambiguation.yaml  # cascade vs embedding ER, etc.
```

## What to look for / known edges
- **Determinism:** two runs of the same stack over the same corpus must be byte-identical (the harness
  checks this). If not, a provider introduced nondeterminism — a bug.
- **No PII leak:** the redacted RAG export must contain none of the planted PII substrings (the harness
  checks the whole export surface — jsonl/parquet/ttl/graphml).
- **Labels are lower/title case** for GLiNER (`organization`/`person`, not `ORG`) — and the
  disambiguation KB `entity_type` must match that case (the KB linker is case-sensitive).
- **Scale:** validated at bundled-corpus scale (3 docs). 100k/1M throughput/memory is a separate
  perf-rig exercise (backlog), not covered here.
- **Link prediction** (predicted KG edges via ULTRA) lands as an optional post-Assembly
  `graph_completion` stage — in flight; ULTRA zero-shot already proven on the pod.

# Validate `graph_completion.ultra` on a real GPU — turnkey pod steps (W2-linkpred Slice 2, ADR-0037)

The offline slice is validated against the **faithful FakeScorer** (the Scorer *interface* is real —
`prepare` / `score_tails` returning a per-entity vector — only the learned floats are stubbed) plus
an **API-surface anchor** that asserts the VENDORED ULTRA subset really exposes the pinned API
(`Ultra(rel_model_cfg, entity_model_cfg)`, `model(data, batch) -> [B, num_nodes]`,
`tasks.build_relation_graph / all_negative / compute_ranking`). The **real** inductive-ULTRA
zero-shot numbers — real inferred edges + real **MRR / Hits@k** on a held-out split — come from a
**CUDA pod run**. Until this runs, the GPU numbers are explicitly **"pending pod validation"**, never
fabricated.

> **The false-green catch (the PaddleOCR/relex lesson).** The offline provider must be FAITHFUL to
> the real ULTRA API: an un-faithful shim (a wrong method, a mis-shaped output) FAILS
> `tests/test_scorer_faithful.py`. The pod run below is where the real recipe is exercised end to
> end: `ultra_3g` must load with **0 missing / 0 unexpected** params and rank a held-out edge back at
> a sensible MRR. If the load reports missing params, or the MRR is ~random (≈1/N), the recipe is
> wrong — do NOT fold in a number until the load is clean and the ranking is real.

Run on a single ≥8 GB CUDA GPU (an RTX 2000 Ada, an A10, a RunPod pod, …). The warm pod is at
`/root/ULTRA` with `ultra_3g` already downloaded.

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device is visible
python --version               # 3.11+
```

## 1. Install the framework + the learned scorer WITH its heavy deps

```bash
git clone https://github.com/ddickmann/latence-framework && cd latence-framework
uv sync
# graph_completion.ultra ships the torch/torch-geometric/torch-scatter/easydict stack (the vendored
# MIT ULTRA inference subset) — install it WITH the [ultra] extra on the pod. torch-geometric and
# torch-scatter come from the PyG wheel index MATCHED to the installed torch/CUDA:
uv pip install -e "packages/latence-linkpred-ultra[ultra]" \
    torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
# (Optional) the PyKEEN fallback:
uv pip install -e "packages/latence-linkpred-ultra[pykeen]"
```

The ULTRA checkpoint is a **config path** — weights are NOT vendored. Point `checkpoint` at your
local ckpt (the warm pod has `ultra_3g` under `/root/ULTRA/ckpts/`, or fetch from HF
`mgalkin/ultra_3g`, `license: mit`). ULTRA's `rspmm` C++/CUDA extension compiles on first use.

## 2. The decisive load + zero-shot ranking check — the REAL recipe

This is the pod-proven recipe (introspected + run on an RTX 2000 Ada). It loads `ultra_3g` (0
missing / 0 unexpected), builds a fresh 5-entity KG the model never saw, and ranks a held-out edge
`(Acme, partner_of, ?) -> IBM`.

```bash
CKPT=/root/ULTRA/ckpts/ultra_3g.pth   # your local ULTRA checkpoint (weights NOT vendored)
PYTHONHASHSEED=0 python - <<PY
import torch
from latence_linkpred_ultra.scorer import UltraScorer, BuiltKG

# 5-entity KG (indices): Acme=0, Beta=1, Gamma=2, IBM=3, Globex=4; relation partner_of=0.
# Observed: Acme-Beta, Beta-Gamma, Gamma-IBM, Acme-Globex. Held-out target: Acme partner_of IBM.
kg = BuiltKG(num_entities=5, num_relations=1,
             edges=[(0,0,1),(1,0,2),(2,0,3),(0,0,4)])
s = UltraScorer(checkpoint="${CKPT}", device="cuda")
s.prepare(kg)                          # loads ckpt["model"] — expect 0 missing / 0 unexpected
scores = s.score_tails([(0,0)])[0]     # score every tail for (Acme, partner_of, ?)
ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
print("tail ranking (best first):", ranked)
print("IBM (idx 3) rank:", ranked.index(3) + 1)
PY
```

Expected: the model loads cleanly and IBM ranks near the top (a sensible zero-shot completion). A
random-looking ranking means the recipe/checkpoint is wrong — do not fold in a number.

## 3. The full pod stack — real inferred edges + real MRR / Hits@k

Run the pod SOTA stack (the ULTRA link-prediction Stage RUNS here; `skip_if_unavailable: false`):

```bash
CORPUS_DIR=/path/to/messy/docs STORAGE_URI=file:///tmp/linkpred-pod \
  PYTHONHASHSEED=0 uv run latence run stacks/gpu-sota-pod.yaml --run-id linkpred-pod
```

Then read the real metrics the provider computed (held-out MRR / Hits@k with leak detection) and the
separately-counted inferred edges off the Quality Report:

```bash
python - <<'PY'
import json, glob
rep = json.load(open(glob.glob("/tmp/linkpred-pod/_latence/runs/linkpred-pod/quality-report.json")[0]))
g = rep["graph"]
print("inferred_edge_count:", g["inferred_edge_count"])
print("inferred_edges_by_scorer:", g["inferred_edges_by_scorer"])   # expect {"ultra": N}
# The MRR/Hits@k are logged by the provider (graph_completion.ultra: held-out eval ...); grep the run log.
PY
```

Fold the real `inferred_edge_count`, `inferred_edges_by_scorer` (must be `{"ultra": N}`), and the
logged **MRR / Hits@1/3/10** back into the issue / report. Until this pod run is done these are
**"pending pod validation"** — never guessed.

## 4. The bake-off (reference vs ultra) on the pod

```bash
PYTHONHASHSEED=0 uv run latence bake-off matrix/graph_completion.yaml --out-dir /tmp/linkpred-bakeoff
```

On the pod the `graph_completion.ultra` row RUNS (device: cuda) and carries the real inferred-edge
count; on a CPU host it is an honest **skip-with-flag** row (no fabricated number). The
`graph_completion.reference` row is the deterministic 2-hop baseline the learned scorer must beat.

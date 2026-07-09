# W2-linkpred (Slice 2/2) — `latence-linkpred-ultra`: learned ULTRA/PyKEEN graph completion

**The learned link-prediction Provider on the GraphCompleter seam merged in Slice 1 (#131, ADR-0037).**
Registers `graph_completion.ultra` in a new heavy package `latence-linkpred-ultra`, an `AdapterBase`
GraphCompleter that predicts missing KG edges with an **inductive ULTRA scorer** (ICLR'24) + a **PyKEEN
fallback**, a **type-aware tiered 2-hop candidate generator**, and a **temperature/precision
calibrator** — reusing Slice 1's inferred-edge contract verbatim (properties.inferred, honest empty-
mention Evidence, deterministic edge_id, asserted-collision drop, separate inferred_edge_count).

## Licenses (verified 2026-07-09 — research diligence)
- **ULTRA** (DeepGraphLearning/ULTRA, ICLR'24): **MIT** code (MilaGraph) + **MIT** weights (checkpoints
  `ultra_3g/4g/50g` ship in the repo `/ckpts`; no separate restriction). Cite the repo + commit.
- **PyKEEN**: **MIT** (PyKEEN Project Team). Evidence-bearing license entry; passes conformance C3.
- ULTRA is not on PyPI → **vendor the minimal inference subset** of the MIT ULTRA model code
  (`ultra/models.py` Ultra + RelNBFNet/EntityNBFNet, `ultra/layers.py`, `ultra/tasks.py`
  build_relation_graph/all_negative/compute_ranking) into the package under its MIT license, attributed
  in NOTICE/THIRD-PARTY-LICENSES. Do NOT vendor weights — the checkpoint is a config path/URL
  (default `ultra_3g`), documented. (Alternatively depend on ULTRA via a pinned git URL; vendoring the
  inference subset is cleaner for an installable OSS package — pick one, document it.)

## The REAL ULTRA recipe (introspected + a real zero-shot inference RUN on the pod — pin to this)
Proven on the RTX 2000 Ada: `ultra_3g` loads with **0 missing / 0 unexpected** params and ranks a
held-out edge `(Acme, partner_of, ?)→IBM` **#1** on a fresh 5-entity KG it never saw.
```python
from ultra.models import Ultra
from ultra import tasks
from torch_geometric.data import Data
from easydict import EasyDict
CFG = dict(input_dim=64, hidden_dims=[64]*6, message_func="distmult",
           aggregate_func="sum", short_cut=True, layer_norm=True)
model = Ultra(rel_model_cfg=EasyDict({"class":"RelNBFNet", **CFG}),
              entity_model_cfg=EasyDict({"class":"EntityNBFNet", **CFG}))
model.load_state_dict(torch.load(ckpt_path, map_location=dev)["model"])   # ["model"] key; 0 missing/unexpected
# Build the KG (ULTRA doubles relations with inverses: for edge (h,r,t) add (t, r+R, h)):
data = Data(edge_index=torch.stack([torch.cat([h,t]), torch.cat([t,h])]),
            edge_type=torch.cat([r, r+R]), num_nodes=N, num_relations=2*R)
data.relation_graph = tasks.build_relation_graph(data)
# Score a query batch [B,3] = (head, tail, relation):
t_batch, h_batch = tasks.all_negative(data, batch)          # all-entity negatives
scores = model(data, t_batch)                                # [B, num_nodes] scores over tails
ranking = tasks.compute_ranking(pred, target, mask)          # MRR / Hits@k (proper filter mask)
```
Deps: `torch`, `torch-geometric`, `torch-scatter` (matched wheel:
`-f https://data.pyg.org/whl/torch-2.4.1+cu124.html`), `easydict`; ULTRA's `rspmm` C++/CUDA ext compiles
on first use. ALL heavy deps live ONLY in this package (ADR-0016); provider discovery stays torch-free
(lazy import). `compute="either"` (CPU works, GPU accelerates via the device seam, skip-with-flag).

## What to build
1. **`GlinerRelex`-style AdapterBase GraphCompleter** `graph_completion.ultra`: `complete(records) ->
   Iterator[GraphRecord]`. Build the entity vocab from NODE records (node_id→idx), the relation vocab
   from asserted EDGE labels, and the observed edge_index/edge_type (with inverses) — a PyG `Data`.
2. **Candidate generation** — type-aware tiered 2-hop: for each relation, propose (h, r, t) candidates
   from 2-hop neighbourhoods, filtered by the relation's observed head/tail entity TYPES (from node
   labels). **Bounded** (a per-node/per-relation cap, like the relation quadratic guard) with a
   **recall guarantee for rare relations** (a rare relation retains all its type-valid candidates;
   only dense relations are capped). Log any cap that drops candidates (no silent truncation).
3. **Scorer abstraction** — ULTRA (zero-shot from a pretrained checkpoint; optional fine-tune out of
   scope) behind a `Scorer` interface, with a **PyKEEN model as graceful fallback** (if ULTRA
   deps/checkpoint absent → PyKEEN; if neither → typed ProviderError, never a silent no-op).
4. **Calibrator** — temperature scaling + a **precision-targeted threshold**: only emit a predicted
   edge whose calibrated probability clears a configurable target precision; `properties.calibrated=True`,
   `properties.score` = calibrated prob. Uncalibrated raw scores must never be emitted as confidence.
5. **Emit inferred edges via Slice 1's contract** — reuse the seam's inferred-edge representation
   exactly (properties.inferred=True, scorer="ultra", score, rank; honest Evidence mention_ids=[],
   document_ids=union of endpoints, snippet naming ULTRA+score; deterministic edge_id; drop collisions
   with asserted edges). Deterministic + seeded → byte-identical (fixed candidate order, greedy/no-sampling scoring).
6. **KG-quality metrics (the AC)** — MRR / Hits@k on a **held-out edge split** with **leak detection**
   (the held-out edges + their inverses removed from the observed graph before scoring; assert no leak).
   Emit into the Quality Report / a metrics artifact. Real numbers from the pod, never fabricated.

## Anti-false-green (the PaddleOCR/relex lesson — non-negotiable)
The offline stub MUST be faithful to the REAL ULTRA API above (`Ultra(cfg,cfg)`,
`model(data, batch)` returning per-entity scores, the `tasks` helpers returning the real shapes). A
provider that calls a non-existent method or mis-parses the output MUST fail an anchor test. The
real-inference test is dep+CUDA-gated (skipped offline) and overrides any autouse `LATENCE_CUDA=0` pin.
I re-validate on the pod (real MRR/Hits@k) before merge — the pod env is warm at `/root/ULTRA`.

## Definition of Done
- `graph_completion.ultra` builds, registers, AdapterBase GraphCompleter, drop-in on the Slice-1 seam;
  scorer abstraction (ULTRA zero-shot + PyKEEN fallback, graceful); bounded type-aware candidate gen
  with rare-relation recall; precision-targeted calibrator; inferred edges via the Slice-1 contract;
  deterministic byte-identical.
- MRR/Hits@k on a held-out split with leak detection; predictions carry calibrated confidence + honest
  Evidence. Evidence-bearing MIT license (ULTRA+PyKEEN, weights+code, cited); vendored ULTRA subset
  attributed. Passes conformance C1–C6 (roster +1); heavy deps isolated; native fault → typed error.
- `graph_completion.ultra` added to `stacks/gpu-sota.yaml` + `gpu-sota-pod.yaml` as an OPTIONAL trailing
  stage (skip-with-flag on CPU). `matrix/graph_completion.yaml` bake-off (reference vs ultra).
  `bash scripts/verify-local.sh` green. ADR-0037 note updated (learned scorer). `scripts/validate-linkpred-on-gpu.md`.
- Pod-validated: real ULTRA zero-shot over an assembled KG predicts sensible inferred edges + real
  MRR/Hits@k on a held-out split (I run this over SSH; numbers folded in, never fabricated).

## Reviewer should check
- The stub is faithful to the REAL ULTRA API (un-faithful provider FAILS) — the false-green trap.
- Inferred edges reuse the Slice-1 contract EXACTLY (inferred marker, honest empty-mention Evidence,
  collision drop, separate count) — never merged into asserted edges.
- Candidate gen is bounded AND rare relations keep full recall (no silent truncation; caps logged).
- Calibration is real (precision-targeted; raw scores never emitted as confidence); leak detection on
  the held-out split is genuine. Determinism byte-identical. MIT license verified weights+code, cited.

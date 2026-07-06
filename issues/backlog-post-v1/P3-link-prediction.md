# P3 — Link-prediction plugin (preserved)

**Parent**: post-v1 · **Plugin · Hardening bar**

## What to build
An optional Graph-completion plugin that predicts missing KG edges, exposed as a post-Assembly Capability. Ships the strongest link-prediction IP from the private stack: an **inductive/zero-shot ULTRA scorer** (ICLR 2024) plus PyKEEN models behind one scorer abstraction, a **type-aware tiered 2-hop candidate generator** with a recall guarantee for rare relations, and a **temperature-scaling + precision-targeted threshold calibrator**.

## Ports (algorithmic assets — PORT-AS-IS, 5/5)
- `link_prediction/{scorer,ultra_scorer,candidate_generator,calibrator}.py` (~876 LOC). Decouple from RotatE-only assumptions; ULTRA gives inductive generalization the earlier design missed.

## Acceptance criteria
- [ ] Scorer abstraction supports ULTRA (zero-shot + fine-tuned) and ≥1 PyKEEN model with graceful fallback.
- [ ] Candidate generation is bounded and type-aware; rare relations retain full recall.
- [ ] Calibrator hits a configurable target precision for accepted edges; predictions carry confidence + Evidence.
- [ ] Emits KG-quality metrics (MRR, Hits@k) on a held-out split with leak detection.
- [ ] Hardening bar met.

## Blocked by
v1 spine stable through S9. Independent of P1/P2/P4.

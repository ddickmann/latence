# Task: record the baseline: one text-path run and ULTRA's real link-prediction numbers

Type: task
Mode: HITL
Status: open
Blocked by: 07

## Question

The repo records no numbers for the learned text path:

- docs/hour for every learned Provider is "Not yet measured";
- every value in `docs/BENCHMARKS-E2E.md` is `PENDING`;
- there is no MRR or Hits@k for ULTRA, only a 5-entity smoke test, and the 82-RFC dogfood run
  emitted 0 inferred edges.

Before the text path is upgraded, record a baseline on the GPU pod:

- per-stage wall time and GPU memory, and docs/hour, for the current best stack on one fixed
  corpus (the dogfood RFC set plus one German corpus);
- entity, relation and merge quality on whatever gold exists (`goldset.py`), with the gap
  stated where gold is missing;
- ULTRA filtered MRR / Hits@1/3/10 with leak detection (`latence-linkpred-ultra/evaluation.py`),
  on the assembled graph and on one public KG benchmark;
- why the dogfood run emitted 0 edges: a skip, or a calibration refusal.

HITL: the owner runs the pod. The agent prepares the exact commands and the result template,
then records the numbers. No number is reported without the real dependency behind it.

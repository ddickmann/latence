# 09 — Task: per-regime γ calibration as a shipped workflow

Type: task
Status: open
Blocked by: 01

## Problem (measured)

γ is NOT globally monotone vs correctness — inverted U peaking at γ≈0.5, because
γ≤0.5 ⟺ entities-only (weak) BY CONSTRUCTION (`_WEAK_GAMMA_FACTOR = 0.5` encodes
`has_relations` in γ's high bit). Within each regime γ IS monotone (weak: 0.559→0.971;
relation-bearing: 0.500→0.656). A single global isotonic fit flattens both signals — the M6
A/B proved per-regime calibration is recall-NEUTRAL for selection (γ is ~15% of utility), so
its production value is **honest confidence for consumers and the Quality Report**, not
retrieval gains. Ship it as truth-in-labeling, and say exactly that.

## Task

Productize the harness pieces (`calibration-map-per-regime.json` + `calib_ab.py` logic) into
the labeling workflow: `latence-label calibrate` fits per-regime isotonic from a judgments
file; export can stamp `gamma_calibrated` beside raw `gamma` (additive, opt-in — the KV
solver fields keep raw γ; ADR-0058 identity discipline applies). Document the measured
recall-neutrality so nobody sells it as a retrieval lever.

## Outcome

Workflow command + opt-in export field + tests; docs stating the measured neutrality.

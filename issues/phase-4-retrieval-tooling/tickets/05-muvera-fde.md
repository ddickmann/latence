## Parent
Spec: retrieval-quality tooling (#158)

## What to build
A canonical, training-free MUVERA-FDE converter turning multi-vectors into one dense vector, plus a multi-vector generator and the raw-multivector sidecar — all real but experimental and off by default.

## Acceptance criteria
- [ ] A canonical, training-free MUVERA FDE reference produces a fixed-dimension dense vector approximating MaxSim; no fake/passthrough (anti-false-green, ADR-0050).
- [ ] The FDE dense column and the raw-multivector sidecar emit when enabled; both are off by default and clearly marked experimental.
- [ ] A toggle selects FDE-only vs FDE + kept multi-vectors.
- [ ] Tested for shape and MaxSim-approximation on synthetic multi-vectors; real-model validation done out of band.

## Blocked by
- #160 — index-time signal seam

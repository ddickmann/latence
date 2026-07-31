# 09 — Grilling: gamma calibration — product feature or adopter task?

Type: grilling
Status: resolved
Blocked by: 01

## Question

Spec 1.5 calibrates raw GLiNER/GLiREL scores by isotonic regression against ~200 hand-labeled EUs per domain. Is the calibration harness a shipped product feature (with our own labels for the eval/reference domains) or a documented adopter procedure + tooling? Who labels the ~200 EUs for the eval corpora, and does uncalibrated gamma (weighted raw scores) suffice for M1–M4 to proceed? Outcome: decision + calibration tooling scope.

## Answer

**Dennis (2026-07-31): calibration is shipped TOOLING; the labels are the adopter's.**

- A `latence calibrate` harness fits isotonic regression from a small labeled sample
  (~200 Hyperedges per domain, spec 1.5) and emits a **calibration sidecar** the pipeline
  applies when present. **Absent sidecar ⇒ uncalibrated γ, exactly today's behavior** — the
  default path stays byte-identical (the standing rule).
- Shipped default curves were declined: reference-domain curves would be silently wrong on
  the adopter's domain — the hidden-default class this repo refuses.
- **Synthesis with ticket 08/21:** the ~200-EU calibration sample rides the SAME labeling
  harness ticket 21 builds for the manual gold set — one labeling workflow, two artifacts.
  Ticket 21's scope now includes a Hyperedge-γ labeling mode (span + correct/incorrect
  judgment), and the M6 eval reports **calibrated-vs-uncalibrated as an A/B** on the dev
  subset (we label that sample ourselves as part of eval work, which is eval-harness labor,
  not a shipped-curve product claim).
- Build lands as its own small ticket AFTER M5 (it needs the emitted Hyperedges to label
  against); scikit-learn (BSD-3, ticket 02) behind an optional extra, never in core.

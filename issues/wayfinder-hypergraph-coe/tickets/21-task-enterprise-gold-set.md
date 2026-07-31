# 21 — Task (HITL): the manual enterprise gold set

Type: task
Status: open
Blocked by: —

## Question

Ticket 08 (Dennis): the enterprise-shaped eval gates on a **manual** gold set — ~100
multi-hop questions with hand-marked supporting facts over the real benchmark/e2e messy
corpus. Two halves:

1. **AFK — the labeling harness.** Build the flow that makes Dennis's labeling efficient:
   corpus browsing, question capture, supporting-fact span marking, JSONL gold format
   compatible with the ticket-17 harness, resumable sessions, and a coverage dashboard
   (multi-hop depth distribution — the set must not collapse into 1-hop questions).
2. **HITL — the labeling itself.** Dennis (or a delegate) writes the questions and marks
   the facts. Estimated effort: hours, spread over sessions.

3. **Calibration mode (ticket 09):** the same harness labels the ~200-Hyperedge γ-calibration
   sample (span + correct/incorrect judgment) — one labeling workflow, two artifacts (gold
   set + calibration sidecar input).

Resolved when ≥100 questions with verified supporting-fact marks exist and ticket 17 can
read them. Feeds the M6 gate; blocks ticket 17 alongside 08/16.

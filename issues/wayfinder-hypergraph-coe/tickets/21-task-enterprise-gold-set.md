# 21 — Task (HITL): the manual enterprise gold set

Type: task
Status: resolved
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

## Answer

**Complete** — harness, gold set, and calibration judgments all exist and validate; the one
open invitation is Dennis's spot-check (the UI resumes on the same files at any time).

1. **Harness** (subagent, verified): `latence-label` in latence-benchmark — gold mode (local
   HTML UI, span-by-selection, hop tagging, resumable, 1-hop coverage guard), calibration
   mode (ticket 09), `stats`, and the `validate` gate ticket 17 calls. Spans round-trip at
   write time AND validation.
2. **Data** (two Opus subagents at Dennis's direction, 2026-08-01 — PROVENANCE.md records
   plainly that this is LLM-generated, not the manual set ticket 08 first specified, pending
   human spot-check): a 40-document invented-entity enterprise corpus (30% DE, planted
   80 cross-document chains, ground truth in CHAINS.md), a byte-deterministic pipeline run
   (226 hyperedges, redaction exercised, zero PII in exported text), **gold.jsonl: 100
   questions / 225 spans / 20-45-25-10 hop split / 1-hop share 20%**, and
   **judgments.jsonl: 226 calibration judgments (224 correct / 1 partial / 1 incorrect)**.
   `latence-label validate` exit 0 over both, against the regenerated export.
3. **The data found two real framework defects** (stream-copy support inflation with a
   PII-primary-variant risk; duplicate relation triples) — both fixed at the source with
   regression tests before the data was accepted. The dataset's first contribution to
   quality happened before the eval even ran.
4. Residuals: γ is two-valued on this corpus, so the calibration A/B runs on the
   MuSiQue/HotpotQA side (map fog); the redaction digit-run-as-SSN over-trigger is a
   spawned side-task; the export regenerates deterministically via run_pipeline.py (store/
   is gitignored, not committed).

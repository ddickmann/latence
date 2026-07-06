# S3 — Chunk + Screening

**Parent**: S2 · **Phase 1 · Baseline bar**

## What to build
Two Stages that together make the corpus safe and retrieval-shaped. **Chunk**: split parsed documents into retrieval-sized pieces while preserving offsets, page alignment, Provenance, and Classification losslessly. **Screening** at two checkpoints: **Intake Screening** before Parse (malware, zip bombs, file-type spoofing, oversized/corrupt files) and **Content Screening** after Chunk (prompt injection, harmful content, sensitivity escalation). Screening either **Quarantines** a record (excluded from all downstream Stages and exports, retained and inspectable with reason) or flags it with a risk marker that propagates downstream.

## Ports (algorithmic assets)
- Orchestrator chunking + 18-pattern markup stripping + token budgeting (`pipeline-worker/orchestrator.py`) → Chunk internals. PORT-AS-IS (strip RunPod/B2).

## Acceptance criteria
- [ ] Chunk preserves offsets, page mapping, Provenance, and Classification on every chunk — verified by a round-trip test (chunk → offset → original page).
- [ ] Intake Screening runs before Parse; a bundled malicious/zip-bomb/spoofed fixture is Quarantined and never reaches Parse.
- [ ] Content Screening runs after Chunk; a prompt-injection fixture is flagged with a risk marker that survives into the corpus so RAG consumers can exclude it.
- [ ] Screening reference Providers ship permissively-licensed and pluggable (e.g. file-signature/size checks; a lightweight injection classifier).
- [ ] Quarantine dispositions and reasons appear in the Quality Report.
- [ ] Baseline bar met.

## Blocked by
S2.

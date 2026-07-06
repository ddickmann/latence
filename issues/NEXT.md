# NEXT — work queue (WIP = 1)

Phase 1 is a strict build-through. Pull the top unblocked issue, meet the **Baseline bar**, mark done, move the pointer.

## Done
- [x] **S1 — walking-skeleton**
- [x] **S2 — parse** (document Parser: PDF/text → markdown + page map; ported PageOffsetIndex; Classification at Parse; graceful PARSE_ERROR; drift diagnostics in the Quality Report)
- [x] **S3 — chunk+screening** (offset-preserving Chunker over the ported orchestrator asset — 18-pattern strip + token budgeting, page-aligned; Intake Screening quarantines oversized/type-spoof/zip-bomb before Parse; Content Screening flags prompt-injection with a RiskMarker that survives into the corpus; quarantine + findings sidecars → Quality Report; ADR-0021)
- [x] **S4 — entity-extraction** (zero-shot NER: EntityMention as a first-class inter-Stage Record with mention-level page resolution via the ported PageOffsetIndex; in-core deterministic `entity.gazetteer` + learned `entity.gliner`; per-type counts + confidence distribution in the Quality Report; ADR-0022)
- [x] **S5 — relation-extraction** (RelationMention as a first-class Record with a covering-span → page round-trip; in-core deterministic `relation.pattern` (trigger-phrase, zero-shot) + LLM `relation.llm` (any OpenAI-compatible endpoint incl. Ollama) + FUSED `relation.gliner_relex` satisfying S4+S5 in one pass via mixed-carrier checkpoint routing, no double-extraction; per-type relation counts + confidence in the Quality Report; ADR-0023)
- [x] **S6 — redaction** (metadata-aware PII detection + masked/replaced variant per document; ADR-0024)
- [x] **S7 — profiling** (corpus-level statistical + quality features, streaming; ADR-0025)
- [x] **S8 — disambiguation** (ported 5-strategy resolver cascade + Union-Find + audited merge log + GLinker linking + RelationNormalizer; ADR-0026)
- [x] **S9 — graph-assembly+export** (canonical KG with content-addressed nodes/edges + per-edge Evidence; KG Export as Parquet/TTL/GraphML; ADR-0027)
- [x] **S10 — quality-report** (Quality Report promoted to the full first-class artifact — offset-aligned completeness + corpus drift roll-up + machine- & human-readable render — plus the gold-set harness: per-Stage precision/recall/F1 on the adopter's own annotated corpus; ADR-0028)

## Now
- [ ] **S11 — incremental-deltas** ← start here

## Phase-1 queue (in order)
S11 incremental-deltas · S12 performance-path+runner · S13 demo-package

## Then — HARD STOP
- [ ] **G1 — profound-testing-gate** (do not start Phase 2 until every G1 box is green)

## Phase-2 queue (after G1)
H-core · H2 · H3 · H4 · H5 · H6 · H7 · H8 · H9 · H10 · H11 · H12 · H13

## Rules
- One issue in progress at a time on the critical path.
- An issue is done only when its DoD bar is met AND CI is green.
- Detours go to `../docs/` or a new backlog issue — do not expand the active issue's scope.

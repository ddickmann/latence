# Hostile Audit — Summary (conduct-audit, branch `feat/retrieval-tooling`)

**Scope:** query-time retrieval tooling (rerank / fuse / pack / expand / multihop,
query-signal-gen, orchestrator, MCP server) + index-time signal generators
(SPLADE composition, multi-vector, MUVERA-FDE, graph features, BM25 artifact) and the
Export seam that surfaces them — packages `latence-retrieval`, `latence-splade`,
`latence-muvera`, `latence-core`.

**Rounds run:** 6. Rounds R1–R5 each surfaced ≥1 REFUTE-verified critical/major defect;
the 6th (final) re-audit round surfaced **zero** confirmed critical/major findings — the
stop condition — so this pass certifies the branch rather than opening new tickets.

## Final gate (this pass)

- **Feature tests:** `PYTHONHASHSEED=0 LATENCE_CUDA=0 uv run pytest packages/latence-retrieval packages/latence-splade packages/latence-muvera packages/latence-core` → **1821 passed, 8 skipped** (the 8 skips are pre-existing optional-dependency / heavy-model guards, not audit regressions).
- **Lint:** `ruff check` over the four packages → **All checks passed.**
- **Types:** `mypy --strict` over src + tests of the four packages → **Success, no issues in 246 source files.**
- **Protected invariants:** all re-verified green through live, non-vacuous tests (see below). No fix regressed a protected invariant.

## Confirmed defects fixed (by round)

Each row = a REFUTE-verified critical/major finding, its ticket under `issues/audit/`,
and the fix commit(s). Every fix ships a regression test that fails on the pre-fix code.

| Round | Sev | Ticket | Defect (one line) | Fix commit(s) |
|-------|-----|--------|-------------------|---------------|
| R1 | critical | `R1-invariants-0.md` | Index-time SPARSE signal encoded the un-redacted W13 context header → PII re-leak (ADR-0053) | `16efa09` |
| R1 | major | `R1-spec_ac_adr_compliance-1.md` | ADR-0053 SPLADE composition never applied at index time → asymmetric query/doc sparse space | `3ffc68e` |
| R1 | major | `R1-anti_false_green-2.md`, `R1-deep_module_quality-4.md` (dup) | SPLADE redaction gate used substring `in` → masked PII surface re-injected as a term | `3d8c561`, `363e626` |
| R1 | major | `R1-security-3.md` | Multi-hop DuckDB/Neo4j traversal had no cycle/visited guard → exponential work (DoS) on cyclic graphs | `042688a` |
| R2 | major | `R2-security-2.md`, `R2-spec_ac_adr_compliance-0.md`, `R2-deep_module_quality-4.md` (dup) | Non-ASCII bearer token → uncaught `TypeError` in MCP auth (unauth, un-throttled DoS; breaks leak-free-envelope) | `2d7f728`, `1ee7d21` |
| R2 | major | `R2-security-3.md` | Injected query-signal-gen / bad signal key faults escaped the dispatch guard → raw text leak across tool boundary | `307e4ab` |
| R2 | major | `R2-correctness_algorithms-1.md` | BM25 rescorer accepted `df > doc_count` → NEGATIVE IDF → inverted ranking (silent quiet-wrong) | `4a3b6f2` |
| R3 | major | `R3-security-0.md` | Candidate-processor (expand / multihop) faults escaped the MCP dispatch guard → raw store metadata leak | `5070fab` |
| R4 | major | `R4-wired_but_dead-0.md` | ADR-0051 graph features computed + stamped but never surfaced as Export columns → whole graph-augmented seam wired-but-dead | `e59a89c` |
| R5 | critical | `R5-invariants-0.md` | Export wrote the un-redacted ContextHeader (canonical names + relation triples) into inspectable corpus cells → PII re-leak (ADR-0053) | `ee8b2d2` |
| R5 | major | `R5-spec_ac_adr_compliance-1.md` | SPLADE `confidence_threshold` gate inert on the emission path — real per-mention confidence discarded, hardcoded `1.0` → weak NER pollutes the sparse signal | `a5ec44d` |
| R5 | major | `R5-correctness_algorithms-2.md`, `R5-wired_but_dead-3.md` (dup) | pack could not consume the framework's own `kg_node_centrality` (list<double> vs scalar) → centrality leg raised `ValueError` on the real corpus | `e984dda`, `f16888a` |

Total: **12 unique confirmed defects** (2 critical, 10 major) across 16 fix commits;
duplicate cross-dimension re-surfacings were recorded for traceability and marked
`wontfix`/`resolved (duplicate)` with no redundant code churn.

## Protected-invariant verification (live tests, all green)

- **STATELESS (ADR-0048):** `packages/latence-retrieval/tests/test_stateless_seam.py` — tooling holds no index/vectors/graph service and runs no first-stage search; transforms caller-supplied candidates only.
- **FILES-ONLY + BYTE-IDENTICAL-WHEN-UNWIRED (ADR-0017/0049):** `test_retrieval_signal_export.py::test_no_generator_wired_is_byte_identical`, `::test_sparse_columns_absent_by_default`; `test_context_enrichment_export.py::test_context_columns_absent_by_default_schema_byte_identical`, `::test_no_header_embeds_plain_content_byte_identical`; `test_export_embedding_seam.py::test_embeddings_are_optional_no_embedder_no_vectors`. These assert observable byte-identity of the export, not a config flag.
- **ANTI-FALSE-GREEN:** every fix's regression test exercises observable behaviour and fails on pre-fix code (verified per-commit); duplicate tickets took no hollow edit.
- **REDACTION-BY-DEFAULT (ADR-0053):** `test_redaction_*`, `test_extract_on_masked.py`, and the sparse-gate tests in `test_retrieval_signal_export.py` (redacted surface never re-enters the sparse signal); the R1/R5 re-leak paths are now covered.
- **EXPERIMENTAL-OFF-BY-DEFAULT (ADR-0050):** multi-vector / FDE columns absent by default (`test_retrieval_signal_export.py`, `test_sparse.py`); real-but-experimental, no fake FDE.
- **DEEP MODULES / two-adapter seams:** graph-column seam validated end-to-end through the store→candidate metadata path (`test_export_graph_columns_e2e.py`), including the graceful-when-absent no-op.
- **NO REGRESSION to the existing pipeline:** the 9 stacks, provider conformance, and bake-off matrices stay green (`test_stacks_corpus.py`, `test_conformance.py`, `test_bakeoff.py`); all changes additive.

## Residual known-issues (below the critical/major bar)

None open above the reporting bar. No residual critical/major finding remains; the
final re-audit round was clean. The 8 skipped tests are intentional
optional-dependency / heavy-model guards and predate this audit.

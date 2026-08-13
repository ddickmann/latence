# Wave 2-F — the KG-column contract gets one home; GraphRescueRetrieval gets a decoration seam

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 2-F). Two phases, two commits,
never merged: phase A is byte-identical refactoring proven by oracle; phase B is additive and
opt-in, changing nothing by default. All gates green; byte-identity proven by running the gates,
not by assertion.

## Phase A — `kg_columns`: one implementation, four consumers (commit `85b8e59`)

- **New module** `packages/latence-retrieval/src/latence_retrieval/kg_columns.py` — the single
  home of the denormalised-KG-column read contract (graceful-absent / loud-malformed):
  - `parse_id_list(candidate, key, context=...)` replaces the **triplicated** list-of-node-ids
    reader (`expand._id_set`, `MultiHopExpander._node_ids`, `rescue._node_ids`) which differed
    only in error-message prefix. Each consumer passes its `context`, so every pinned diagnostic
    string is byte-identical to before (asserted per-prefix in `test_kg_columns.py`).
  - `peak()` is the one per-entity-list → peak reduction. `pack._signal_scalar`'s list branch and
    the `Expander`'s hub bonus both call it — the lockstep invariant pack.py's prose comment
    enforced ("identically to `expand._peak_centrality`") is now **code**, not documentation.
  - `KgColumns` / `KgColumnKeys`: the parse-once-per-candidate value view (`nodes` / `reach` /
    `communities` / `peak_centrality`) the 1-hop `Expander` scores from; columns are read in the
    emitted order so single-candidate error precedence is unchanged.
- `expand.py` drops ~90 lines of private parsing; `multihop.py` / `rescue.py` / `pack.py` each
  delegate one reader/reduction. Public APIs, configs, and every error string unchanged.
- Apache seven-file pin untouched: `multihop.py`/`pack.py` keep their exact headers
  (`test_publish_readiness` green — note `expand.py` is **not** in the seven and carries no SPDX
  header; left exactly as found). `kg_columns.py` is PolyForm-Noncommercial like the package.

### Phase-A byte-identity gate (run, both sides)

`benchmark/serving/parity.py` runs reference+served together and needs a live Qdrant, so the
oracle is its **reference lane extracted**: `GraphRescueRetrieval` with `KgDenseReference`
(brute-force dense, KG payload columns attached), `Bm25Backend`, `GraphRetriever`
(EdgeGraphSource + ChunkGraphDocumentSource), and the hyperedge lane — over the local S3 exports,
reusing the checkpointed query vectors in `benchmark/serving/cache` (no embedding, no Qdrant).
**All three selectors (`none` / `packed` / `coe`) per query**, so every KG-column consumer —
Expander rescore, `_graph_seeds` harvest, the Packer's `kg_node_centrality` value signal, CoE —
is on the measured path. Full ranked `(id, score)` lists + rescued flags + full `GateDecision` +
selector spend, canonical JSON:

| run | queries | sha256 |
|---|---|---|
| `bf64438` (pre) | multihop_rag limit50 + wiki2multihop limit200 (250 usable) | `405c7ab13127aed9…` |
| `85b8e59` (post) | same | `405c7ab13127aed9…` — **byte-equal** (`cmp` clean) |
| `768b8d8` (post phase B, `decorate=None`) | same | **byte-equal** (`cmp` clean) |

Plus: latence-retrieval suite 796 passed / 4 skipped; s5 + publish-readiness + latence-benchmark
787 passed; `ruff` clean; `mypy --strict` clean.

## Phase B — `CandidateDecorator` seam, opt-in, default-identical (commit `768b8d8`)

**The divergence, documented honestly** (rescue.py module docstring + the graph-rescue guide):
the S5b bench decorates every candidate with the export's KG columns and a precomputed pairwise
`redundancy` map before the expander/selector; the shipped default cannot. Concretely:

1. `RrfFuser` keeps **first-seen** metadata per id and the base fusion is `(dense, bm25)`, so a
   base candidate only BM25 retrieved carries **no KG columns** — the Expander scored it with
   zero nodes silently and `_graph_seeds` harvested nothing from it. Both sides of the ticket-02
   serving A/B shared this composition, so its 0-delta could not see it.
2. The default `similarity_key="redundancy"` never fired (nothing populates the key on the
   shipped path) — the packer re-tokenised O(selected × pool) lexical Jaccard per query.

**The seam**: `GraphRescueRetrieval(..., decorate=CandidateDecorator | None)`. Applied exactly
where the bench decorates — the base ahead of the expander (gate-fired only, mirroring
`RescuePipeline.retrieve`) and the deduped union ahead of the selector. The pipeline **verifies**
decoration enriches metadata only (ids/order/scores unchanged) and fails loudly on a decorator
that reranks. `decorate=None` (default) is the identity.

**The reference adapter**: `KgColumnDecorator(columns, redundancy_key=None)` — candidate-id →
column row (row keys overwrite, the export is authoritative; the `rescued` stamp is preserved),
optional full pairwise redundancy map built on the packer's **own token regex (imported, not
mirrored)** so the numbers cannot drift from `pack._lexical_similarity`. Mirrors the bench's
`_kg_metadata`/`_decorate` and serving `KgDenseReference`; the benchmark keeps its own wiring
until Wave 5.

### Phase-B gates (tests, `tests/test_rescue.py::TestDecoration`)

- `decorate=None` byte-identity: the **full `RescueResult`** (ids, scores, metadata, gate,
  spend) of a dual-retrieved + bm25-only + rescue-overlap fixture, captured verbatim on the
  pre-seam code, reproduced exactly.
- bm25-only candidate produces traversal seeds **only when decorated** (recording-graph spy;
  query seeds still first).
- supplied redundancy short-circuits the packer's lexical Jaccard (counting spy on
  `pack._lexical_similarity`: >0 calls undecorated, **0** decorated).
- adapter redundancy == `pack._lexical_similarity` exactly; stale engine columns overwritten,
  other keys preserved; reranking decorator is loud; protocol accepts the adapter, rejects
  non-decorators at construction.

## Rider

`benchmark/s5/rescue.py`'s RescuePipeline and the union-vs-select split (arch-pass row J, the
~600 lockstep lines) stay untouched — Wave 5 owns that deletion behind its own A/B oracle; this
wave only shipped the seam it will need.

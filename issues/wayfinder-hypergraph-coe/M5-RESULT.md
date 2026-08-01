# M5 result — chain assembly + selection-trace logging (ticket 16)

Status: built + green. Module: `packages/latence-retrieval/src/latence_retrieval/chains.py`;
trace hook added to `coe.py` (default path byte-identical, `test_coe.py` unchanged); tests:
`packages/latence-retrieval/tests/test_chains.py` (14, all deterministic, PYTHONHASHSEED=0).

## What was built

### Chain assembly (spec 3.4) — `assemble_chains(selection, signals, *, tau_lookup, …)`

- **G_sel components = chains**, found by BFS over an inverted node→selected-candidate index —
  never a dense pairwise structure (the same ticket-20 sparse discipline as the selector).
  Singletons are fine; a component whose entities never touch V0 is peripheral.
- **Intra-chain order**: greedy minimal key `(hop, direction_rank, −ρ, id)`. `hop` is BFS layer
  distance of the Hyperedge's closest entity to the anchors (V0) over shared entities — evidence
  distance stays the PRIMARY order; ρ desc then ascending id break ties. Identical inputs yield
  identical output **bytes** (pinned twice, unit + e2e).
- **Peripheral chains** render LAST under one `## Peripheral evidence` header, each chain marked
  `(peripheral)`. No V0 to BFS from, so a peripheral chain seeds at its (ρ desc, id asc) best
  Hyperedge (step 1 by definition), hops measured from that seed. Anchored chains order among
  themselves by max-ρ desc (smallest-id tie-break); peripheral likewise, after all anchored.
- **Emission** per spec 3.4: `## Chain k (anchor: <node ids ∩ V0>)`, then per step
  `[step i] (rho=…, gamma=…, src=…) <tau_text>` + an indented `entities: … | relations: …`
  detail line (relations only when a lookup supplied them). `src=` uses the optional
  `src_lookup(hid) → doc/chunk ref`; absent, the content-addressed Hyperedge id — itself a
  stable reference — fills the slot (solver fields deliberately carry no source ref).
- **τ text is redaction-safe by construction**: `tau_lookup` is typically
  `HyperedgeKV.tau_text`, and ADR-0057 enforces at the WRITER that stored `tau_text` is the
  corpus redaction variant — the rendered context can never contain text the exported corpus
  would not. A selected id with no fetchable τ, or absent from the signals, fails loudly.

### Spec 6.3 directed-relation preference — the exact rule

Solver fields carry no relation triples by design, so direction rides an optional caller seam
`relations_lookup(hyperedge_id) → [(head_node, label, tail_node), …]` (e.g. backed by the
emitted graph parquet). The implemented rule: **with the lookup supplied,
`direction_rank(e) = 0` when ≥ 1 of e's triples has its head node already *resolved* (V0 ∩
chain, plus every entity of the steps emitted so far — the step continues an established
head→tail path), `2` when e carries triples but none of their heads is resolved (its relations
point backward into the resolved evidence at best), `1` when e carries no triples (no
directional testimony either way).** The rank sits BETWEEN hop and ρ in the key: it refines
within a hop layer and never overrides evidence distance (pinned:
`test_directed_preference_never_overrides_hop_distance`). Without the lookup the rank is
constant, so the order is EXACTLY the undirected `(hop, −ρ, id)` of spec 3.4 (pinned:
`test_directed_preference_follows_head_to_tail_only_when_lookup_is_supplied`). This is the
spec's *partial* mitigation for sequentially-constrained multi-hop; true
retrieve-resolve-retrieve stays out of scope per 6.3's last paragraph.

### Selection-trace logging (spec 3.5) — the weak-supervision seed

- `select(…, trace_sink: Callable[[dict], None] | None = None)` in `coe.py` — the ONLY edit to
  an existing module (plus `__init__.py` exports). Default-`None` path byte-identical: all trace
  code is guarded behind `if trace_sink is not None`; `test_coe.py` passes UNCHANGED.
- ONE dict per query: `stage="select"`, sorted `anchors`, `expanded_candidates` /
  `pruned_candidates` / `candidate_count`, `budget`, per-candidate `steps` (hyperedge_id, v0,
  marginal, decision accepted/rejected, trimmed flag), final `selected`, `trimmed`, `verified`,
  `total_cost`. Every candidate appears exactly once as accepted/rejected; accepted-then-trimmed
  keeps decision `accepted` with `trimmed: true` and is listed in `trimmed`.
- A rejected candidate's marginal is evaluated against the FINAL selected set — the
  counterfactual a learned utility wants to score; an accepted one's is its value at acceptance,
  exactly what drove the decision.
- **No raw `tau_text` in any trace — ids/counts/scores only.** Deliberate privacy choice,
  recorded in both module docstrings: the trace is the future classifier's training seed and may
  travel further than the corpus it describes (pinned: `test_no_tau_text_ever_reaches_a_trace`).
- `assemble_chains` emits its own `stage="chains"` dict through the same sink shape: chain
  count/sizes, anchored/peripheral counts, directed flag, per-chain ordered ids.

## The e2e (KV fixture → signals → select → chains)

`test_e2e_two_hop_corpus_renders_one_chain_with_the_bridge_after_the_direct`: the M3/M4 fixture
corpus (`_build_kv`, planted seed—direct—mid—bridge—answer chain + weak lexical distractor + 20
noise edges) through the full path. Selection assembles {h_direct, h_bridge} (M4's e2e); chain
assembly renders them as **ONE anchored chain** (`anchor: n_seed`), step 1 = h_direct (hop 0),
step 2 = h_bridge (hop 1) — the bridge reads after the direct evidence. Byte-deterministic
across two calls; trace sequence `["select", "chains"]`.

## Numbers

- `PYTHONHASHSEED=0 uv run pytest packages/latence-retrieval/tests/`: **653 passed, 17 skipped**
  (skips pre-existing, dep-gated), including the 14 new chains tests; `test_coe.py` unchanged.
- `uv run mypy --strict packages/latence-retrieval/src`: Success, 28 source files.
- `uv run ruff check packages/latence-retrieval`: All checks passed. (`ruff check packages`
  currently reports 3 findings, all inside the concurrent session's uncommitted
  `packages/latence-benchmark/src/latence_benchmark/labeling/` files — out of this lane's
  scope and untouched per the workstream split.)
- Full suite (`PYTHONHASHSEED=0 make test`): see the final report line (run at the end of M5).

## Deviations / notes

- Anchor naming in chain headers uses entity **node ids** (what signals carry); a display-name
  lookup is a caller concern, same as `src_lookup`.
- Files touched: NEW `chains.py`, NEW `test_chains.py`, `coe.py` (trace hook + docstring),
  `__init__.py` (exports only), this file.

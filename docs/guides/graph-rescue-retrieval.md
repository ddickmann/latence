# Graph-rescue retrieval

> **Status: production component, measured defaults.** The composition lives in
> `latence_retrieval.rescue.GraphRescueRetrieval`; every default below is the operating point
> measured on the framework's own end-to-end benchmark (ticket `s5b-retrieval-architecture-fix`),
> not a guess. The campaign wizard configures it via the `retrieval:` block of `campaign.yaml`.

`GraphRescueRetrieval` is the shipped answer to the question *"how do I actually apply the
knowledge graph at query time without making retrieval worse?"* — base retrieval first, the graph
strictly **additive** on top, a solver spending the token budget, and a per-query policy gate
deciding whether the graph lane is worth paying for at all. There are **zero LLM calls anywhere on
the online path**: every signal is a precomputed export column, a graph traversal, or a
deterministic solver — recall-per-token at zero marginal LLM cost is the positioning.

> **Before you enable this: what it measurably buys.** On the three datasets of the ticket-06
> experiment, the traversal and hypergraph lanes contribute **nothing a top-10 or top-100 metric
> can see** — an oracle that promotes every rescued gold chunk to the front is worth +0.0056
> (multihop_rag), +0.0077 (wiki2multihop) and **0.0000** (vidoseek), against 221–538 ms/query.
> The base stage (dense+BM25 RRF) is doing the work, and the next unit of quality is a rescore
> over *its* top-100, worth up to +0.37. Enable the graph lanes for the property below — that
> they cannot make retrieval worse — not for a recall gain. See
> `../evidence/graph-inertia.md`.

## Why this shape (and not fusion)

The obvious wiring — run a graph traversal as one more retrieval leg and RRF-fuse it with dense
and BM25 — was **measured to regress recall on every dataset** of the S5 benchmark matrix
(R@1 falling monotonically as graph legs were added, e.g. ohr_bench 0.228 → 0.122). Ticket 06
reproduced this independently as a continuous weight sweep — RRF weight on the graph lane is
monotonically harmful, −0.0187 recall@10 at `w=0.25` and −0.1121 at `w=1.00` on multihop_rag, with
no weight small enough to help without being small enough to do nothing — so **the design choice
below is confirmed by measurement, not merely by the original observation.** The components were
fine; the application was wrong, four ways:

1. **Graph lists are not peers of dense/BM25.** A weak alias-seeded traversal hit competed at
   equal RRF weight with a strong dense hit and dragged it down. Dense and BM25 *are* peers (both
   full-corpus first stages over the same corpus), so they stay fused; the graph result is
   **appended below the base and deduplicated by id — never fused into it**. The base ordering is
   never reordered by the graph lane.
2. **Entity resolution ignored the strongest signal.** Seeding traversal from query n-grams alone
   throws away the entities of the top first-stage documents. The fix seeds from the **query and
   the top-5 first-stage documents** (query seeds first, never displaced by the cap).
3. **The graph lane ran unconditionally.** Known-item and navigational queries can only get worse
   from traversal; a **policy gate** now decides per query.
4. **The selector was under-budgeted and mis-measured.** A 4,000-token knapsack budget came from
   neither the source paper nor a measurement (and actually packed ~1,895 tokens); selection
   stages are now budgeted at their measured operating points and evaluated on selection metrics.

This is the same architectural cut as the graph-sidecar reference flow (`colsearch`): *"low-level
entity/relation rescue, high-level thematic rescue, and additive evidence stitching — without
turning graph traversal into the default first-stage router."*

## The pipeline

```text
query bundle (text, dense vector, entity seed node ids)
  -> dense + BM25 first stages (caller's backends, top-100 each)
  -> RRF fuse (those two ONLY), max-normalise to 1.0
  -> policy gate (graph_mode: off | auto | force)
       not fired -> selector
       fired     -> Expander rescore of the base (1-hop KG columns, bounded bonuses)
                 -> seed entities from query + top-5 first-stage docs (cap 32)
                 -> GraphRetriever traversal  ┐ fused with each other,
                 -> hypergraph_signals lane   ┘ APPENDED below the base, dedupe, cap 50
  -> RescueRetrieval (base + union + GateDecision + rescued count + hypergraph signals)
  -> selector over the union: Packer (knapsack) | coe.select | none
  -> RescueResult (candidates + GateDecision + rescued count + tokens spent)
```

The `RescueRetrieval` line is a real seam, not a diagram step: `run(bundle)` is exactly
`select(bundle.text, retrieve(bundle))`, and the two halves are callable on their own — see
[budget A/Bs without re-retrieval](#budget-abs-without-re-retrieval).

Each step is computed by the shipped component named beside it — the composition adds **policy
only** (ADR-0048: the caller supplies engines and signals; the library holds no index and fetches
nothing of its own).

Two score-scale rules are load-bearing:

- The fused base is **max-normalised to 1.0** before the rescore. RRF scores sit near 0.03; the
  expander's bonuses are absolute numbers. Without the (strictly monotone, order-preserving)
  normalisation, one graph link would outvote the entire base ranking — the exact defect this
  component removes.
- The deduped union is re-scored by rank as `1/(60 + rank)` — the fuser's own decay shape — so a
  candidate deep in the pool keeps enough value for the packer to consider it instead of the
  greedy stalling at a fraction of the budget. Every appended candidate is stamped
  `metadata["rescued"] = True`, so selectors and callers can always tell what the base earned
  from what the graph carried in.

## Wiring it

```python
from latence_retrieval import GraphRescueRetrieval, GraphRetriever, QueryBundle

pipeline = GraphRescueRetrieval(
    {
        "graph_mode": "auto",          # off | auto | force
        "selector": "packed",          # packed | coe | none
        "hyperedge_kv_path": "export/graph-hyperedges.kv",   # optional hypergraph lane
    },
    dense=my_dense_backend,            # your engine, as a RetrievalBackend adapter
    bm25=my_bm25_backend,
    graph=GraphRetriever(              # optional traversal lane, zero-ops over the export
        {
            "graph": {"source": "duckdb", "edges_uri": "export/graph-edges.parquet"},
            "documents": {"source": "duckdb", "chunks_uri": "export/chunks.parquet"},
        }
    ),
    hyperedge_documents=my_hyperedge_chunk_resolver,  # optional, paired with the kv path
)

result = pipeline.run(
    QueryBundle(
        text="why did the merger between Acme and Beta collapse",
        vector=my_query_embedding,          # caller-run query-signal-gen
        seed_node_ids=my_linker_seeds,      # caller's entity linker, or AliasQuerySeeder
    )
)
result.candidates      # base order preserved, rescue appendix stamped `rescued`
result.gate            # the policy decision: fired, reasons, veto, agreement, confidence
result.tokens_packed   # the selector's actual spend against result.budget
```

### Budget A/Bs without re-retrieval

`run` is the one-shot call; underneath it the pipeline is two halves you can call yourself —
`retrieve(bundle)` (the expensive one: two first stages, the rescore, the traversal and the
hypergraph propagation) returning a frozen `RescueRetrieval`, and `select(text, retrieval, …)`
(the cheap one) spending a budget over it. One retrieval serves any number of selections:

```python
retrieval = pipeline.retrieve(bundle)          # engines run ONCE
for budget in (4_000, 10_000, 16_000):
    result = pipeline.select(bundle.text, retrieval, budget=budget)
    print(budget, result.tokens_packed, [c.id for c in result.candidates][:5])

coe = pipeline.select(bundle.text, retrieval, selector="coe")   # selector A/B, same union
```

`selector=` / `budget=` override the configured values for that call only; omitted, they *are*
the configured values, so `select(bundle.text, pipeline.retrieve(bundle))` is byte-identical to
`run(bundle)` (pinned per selector in `tests/test_rescue.py::TestTheSplit`). An overriding
selector with no explicit budget falls back to **its own** measured default (10k packed / 16k
coe), and `selector="coe"` still requires the hypergraph lane. This is the shape a budget curve
needs: every operating point is settled against the identical candidate union, so the curve
measures the *selector* rather than N slightly different retrievals — which is exactly how the
S5b benchmark's `BUDGET_CURVE` cells are produced.

Both graph lanes are optional and degrade honestly: no `graph` backend means no traversal lane,
no `hyperedge_kv_path`/`hyperedge_documents` pair means no hypergraph lane (supplying only half
the pair is a loud construction error), and a base candidate list without KG columns makes the
rescore an order-preserving no-op. At minimum one base first stage (`dense` and/or `bm25`) is
required.

## The policy gate

`graph_mode` is the switch; `auto` is the shipped policy. Four triggers turn the lane **on**:

| Trigger (reason string) | Fires when | Default |
| --- | --- | --- |
| `relation-cue` | the query carries a relation word (`why`, `between`, `depends`, `caused`, `compare`, …) | shipped lexicon (`relation_cues`) |
| `entity-heavy` | the query itself names ≥ `entity_heavy_seeds` linked entities | 2 |
| `low-agreement` | Jaccard overlap of the dense and BM25 top-`agreement_k` ids < `low_agreement` | top-10 < 0.30 |
| `low-confidence` | dense top-1 score < `low_confidence` | 0.60 |

One **veto**: a query of ≤ `navigational_max_words` (6) words with no relation cue whose two
routes agree at ≥ `high_agreement` (0.50) is known-item/navigational and stays graph-off whatever
else fired (`vetoed_by="navigational"`).

Every decision is returned as a `GateDecision` on the result — the fired flag, the named reasons,
the veto, the measured agreement/confidence — so the gate's hit rate is a number you report, not
an assumption. `off` and `force` still record what `auto` *would* have decided (`vetoed_by:
"mode-off"`, reason `mode-force`): a forced A/B can separate a gate that is too tight from a lane
that is too weak. `low_confidence` is the one corpus-dependent knob — it compares against your
dense engine's score scale, so check it against a sample of your own top-1 scores.

## What the export must carry

The rescore and the packer's graph feature read the KG columns the pipeline denormalises **onto
every chunk** (ADR-0051 — no graph database in the default path; see the
[retrieval-engine input spec](../RETRIEVAL-ENGINE-INPUT-SPEC.md)). Your backend adapter must
surface them on each returned `Candidate.metadata`:

| Column | Content | Used by |
| --- | --- | --- |
| `kg_node_ids` | node ids of the chunk's **own** entities (list of strings) | 1-hop links, co-mention, doc-derived traversal seeds |
| `neighbor_node_ids` | node ids the chunk's triples reach (its 1-hop expansion keys) | 1-hop links (both directions are checked) |
| `kg_node_community` | community id per entity (optional graph-features projection) | shared-community bonus |
| `kg_node_centrality` | centrality float per entity | the expander's hub bonus **and** the packer's `kg_node_centrality` value signal (both reduce the per-entity list to its **peak**) |

An absent column simply contributes nothing (graceful degradation); a present-but-malformed
column fails loudly at the boundary — never a silently wrong score. The traversal lane reads the
emitted `graph-edges.parquet` + chunk parquet through `GraphRetriever`'s zero-ops DuckDB
adapters; the hypergraph lane reads the emitted `graph-hyperedges.kv` sidecar. (The read
contract itself — graceful-absent / loud-malformed, and the per-entity-list → peak reduction —
has one home: `latence_retrieval.kg_columns`; every consumer above runs the same code.)

## Decoration: the shipped default vs the benchmarked composition

The S5b benchmark decorates every candidate with the export's KG columns **and** a precomputed
pairwise `redundancy` map before the expander and selector run. The shipped default composition
does not — candidates carry only what your engines put on them — and that opens a real, honest
divergence from the benchmarked numbers:

- **A base candidate only BM25 retrieved carries no KG columns.** `RrfFuser` keeps the
  first-seen metadata per id, and the base fusion order is `(dense, bm25)` — so a bm25-only hit
  reaches the expander with zero nodes (scored silently as column-free) and contributes no
  entities to the traversal seed harvest. (Both sides of the serving-parity A/B shared this
  composition, so its 0-delta could not surface it.)
- **The packer's default `similarity_key: "redundancy"` never fires.** Nothing populates that
  key on the default path, so every pair falls back to the packer's own lexical Jaccard,
  re-tokenising `O(selected × pool)` per query — the recomputation the bench precomputes once.

The opt-in fix is the `decorate=` dependency — a `CandidateDecorator` applied to the base ahead
of the expander and to the deduped union ahead of the selector, exactly where the bench
decorates. `decorate=None` (the default) preserves today's behaviour byte-for-byte (pinned by
test). The shipped reference adapter mirrors the bench's decoration:

```python
from latence_retrieval import GraphRescueRetrieval, KgColumnDecorator

decorator = KgColumnDecorator(
    {   # candidate id -> the export's denormalised KG columns for that chunk
        chunk_id: {
            "kg_node_ids": row.node_ids,
            "neighbor_node_ids": row.neighbor_node_ids,
            "kg_node_community": row.community_ids,
            "kg_node_centrality": row.centralities,
        }
        for chunk_id, row in my_export_rows.items()
    },
    redundancy_key="redundancy",   # optional: precompute the packer's pairwise redundancy
)

pipeline = GraphRescueRetrieval(config, dense=..., bm25=..., decorate=decorator)
```

A decorator may only **enrich metadata**: the pipeline verifies ids, order, and scores are
unchanged and fails loudly on a decorator that reranks. With `redundancy_key` set, the stored
numbers are computed with the packer's own tokeniser (imported, not mirrored), so they are
byte-identical to what the packer would have recomputed — supplying them is purely a
per-query-cost optimisation, never a semantic change.

## The two selectors

Both spend a token budget over the same union; they sit at different points on the
precision/recall frontier, and the defaults encode where each was measured to operate best.

**`packed` — the knapsack QKP (`Packer`). The precision point.** Maximises relevance + graph
value − redundancy under the budget. Measured behaviour: the greedy solver **saturates around
~5.5k tokens** (10k and 16k budgets produced identical selections), and pure greedy stranded
~45% of a 10k budget — on a retrieval bench, unspent budget is unrecovered recall. Hence the
default pairing: `budget: 10_000` with `fill: "backfill"`, which spends the remainder in value
order **without touching the QKP phase's picks** (the greedy selection is unchanged; backfill
only appends). `pack_centrality_weight: 0.25` puts each candidate's peak `kg_node_centrality`
into the solver's value — the one graph feature that reaches selection as a *value* rather than
an ordering. Choose `packed` when the consumer is context-window-sensitive and you want the
densest defensible context.

**`coe` — chain-of-evidence (`coe.select`). The recall point.** CoE selects *Hyperedges*, not
arbitrary chunks, so the composition lets CoE decide which **rescue** candidates are evidence
worth budget, then fills with the base in its own rank order up to the budget. Measured
behaviour: unlike the knapsack, CoE **keeps spending productively as the budget grows** (9,944
tokens at a 10k budget → 15,942 at 16k, with document recall still rising) — so its default
budget is `16_000`, where the spend still buys recall. Modern LLMs are comfortable well past
that (the ruling: "going up to 16k for a retrieval is not a problem at all"). Choose `coe` when
multi-hop chain coverage matters more than context density. Requires the hypergraph lane.

**`none`** returns the full ranked union (base + rescue appendix) — for callers that run their
own downstream selection, and for like-for-like `recall@k` comparisons against plain rankings.

## Config reference (measured defaults)

| Key | Default | What it is |
| --- | --- | --- |
| `graph_mode` | `"auto"` | the policy gate: `off` / `auto` / `force` |
| `selector` | `"packed"` | `packed` / `coe` / `none` |
| `budget` | `10_000` (packed) / `16_000` (coe) | the selector's token budget |
| `pool_k` | `100` | first-stage depth per fetch leg |
| `select_pool_k` | `100` | union candidates the selector may choose from |
| `seed_top_k` | `5` | expander anchor: top-k base candidates |
| `hop_weight` | `0.05` | expander bonus per 1-hop-linked seed (units of the unit-scaled base; the base's head gap is ~0.016, so this lifts a corroborated candidate a few ranks and never vaults the head) |
| `community_weight` | `0.01` | expander shared-community bonus |
| `centrality_weight` | `0.05` | expander peak-centrality bonus |
| `seed_docs_k` | `5` | first-stage docs seeding entity resolution |
| `seed_node_cap` | `32` | ceiling on traversal seeds (query seeds first; five news chunks carry ~68 entities — uncapped expansion resolves back to half the corpus) |
| `rescue_cap` | `50` | rescue candidates appended per query (a tail, not a second ranking) |
| `pack_centrality_weight` | `0.25` | `kg_node_centrality` weight in the packer's value |
| `similarity_key` | `"redundancy"` | packer's precomputed-redundancy metadata key; absent pairs fall back to lexical Jaccard |
| `fill` | `"backfill"` | packer's unspent-budget policy (`stop` = pure QKP) |
| `rank_score_k` | `60` | the union's `1/(k+rank)` decay constant |
| `relation_cues` | shipped lexicon | the gate's relation-cue words |
| `entity_heavy_seeds` | `2` | query seeds at/above which a query is entity-heavy |
| `agreement_k` | `10` | top-k window of the dense/BM25 agreement signal |
| `low_agreement` | `0.30` | below = the first stages disagree (gate on) |
| `high_agreement` | `0.50` | at/above = they agree (known-item veto input) |
| `low_confidence` | `0.60` | dense top-1 below = unconfident (corpus-dependent) |
| `navigational_max_words` | `6` | short-query bound of the known-item veto |
| `hyperedge_kv_path` | `None` | emitted hyperedge store; pairs with the injected `hyperedge_documents` resolver |

Unknown keys are rejected loudly at construction; so is every contradiction (`coe` without the
hypergraph lane, half a hyperedge pair, no base first stage).

## Configuring it from the campaign wizard

`latence-campaign init` prompts for the two headline choices (policy gate, selector) and writes
the full block with the measured defaults — every other knob is flag-only
(`--retrieval-packed-budget`, `--retrieval-hop-weight`, …), and identical answers produce a
byte-identical file:

```yaml
retrieval:
  graph:
    mode: auto
    seed_docs_k: 5
    seed_node_cap: 32
    rescue_cap: 50
  expand:
    seed_top_k: 5
    hop_weight: 0.05
    community_weight: 0.01
    centrality_weight: 0.05
  select:
    selector: packed
    packed_budget: 10000
    coe_budget: 16000
    pack_centrality_weight: 0.25
```

A pre-existing campaign.yaml without the block keeps loading and parses to the defaults; a typo'd
knob is a loud parse error, never a silent fallback.

The block maps onto `GraphRescueRetrieval`'s config — same names, same semantics — so a
campaign.yaml is a valid production config source, not a bench-only dialect. That mapping is a
FUNCTION, not a prose claim: `latence_benchmark.campaign.retrieval_lib.rescue_config(answers)`
is its one home, and you can print what your own config resolves to:

```console
$ latence-campaign retrieval-config -c campaign.yaml
```

Two tests enforce it (`packages/latence-benchmark/tests/test_campaign_retrieval_wizard.py`):
`TestRescueConfig::test_every_emitted_key_is_a_real_graph_rescue_config_key` pins every emitted
key to the component's own `_CONFIG_KEYS`, so renaming a knob on either side fails a test instead
of surfacing as `unknown config key(s)` on a run; and
`TestRescueConfig::test_the_defaults_construct_the_production_component` constructs a real
`GraphRescueRetrieval` from the defaults, so the VALUES are pinned too, not just the names.

Two deliberate non-identities, both documented on the function:

* the block carries **both** per-selector budgets (`packed_budget`, `coe_budget`) so switching
  selectors keeps each one's measured operating point; the component takes the single `budget` of
  the selector in force, and selector `none` gets none at all;
* the deployment-side knobs (`pool_k`, `select_pool_k`, `rank_score_k`, the gate thresholds,
  `hyperedge_kv_path`) are **not** campaign answers — their component defaults are the measured
  operating point, and the deployment merges its own on top. Selector `coe` is the case that
  matters: it selects over the hypergraph lane, so the deployment must add `hyperedge_kv_path`
  plus the injected `hyperedge_documents` resolver or construction is refused.

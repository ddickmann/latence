# 4 · Enrichment and export

*Chapter 3 finished with a graph: canonical nodes, evidence-linked edges, hyperedges, and a
centrality/community stamp on every node. That graph is in memory, and the chunks that produced
it still know nothing about it. This chapter closes the loop — it projects the graph back onto
the chunks, then writes everything to disk. From here on there is no framework in the loop: what
lands in `export/` is the whole interface to everything downstream.*

Two stages do the work, and they are wired to different parents:

```
redact ─┐
        ├──► enrich (context_enrichment / context.kg_header) ──► export_corpus (export.jsonl_parquet)
graph ──┤
        └──► export_kg (export.knowledge_graph)
```

`enrich` depends on **both** `redact` and `graph` — it needs the chunk stream and the assembled
KG in the same pass. `export_kg` hangs off `graph` alone and never sees a chunk. The two exports
are independent and write disjoint file sets into the same directory.

---

## 4.1 The context header

### What it is

For each chunk, the enricher builds a `ContextHeader`: the chunk's own canonical entities, plus
the top few KG relations those entities participate in, rendered as compact triples. It is a
handful of short strings — never neighbour prose, never a subgraph dump.

The stage never drops, reorders or rewrites a chunk. It attaches one additive field:

```python
# packages/latence-core/src/latence_core/stages/context_enrichment.py:338-345
for chunk in chunks:
    entities = by_chunk.get(chunk.record_id)
    if entities is None:
        # No gated canonical entity: nothing to add — pass the chunk through byte-identical.
        yield chunk
    else:
        header = self._build_header(entities, nodes, adjacency)
        yield chunk.model_copy(update={"context_header": header})
```

A chunk with no confidence-gated canonical entity keeps `context_header = None` and is
byte-identical to a run without the stage. `content` is never touched by this stage — not here,
and not later.

### The four knobs

| Key | Type | Default (`context_enrichment.py:307-310`) | Campaign value | What it bounds |
|---|---|---|---|---|
| `max_neighbors_per_entity` | int | `4` | `4` | Relations **one entity** may contribute. `0` = unbounded. |
| `max_triples` | int | `6` | `6` | Triples in the **whole header**, after dedup. `0` = unbounded. |
| `min_entity_confidence` | float | `0.0` | *(unset)* | Floor on both the mention confidence **and** the node confidence. |
| `min_edge_confidence` | float | `0.0` | *(unset)* | Floor on edge confidence, applied before ranking. |

The campaign pipelines set only the two caps. Both confidence floors sit at `0.0`, which means
**every** gated mention and **every** edge the assembler emitted is a candidate — the caps, not
a confidence threshold, are what keep the header small.

### How the header is built

Three passes, in this order.

**1 — index the graph** (`_index_graph`, `context_enrichment.py:347-382`). Node records become
a `node_id → _NodeView` map and a `mention_id → node_id` join; edge records that clear
`min_edge_confidence` become `_EdgeMeta`. Then adjacency is built so that **each edge is
registered under both endpoints**, with the rendered triple identical from either side:

```python
# context_enrichment.py:373-381
triple = f"{src.canonical_name}{_ARROW_L}{meta.label}{_ARROW_R}{tgt.canonical_name}"
# Seen from the source node, the neighbor is the target; from the target it is the
# source. The rendered triple is identical (the edge's asserted direction) either way.
adjacency.setdefault(meta.source_node_id, []).append(
    _Neighbor(meta.confidence, triple, meta.target_node_id, meta.edge_id)
)
adjacency.setdefault(meta.target_node_id, []).append(
    _Neighbor(meta.confidence, triple, meta.source_node_id, meta.edge_id)
)
```

`_ARROW_L` is `" —"` and `_ARROW_R` is `"→ "`, so the rendered surface is `A —label→ B` in the
edge's **asserted** direction regardless of which endpoint reached it. An edge whose endpoint is
missing from the node set is skipped, not rendered with a placeholder.

**2 — fold the mentions** (`_fold_mentions`, `context_enrichment.py:384-421`). The corpus-wide
mention stream is reduced, on arrival, to `chunk_record_id → {node_id: strongest gated mention
confidence}`. Both the mention confidence and the node confidence must clear
`min_entity_confidence`. Nothing mention-shaped survives the fold — the mention→node join table
is explicitly deleted before the chunk loop starts (`context_enrichment.py:336`). That is a
memory fix, not a semantic one: a maximum is order-independent, so the folded result is exactly
what a per-chunk pass would compute.

**3 — build one header** (`_build_header`, `context_enrichment.py:423-501`):

- entities are sorted by `(canonical_name, node_id)`; `entities`, `entity_types`,
  `entity_confidences`, `kg_node_ids`, `kg_node_centrality`, `kg_node_community` are all
  **aligned 1:1** with that order;
- `entity_confidences[i]` is `min(best gated mention confidence, node confidence)` — the exact
  value at which this chunk's admission gate would admit the entity, carried so an index-time
  sparse embedder can apply the identical rule (ADR-0053);
- the graph features are **all-or-nothing**: if any one entity lacks a well-typed
  `centrality`+`community` pair, both lists come back empty (`_graph_features`,
  `context_enrichment.py:234-253`);
- then the two caps, in order.

### The two caps, precisely

```python
# context_enrichment.py:460-480
kept: list[_Neighbor] = []
for node_id in own_ids:
    neighbors = sorted(adjacency.get(node_id, []), key=lambda n: n.sort_key())
    cap = self._max_neighbors if self._max_neighbors > 0 else len(neighbors)
    if len(neighbors) > cap:
        self.hub_entities_capped += 1
    kept.extend(neighbors[:cap])

best_by_triple: dict[str, _Neighbor] = {}
for neigh in kept:
    existing = best_by_triple.get(neigh.triple)
    if existing is None or neigh.sort_key() < existing.sort_key():
        best_by_triple[neigh.triple] = neigh
ranked = sorted(best_by_triple.values(), key=lambda n: n.sort_key())
overall_cap = self._max_triples if self._max_triples > 0 else len(ranked)
capped = ranked[:overall_cap]
```

`sort_key()` is `(-confidence, triple, edge_id)` — a **total** order, so ties never depend on
iteration order and a seeded run is byte-identical. `edge_id` is content-addressed, so the
tiebreak survives a re-run on another host.

The per-entity cap is the **hub guard**, and it is the thing that makes a graph header safe to
build over a real corpus. A corpus-wide hub — `London`, a company name, a product line — has
hundreds of relations; without the cap, every chunk mentioning it would inherit all of them and
the header would swamp the chunk. With `max_neighbors_per_entity: 4` the hub contributes its
four highest-confidence relations and nothing else. The stage counts how often that bites, as a
counts-only witness (`self.hub_entities_capped`, `context_enrichment.py:311-315`) — no content,
so it is safe to surface.

The overall cap then applies **after** deduplication, so an internal relation between two of the
chunk's own entities (reachable from both endpoints) costs one slot, not two.

Finally the expansion keys:

```python
# context_enrichment.py:483-488
own_id_set = set(kg_node_ids)
neighbor_node_ids = sorted(
    {n.neighbor_node_id for n in capped} - own_id_set
)
```

`neighbor_node_ids` is the set of node ids the surviving triples reach **excluding the chunk's
own entities** — an internal-only relation contributes no external neighbour. These are the
1-hop expansion keys the retrieval side reads; expansion itself is the retriever's job (ADR-0017:
files, not a live database — the pipeline only emits the keys).

### Worked example

One chunk, run through the real code with the campaign's `4`/`6` caps:

```
content: "Ada Lovelace worked with Charles Babbage on the Analytical Engine in London."
mentions: Ada Lovelace (0.93), Charles Babbage (0.89), Analytical Engine (0.81)
KG edges touching those entities:
  Charles Babbage —created_by→   Analytical Engine   0.90
  Ada Lovelace    —created_by→   Analytical Engine   0.86
  Charles Babbage —located_in→   London              0.80
  Ada Lovelace    —located_in→   London              0.72
  Ada Lovelace    —member_of→    Royal Institution   0.65
  Ada Lovelace    —works_for→    Royal Institution   0.55
  Ada Lovelace    —spouse_of→    Charles Babbage     0.40
```

`Ada Lovelace` has five relations, one more than `max_neighbors_per_entity: 4`, so her weakest
(`spouse_of`, 0.40) is dropped and `hub_entities_capped` increments to 1. That same triple is
still reachable from `Charles Babbage` (who has only three relations, all kept) — so after
dedup, seven distinct triples remain and `max_triples: 6` cuts the weakest, which is again
`spouse_of`. Both caps bite, in that order. The produced header:

```json
{
  "entities":           ["Ada Lovelace", "Analytical Engine", "Charles Babbage"],
  "entity_types":       ["PERSON", "PRODUCT", "PERSON"],
  "entity_confidences": [0.93, 0.81, 0.89],
  "kg_node_ids":        ["n-ada", "n-ae", "n-cb"],
  "kg_node_centrality": [0.5, 0.5, 0.5],
  "kg_node_community":  ["c-1", "c-1", "c-1"],
  "neighbor_node_ids":  ["n-lon", "n-ri"],
  "neighbor_triples": [
    "Charles Babbage —created_by→ Analytical Engine",
    "Ada Lovelace —created_by→ Analytical Engine",
    "Charles Babbage —located_in→ London",
    "Ada Lovelace —located_in→ London",
    "Ada Lovelace —member_of→ Royal Institution",
    "Ada Lovelace —works_for→ Royal Institution"
  ],
  "situating_sentence": null,
  "model_id": null
}
```

Note `neighbor_node_ids` holds only `n-lon` and `n-ri`. `n-ada`, `n-ae` and `n-cb` are the
chunk's own entities and are excluded.

`situating_sentence` and `model_id` are always `null` in the shipped build. The LLM
situating-sentence mode (Contextual-Retrieval style) is a documented follow-on in
[ADR-0039](../adr/0039-context-enrichment-optional-stage-kg-projected-onto-chunks-embedding-input-only.md),
deliberately not implemented here — it would make the stage non-deterministic and require a model
in the default path.

---

## 4.2 Where the header goes — the split that makes it measurable

This is the part most easily misread, and it is worth being exact about, because it is the
mechanism the ablation in chapter 5 measures.

The header is rendered into text by the **export**, not the enricher:

```python
# packages/latence-core/src/latence_core/stages/export.py:115-127
entities = header.get("entities") or []
entity_types = header.get("entity_types") or []
triples = header.get("neighbor_triples") or []
parts: list[str] = []
if entities:
    rendered = ", ".join(
        f"{name} ({etype})" if etype else str(name)
        for name, etype in zip(entities, entity_types, strict=False)
    )
    parts.append(f"Entities: {rendered}")
if triples:
    parts.append("Relations: " + "; ".join(str(t) for t in triples))
return " | ".join(parts)
```

and prepended with a single newline (`_corpus_text`, `export.py:177-183`):

```python
base = _corpus_content(row, unmasked=unmasked)
header = row.get("context_header")
if isinstance(header, dict):
    header_text = _context_header_text(header)
    if header_text:
        return f"{header_text}\n{base}"
return base
```

For the worked example above, the **embedding input** is exactly:

```
Entities: Ada Lovelace (PERSON), Analytical Engine (PRODUCT), Charles Babbage (PERSON) | Relations: Charles Babbage —created_by→ Analytical Engine; Ada Lovelace —created_by→ Analytical Engine; Charles Babbage —located_in→ London; Ada Lovelace —located_in→ London; Ada Lovelace —member_of→ Royal Institution; Ada Lovelace —works_for→ Royal Institution
Ada Lovelace worked with Charles Babbage on the Analytical Engine in London.
```

An empty header renders `""` and no newline is added, so a chunk with nothing to say embeds
exactly its own text.

### Which sinks see the header, and which do not

`_attach_signals` (`export.py:803-861`) builds **two text views per batch** and hands them to
different consumers:

| Sink | Text it sees | Header? | Why |
|---|---|---|---|
| Dense `embedding` column | `_corpus_text` | **yes** | A vector exposes no readable term — the coherence inflates the vector, not the text. |
| `sparse_indices`/`sparse_values` | `_corpus_content` | no | Sparse term ids are an inspectable index; a header term would be readable. |
| Multi-vector / `fde_embedding` | `_corpus_content` | no | Same reason — raw per-token vectors go into an inspectable index. |
| `bm25-postings.parquet` / `bm25-stats.json` | `_corpus_content` | no | The artifact must match what a store actually indexes. |
| Stored `content` cell (JSONL + Parquet) | — | no | Never touched. The LLM window stays the clean chunk. |
| `context_header` field/column | the header itself | (see §4.5) | Inspectable, so it goes through a redaction gate first. |

So exactly one thing changes when enrichment is on: **the dense vector**. Everything else —
stored text, lexical statistics, sparse signals — is byte-identical to a run without the stage.
That is what makes the intervention isolable.

### The ablation: `dense_content_only`

Chapter 5's leg table carries one ablation, and this stage is its entire subject:

```python
# benchmark/s5/legs.py:215-226
"dense_content_only": LegSpec(
    family="ablation",
    signals=("dense_content_only",),
    required_files=("corpus.parquet", CONTENT_ONLY_VECTORS),
    ...
    vector_source="sidecar",
```

It is **not** a ladder rung. It re-runs the `dense` rung against a second set of vectors built
from each chunk's content **only**, and the report subtracts it from `dense`
(`ABLATION_BASELINE = {"dense_content_only": "dense"}`, `legs.py:247`). Only the corpus side
changes — queries have no context header, so both legs rank against identical query vectors.

The counterfactual vectors are produced by `benchmark/s5/build_vectors.py`, and the switch is
one line:

```python
# benchmark/s5/build_vectors.py:802-804
if with_context_header:
    return _corpus_text(row, unmasked=True)
return _corpus_content(row, unmasked=True)
```

It calls the **framework's own** selectors, not a local re-implementation — which is what makes
the comparison a controlled one rather than two different pipelines. The content-only matrix is
written to a sidecar (`corpus-vectors-content-only.parquet` — `record_id` + a fixed-width
`float32` `embedding`, in corpus row order, nothing else), and
`benchmark.s5.corpus.read_vector_sidecar` refuses any sidecar whose ids are not the corpus's ids
in the corpus's order — a misaligned matrix would still load as a perfectly shaped array and
every rank it produced would be wrong-but-plausible, so the disagreement raises rather than warns.

**To reproduce or refute the ablation** you need: an export directory with `corpus.parquet`, the
same checkpoint the export used, and one command per variant. The default variant re-derives the
header-enriched vectors; `--no-context-header --vectors-out <path>` derives the counterfactual.
Both are checkpointed into separate shard directories, precisely because the two variants embed
*different text for the same ids* and a shared cache would let one silently inherit the other's
vectors.

### Why a graph header helps dense retrieval at all

The honest mechanism is unglamorous. A chunk is embedded in isolation, so a query whose answer
depends on a fact stated *elsewhere in the corpus* has nothing to match on. The header injects,
into the vector, two things the chunk's own bytes do not carry:

1. **Canonicalised surfaces.** The chunk may say "the Engine"; the header says
   `Analytical Engine (PRODUCT)` because disambiguation resolved it. A query using the canonical
   name now has lexical *and* semantic mass to hit.
2. **One hop of corpus-level structure.** `Ada Lovelace —located_in→ London` is a fact the chunk
   never states. A query about London and the Analytical Engine has somewhere to land.

And the risks, which the code bounds explicitly rather than hoping about:

| Risk | What the code does |
|---|---|
| **Bloat** — the header crowds out the chunk in the vector | Compact triples, never neighbour prose; two hard caps; the header rides the *embedding input only*, so the stored chunk and the LLM window never grow. |
| **Hub domination** — one popular entity's relations flood every chunk that mentions it | `max_neighbors_per_entity` truncates per entity, and the truncation is counted (`hub_entities_capped`). |
| **Redundancy** — the same triple counted twice | Dedup by rendered triple before the overall cap, keeping the highest-confidence witness. |
| **Nondeterminism** — a re-run produces different headers | Total sort order with a content-addressed tiebreak; every collection sorted. |
| **Weak NER pollution** | `min_entity_confidence`/`min_edge_confidence` floors exist — though the campaign leaves both at `0.0`, so on that corpus the caps are the only filter. |

What the code does **not** do: measure whether the header helps. That is the ablation's job, and
its result is chapter 5's to report.

---

## 4.3 `export.knowledge_graph` — the graph as portable files

Config in the campaign pipelines is two lines: `basename: graph`, `hyperedge_kv: true`.

`hyperedge_kv` defaults to **`true`** in the code (`graph_export.py:131`) — within the
already-opt-in hyperedge path, the strong configuration is the default and the cheap mode is the
documented opt-out. The pipeline YAML states it explicitly rather than relying on that.

The stage consumes graph records in **one pass**, keeping only the inner records and dropping the
envelopes as it goes (`graph_export.py:144-158`), and writes:

| File | Written when | Contents |
|---|---|---|
| `graph-nodes.parquet` | always | one row per `GraphNode` |
| `graph-edges.parquet` | always | one row per `GraphEdge` |
| `graph.ttl` | always | RDF Turtle, hand-emitted, no `rdflib` dependency |
| `graph.graphml` | always | GraphML XML — loads in yEd, Gephi, networkx |
| `graph-hyperedges.parquet` | any hyperedge present | one row per `GraphHyperedge` |
| `graph-hyperedges.kv` | hyperedges present **and** `hyperedge_kv` | the derived point-lookup KV |

All of them ride **one** `Storage.atomic_writers` publication (`graph_export.py:187-196`), so a
consumer can never observe a new `graph.ttl` beside a stale `graph-hyperedges.kv`. A crash
mid-export leaves the previous complete set.

### `graph-nodes.parquet`

`node_id`, `entity_id`, `type`, `canonical_name`, `kb_id`, `confidence`,
`member_mention_ids` (JSON), `source_document_ids` (JSON), `properties` (JSON), `evidence` (JSON)
— **plus** `centrality` (`double`) and `community` (`string`) when any node carries them
(`graph_export.py:207,247-249`). The entity type column is named `type`, not `label`, so it
matches GraphML's `<data key="type">` and TTL's `lg:entityType`: the same entity type under the
same name in all three formats.

> **Doc drift.** `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md` §2.4 documents this table without the
> `centrality`/`community` columns. The code emits them whenever the `graph_features` enrichment
> ran — which the campaign pipelines configure (`graph_features.degree`).

### `graph-edges.parquet`

`edge_id`, `relation_id`, `source_node_id`, `target_node_id`, `label`, `confidence`,
`properties` (JSON), `evidence` (JSON). Every edge carries `Evidence` — the mention and document
ids that justify it.

### `graph.ttl` and `graph.graphml`

TTL emits each node as a subject under a non-dereferenceable `urn:latence:graph:` base, and each
edge **twice**: once as a direct triple and once as a reified `rdf:Statement` carrying
`lg:confidence` and `lg:evidenceCount` (`graph_export.py:352-365`). Confidences are formatted
`"%.6f"` so two runs are byte-identical.

Both writers scrub characters XML 1.0 and strict Turtle forbid (`_XML_ILLEGAL`,
`graph_export.py:70-77`), because node names and edge labels are verbatim spans sliced out of
messy source documents — a stray form-feed or NUL would otherwise make the file unparseable in
every standard tool.

### `graph-hyperedges.kv` — the on-disk shape retrieval reads back

The KV is **derived and always rebuildable** from `graph-hyperedges.parquet`; the parquet is the
durable form. The reference implementation is dependency-free by design: a single file of sorted
keys, `mmap`-ed and binary-searched (`hyperedge_store.py`). LMDB/RocksDB adapters are a Provider
option behind the same read interface — the *format* is the contract, the library is not.

```
magic  b"LHKV1\n"
n                       number of entries          (all integers little-endian u64)
key_offsets  [n+1]      into the key blob
value_offsets[n+1]      into the value blob
key blob                keys, concatenated, SORTED bytewise
value blob              values, concatenated
```

One file, four keyspaces distinguished by prefix (`hyperedge_store.py:78-124`):

| Key | Value | Read by |
|---|---|---|
| `e\|<hyperedge_id>` | solver fields as sorted-key JSON: `entity_node_ids`, `gamma`, `has_relations`, `support`, `token_cost`, `token_counter` — no text | the chain-of-evidence selector |
| `p\|<node_id>` | `[[hyperedge_ids…], [gammas…]]`, **γ-sorted descending with γ inline** | candidate lookup — top-k truncation needs no row fetch |
| `t\|<hyperedge_id>` | `tau_text` as UTF-8 | the packer, when it needs the span text |
| `a\|<normalised surface>` | JSON list of node ids | the zero-model alias entry point |

Alias keys are normalised NFKC + casefold + whitespace-collapsed (`normalize_alias`,
`hyperedge_store.py:66-72`) — deterministic and locale-independent. A reader is opened per call
and closed (`HyperedgeStoreReader`, `hyperedge_store.py:154-261`); it holds nothing between
queries, which is structural rather than a promise.

---

## 4.4 `export.jsonl_parquet` — the RAG corpus

### What lands on disk

With the campaign's config (`basename: corpus`, an embedder, `bm25: tokenizer.regex`,
`context_columns: true`):

| File | Written when | Contents |
|---|---|---|
| `corpus.jsonl` | always | one sorted-key JSON object per line — the **full** record dump |
| `corpus.parquet` | always | the flat columnar projection (below) |
| `bm25-postings.parquet` | `bm25` block present | `record_id`, `doc_len`, parallel `terms`/`term_frequencies` |
| `bm25-stats.json` | `bm25` block present | corpus term statistics, versioned and self-describing |
| `multivectors.parquet` | `keep_multivectors` | `record_id` + `list<list<double>>` sidecar (off here) |

Everything except `bm25-stats.json` rides one atomic publication (`export.py:712-771`). The
stats JSON is a small buffered artifact derived from a completed fold, so it is written
afterwards — and only once the postings it describes actually exist (`export.py:778-786`).

The export streams: records are consumed in batches of `_EXPORT_BATCH_ROWS = 1024`
(`export.py:85`), each batch becoming one Parquet row group and one JSONL flush, so peak memory
is O(batch) rather than O(corpus). An empty corpus still writes the schema-only Parquet, so a
reader always gets the column set.

### The Parquet column schema

Read straight off `_parquet_schema` (`export.py:929-972`). Column order is exactly this:

| # | Column | Arrow type | Present when | Meaning |
|---|---|---|---|---|
| 1 | `record_id` | `string` | always | primary key |
| 2 | `schema_version` | `int64` | always | the contract version this row was written at |
| 3 | `provenance` | `string` | always | `Provenance` as sorted-key JSON |
| 4 | `classification` | `string` | always | `Classification` as sorted-key JSON |
| 5 | `content` | `string` | always | **the corpus text** — see §4.5 |
| 6 | `media_type` | `string` | always | e.g. `text/markdown` |
| 7 | `embedding` | `list<double>` | `embedder` wired | fixed-width dense vector |
| 8 | `sparse_indices` | `list<int64>` | `sparse_embedder` wired | *(off in the campaign)* |
| 9 | `sparse_values` | `list<double>` | `sparse_embedder` wired | *(off in the campaign)* |
| 10 | `fde_embedding` | `list<double>` | `fde_converter` wired | *(off in the campaign)* |
| 11 | `context_header` | `string` | `context_columns: true` | the header as sorted-key JSON, `""` when absent |
| 12 | `kg_node_ids` | `list<string>` | `context_columns: true` | the chunk's own entity node ids |
| 13 | `neighbor_node_ids` | `list<string>` | `context_columns: true` | 1-hop expansion keys |
| 14 | `kg_node_community` | `list<string>` | `context_columns: true` | per-entity community id |
| 15 | `kg_node_centrality` | `list<double>` | `context_columns: true` | per-entity centrality |

The four native list columns are `[]` — never null — when the row has no header, when the row is
not a chunk, or (for the last two) when `graph_features` was off.

> **Doc drift, and this one bites.** `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md` §2.2 documents the
> context columns as `context_header` + **`neighbor_entity_ids`**. There is no
> `neighbor_entity_ids` column in the code and there never was on this branch: the column is
> `neighbor_node_ids`, and there are four of them, not one. A consumer coded against the spec
> gets a missing-column error. The code wins; the spec is stale.

### `context_columns: true` — exactly what it adds

It adds columns 11–15 above, and **nothing else**. Specifically:

- it does not change `content`, the embedding, the BM25 artifacts, or the JSONL in any way;
- the JSONL always carries the full `context_header` object (or `null`) via the ordinary record
  dump, so the flag is Parquet-only (`export.py:466-471`);
- with it absent or false, the Parquet schema is byte-identical to a build without the feature.

The column **names** are load-bearing. A vector store that copies non-text corpus columns into
candidate metadata (Qdrant's backend does) makes these readable by the query-time consumers under
their *default* metadata keys — `expand`'s `node_ids_key` / `neighbor_ids_key` / `community_key` /
`centrality_key`, and the packer's centrality signal
([ADR-0051](../adr/0051-zero-ops-multi-hop-graph-retrieval-via-duckdb-over-the-emitted-graph-parquet.md)).
Rename them and the computed graph features become dead weight nothing can read.

### JSONL vs Parquet — what the columnar file drops

The JSONL row is `record.model_dump(mode="json")` plus whatever signals were attached. For a
redacted, enriched chunk the keys are:

```
classification, content, context_header, document_record_id, embedding, index,
induced_labels, masked_content, media_type, offset_map, page_slice, pii_spans,
provenance, record_id, redaction_disabled_for_sensitive, risk_markers,
schema_version, token_count
```

The Parquet projection keeps six of those plus the opt-in columns. **`masked_content`,
`pii_spans`, `risk_markers`, `induced_labels`, `offset_map`, `page_slice`,
`document_record_id` and `token_count` exist only in the JSONL.** For most consumers that is
fine — `provenance` carries `document_id`, `page_start/page_end`, `char_start/char_end`. For a
governance consumer it matters a great deal, and §4.5 says why.

### The embedder, from source

The campaign wires `embedding.sentence_transformers` with
`model: ibm-granite/granite-embedding-311m-multilingual-r2`, `dimension: 768`, `device: cuda`.
Every one of those is the provider's own default except the device
(`latence-embedder-st/src/latence_embedder_st/provider.py:76,85,88`):

| Fact | Value | Source |
|---|---|---|
| Default model | `ibm-granite/granite-embedding-311m-multilingual-r2` | `provider.py:76` |
| Default dimension | `768` | `provider.py:85` |
| CPU tier | `ibm-granite/granite-embedding-97m-multilingual-r2` (384) | `provider.py:79,31-33` |
| Default prefix | `""` — Granite r2 needs no instruction prefix | `provider.py:88` |
| Normalisation | `normalize_embeddings=True`, always | `provider.py:308-313` |
| Batch size | `config.batch_size`, default **32** | `providers/adapter.py:76,266-273` |
| Determinism | declared **`False`** | `provider.py:180` |
| Licence | Apache-2.0 weights **and** Apache-2.0 code, verified 2026-07-14 | `provider.py:163-179` |

Mechanics worth knowing:

- **The model loads lazily**, on the first `embed` call, not at construction — so the Runner
  constructing every stage never downloads weights (`provider.py:212-238`).
- **The declared dimension is checked against the model**, on the eager module before any compile
  wrap. A mismatch is a loud `ProviderError`, never a silently wrong column
  (`_check_dimension`, `provider.py:258-273`). The width is then re-asserted per returned vector.
- **Device handling** goes through the shared device seam; a *skipped* device decision is a loud
  `ConfigError`, not a silent CPU fallback. On GPU the perf seam resolves bf16 + flash-attention-2
  with a graceful `sdpa` fallback; on CPU, fp32 + `sdpa`.
- **`deterministic=False` is declared honestly.** bf16 + FA2 is explicitly not bitwise
  reproducible, so the profile does not claim it is. The deterministic `embedding.hashing`
  reference (in-core, zero-dep, default `dimension = 64`, `embedding.py:34`) stays the default on
  the golden/unit-test path.

The embedder is a **nested sub-provider of the Export**, resolved at wiring time
(`export.py:575`, `_resolve` at `export.py:622-641`). A malformed block is a `ConfigError` and a
provider that does not satisfy the `Embedder` protocol is a `ContractError` — both before the
pipeline runs, never a silently-absent signal.

With no `embedder` block the corpus still exports its text with full provenance and
classification, so any vector store can ingest it and embed later (ADR-0017). The embedding is an
augmentation, not a requirement.

### BM25 — an artifact, not an index

The `bm25` block names a `Tokenizer`; the campaign uses `tokenizer.regex`, whose term unit is
`[^\W_]+` over lower-cased text — maximal runs of Unicode letters, digits and combining marks,
with the underscore as a boundary (`bm25.py:69,126-128`). This is the *same* pattern the in-core
dense and sparse hashing references use, so the lexical statistics line up with those signals.

The tokenizer is honest about its limit: it is Unicode-aware, **not** a word segmenter. `müller`,
`für` and `таможенный` survive whole; Chinese, Japanese and Thai split only at whitespace, so an
unspaced run becomes one term. A corpus in those scripts plugs its own segmenter in behind the
`Tokenizer` seam — which is why the tokenisation is a seam rather than hard-coded.

Two files come out.

`bm25-postings.parquet` — one row per record: `record_id`, `doc_len` (`int64`), and parallel
`terms` (`list<string>`) / `term_frequencies` (`list<int64>`). `Bm25Posting` enforces the
canonical shape at construction: terms strictly ascending (hence unique), frequencies all `≥ 1`,
and `doc_len == sum(term_frequencies)` (`bm25.py:174-198`). A malformed posting cannot reach the
writer.

`bm25-stats.json` — the corpus fold:

```json
{
  "artifact": "latence.bm25.termstats",
  "version": 1,
  "tokenizer": "tokenizer.regex",
  "doc_count": 1,
  "total_tokens": 11,
  "avgdl": 11.0,
  "terms": [{"term": "analytical", "df": 1, "idf": 0.28768207}, ...]
}
```

`idf` is the Robertson/Lucene form `ln(1 + (N − df + 0.5) / (df + 0.5))` (`bm25.py:233-242`) —
always non-negative, and the form Lucene/Elasticsearch/Qdrant use, so a store can adopt the
precomputed value directly. Terms ascend and every derived value is rounded to 8 decimals, so two
runs are byte-identical. The tokenizer name is stamped into the artifact because **a store must
analyse queries the same way** — that field is the contract, not a courtesy.

Memory is O(vocabulary): only the `df` map and two scalars are retained across the corpus; the
heavy per-document postings are returned to the caller and streamed one row group per batch
(`Bm25Accumulator`, `bm25.py:265-330`).

What this is **not** is a search index. Latence never holds one and never runs a first-stage
search ([ADR-0048](../adr/0048-retrieval-tooling-is-strictly-stateless.md)). These two files are
what a store loads to drive **its own** BM25.

### The other knobs on this stage

| Key | Default | Effect |
|---|---|---|
| `basename` | `"records"` | file stem; the campaign uses `corpus` |
| `unsafe_unmasked_corpus` | `False` | §4.5 |
| `embedder` | absent | dense `embedding` column |
| `context_columns` | `False` | Parquet columns 11–15 |
| `sparse_embedder` | absent | `sparse_indices`/`sparse_values` |
| `multivector_embedder` | absent | source for FDE + the sidecar; **adds no column by itself** |
| `fde_converter` | absent | `fde_embedding`; requires `multivector_embedder` |
| `keep_multivectors` | `False` | `multivectors.parquet` sidecar; requires `multivector_embedder` |
| `bm25` | absent | the two BM25 files |
| `iceberg` | absent | see below |

The two wiring invariants — `fde_converter` and `keep_multivectors` each require
`multivector_embedder` — are checked at construction and raise `ConfigError`
(`export.py:608-620`), so you get a loud failure at wiring time rather than a silently empty
column.

**Yes, the export can target Iceberg**, and the campaign does not use it. An `iceberg` block
(`{"uri": <REST catalog>, "namespace", "table", "mode": "append"|"overwrite"}`) *additionally*
registers the run's corpus as an Apache Iceberg table in a REST catalog (Unity, Polaris,
Glue-via-REST, Lakekeeper). The details matter for anyone evaluating it:

- the Parquet + JSONL deliverable stays the source of truth and is **byte-identical** whether or
  not the block is present — the export directory itself is untouched;
- what is added is an immutable, **content-addressed** copy of the Parquet inside the *target
  table's own* warehouse location, named after the sha256 of its own bytes, plus the catalog
  registration. Not in the ephemeral run tree — a data file published there would be destroyed by
  an unrelated purge while the snapshot went on citing its path;
- registration happens **last**, after the deliverable is atomically published, so a crash before
  it leaves a normal export and simply no table;
- `pyiceberg` is imported lazily inside the call and is not a `latence-core` runtime dependency
  (it ships as the `latence-core[iceberg]` extra). Absent-and-unconfigured costs nothing;
  configured-but-absent fails loudly with the install line;
- `mode` is documented honestly: `append` adds a snapshot (right for a delta run — a repeated
  *full* export therefore duplicates rows), `overwrite` replaces the contents in one atomic
  commit (right for a full re-export).

---

## 4.5 Governance — the export is where it is kept or lost

Everything upstream can be impeccable and still leak here, because this is the only stage that
writes bytes an outside system reads.

### The default: masked text is the corpus text

```python
# export.py:1118-1119
if not unmasked and "content" in row and row.get("masked_content") is not None:
    row["content"] = row["masked_content"]
```

Whenever the redaction stage produced a `masked_content`, **that** is the exported `content` —
the JSONL cell, the Parquet cell, the embedder input, the sparse input, the BM25 postings. The
raw text never reaches the RAG corpus. A row with no `masked_content` exports its own `content`
unchanged, byte-identical to a run without redaction.

### `unsafe_unmasked_corpus: true` — precisely what changes

The campaign sets it. Here is the complete delta, from the code rather than the YAML comment:

| Surface | Default (`False`) | With `unsafe_unmasked_corpus: true` |
|---|---|---|
| `content` (JSONL + Parquet) | `masked_content` | **raw `content`** |
| dense `embedding` | over masked text + header | over **raw** text + header |
| sparse / multi-vector / FDE | over masked text | over **raw** text |
| `bm25-postings.parquet` / `bm25-stats.json` | over masked text | over **raw** text |
| `masked_content` (JSONL) | present | **still present** |
| `pii_spans` (JSONL) | present | **still present** |
| exported `context_header` | redaction-gated | **still redaction-gated** (see below) |
| log output | — | one `WARNING` per export, before the first byte |

The warning is a module constant so the banner, the docstrings and the regression test read the
same words (`export.py:96-99`):

> Export config 'unsafe_unmasked_corpus' is ON: exporting UNMASKED corpus text; PII detections
> are recorded in pii_spans but NOT substituted — benchmark/offline use only.

**The trade.** What you buy is a corpus whose text is the real text. The measured reason: with
masking on, 82% of the exported 2WikiMultiHopQA / MultiHop-RAG rows carried `[PERSON]`-style
placeholders *including the article titles the questions target*, and the reused musique/hotpot
exports were unmasked — so retrieval scores were neither valid in themselves nor comparable
across datasets. What you give up is the substitution, and the give-up is broader than "the text
column":

- the **BM25 artifact now contains the PII terms**. Verified on the worked example: with masking
  on, the vocabulary is `analytical, babbage, charles, engine, in, london, on, person, the,
  with, worked` — note `person`, the placeholder's own token, and note that `ada` and `lovelace`
  are gone. With the flag on, `ada` and `lovelace` are back in `bm25-stats.json` with a df, an
  idf, and a posting. A lexical index built from that artifact is queryable *for the name*;
- the audit evidence survives only in the **JSONL**. `pii_spans` and `masked_content` are not
  Parquet columns. A consumer who loads `corpus.parquet` — the file every vector-store ingestion
  path actually reads — has the raw text and no record that anything was ever detected.

That is why the flag is named the way it is. It is meant to be impossible to read
`unsafe_unmasked_corpus: true` in a stack YAML, a diff, or a `grep` and not know what was traded
away. It exists for retrieval benchmarking, and the code says so in three places. It is not a
production posture.

### The header redaction gate

The header is built from entities the NER found over **raw** text (`extract_on: unmasked` in the
campaign), so a person entity's `canonical_name` is the real name — the same one redaction masked
out of `masked_content`. That is harmless in a vector and a re-leak in an inspectable cell. So
between the signal generation and the inspectable sinks, the export re-gates the header
(`_redacted_context_header`, `export.py:237-320`):

- a **name** is kept only if its surface still occurs, **as a whole word or phrase**, in the
  redacted corpus text — never as a bare substring, so a masked `Ann` does not "survive" inside
  `annual` (`_occurs_as_phrase`, `export.py:207-219`);
- when an entity is dropped, its aligned `entity_types`, `entity_confidences`, `kg_node_ids`,
  `kg_node_centrality` and `kg_node_community` entries are dropped **together**, so the arrays
  stay consistent;
- a triple is dropped unless **both** endpoint names survive; endpoints are recovered by
  splitting on the exact arrow markers, so the relation *label* is never mistaken for a name
  (`_triple_endpoint_surfaces`, `export.py:222-234`);
- an unparseable triple is treated as one surface and, failing to find it, dropped — redaction by
  default;
- `neighbor_node_ids` (sha256 join keys) are **retained**: they carry no readable name, so the
  query-time expansion seam keeps working off content-addressed ids.

On the worked example, redacting `Ada Lovelace` out of the content yields an exported header of
`["Analytical Engine", "Charles Babbage"]` and exactly two surviving triples — every triple
naming her is gone, while `neighbor_node_ids` still lists `n-lon` and `n-ri`.

Two properties of this gate are easy to miss and both are deliberate:

1. **It does not follow the `unsafe_unmasked_corpus` opt-out.** The haystack stays the *masked*
   text (`export.py:879-893`). A benchmark export that materialises raw corpus text still emits
   the conservatively-gated header. Verified: the exported header is identical in both runs. The
   opt-out buys back benchmark-valid corpus text; it is not a licence to widen every other
   redaction surface, and gating against the masked text can only ever drop names, never add them.
2. **It runs only on rows that were actually redacted** (`masked_content is not None`). A row no
   redactor touched removed nothing, so its header is exported unchanged.

### What the KG export does *not* redact

Say it plainly, because it is the asymmetry a reviewer will find:

- `GraphHyperedge.tau_text` **is** redacted — enforced upstream at the writer
  ([ADR-0057](../adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)),
  so the span text in `graph-hyperedges.parquet` and the KV's `t|` keyspace is the corpus
  redaction variant;
- `GraphNode.canonical_name` **is not**. Under `extract_on: unmasked`, the KG's entity names are
  the real surfaces, and `graph_export.py` applies no redaction gate at all. They appear in
  `graph-nodes.parquet.canonical_name`, in `graph.ttl` as `rdfs:label`, in `graph.graphml` as
  `<data key="label">`, and — normalised — as the KV's `a|` alias keys.

That is not a bug; it is the documented consequence of the extraction posture, and the framework
gives you the lever. `extract_on: masked` redirects the extraction stages to read
`masked_content` instead, so PII never reaches extraction or the KG at all — and the pipeline
validator enforces that the extraction stages transitively depend on a redaction stage, so the
mode cannot be set without the ordering that makes it true
([ADR-0044](../adr/0044-universal-redaction-floor-and-optional-mask-before-extract.md)). The
trade is the obvious one: the KG loses the real entities. Pick deliberately; do not discover it
after the graph ships.

### Feeding an existing store rather than replacing one

The reason the deliverable is files rather than a service is that most organisations already own
the store. `corpus.parquet` is the whole ingestion contract:

- **the vector** is a native `list<double>` column, L2-normalised, of a declared and
  runtime-verified width — read it straight into the index;
- **the text** is one `string` column with a stated masking posture;
- **provenance travels with the row**: `source_uri`, `file_name`, `file_type`, `file_size`,
  `document_id`, and the page/char offsets, as JSON. Nothing downstream has to guess where a
  chunk came from;
- **classification travels with the row**: `language`, `category`, `sensitivity` — the fields a
  store's filters and a tenancy policy key on;
- **the lexical statistics are separate**, so a store drives its own BM25 with the same
  tokenisation the corpus was folded with;
- **the graph columns are denormalised** onto the row, so graph-aware rerank needs no join back
  to the KG at query time;
- and `schema_version` is on every row, so a consumer can tell what contract it is reading.

No component of this requires latence to be running. That is the point.

---

## 4.6 Reading it back

A minimal round-trip, using nothing but standard tools:

```python
import json, pyarrow.parquet as pq

t = pq.read_table("export/corpus.parquet")
row = t.to_pylist()[0]

vec  = row["embedding"]                       # list[float], unit-length, 768 wide
text = row["content"]                         # the corpus text, masking posture per §4.5
prov = json.loads(row["provenance"])          # document_id, file_name, char/page offsets
cls  = json.loads(row["classification"])      # language, category, sensitivity
hdr  = json.loads(row["context_header"] or "{}")   # redaction-gated
hops = row["neighbor_node_ids"]               # 1-hop keys into graph-nodes.parquet

stats = json.load(open("export/bm25-stats.json"))  # doc_count, avgdl, per-term df + idf
```

`hops` joins to `graph-nodes.parquet.node_id`; `row["kg_node_ids"]` joins the same way for the
chunk's own entities; `graph-edges.parquet.source_node_id`/`target_node_id` join both. The
per-edge `evidence` joins back to mention ids. The corpus and the graph are one artifact — the
chunks carry the graph's node ids and the graph's edges carry the chunks' evidence.

For anything beyond this, the exact per-file contract is
[the retrieval-engine input spec](../RETRIEVAL-ENGINE-INPUT-SPEC.md) — with the two corrections
flagged above (`neighbor_entity_ids` does not exist; the node table has `centrality`/`community`
columns when features ran).

---

**Handoff.** The corpus is now queryable, and by anything: a vector store reading `embedding`, a
lexical engine reading the BM25 artifacts, a graph engine reading the parquet/TTL/GraphML, a
lakehouse reading the Parquet directly, or an Iceberg catalog if one was configured. Chapter 5
(`05-retrieval.md`) takes the *shipped* retrieval library — stateless, holding no index, running
no first-stage search — and shows what it does with these files: how dense and BM25 fuse, how the
denormalised graph columns rescore that fusion, how graph traversal *appends* what the base
ranking missed rather than outvoting it, how a token budget is spent over the union, and what all
of that measured — including what the `dense_content_only` ablation says about the header this
chapter just built.

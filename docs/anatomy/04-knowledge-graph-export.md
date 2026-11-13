# 4 · The knowledge-graph export

*Chapter 3 finished with a graph in memory: canonical nodes and evidence-linked edges. This
chapter covers the one Stage that writes it to disk, and what that file set does and does not
redact. From here on there is no framework in the loop: what lands in `export/` is the whole
interface to everything downstream.*

The Stage is `export.knowledge_graph` (`packages/latence-core/src/latence_core/stages/graph_export.py`).
It hangs off the graph Stage alone and never sees a chunk. Its only configuration key is
`basename`, the file stem, which defaults to `graph`.

## 4.1 The file set

| File | Contents |
|---|---|
| `graph-nodes.parquet` | one row per `GraphNode` |
| `graph-edges.parquet` | one row per `GraphEdge` |
| `graph.ttl` | RDF Turtle, emitted by hand with no RDF library |
| `graph.graphml` | GraphML XML, which loads in yEd, Gephi and networkx |

Three properties hold for the set as a whole:

- **One publication.** All four files ride a single `Storage.atomic_writers` publication, so a
  consumer never sees a new `graph.ttl` beside a stale `graph-nodes.parquet`. A crash in the
  middle of an export leaves the previous complete set.
- **Deterministic bytes.** Confidences are formatted with six decimals in the Turtle and
  GraphML files, and every JSON column is written with sorted keys, so two runs over the same
  input produce the same files.
- **An empty graph is not an error.** It writes valid files with headers only.

Only `GraphRecord`s are written. Any other record kind in the input is ignored.

### `graph-nodes.parquet`

| Column | Type | Notes |
|---|---|---|
| `node_id` | string | |
| `entity_id` | string | |
| `type` | string | the entity type |
| `canonical_name` | string | the readable name; see §4.2 |
| `kb_id` | string | empty when the node is not linked to a knowledge base |
| `confidence` | double | |
| `member_mention_ids` | string | a JSON list |
| `source_document_ids` | string | a JSON list |
| `properties` | string | a JSON object |
| `evidence` | string | a JSON object: the node's `Evidence` |

The entity type column is named `type`, not `label`, so it carries the same name as GraphML's
`<data key="type">` and Turtle's `lg:entityType`.

### `graph-edges.parquet`

`edge_id`, `relation_id`, `source_node_id`, `target_node_id`, `label`, `confidence` (double),
`properties` (a JSON object) and `evidence` (a JSON object). Every edge carries `Evidence`: the
mention and document ids that justify it.

### `graph.ttl` and `graph.graphml`

Turtle writes each node as an `lg:Entity` under the non-dereferenceable base
`urn:latence:graph:`, with `rdfs:label` (the canonical name), `lg:entityType`, `lg:kbId` when
the node has one, and `lg:confidence`. Each edge is written twice: once as a direct triple, and
once as a reified `rdf:Statement` that carries `lg:confidence` and `lg:evidenceCount`.

GraphML writes a directed graph. A node carries `label` (the canonical name), `type`, `kb_id`
when present and `confidence`. An edge carries its label, its confidence and an evidence count.

Both writers strip the characters that XML 1.0 and strict Turtle forbid, because node names and
edge labels are spans cut out of source documents, and one stray control character would make
the file unreadable in every standard tool.

## 4.2 What the export does not redact

`GraphNode.canonical_name` is **not** redacted, and this Stage has no redaction gate. Under
`extract_on: unmasked`, which is the default, the graph's entity names are the real surface
strings, and they are written to four places:

| Surface | Where |
|---|---|
| the `canonical_name` column | `graph-nodes.parquet` |
| the `evidence` column, in its `snippet` | `graph-nodes.parquet`: Graph Assembly renders it as `node '<name>' (<type>) from N mention(s)` |
| `rdfs:label` | `graph.ttl` |
| `<data key="label">` | `graph.graphml` |

This is deliberate. A graph whose person nodes are all `[PERSON]` is a graph about nobody, and
resolution and disambiguation degrade with it. The Stage also could not apply a gate of its
own: it receives only graph records, with no chunk text and no PII spans to act on.

**The export warns about the surprising case.** When a redaction Stage is wired while
`extract_on` is `unmasked`, the corpus is masked and the graph is not. The Runner then logs one
`WARNING` per knowledge-graph export, before the first byte is written
(`runner.KG_ENTITY_NAMES_WARNING`), naming the surfaces above and the setting that changes it.
Both halves of the condition are required. With no redaction Stage, nothing is masked anywhere,
so there is no asymmetry to report. Under `extract_on: masked`, PII never reached extraction.

**The lever is `extract_on: masked`**, described in
[chapter 2](02-extraction-and-privacy.md#6--extract_on-unmasked--the-flag-that-decides-what-the-graph-knows).
It puts redaction in front of extraction, so no PII span text reaches a node label. No setting
on the export Stage changes any of this.

The behaviour is pinned by `packages/latence-core/tests/test_kg_export_name_posture.py`: the
warning fires exactly once in the surprising configuration, the real name is present in every
surface the warning names, and the same pipeline with `extract_on: masked` puts no PII span
text in any node label and warns about nothing.

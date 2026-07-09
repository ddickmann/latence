# Graph Assembly emits scoped GraphRecords over the ported builder; Export writes the KG as portable files (Parquet + TTL + GraphML) and the RAG corpus, never a live DB

S9 adds **Graph Assembly** (CONTEXT): the corpus-level Stage that builds the canonical
knowledge graph — nodes, edges, and per-edge **Evidence** — from the disambiguated canonical
entities and normalized relations, and the **Export** that materializes the two AI-ready
deliverables on Storage (ADR-0017). This is the slice that closes the first end-to-end
messy-docs → KG demo. It extends ADR-0004/0007 (the Capability + CPU-first-reference pattern)
and mirrors how ADR-0026/0025 handled the earlier corpus-level Stages, with the concrete
Graph contract, the ported builder asset, and the files-not-a-DB Export discipline.

## A GraphRecord is a per-scope inter-Stage Record, like the DisambiguationRecord

Graph Assembly is corpus-level and produces two kinds of output — **nodes** (one per
canonical entity) and **edges** (one per normalized relation). Rather than two record types,
a single `GraphRecord` carries a `GraphScope` tag (`NODE` / `EDGE`) and exactly one populated
payload (`GraphNode` / `GraphEdge`), exactly as `DisambiguationRecord` carries a
`DisambiguationScope` (ADR-0026) and `FeatureRecord` a `FeatureScope` (ADR-0025). It is a
versioned core contract (`SCHEMA_VERSION` → 9), a subclass of `Record`, so it carries
Provenance and Classification like every other inter-Stage carrier, threads through the DAG,
is checkpointed/resumed uniformly, and exports directly. The scope/payload consistency is
contract-enforced, so a Provider cannot emit a mis-scoped record.

**Every edge carries Evidence, reusing the S8 `Evidence` contract verbatim.** CONTEXT
separates Provenance (source lineage) from Evidence (justification for an inference). A
`GraphNode` and a `GraphEdge` each carry an `Evidence` back to the source mentions/documents
that justify them — the S9 "every edge carries Evidence (source docs/mentions/offsets) and
confidence" acceptance criterion — while their Provenance remains the page-accurate lineage
inherited from the source canonical entity / normalized relation. S8's ADR-0026 anticipated
this: "Graph Assembly (S9) reuses [Evidence] verbatim for per-edge justification."

## Deterministic, content-addressed IDs (the ported builder asset)

The named asset for this slice is the **Graph builder + EvidenceRecord provenance**
(`dataset-intelligence/graph/builder.py`, ASSET-INVENTORY.md, 5/5, **PORT-AS-IS**), re-homed
in the pure `latence_core.graph` engine. Its algorithmic value is preserved verbatim:

- **`make_graph_id`** — the ported `_make_id`: a node/edge id is a sha256 over a stable key
  (the corpus id + the entity/relation id), truncated to 16 hex chars. The same entity always
  maps to the same node, so a seeded run is byte-identical and two runs over the same corpus
  produce the same graph (the S9 "deterministic IDs" criterion, and the S11 affected-set key).
- **Endpoint validation** — an edge is emitted only when *both* endpoints resolved to a node
  (the ported `_build_edges` guard); a relation pointing at a missing or collapsed endpoint is
  dropped, never a dangling or self-loop edge.
- **Evidence links** — every node links back to its member mentions and every edge to the
  source relation's mentions/documents (the ported `_build_evidence` step), folded onto the
  contract's per-record `evidence` field, with the F8 `max_evidence_per_node` bound preserved
  so a hub entity does not produce an unbounded Evidence blob.

The private stack's `§19` predicted-vs-extracted split is preserved as an `is_predicted` edge
property: v1 has no link-prediction Stage (that is the post-v1 P3 plugin), so every v1 edge is
extracted (`is_predicted=False`), but the attribute is carried so a future predicted edge is
never silently mixed with an extracted one.

**Update (W2-linkpred, ADR-0037):** that "future predicted edge" now exists — the OPTIONAL
post-Assembly `GraphCompleter` Capability appends predicted EDGE-scope GraphRecords marked
`properties["inferred"] = True` (with `scorer`/`score`/`rank`/`calibrated`), counted separately in
the Quality Report and never merged into the asserted edges. The `is_predicted` attribute this ADR
carried is the seed of that split; ADR-0037 promotes it to its own Stage's output with the honest
inferred-edge contract (empty-mention Evidence, deterministic `edge_id` distinct from asserted).

## Export writes files, never a live database (ADR-0017), and serves BOTH RAG and graph-RAG

The Export Stage materializes the two AI-ready deliverables **as portable files on Storage,
never a live database** (ADR-0017): (1) the **RAG-ready corpus** — the cleaned,
Screened, PII-handled text with full Provenance/Classification — via the existing S1
`export.jsonl_parquet` Provider (Parquet + JSONL); and (2) the **knowledge graph** — nodes,
edges, per-edge Evidence — via a new `export.knowledge_graph` Provider as **Parquet + TTL +
GraphML**. Both are wired as ordinary Export Stages off their respective upstreams (the KG
export off Graph Assembly), so the one-command demo fans out to both deliverables from the one
converging DAG. **The corpus export is wired off a Redaction Stage by default** (both the S13
`demo-pipeline.yaml` and the `graph-slice.yaml` example route `export.jsonl_parquet` off
`redaction.hybrid_rule`), so the shipped corpus is genuinely PII-handled — the "Screened,
PII-handled" claim is honest by default, not aspirational (the reconciliation tracked in #37).
An operator who wants the raw, un-redacted chunk text drops the Redaction Stage and points the
corpus export back at Chunk. A downstream Neo4j / triple-store /
graph-RAG consumer loads the files with no framework code in the loop; writing directly into
an adopter's DB is rejected for v1 (ADR-0017) and is opt-in Provider work post-v1 behind the
same seam.

The TTL/GraphML are hand-emitted with strict IRI/literal/XML escaping and **no `rdflib` /
`networkx` dependency** (ADR-0016): the TTL is well-formed RDF Turtle (each edge is a triple
plus a reified statement carrying its confidence and Evidence count), and the GraphML is XML
that loads in yEd / Gephi / networkx. Every file is byte-stable (records are emitted in the
builder's node-id-then-edge-id order, every attribute in a fixed order), so a seeded run is
reproducible (Baseline bar).

## Embedding is an opt-in Embedder Provider, not required for Export

Per the S9 acceptance criterion and ADR-0017, embeddings are **optional**: the RAG corpus
exports its text with full Provenance/Classification whether or not an embedder is wired, so
any vector DB can ingest it. The `Embedder` Capability is a thin opt-in seam; the CPU
reference in core is a deterministic, dependency-free **hashing embedder** (no model, no
weights, no license to verify — ADR-0012/0016), so the seam is exercisable end to end offline.
The default learned Embedder (`multilingual-e5-small`, MIT) is a **pluggable Provider
package**, not a core dependency — the framework names the seam, not the model.

**How it is wired (the #37 seam completion).** Embedding is not a standalone DAG node — it is an
augmentation of the corpus **Export** Stage. A corpus Export names its Embedder Provider in its
own `config['embedder']` (a `{"provider": ..., "config": {...}}` block); the corpus Export
resolves that Provider from the `latence.providers` registry, checks it against the `Embedder`
seam, and threads each exported record's corpus text (a RedactionRecord's PII-handled
`masked_content`, else the `content`) through it, attaching a fixed-width `embedding` vector to
every row — a JSONL key and a native Parquet `list<double>` column. An Export with no `embedder`
attaches nothing (the text is still exported, so a vector DB can embed later). The
`CapabilityKind.EMBEDDING` value names the seam so a Provider registers + is discovered under it;
declaring a *bare* `embedding` DAG Stage is a wiring error the Runner rejects with a pointer to
the Export-config form. The learned served model is the pluggable `embedding.endpoint` package,
swapped in by pointing `config['embedder']['provider']` at it — one config line, no code change.

## Providers shipped (CPU-first reference, ADR-0007)

* **`graph.canonical`** ships in `latence-core`: the ported deterministic builder behind the
  `GraphAssembler` seam — content-addressed nodes/edges, endpoint validation, per-node/-edge
  Evidence — all pure-Python, no extra deps.
* **`export.knowledge_graph`** ships in `latence-core`: the KG Export as Parquet + TTL +
  GraphML, hand-emitted with no RDF/graph library.
* **`embedding.hashing`** ships in `latence-core`: the deterministic opt-in reference Embedder
  (the learned e5-small is a pluggable package).

The KG statistics land in the Quality Report as `GraphQuality`: node/edge counts, nodes by
type, edges by label, KB-linked node count, and **evidence coverage** (the fraction of edges
that carry Evidence — 1.0 means every edge is justified) — counts only (no raw document text)
— satisfying the S9 "KG statistics (node/edge counts, evidence coverage) land in the Quality
Report" criterion.

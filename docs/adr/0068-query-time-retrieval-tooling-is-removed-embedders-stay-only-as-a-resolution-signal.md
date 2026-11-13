# Query-time retrieval tooling is removed; embedders stay only as a resolution signal

**Status:** Accepted (2026-11-08)

The retrieval layer (ADR-0048–0053), hyperedges and their evidence sidecar (ADR-0056–0059) and
Context Enrichment (ADR-0039) leave the product. Embedder Providers stay, but only as a signal
the resolution phase uses: the embedding rung of the resolver, the semantic rung of type
consolidation, and the STS merge in `disambiguation.glinker`. They are no longer an index-time
retrieval output. The owner decided this on 2026-11-08, together with the re-scope in ADR-0067.

## Context

ADR-0067 re-scopes Latence into a semantic-resolution and ontology engine. Retrieval was a
downstream consumer layer: signal generators, candidate processors, the retrieval MCP server,
hyperedges packed into a token budget, and Context Enrichment that existed to feed the
embedding input. None of it builds or governs a semantic layer. Keeping it means carrying
about 15,000 lines of package code plus core hooks, contracts, stacks and tests that the new
product does not use. It would also blur what the product is.

The carve-out was inventoried on 2026-11-08 (map ticket *Carve-out inventory*). No kept package
imports a retrieval package. In core, retrieval code is reached at about five import sites and
through optional config branches:

- `stages/graph_assembly.py` (hyperedges);
- `stages/graph_export.py` (the hyperedge KV);
- `stages/export.py` (BM25 accumulator, context headers, redaction survival);
- `cli.py` (the `retrieval` command group, the SPLADE index);
- `conformance/cases.py` (the sparse hashing embedder).

Further hooks:

- capability protocols for sparse, multi-vector, FDE, tokenizer and context enrichment;
- the hyperedge contracts and `GraphScope.HYPEREDGE`;
- hyperedge, token-cost and enrichment sections in the Quality Report.

## Decision

1. **Removed:**
   - packages: `latence-retrieval`, `latence-sink-qdrant`, `latence-splade`, `latence-muvera`;
   - core modules: `hyperedge.py`, `hyperedge_store.py`, `sentences.py`, `bm25.py`,
     `signals.py`, `splade_index.py`, `stages/context_enrichment.py`;
   - the retrieval Capability protocols and descriptors;
   - the hyperedge contracts;
   - the retrieval columns in the corpus export;
   - the retrieval benchmark lanes;
   - every stack that exists only for retrieval.
2. **Kept:**
   - `latence-embedder-st` and `latence-embedder-endpoint`, and `embedding.hashing` as the
     offline fallback, used only as resolution signals;
   - the corpus export, as the record of what text was processed.
3. **Removal is one build slice** (map ticket *Remove the retrieval tooling*). It has a
   `SCHEMA_VERSION` bump and a green gate, and it removes code, tests, stacks and docs together.
   No dead seam is left behind. The inventory ticket fixes the exact file list. Removing
   `graph_features` (PageRank and communities, whose only consumers were retrieval expansion
   keys) is decided there too.
4. **Superseded:** ADR-0039, ADR-0048, ADR-0049, ADR-0050, ADR-0051, ADR-0052, ADR-0053,
   ADR-0056, ADR-0057, ADR-0058 and ADR-0059. They stay in the log as the dated record. Each
   carries a superseded notice.
5. **ADR-0045 is narrowed.** Granite r2 stays the default learned Embedder, but as a resolution
   signal. ADR-0046's finding that retrieval embedders over-merge short names stays binding.

## Alternatives rejected

- **Keep retrieval as an optional extra.** Rejected: an unused optional layer still costs gates,
  dependency pins and contract surface, and it keeps the product's identity split.
- **Freeze retrieval in place and stop maintaining it.** Rejected: frozen code inside a gated
  monorepo still breaks on every contract change, and it would block the `SCHEMA_VERSION` bump
  the re-scope needs.
- **Move retrieval to its own repository now.** Not needed: Metis owns retrieval for its agents.
  Git history keeps the code if it is ever wanted again.

## Consequences

- The retrieval documentation (`docs/retrieval-tooling.md`, `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`,
  the retrieval tutorials) describes code that is about to go. It is removed in the same slice.
- Graph Assembly emits nodes and edges only. The m-ary context a hyperedge carried is not lost
  as Evidence, because edges keep their mention-level Evidence.
- Consumers that loaded the Qdrant sink or the retrieval MCP must use their own retrieval
  engine, or Metis.

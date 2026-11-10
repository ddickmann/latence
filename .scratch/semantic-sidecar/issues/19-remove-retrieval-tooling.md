# Build: remove the retrieval tooling

Type: task
Mode: AFK
Status: resolved
Blocked by: 07

## Question

Build slice (graduated at charting because its decision, ADR-0068, is already made).

## Build

Remove everything ADR-0068 and *Inventory the carve-out* list:

- the packages;
- the core modules;
- the capability protocols and descriptors;
- the hyperedge contracts (bump `SCHEMA_VERSION`);
- the export columns;
- the CLI group;
- the Quality Report sections;
- the stacks, benchmark lanes, tests and docs.

Keep the embedders as resolution signals.

## Done when

- `scripts/verify-local.sh` is green;
- no import, entry point, stack or doc references a removed module;
- the doc-truth tests pass with corrected counts;
- `uv lock` is clean;
- a seeded text-path run produces the same nodes and edges as before the removal (only
  hyperedges gone).

## Answer

Resolved 2026-11-10. The removal is merged into the working branch.

**What left the tree**

- Packages: `latence-retrieval`, `latence-sink-qdrant`, `latence-splade`, `latence-muvera`
  (34 → 30 packages).
- Core: hyperedges, Context Enrichment, graph features, the BM25 / sparse / multi-vector / FDE
  export signals, redaction survival, the SPLADE index driver, and the Capability protocols,
  descriptors, conformance cases, CLI group and setup-recipe parts that existed only for them.
- Contracts: `SCHEMA_VERSION` 21 → 22 and `QUALITY_SCHEMA_VERSION` → 21.
- Stacks: `context-enrichment`, `graph-features`, `retrieval-signals`, and the Qdrant compose
  file. Benchmarks: `benchmark/s5`, `benchmark/serving` and the campaign's `s5-retrieval` stage.
- Docs: the retrieval tutorials, guides, evidence pages, the deep dive and the anatomy chapters
  on enrichment and retrieval.
- In total 457 files changed and about 78,000 lines removed. No ADR was deleted or rewritten.
  The embedder packages and `embedding.py` stay, as a resolution signal.

**Checked against "Done when"**

| Item | Result |
|---|---|
| `scripts/verify-local.sh` green | Every lane after the secret scan passed on the removal tree: workspace sync, editable installs, ruff, strict mypy, the full suite (4,332 passed, 17 skipped), Provider conformance, stack validation, environment specs, the compatibility matrix and the bake-off matrices. **The secret-scan lane fails locally** on a fixture string in `test_redaction_recall_benchmark.py` that is not in the audited baseline. It fails the same way without the removal, the file is unchanged from `main`, and CI's own secret-scan lane passes; the cause of the local difference is not known. |
| No import, entry point or stack references a removed module | Met. The only code mentions left are dated schema-history comments. |
| **No doc references a removed module** | **Not met; moved by decision.** Ten pages still describe removed features under a dated warning that names what no longer applies: `docs/BENCHMARKS-E2E.md`, `docs/deployment.md`, `docs/anatomy/index.md`, `docs/tutorials/01`, `02`, `04` and `05`, `docs/guides/benchmark-campaign.md`, `docs/guides/incremental-ingest.md` and `docs/evidence/enterprise-serving-03-incremental-ingest.md`. Their rewrite joins the docs-and-site rewrite in the map. |
| Doc-truth tests pass with corrected counts | Met (54 passed). A new test checks that documented lint commands name paths that exist; it was red on two removed directories first. |
| `uv lock` clean | Met (`uv lock --check` exits 0). |
| Same nodes and edges as before | Met. Before and after runs of the demo, the example pipelines and every kept stack give equal node and edge records, apart from the schema version (36 node rows and 176 edge rows in the Parquet exports). |

**Not run here:** the MinIO `s3://` lane (the image pull is refused in this environment and in
CI), the Airflow DAG lane, GPU runs, and the strict docs build to completion (its only warning
is a diagram script host this environment's network blocks).

**Kept on purpose.** The removed export chapter also documented behaviour that stays, so a
shorter chapter replaces it: `docs/anatomy/04-knowledge-graph-export.md` covers the four files
the graph export writes, their columns, and that it does not redact entity names.

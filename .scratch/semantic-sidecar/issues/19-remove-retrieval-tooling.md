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
| `scripts/verify-local.sh` green | Every lane after the secret scan passed on the removal tree: workspace sync, editable installs, ruff, strict mypy, the full suite (4,332 passed, 17 skipped), Provider conformance, stack validation, environment specs, the compatibility matrix and the bake-off matrices. **The secret-scan lane fails locally** on a fixture string in `test_redaction_recall_benchmark.py` that is not in the audited baseline. It fails the same way without the removal, the file is unchanged from `main`, and CI's own secret-scan lane passes. **Cause (found 2026-11-10):** `detect-secrets` verifies some key shapes against the vendor's API. For this Stripe-shaped fixture it calls `api.stripe.com`; where that host is reachable the key is reported invalid and the finding is dropped, and where it is blocked, as here, the finding stays unverified and is kept. The gate's result therefore depends on the network. Scanning with verification off and auditing the fixture into the baseline would make it deterministic; that is a change to a security gate and is the owner's call. |
| No import, entry point or stack references a removed module | Met for imports, entry points and stacks. **Not complete in behaviour:** the benchmark campaign's reuse check and the labelling tool's hyperedge mode still serve removed features, and two help texts are stale. See *What the retrieval removal left behind*. |
| **No doc references a removed module** | **Mostly met.** The ten pages that described removed features were rewritten against the current code on 2026-11-10; measurements from the earlier pipeline stay only as dated records that name what that pipeline contained. Still open and moved to *What the retrieval removal left behind*: the release gates, `ROADMAP.md`, and anatomy chapters 2 and 3. |
| Doc-truth tests pass with corrected counts | Met (54 passed). A new test checks that documented lint commands name paths that exist; it was red on two removed directories first. |
| `uv lock` clean | Met (`uv lock --check` exits 0). |
| Same nodes and edges as before | Met. Before and after runs of the demo, the example pipelines and every kept stack give equal node and edge records, apart from the schema version (36 node rows and 176 edge rows in the Parquet exports). |

**Not run here:** the MinIO `s3://` lane (the image pull is refused in this environment and in
CI), the Airflow DAG lane, GPU runs, and the strict docs build to completion (its only warning
is a diagram script host this environment's network blocks).

**Kept on purpose.** The removed export chapter also documented behaviour that stays, so a
shorter chapter replaces it: `docs/anatomy/04-knowledge-graph-export.md` covers the four files
the graph export writes, their columns, and that it does not redact entity names.

**Follow-up.** [What the retrieval removal left behind](23-retrieval-removal-leftovers.md)
lists what this removal missed and the unrelated defects the page rewrite found.

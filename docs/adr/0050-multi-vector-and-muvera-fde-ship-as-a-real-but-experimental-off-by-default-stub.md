# Multi-vector / MUVERA-FDE ships as a real-but-experimental, off-by-default stub behind the FdeConverter seam

The multi-vector path — raw ColBERT-style token vectors, their MUVERA fixed-dimensional-encoding (FDE)
conversion to a single dense vector, and MaxSim reranking over them — ships in v1 as a **real,
training-free reference that is experimental, off by default, and not blessed for production**, behind
a stable seam. What is *deferred* is not the code but the *production converter decision*: which cheap,
reliable multi-vector→dense method to bless (canonical MUVERA FDE vs a regularizer-refined variant) is
an open research question the maintainer is still resolving.

## Context

MUVERA FDE is a published, **data-oblivious, training-free** algorithm: random space-partitioning
buckets ColBERT token vectors and sums per bucket into one high-dimensional dense vector whose
dot-product approximates the Chamfer/MaxSim similarity — so it slots into a customer's ordinary **dense**
ANN index with no special store. A higher-quality alternative (LightOn's hierarchical-pooling
*regularizer*, which trains through hard cluster assignment with a straight-through estimator for ~5×
lossless compression) exists but requires training the maintainer explicitly wants to avoid for now.
Whether FDE-alone is reliable enough, or the regularizer route is worth its cost, is undecided.

The framework's anti-false-green rule forbids a "stub" that returns plausible-looking vectors while
pretending to be FDE. A stub must be **either real or honestly absent**.

## Decision

- **Emission contract (index-time, files, ADR-0017/0049).** Dense → `embedding` column (exists). SPLADE
  sparse → parallel `sparse_indices: list<int>` + `sparse_values: list<float>` columns (the shape every
  vector DB expects; ADR-0053). **MUVERA FDE → `fde_embedding: list<double>` column** — deliberately
  shaped to land in the customer's *dense* index. **Raw multi-vectors → a sidecar
  `multivectors.parquet`** (`list<list<double>>` keyed by `record_id`), *not* in `records.parquet` (too
  heavy, different access pattern). The "EMBEDDING GENERATION" options map to one toggle: **FDE-only**
  (emit `fde_embedding`) vs **FDE + keep multi-vectors** (also emit the sidecar for later MaxSim rerank).
- **Stub interpretation = real-but-experimental, off by default.** Ship the seams (`MultiVectorEmbedder`,
  `FdeConverter`, and the MaxSim reranker) fully typed/wired/tested, *and* a **faithful, canonical,
  data-oblivious MUVERA FDE reference** (no training) and a reference MaxSim scorer — marked
  **experimental, off by default, quality-uncaveated on any given corpus**. Rejected the alternative of
  an interface-only `NotImplementedError` stub: it leaves the multi-vector path un-evaluable, which
  defeats the point of being able to measure FDE-vs-regularizer later.
- **Deferred:** the LightOn-regularizer-refined converter; any bake-off that *blesses* a production
  default; turning multi-vector on by default; and the SOTA MaxSim kernels (ADR-0049 defers these to a
  relicensed accelerator — the FP8 Triton kernel adapted from unsloth needs its own license clearance).

## Consequences

- v1 ships a runnable, evaluable multi-vector path with **zero fake code** — honest per the
  anti-false-green rule (experimental-but-real, not a passthrough that lies).
- One honest downstream dependency: the Knapsack packer's `coverage_matrix` input is the per-query-token
  MaxSim coverage, which rides this deferred path — so the packer runs on every *other* signal in v1 and
  **gains the coverage term when multi-vector goes real** (ADR-0049).
- The seam is stable, so the future production converter drops in behind `FdeConverter` without a
  contract change.

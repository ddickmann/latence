# latence-muvera

The framework's **canonical, training-free MUVERA FDE** `FdeConverter` Provider (#163,
ADR-0050/0049/0012/0016). It is the same index-time `FdeConverter` Capability as the in-core
deterministic `fde.reference` — swap one for the other with **one config line** — but its dense vector
is a genuine MUVERA Fixed-Dimensional Encoding whose **dot-product approximates the Chamfer / MaxSim
similarity**, so ColBERT-style late-interaction quality can be approximated inside a customer's
ordinary **dense** ANN index with no special multi-vector store. It emits the `fde_embedding` column
via #160's Export seam.

## What "canonical MUVERA" means (vs the reference)

MUVERA (Dai et al., *Multi-Vector Retrieval via Fixed Dimensional Encodings*, 2024) is a
**data-oblivious, training-free** reduction. `fde.muvera` implements the real construction the
reference deliberately omits:

| Step | `fde.muvera` (canonical) | `fde.reference` (in-core) |
|---|---|---|
| Space partition | `k_sim = log2(buckets)` fixed **Gaussian** SimHash hyperplanes | fixed SimHash hyperplanes |
| Document cell aggregate | **average** (cell centroid) | sum |
| Empty document cell | **filled** from the nearest non-empty cell (Hamming distance) | left zero |
| Query cell aggregate | **sum** | (n/a — symmetric) |
| Repetitions | `repetitions` independent partitions concatenated (variance reduction) | 1 |
| MaxSim guarantee | **yes** — `⟨FDE_q, FDE_d⟩ ≈ Σ_q max_d ⟨q,d⟩` | none (honest but quality-uncaveated) |

The **query = sum / document = average-and-fill** asymmetry is the crux: for each query token `q`,
`⟨FDE_q, FDE_d⟩` contributes `⟨q, centroid of q's cell in D⟩ ≈ max_d ⟨q,d⟩`, so the single dense
dot-product estimates the late-interaction score. `repetitions` averages independent estimates to cut
variance. `repetitions = 1` (the default) keeps the output width a drop-in
`buckets × input_dimension` match for the reference.

## Two clocks, one construction

- **Index time** — `convert(multivectors)` (the `FdeConverter` Protocol method #160's Export calls)
  produces the **document** FDE that becomes the `fde_embedding` column, destined for the customer's
  dense index.
- **Query time** — `encode_query(multivectors)` produces the **query** FDE from the identical
  partition, for the dense lookup against that column. One converter, two clocks (mirroring how
  `sparse.splade` composes both index- and query-time sparse vectors).

## Usage

```yaml
# In a corpus Export (ADR-0017 opt-in), experimental / off by default (ADR-0050):
config:
  multivector_embedder:                 # the ColBERT-style token vectors to reduce
    provider: multivector.hashing       # in-core reference; a real ColBERT encoder is a package
    config: { dimension: 128 }
  fde_converter:
    provider: fde.muvera
    config:
      input_dimension: 128              # REQUIRED — must equal the multi-vector's token width
      buckets: 16                       # power of two; more cells → tighter approximation, wider vector
      repetitions: 4                    # independent partitions concatenated to reduce variance
      fill_empty: true                  # the MUVERA fill-empty-clusters step (documents)
      seed: 0
  keep_multivectors: false              # FDE-only; set true to also persist the raw sidecar for exact MaxSim rerank
```

The FDE column (and the raw-multivector sidecar) are **off by default and experimental** (ADR-0050);
`fde_converter` requires `multivector_embedder`, and the `keep_multivectors` toggle selects **FDE-only**
vs **FDE + kept multi-vectors** (the sidecar for a later exact-MaxSim rerank).

## No model, no weights — training-free (ADR-0050/0012)

MUVERA FDE is **data-oblivious and training-free**: it ships no model and no weights, so there is no
separate weights license to defer (unlike `sparse.splade` or a real ColBERT multi-vector encoder). The
`ProviderProfile` declares `model_id=None` and a **verified** Apache-2.0 (the package's own code, this
repo's `LICENSE`) — an honest, non-fabricated attestation, because there genuinely is nothing else to
verify. The deferred, higher-quality **LightOn-regularizer** converter (which *does* train) is out of
scope (ADR-0050); so is any bake-off that *blesses* a production FDE default.

## Offline / CI discipline (anti-false-green)

Pure-Python, dependency-free, deterministic (seeded Gaussian hyperplanes, rounded sums → a seeded run
is byte-reproducible). Installed in the CPU-first offline dev env, so the **real** converter is
exercised by the conformance suite and by a **MaxSim-approximation** test on synthetic multi-vectors —
not a shape check: it asserts the FDE dot-product tracks true MaxSim as a document's overlap with the
query grows. A wrong-width token vector is **rejected** with a typed error (the faithful contract a
real converter enforces). Real-model validation (a licensed ColBERT encoder feeding the converter) is
done out of band.

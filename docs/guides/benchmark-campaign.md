# The benchmark campaign wizard

!!! warning "Written before the retrieval tooling was removed"
    This page predates [ADR-0068](../adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md), which removed the query-time retrieval
    tooling together with Context Enrichment (`enrich`), graph features, hyperedges, the
    BM25 / sparse / multi-vector export signals and the Qdrant sink. The campaign's `retrieval:` and `s5:` blocks and its `s5-retrieval` stage were removed; the campaign now ends at `s3-pipeline`.
    The page is kept until it is rewritten.

`latence-campaign` drives the framework's end-to-end SOTA benchmark campaign — acquire real
public corpora, OCR them, push them through the full enterprise pipeline, then measure
retrieval — as **gated, resumable stages** with filesystem done-probes and an atomic
`STATE.json`. It is how the numbers on the [front page](../index.md) were produced, and it is
deliberately drivable from a clone with the framework installed and nothing else (stdlib
`argparse`, no extra deps).

```text
latence-campaign init      # the wizard: answers -> a validated campaign.yaml
latence-campaign status    # the stage table + probe results
latence-campaign run <stage>   # runs ONE stage (refused unless the previous one is approved)
latence-campaign approve <stage>  # operator sign-off on a done stage (opens the next gate)
latence-campaign resume    # continue an interrupted campaign (filesystem probes, not exit codes)
```

## The wizard contract

`init` follows the `latence setup` contract: **flag > prompt > default**, a "why it matters"
line per question, one frozen answers struct from either path, generate → dry-validate →
write, and exactly one next command printed. Identical answers produce a **byte-identical**
`campaign.yaml` — the config is an artifact, not a session. `--non-interactive` takes every
default/flag for CI use. Exit codes: `2` usage/contradiction, `1` invalid artifact or a
blocked/failed stage, `0` success.

The prompts cover: the campaign root, the dataset selection (from the known-dataset registry),
the pod SSH target for GPU stages, the S3 induction mode, and the two headline retrieval
choices (policy-gate mode, selector) — the full `retrieval:` block with measured defaults is
documented in the graph-rescue retrieval guide (removed with the tooling).
The `s5:` matrix block is flag-only (`--s5-datasets`, `--s5-legs`, `--s5-limit`, `--s5-fusion`):
its default *is* the full matrix over the corpus selection, so every deviation is a deliberate
narrowing rather than an answer worth interrogating up front.

## The stage chain

Order **is** the gate chain — each stage refuses to run until the previous one is approved,
and `approve` is an explicit operator sign-off, never automatic:

| Stage | What it does | Gate artifact |
| --- | --- | --- |
| `s1-corpus` | Acquires the selected public datasets under their **machine-enforced license lanes** (a research-only corpus requires `--accept-non-commercial`; an unverifiable one is refused unconditionally) | acquisition manifest + per-dataset counts |
| `s2-ocr` | OCRs the scanned corpora into markdown (served LightOn-OCR-2), page maps carried in sidecars | OCR metrics + spot-check dir |
| `s3-pipeline` | Runs the full enterprise pipeline per dataset on the pod (extraction, redaction, KG, hyperedges, embeddings, exports) | per-dataset `metrics.json` — the campaign scale numbers |
| `s5-retrieval` | Runs the retrieval feature matrix — every (dataset × leg) cell from `dense` through `+bm25 / +KG / +hyperedges / +CoE / +packer`, the content-only ablation and the three graph-rescue legs | one JSON cell per (dataset, leg) under `stages/s5-retrieval/results/`, plus `REPORT.md` + `metrics.json` |

The matrix is configured by the `s5:` block (`datasets`, `legs`, `limit`, `fusion`) and, when
`s5.datasets` is left empty, follows the campaign's own `corpus.datasets` — so trimming the
corpus trims the matrix with it. The stage derives its plan from that block and calls the
harness's API (`benchmark.s5.run_matrix.run`) directly; the same harness is still drivable
standalone as `python -m benchmark.s5.run_matrix`, and a cell produced either way is the same
file. Resumability is per cell: a matrix that takes hours can be interrupted and re-entered, and
`run` never re-runs a cell it already has.

Two refusals are load-bearing. A leg whose export artifacts are absent is recorded as
`missing-inputs` with the file list, never as a zero — the reused MuSiQue/HotpotQA exports
genuinely carry no embedding column, and the matrix says so. But a *dataset* whose every leg
refused blocks the stage: that is an absent or unfinished export being advanced as a
measurement. Gold resolution is audited per dataset before any recall number is trusted, and
single-hop control datasets are reported as controls, not headlines.

### The run has to be pinned, twice over

The harness refuses to start unless **`PYTHONHASHSEED=0`** *and* **`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` are all `1`**:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmark.s5.run_matrix --datasets all --legs all
```

The hash seed is the obvious one — the hypergraph entry is hash-order sensitive. The thread pin
is the one that costs people a day. The dense leg is a brute-force cosine (a BLAS `gemv`) and
the query embedder is a torch forward pass; both reassociate their floating-point sums
differently at different thread counts, and that is enough to change *which documents land in
the pool*. Measured on this corpus: running musique/bm25 at 4 threads instead of 1, with
identical code, gives 33 of 1,209 queries a different dense pool and one query a different
gold-hit count — moving `recall@10` from 0.3404 to 0.3406. An unpinned harness reports thread
scheduling as a quality change. Because the campaign stage calls the harness in-process, the
pins have to be in the environment you launch `latence-campaign` from.

Every dataset's corpus/toolkit licenses and lanes are recorded in one place —
[`THIRD-PARTY-LICENSES.md`](https://github.com/ddickmann/latence/blob/main/THIRD-PARTY-LICENSES.md)
— and corpus-derived artifacts (raw acquisitions, OCR output, exports, reports) stay local by
`.gitignore` discipline; only aggregate metrics and stage reports are committed.

## Interruption is a state, not a failure

Stages checkpoint through the same WAL/checkpoint discipline as the pipeline Runner; `resume`
re-probes the filesystem (never trusts a recorded exit code) and relaunches only what is not
provably done. A blocked stage records its reason in `STATE.json` and `status` renders it —
fix the cause, `resume` under the same run id, finalized checkpoints replay from disk.

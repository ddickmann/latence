# The benchmark campaign wizard

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
documented in [Graph-rescue retrieval](graph-rescue-retrieval.md#configuring-it-from-the-campaign-wizard).

## The stage chain

Order **is** the gate chain — each stage refuses to run until the previous one is approved,
and `approve` is an explicit operator sign-off, never automatic:

| Stage | What it does | Gate artifact |
| --- | --- | --- |
| `s1-corpus` | Acquires the selected public datasets under their **machine-enforced license lanes** (a research-only corpus requires `--accept-non-commercial`; an unverifiable one is refused unconditionally) | acquisition manifest + per-dataset counts |
| `s2-ocr` | OCRs the scanned corpora into markdown (served LightOn-OCR-2), page maps carried in sidecars | OCR metrics + spot-check dir |
| `s3-pipeline` | Runs the full enterprise pipeline per dataset on the pod (extraction, redaction, KG, hyperedges, embeddings, exports) | per-dataset `metrics.json` — the campaign scale numbers |

Retrieval measurement (the S5 feature matrix and the graph-rescue cells) runs from the same
config through the harness under `benchmark/s5/` (`run_matrix.py`; results land under
`benchmark/sota-campaign/stages/s5-retrieval/`). Gold resolution is audited per dataset before
any recall number is trusted, and single-hop control datasets are reported as controls, not
headlines.

Every dataset's corpus/toolkit licenses and lanes are recorded in one place —
[`THIRD-PARTY-LICENSES.md`](https://github.com/ddickmann/latence-framework/blob/main/THIRD-PARTY-LICENSES.md)
— and corpus-derived artifacts (raw acquisitions, OCR output, exports, reports) stay local by
`.gitignore` discipline; only aggregate metrics and stage reports are committed.

## Interruption is a state, not a failure

Stages checkpoint through the same WAL/checkpoint discipline as the pipeline Runner; `resume`
re-probes the filesystem (never trusts a recorded exit code) and relaunches only what is not
provably done. A blocked stage records its reason in `STATE.json` and `status` renders it —
fix the cause, `resume` under the same run id, finalized checkpoints replay from disk.

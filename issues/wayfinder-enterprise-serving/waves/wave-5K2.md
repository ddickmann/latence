# Wave 5-K part 2 — S5 becomes a real campaign stage (`run(plan)` + StageSpec + wizard `s5:`)

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 5-K), campaign-exploration
candidate C2. Part 1 (`1f9386e`) gave the `retrieval:` block its consumer; this is items 1, 2, 4
of the binding contract plus the committed-config regeneration. Two commits, in order — the API
first, the stage on top of it:

| commit | what |
|---|---|
| `c0edd79` | `run(plan) -> MatrixOutcome` is the harness API; `datasets.py` takes a `CorpusLayout` |
| `5d65349` | `s5-retrieval` StageSpec + wizard `s5:` block + the regenerated `campaign.yaml` |

## The friction, verified on the current tree

The S5 harness already honoured every rule this campaign is built on — per-cell resumability,
refuse-don't-fake on absent inputs, `REPORT.md` + `metrics.json` as gate artifacts. It honoured
them through machinery a campaign could not call:

- no `StageSpec` in `campaign/stages.py::registry()` (three stages), so no done-probe, no
  progress-probe, and no gate on the s3-pipeline approval;
- `run_matrix` was argparse-only with `DEFAULT_OUT` hardcoded — a campaign root could not
  redirect the output, and `main()` was the only entry;
- `datasets.py` held `REPO_ROOT` / `SOTA_CORPUS` / `S3_EXPORTS` / `S2_OUTPUT` / `OFFICIAL_DATA`
  as module constants, so where the harness read its inputs was a property of where its source
  file sat, not of what it was asked to measure;
- `docs/guides/benchmark-campaign.md` claimed S5 "runs from the same config" while `run_matrix`
  never opened `campaign.yaml`. That sentence is now true rather than deleted.

## 1. The API (`c0edd79`)

`run(plan) -> MatrixOutcome` is the entry; `main` is one adapter (flags → `Plan` → `run` → one
line and an exit code). The properties that had to move WITH it:

| moved to the API entry | why |
|---|---|
| `require_deterministic()` (the `PYTHONHASHSEED=0` refusal) | a caller that never touches argparse must still be unable to write a non-reproducible cell. The CLI's stderr line and exit code are byte-unchanged — the message is now one constant, `HASHSEED_REFUSAL`, asserted by the API and printed by the CLI |
| `SAMPLE_SEED` + `sample()` | already past `parse_args`; unchanged, now pinned by a test |
| `planned_cells(plan)` | the done-probe's definition of "finished", derived FROM the plan, so the stage never restates the cell naming beside the writer |

`MatrixOutcome` reports `cells` (planned cells present afterwards — the resume view), `written`
(this invocation only), `report`, `metrics`. `run_dataset` returns the cells it wrote, refusals
included, so a resumed run reports what it actually did instead of re-deriving it from mtimes.

**`CorpusLayout`** replaces the four constants: two roots (`sota_corpus`, `official_data`), the
rest derived (`s3_exports`, `s2_output`). `DEFAULT_LAYOUT = CorpusLayout.under(REPO_ROOT)` is
this checkout and the module constants are now *derived from it*, so every existing call site is
byte-identical. `dataset_specs(layout)` binds each loader to the SAME layout with `partial`, so a
spec's queries and its export can never come from different roots — which is exactly the silent
failure a caller redirecting only *some* of the constants would have produced. `DatasetSpec` lost
its `s2_dataset` + derived `s2_root` property in favour of a resolved `s2_root` field;
`CONTROL_DATASETS` stays derived from the table's own `single_hop` flag.

**One relocation:** `RESULTS_DIRS` and a new `results_dir(out, fusion)` moved from `run_matrix`
to `cell.py`, beside `cell_path`. Where a cell lives and what it is called are one fact, and the
campaign's done-probe needs that fact without importing numpy / pyarrow / latence-retrieval —
`cell.py` is stdlib-only, which is what makes a cheap, drift-proof probe possible at all.

## 2. The stage (`5d65349`)

`campaign/retrieval_stage.py`, registry position 4 — so `s5-retrieval` refuses until
`s3-pipeline` is **approved**, by the existing gate chain, with no new gating code.

### The StageSpec

```
name          s5-retrieval
title         Retrieval feature matrix (dense → +bm25 → +KG → +hyperedges → +CoE → +packer)
done_probe    every planned cell file present  AND  REPORT.md  AND  metrics.json
progress_probe "<done>/<total> matrix cell(s) written"
run           request_for(ctx) → MatrixRunner → the no-fake-green floor → one operator line
```

The contract asked for "every planned cell file present". The two gate artifacts are added
deliberately and stated here: `approve` is an approval **of** `REPORT.md` + `metrics.json`, and
the harness writes them last — so requiring them is also what distinguishes a completed run from
a cell directory filled in by hand. Both probes ask `benchmark.s5.cell` for the file names, so
probe and writer cannot disagree; an unimportable harness makes the probe unsatisfied and the
run block loudly, never quietly pass.

### What the stage owns (it is not a pass-through)

- **Plan derivation.** `request_for(ctx)` resolves the `s5:` block: an empty `s5.datasets`
  follows the campaign's own `corpus.datasets` (trim the corpus, trim the matrix — the two can
  never silently disagree); the legs are canonicalised into matrix order, because that order is
  the report's column order and not something an operator's typing should change; `out` is
  `stage_dir(campaign_root, "s5-retrieval")`.
- **The layout redirect.** The `Plan` is built with `CorpusLayout.under(ctx.repo_root)` — the
  checkout the operator pointed the campaign at, which is the whole point of item 2.
- **The no-fake-green floor.** A per-leg `missing-inputs` stays legitimate and is left alone (the
  reused M6 exports genuinely carry no `embedding` column, and the matrix says so). A **dataset**
  whose every leg refused blocks: that is an absent or unfinished S3 export being advanced as a
  measurement — the retrieval twin of the empty quality report `s3` blocks on.
- **`force=False`, always.** The campaign never re-runs a cell it already has; that is what makes
  a multi-hour matrix resumable across sessions, and `--fresh` is the operator's override.

### The import seam

`benchmark/s5` is repo code, not a dependency of `latence-benchmark` — it pulls numpy, pyarrow
and latence-retrieval, none of which the campaign package imports. So `harness_module(repo_root,
name)` is a seam against the checkout the context names: it **checks the file exists first**
(`sys.path` is process-global; a bare import could otherwise resolve against some other importable
checkout and measure a working copy the operator never named), then imports. Probes load only the
stdlib-only `cell`; the heavy modules load only when work starts. `MatrixRunner` is the injection
point — the offline suite drives every gate/probe/floor claim through a fake, the `PodTransport`
pattern.

## 3. The `s5:` block

```yaml
s5:
  datasets: []          # [] = the campaign's own corpus.datasets
  fusion: equal         # equal | tuned  (mirrors run_matrix.FUSIONS)
  legs:                 # mirrors benchmark.s5.legs.ALL_LEG_NAMES, in matrix order
  - dense
  - bm25
  - kg
  - hyperedges
  - coe
  - packer
  - dense_content_only
  - graph_rescue
  - graph_rescue_packed
  - graph_rescue_coe
  limit: null           # null = the full gold sets
```

Flat, not nested: the four knobs are one concern (the matrix's shape), not four. The block
discipline is the `s3:`/`retrieval:` one unchanged — an older campaign.yaml without the block
keeps loading, a typo'd knob is LOUD, a wrongly-typed value is rejected at parse. Header line:

```
# s5 matrix     : 7 dataset(s) (the corpus selection) × 10 leg(s); full query sets; fusion equal
```

Four CLI flags (`--s5-datasets`, `--s5-legs`, `--s5-limit`, `--s5-fusion`), every one resolved
`is not None` per Wave 1-C: `--s5-legs ''` reaches the empty-legs refusal instead of silently
restoring the full ladder, `--s5-limit 0` and `--s5-limit ''` are loud, `--s5-datasets ''` is the
explicit "follow the corpus selection" answer. **No s5 prompt** — deliberately, and documented:
the default *is* the full matrix, so every deviation is a narrowing an operator makes on purpose,
not a headline choice worth interrogating before the S3 gate is even approved.

The lockstep is pinned from the test side, where importing both is free:
`S5_LEGS == benchmark.s5.legs.ALL_LEG_NAMES` (order included), `S5_FUSIONS == run_matrix.FUSIONS`,
`set(KNOWN_DATASETS) == set(datasets.DATASET_NAMES)`.

## 4. `campaign.yaml` regenerated (same commit)

K1's report found the committed `benchmark/sota-campaign/campaign.yaml` was a stale pre-S3
artifact. Regenerated with:

```
latence-campaign init --non-interactive \
  --campaign-root benchmark/sota-campaign \
  --pod-ssh 'ssh root@… -p … -i ~/.ssh/id_ed25519' \
  -o benchmark/sota-campaign/campaign.yaml
```

**Diff: +77 lines, −0.** A pure addition — no answer changed, no value edited:

| added | contents |
|---|---|
| header (8 lines) | the `s3 pipeline`, `retrieval` and `s5 matrix` summary lines the generator derives |
| `retrieval:` (16) | the measured S5b operating point (graph `auto`, packed @ 10k, coe @ 16k, the expander weights, the seed/rescue caps) |
| `s3:` (38) | chunk 640/80/8, gliner2 @ 0.5, induction `per_chunk` on deepseek-v4-flash (reasoning off, $150 cap, concurrency 32), the hand-authored label sets, learned redaction with guardrails on |
| `s5:` (14) | the block above — 7 datasets (the corpus selection) × 10 legs, full query sets, fusion `equal` |

Every one of those values was already in force; they were simply not written down in the file the
campaign claims to be driven by.

## Gates

| gate | result |
|---|---|
| **Dual-entry oracle** | `vidoseek/dense --limit 3`, real local export, `PYTHONHASHSEED=0`. Entry A = `python -m benchmark.s5.run_matrix --datasets vidoseek --legs dense --limit 3 --out <scratch>`. Entry B = `latence-campaign init` → seed s1/s2/s3 approved → `latence-campaign run s5-retrieval`. **Cells identical modulo the two wall-clock keys** (`latency_ms_median`, `latency_ms_p95`), asserted programmatically: same `status`/`control`/`fusion`/`limit`/`vector_source`, same `metrics` (`recall@1` 0.3333, `recall@5` 1.0, `recall@10` 1.0, `ndcg@10` 0.6872, `doc_recall@5/@10` 1.0), same full `resolution` block (3/3 refs, page granularity 3/0, `trustworthy: true`) |
| Default `--out` + cell naming | pinned: `DEFAULT_OUT == <repo>/benchmark/sota-campaign/stages/s5-retrieval`, `{dataset}__{leg}.json`, `results/` + `results-weighted/` |
| Refusal at the stage's entry | `run(plan)` raises `MatrixRefused` without `PYTHONHASHSEED=0` (not only `main`); the CLI's stderr line is asserted verbatim |
| STATE.json byte test | four-stage registry, injected ticking clock, exact bytes asserted |
| `latence-campaign init --non-interactive` byte-diff | changed as expected (`s5:` added); committed campaign.yaml regenerated in the same commit, diff summarised above |
| `packages/latence-benchmark/tests` + `benchmark/s5/tests` + `benchmark/serving` | **826 passed** (+44 new: 13 `test_run_matrix`, 19 `test_campaign_s5_stage`, 26 `test_campaign_s5_wizard`, less the two updated registry pins) |
| `ruff check packages benchmark/s5 benchmark/serving issues/wayfinder-enterprise-serving` | clean |
| `mypy --strict packages/latence-benchmark/src` | clean (35 files) |
| `mypy --strict -p benchmark.s5` (touched files) | clean — the one remaining error is pre-existing in `tests/test_rescue.py` (Wave 5-J), outside CI's target list |

## Notes for whoever picks this up

- **The registry pin moved.** `test_campaign_s3_stage::test_s3_is_registered_directly_after_s2`
  now asserts a prefix rather than the whole list, so stage five does not have to touch it.
- **The CLI flow test's last step changed meaning.** `resume` now advances into `s5-retrieval`;
  its fixture repo is a corpus tree, not a checkout, so the harness is refused (exit 1) rather
  than run against whatever happens to be importable. That refusal IS the assertion.
- **Not done here** (out of the binding contract's scope): the `s5-retrieval` stage does not write
  a consolidated stage `metrics.json` of its own — the harness's matrix `metrics.json` at the
  stage root IS the gate artifact, which is why `status` renders it as present.
- The pod sweep was untouched: nothing here runs on the pod, and the oracle wrote only into a
  scratch directory outside the campaign root.

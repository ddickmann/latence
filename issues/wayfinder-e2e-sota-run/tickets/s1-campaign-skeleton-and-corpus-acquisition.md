# S1 — Campaign skeleton + corpus selection/acquisition

Type: task
Status: claimed
Blocked by: — (S0 gate passed 2026-08-02)

## Deliverable

Two pieces, built together because acquisition RUNS as the first campaign stage:

### 1. The `latence-campaign` CLI skeleton (latence-benchmark package)

Console script `latence-campaign` (argparse, matching `latence-label` precedent) with:

- `init` — the wizard: prompts (interactive) / flags (non-interactive) → ONE frozen
  `CampaignAnswers` dataclass → pure byte-deterministic generator →
  `campaign.yaml` (commented header derived from the assembled config, `latence setup`
  house style: flag > prompt > default, "why" lines to stderr, dry-validate before write,
  exit 2 usage / 1 invalid / 0 written).
- `status` — render the stage table: per-stage state (pending/running/done/blocked/
  approved), done-probe results, metrics summary paths.
- `run <stage>` — execute one stage; REFUSES if the previous stage's gate is not approved
  (`approve <stage>` records Dennis's sign-off into STATE). Re-run of a done stage is a
  no-op unless `--fresh`.
- `resume` — continue an interrupted stage (filesystem probes, not exit codes).

Engine (module `latence_benchmark/campaign/`): stage registry (name, done_probe,
progress_probe, run callable), atomic STATE.json (temp+rename, one per campaign dir),
per-stage `metrics.json` + artifacts dir, `blocked` as a terminal state carried with a
reason. Port supervise.sh's shape; no shell. Everything offline-testable: probes and
fetchers behind seams, pytest with fakes, no network in tests.

Campaign dir layout: `<campaign_root>/{campaign.yaml, STATE.json,
stages/<NN-name>/{metrics.json, artifacts...}}`.

### 2. Stage `s1-corpus`: acquisition of the 6+2 with license gates

Per-dataset fetchers (idempotent, resumable, skip-if-exists, each behind a seam so tests
inject fakes) writing under `benchmark/sota-corpus/<dataset>/raw/` (gitignored — see lanes
below). BEFORE any byte is fetched, a license gate records the lane per dataset (reuse
`latence_benchmark.licensing.require_usable` pattern):

| dataset | source | lane |
|---|---|---|
| musique / hotpotqa | already local (`benchmark/official-data/`) — symlink/reference, do NOT re-download | attribution / share-alike-local |
| 2wikimultihopqa | Alab-NII/2wikimultihop release (or HF mirror) | dataset Apache-2.0; paragraph text share-alike-local |
| multihop-rag | HF `yixuantt/MultiHopRAG` (corpus + queries) | ODC-BY annotations; news text strictly local |
| ohr-bench | HF `opendatalab/OHR-Bench` (pdfs.zip + gt + qa) — `benchmark/corpus/fetch_corpus.sh` FETCH_OHR precedent | research-only, strictly local |
| vidoseek | HF `autumncc/ViDoSeek` (~1.7GB pdf zip + annotations) | annotations Apache-2.0; PDFs local |
| uda | `qinchuanhui/UDA-Benchmark` (FinHybrid + PaperTab subsets ONLY; skip NqText/FetaTab — HTML unsupported by any parser, recorded as the S1 ruling) | CC BY-SA → share-alike-local |

Gate artifact: `stages/s1-corpus/metrics.json` + a **corpus manifest** rendered via
`benchmark/corpus/manifest.py` (`build_manifest`/`write_manifest`) — per-dataset file
counts, byte sizes, type histogram, license lane, provenance URL, verified-at date.
`.gitignore` extended for `benchmark/sota-corpus/*/raw/`.

## Constraints

- No fake green: fetchers are seam-tested offline, but the S1 gate itself is only passed
  by the REAL downloads + real manifest.
- DoD: ruff + mypy clean, pytest for the engine + fetcher logic, no new hard deps in
  latence-benchmark beyond what's present (hf download via `huggingface_hub` behind the
  existing `[hub]` extra; stdlib urllib for direct URLs).
- The wizard's `init` for now covers campaign root + corpus selection (subset of the 6+2)
  + pod SSH target; later stages append their config blocks — design `CampaignAnswers`
  extensible.
- Commit style: `campaign(s1): ...`; work on branch hypergraph-m1 (== main), commit
  directly, no force-push.

# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stage ``s3-pipeline`` — full-feature pipeline data generation on the pod, per dataset.

The stage runs CLIENT-side (the ocr_stage pattern: the pod is a file server plus one
long-lived job at a time) and, per dataset of the plan, in a fixed sequential order — the
pod has ONE GPU, and the framework runs its GPU stages sequentially anyway:

1. **Local corpus preps** (pure functions + skip-if-exists writes): MultiHop-RAG's 609
   news articles and the 2Wiki dev question-pool paragraphs become deterministic
   content-addressed markdown under ``benchmark/sota-corpus/<ds>/markdown/``.
2. **Pipeline generation** — one full-feature pipeline.yaml per dataset, rendered from the
   campaign's ``s3:`` block (:func:`latence_benchmark.campaign.s3_lib.generate_pipeline_yaml`)
   into ``stages/s3-pipeline/pipelines/<ds>.yaml`` — committed alongside the campaign, so
   the exact configs that produced the data are reviewable forever.
3. **Remote run** behind the same :class:`~latence_benchmark.campaign.ocr_stage.PodTransport`
   seam: push markdown + yaml, launch ``latence process --run-id s3-<ds>-v1`` under
   ``nohup`` with a pid file, then poll a one-line remote status probe (done/running/dead +
   a monotone byte counter) under a stall budget. Re-invoking with the same run_id IS the
   framework's resume (finalized checkpoints replay from disk), so a dead run is relaunched
   with the same id and a live one is re-entered, never doubled.
4. On completion per dataset: pull ``quality-report.{json,md}`` + the export file listing
   into the stage dir, then fold the report's numbers (chunks, mentions, relations,
   entities, PII, induction cost actuals) into one consolidated metrics.json —
   the stage's gate artifact.

``musique`` and ``hotpotqa`` are selectable (s1 references their source files in place, s2
records them as text-native) but this stage has no pipeline for them. It used to verify an
export of an earlier campaign for the two and record it as reused; that export fed the
retrieval matrix removed by ADR-0068 and nothing reads it now, so the check is gone. The
stage records the two under ``not_processed`` in its metrics and names them in its one-line
note: a dataset it did nothing for is named, never silently dropped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from latence_core.run_store import run_dir_for

from latence_benchmark.campaign import s3_lib
from latence_benchmark.campaign.corpus_stage import SOTA_CORPUS_DIR
from latence_benchmark.campaign.ocr_stage import PodTransport, SubprocessTransport
from latence_benchmark.campaign.s3_lib import REMOTE_ROOT
from latence_benchmark.campaign.stages import (
    CampaignContext,
    StageBlockedError,
    StageSpec,
    stage_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from latence_benchmark.campaign.s3_lib import PrepFile

STAGE_NAME = "s3-pipeline"

#: The datasets S3 processes, in run order (one at a time — one GPU): the OCR'd group
#: (LLM label induction under the budget cap) first, then the two text-native preps
#: (hand-authored labels, no induction).
INDUCED_DATASETS: tuple[str, ...] = ("ohr_bench", "vidoseek", "uda")
TEXT_DATASETS: tuple[str, ...] = ("multihop_rag", "wiki2multihop")
PROCESS_ORDER: tuple[str, ...] = INDUCED_DATASETS + TEXT_DATASETS

#: Selectable datasets this stage has no pipeline for, in record order. They are recorded
#: under ``not_processed`` in the stage metrics (see the module docstring).
NOT_PROCESSED: tuple[str, ...] = ("musique", "hotpotqa")

#: The pod venv's `latence` CLI (the framework + GPU provider packages live there; see the
#: operator runbook in the S3 ticket). A missing venv surfaces as a blocked launch.
REMOTE_LATENCE = "/workspace/venvs/latence/bin/latence"

#: Pipeline runs move slower than OCR pages but write checkpoints continuously; an hour of
#: zero byte growth on the run store means a wedged stage, not a slow one.
STALL_BUDGET_S = 3600.0
POLL_INTERVAL_S = 60.0


def run_id_for(dataset: str) -> str:
    """The per-dataset run id. Same id on re-entry = the framework's checkpoint resume."""
    return f"s3-{dataset}-v1"


def _remote_store(dataset: str) -> str:
    return f"{REMOTE_ROOT}/{dataset}"


def _remote_run_dir(dataset: str) -> str:
    # The framework's own run layout (``latence_core.run_store``), addressed remotely: the
    # pod-side ``latence process`` writes under exactly this directory, so the probe/pull
    # paths can never drift from where the Runner actually lands its artifacts.
    return run_dir_for(_remote_store(dataset), run_id_for(dataset))


def _remote_pid(dataset: str) -> str:
    return f"{REMOTE_ROOT}/logs/{dataset}.pid"


def _remote_log(dataset: str) -> str:
    return f"{REMOTE_ROOT}/logs/{dataset}.log"


def status_cmd(dataset: str) -> str:
    """The one-line remote probe: ``<phase> <kbytes>`` (done | running | absent).

    ``done`` = the run's quality-report.json exists (the byte-backed completion signal —
    the runner writes it last); ``running`` = the pid-file process is alive; ``absent`` =
    neither, i.e. nothing launched yet OR the process died before finishing.

    The progress counter sums THREE observables, because no single one covers every stage
    shape this pipeline runs (both gaps were found the hard way, in production, 2026-08-04/05):

    * **store kbytes** — checkpoints + WAL; the natural signal for streaming GPU stages.
    * **run-log kbytes** — a network-bound phase (LLM schema induction; UDA's 230k calls ran
      seven hours) streams progress into the log while writing NOTHING to the store.
    * **process CPU-seconds** — a compute-bound phase (type consolidation and disambiguation
      over UDA's 2.6M mentions) holds its records in memory, logs nothing, and writes its
      checkpoint only on completion: store AND log are both flat for hours while one core
      runs at 100%. CPU time is the only monotone evidence of work in that window.

    A stall is therefore "the process is alive but has consumed no CPU and written no bytes
    for the whole budget" — which is what a genuinely wedged run looks like, and what a
    slow-but-working one never does.
    """
    return (
        f"sh -c 'if [ -f {_remote_run_dir(dataset)}/quality-report.json ]; then p=done; "
        f"elif kill -0 \"$(cat {_remote_pid(dataset)} 2>/dev/null)\" 2>/dev/null; "
        f"then p=running; else p=absent; fi; "
        f"b=$(du -sk {_remote_store(dataset)} 2>/dev/null | cut -f1); "
        f"lb=$(du -sk {_remote_log(dataset)} 2>/dev/null | cut -f1); "
        # ``times`` (not ``time``) prints cumulative CPU SECONDS as a bare integer — no
        # HH:MM:SS parsing, so the probe stays a single quote-safe shell line.
        # No quoted whitespace anywhere in here: the whole probe rides inside a single-quoted
        # ``sh -c '...'``, so a nested ``tr -d ' '`` would terminate that quote (it did).
        # ``$(( ))`` strips the leading padding ps emits, so no trimming is needed.
        f"c=$(ps -o times= -p \"$(cat {_remote_pid(dataset)} 2>/dev/null)\" 2>/dev/null); "
        f'echo "$p $(( ${{b:-0}} + ${{lb:-0}} + ${{c:-0}} ))"\''
    )


def launch_cmd(dataset: str) -> str:
    """nohup + pid file, logging to the dataset's log; PYTHONHASHSEED pinned (repo rule)."""
    return (
        f"mkdir -p {REMOTE_ROOT}/logs && cd {REMOTE_ROOT} && "
        f"(nohup env PYTHONHASHSEED=0 {REMOTE_LATENCE} process "
        f"--config pipelines/{dataset}.yaml --run-id {run_id_for(dataset)} "
        f">> {_remote_log(dataset)} 2>&1 & echo $! > {_remote_pid(dataset)})"
    )


# -- remote state ----------------------------------------------------------------------------


def _read_run_state(transport: PodTransport, dataset: str) -> tuple[str, int] | None:
    """``(phase, progress)`` from the status probe, or ``None`` on an unreadable reply.

    An unreadable reply (dropped ssh round-trip, garbled output) is NOT fatal by itself —
    the stall budget, not the read error, decides when persistent silence becomes a block
    (the same posture as the s2 poll loop).
    """
    rc, out = transport.run(status_cmd(dataset))
    if rc != 0:
        return None
    parts = out.split()
    if len(parts) < 2 or parts[-2] not in ("done", "running", "absent"):
        return None
    try:
        progress = int(parts[-1])
    except ValueError:
        return None
    return parts[-2], progress


def _log_tail(transport: PodTransport, dataset: str) -> str:
    rc, out = transport.run(f"tail -n 20 {_remote_log(dataset)}")
    return out.strip()[-600:] if rc == 0 and out.strip() else "(log unavailable)"


def poll_pipeline_run(
    transport: PodTransport,
    dataset: str,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    stall_budget: float = STALL_BUDGET_S,
    interval: float = POLL_INTERVAL_S,
) -> None:
    """Poll one dataset's run until ``done``; a dead process or a stalled store blocks.

    ``absent`` mid-poll means the launched process died without writing its quality report
    — blocked immediately, carrying the remote log tail, because waiting out the stall
    budget on a corpse would just delay the same answer by an hour.
    """
    best_progress = -1
    last_advance = clock()
    while True:
        state = _read_run_state(transport, dataset)
        if state is not None:
            phase, progress = state
            if phase == "done":
                return
            if phase == "absent":
                msg = (
                    f"{dataset}: the pod run died before its quality report "
                    f"(run_id {run_id_for(dataset)}). Log tail: {_log_tail(transport, dataset)} "
                    "— fix the cause, then `latence-campaign resume` relaunches under the "
                    "same run_id (finalized checkpoints replay from disk)."
                )
                raise StageBlockedError(msg)
            if progress > best_progress:
                best_progress = progress
                last_advance = clock()
        if clock() - last_advance > stall_budget:
            msg = (
                f"{dataset}: pod run stalled — no byte growth under {_remote_store(dataset)} "
                f"for more than {stall_budget:.0f}s (run_id {run_id_for(dataset)}; see "
                f"{_remote_log(dataset)} on the pod). Fix the cause, then "
                "`latence-campaign resume` re-enters the poll."
            )
            raise StageBlockedError(msg)
        sleep(interval)


# -- local preps -----------------------------------------------------------------------------


def write_prep_files(dest: Path, files: Sequence[PrepFile]) -> tuple[int, int]:
    """Materialise a prep: skip-if-exists per file, temp+rename. Returns (written, skipped)."""
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for prep in files:
        target = dest / prep.filename
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        tmp = target.with_name(target.name + ".part")
        tmp.write_text(prep.content, encoding="utf-8")
        tmp.replace(target)
        written += 1
    return written, skipped


def prepare_multihop_rag(repo_root: Path) -> dict[str, int]:
    """609 news articles (``corpus.json``) → content-addressed markdown, skip-if-exists."""
    raw = repo_root / SOTA_CORPUS_DIR / "multihop_rag" / "raw" / "corpus.json"
    if not raw.is_file():
        msg = f"multihop_rag: expected {raw} — run/approve s1-corpus first."
        raise StageBlockedError(msg)
    files = s3_lib.multihop_rag_articles(raw.read_text(encoding="utf-8"))
    dest = repo_root / SOTA_CORPUS_DIR / "multihop_rag" / "markdown"
    written, skipped = write_prep_files(dest, files)
    return {"files": len(files), "written": written, "skipped": skipped}


def prepare_wiki2multihop(repo_root: Path) -> dict[str, int]:
    """The 2Wiki dev question-pool corpus: one md per distinct context paragraph (M6-style)."""
    raw = repo_root / SOTA_CORPUS_DIR / "wiki2multihop" / "raw" / "dev.parquet"
    if not raw.is_file():
        msg = f"wiki2multihop: expected {raw} — run/approve s1-corpus first."
        raise StageBlockedError(msg)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "wiki2multihop: pyarrow is required to read the dev split parquet."
        raise StageBlockedError(msg) from exc
    # ``cast`` rather than a type-ignore: pyarrow is dynamically typed in this repo's mypy
    # config (the ocr_stage posture).
    column = cast("Any", pq).read_table(raw, columns=["context"]).column("context")
    files = s3_lib.wiki2multihop_paragraphs(str(cell) for cell in column.to_pylist())
    dest = repo_root / SOTA_CORPUS_DIR / "wiki2multihop" / "markdown"
    written, skipped = write_prep_files(dest, files)
    return {"files": len(files), "written": written, "skipped": skipped}


# -- pipeline generation ---------------------------------------------------------------------


def _active(ctx: CampaignContext) -> tuple[list[str], list[str]]:
    """``(processed, not_processed)`` for the selection, each in its fixed order."""
    processed = [d for d in PROCESS_ORDER if d in ctx.answers.datasets]
    not_processed = [d for d in NOT_PROCESSED if d in ctx.answers.datasets]
    return processed, not_processed


def _induced_active(ctx: CampaignContext, processed: Sequence[str]) -> list[str]:
    if ctx.answers.s3.induction_mode == "none":
        return []
    return [d for d in processed if d in INDUCED_DATASETS]


def pipelines_dir(ctx: CampaignContext) -> Path:
    return stage_dir(ctx.campaign_root, STAGE_NAME) / "pipelines"


def generate_pipelines(ctx: CampaignContext, processed: Sequence[str]) -> dict[str, str]:
    """Render + atomically write every dataset's pipeline.yaml; returns ``{ds: text}``.

    When latence-core is importable (the dev checkout always has it) each rendered file is
    additionally validated through the REAL ``Pipeline`` pydantic model before it is written
    — a generator bug surfaces here, not three hours into a pod run. The offline test suite
    pins the same validation, so environments without core lose only the redundant check.
    """
    induced = _induced_active(ctx, processed)
    call_budgets = s3_lib.induction_call_budgets(
        ctx.answers.s3.induction_budget_usd_cap, ctx.answers.s3.induction_model, induced
    )
    rendered: dict[str, str] = {}
    for dataset in processed:
        text = s3_lib.generate_pipeline_yaml(
            dataset,
            ctx.answers.s3,
            induce=dataset in induced,
            max_induction_calls=call_budgets.get(dataset, 0),
        )
        try:
            from latence_core.pipeline import Pipeline
        except ImportError:
            pass
        else:
            try:
                Pipeline.from_yaml_text(text)
            except Exception as exc:  # pydantic error types vary; the reason is the point
                msg = f"{dataset}: generated pipeline.yaml failed core validation: {exc}"
                raise StageBlockedError(msg) from exc
        rendered[dataset] = text
    out = pipelines_dir(ctx)
    out.mkdir(parents=True, exist_ok=True)
    for dataset, text in rendered.items():
        target = out / f"{dataset}.yaml"
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
    return rendered


# -- probes ----------------------------------------------------------------------------------


def _metrics_path(ctx: CampaignContext) -> Path:
    return stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json"


def _report_dir(ctx: CampaignContext, dataset: str) -> Path:
    return stage_dir(ctx.campaign_root, STAGE_NAME) / "reports" / dataset


def s3_done_probe(ctx: CampaignContext) -> bool:
    """Metrics cover exactly the selection AND every processed dataset's report landed.

    Same contract as the earlier stages: editing ``corpus.datasets`` re-opens the stage,
    and a metrics file without the byte-backed reports behind it does not count.
    """
    path = _metrics_path(ctx)
    if not path.is_file():
        return False
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    datasets = recorded.get("datasets")
    if not isinstance(datasets, dict):
        return False
    processed, not_processed = _active(ctx)
    if set(datasets) != set(processed):
        return False
    if recorded.get("not_processed") != not_processed:
        return False
    return all((_report_dir(ctx, ds) / "quality-report.json").is_file() for ds in processed)


def s3_progress_probe(ctx: CampaignContext) -> str:
    processed, _ = _active(ctx)
    landed = sum(
        1 for ds in processed if (_report_dir(ctx, ds) / "quality-report.json").is_file()
    )
    return f"{landed}/{len(processed)} dataset quality report(s) pulled"


# -- the stage run ---------------------------------------------------------------------------


def _push_dataset(ctx: CampaignContext, transport: PodTransport, dataset: str) -> None:
    markdown = ctx.repo_root / SOTA_CORPUS_DIR / dataset / "markdown"
    if not markdown.is_dir() or not any(markdown.glob("*.md")):
        msg = (
            f"{dataset}: no markdown under {markdown} — "
            + (
                "run/approve s2-ocr first."
                if dataset in INDUCED_DATASETS
                else "the local corpus prep should have produced it (stage bug)."
            )
        )
        raise StageBlockedError(msg)
    rc, out = transport.run(f"mkdir -p {REMOTE_ROOT}/input/{dataset} {REMOTE_ROOT}/pipelines")
    if rc != 0:
        msg = f"pod refused mkdir under {REMOTE_ROOT} (rc={rc}): {out.strip()[:300]}"
        raise StageBlockedError(msg)
    # rsync no-trailing-slash: the local `markdown` dir lands AS input/<ds>/markdown — the
    # exact path the generated pipeline's source stage reads.
    transport.push(markdown, f"{REMOTE_ROOT}/input/{dataset}/")
    transport.push(pipelines_dir(ctx) / f"{dataset}.yaml", f"{REMOTE_ROOT}/pipelines/")


def _launch(transport: PodTransport, dataset: str) -> None:
    rc, out = transport.run(launch_cmd(dataset))
    if rc != 0:
        msg = f"{dataset}: pod launch failed (rc={rc}): {out.strip()[-500:]}"
        raise StageBlockedError(msg)


def _pull_dataset_results(
    ctx: CampaignContext, transport: PodTransport, dataset: str
) -> dict[str, Any]:
    out_dir = _report_dir(ctx, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _remote_run_dir(dataset)
    transport.pull(f"{run_dir}/quality-report.json", out_dir / "quality-report.json")
    transport.pull(f"{run_dir}/quality-report.md", out_dir / "quality-report.md")
    rc, listing = transport.run(f"find {run_dir}/export -type f | sort")
    if rc == 0:
        (out_dir / "export-files.txt").write_text(listing, encoding="utf-8")
    try:
        report = json.loads((out_dir / "quality-report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"{dataset}: pulled quality-report.json is unreadable: {exc}"
        raise StageBlockedError(msg) from exc
    if not isinstance(report, dict):
        msg = f"{dataset}: pulled quality-report.json is not a JSON object."
        raise StageBlockedError(msg)
    return report


def _require_nonempty_run(
    dataset: str, report_metrics: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    """The no-fake-green floor: a quality report over an EMPTY corpus blocks, never passes.

    A ``latence process`` run over zero documents still writes its quality-report.json, so
    the status probe reads ``done``, the report is pulled, and — without this floor — the
    stage would advance an empty dataset as if it were data. That is exactly
    what happened in the field when the markdown push landed as a dangling symlink on the
    pod. Zero documents OR zero chunks means the input push (or the corpus behind it)
    likely failed; the block names the dataset and the counts so the operator looks at the
    pod's input dir, not the pipeline.

    Second floor, from the crashed-graph incident (production S3, 2026-08-04): a run that
    CRASHED inside Graph Assembly can still leave a quality-report.json behind, so the
    ``done`` probe fires and the pull path would advance a graph-less run. Every
    campaign-generated pipeline configures ``graph.canonical`` (s3_lib), so
    ``graph.nodes_total == 0`` in the RAW report is a crashed or truncated graph stage,
    never a legitimate result — block, pointing the operator at the pod's run log.
    """
    documents = int(report_metrics.get("documents", 0))
    chunks = int(report_metrics.get("chunks", 0))
    if documents <= 0 or chunks <= 0:
        msg = (
            f"{dataset}: quality report records an EMPTY run — {documents} document(s), "
            f"{chunks} chunk(s). The input push likely failed (e.g. markdown arrived as a "
            f"dangling symlink); verify {REMOTE_ROOT}/input/{dataset}/markdown on the pod, "
            f"remove the empty run store {_remote_store(dataset)}, then "
            "`latence-campaign resume`."
        )
        raise StageBlockedError(msg)
    graph_raw = report.get("graph")
    graph = graph_raw if isinstance(graph_raw, dict) else {}
    nodes_total = int(graph.get("nodes_total", 0))
    if nodes_total <= 0:
        msg = (
            f"{dataset}: quality report shows a CRASHED/TRUNCATED graph run — "
            f"graph.nodes_total={nodes_total} — on a graph-configured "
            f"pipeline. Inspect the pod log {_remote_log(dataset)} for the crash, fix the "
            f"cause, remove the run store {_remote_store(dataset)}, then "
            "`latence-campaign resume`."
        )
        raise StageBlockedError(msg)


def run_s3_stage(
    ctx: CampaignContext,
    transport: PodTransport | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Prep → generate → (push → launch/re-enter → poll → pull) per dataset.

    :param transport: The pod seam; ``None`` builds the real ssh/rsync transport from the
        campaign config's ``pod_ssh``. Tests inject fakes here.
    :param clock: Monotonic-seconds source for the stall budget (seam, tests fake it).
    :param sleep: Poll-interval sleeper (seam, tests fake it).
    """
    processed, not_processed = _active(ctx)
    s3 = ctx.answers.s3
    out_dir = stage_dir(ctx.campaign_root, STAGE_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_metrics: dict[str, Any] = {}

    # 1. Local corpus preps (pure functions + skip-if-exists writes; never touch the pod).
    preps: dict[str, dict[str, int]] = {}
    if "multihop_rag" in processed:
        preps["multihop_rag"] = prepare_multihop_rag(ctx.repo_root)
    if "wiki2multihop" in processed:
        preps["wiki2multihop"] = prepare_wiki2multihop(ctx.repo_root)

    # 2. Generate every pipeline.yaml (committed-able gate artifacts) + the budget split
    # (proportional to the measured chunk counts, scaled to the cap — see s3_lib).
    induced = _induced_active(ctx, processed)
    call_budgets = s3_lib.induction_call_budgets(
        s3.induction_budget_usd_cap, s3.induction_model, induced
    )
    generate_pipelines(ctx, processed)

    # 3. The transport, built late: a selection with nothing to process never needs ssh.
    if processed:
        if transport is None:
            if not ctx.answers.pod_ssh:
                msg = (
                    "s3-pipeline runs on the GPU pod but campaign.pod_ssh is empty — set it "
                    "in campaign.yaml (the full `ssh root@IP -p PORT -i KEY` line)."
                )
                raise StageBlockedError(msg)
            transport = SubprocessTransport(ctx.answers.pod_ssh)

        # 4. Sequential per-dataset runs (one GPU; the pod runs exactly one job at a time).
        for dataset in processed:
            local_report = _report_dir(ctx, dataset) / "quality-report.json"
            resumed = False
            if not local_report.is_file():
                state = _read_run_state(transport, dataset)
                phase = state[0] if state is not None else "absent"
                if phase == "absent":
                    # Fresh launch OR relaunch-after-death: same run_id, so finalized
                    # checkpoints replay from disk (the framework's resume semantics).
                    _push_dataset(ctx, transport, dataset)
                    _launch(transport, dataset)
                else:
                    resumed = True  # live or already-done job: re-entered, never doubled
                if phase != "done":
                    poll_pipeline_run(transport, dataset, clock=clock, sleep=sleep)
                report = _pull_dataset_results(ctx, transport, dataset)
            else:
                # Pulled on a previous pass: the local bytes are the authority (resume
                # never redoes finished work, and never needs the pod for it).
                resumed = True
                report = json.loads(local_report.read_text(encoding="utf-8"))
            report_metrics = s3_lib.dataset_metrics_from_report(
                report, induction_model=s3.induction_model
            )
            # Sanity floor on BOTH paths (fresh pull and local resume): empty blocks,
            # and so does a crashed-graph report (zero nodes).
            _require_nonempty_run(dataset, report_metrics, report)
            dataset_metrics[dataset] = {
                **report_metrics,
                "run_id": run_id_for(dataset),
                "induction_mode": (s3.induction_mode if dataset in induced else "none"),
                "pipeline_yaml": f"pipelines/{dataset}.yaml",
                "resumed": resumed,
            }

    # 5. Consolidated gate metrics — the one artifact Dennis reads before approving.
    usd_actual_total = round(
        sum(
            m["induction"]["usd_actual"]
            for m in dataset_metrics.values()
            if isinstance(m.get("induction"), dict)
        ),
        4,
    )
    metrics = {
        "stage": STAGE_NAME,
        "config": {
            "chunk": {
                "max_tokens": s3.chunk_max_tokens,
                "overlap_tokens": s3.chunk_overlap_tokens,
                "min_tokens": s3.chunk_min_tokens,
            },
            "extraction": {
                "provider": s3.extraction_provider,
                "model": s3.extraction_model,
                "threshold": s3.extraction_threshold,
            },
            "redaction": {
                "provider": s3.redaction_provider,
                "guardrails": s3.redaction_guardrails,
            },
        },
        "induction_budget": {
            "mode": s3.induction_mode,
            "model": s3.induction_model,
            "reasoning": s3.induction_reasoning,
            "usd_cap": s3.induction_budget_usd_cap,
            "induced_datasets": induced,
            "max_calls_by_dataset": call_budgets,
            "est_usd_per_call": s3_lib.est_usd_per_call(s3.induction_model),
            "usd_actual_total": usd_actual_total,
        },
        "preps": preps,
        "datasets": dataset_metrics,
        "not_processed": not_processed,
        "pod": {"remote_root": REMOTE_ROOT, "remote_latence": REMOTE_LATENCE},
    }
    tmp = _metrics_path(ctx).with_name("metrics.json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(_metrics_path(ctx))

    total_chunks = sum(
        int(m.get("chunks", 0)) for m in dataset_metrics.values() if "chunks" in m
    )
    skipped = (
        f"{len(not_processed)} not processed, this stage has no pipeline for them "
        f"({', '.join(not_processed)})"
        if not_processed
        else "0 not processed"
    )
    return (
        f"{len(processed)} dataset(s) processed ({total_chunks} chunks), {skipped}; "
        f"induction actuals ${usd_actual_total:.2f} "
        f"(cap ${s3.induction_budget_usd_cap:.2f}); reports under {out_dir / 'reports'}"
    )


S3_PIPELINE_STAGE = StageSpec(
    name=STAGE_NAME,
    title="Full-feature pipeline data generation on the pod (GLiNER2 + induction, per dataset)",
    done_probe=s3_done_probe,
    progress_probe=s3_progress_probe,
    run=run_s3_stage,
)

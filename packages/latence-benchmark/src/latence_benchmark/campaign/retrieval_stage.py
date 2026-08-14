# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Stage ``s5-retrieval`` — the retrieval feature matrix, as a campaign stage.

The S5 harness (``benchmark/s5/``) already honoured every rule this campaign is built on:
per-cell resumability, refuse-don't-fake on absent inputs, ``REPORT.md`` + ``metrics.json`` as
gate artifacts. It honoured them through machinery no campaign could reach — an argparse main
with a hardcoded output directory, module-level corpus constants, and no entry in
:func:`~latence_benchmark.campaign.stages.registry`. So the campaign's own guide claimed S5
"runs from the same config" while ``run_matrix`` never opened ``campaign.yaml``.

This module closes that. It owns three things and delegates the fourth:

1. **Plan derivation** — :func:`request_for` turns the campaign's ``s5:`` block (plus its
   ``corpus.datasets`` and its campaign root) into a :class:`MatrixRequest`. The stage decides
   what to measure; the harness decides how.
2. **The probes** — :func:`s5_done_probe` / :func:`s5_progress_probe` ask the FILESYSTEM,
   through the harness's own cell naming (:mod:`benchmark.s5.cell`, stdlib-only), so a probe
   can never drift from where the writer puts a cell.
3. **The floor** — a dataset whose every leg recorded ``missing-inputs`` did not measure
   nothing, it measured *no export*; that blocks (see :func:`_require_measured`), exactly as
   an empty quality report blocks the s3 stage.
4. **The run** — delegated to :func:`benchmark.s5.run_matrix.run`, the harness's API. Not
   ``main()``: a stage that shelled through argparse would be a pass-through that could not
   own its plan.

**Why the harness is imported lazily, from ``ctx.repo_root``.** ``benchmark/s5`` is a
measurement harness in the repo, not a dependency of this package — it pulls numpy, pyarrow and
latence-retrieval, and the campaign package deliberately imports none of them. The context
already carries the checkout the stage operates on, so the import is a *seam* against that
checkout (:func:`harness_module`) rather than a package dependency: probes load only the
stdlib-only cell schema, and the heavy modules load only when work actually starts.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_benchmark.campaign.stages import (
    CampaignContext,
    StageBlockedError,
    StageSpec,
    stage_dir,
)
from latence_benchmark.campaign.wizard import S5_LEGS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import ModuleType

STAGE_NAME = "s5-retrieval"


# -- the plan ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixRequest:
    """What this campaign asks the S5 harness for — resolved answers, no paths derived twice.

    Deliberately campaign-side data (no harness types), so plan derivation, both probes and
    every test of them run without importing numpy, pyarrow or latence-retrieval. It is turned
    into the harness's own ``Plan`` at the last moment, inside :func:`harness_runner`.
    """

    campaign_root: Path
    repo_root: Path
    datasets: tuple[str, ...]
    legs: tuple[str, ...]
    limit: int | None
    fusion: str

    @property
    def out(self) -> Path:
        """``<campaign_root>/stages/s5-retrieval/`` — cells, REPORT.md, metrics.json."""
        return stage_dir(self.campaign_root, STAGE_NAME)


@dataclass(frozen=True)
class MatrixSummary:
    """One matrix run, as the stage reports it. The seam's return type."""

    written: int
    present: int
    planned: int
    report: Path
    metrics: Path


#: The seam: anything that can satisfy a :class:`MatrixRequest`. The real one is
#: :func:`harness_runner`; tests inject a fake so the stage's gate/probe/floor behaviour is
#: provable offline, without a 100 MB export or a 1.2 GB embedding checkpoint.
MatrixRunner = Callable[["MatrixRequest"], MatrixSummary]


def request_for(ctx: CampaignContext) -> MatrixRequest:
    """The campaign's ``s5:`` block → one :class:`MatrixRequest`.

    Two resolutions happen here and nowhere else:

    * **datasets** — an empty ``s5.datasets`` means *the campaign's own corpus selection*, so
      trimming ``corpus.datasets`` trims the matrix with it. An explicit list is taken verbatim
      (it was validated against the known datasets at parse time).
    * **leg order** — the requested legs are re-sorted into :data:`S5_LEGS` order, because that
      order is the report's column order and a matrix whose columns follow the order an operator
      happened to type would be a different artifact for the same answers.
    """
    s5 = ctx.answers.s5
    datasets = s5.datasets or ctx.answers.datasets
    legs = tuple(leg for leg in S5_LEGS if leg in set(s5.legs))
    return MatrixRequest(
        campaign_root=ctx.campaign_root,
        repo_root=ctx.repo_root,
        datasets=tuple(datasets),
        legs=legs,
        limit=s5.limit,
        fusion=s5.fusion,
    )


# -- the harness seam --------------------------------------------------------------------------


def harness_module(repo_root: Path, name: str) -> ModuleType:
    """Import ``benchmark.s5.<name>`` from the checkout at ``repo_root``.

    The harness is repo code, not a package dependency, so the checkout the context names is
    the one authority on where it lives. The file is checked FIRST and its absence raises
    :class:`ModuleNotFoundError`: ``sys.path`` is process-global, so a bare import could
    otherwise resolve against some *other* checkout that happens to be importable and measure
    a working copy the operator never named.
    """
    source = repo_root / "benchmark" / "s5" / f"{name}.py"
    if not source.is_file():
        msg = f"no S5 harness at {source}"
        raise ModuleNotFoundError(msg)
    root = str(repo_root)
    if root not in sys.path:  # pragma: no cover - process-global, order-dependent
        sys.path.insert(0, root)
    return importlib.import_module(f"benchmark.s5.{name}")


def _cell_paths(request: MatrixRequest) -> tuple[Path, ...]:
    """Every cell file this request is responsible for, or ``()`` if the harness is absent.

    Asks :mod:`benchmark.s5.cell` — which owns both the ``<dataset>__<leg>.json`` naming and the
    per-fusion cell directory, and imports nothing outside the stdlib — so the probe and the
    writer cannot disagree about where a cell is. An unimportable harness yields ``()``, which
    reads as "unsatisfied" in the probes and becomes a loud block in :func:`run_s5_stage`.
    """
    try:
        cell = harness_module(request.repo_root, "cell")
    except ImportError:
        return ()
    directory = cell.results_dir(request.out, request.fusion)
    return tuple(
        cell.cell_path(directory, dataset, leg)
        for dataset in request.datasets
        for leg in request.legs
    )


def harness_runner(request: MatrixRequest) -> MatrixSummary:
    """Run the matrix through :func:`benchmark.s5.run_matrix.run` — the harness's own API.

    The plan is built here, from the request: the campaign's stage directory as ``out``, the
    context's checkout as the corpus layout, ``force=False`` (the campaign never re-runs a cell
    it already has — that is what makes a multi-hour matrix resumable across sessions).
    """
    try:
        run_matrix = harness_module(request.repo_root, "run_matrix")
        datasets_mod = harness_module(request.repo_root, "datasets")
    except ImportError as exc:
        msg = (
            f"the S5 harness is not importable from {request.repo_root} "
            f"(benchmark/s5): {exc}. Point --repo-root at a latence-framework checkout with "
            "the harness's dependencies installed."
        )
        raise StageBlockedError(msg) from exc
    plan = run_matrix.Plan(
        datasets=request.datasets,
        legs=request.legs,
        limit=request.limit,
        out=request.out,
        force=False,
        fusion=request.fusion,
        layout=datasets_mod.CorpusLayout.under(request.repo_root),
    )
    try:
        outcome = run_matrix.run(plan)
    except run_matrix.MatrixRefused as exc:
        raise StageBlockedError(str(exc)) from exc
    return MatrixSummary(
        written=len(outcome.written),
        present=len(outcome.cells),
        planned=len(run_matrix.planned_cells(plan)),
        report=outcome.report,
        metrics=outcome.metrics,
    )


# -- probes ------------------------------------------------------------------------------------


def s5_done_probe(ctx: CampaignContext) -> bool:
    """Every planned cell on disk, plus the two gate artifacts a human signs off on.

    The cells alone are the contract's definition of the work; ``REPORT.md`` and
    ``metrics.json`` are added because they are what ``approve`` is an approval OF, and the
    harness writes them last — so requiring them is also what distinguishes a completed run
    from a matrix half-filled by hand.
    """
    request = request_for(ctx)
    paths = _cell_paths(request)
    if not paths or not all(path.is_file() for path in paths):
        return False
    out = request.out
    return (out / "REPORT.md").is_file() and (out / "metrics.json").is_file()


def s5_progress_probe(ctx: CampaignContext) -> str:
    """``<done>/<total> matrix cell(s)`` — the count a long matrix advances through."""
    paths = _cell_paths(request_for(ctx))
    if not paths:
        return "0/0 matrix cell(s) — no plan (S5 harness not importable?)"
    done = sum(1 for path in paths if path.is_file())
    return f"{done}/{len(paths)} matrix cell(s) written"


# -- the no-fake-green floor --------------------------------------------------------------------


def _cell_status(path: Path) -> str:
    """One cell file's recorded ``status``, or ``"unreadable"``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unreadable"
    return str(payload.get("status", "unknown")) if isinstance(payload, dict) else "unreadable"


def _require_measured(request: MatrixRequest, paths: Sequence[Path]) -> dict[str, Any]:
    """Refuse a matrix in which some dataset measured nothing at all.

    The harness's per-leg refusal is correct and stays: the reused M6 exports genuinely carry
    no embedding column, so ``musique/dense`` is honestly ``missing-inputs`` and the matrix
    reports it as such. What must never pass is a *dataset* whose every leg refused — that is
    an absent or unfinished export being advanced as a measurement, the retrieval twin of the
    empty quality report the s3 stage blocks on. Returns the per-dataset status histogram that
    goes into the stage's completion note.
    """
    by_dataset: dict[str, dict[str, int]] = {dataset: {} for dataset in request.datasets}
    for path in paths:
        dataset = path.name.split("__", 1)[0]
        counts = by_dataset.setdefault(dataset, {})
        status = _cell_status(path) if path.is_file() else "absent"
        counts[status] = counts.get(status, 0) + 1
    barren = sorted(
        dataset for dataset, counts in by_dataset.items() if counts.get("ok", 0) == 0
    )
    if barren:
        msg = (
            f"{', '.join(barren)}: every leg recorded a refusal — no cell was measured. "
            "That is an absent or unfinished S3 export, not a retrieval result; check "
            f"benchmark/sota-corpus/s3-exports/<dataset> under {request.repo_root}, then "
            "`latence-campaign run s5-retrieval --fresh`."
        )
        raise StageBlockedError(msg)
    return {dataset: dict(sorted(counts.items())) for dataset, counts in by_dataset.items()}


# -- the stage run -------------------------------------------------------------------------------


def run_s5_stage(ctx: CampaignContext, matrix: MatrixRunner | None = None) -> str:
    """Derive the plan, run the matrix, enforce the floor; returns the operator's one line.

    :param matrix: The harness seam; ``None`` uses :func:`harness_runner` (the real
        ``run_matrix.run``). Tests inject fakes here, the ``PodTransport`` pattern.
    """
    request = request_for(ctx)
    summary = (matrix or harness_runner)(request)
    statuses = _require_measured(request, _cell_paths(request))
    ok = sum(counts.get("ok", 0) for counts in statuses.values())
    return (
        f"{len(request.datasets)} dataset(s) × {len(request.legs)} leg(s) = "
        f"{summary.planned} cell(s), {ok} measured, {summary.written} written this run "
        f"(fusion {request.fusion}); gate artifacts {summary.report.name} + "
        f"{summary.metrics.name} under {request.out}"
    )


S5_RETRIEVAL_STAGE = StageSpec(
    name=STAGE_NAME,
    title="Retrieval feature matrix (dense → +bm25 → +KG → +hyperedges → +CoE → +packer)",
    done_probe=s5_done_probe,
    progress_probe=s5_progress_probe,
    run=run_s5_stage,
)

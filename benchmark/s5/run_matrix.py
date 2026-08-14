# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``run_matrix.py`` — the S5 feature matrix: (dataset × leg) → one comparable table.

**The API is :func:`run`, not the CLI.** ``run(plan) -> MatrixOutcome`` is the whole harness:
a :class:`Plan` in, the cells + report on disk and a record of what happened out. Everything
argparse-shaped (:func:`parse_args`, :func:`main`) is one thin adapter over it, and so is the
``s5-retrieval`` campaign stage — which derives its own :class:`Plan` from ``campaign.yaml``
and calls :func:`run` directly. The determinism refusal and the sample seed therefore live at
that entry, not in ``main``: a caller that never touches argparse must still be unable to
produce a non-reproducible cell.

Resumable by construction: every (dataset, leg) cell is written to its own JSON file under
``benchmark/sota-campaign/stages/s5-retrieval/results/`` the moment it finishes, and a re-run
skips cells whose file already exists (``--force`` re-runs them). A matrix that takes hours can
therefore be interrupted, extended one dataset at a time, or resumed after a crash without
losing or silently re-mixing work.

Deterministic: ``PYTHONHASHSEED=0`` **and** a pinned BLAS thread count are asserted at start,
query sampling under ``--limit`` is a fixed-seed shuffle, and every rank list breaks ties on
ascending id. The thread pin joined the contract after a measurement: the dense leg is a
brute-force cosine (a BLAS ``gemv``) and the query embedder is a torch forward pass, and both
reassociate their float sums per thread count — enough to change which documents land in the
pool. See :data:`THREADS_REFUSAL` for the numbers.

Refuses rather than fakes: a leg whose export artifacts are absent (the reused M6
MuSiQue/HotpotQA exports carry no ``embedding`` column and no BM25 artifacts) is recorded as
``status: "missing-inputs"`` with the file list, never as a zero.

Ablation legs sit beside the ladder rather than on it: ``dense_content_only`` re-runs the dense
rung against ``corpus-vectors-content-only.parquet`` (see
:func:`benchmark.s5.build_vectors.write_vector_sidecar`) so the report can state what the W13
context header bought, instead of leaving it folded invisibly into every dense number.

Two fusion variants, in two namespaces that never touch. ``--fusion equal`` (the default, and the
one the matrix baseline was run under) fuses every leg's signals at equal RRF weight and writes
``results/``. ``--fusion tuned`` fits per-signal weights on a dev split and reports them on the
disjoint test split (:mod:`benchmark.s5.tuning`), writing ``results-weighted/``. The default is
untouched by the second variant's existence — same code path, same cells, same directory — and
``REPORT.md`` shows both side by side whenever weighted cells exist.

Usage — ``PIN`` is the determinism contract, and the harness refuses without all of it::

    PIN="PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

    env $PIN .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets vidoseek,multihop_rag --legs all --limit 50
    env $PIN .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets multihop_rag --legs dense,dense_content_only --limit 200
    env $PIN .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets all --legs all --fusion tuned
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5 import goldmap  # noqa: E402
from benchmark.s5.cell import (  # noqa: E402
    RESULTS_DIRS,
    Header,
    cell_path,
    ladder_body,
    load_cells,
    missing_inputs_cell,
    not_fusion_tunable_body,
    rescue_body,
    tuned_body,
    write_cell,
)
from benchmark.s5.cell import results_dir as fusion_results_dir  # noqa: E402
from benchmark.s5.corpus import (  # noqa: E402
    CONTENT_ONLY_VECTORS,
    Export,
    load_export,
    read_vector_sidecar,
)
from benchmark.s5.datasets import (  # noqa: E402
    CONTROL_DATASETS,
    DATASET_NAMES,
    DEFAULT_LAYOUT,
    CorpusLayout,
    DatasetSpec,
    dataset_specs,
)
from benchmark.s5.embed import GraniteQueryEmbedder  # noqa: E402
from benchmark.s5.legs import (  # noqa: E402
    ALL_LEG_NAMES,
    Runner,
    build_runner,
    has_embeddings,
    leg_spec,
    missing_inputs,
    vector_source,
)
from benchmark.s5.metrics import CellAccumulator, documents_of  # noqa: E402
from benchmark.s5.report import write_report  # noqa: E402
from benchmark.s5.rescue import (  # noqa: E402
    BUDGET_CURVE,
    GateStats,
    build_pipeline,
    bundle_for,
    headline_budget,
    selector_for,
)
from benchmark.s5.tuning import (  # noqa: E402
    InsufficientSplit,
    ScoredQuery,
    is_tunable_leg,
    run_tuned_leg,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_OUT = REPO_ROOT / "benchmark" / "sota-campaign" / "stages" / "s5-retrieval"
SAMPLE_SEED = 20260809

#: The fusion variants, in declaration order — each owning its own cell directory (the layout
#: itself lives with the cell schema, :data:`benchmark.s5.cell.RESULTS_DIRS`).
FUSIONS = tuple(RESULTS_DIRS)

#: The refusal text when ``PYTHONHASHSEED`` is not pinned. One constant, because it is asserted
#: at the API entry AND printed by the CLI, and the two must say the same thing.
HASHSEED_REFUSAL = (
    "set PYTHONHASHSEED=0 (the hypergraph entry is hash-order sensitive; a run without it is "
    "not reproducible)."
)

#: The BLAS thread-count variables. The dense leg is a brute-force cosine (a BLAS ``gemv``) and
#: the query embedder is a torch forward pass; both reassociate their floating-point sums
#: differently at different thread counts, so the thread count is part of the run's identity in
#: exactly the way the hash seed is.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

#: The thread count every recorded cell was measured at. Pinning to one is not a performance
#: choice — it is the only value that makes two runs comparable without also pinning the machine.
PINNED_THREADS = "1"

#: The refusal text when the thread count is not pinned.
#:
#: This is not theoretical. Running musique/bm25 at ``OMP_NUM_THREADS=4`` instead of ``1``, same
#: code and same corpus, gives 33 of 1,209 queries a different dense pool, 4 a different top-10
#: membership, and exactly one a different gold-hit count — which moves ``recall@10`` from
#: 0.3404 to 0.3406. That single query is the whole delta: 1/(1209x6) = 1.37855e-04, matching the
#: observed difference exactly. An unpinned harness reports thread scheduling as a quality change.
THREADS_REFUSAL = (
    f"pin {' = '.join(THREAD_VARS)} = {PINNED_THREADS} (the dense leg's BLAS gemv and the query "
    "embedder reassociate their float sums per thread count, which changes rankings: measured, "
    "33/1209 musique queries get a different dense pool at 4 threads vs 1)."
)


class MatrixRefused(RuntimeError):
    """The harness refused to run — the message is the reason, verbatim."""


@dataclass(frozen=True)
class Plan:
    """One resolved run request — the harness's whole input.

    ``layout`` is where the inputs are read from (:class:`~benchmark.s5.datasets.CorpusLayout`);
    it defaults to this checkout, so a plan built the old way behaves identically. ``out`` is
    where cells, ``REPORT.md`` and ``metrics.json`` are written, and it is independent of the
    layout on purpose: the campaign stage writes into its own stage directory while reading a
    corpus that lives wherever the checkout put it.
    """

    datasets: tuple[str, ...]
    legs: tuple[str, ...]
    limit: int | None
    out: Path
    force: bool
    resolution_only: bool = False
    fusion: str = "equal"
    layout: CorpusLayout = DEFAULT_LAYOUT

    @property
    def results_dir(self) -> Path:
        """The cell directory this fusion variant owns."""
        return fusion_results_dir(self.out, self.fusion)

    @property
    def specs(self) -> dict[str, DatasetSpec]:
        """This plan's dataset table, resolved against its own layout."""
        return dataset_specs(self.layout)


@dataclass(frozen=True)
class MatrixOutcome:
    """What one :func:`run` did — the record a caller reports or gates on.

    ``cells`` is every planned cell that EXISTS on disk afterwards (the resume view: cells a
    previous run left count), ``written`` is only what this invocation produced. A stage's
    done-probe reads the filesystem itself rather than trusting this struct — but the two must
    agree, and a run that returns cleanly with ``len(cells) < len(planned_cells(plan))`` is
    precisely the "exited 0, produced nothing" shape the campaign contract blocks on.
    """

    plan: Plan
    cells: tuple[Path, ...]
    written: tuple[Path, ...]
    report: Path
    metrics: Path


def require_deterministic() -> None:
    """Refuse unless the run is reproducible — at the API entry, not only in ``main``.

    Two variables, not one. ``PYTHONHASHSEED`` was always asserted; the BLAS thread count was
    not, and it is the one that silently moved numbers between the two full-matrix sweeps. Both
    are checked here so a cell cannot be written by a process that was never pinned.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise MatrixRefused(HASHSEED_REFUSAL)
    unpinned = [var for var in THREAD_VARS if os.environ.get(var) != PINNED_THREADS]
    if unpinned:
        raise MatrixRefused(f"{THREADS_REFUSAL} Unpinned: {', '.join(unpinned)}.")


def planned_cells(plan: Plan) -> tuple[Path, ...]:
    """Every ``<dataset>__<leg>.json`` this plan is responsible for, in matrix order.

    The done-probe's definition of "finished", derived from the plan rather than restated
    beside it — the campaign stage asks this function instead of rebuilding the file naming.
    A ``--resolution-only`` plan writes no cells and owns none.
    """
    if plan.resolution_only:
        return ()
    return tuple(
        cell_path(plan.results_dir, dataset, leg) for dataset in plan.datasets for leg in plan.legs
    )


def parse_args(argv: Sequence[str] | None = None) -> Plan:
    """Parse the CLI into a :class:`Plan` (``all`` expands to every dataset / every leg).

    One adapter over :func:`run`, nothing more: it resolves flags into the same struct the
    campaign stage builds from ``campaign.yaml``, and every rule that must hold for BOTH
    callers (the hash-seed refusal, the sample seed, the cell naming) lives past this function.
    """
    parser = argparse.ArgumentParser(description="S5 retrieval feature-matrix harness")
    parser.add_argument("--datasets", default="all", help="comma list, or 'all'")
    parser.add_argument(
        "--legs",
        default="all",
        help=f"comma list of {','.join(ALL_LEG_NAMES)}, or 'all' (which includes the "
        "ablation legs — a leg whose inputs are absent is recorded as missing-inputs, not run)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="query cap per dataset (smoke runs)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--force", action="store_true", help="re-run cells that already have results"
    )
    parser.add_argument(
        "--fusion",
        default="equal",
        choices=FUSIONS,
        help=(
            "how a leg's signals are combined. 'equal' (default) is the untuned RRF baseline "
            "and writes results/; 'tuned' fits per-signal weights on a dev split, reports them "
            "on the disjoint test split, and writes results-weighted/ — a separate namespace, so "
            "the baseline is never touched."
        ),
    )
    parser.add_argument(
        "--resolution-only",
        action="store_true",
        help=(
            "measure and write resolution/<dataset>.json only — no query embedding, no legs. "
            "This is the gold-mapping health check on its own, cheap enough to run over every "
            "dataset's FULL query set."
        ),
    )
    args = parser.parse_args(argv)
    datasets = (
        tuple(DATASET_NAMES)
        if args.datasets == "all"
        else tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    )
    unknown = [d for d in datasets if d not in DATASET_NAMES]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; known: {sorted(DATASET_NAMES)}")
    legs = (
        ALL_LEG_NAMES
        if args.legs == "all"
        else tuple(leg.strip() for leg in args.legs.split(",") if leg.strip())
    )
    bad = [leg for leg in legs if leg not in ALL_LEG_NAMES]
    if bad:
        parser.error(f"unknown leg(s) {bad}; known: {list(ALL_LEG_NAMES)}")
    return Plan(
        datasets=datasets,
        legs=tuple(leg for leg in ALL_LEG_NAMES if leg in legs),
        limit=args.limit,
        out=args.out,
        force=bool(args.force),
        resolution_only=bool(args.resolution_only),
        fusion=str(args.fusion),
    )


def sample(items: list[goldmap.GoldQuery], limit: int | None) -> list[goldmap.GoldQuery]:
    """A fixed-seed sample of ``limit`` queries — reproducible across runs and machines."""
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(SAMPLE_SEED)
    return sorted(rng.sample(items, limit), key=lambda q: q.query_id)


def measure_resolution(
    name: str, plan: Plan, *, with_embeddings: bool, with_graph_features: bool = False
) -> tuple[goldmap.ResolutionReport, list[goldmap.ResolvedGold], Export]:
    """Resolve one dataset's full (or sampled) gold onto its export and write its resolution JSON.

    The page-span reconstruction is limited to the documents a PAGE reference actually names —
    UDA's 393 paper_tab queries are document-level gold, and reassembling their 307 papers to
    answer a question no reference asks would dominate the measurement.
    """
    spec = plan.specs[name]
    queries = sample(spec.queries(), plan.limit)
    page_docs = frozenset(
        Path(ref.file_name).stem
        for query in queries
        for ref in query.refs
        if ref.kind is goldmap.RefKind.PAGE
    )
    print(f"[{name}] loading export ({len(queries)} queries) …", flush=True)
    export = load_export(
        spec.export_dir,
        name,
        s2_root=spec.s2_root,
        with_embeddings=with_embeddings,
        page_docs=page_docs,
        with_graph_features=with_graph_features,
    )
    resolved = [goldmap.resolve(q, export) for q in queries]
    resolution = goldmap.report(name, resolved)
    print(
        f"[{name}] gold resolution {resolution.rate:.1%} "
        f"({resolution.refs_resolved}/{resolution.refs_total} refs; "
        f"{resolution.refs_page} page / {resolution.refs_document} document; "
        f"{resolution.queries_excluded} queries excluded {dict(resolution.exclusions)}; "
        f"{resolution.queries_usable}/{resolution.queries_total} queries usable)",
        flush=True,
    )
    (plan.out / "resolution").mkdir(parents=True, exist_ok=True)
    (plan.out / "resolution" / f"{name}.json").write_text(
        json.dumps(resolution.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return resolution, resolved, export


def run_dataset(name: str, plan: Plan) -> list[Path]:
    """Run every requested leg for one dataset; write each cell as it completes.

    Returns the cell files THIS call wrote (refusals included — a recorded missing-inputs cell
    is a written cell), so the caller can report what a resumed run actually did instead of
    re-deriving it from timestamps.
    """
    spec = plan.specs[name]
    results_dir = plan.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if plan.resolution_only:
        measure_resolution(name, plan, with_embeddings=False)
        return written

    pending = [
        leg for leg in plan.legs if plan.force or not cell_path(results_dir, name, leg).is_file()
    ]
    blocked = {leg: missing_inputs(spec.export_dir, leg) for leg in pending}
    for leg, missing in blocked.items():
        if missing:
            write_cell(results_dir, name, leg, missing_inputs_cell(name, leg, missing))
            written.append(cell_path(results_dir, name, leg))
    runnable = [leg for leg in pending if not blocked[leg]]
    # The embedding COLUMN is a per-leg input, not a per-dataset one: an ablation leg reads its
    # vectors from the sidecar and can run on an export whose corpus table has no column at all.
    from_corpus = [leg for leg in runnable if vector_source(leg) == "corpus"]
    if from_corpus and not has_embeddings(spec.export_dir):
        for leg in from_corpus:
            write_cell(
                results_dir,
                name,
                leg,
                missing_inputs_cell(name, leg, ["corpus.parquet:embedding"]),
            )
            written.append(cell_path(results_dir, name, leg))
        runnable = [leg for leg in runnable if vector_source(leg) != "corpus"]
        from_corpus = []

    if not runnable:
        return written

    resolution, resolved, export = measure_resolution(
        name,
        plan,
        with_embeddings=bool(from_corpus),
        # Only the rescue family reads the denormalised KG projection, and it is three more list
        # columns per row over corpora of up to 220k chunks — so it is loaded when it is used.
        with_graph_features=any(leg_spec(leg).needs_graph_features for leg in runnable),
    )
    usable = [r for r in resolved if r.is_usable]
    if not usable:
        return written

    # Loaded here rather than inside `build_runner`, because this is where the corpus row ids the
    # sidecar must line up with are known; `read_vector_sidecar` raises if it does not.
    content_only = None
    if any(vector_source(leg) == "sidecar" for leg in runnable):
        print(f"[{name}] loading {CONTENT_ONLY_VECTORS} …", flush=True)
        content_only = read_vector_sidecar(
            spec.export_dir / CONTENT_ONLY_VECTORS,
            expected_ids=[row.record_id for row in export.rows],
        )[1]

    print(f"[{name}] embedding {len(usable)} queries …", flush=True)
    vectors = GraniteQueryEmbedder().encode([r.question for r in usable])
    runner = build_runner(export, runnable, content_only_vectors=content_only)

    scored = [
        ScoredQuery(
            query_id=item.query_id,
            question=item.question,
            vector=vectors[position],
            gold=item.chunk_ids,
            chain=item.chain,
        )
        for position, item in enumerate(usable)
    ]
    for leg in runnable:
        header = Header(
            dataset=name,
            leg=leg,
            fusion=plan.fusion,
            vector_source=vector_source(leg),
            control=name in CONTROL_DATASETS,
            limit=plan.limit,
            resolution=resolution.to_dict(),
        )
        if leg_spec(leg).cell_kind == "rescue":
            body = _rescue_cell(runner, export, leg, scored, fusion=plan.fusion)
        elif plan.fusion == "equal":
            body = _equal_cell(runner, export, leg, scored)
        else:
            body = _tuned_cell(runner, leg, scored)
        payload = header.merge(body)
        write_cell(results_dir, name, leg, payload)
        written.append(cell_path(results_dir, name, leg))
        print(f"[{name}/{leg}] {json.dumps(payload.get('metrics'))}", flush=True)
    return written


def _equal_cell(
    runner: Runner, export: Export, leg: str, scored: Sequence[ScoredQuery]
) -> dict[str, object]:
    """The baseline cell: every usable query, every signal at equal RRF weight."""
    cell = CellAccumulator()
    for query in scored:
        outcome = runner.run(query.question, query.vector, leg)
        _score(
            cell,
            export,
            query,
            outcome.ranked,
            outcome.latency_ms,
            tokens_packed=outcome.tokens_packed,
            budget=outcome.budget,
        )
    return ladder_body(cell.summary())


def _score(
    cell: CellAccumulator,
    export: Export,
    query: ScoredQuery,
    ranked: Sequence[str],
    latency_ms: float,
    *,
    tokens_packed: int | None = None,
    budget: int | None = None,
) -> None:
    """Score one query into ``cell`` at BOTH granularities — chunk-level and document-level.

    The document view is derived here rather than by the legs, because it is a property of the
    export (which document a chunk came from) and not of the retriever, so every leg's number is
    deduplicated by exactly the same map.
    """
    cell.add(
        ranked,
        query.gold,
        query.chain,
        latency_ms,
        tokens_packed=tokens_packed,
        budget=budget,
        ranked_documents=documents_of(ranked, export.document_of),
        gold_documents=frozenset(documents_of(sorted(query.gold), export.document_of)),
    )


def _rescue_cell(
    runner: Runner,
    export: Export,
    leg: str,
    scored: Sequence[ScoredQuery],
    *,
    fusion: str,
) -> dict[str, object]:
    """A graph-rescue cell: the leg at its headline budget, plus its recall-over-budget curve.

    The shipped :class:`~latence_retrieval.rescue.GraphRescueRetrieval` IS the pipeline (see
    :mod:`benchmark.s5.rescue`): each query is
    :meth:`~latence_retrieval.rescue.GraphRescueRetrieval.retrieve`\\ d **once** and then
    :meth:`~latence_retrieval.rescue.GraphRescueRetrieval.select`\\ ed at every budget in
    :data:`~benchmark.s5.rescue.BUDGET_CURVE`, so the curve varies the budget and nothing else.
    ``graph_rescue`` has no selector, so its curve is a single entry: repeating one ranking under
    seven budget labels would imply a measurement that was not made.

    Latency is measured the way the cell consumes it — the retrieval paid once, plus this budget's
    own selection — so a curve point's number is what that operating point actually costs.

    The ``tuned`` fusion variant does not apply here and says so instead of producing a number: the
    rescue family fuses only ``dense``+``bm25``, and its graph lanes are appended rather than
    weighted, so there is no weight vector whose fitting could be reported.
    """
    if fusion != "equal":
        return not_fusion_tunable_body()
    selector = selector_for(leg)
    headline = headline_budget(leg)
    budgets = (headline,) if selector == "none" else BUDGET_CURVE
    cells = {budget: CellAccumulator() for budget in budgets}
    gate = GateStats()
    pipeline = build_pipeline(runner, export)
    for query in scored:
        started = time.perf_counter()
        retrieval = pipeline.retrieve(bundle_for(runner, query.question, query.vector))
        retrieve_ms = (time.perf_counter() - started) * 1000.0
        gate.add(retrieval.gate)
        for budget, cell in cells.items():
            started = time.perf_counter()
            result = pipeline.select(query.question, retrieval, selector=selector, budget=budget)
            _score(
                cell,
                export,
                query,
                [candidate.id for candidate in result.candidates],
                retrieve_ms + (time.perf_counter() - started) * 1000.0,
                tokens_packed=result.tokens_packed,
                budget=result.budget,
            )
    return rescue_body(
        metrics=cells[headline].summary(),
        budget_curve={str(budget): cell.summary() for budget, cell in cells.items()},
        gate=gate.summary(),
        select_budget=headline,
    )


def _tuned_cell(runner: Runner, leg: str, scored: Sequence[ScoredQuery]) -> dict[str, object]:
    """The weighted cell: weights fitted on dev, BOTH variants scored on the disjoint test half.

    ``metrics`` is the tuned variant on test and ``metrics_equal_weight_on_test`` is the
    equal-weight variant on the *same* test queries from the *same* retrieved lists — so the
    report's delta is a difference in fusion, not in which queries were scored. The full-set
    equal-weight cell in ``results/`` is a different query set and is never subtracted from these.
    """
    outcome = run_tuned_leg(runner, leg, scored)
    if isinstance(outcome, InsufficientSplit):
        return outcome.to_dict()
    return tuned_body(
        metrics=outcome.tuned,
        metrics_equal_weight_on_test=outcome.equal,
        tuning=outcome.tuning.to_dict(),
        split=outcome.split,
        tunable=is_tunable_leg(leg),
    )


def run(plan: Plan) -> MatrixOutcome:
    """THE harness entry: run ``plan``, rewrite ``REPORT.md`` + ``metrics.json``, report back.

    Both the CLI (:func:`main`) and the campaign stage come through here, which is why the
    determinism refusal is the first thing it does: the guarantee "every committed cell was
    produced under a pinned hash seed" belongs to the harness, not to one of its two callers.
    """
    require_deterministic()
    written: list[Path] = []
    for name in plan.datasets:
        written.extend(run_dataset(name, plan))
    # Both namespaces are read back from disk regardless of which one this invocation wrote, so a
    # plain `--fusion equal` run still renders any weighted cells a previous run left, and a
    # `--fusion tuned` run still renders the baseline it must be compared against. Cells are the
    # source of truth; the report is a view over them.
    equal = _matrix(plan, "equal")
    # Only datasets that actually have weighted cells: an empty weighted matrix must render as
    # "the variant was not run", not as a table of blanks implying it found nothing.
    weighted = {name: cells for name, cells in _matrix(plan, "tuned").items() if cells}
    write_report(plan.out, equal, legs=plan.legs, limit=plan.limit, weighted=weighted)
    return MatrixOutcome(
        plan=plan,
        cells=tuple(path for path in planned_cells(plan) if path.is_file()),
        written=tuple(written),
        report=plan.out / "REPORT.md",
        metrics=plan.out / "metrics.json",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI adapter: flags → :class:`Plan` → :func:`run` → one line and an exit code."""
    try:
        require_deterministic()
    except MatrixRefused as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    outcome = run(parse_args(argv))
    print(f"wrote {outcome.report} and {outcome.metrics}")
    return 0


def _matrix(plan: Plan, fusion: str) -> dict[str, dict[str, object]]:
    """One fusion variant's cells for every requested dataset (a dataset with none maps to {})."""
    directory = fusion_results_dir(plan.out, fusion)
    return {name: load_cells(directory, name, plan.legs) for name in plan.datasets}


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

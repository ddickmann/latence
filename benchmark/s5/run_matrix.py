# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``run_matrix.py`` — the S5 feature-matrix CLI: (dataset × leg) → one comparable table.

Resumable by construction: every (dataset, leg) cell is written to its own JSON file under
``benchmark/sota-campaign/stages/s5-retrieval/results/`` the moment it finishes, and a re-run
skips cells whose file already exists (``--force`` re-runs them). A matrix that takes hours can
therefore be interrupted, extended one dataset at a time, or resumed after a crash without
losing or silently re-mixing work.

Deterministic: ``PYTHONHASHSEED=0`` is asserted at start (the hypergraph entry and several
dict-ordered paths are hash-order sensitive), query sampling under ``--limit`` is a fixed-seed
shuffle, and every rank list breaks ties on ascending id.

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

Usage::

    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets vidoseek,multihop_rag --legs all --limit 50
    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets multihop_rag --legs dense,dense_content_only --limit 200
    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets all --legs all --fusion tuned
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5 import goldmap  # noqa: E402
from benchmark.s5.cell import (  # noqa: E402
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
from benchmark.s5.corpus import (  # noqa: E402
    CONTENT_ONLY_VECTORS,
    Export,
    load_export,
    read_vector_sidecar,
)
from benchmark.s5.datasets import CONTROL_DATASETS, DATASETS  # noqa: E402
from benchmark.s5.embed import GraniteQueryEmbedder  # noqa: E402
from benchmark.s5.legs import (  # noqa: E402
    ABLATION_LEG_NAMES,
    ALL_LEG_NAMES,
    RESCUE_LEG_NAMES,
    Runner,
    build_runner,
    has_embeddings,
    missing_inputs,
    vector_source,
)
from benchmark.s5.metrics import CellAccumulator, documents_of  # noqa: E402
from benchmark.s5.report import write_report  # noqa: E402
from benchmark.s5.rescue import (  # noqa: E402
    BUDGET_CURVE,
    COE_SELECT_BUDGET,
    SELECT_BUDGET,
    GateStats,
    RescuePipeline,
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

#: Where each fusion variant's cells live. Separate directories, never separate *keys* in one
#: directory: the equal-weight matrix is a long-running job, and a variant must not be able to
#: overwrite, reorder or shadow one of its cells even by accident.
RESULTS_DIRS = {"equal": "results", "tuned": "results-weighted"}
FUSIONS = tuple(RESULTS_DIRS)


@dataclass(frozen=True)
class Plan:
    """One resolved run request."""

    datasets: tuple[str, ...]
    legs: tuple[str, ...]
    limit: int | None
    out: Path
    force: bool
    resolution_only: bool = False
    fusion: str = "equal"

    @property
    def results_dir(self) -> Path:
        """The cell directory this fusion variant owns."""
        return self.out / RESULTS_DIRS[self.fusion]


def parse_args(argv: Sequence[str] | None = None) -> Plan:
    """Parse the CLI into a :class:`Plan` (``all`` expands to every dataset / every leg)."""
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
    datasets = tuple(DATASETS) if args.datasets == "all" else tuple(
        d.strip() for d in args.datasets.split(",") if d.strip()
    )
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; known: {sorted(DATASETS)}")
    legs = ALL_LEG_NAMES if args.legs == "all" else tuple(
        leg.strip() for leg in args.legs.split(",") if leg.strip()
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
    spec = DATASETS[name]
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


def run_dataset(name: str, plan: Plan) -> None:
    """Run every requested leg for one dataset; write each cell as it completes."""
    spec = DATASETS[name]
    results_dir = plan.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if plan.resolution_only:
        measure_resolution(name, plan, with_embeddings=False)
        return

    pending = [
        leg
        for leg in plan.legs
        if plan.force or not cell_path(results_dir, name, leg).is_file()
    ]
    blocked = {leg: missing_inputs(spec.export_dir, leg) for leg in pending}
    for leg, missing in blocked.items():
        if missing:
            write_cell(results_dir, name, leg, missing_inputs_cell(name, leg, missing))
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
        runnable = [leg for leg in runnable if vector_source(leg) != "corpus"]
        from_corpus = []

    if not runnable:
        return

    resolution, resolved, export = measure_resolution(
        name,
        plan,
        with_embeddings=bool(from_corpus),
        # Only the rescue family reads the denormalised KG projection, and it is three more list
        # columns per row over corpora of up to 220k chunks — so it is loaded when it is used.
        with_graph_features=any(leg in RESCUE_LEG_NAMES for leg in runnable),
    )
    usable = [r for r in resolved if r.is_usable]
    if not usable:
        return

    # Loaded here rather than inside `build_runner`, because this is where the corpus row ids the
    # sidecar must line up with are known; `read_vector_sidecar` raises if it does not.
    content_only = None
    if any(leg in ABLATION_LEG_NAMES for leg in runnable):
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
        if leg in RESCUE_LEG_NAMES:
            body = _rescue_cell(runner, export, leg, scored, fusion=plan.fusion)
        elif plan.fusion == "equal":
            body = _equal_cell(runner, export, leg, scored)
        else:
            body = _tuned_cell(runner, leg, scored)
        payload = header.merge(body)
        write_cell(results_dir, name, leg, payload)
        print(f"[{name}/{leg}] {json.dumps(payload.get('metrics'))}", flush=True)


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
    """A graph-rescue cell: the leg at :data:`SELECT_BUDGET`, plus its recall-over-budget curve.

    Each query is retrieved **once** and then selected at every budget in
    :data:`~benchmark.s5.rescue.BUDGET_CURVE`, so the curve varies the budget and nothing else.
    ``graph_rescue`` has no selector, so its curve is a single entry: repeating one ranking under
    five budget labels would imply a measurement that was not made.

    The ``tuned`` fusion variant does not apply here and says so instead of producing a number: the
    rescue family fuses only ``dense``+``bm25``, and its graph lanes are appended rather than
    weighted, so there is no weight vector whose fitting could be reported.
    """
    if fusion != "equal":
        return not_fusion_tunable_body()
    # Each selector's headline sits at its own measured operating point: the knapsack saturates
    # ~5.5k so 10k is generous, while CoE keeps buying recall through 16k (operator ruling).
    headline = COE_SELECT_BUDGET if leg == "graph_rescue_coe" else SELECT_BUDGET
    budgets = (headline,) if leg == "graph_rescue" else BUDGET_CURVE
    cells = {budget: CellAccumulator() for budget in budgets}
    gate = GateStats()
    pipeline = RescuePipeline(runner=runner)
    for query in scored:
        retrieved = pipeline.retrieve(query.question, query.vector)
        gate.add(retrieved.gate)
        for budget, cell in cells.items():
            outcome = pipeline.select(query.question, retrieved, leg, budget)
            _score(
                cell,
                export,
                query,
                outcome.ranked,
                outcome.latency_ms,
                tokens_packed=outcome.tokens_packed,
                budget=outcome.budget,
            )
    return rescue_body(
        metrics=cells[headline].summary(),
        budget_curve={str(budget): cell.summary() for budget, cell in cells.items()},
        gate=gate.summary(),
        select_budget=headline,
    )


def _tuned_cell(
    runner: Runner, leg: str, scored: Sequence[ScoredQuery]
) -> dict[str, object]:
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested matrix, then rewrite ``REPORT.md`` + ``metrics.json`` from the cells."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "refusing to run: set PYTHONHASHSEED=0 (the hypergraph entry is hash-order "
            "sensitive; a run without it is not reproducible).",
            file=sys.stderr,
        )
        return 2
    plan = parse_args(argv)
    for name in plan.datasets:
        run_dataset(name, plan)
    # Both namespaces are read back from disk regardless of which one this invocation wrote, so a
    # plain `--fusion equal` run still renders any weighted cells a previous run left, and a
    # `--fusion tuned` run still renders the baseline it must be compared against. Cells are the
    # source of truth; the report is a view over them.
    equal = _matrix(plan, "equal")
    # Only datasets that actually have weighted cells: an empty weighted matrix must render as
    # "the variant was not run", not as a table of blanks implying it found nothing.
    weighted = {name: cells for name, cells in _matrix(plan, "tuned").items() if cells}
    write_report(plan.out, equal, legs=plan.legs, limit=plan.limit, weighted=weighted)
    print(f"wrote {plan.out / 'REPORT.md'} and {plan.out / 'metrics.json'}")
    return 0


def _matrix(plan: Plan, fusion: str) -> dict[str, dict[str, object]]:
    """One fusion variant's cells for every requested dataset (a dataset with none maps to {})."""
    directory = plan.out / RESULTS_DIRS[fusion]
    return {name: load_cells(directory, name, plan.legs) for name in plan.datasets}


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

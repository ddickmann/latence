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

Usage::

    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.run_matrix \\
        --datasets vidoseek,multihop_rag --legs all --limit 50
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
from benchmark.s5.corpus import load_export  # noqa: E402
from benchmark.s5.datasets import CONTROL_DATASETS, DATASETS  # noqa: E402
from benchmark.s5.embed import GraniteQueryEmbedder  # noqa: E402
from benchmark.s5.legs import LEG_NAMES, build_runner, has_embeddings, missing_inputs  # noqa: E402
from benchmark.s5.metrics import CellAccumulator  # noqa: E402
from benchmark.s5.report import write_report  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_OUT = REPO_ROOT / "benchmark" / "sota-campaign" / "stages" / "s5-retrieval"
SAMPLE_SEED = 20260809


@dataclass(frozen=True)
class Plan:
    """One resolved run request."""

    datasets: tuple[str, ...]
    legs: tuple[str, ...]
    limit: int | None
    out: Path
    force: bool


def parse_args(argv: Sequence[str] | None = None) -> Plan:
    """Parse the CLI into a :class:`Plan` (``all`` expands to every dataset / every leg)."""
    parser = argparse.ArgumentParser(description="S5 retrieval feature-matrix harness")
    parser.add_argument("--datasets", default="all", help="comma list, or 'all'")
    parser.add_argument(
        "--legs", default="all", help=f"comma list of {','.join(LEG_NAMES)}, or 'all'"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="query cap per dataset (smoke runs)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--force", action="store_true", help="re-run cells that already have results"
    )
    args = parser.parse_args(argv)
    datasets = tuple(DATASETS) if args.datasets == "all" else tuple(
        d.strip() for d in args.datasets.split(",") if d.strip()
    )
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; known: {sorted(DATASETS)}")
    legs = LEG_NAMES if args.legs == "all" else tuple(
        leg.strip() for leg in args.legs.split(",") if leg.strip()
    )
    bad = [leg for leg in legs if leg not in LEG_NAMES]
    if bad:
        parser.error(f"unknown leg(s) {bad}; known: {list(LEG_NAMES)}")
    return Plan(
        datasets=datasets,
        legs=tuple(leg for leg in LEG_NAMES if leg in legs),
        limit=args.limit,
        out=args.out,
        force=bool(args.force),
    )


def sample(items: list[goldmap.GoldQuery], limit: int | None) -> list[goldmap.GoldQuery]:
    """A fixed-seed sample of ``limit`` queries — reproducible across runs and machines."""
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(SAMPLE_SEED)
    return sorted(rng.sample(items, limit), key=lambda q: q.query_id)


def run_dataset(name: str, plan: Plan) -> dict[str, object]:
    """Run every requested leg for one dataset; write each cell as it completes."""
    spec = DATASETS[name]
    results_dir = plan.out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        leg
        for leg in plan.legs
        if plan.force or not (results_dir / f"{name}__{leg}.json").is_file()
    ]
    blocked = {leg: missing_inputs(spec.export_dir, leg) for leg in pending}
    if not spec.export_dir.is_file() and (spec.export_dir / "corpus.parquet").is_file():
        pass
    for leg, missing in blocked.items():
        if missing:
            _write_cell(
                results_dir,
                name,
                leg,
                {"status": "missing-inputs", "missing": missing, "dataset": name, "leg": leg},
            )
    runnable = [leg for leg in pending if not blocked[leg]]
    if runnable and not has_embeddings(spec.export_dir):
        for leg in runnable:
            _write_cell(
                results_dir,
                name,
                leg,
                {
                    "status": "missing-inputs",
                    "missing": ["corpus.parquet:embedding"],
                    "dataset": name,
                    "leg": leg,
                },
            )
        runnable = []

    if not runnable:
        return _collect(results_dir, name, plan.legs)

    queries = sample(spec.queries(), plan.limit)
    # Reconstruct page spans only for the documents this run's gold actually references.
    page_docs = frozenset(
        Path(ref.file_name).stem for query in queries for ref in query.refs
    )
    print(f"[{name}] loading export ({len(queries)} queries) …", flush=True)
    export = load_export(spec.export_dir, name, s2_root=spec.s2_root, page_docs=page_docs)
    resolved = [goldmap.resolve(q, export) for q in queries]
    resolution = goldmap.report(name, resolved)
    usable = [r for r in resolved if r.is_usable]
    print(
        f"[{name}] gold resolution {resolution.rate:.1%} "
        f"({resolution.refs_resolved}/{resolution.refs_total} refs, "
        f"{resolution.queries_usable}/{resolution.queries_total} queries usable)",
        flush=True,
    )
    (plan.out / "resolution").mkdir(parents=True, exist_ok=True)
    (plan.out / "resolution" / f"{name}.json").write_text(
        json.dumps(resolution.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if not usable:
        return _collect(results_dir, name, plan.legs)

    print(f"[{name}] embedding {len(usable)} queries …", flush=True)
    vectors = GraniteQueryEmbedder().encode([r.question for r in usable])
    runner = build_runner(export, runnable)

    for leg in runnable:
        cell = CellAccumulator()
        for position, item in enumerate(usable):
            outcome = runner.run(item.question, vectors[position], leg)
            cell.add(
                outcome.ranked,
                item.chunk_ids,
                item.chain,
                outcome.latency_ms,
                tokens_packed=outcome.tokens_packed,
                budget=outcome.budget,
            )
        payload: dict[str, object] = {
            "status": "ok",
            "dataset": name,
            "leg": leg,
            "control": name in CONTROL_DATASETS,
            "limit": plan.limit,
            "resolution": resolution.to_dict(),
            "metrics": cell.summary(),
        }
        _write_cell(results_dir, name, leg, payload)
        print(f"[{name}/{leg}] {json.dumps(cell.summary())}", flush=True)
    return _collect(results_dir, name, plan.legs)


def _write_cell(results_dir: Path, dataset: str, leg: str, payload: dict[str, object]) -> None:
    path = results_dir / f"{dataset}__{leg}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect(results_dir: Path, dataset: str, legs: Sequence[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for leg in legs:
        path = results_dir / f"{dataset}__{leg}.json"
        if path.is_file():
            out[leg] = json.loads(path.read_text(encoding="utf-8"))
    return out


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
    matrix: dict[str, dict[str, object]] = {}
    for name in plan.datasets:
        matrix[name] = run_dataset(name, plan)
    write_report(plan.out, matrix, legs=plan.legs, limit=plan.limit)
    print(f"wrote {plan.out / 'REPORT.md'} and {plan.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

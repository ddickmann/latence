"""``latence-benchmark`` — the one command an outsider runs.

    latence-benchmark run --suite parse

Acquires the pinned dataset (license-gated first), runs the Stage under the pinned stack
config, scores it, and writes both published artifacts. ``--dataset-root`` runs the identical
pipeline against an already-downloaded copy, which is what CI and air-gapped hosts use.

Stdlib ``argparse`` deliberately: the harness must be runnable straight from a clone with the
framework installed and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from latence_benchmark.datasets import (
    DatasetUnavailableError,
    HuggingFaceDatasetSource,
    LocalDatasetSource,
)
from latence_benchmark.licensing import LicenseGateError, require_usable
from latence_benchmark.results import write_results
from latence_benchmark.suite import get_suite, suite_names
from latence_benchmark.unittests import BenchmarkDataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_benchmark.datasets import DatasetSource

DEFAULT_MARKDOWN = Path("docs/BENCHMARKS.md")
DEFAULT_JSON = Path("docs/benchmarks/results.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latence-benchmark",
        description="Run the Latence framework's reproducible public benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark suite end-to-end and publish the results.")
    run.add_argument(
        "--suite",
        default="parse",
        choices=list(suite_names()),
        help="Which Stage benchmark suite to run (default: parse).",
    )
    run.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Use an already-downloaded dataset at this path instead of fetching it.",
    )
    run.add_argument(
        "--provider",
        default=None,
        help="Re-pin the Stage's Provider (e.g. parser.lighton). Recorded in the results.",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap documents scored per split. Produces a clearly-labelled SUBSET result.",
    )
    run.add_argument(
        "--accept-non-commercial",
        action="store_true",
        help="Acknowledge a research-only dataset license. Never lifts an UNVERIFIED license.",
    )
    run.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    run.add_argument("--json-out", type=Path, default=DEFAULT_JSON)

    sub.add_parser("list", help="List the registered benchmark suites.")
    return parser


def _source(dataset_root: Path | None) -> DatasetSource:
    return LocalDatasetSource(dataset_root) if dataset_root else HuggingFaceDatasetSource()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        for name in suite_names():
            suite = get_suite(name)
            print(f"{name}\tstage={suite.stage}\tdataset={suite.dataset.name}")
        return 0

    from latence_benchmark.parse_suite import ParseSuite, ProviderUnavailableError, parse_suite

    suite = get_suite(args.suite)
    if args.provider and isinstance(suite, ParseSuite):
        suite = parse_suite(provider=args.provider)

    try:
        # The license gate runs BEFORE acquisition — restricted bytes never land on disk.
        require_usable(
            suite.dataset.name,
            suite.dataset.license,
            accept_non_commercial=args.accept_non_commercial,
        )
        root = _source(args.dataset_root).materialise(suite.dataset)
        result = suite.run(root, limit=args.limit)
    except (LicenseGateError, DatasetUnavailableError, BenchmarkDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    generated_on = datetime.now(UTC).date().isoformat()
    write_results(
        [result],
        markdown_path=args.markdown_out,
        json_path=args.json_out,
        generated_on=generated_on,
    )
    overall = result.overall
    rate = overall.pass_rate
    rate_text = "n/a" if rate is None else f"{rate * 100:.1f}%"
    print(
        f"{result.suite}: {overall.passed}/{overall.scored} scored unit tests passed "
        f"({rate_text}); {overall.unsupported} unsupported. "
        f"Wrote {args.markdown_out} and {args.json_out}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

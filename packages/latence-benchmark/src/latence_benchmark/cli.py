"""``latence-benchmark`` — the one command an outsider runs.

    latence-benchmark run --suite parse
    latence-benchmark run --suite all      # every registered suite, one results page

Acquires each suite's pinned dataset (license-gated first), runs the Stage under the pinned
stack config, scores it, and writes both published artifacts. ``--dataset-root`` runs the
identical pipeline against an already-downloaded copy, which is what CI and air-gapped hosts
use; with ``--suite all`` it may be repeated as ``--dataset-root <dataset>=<path>`` because the
suites do not share one corpus.

No single host usually has every dataset, so a run covering a subset of the suites republishes
the previously published measurements for the Stages it did not score instead of deleting them
(``--no-carry-forward`` opts out). Each measurement records the date it was produced.

Stdlib ``argparse`` deliberately: the harness must be runnable straight from a clone with the
framework installed and nothing else.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from latence_benchmark.datasets import (
    DatasetUnavailableError,
    HuggingFaceDatasetSource,
    LocalDatasetSource,
    ShippedCorpusSource,
)
from latence_benchmark.licensing import LicenseGateError, require_usable
from latence_benchmark.providers import ProviderUnavailableError
from latence_benchmark.results import PublishedResultsError, carry_forward, write_results
from latence_benchmark.suite import StackPin, get_suite, suite_names
from latence_benchmark.unittests import BenchmarkDataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_benchmark.datasets import DatasetSource, DatasetSpec
    from latence_benchmark.suite import BenchmarkSuite, SuiteResult

DEFAULT_MARKDOWN = Path("docs/BENCHMARKS.md")
DEFAULT_JSON = Path("docs/benchmarks/results.json")

#: ``--suite`` value meaning "every registered suite, onto one results page".
ALL = "all"


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
        choices=[*suite_names(), ALL],
        help="Which Stage benchmark suite to run, or 'all' (default: parse).",
    )
    run.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="[DATASET=]PATH",
        help=(
            "Use an already-downloaded dataset instead of fetching it. Repeat as "
            "'<dataset>=<path>' when running several suites; a bare path applies to every "
            "suite that does not ship its own corpus."
        ),
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
    run.add_argument(
        "--no-carry-forward",
        dest="carry_forward",
        action="store_false",
        help=(
            "Write a page containing ONLY the suites this run scored, dropping the previously "
            "published measurements for every other Stage. Off by default: a partial run "
            "republishes the other Stages rather than deleting them."
        ),
    )
    run.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    run.add_argument("--json-out", type=Path, default=DEFAULT_JSON)

    sub.add_parser("list", help="List the registered benchmark suites.")
    return parser


def _dataset_roots(values: Sequence[str]) -> dict[str | None, Path]:
    """Parse ``--dataset-root`` values into ``{dataset name or None (default): path}``."""
    roots: dict[str | None, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        roots[name if separator else None] = Path(path if separator else value)
    return roots


def _source(spec: DatasetSpec, roots: dict[str | None, Path]) -> DatasetSource:
    """Pick the acquisition adapter for one dataset."""
    if spec.revision_kind == "content-digest":
        # Shipped in this package; there is nothing to download and nothing to point at.
        return ShippedCorpusSource()
    root = roots.get(spec.name, roots.get(None))
    return LocalDatasetSource(root) if root is not None else HuggingFaceDatasetSource()


def _repin(suite: BenchmarkSuite, provider: str) -> BenchmarkSuite:
    """Re-pin a suite's Provider, keeping everything else identical.

    Works for every suite because a suite is a frozen dataclass — the immutability the protocol
    asks for is what makes this a copy rather than a mutation.
    """
    if not dataclasses.is_dataclass(suite):  # pragma: no cover - all shipped suites are
        msg = f"Suite {suite.name!r} cannot be re-pinned: it is not a dataclass."
        raise TypeError(msg)
    pin = StackPin(
        stage=suite.stack_pin.stage,
        provider=provider,
        config=dict(suite.stack_pin.config),
    )
    return dataclasses.replace(suite, stack_pin=pin)


def _summarise(result: SuiteResult) -> str:
    headline = result.headline
    if headline is not None:
        precision = "n/a" if headline.precision is None else f"{headline.precision * 100:.1f}%"
        recall = "n/a" if headline.recall is None else f"{headline.recall * 100:.1f}%"
        f1 = "n/a" if headline.f1 is None else f"{headline.f1 * 100:.1f}%"
        return (
            f"{result.suite}: precision {precision}, recall {recall}, F1 {f1} "
            f"(micro over {headline.dimension}; {headline.gold} gold items, "
            f"{result.documents_scored} documents)"
        )
    overall = result.overall
    rate = overall.pass_rate
    rate_text = "n/a" if rate is None else f"{rate * 100:.1f}%"
    return (
        f"{result.suite}: {overall.passed}/{overall.scored} scored unit tests passed "
        f"({rate_text}); {overall.unsupported} unsupported"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        for name in suite_names():
            suite = get_suite(name)
            print(f"{name}\tstage={suite.stage}\tdataset={suite.dataset.name}")
        return 0

    selected = suite_names() if args.suite == ALL else [args.suite]
    if args.provider and len(selected) > 1:
        # One Provider name cannot be the right pin for four different Stages, and quietly
        # applying it to all of them would publish a page of nonsense.
        print(
            "error: --provider re-pins ONE Stage, so it cannot be combined with --suite all. "
            f"Re-run it for a single suite, e.g. --suite {selected[0]} --provider {args.provider}",
            file=sys.stderr,
        )
        return 2
    roots = _dataset_roots(args.dataset_root)
    results: list[SuiteResult] = []

    for name in selected:
        suite = get_suite(name)
        if args.provider:
            suite = _repin(suite, args.provider)
        try:
            # The license gate runs BEFORE acquisition — restricted bytes never land on disk.
            require_usable(
                suite.dataset.name,
                suite.dataset.license,
                accept_non_commercial=args.accept_non_commercial,
            )
            root = _source(suite.dataset, roots).materialise(suite.dataset)
            results.append(suite.run(root, limit=args.limit))
        except (LicenseGateError, DatasetUnavailableError, BenchmarkDataError) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 2
        except ProviderUnavailableError as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 3

    generated_on = datetime.now(UTC).date().isoformat()
    # Every number carries the date it was produced, so a page assembled from several runs
    # never implies that a carried-forward measurement was measured today.
    measured = [dataclasses.replace(result, measured_on=generated_on) for result in results]
    published: Sequence[SuiteResult] = measured
    if args.carry_forward:
        try:
            published = carry_forward(measured, json_path=args.json_out)
        except PublishedResultsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    write_results(
        published,
        markdown_path=args.markdown_out,
        json_path=args.json_out,
        generated_on=generated_on,
    )
    for result in results:
        print(_summarise(result))
    republished = [r.suite for r in published if r.suite not in {x.suite for x in results}]
    if republished:
        print(f"Republished previously measured suites unchanged: {', '.join(republished)}.")
    print(f"Wrote {args.markdown_out} and {args.json_out}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

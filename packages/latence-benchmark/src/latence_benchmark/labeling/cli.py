# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""``latence-label`` — the ticket-21 labeling harness CLI.

    latence-label gold      --export-dir <dir> --out gold.jsonl
    latence-label stats     --gold gold.jsonl
    latence-label validate  --export-dir <dir> --gold gold.jsonl

``gold`` serves the local single-page UI on ``127.0.0.1`` and opens the browser; the output
file is append-only JSONL and resumes automatically when the file already exists.
``validate`` is the machine gate ticket 17 runs before the eval reads the gold set: exit 0
only when every span round-trips.

Stdlib ``argparse`` deliberately, matching :mod:`latence_benchmark.cli`: the harness must be
runnable straight from a clone with the framework installed and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from latence_benchmark.labeling.goldset import (
    coverage,
    load_corpus,
    load_gold,
    validate_gold,
)
from latence_benchmark.labeling.server import HOST, GoldSession, make_server

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_benchmark.labeling.goldset import Coverage

DEFAULT_PORT = 8765


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latence-label",
        description="Label the manual enterprise gold set.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gold = sub.add_parser("gold", help="Serve the gold-set labeling UI over an exported corpus.")
    gold.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="An S1 export directory (records.jsonl / records.parquet) or the records file itself.",
    )
    gold.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Append-only gold JSONL. An existing file resumes the session.",
    )
    gold.add_argument("--port", type=int, default=DEFAULT_PORT)
    gold.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="Do not open the default browser automatically.",
    )

    stats = sub.add_parser("stats", help="Coverage summary of a gold file (hop distribution).")
    stats.add_argument("--gold", type=Path, required=True)

    validate = sub.add_parser(
        "validate",
        help="Verify every stored span round-trips against the corpus (ticket-17 gate).",
    )
    validate.add_argument("--export-dir", type=Path, required=True)
    validate.add_argument("--gold", type=Path, required=True)
    return parser


def _print_coverage(cov: Coverage) -> None:
    print(f"questions labeled: {cov.total}")
    for hop, count in sorted(cov.hop_distribution.items()):
        print(f"  {hop}-hop: {count}")
    if cov.total:
        print(f"1-hop share: {cov.one_hop_share:.0%}")
    if cov.warning:
        print(f"WARNING: {cov.warning}")


def _serve(session: GoldSession, port: int, open_browser: bool) -> int:
    server = make_server(session, port=port)
    url = f"http://{HOST}:{server.server_port}/"
    print(f"labeling UI: {url}  (Ctrl+C to stop; the output file is append-only and safe)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


def _cmd_gold(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.export_dir)
    if not corpus.chunks:
        print(f"error: no labelable chunks found under {args.export_dir}", file=sys.stderr)
        return 1
    session = GoldSession.open(corpus, args.out)
    cov = coverage(session.questions.values())
    print(
        f"corpus: {len(corpus.documents)} documents, {len(corpus.chunks)} chunks; "
        f"resuming with {cov.total} questions already labeled"
    )
    _print_coverage(cov)
    return _serve(session, args.port, args.open_browser)


def _cmd_stats(args: argparse.Namespace) -> int:
    try:
        questions = load_gold(args.gold)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_coverage(coverage(questions.values()))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.export_dir)
    errors = validate_gold(corpus, args.gold)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"FAIL: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    questions = load_gold(args.gold)
    cov = coverage(questions.values())
    print("OK: gold valid — every span round-trips against the corpus")
    _print_coverage(cov)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    commands = {
        "gold": _cmd_gold,
        "stats": _cmd_stats,
        "validate": _cmd_validate,
    }
    handler = commands[args.command]
    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

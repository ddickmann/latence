"""``latence-sink-qdrant`` — the one command between "pipeline finished" and "my engine serves it".

Two verbs, both thin: they parse, hand everything to :class:`~latence_sink_qdrant.sink.QdrantSink`,
and print what the *store* reports.

    latence-sink-qdrant load ./out/export --collection corpus --url http://localhost:6333
    latence-sink-qdrant verify-absent doc-7#chunk-0 --collection corpus --url http://localhost:6333

``load`` is idempotent: run it again and it upserts only what changed and sweeps what the Export no
longer contains. ``verify-absent`` is the audit step a Retraction/Purge runbook ends with — it exits
non-zero, listing the offenders, if any named record is still in the collection, and exits ``2``
rather than reporting absence if the collection it was pointed at does not exist (a mistyped
``--collection`` or the wrong cluster must never pass for erasure evidence).

Config may come from a YAML/JSON file (``--config``) with the flags overriding it, so a cluster's
connection block lives in one file and the day-to-day command stays short. ``stdlib argparse``, not
typer: a sink is a standalone tool, and this package stays as thin as its dependency list claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from latence_sink_qdrant.errors import SinkConfigError, SinkError
from latence_sink_qdrant.sink import QdrantSink

_CONNECTION_FLAGS = ("url", "host", "port", "api_key", "timeout")


def build_parser() -> argparse.ArgumentParser:
    """The command surface (built separately so the tests can drive it without a subprocess)."""
    parser = argparse.ArgumentParser(
        prog="latence-sink-qdrant",
        description=(
            "Load a latence Export (records.parquet) into a Qdrant collection: create-if-absent, "
            "batched idempotent upserts keyed by record identity, and a reconciling sweep that "
            "makes retracted/purged records verifiably absent."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    load = subparsers.add_parser("load", help="load an Export into a collection")
    load.add_argument("export_dir", help="the directory the Export Stage wrote (local path or URI)")
    _add_common(load)
    load.add_argument("--basename", help="Export file stem (default: records)")
    load.add_argument("--batch-size", type=int, help="records per store round-trip (default: 256)")
    load.add_argument(
        "--distance",
        choices=("cosine", "dot", "euclid", "manhattan"),
        help="metric used only when creating the collection (default: cosine)",
    )
    load.add_argument("--vector-name", help="named vector to write (default: the unnamed vector)")
    load.add_argument(
        "--load-id",
        help="audit label for this load, prefixed onto the per-run sweep marker (default: none)",
    )
    load.add_argument(
        "--no-reconcile",
        action="store_true",
        help=(
            "do NOT delete points absent from this Export. Use when loading an INCREMENTAL Export "
            "into a collection that also holds records the increment omits; the default (reconcile "
            "on) is what makes a Retraction/Purge take effect in the store. Reconciling only ever "
            "sweeps points of this load's id_namespace, so a co-tenant corpus is never touched."
        ),
    )

    verify = subparsers.add_parser(
        "verify-absent",
        help=(
            "assert that the named record ids are NOT in the collection (exit 1 if any remain, "
            "exit 2 if the collection does not exist — an audit never certifies what it did "
            "not see)"
        ),
    )
    verify.add_argument("record_id", nargs="+", help="corpus record id(s) that must be gone")
    _add_common(verify)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--collection", help="target Qdrant collection")
    parser.add_argument("--config", type=Path, help="YAML/JSON file with the sink config block")
    parser.add_argument("--url", help="Qdrant URL, e.g. http://localhost:6333")
    parser.add_argument("--host", help="Qdrant host (alternative to --url)")
    parser.add_argument("--port", type=int, help="Qdrant HTTP port")
    parser.add_argument("--api-key", help="Qdrant api key")
    parser.add_argument("--timeout", type=int, help="per-request timeout in seconds")


def _load_config_file(path: Path) -> dict[str, Any]:
    """Read one YAML/JSON config block — every way it can fail is a :class:`SinkConfigError`.

    A missing file and a malformed one are operator typos, not sink faults, and ADR-0034 says the
    operator sees a typed message (exit ``2``), never ``FileNotFoundError`` /
    ``yaml.parser.ParserError`` as a traceback with absolute paths. The parser's own text is kept
    as the *reason* and chained as ``__cause__``, so nothing is swallowed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Config file {str(path)!r} could not be read: {exc.strerror or exc}."
        raise SinkConfigError(msg) from exc

    is_yaml = path.suffix in {".yaml", ".yml"}
    kind = "YAML" if is_yaml else "JSON"
    try:
        if is_yaml:
            import yaml  # pyyaml arrives with latence-core, so this import cannot be the fault

            loaded = yaml.safe_load(text) or {}
        else:
            loaded = json.loads(text) if text.strip() else {}
    except Exception as exc:  # noqa: BLE001 - normalise ANY parser fault into the typed error
        msg = f"Config file {str(path)!r} is not valid {kind}: {exc}"
        raise SinkConfigError(msg) from exc

    if not isinstance(loaded, dict):
        msg = f"Config file {str(path)!r} must contain a mapping of sink config keys."
        raise SinkConfigError(msg)
    return dict(loaded)


def _config_from(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the config file (if any) with the flags — flags win, unset flags never override."""
    config: dict[str, Any] = _load_config_file(args.config) if args.config else {}
    for key in ("collection", "basename", "batch_size", "distance", "vector_name", "load_id"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    for key in _CONNECTION_FLAGS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if getattr(args, "no_reconcile", False):
        config["reconcile"] = False
    return config


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises a sink error at the user."""
    args = build_parser().parse_args(argv)
    try:
        sink = QdrantSink(_config_from(args))
        if args.command == "load":
            report = sink.load(args.export_dir)
            print(report.summary())
            return 0
        still_present = sink.verify_absent(args.record_id)
        if still_present:
            print(
                "STILL PRESENT in the collection: " + ", ".join(still_present),
                file=sys.stderr,
            )
            return 1
        print(f"verified absent: {len(args.record_id)} record id(s) are not in the collection")
        return 0
    except SinkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]

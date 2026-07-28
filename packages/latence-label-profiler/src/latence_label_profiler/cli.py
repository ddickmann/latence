"""``latence-label-profile`` — profile a corpus sample and write the candidate pool + artifact.

Reads plain-text files (one document per file) from a directory tree, or newline-delimited JSON
records with a text field, and writes the full artifact JSON to stdout or a file. Deliberately does
not reach into the framework's Source/Parser seams: the profiler runs BEFORE a pipeline exists, over
whatever text is already extractable, so its input contract is "text", nothing more.

Zero network by construction: it opens local paths and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from latence_label_profiler.profiler import (
    DEFAULT_MAX_TYPES,
    CorpusProfile,
    CorpusProfiler,
    ProfilerConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def _iter_texts(root: Path, pattern: str, json_field: str | None) -> Iterator[str]:
    """Yield one text per file, in SORTED path order — the sample must not depend on the OS."""
    paths = sorted(p for p in root.rglob(pattern) if p.is_file())
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - filesystem-dependent
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        if json_field is None:
            yield raw
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get(json_field) if isinstance(record, dict) else None
            if isinstance(value, str):
                yield value


def _build(args: argparse.Namespace) -> CorpusProfile:
    config = ProfilerConfig(
        seed=args.seed,
        sample_size=args.sample_size,
        max_types=args.max_types,
        min_mention_mass=args.min_mention_mass,
        min_document_frequency=args.min_document_frequency,
    )
    texts = _iter_texts(Path(args.corpus), args.pattern, args.json_field)
    return CorpusProfiler(config).profile(texts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latence-label-profile",
        description=(
            "Mine a candidate type space from a corpus sample and select it by greedy submodular "
            "max-coverage. Emits the label_inducer.lexicon 'candidates' block plus the profiler's "
            "own optimality certificate, coverage curve, Zipf fit and Good-Turing residual. "
            "No LLM, no model, no network."
        ),
    )
    parser.add_argument("corpus", help="directory to walk for text")
    parser.add_argument("--pattern", default="*.txt", help="glob for input files (default: *.txt)")
    parser.add_argument(
        "--json-field",
        default=None,
        help="treat inputs as JSONL and read this field as the text (e.g. 'content')",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--max-types", type=int, default=DEFAULT_MAX_TYPES)
    parser.add_argument("--min-mention-mass", type=int, default=3)
    parser.add_argument("--min-document-frequency", type=int, default=2)
    parser.add_argument(
        "-o", "--out", default=None, help="write the artifact here (default: stdout)"
    )
    parser.add_argument(
        "--candidates-out",
        default=None,
        help="also write ONLY the label_inducer.lexicon candidates block here",
    )
    args = parser.parse_args(argv)

    profile = _build(args)
    artifact = profile.to_json()
    if args.out:
        Path(args.out).write_text(artifact, encoding="utf-8")
    else:
        sys.stdout.write(artifact)
    if args.candidates_out:
        Path(args.candidates_out).write_text(
            json.dumps(profile.lexicon_candidates(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

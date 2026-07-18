#!/usr/bin/env python
"""Measure per-Stage throughput (documents/hour) for a stack on THIS host.

The reproduction harness behind the numbers in ``PERF-RESULTS.md``. It stages a synthetic
but *real-shaped* corpus (deterministic markdown/text/PDF documents built from the bundled
validation corpus templates, each copy given unique bytes so nothing dedupes), runs the stack
on the ``LocalRunner``, and reads the per-Stage ``duration_seconds`` the Quality Report
already records. It measures nothing itself — the Runner's own instrumentation is the source
of truth, so the reported numbers are exactly what an operator sees in their own report.

Usage (from the repo root)::

    uv run python scripts/bench_throughput.py --docs 200 --repeat 3
    uv run python scripts/bench_throughput.py --stack stacks/default.yaml --docs 400

Every run prints the host, the stack, the corpus shape, and a per-Stage markdown table. Only
run it on an otherwise idle machine, and paste the table into ``PERF-RESULTS.md`` **with the
hardware/date/command recorded** — a throughput number without its host is not a result.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.stacks.corpus import make_text_pdf
from latence_core.stacks.validate import load_stack
from latence_core.storage import Storage

# The benign document templates: one markdown memo, one plain-text note, one 2-page text PDF
# and one support ticket. Entity-rich and cross-referencing, so Extraction/Disambiguation/Graph
# do real work rather than idling on empty input. ``{i}`` makes every copy's bytes unique
# (documents are content-addressed — identical bytes would collapse to one document_id).
_MEMO = """# Acquisition memo {i}

Alice works at Acme Corporation. She joined Acme last year as head of research.
International Business Machines is a partner of Acme Corporation, and Globex is a
competitor. Case reference {i}.
"""

_NOTE = (
    "Bob joined Acme last year. IBM announced strong quarterly results. "
    "Carol works at Globex. Alice is employed by Acme Corporation. Note {i}.\n"
)

_TICKET = """# Support ticket transcript {i}

Alice at Acme Corporation escalated ticket {i}. Her email is alice@acme.example
and her phone is 555-0142. Bob at Globex is the partner contact.
"""

_PDF_PAGE = (
    "Quarterly report {i}, page {p}. Alice works at Acme Corporation, and "
    "International Business Machines partnered with Acme while Bob works at Globex."
)

# The filler that grows a template to the requested document size. Real enterprise documents are
# kilobytes of prose, not one paragraph — benching one-paragraph documents would report a
# throughput dominated by per-document overhead and flatter the engine.
_BODY = """
## Section {n}

Acme Corporation reported that Alice, head of research, met Bob of Globex to review the
partnership with International Business Machines. Carol confirmed the quarterly figures and
escalated case {i} to the compliance desk for review under the standing agreement.
"""


def _grow(text: str, *, index: int, target_bytes: int) -> str:
    """Pad ``text`` with realistic prose sections until it reaches ``target_bytes``."""
    out = text
    section = 0
    while len(out.encode("utf-8")) < target_bytes:
        section += 1
        out += _BODY.format(n=section, i=index)
    return out


def _pdf_pages(index: int, target_bytes: int) -> list[str]:
    """Enough single-line PDF pages to reach roughly ``target_bytes`` of PDF."""
    pages = max(2, target_bytes // 2048)
    return [_PDF_PAGE.format(i=index, p=p) for p in range(pages)]


def _corpus_documents(count: int, doc_bytes: int) -> dict[str, bytes]:
    """``count`` unique documents of ~``doc_bytes`` each, cycling through the four templates."""
    builders = (
        lambda i: (
            f"memo-{i}.md",
            _grow(_MEMO.format(i=i), index=i, target_bytes=doc_bytes).encode("utf-8"),
        ),
        lambda i: (
            f"note-{i}.txt",
            _grow(_NOTE.format(i=i), index=i, target_bytes=doc_bytes).encode("utf-8"),
        ),
        lambda i: (
            f"ticket-{i}.md",
            _grow(_TICKET.format(i=i), index=i, target_bytes=doc_bytes).encode("utf-8"),
        ),
        lambda i: (f"report-{i}.pdf", make_text_pdf(_pdf_pages(i, doc_bytes))),
    )
    return dict(builders[i % len(builders)](i) for i in range(count))


@dataclass(frozen=True)
class StageTiming:
    """One Stage's measured wall time across the repeats, plus its record counts."""

    name: str
    capability: str
    provider: str
    records_in: int
    records_out: int
    seconds: list[float]

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.seconds)

    def docs_per_hour(self, documents: int) -> float:
        return documents * 3600.0 / self.median_seconds if self.median_seconds > 0 else float("inf")


def _run_once(stack: Path, documents: dict[str, bytes], run_id: str) -> QualityReport:
    with tempfile.TemporaryDirectory(prefix="latence-bench-") as tmp:
        root = str(Path(tmp))
        corpus_dir = f"{root}/corpus"
        storage = Storage.for_uri(root)
        storage.makedirs(corpus_dir)
        for name, data in documents.items():
            storage.write_atomic(f"{corpus_dir}/{name}", data)
        pipeline = load_stack(stack, storage_uri=root, corpus_dir=corpus_dir)
        return LocalRunner().run(pipeline, run_id=run_id)


@dataclass(frozen=True)
class Measurement:
    """What a published throughput row needs: the timings AND the corpus that produced them."""

    stages: list[StageTiming]
    totals: list[float]
    documents: int
    corpus_bytes: int


def measure(stack: Path, *, docs: int, doc_bytes: int, repeat: int) -> Measurement:
    """Run the stack ``repeat`` times over ``docs`` documents; return per-Stage + total timings."""
    documents = _corpus_documents(docs, doc_bytes)
    by_stage: dict[str, StageTiming] = {}
    totals: list[float] = []
    for attempt in range(repeat):
        report = _run_once(stack, documents, run_id=f"bench-{attempt:04d}")
        totals.append(report.total_duration_seconds)
        for metric in report.stages:
            existing = by_stage.get(metric.name)
            seconds = [*(existing.seconds if existing else []), metric.duration_seconds]
            by_stage[metric.name] = StageTiming(
                name=metric.name,
                capability=metric.capability,
                provider=metric.provider,
                records_in=metric.records_in,
                records_out=metric.records_out,
                seconds=seconds,
            )
    return Measurement(
        stages=list(by_stage.values()),
        totals=totals,
        documents=docs,
        corpus_bytes=sum(len(data) for data in documents.values()),
    )


def render(stack: Path, result: Measurement) -> str:
    """The markdown table + provenance header that belongs in ``PERF-RESULTS.md``."""
    docs = result.documents
    mib = result.corpus_bytes / 1024 / 1024
    lines = [
        f"Host: {platform.machine()} · {platform.system()} {platform.release()} · "
        f"Python {platform.python_version()} (record the real chip/instance name by hand)",
        f"Stack: {stack} · documents: {docs} · corpus: {mib:.2f} MiB "
        f"({result.corpus_bytes / docs / 1024:.1f} KiB/doc) · repeats: {len(result.totals)} "
        "(median reported)",
        f"Date (UTC): {datetime.now(UTC).date().isoformat()}",
        "",
        "| Stage | Capability | Provider | records in→out | median s | docs/hour | MiB/hour |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {t.name} | {t.capability} | {t.provider} | {t.records_in}→{t.records_out} "
        f"| {t.median_seconds:.3f} | {t.docs_per_hour(docs):,.0f} "
        f"| {mib * 3600.0 / t.median_seconds if t.median_seconds else float('inf'):,.0f} |"
        for t in result.stages
    )
    total = statistics.median(result.totals)
    lines.append(
        f"| **end-to-end** | — | — | — | **{total:.3f}** "
        f"| **{docs * 3600.0 / total:,.0f}** | **{mib * 3600.0 / total:,.0f}** |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=Path("stacks/default.yaml"))
    parser.add_argument("--docs", type=int, default=200, help="documents in the staged corpus")
    parser.add_argument("--doc-kib", type=int, default=16, help="approx. size of each document")
    parser.add_argument("--repeat", type=int, default=3, help="runs to take the median over")
    args = parser.parse_args(argv)

    result = measure(args.stack, docs=args.docs, doc_bytes=args.doc_kib * 1024, repeat=args.repeat)
    print(render(args.stack, result))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())

# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Recover per-page character spans for the OCR'd datasets — the page gold mapping's foundation.

**The problem.** ViDoSeek and OHR-Bench identify gold evidence by *page*, but the S3 export's
``provenance.page_start``/``page_end`` are ``1`` for every row of every OCR'd dataset: the
markdown handed to the pipeline is one assembled document, so the chunker had no page signal to
carry. Page-level gold would therefore have to degrade to document level — a large, silent
inflation of recall on exactly the two single-hop CONTROL datasets whose job is to show *no*
gain from the hypergraph legs.

**The fix.** The page boundaries were never lost, only unindexed. S2 wrote one artifact per page
(``s2-output/pages/<dataset>/<doc_id>/0001.md``) and assembled the document from them with a
pure, deterministic function
(:func:`latence_benchmark.campaign.ocr_lib.assemble_document` over
:func:`~latence_benchmark.campaign.ocr_lib.normalize_html_tables`). Re-running that exact
assembly locally reproduces the markdown **byte for byte** — verified on real docs by
``tests/test_pagemap.py``, which fails if the reconstruction ever drifts — and the running
offsets of the join are the page char spans. The export's chunk provenance indexes the same
string (its final chunk's ``char_end`` equals the markdown's length), so a chunk is on a page
exactly when the two spans overlap.

This module is therefore *derivation, not estimation*: it reads the same bytes S2 wrote and
recomputes an index S3 did not persist. The byte-equality assertion is what keeps it honest — a
reconstruction that no longer matches raises rather than returning plausible-looking spans.
"""

from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

_BENCH_SRC = Path(__file__).resolve().parents[2] / "packages" / "latence-benchmark" / "src"
if str(_BENCH_SRC) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(_BENCH_SRC))

from latence_benchmark.campaign.ocr_lib import (  # noqa: E402
    assemble_document,
    normalize_html_tables,
)

#: The OCR'd datasets: the ones with an S2 ``pages/`` manifest and therefore page-level gold.
PAGINATED_DATASETS = frozenset({"vidoseek", "ohr_bench", "uda"})


class PageMapError(RuntimeError):
    """The page reconstruction did not reproduce the assembled markdown — never guess instead."""


@dataclass(frozen=True)
class DocumentPages:
    """One document's page char spans in the assembled markdown, 1-based.

    ``starts[i]`` / ``ends[i]`` bound page ``i + 1``. ``starts`` is ascending, so a chunk's page
    range is two binary searches rather than a scan over a 900-page 10-K.
    """

    doc_id: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    @property
    def page_count(self) -> int:
        """How many pages the document has."""
        return len(self.starts)

    def span(self, page: int) -> tuple[int, int] | None:
        """The ``[start, end)`` char span of 1-based ``page``, or None when out of range."""
        if page < 1 or page > len(self.starts):
            return None
        return (self.starts[page - 1], self.ends[page - 1])

    def pages_overlapping(self, char_start: int, char_end: int) -> tuple[int, ...]:
        """The 1-based pages a ``[char_start, char_end)`` span touches (ascending)."""
        if char_end <= char_start:
            return ()
        first = max(0, bisect_right(self.starts, char_start) - 1)
        last = bisect_left(self.starts, char_end)
        return tuple(
            page + 1
            for page in range(first, min(last, len(self.starts)))
            if self.ends[page] > char_start
        )


def header_fields(markdown_text: str) -> dict[str, str]:
    """The ``doc_id``/``dataset``/``pages``/``model``/``revision`` of an assembled doc's header.

    Read back rather than configured: the header S2 wrote names the OCR model and revision, so
    the reconstruction never depends on the caller remembering which model produced which
    dataset. A file without the header yields ``{}`` and the reconstruction fails loudly.
    """
    first = markdown_text.split("\n", 1)[0]
    if not first.startswith("<!-- latence-campaign s2-ocr"):
        return {}
    body = first.removeprefix("<!--").removesuffix("-->").strip()
    fields: dict[str, str] = {}
    for part in body.split("|")[1:]:
        key, _, value = part.partition("=")
        fields[key.strip()] = value.strip()
    return fields


def build_document_pages(pages_dir: Path, markdown_path: Path) -> DocumentPages:
    """Reconstruct ``markdown_path``'s page spans from its S2 page artifacts, or raise.

    The assembly is re-run with the *same* pure functions S2 used — over the model/revision the
    document's own header records — and the result compared to the markdown on disk byte for
    byte; a mismatch is a :class:`PageMapError` rather than a set of spans that are subtly off
    by a table normalisation.
    """
    doc_id = markdown_path.stem
    actual = markdown_path.read_text(encoding="utf-8")
    fields = header_fields(actual)
    if not fields:
        msg = f"pagemap: {markdown_path} has no S2 provenance header — cannot reconstruct pages."
        raise PageMapError(msg)
    page_files = sorted(p for p in pages_dir.iterdir() if p.suffix == ".md")
    if not page_files:
        msg = f"pagemap: no page artifacts under {pages_dir}."
        raise PageMapError(msg)
    pages = [normalize_html_tables(p.read_text(encoding="utf-8")).text for p in page_files]
    assembled = assemble_document(
        pages,
        doc_id=fields.get("doc_id", doc_id),
        dataset=fields.get("dataset", ""),
        model=fields.get("model", ""),
        revision=fields.get("revision", ""),
    )
    if assembled != actual:
        msg = (
            f"pagemap: reconstruction of {doc_id!r} does not match the assembled markdown "
            f"({len(assembled)} vs {len(actual)} chars) — the page spans would be wrong."
        )
        raise PageMapError(msg)
    header_len = assembled.index(pages[0]) if pages[0] else len(assembled)
    starts: list[int] = []
    ends: list[int] = []
    cursor = header_len
    for index, page in enumerate(pages):
        starts.append(cursor)
        ends.append(cursor + len(page))
        cursor += len(page) + (2 if index + 1 < len(pages) else 0)
    return DocumentPages(doc_id=doc_id, starts=tuple(starts), ends=tuple(ends))


def load_dataset_pages(
    s2_root: Path,
    dataset: str,
    *,
    only: frozenset[str] | None = None,
) -> dict[str, DocumentPages]:
    """Every document's page spans for ``dataset`` (``{doc_id: DocumentPages}``).

    ``only`` restricts the work to the doc ids a run actually needs — a smoke run over 50
    ViDoSeek queries touches ~50 of 292 documents, and UDA has 1,917. Documents whose
    reconstruction fails are **omitted and reported by the caller** through the gold-resolution
    rate rather than silently mapped at document level.
    """
    pages_root = s2_root / "pages" / dataset
    markdown_root = s2_root / "markdown" / dataset
    out: dict[str, DocumentPages] = {}
    if not pages_root.is_dir():
        return out
    for doc_dir in sorted(pages_root.iterdir()):
        if not doc_dir.is_dir() or (only is not None and doc_dir.name not in only):
            continue
        markdown_path = markdown_root / f"{doc_dir.name}.md"
        if not markdown_path.is_file():
            continue
        try:
            out[doc_dir.name] = build_document_pages(doc_dir, markdown_path)
        except PageMapError:
            continue
    return out


__all__ = [
    "PAGINATED_DATASETS",
    "DocumentPages",
    "PageMapError",
    "build_document_pages",
    "load_dataset_pages",
]

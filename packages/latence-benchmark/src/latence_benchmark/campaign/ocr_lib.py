# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pure S2 OCR logic shared by the campaign stage and the pod-side worker.

This module is **stdlib-only and flat-importable by design**: the s2-ocr stage pushes this
exact file to the pod next to ``ocr_worker.py``, which imports it as a plain ``import
ocr_lib`` — so the table normalizer, repetition detector, manifest planner, fin-cap
selector and provenance format are implemented once, unit-tested once (offline), and the
pod executes the very bytes the suite proved. Nothing here touches the network, a clock,
a GPU, or ``latence_benchmark`` itself.

The three OCR-specific rules live here because each was decided with evidence
(issues/wayfinder-e2e-sota-run/tickets/s2-ocr-markdown-normalization.md):

* **HTML tables become GitHub-markdown tables.** LightOnOCR-2 emits ``<table>`` HTML
  inside its markdown stream, and the chunker's ``strip_markup`` deletes raw tags keeping
  inner text only — an unnormalized table loses ALL row/column structure downstream.
  Colspan is flattened by repeating the cell; nested tables are linearized into the outer
  cell; a table that cannot be parsed is KEPT RAW (and counted), never dropped.
* **Repetition flag = any 8-gram repeated more than 4 times** (the ~1% LightOnOCR loop
  failure mode) — the worker retries such a page exactly once.
* **The manifest is the filesystem**: one ``pages/<doc_id>/<page:04d>.md`` per page,
  skip-if-exists, so a crash loses at most in-flight pages and a rerun is a no-op.
* **The assembly emits its page spans** alongside the markdown
  (:func:`assemble_document_with_pages`) and the worker writes them beside the document as a
  ``.pagemap.json`` sidecar. Joining the pages used to DESTROY the page boundaries the worker
  was holding, so the re-ingested markdown resolved every offset to page 1 across the whole
  S3 corpus; the boundaries now travel with the artifact (ADR-0060). ``assemble_document`` is
  the same function's ``.text``, so the markdown is byte-identical to what S2 already wrote.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

T = TypeVar("T")

#: 8-gram repeated more than this many times flags a page as a repetition loop.
REPETITION_NGRAM = 8
REPETITION_MAX_REPEATS = 4

#: Colspan values are clamped here — a hallucinated ``colspan="9999"`` must not
#: manufacture a nine-thousand-column markdown table.
_MAX_COLSPAN = 50

_TABLE_TAG = re.compile(r"<\s*(/?)\s*table\b[^>]*>", re.IGNORECASE)
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


# -- fin-cap selection -----------------------------------------------------------------------


def select_capped_docs(qa_counts: Mapping[str, int], cap: int) -> tuple[str, ...]:
    """The top-``cap`` most-questioned docs; ties broken by ASCENDING doc name.

    Deterministic by construction (no set iteration, no hash order): the ranking key is
    ``(-count, name)``, so the same counts always select the same documents — the property
    the fin-cap metrics record and the S2 gate relies on.
    """
    ranked = sorted(qa_counts, key=lambda doc: (-qa_counts[doc], doc))
    return tuple(ranked[:cap])


def qa_coverage(qa_counts: Mapping[str, int], selected: Collection[str]) -> tuple[int, int]:
    """``(qa_covered_by_selection, qa_total)`` — the honest coverage number for metrics."""
    covered = sum(qa_counts.get(doc, 0) for doc in selected)
    return covered, sum(qa_counts.values())


# -- zip-slip guard --------------------------------------------------------------------------


def safe_zip_entry(name: str) -> bool:
    """Whether a zip entry name stays inside the extraction root.

    Rejects absolute paths, Windows drive prefixes, backslash separators, and any ``..``
    component — the classic zip-slip vectors. Pure so the guard is testable without
    building hostile archives.
    """
    if not name or name.startswith(("/", "\\")):
        return False
    if "\\" in name or _DRIVE_PREFIX.match(name):
        return False
    return all(part not in ("..", "") for part in name.split("/"))


# -- repetition detector ---------------------------------------------------------------------


def has_repetition(
    text: str,
    *,
    ngram: int = REPETITION_NGRAM,
    max_repeats: int = REPETITION_MAX_REPEATS,
) -> bool:
    """True when any ``ngram``-word sequence occurs MORE than ``max_repeats`` times.

    Word-level (whitespace tokenization): the LightOn failure mode is a looping phrase,
    and word grams are insensitive to the punctuation the loop happens to carry.
    Exactly ``max_repeats`` occurrences is legitimate text (boilerplate headers repeat);
    the flag needs strictly more.
    """
    words = text.split()
    if len(words) < ngram:
        return False
    grams = Counter(tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1))
    return any(count > max_repeats for count in grams.values())


# -- HTML-table -> markdown normalization ----------------------------------------------------


@dataclass(frozen=True)
class TableNormalization:
    """One page's normalization outcome: the text plus the two honest counters."""

    text: str
    converted: int
    kept_raw: int


class _TableParser(HTMLParser):
    """Collects ``(cell_text, colspan)`` rows at table depth 1; deeper markup linearizes.

    A nested ``<table>`` raises the depth, so its rows/cells never open rows here — their
    text lands in the currently open outer cell (separated by spaces), which is exactly
    the "nested tables linearized" ruling.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int]]] = []
        self._row: list[tuple[str, int]] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._depth = 0

    # HTMLParser is forgiving, not validating: implied closes (a new <tr> while a row is
    # open, a new <td> while a cell is open) flush the previous element, mirroring how
    # browsers repair the same soup.

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._depth += 1
            return
        if self._depth != 1:
            if self._cell is not None and tag in ("tr", "td", "th", "br"):
                self._cell.append(" ")
            return
        if tag == "tr":
            self._flush_cell()
            self._flush_row()
            self._row = []
        elif tag in ("td", "th"):
            self._flush_cell()
            if self._row is None:
                self._row = []
            self._cell = []
            self._colspan = _parse_colspan(attrs)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            if self._depth == 1:
                self._flush_cell()
                self._flush_row()
            self._depth = max(self._depth - 1, 0)
            return
        if self._depth != 1:
            return
        if tag in ("td", "th"):
            self._flush_cell()
        elif tag == "tr":
            self._flush_cell()
            self._flush_row()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _flush_cell(self) -> None:
        if self._cell is None:
            return
        text = " ".join("".join(self._cell).split())
        if self._row is None:
            self._row = []
        self._row.append((text, self._colspan))
        self._cell = None
        self._colspan = 1

    def _flush_row(self) -> None:
        if self._row:
            self.rows.append(self._row)
        self._row = None


def _parse_colspan(attrs: list[tuple[str, str | None]]) -> int:
    for key, value in attrs:
        if key.lower() == "colspan" and value is not None:
            try:
                return min(max(int(value.strip()), 1), _MAX_COLSPAN)
            except ValueError:
                return 1
    return 1


def _table_spans(text: str) -> tuple[list[tuple[int, int]], int]:
    """Balanced OUTERMOST ``<table>…</table>`` spans, plus 1 if an open tag never closes.

    Nesting is tracked by depth so an inner table never terminates the outer span; a
    dangling ``<table>`` with no close is malformed — it stays raw and is counted.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for match in _TABLE_TAG.finditer(text):
        if match.group(1) != "/":
            if depth == 0:
                start = match.start()
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, match.end()))
    return spans, (1 if depth > 0 else 0)


def _render_markdown_table(rows: Sequence[Sequence[tuple[str, int]]]) -> str | None:
    expanded: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for text, colspan in row:
            escaped = text.replace("|", "\\|")
            cells.extend([escaped] * colspan)  # colspan flattened with repeated cells
        expanded.append(cells)
    width = max((len(row) for row in expanded), default=0)
    if width == 0:
        return None
    padded = [row + [""] * (width - len(row)) for row in expanded]
    lines = [f"| {' | '.join(padded[0])} |", f"| {' | '.join(['---'] * width)} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in padded[1:])
    return "\n".join(lines)


def _convert_table(raw: str) -> str | None:
    parser = _TableParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # Any parser escape means "keep the raw HTML", never "drop the page".
        return None
    if not parser.rows:
        return None
    return _render_markdown_table(parser.rows)


def normalize_html_tables(text: str) -> TableNormalization:
    """Replace every parseable HTML table with a GitHub-markdown table, in place.

    A table that fails to parse (or yields no rows) is left EXACTLY as it was and counted
    in ``kept_raw`` — losing a broken table's raw text would be strictly worse than
    shipping it un-normalized. Text outside table spans is byte-identical in the output.
    """
    spans, unclosed = _table_spans(text)
    converted = 0
    kept_raw = unclosed
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        raw = text[start:end]
        markdown = _convert_table(raw)
        if markdown is None:
            kept_raw += 1
            pieces.append(raw)
        else:
            converted += 1
            pieces.append(markdown)
        cursor = end
    pieces.append(text[cursor:])
    return TableNormalization(text="".join(pieces), converted=converted, kept_raw=kept_raw)


# -- file-manifest planning ------------------------------------------------------------------


def page_filename(page: int) -> str:
    """``0001.md``-style page artifact name (1-based, 4 digits — the manifest key)."""
    return f"{page:04d}.md"


def missing_pages(total: int, existing: Collection[str]) -> tuple[int, ...]:
    """Which of pages ``1..total`` have no artifact yet — the resumable work plan.

    ``existing`` is the set of filenames already present in the doc's pages dir;
    skip-if-exists at planning time is what makes a rerun after a crash lose nothing.
    """
    return tuple(page for page in range(1, total + 1) if page_filename(page) not in existing)


def provenance_header(
    *, doc_id: str, dataset: str, page_count: int, model: str, revision: str
) -> str:
    """The assembled document's first line — machine-greppable provenance, as a comment."""
    return (
        f"<!-- latence-campaign s2-ocr | doc_id={doc_id} | dataset={dataset} | "
        f"pages={page_count} | model={model} | revision={revision} -->"
    )


#: Suffix of the page-map sidecar written beside each assembled document. Must equal
#: :data:`latence_core.contracts.PAGE_MAP_SIDECAR_SUFFIX` — it is repeated here rather than
#: imported because this module is pushed to the pod as a bare file and may not import
#: ``latence_core``; ``test_campaign_ocr_lib`` asserts the two never drift.
PAGE_MAP_SIDECAR_SUFFIX = ".pagemap.json"


@dataclass(frozen=True)
class AssembledDocument:
    """The assembled markdown plus where each source page landed inside it.

    ``page_spans`` is one ``(page_number, char_start, char_end)`` per input page, half-open
    and **contiguous from 0 to len(text)** — the shape
    :class:`latence_core.contracts.PageMap` validates. Contiguity means the two joins that are
    not page content have to be attributed to a page rather than left as gaps: the provenance
    header goes to page 1, and each ``\\n\\n`` separator (and the trailing newline) goes to the
    page that PRECEDES it, exactly as
    :meth:`latence_core.contracts.PageMap.from_page_texts` does it. Offsets that fall in real
    page text are unaffected either way; this only decides who owns the seams.
    """

    text: str
    page_spans: tuple[tuple[int, int, int], ...]


def assemble_document(
    pages: Sequence[str], *, doc_id: str, dataset: str, model: str, revision: str
) -> str:
    """Provenance header + the pages joined with a blank line — the ``markdown/`` artifact."""
    return assemble_document_with_pages(
        pages, doc_id=doc_id, dataset=dataset, model=model, revision=revision
    ).text


def assemble_document_with_pages(
    pages: Sequence[str], *, doc_id: str, dataset: str, model: str, revision: str
) -> AssembledDocument:
    """:func:`assemble_document`'s markdown **plus** the page char spans inside it.

    The page boundaries were never unknown at this point — the worker is holding the pages —
    they were simply thrown away by the join, which is how every chunk of 3,470 OCR'd
    documents came to cite page 1 (ADR-0060). Computing them HERE, in the same function that
    builds the string, is what keeps them exact: there is no second implementation of the join
    to drift from, and :func:`assemble_document` is now literally this function's ``.text``,
    so the artifact the S3 corpus was built from is byte-identical by construction.

    A zero-page document is coerced to one page covering the (header-only) text, mirroring
    ``PageMap.from_page_texts([])`` — a map must have at least one span.
    """
    header = provenance_header(
        doc_id=doc_id,
        dataset=dataset,
        page_count=len(pages),
        model=model,
        revision=revision,
    )
    text = header + "\n\n" + "\n\n".join(pages) + "\n"
    if not pages:
        return AssembledDocument(text=text, page_spans=((1, 0, len(text)),))
    spans: list[tuple[int, int, int]] = []
    # Page 1 absorbs the header; every page absorbs the separator that follows it, and the
    # last absorbs the trailing newline — so the spans tile [0, len(text)) with no gap.
    cursor = 0
    content = len(header) + 2
    last = len(pages) - 1
    for i, page in enumerate(pages):
        content += len(page)
        content += 2 if i != last else 1
        spans.append((i + 1, cursor, content))
        cursor = content
    return AssembledDocument(text=text, page_spans=tuple(spans))


def page_map_sidecar_json(assembly: AssembledDocument) -> str:
    """``assembly``'s page spans as the sidecar file's bytes — a serialized ``PageMap``.

    Written beside the markdown as ``<doc>.md`` + :data:`PAGE_MAP_SIDECAR_SUFFIX`, where
    ``source.local_folder`` finds it and ``parser.plaintext`` turns it back into a real
    :class:`latence_core.contracts.PageMap` (ADR-0060). ``origin`` is deliberately NOT written:
    a file cannot vouch for its own provenance, so the consumer stamps
    :attr:`~latence_core.contracts.PageMapOrigin.SIDECAR` itself. Keys are emitted in a fixed
    order with no whitespace variance, so the sidecar is byte-deterministic like every other
    campaign artifact.
    """
    pages = [
        {"page_number": number, "char_start": start, "char_end": end}
        for number, start, end in assembly.page_spans
    ]
    return json.dumps({"pages": pages, "total_chars": len(assembly.text)}) + "\n"


# -- spot-check sampling ---------------------------------------------------------------------
#
# Two-stage and doc-first BY NECESSITY: the streaming worker deletes a doc's page images
# the moment its markdown is assembled (the /workspace quota bound), so the sample must be
# decidable per doc AT CLEANUP TIME — docs are sampled from the upfront doc list, and the
# page within each sampled doc from a per-doc-keyed generator once its page count is known.
# Both stages are fixed-seed deterministic, so relaunches retain the same evidence.


def sample_spot_docs(
    doc_keys: Sequence[str], *, count: int = 12, seed: int = 0
) -> tuple[str, ...]:
    """``count`` deterministic docs (sorted) whose one spot page the worker retains."""
    keys = sorted(doc_keys)
    if len(keys) <= count:
        return tuple(keys)
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(keys, count)))


def spot_page_for(doc_key: str, page_count: int, *, seed: int = 0) -> int:
    """The one page (1-based) retained for ``doc_key`` — stable given key and count."""
    if page_count < 2:
        return 1
    return random.Random(f"{seed}:{doc_key}").randint(1, page_count)

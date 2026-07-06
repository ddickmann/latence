"""Test fixtures for the document Parser: a dependency-free PDF builder.

Provides ``make_text_pdf`` as a fixture (a callable) so tests exercise the real
per-page pypdf extraction path without pulling reportlab or shipping a binary
fixture. Deterministic: identical text yields identical bytes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest


def _make_text_pdf(pages: list[str]) -> bytes:
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def _content(text: str) -> str:
        return f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET"

    if not pages:
        pages = [""]
    n = len(pages)

    offsets: dict[int, int] = {}
    buf = "%PDF-1.4\n"

    def add(objnum: int, body: str) -> None:
        nonlocal buf
        offsets[objnum] = len(buf.encode("latin-1"))
        buf += f"{objnum} 0 obj\n{body}\nendobj\n"

    page_obj_nums = [4 + 2 * i for i in range(n)]
    content_obj_nums = [5 + 2 * i for i in range(n)]
    kids = " ".join(f"{p} 0 R" for p in page_obj_nums)

    add(1, "<< /Type /Catalog /Pages 2 0 R >>")
    add(2, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>")
    add(3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i in range(n):
        cs = _content(pages[i])
        add(
            page_obj_nums[i],
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_obj_nums[i]} 0 R >>",
        )
        add(content_obj_nums[i], f"<< /Length {len(cs)} >>\nstream\n{cs}\nendstream")

    xref_pos = len(buf.encode("latin-1"))
    max_obj = max(offsets)
    buf += f"xref\n0 {max_obj + 1}\n0000000000 65535 f \n"
    for i in range(1, max_obj + 1):
        buf += f"{offsets.get(i, 0):010d} 00000 n \n"
    buf += f"trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
    return buf.encode("latin-1")


@pytest.fixture()
def make_text_pdf() -> Callable[[list[str]], bytes]:
    """Return a builder that emits a valid PDF with one text line per page."""
    return _make_text_pdf

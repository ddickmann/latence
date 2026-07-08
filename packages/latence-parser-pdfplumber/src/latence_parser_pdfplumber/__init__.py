"""``latence-parser-pdfplumber`` — the first REAL Wave-1 adapter over ``AdapterBase`` (H-A1).

A CPU, page-aware PDF (+ plain-text) Parser Provider (``parser.pdfplumber``) over the pure-Python
`pdfplumber <https://github.com/jsvine/pdfplumber>`_ library (MIT), implemented as an
:class:`~latence_core.providers.adapter.AdapterBase` subclass. It turns the hypothetical P3-F2
provider seam into a real second consumer and is the template every subsequent Wave-1 adapter
copies. See :mod:`latence_parser_pdfplumber.provider` for the seam-report and library-choice
rationale (why pdfplumber substitutes for the preferred-but-unsuitable kreuzberg).
"""

from __future__ import annotations

from latence_parser_pdfplumber.provider import PdfPlumberParser, size_after_parse

__all__ = ["PdfPlumberParser", "size_after_parse"]

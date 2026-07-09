"""``latence-parser-render`` — the universal any-file Parser (W5-universal-parser).

A license-clean port of the colsearch batch converter: one Parser Provider (``parser.render``) that
turns ANY common file type — PDF, image, or office (docx/pptx/xlsx) — into a
:class:`~latence_core.contracts.DocumentRecord` + :class:`~latence_core.contracts.PageMap`,
dispatching by type. PDF via ``pypdfium2`` (Apache/BSD, the same renderer parser.lighton uses),
images via ``Pillow`` (HPND), office via LibreOffice-headless (``soffice``, MPL-2.0 external
subprocess) when present else ``python-docx``/``openpyxl``/``python-pptx`` (MIT) text extraction.
Page images are OCR'd via a **configured OCR Provider** (``ocr: {provider: parser.lighton |
parser.glm}``) through its ``Parser.parse`` seam — reusing the pod-validated OCR models, never
reinventing them. It carries the parser security lessons (pixel-bomb cap, offset-0 type-spoof guard,
soffice subprocess timeout no-shell, native fault → typed error no leak) and NEVER imports
``fitz``/PyMuPDF (AGPL — a test enforces it). Built THROUGH
:class:`~latence_core.providers.adapter.AdapterBase` (H-A1), ``compute="either"``. See
:mod:`latence_parser_render.provider` for the seam-report and the license verification.
"""

from __future__ import annotations

from latence_parser_render.provider import UniversalRenderParser, size_after_parse

__all__ = ["UniversalRenderParser", "size_after_parse"]

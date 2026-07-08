"""``latence-parser-lighton`` — a real GPU OCR Parser over LightOn-OCR (W1-lighton, #91).

A GPU-class, **image→markdown** Parser Provider (``parser.lighton``) over ``lightonai/
LightOnOCR-1B-1025`` (a compact 1B vision-language OCR model, **Apache-2.0** weights AND code),
implemented as an :class:`~latence_core.providers.adapter.AdapterBase` subclass (H-A1) exactly like
``parser.pdfplumber``. It unlocks scanned / image-only pages the CPU text-layer parsers cannot read
(it reads pixels, not a text layer), emits markdown per page, and takes that markdown onto the
exact-offset :func:`~latence_core.contracts.PageMap.from_page_texts` seam (ADR-0031). Heavy deps
(transformers/torch/pillow/pypdfium2) ship HERE, not in the thin core (ADR-0016), and are
lazy-imported so Provider discovery stays torch-free. See :mod:`latence_parser_lighton.provider`
for the seam-report, the real-model API the offline fake mirrors, and the license verification.
"""

from __future__ import annotations

from latence_parser_lighton.provider import LightOnOcrParser, size_after_parse

__all__ = ["LightOnOcrParser", "size_after_parse"]

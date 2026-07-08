"""``latence-parser-glm`` — a real GPU OCR Parser over GLM-OCR (W1-glm, #91).

A GPU-class, **image→markdown** Parser Provider (``parser.glm``) over ``zai-org/GLM-OCR`` (a compact
0.9B vision-language OCR model, **MIT** weights AND code; the pipeline's PP-DocLayoutV3 layout
component is Apache-2.0), implemented as an :class:`~latence_core.providers.adapter.AdapterBase`
subclass (H-A1) — the **near-twin** of the pod-validated ``parser.lighton``. It unlocks scanned /
image-only pages the CPU text-layer parsers cannot read (it reads pixels, not a text layer), emits
markdown per page, and takes that markdown onto the exact-offset
:func:`~latence_core.contracts.PageMap.from_page_texts` seam (ADR-0031). A **pixel-bomb DoS cap**
(max-megapixel render clamp + ``Image.MAX_IMAGE_PIXELS``) is built in from the start — GLM does not
defer the LightOn security finding. Heavy deps (transformers/torch/pillow/pypdfium2) ship HERE, not
in the thin core (ADR-0016), and are lazy-imported so Provider discovery stays torch-free. See
:mod:`latence_parser_glm.provider` for the seam-report, the real-model API the offline fake mirrors,
and the license verification.
"""

from __future__ import annotations

from latence_parser_glm.provider import GlmOcrParser, size_after_parse

__all__ = ["GlmOcrParser", "size_after_parse"]

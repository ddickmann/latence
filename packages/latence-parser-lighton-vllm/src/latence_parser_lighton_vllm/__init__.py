"""``latence-parser-lighton-vllm`` — the PRODUCTION LightOn-OCR Parser over a served vLLM (W10).

The production OCR path: a drop-in :class:`~latence_core.capability.Parser`
(``parser.lighton_vllm``) that hits a **running vLLM OpenAI-compatible endpoint** — ~2.7x faster at
concurrency than optimised in-process transformers (LightOn's recommended path). It reuses
``parser.lighton``'s exact rasteriser (200 dpi / megapixel bomb-cap / 1540px LANCZOS longest-side —
the shared :class:`~latence_parser_lighton.rasterize.PageRasteriser`, so the two OCR back-ends
cannot diverge), POSTs each page as an OpenAI multimodal image to ``{base_url}/v1/chat/completions``
(max_tokens 4096 / temperature 0.2 / top_p 0.9 — LightOn's official), assembles the per-page
markdown onto the exact-offset :func:`~latence_core.contracts.PageMap.from_page_texts` seam, fans
the page requests out concurrently (bounded + retried, per-page fail-open), keeping the public
``parse`` sync. The in-process ``parser.lighton`` stays for single-box / simple. See
:mod:`latence_parser_lighton_vllm.provider` for the served vLLM recipe (cudagraphs in production)
and the license verification, and ``docs/serving-lighton-vllm.md`` for the turnkey serve.
"""

from __future__ import annotations

from latence_parser_lighton_vllm.provider import LightOnVllmParser, size_after_parse

__all__ = ["LightOnVllmParser", "size_after_parse"]
__version__ = "0.1.0"

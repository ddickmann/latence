"""Render the one-page document the served-OCR smoke parses (issue #196).

The fixture is GENERATED rather than committed as a binary, and that is deliberate: it is rendered
by the *pinned Pillow of this very environment*, so the smoke's input is produced by the stack it
is testing. A committed PNG would be rendered by whatever machine happened to make it once.

The page carries a sentinel line whose tokens the smoke asserts came back out of the served OCR
endpoint. If the endpoint is not really reached — or reached and its output dropped — those tokens
cannot appear downstream, because this pipeline has no other source of text.

    python make-fixture.py <output.pdf>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# A4 at 150 dpi. Comfortably above the rasteriser's 1540px longest-side downscale, so the page goes
# through the same resize path a real 200-dpi scan does.
PAGE = (1240, 1754)

# The tokens the smoke looks for on the way out. Distinctive enough that no template, prompt
# scaffolding or error message could produce them by accident.
SENTINEL_TOKENS = ("LATENCE", "SERVED", "OCR", "SMOKE", "4711")

LINES = (
    "LATENCE SERVED OCR SMOKE 4711",
    "",
    "This page exists to prove one thing: that the Parse stage",
    "reached the vLLM LightOn-OCR endpoint and its markdown came",
    "back through the pipeline into the exported RAG corpus.",
    "",
    "Invoice number: 4711    Date: 2026-07-18",
    "Vendor: Latence Framework Reproducible Environments",
)


def render(destination: Path) -> None:
    page = Image.new("RGB", PAGE, "white")
    draw = ImageDraw.Draw(page)
    # Pillow >= 10.1 makes the bundled default font scalable; this environment pins 12.3, so real
    # glyphs at a real size — not the 11px bitmap font, which no OCR model should be asked to read.
    heading = ImageFont.load_default(size=64)
    body = ImageFont.load_default(size=40)
    y = 180
    for index, line in enumerate(LINES):
        font = heading if index == 0 else body
        draw.text((120, y), line, fill="black", font=font)
        y += 110 if index == 0 else 70
    destination.parent.mkdir(parents=True, exist_ok=True)
    page.save(destination, "PDF", resolution=150.0)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    destination = Path(argv[1])
    render(destination)
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")
    print(f"sentinel tokens: {' '.join(SENTINEL_TOKENS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

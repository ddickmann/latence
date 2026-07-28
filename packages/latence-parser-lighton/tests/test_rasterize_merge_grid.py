"""T-OCR-2 — both page-image sides must land on the 28-px patch-merge grid.

**The defect.** ``rasterize.py`` capped the LONGEST side at 1540 px and let the shorter side land
wherever the aspect ratio put it. LightOnOCR is a ``mistral3`` model whose vision tower is
``pixtral``: it cuts the page into **14-px patches**, then ``Mistral3PatchMerger.forward`` merges
them 2x2 with ``torch.nn.functional.unfold(kernel_size=2, stride=2)`` — and ``unfold`` emits
**only complete windows**. An odd patch row or column has no partner, so it is **silently dropped**
along with the ~14 px of page it covered. 1540 is safe (1540/14 = 110, even); a US-Letter page
rasterised at 200 DPI is 1700x2200, scales to width **1190 = 85 patches — odd**, and the right-hand
14 px of every such page never reaches the model. Patch 14 x merge 2 = **28**: both sides must be a
multiple of 28.

Scope, stated honestly (DECISIONS.md § D1a): the *mechanism* is verified in the vLLM/transformers
source, the *visible loss* is inferred rather than LightOn-confirmed. It is cheap to satisfy and
cheap to test, so it is satisfied rather than argued about.

:func:`_dropped_pixels` below re-implements the merger's window arithmetic, so these tests fail for
the actual reason rather than against a hard-coded number: the assertions are "no patch row is
dropped", not "the width is 1176".

The core sweep drives :func:`merge_grid_target` — plain integers in, plain integers out, no image
library and no fake anywhere near it. The integration tests then drive the real
``PageRasteriser._downscale_longest`` and the Provider's own ``_image_from_bytes`` through the
package's offline PIL double, so the arithmetic proven above is proven to be the arithmetic the
Provider actually runs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from latence_parser_lighton.provider import LightOnOcrParser
from latence_parser_lighton.rasterize import (
    DEFAULT_MAX_LONGEST_DIM,
    MERGE_GRID,
    PATCH_MERGE,
    PATCH_SIZE,
    PageRasteriser,
    merge_grid_target,
)

_CAP = DEFAULT_MAX_LONGEST_DIM


# --- the vision tower's own arithmetic, re-implemented so the tests fail for the real reason ----


def _dropped_pixels(side_px: int) -> int:
    """Pixels of page lost on one axis to ``unfold(kernel=2, stride=2)``'s complete-windows rule."""
    patches = side_px // PATCH_SIZE
    merged = patches // PATCH_MERGE  # unfold emits floor(n / stride) complete windows
    return (patches - merged * PATCH_MERGE) * PATCH_SIZE


def test_the_merge_grid_is_derived_from_the_model_not_asserted() -> None:
    """14-px patches merged 2x2 == a 28-px grid, and LightOn's 1540 already sits on it."""
    assert PATCH_SIZE == 14
    assert PATCH_MERGE == 2
    assert MERGE_GRID == PATCH_SIZE * PATCH_MERGE == 28
    assert _CAP % MERGE_GRID == 0, "1540 is safe; the ticket's premise would be wrong otherwise"


def test_an_odd_patch_grid_really_does_drop_a_page_strip() -> None:
    """Guard the guard: if this were 0, every assertion below would be vacuous."""
    assert _dropped_pixels(1190) == PATCH_SIZE  # US-Letter's width under the old longest-side cap
    assert _dropped_pixels(1176) == 0  # the same page rounded onto the grid
    assert _dropped_pixels(_CAP) == 0


# --- the property sweep ------------------------------------------------------------------------


def _pages() -> Iterator[tuple[int, int]]:
    """Real page shapes at 200 DPI, extreme aspect ratios, and a seeded random sweep."""
    yield from (
        (1700, 2200),  # US Letter, 200 DPI — the shape that motivated this ticket
        (1654, 2339),  # A4, 200 DPI
        (2200, 1700),  # US Letter, landscape
        (2339, 1654),  # A4, landscape
        (1700, 2800),  # US Legal, 200 DPI
        (2339, 3311),  # A3, 200 DPI
        (1540, 1540),  # already exactly at the cap, square
        (1120, 1540),  # already on the grid, at the cap
        (800, 1000),  # comfortably under the cap — must NOT be upscaled
        (28, 28),  # exactly one merge window
        (29, 31),  # just over one merge window
        (3000, 30),  # a degenerate strip: the short side cannot be aligned without upscaling
    )
    rng = random.Random(20260728)  # seeded: the sweep is reproducible, no hypothesis dep (ADR-0016)
    for _ in range(400):
        yield (rng.randint(1, 6000), rng.randint(1, 6000))


@pytest.mark.parametrize(("width", "height"), list(_pages()))
def test_the_output_always_satisfies_every_invariant(width: int, height: int) -> None:
    """mod 28 on BOTH sides, longest <= 1540, aspect within one grid step, and never upscaled."""
    target = merge_grid_target(width, height, _CAP)
    if target is None:
        # Two licensed refusals, and nothing else: the page is ALREADY aligned and in budget (so a
        # resize would be pure loss), or a side is too narrow to reach 28 px without ENLARGING it.
        already_good = (
            width % MERGE_GRID == 0 and height % MERGE_GRID == 0 and max(width, height) <= _CAP
        )
        assert already_good or min(width, height) < MERGE_GRID, (
            f"{width}x{height} was left off the merge grid but could have been aligned"
        )
        return
    out_w, out_h = target

    assert out_w % MERGE_GRID == 0 and out_h % MERGE_GRID == 0, (
        f"{width}x{height} -> {out_w}x{out_h}: a patch row/column has no 2x2 partner and is "
        f"silently dropped by the merger "
        f"(lost {_dropped_pixels(out_w)}px of width, {_dropped_pixels(out_h)}px of height)"
    )
    assert _dropped_pixels(out_w) == 0 and _dropped_pixels(out_h) == 0
    assert max(out_w, out_h) <= _CAP, f"{width}x{height} -> {out_w}x{out_h} exceeds the 1540 cap"
    assert out_w <= width and out_h <= height, (
        f"{width}x{height} -> {out_w}x{out_h} UPSCALED the page — never do that"
    )
    assert out_w >= MERGE_GRID and out_h >= MERGE_GRID

    # Aspect: rescale the input by the factor the longest side actually took, and the other side
    # must land within one grid step of where proportionality would have put it.
    scale = max(out_w, out_h) / float(max(width, height))
    assert abs(out_w - width * scale) <= MERGE_GRID
    assert abs(out_h - height * scale) <= MERGE_GRID


def test_us_letter_at_200_dpi_stops_losing_its_right_edge() -> None:
    """The concrete case from DECISIONS.md § D1a: 1700x2200 -> width 1190 -> 85 patches -> odd."""
    naive_width = round(1700 * (_CAP / 2200))
    assert naive_width == 1190
    assert _dropped_pixels(naive_width) == PATCH_SIZE  # what the longest-side-only cap produced

    target = merge_grid_target(1700, 2200, _CAP)
    assert target is not None
    assert target == (1176, 1540)
    assert _dropped_pixels(target[0]) == 0


def test_a_page_under_the_cap_is_rounded_down_never_up() -> None:
    """No upscaling is a hard rule, so an unaligned small page snaps DOWN onto the grid."""
    target = merge_grid_target(800, 1000, _CAP)
    assert target == (784, 980)  # 28*28 and 28*35 — the nearest grid point at or below each side


def test_an_already_aligned_page_is_left_completely_alone() -> None:
    """Byte-identical passthrough when there is nothing to fix — no needless resample."""
    assert merge_grid_target(1120, 1540, _CAP) is None
    assert merge_grid_target(MERGE_GRID, MERGE_GRID, _CAP) is None


def test_a_side_too_narrow_to_align_is_refused_rather_than_enlarged() -> None:
    """A 30-px-wide strip has one patch row; aligning it would mean upscaling, so we do not."""
    assert merge_grid_target(3000, 20, _CAP) is None
    assert merge_grid_target(0, 500, _CAP) is None


def test_disabling_the_cap_disables_the_grid_too() -> None:
    """``max_longest_dim = 0`` means "hand me the raw raster" — a deliberate, documented bypass."""
    assert merge_grid_target(1700, 2200, 0) is None


# --- the same arithmetic, through the code the Provider actually runs ---------------------------


class _SizedImage:
    """The narrowest possible stand-in for a Pillow image: a size, and a resize that records."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.resized_to: tuple[int, int] | None = None

    def resize(self, size: tuple[int, int], _resample: Any = None) -> _SizedImage:
        out = _SizedImage(size)
        out.resized_to = size
        return out


@contextmanager
def _noop_guard(_op: str) -> Iterator[None]:
    yield


def _rasteriser(max_longest_dim: int = _CAP) -> PageRasteriser:
    return PageRasteriser(dpi=200, max_longest_dim=max_longest_dim, guard=_noop_guard)


def test_the_rasteriser_applies_the_grid(fake_lighton: None) -> None:
    """``_downscale_longest`` — the one function both OCR back-ends share — must do the rounding."""
    resized = _rasteriser()._downscale_longest(_SizedImage((1700, 2200)))
    assert resized.size == (1176, 1540)

    untouched = _rasteriser()._downscale_longest(_SizedImage((1120, 1540)))
    assert untouched.resized_to is None


def test_the_provider_path_produces_grid_aligned_pages(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """End to end through the Provider: decoded bytes -> a page image on the merge grid."""
    img_bytes = make_lighton_image("scanned page text\n", dims=(1700, 2200))
    image = LightOnOcrParser({})._image_from_bytes(img_bytes)
    width, height = image.size
    assert (width % MERGE_GRID, height % MERGE_GRID) == (0, 0)
    assert max(width, height) <= _CAP
    assert (width, height) == (1176, 1540)

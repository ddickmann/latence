"""A faithful, offline stand-in for a materialised olmOCR-bench root.

Faithful means two things, and the second is the one that matters: the fixture reproduces the
real dataset's on-disk layout (``bench_data/<split>.jsonl`` + ``bench_data/pdfs/<path>.pdf``)
and its exact row schema — *and* it is built from **real PDF bytes** that the real
``parser.pdfplumber`` Provider actually parses. Nothing here fakes the Parse Stage, so a test
that passes against this fixture exercises the same code path a full-dataset run does.

The PDFs are generated (not committed binaries) by :func:`minimal_pdf`, which writes a genuine
single-page PDF with a Helvetica text object. If pdfplumber could not read it, every test built
on it would fail loudly rather than quietly measuring nothing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from latence_benchmark.parse_suite import OLMOCR_SPLITS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def minimal_pdf(lines: Sequence[str]) -> bytes:
    """Return a real single-page PDF rendering ``lines`` top-to-bottom in Helvetica."""
    text_ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    text_ops += [f"({_escape(line)}) Tj T*" for line in lines]
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n".encode()
        + b"%%EOF\n"
    )
    return bytes(out)


def _write_dataset(root: Path, splits: dict[str, list[dict[str, Any]]]) -> Path:
    """Write the FULL manifest — every split file the pinned spec declares.

    Splits the test does not populate are written empty rather than omitted, because the real
    dataset always has all seven and the manifest check (rightly) refuses a partial root.
    """
    bench = root / "bench_data"
    (bench / "pdfs").mkdir(parents=True, exist_ok=True)
    for split in OLMOCR_SPLITS:
        rows = splits.get(split, [])
        (bench / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root


@pytest.fixture
def write_dataset() -> Callable[[Path, dict[str, list[dict[str, Any]]]], Path]:
    """Write ``{split: [row, ...]}`` into a dataset root in the real on-disk layout."""
    return _write_dataset


@pytest.fixture
def write_pdf() -> Callable[[Path, str, Sequence[str]], None]:
    """Write a real PDF at ``root/bench_data/pdfs/<relative>`` rendering ``lines``."""

    def _write(root: Path, relative: str, lines: Sequence[str]) -> None:
        path = root / "bench_data" / "pdfs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(minimal_pdf(lines))

    return _write

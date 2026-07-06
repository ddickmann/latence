"""Unit tests for the bundled messy corpus (S13).

The corpus is generated deterministically in code (no opaque binaries); these assert it is
reproducible, that the planted dangerous fixtures are really dangerous (a spoofed PDF, a
high-ratio ZIP, an oversized file), and that a real multi-page PDF is present.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from latence_core.storage import Storage
from latence_demo.corpus import build_corpus_bytes, stage_corpus


def test_corpus_is_deterministic() -> None:
    a, ma = build_corpus_bytes()
    b, mb = build_corpus_bytes()
    assert a == b  # byte-identical corpus every time (Baseline bar)
    assert ma == mb


def test_corpus_has_mixed_types_and_a_real_pdf() -> None:
    docs, manifest = build_corpus_bytes()
    assert {"memo.md", "note.txt", "report.pdf"} <= set(docs)
    # A real PDF (starts with the %PDF- signature), not a text file renamed .pdf.
    assert docs["report.pdf"].startswith(b"%PDF-")
    assert manifest.documents["report.pdf"] == len(docs["report.pdf"])


def test_dangerous_fixtures_are_actually_dangerous() -> None:
    docs, manifest = build_corpus_bytes()
    # type-spoof: a .pdf whose bytes are not a PDF.
    assert "invoice.pdf" in manifest.dangerous
    assert not docs["invoice.pdf"].startswith(b"%PDF-")
    # zip bomb: a real ZIP whose declared uncompressed size vastly exceeds its on-disk size.
    assert "archive.zip" in manifest.dangerous
    with zipfile.ZipFile(io.BytesIO(docs["archive.zip"])) as zf:
        uncompressed = sum(i.file_size for i in zf.infolist())
    ratio = uncompressed / max(len(docs["archive.zip"]), 1)
    assert ratio > 50.0  # exceeds the demo's intake ratio cap
    # oversized: over the demo's 2 MiB intake cap.
    assert len(docs["dump.txt"]) > 2 * 1024 * 1024


def test_manifest_document_ids_match_source_content_addressing() -> None:
    # The manifest's document_ids must equal the exact id the Source stamps on each file:
    # "sha256:" + sha256(bytes). This is the key _score_dangerous binds findings to files by;
    # a drift here (or a reverted manifest field) would silently break per-file scoring.
    docs, manifest = build_corpus_bytes()
    assert set(manifest.document_ids) == set(docs)
    for name, data in docs.items():
        assert manifest.document_ids[name] == "sha256:" + hashlib.sha256(data).hexdigest()
    # The dangerous fixtures each have a resolvable, distinct document_id.
    dangerous_ids = {manifest.document_ids[n] for n in manifest.dangerous}
    assert len(dangerous_ids) == len(manifest.dangerous)


def test_stage_corpus_writes_every_file(tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    storage = Storage.for_uri(store)
    folder = f"{store}/corpus"
    manifest = stage_corpus(storage, folder)
    for name in manifest.documents:
        assert storage.exists(f"{folder}/{name}")
    # Round-trips the exact bytes onto Storage.
    docs, _ = build_corpus_bytes()
    assert storage.read_bytes(f"{folder}/report.pdf") == docs["report.pdf"]

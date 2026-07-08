"""Unit tests for the bundled stack-validation corpus (P3-F4).

The corpus underpins the harness's determinism/resume checks, so its bytes MUST be reproducible,
and its planted dangerous fixtures MUST be present (the graceful-failure check depends on them).
"""

from __future__ import annotations

from pathlib import Path

from latence_core.stacks.corpus import (
    build_validation_corpus_bytes,
    stage_validation_corpus,
)
from latence_core.storage import Storage


def test_corpus_bytes_are_deterministic() -> None:
    """Two independent builds yield byte-identical documents (the seeded/reproducible bar)."""
    docs_a, manifest_a = build_validation_corpus_bytes()
    docs_b, manifest_b = build_validation_corpus_bytes()
    assert docs_a == docs_b
    assert manifest_a.document_ids == manifest_b.document_ids


def test_corpus_plants_the_four_dangerous_fixtures() -> None:
    """The type-spoof / zip-bomb / oversized / prompt-injection fixtures are all present."""
    _docs, manifest = build_validation_corpus_bytes()
    dispositions = set(manifest.dangerous.values())
    assert dispositions == {
        "quarantine:type_spoof",
        "quarantine:zip_bomb",
        "quarantine:oversized",
        "flag:prompt_injection",
    }
    # Every dangerous file is a real staged document with a content-addressed id.
    for name in manifest.dangerous:
        assert name in manifest.documents
        assert manifest.document_ids[name].startswith("sha256:")


def test_corpus_is_entity_rich() -> None:
    """The shared cross-document entities are present so the KG is non-empty."""
    docs, _manifest = build_validation_corpus_bytes()
    memo = docs["memo.md"].decode("utf-8")
    assert "Acme Corporation" in memo
    assert "International Business Machines" in memo  # collapses to the IBM node in Disambiguation
    # A real (dependency-free) PDF, not a renamed text file.
    assert docs["report.pdf"].startswith(b"%PDF-")
    # The type-spoof .pdf deliberately carries NO %PDF- signature (Intake Screening quarantines it).
    assert not docs["invoice.pdf"].startswith(b"%PDF-")


def test_stage_validation_corpus_writes_every_document(tmp_path: Path) -> None:
    """Staging writes every generated document under the folder, idempotently."""
    storage = Storage.for_uri(f"file://{tmp_path}")
    folder = f"file://{tmp_path}/corpus"
    manifest = stage_validation_corpus(storage, folder)
    for name in manifest.documents:
        assert storage.exists(f"{folder}/{name}")
    # Idempotent: re-staging overwrites with the same bytes (no error, same manifest).
    manifest2 = stage_validation_corpus(storage, folder)
    assert manifest2.document_ids == manifest.document_ids

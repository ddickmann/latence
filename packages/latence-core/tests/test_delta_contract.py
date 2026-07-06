"""S11 delta contract tests: Delta / DeltaEntry / CorpusVersion / DeltaChurn invariants."""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    CorpusVersion,
    DeletionMode,
    Delta,
    DeltaChurn,
    DeltaEntry,
    DeltaKind,
)
from pydantic import ValidationError


def test_delta_entry_deleted_has_no_hash() -> None:
    # A DELETED entry has no current content hash; passing one is a contract error.
    ok = DeltaEntry(kind=DeltaKind.DELETED, document_id="doc-1")
    assert ok.content_hash is None
    with pytest.raises(ValidationError):
        DeltaEntry(kind=DeltaKind.DELETED, document_id="doc-1", content_hash="sha256:x")


def test_delta_entry_non_deleted_requires_hash() -> None:
    for kind in (DeltaKind.ADDED, DeltaKind.UPDATED, DeltaKind.UNCHANGED):
        with pytest.raises(ValidationError):
            DeltaEntry(kind=kind, document_id="d")
        DeltaEntry(kind=kind, document_id="d", content_hash="sha256:a")  # ok


def test_delta_classifies_and_reports_changed() -> None:
    delta = Delta(
        parent_version=3,
        entries=[
            DeltaEntry(kind=DeltaKind.ADDED, document_id="b", content_hash="sha256:b"),
            DeltaEntry(kind=DeltaKind.ADDED, document_id="a", content_hash="sha256:a"),
            DeltaEntry(kind=DeltaKind.UPDATED, document_id="c", content_hash="sha256:c"),
            DeltaEntry(kind=DeltaKind.DELETED, document_id="d"),
            DeltaEntry(kind=DeltaKind.UNCHANGED, document_id="e", content_hash="sha256:e"),
        ],
    )
    assert delta.added == ["a", "b"]  # sorted
    assert delta.updated == ["c"]
    assert delta.deleted == ["d"]
    assert delta.unchanged == ["e"]
    assert delta.changed == ["a", "b", "c"]
    assert not delta.is_empty


def test_empty_delta_is_idempotent_noop() -> None:
    delta = Delta(
        parent_version=1,
        entries=[DeltaEntry(kind=DeltaKind.UNCHANGED, document_id="x", content_hash="sha256:x")],
    )
    assert delta.is_empty


def test_corpus_version_parent_must_be_lower() -> None:
    delta = Delta(entries=[])
    CorpusVersion(
        version=1, parent_version=0, delta=delta, record_count=0, document_count=0,
        fingerprint="sha256:z",
    )
    with pytest.raises(ValidationError):
        CorpusVersion(
            version=1, parent_version=1, delta=delta, record_count=0, document_count=0,
            fingerprint="sha256:z",
        )
    # Version 0 with no parent (bootstrap) is valid.
    CorpusVersion(
        version=0, parent_version=None, delta=delta, record_count=0, document_count=0,
        fingerprint="sha256:z",
    )


def test_delta_churn_defaults_and_versioning() -> None:
    churn = DeltaChurn(to_version=0)
    assert churn.schema_version == SCHEMA_VERSION
    assert churn.from_version is None
    assert churn.deletion_mode is DeletionMode.RETRACTION
    assert churn.records_purged == 0


def test_deletion_mode_values() -> None:
    assert DeletionMode.RETRACTION.value == "retraction"
    assert DeletionMode.PURGE.value == "purge"

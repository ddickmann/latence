"""S11 CorpusStore unit tests: read-N -> write-N+1 transaction, idempotency, inspection."""

from __future__ import annotations

from pathlib import Path

from latence_core.contracts import (
    CanonicalEntity,
    Classification,
    Delta,
    DeltaEntry,
    DeltaKind,
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    Provenance,
)
from latence_core.state.corpus_store import CorpusStore
from latence_core.storage import Storage


def _record(record_id: str, members: int) -> DisambiguationRecord:
    prov = Provenance(
        source_uri="mem://x",
        file_name="x.txt",
        file_type="txt",
        file_size=1,
        document_id="doc-a",
        char_start=0,
        char_end=1,
    )
    cls = Classification(language="en")
    entity = CanonicalEntity(
        record_id=record_id,
        canonical_text=record_id,
        entity_type="ORG",
        member_mention_ids=[f"{record_id}:m{i}" for i in range(members)],
        source_document_ids=["doc-a"],
        evidence=Evidence(mention_ids=[f"{record_id}:m0"], document_ids=["doc-a"], confidence=0.9),
        provenance=prov,
        classification=cls,
    )
    return DisambiguationRecord(
        record_id=f"{record_id}#r",
        scope=DisambiguationScope.ENTITY,
        entity=entity,
        provenance=prov,
        classification=cls,
    )


def _delta(added: list[str]) -> str:
    return Delta(
        entries=[DeltaEntry(kind=DeltaKind.ADDED, document_id=d, content_hash="sha256:h") for d in added]
    ).model_dump_json()


def _store(tmp_path: Path) -> CorpusStore:
    uri = f"file://{tmp_path}/store"
    return CorpusStore(Storage.for_uri(uri), uri)


def test_commit_produces_incrementing_versions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.head_version() is None
    v0 = store.commit(
        parent_version=None, records=[_record("acme", 1)], delta_json=_delta(["doc-a"]),
        document_count=1, reconciled=True,
    )
    assert v0.version == 0
    assert store.head_version() == 0
    v1 = store.commit(
        parent_version=0, records=[_record("acme", 2)], delta_json=_delta(["doc-b"]),
        document_count=2, reconciled=False,
    )
    assert v1.version == 1
    assert v1.parent_version == 0
    assert store.head_version() == 1


def test_idempotent_recommit_same_fingerprint_is_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit(
        parent_version=None, records=[_record("acme", 1)], delta_json=_delta(["doc-a"]),
        document_count=1, reconciled=True,
    )
    # Re-committing the SAME live set (same fingerprint) must not mint a new version.
    again = store.commit(
        parent_version=0, records=[_record("acme", 1)], delta_json=_delta([]),
        document_count=1, reconciled=False,
    )
    assert again.version == 0
    assert store.head_version() == 0


def test_content_change_mints_new_version_even_with_same_ids(tmp_path: Path) -> None:
    # A merge that keeps the lexicographic-min canonical id but changes membership MUST commit a
    # new version — the fingerprint hashes content, not just ids (the split/merge determinism).
    store = _store(tmp_path)
    store.commit(
        parent_version=None, records=[_record("acme", 1)], delta_json=_delta(["doc-a"]),
        document_count=1, reconciled=True,
    )
    changed = store.commit(
        parent_version=0, records=[_record("acme", 2)], delta_json=_delta(["doc-b"]),
        document_count=2, reconciled=False,
    )
    assert changed.version == 1


def test_prior_versions_inspectable_after_advance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit(
        parent_version=None, records=[_record("acme", 1)], delta_json=_delta(["doc-a"]),
        document_count=1, reconciled=True,
    )
    store.commit(
        parent_version=0, records=[_record("acme", 2)], delta_json=_delta(["doc-b"]),
        document_count=2, reconciled=False,
    )
    # Version 0 remains fully inspectable for audit/rollback.
    v0 = store.read_version(0)
    assert v0.version == 0
    v0_records = store.read_records(0)
    assert len(v0_records) == 1
    v0_rec = v0_records[0]
    assert isinstance(v0_rec, DisambiguationRecord)
    assert v0_rec.entity is not None
    assert v0_rec.entity.mention_count == 1  # the pre-merge state

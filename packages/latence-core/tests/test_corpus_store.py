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


def _record(record_id: str, members: int, doc: str = "doc-a") -> DisambiguationRecord:
    prov = Provenance(
        source_uri="mem://x",
        file_name="x.txt",
        file_type="txt",
        file_size=1,
        document_id=doc,
        char_start=0,
        char_end=1,
    )
    cls = Classification(language="en")
    entity = CanonicalEntity(
        record_id=record_id,
        canonical_text=record_id,
        entity_type="ORG",
        member_mention_ids=[f"{record_id}:m{i}" for i in range(members)],
        source_document_ids=[doc],
        evidence=Evidence(mention_ids=[f"{record_id}:m0"], document_ids=[doc], confidence=0.9),
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


def test_the_canonical_type_vocabulary_round_trips_per_version(tmp_path: Path) -> None:
    """T4: the vocabulary a delta EXTENDS is Corpus-Version state, versioned like the records.

    A Version that committed none reads back ``None`` (the pre-T4 / no-consolidation path stays
    byte-identical), and a prior Version keeps the vocabulary IT committed — so an audit of v0 sees
    v0's type space, not the head's.
    """
    from latence_core.contracts import CanonicalType, TypeKind, TypeVocabulary

    store = _store(tmp_path)
    store.commit(
        parent_version=None, records=[_record("acme", 1)], delta_json=_delta(["doc-a"]),
        document_count=1, reconciled=True,
    )
    assert store.head_type_vocabulary() is None

    vocabulary = TypeVocabulary(
        types=[CanonicalType(kind=TypeKind.ENTITY, canonical="organization", aliases=["org"])]
    )
    store.commit(
        parent_version=0, records=[_record("acme", 2)], delta_json=_delta(["doc-b"]),
        document_count=2, reconciled=False, type_vocabulary=vocabulary,
    )
    assert store.head_type_vocabulary() == vocabulary
    assert store.read_type_vocabulary(0) is None, "a prior Version keeps its own type space"


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


# -- Purge integrity (H-D1): abandoned-WAL residue + expunge re-fingerprint --


def _stage_abandoned(store: CorpusStore, version: int, records: list[DisambiguationRecord]) -> str:
    """Simulate a crash mid-commit: open a WAL transaction, stage records.jsonl, no manifest.

    Reproduces the abandoned-transaction residue the audit found (security S2): a fully-staged
    ``wal/v<N>/records.jsonl`` with no ``MANIFEST.json`` — invisible to ``read_records`` but on
    disk in cleartext. Returns the staged records.jsonl URI.
    """
    wal = store._wal  # noqa: SLF001 — test drives the WAL seam the crash path leaves behind
    wal.begin(version)
    uri = wal.stage_text(version, "records.jsonl", store._encode_records(list(records)))  # noqa: SLF001
    assert not wal.is_committed(version)  # the manifest was never written — abandoned
    return uri


def test_expunge_reconciles_head_fingerprint_and_count(tmp_path: Path) -> None:
    # Gap 2 (correctness S3): after expunging a purged doc's records from a committed Version, its
    # version.json fingerprint + record_count must describe the POST-purge set — not the stale
    # pre-purge one — or commit()'s idempotency check compares against a lie.
    store = _store(tmp_path)
    store.commit(
        parent_version=None,
        records=[_record("acme", 1, doc="doc-a"), _record("globex", 1, doc="doc-keep")],
        delta_json=_delta(["doc-a", "doc-keep"]),
        document_count=2,
        reconciled=True,
    )
    stale = store.read_version(0)
    assert stale.record_count == 2

    removed = store.expunge_records({"doc-a"})
    assert removed == 1

    live = store.read_records(0)
    reconciled = store.read_version(0)
    # The stored manifest now matches the erased reality...
    assert reconciled.record_count == 1
    assert len(live) == 1
    # ...and its fingerprint equals what commit() would compute for the identical live set — the
    # exact idempotency compare, using the same _fingerprint helper (no spurious duplicate Version).
    assert reconciled.fingerprint == CorpusStore._fingerprint(live)  # noqa: SLF001
    assert reconciled.fingerprint != stale.fingerprint
    # Immutable lineage is preserved (only the two erasure-derived fields changed).
    assert reconciled.version == 0
    assert reconciled.parent_version == stale.parent_version
    assert reconciled.created_ts == stale.created_ts


def test_expunge_sweeps_abandoned_wal_staging_records(tmp_path: Path) -> None:
    # Gap 1 (security S2): a crash mid-commit leaves an abandoned wal/v<N>/records.jsonl reaching
    # doc-a in cleartext. Purge MUST neutralise it — no records.jsonl anywhere (committed OR
    # abandoned) may reach a purged doc afterwards.
    store = _store(tmp_path)
    store.commit(
        parent_version=None,
        records=[_record("acme", 1, doc="doc-a")],
        delta_json=_delta(["doc-a"]),
        document_count=1,
        reconciled=True,
    )
    # Crash mid-commit on the NEXT version: staged records reach doc-a, but no manifest lands.
    abandoned_uri = _stage_abandoned(
        store, 1, [_record("acme", 2, doc="doc-a"), _record("globex", 1, doc="doc-keep")]
    )
    assert store._wal.abandoned_versions() == (1,)  # noqa: SLF001
    storage = Storage.for_uri(f"file://{tmp_path}/store")
    assert "doc-a" in storage.read_text(abandoned_uri)  # the residue is really there pre-purge

    store.expunge_records({"doc-a"})

    # The abandoned staging file survives (still uncommitted) but no longer reaches doc-a; the
    # co-resident doc-keep record is untouched (reach-scoped, not a blanket wipe).
    surviving = storage.read_text(abandoned_uri)
    assert "doc-a" not in surviving
    assert "doc-keep" in surviving
    # And NO records.jsonl anywhere under the corpus root reaches doc-a (the DoD tree walk).
    for path in (tmp_path / "store").rglob("records.jsonl"):
        assert "doc-a" not in path.read_text(encoding="utf-8")


def test_expunge_leaves_abandoned_dir_present_for_concurrency_safety(tmp_path: Path) -> None:
    # Concurrency guard: the abandoned staging dir is rewritten IN PLACE (shape (a)), never
    # remove_tree'd (shape (b)) — an atomic in-place rewrite cannot tear a concurrent run's
    # in-flight transaction. The v<N>/ dir and its (now-clean) records.jsonl still exist.
    store = _store(tmp_path)
    store.commit(
        parent_version=None,
        records=[_record("acme", 1, doc="doc-a")],
        delta_json=_delta(["doc-a"]),
        document_count=1,
        reconciled=True,
    )
    _stage_abandoned(store, 1, [_record("acme", 2, doc="doc-a")])
    store.expunge_records({"doc-a"})
    staging_dir = tmp_path / "store" / "_latence" / "corpus" / "wal" / "v1"
    assert staging_dir.is_dir()  # not deleted out from under a possible live recovery
    assert (staging_dir / "records.jsonl").exists()
    assert not store._wal.is_committed(1)  # still abandoned — we did not promote it  # noqa: SLF001


def test_expunge_purge_then_recommit_is_a_noop(tmp_path: Path) -> None:
    # The end-to-end correctness property the auditor reproduced: commit, purge doc-a, then
    # re-apply the identical delta over the post-purge live set. With the head manifest reconciled,
    # commit() must return the EXISTING head (a no-op), not mint a spurious duplicate Version.
    store = _store(tmp_path)
    store.commit(
        parent_version=None,
        records=[_record("acme", 1, doc="doc-a"), _record("globex", 1, doc="doc-keep")],
        delta_json=_delta(["doc-a", "doc-keep"]),
        document_count=2,
        reconciled=True,
    )
    store.expunge_records({"doc-a"})
    head_before = store.head_version()
    live = store.read_records(head_before or 0)
    # Re-apply the identical (post-purge) live set — the same corpus content.
    again = store.commit(
        parent_version=head_before,
        records=live,
        delta_json=_delta([]),
        document_count=1,
        reconciled=False,
    )
    assert again.version == head_before  # the existing head, NOT a new duplicate
    assert store.head_version() == head_before


def test_expunge_no_collateral_leaves_unaffected_versions_untouched(tmp_path: Path) -> None:
    # A Version that does not reach the purged doc must not be rewritten OR re-fingerprinted — its
    # manifest stays byte-identical (no gratuitous manifest churn on erasure).
    store = _store(tmp_path)
    store.commit(
        parent_version=None,
        records=[_record("globex", 1, doc="doc-keep")],
        delta_json=_delta(["doc-keep"]),
        document_count=1,
        reconciled=True,
    )
    before = store.read_version(0)
    removed = store.expunge_records({"doc-a"})  # doc-a appears nowhere
    assert removed == 0
    after = store.read_version(0)
    assert after == before  # unchanged manifest, unchanged fingerprint/count

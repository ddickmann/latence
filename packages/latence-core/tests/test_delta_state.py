"""S11 ported-engine unit tests: WriteAheadLog, DeltaDetector, GraphPatcher, drift, affected set."""

from __future__ import annotations

from pathlib import Path

from latence_core.contracts import (
    CanonicalEntity,
    Classification,
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    Provenance,
)
from latence_core.state.delta_detector import DeltaDetector, ManifestEntry, content_hash
from latence_core.state.graph_patcher import GraphPatcher
from latence_core.state.incremental import (
    MentionView,
    compute_affected_set,
    compute_drift,
)
from latence_core.state.write_ahead_log import WriteAheadLog
from latence_core.storage import Storage

# -- Write-Ahead Log ---------------------------------------------------------


def _wal(tmp_path: Path) -> tuple[WriteAheadLog, Storage]:
    store = f"file://{tmp_path}/wal"
    storage = Storage.for_uri(store)
    return WriteAheadLog(storage, store), storage


def test_wal_commit_then_recover_finds_head(tmp_path: Path) -> None:
    wal, _ = _wal(tmp_path)
    wal.begin(0)
    uri = wal.stage_text(0, "records.jsonl", "hello\n")
    wal.commit(0, files={"records": uri}, metadata={"kind": "add"})
    assert wal.is_committed(0)
    rec = wal.recover()
    assert rec.committed == 0
    assert rec.abandoned == ()
    manifest = wal.read_manifest(0)
    assert manifest["committed"] is True


def test_wal_abandoned_transaction_leaves_prior_head_intact(tmp_path: Path) -> None:
    wal, storage = _wal(tmp_path)
    # Commit version 0.
    wal.begin(0)
    u0 = wal.stage_text(0, "records.jsonl", "v0\n")
    wal.commit(0, files={"records": u0}, metadata={})
    # Begin version 1 but crash BEFORE commit (only stage, no manifest).
    wal.begin(1)
    wal.stage_text(1, "records.jsonl", "v1-partial\n")
    rec = wal.recover()
    # The last committed head is still 0; version 1 is abandoned (uncommitted).
    assert rec.committed == 0
    assert rec.abandoned == (1,)
    assert not wal.is_committed(1)


def test_wal_committed_versions_ascending(tmp_path: Path) -> None:
    wal, _ = _wal(tmp_path)
    for v in (0, 1, 2):
        wal.begin(v)
        u = wal.stage_text(v, "r.jsonl", f"v{v}")
        wal.commit(v, files={"r": u}, metadata={})
    assert wal.committed_versions() == [0, 1, 2]


# -- Delta detector ----------------------------------------------------------


def test_content_hash_is_deterministic_and_order_independent() -> None:
    a = content_hash("doc", "text", ["Bob", "Alice"], ["works_for"])
    b = content_hash("doc", "text", ["Alice", "Bob"], ["works_for"])
    assert a == b  # entity order does not matter
    c = content_hash("doc", "text", ["Alice"], ["works_for"])
    assert a != c  # a changed derived set flips the hash


def test_content_hash_reextraction_flips_even_on_same_markdown() -> None:
    same_text = content_hash("doc", "same markdown")
    with_entities = content_hash("doc", "same markdown", ["Acme"])
    assert same_text != with_entities  # a re-extraction is an UPDATE, not UNCHANGED


def test_delta_detector_classifies_add_update_delete_unchanged() -> None:
    prev = {
        "a": ManifestEntry("a", "sha256:1"),
        "b": ManifestEntry("b", "sha256:2"),
        "c": ManifestEntry("c", "sha256:3"),
    }
    curr = {
        "a": ManifestEntry("a", "sha256:1"),  # unchanged
        "b": ManifestEntry("b", "sha256:CHANGED"),  # updated
        "d": ManifestEntry("d", "sha256:4"),  # added
    }
    diff = DeltaDetector.diff(prev, curr)
    assert diff.added == ["d"]
    assert diff.updated == ["b"]
    assert diff.deleted == ["c"]
    assert diff.unchanged == ["a"]
    assert diff.changed == ["b", "d"]
    assert not diff.is_empty


def test_delta_detector_no_change_is_empty() -> None:
    m = {"a": ManifestEntry("a", "sha256:1")}
    assert DeltaDetector.diff(m, dict(m)).is_empty


# -- Graph patcher -----------------------------------------------------------


def _entity(record_id: str, docs: list[str]) -> DisambiguationRecord:
    prov = Provenance(
        source_uri="mem://x",
        file_name="x.txt",
        file_type="txt",
        file_size=1,
        document_id=docs[0],
        char_start=0,
        char_end=1,
    )
    cls = Classification(language="en")
    entity = CanonicalEntity(
        record_id=record_id,
        canonical_text=record_id,
        entity_type="ORG",
        member_mention_ids=[f"{record_id}:m0"],
        source_document_ids=docs,
        evidence=Evidence(mention_ids=[f"{record_id}:m0"], document_ids=docs, confidence=0.9),
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


def test_graph_patcher_retracts_by_reach() -> None:
    only_a = _entity("only_a", ["doc-a"])
    only_b = _entity("only_b", ["doc-b"])
    both = _entity("both", ["doc-a", "doc-b"])
    result = GraphPatcher.retract([only_a, only_b, both], {"doc-b"})
    kept_ids = {r.record_id for r in result.kept}
    retracted_ids = {r.record_id for r in result.retracted}
    partial_ids = {r.record_id for r in result.partially_affected}
    assert kept_ids == {"only_a#r"}  # no reach into doc-b
    assert retracted_ids == {"only_b#r"}  # reach entirely within doc-b
    assert partial_ids == {"both#r"}  # reach spans deleted + surviving
    assert "doc-a" in result.affected_document_ids  # the surviving side to re-resolve


def test_graph_patcher_empty_delete_keeps_all() -> None:
    e = _entity("e", ["doc-a"])
    result = GraphPatcher.retract([e], set())
    assert result.kept == [e]
    assert not result.retracted


# -- drift + affected set ----------------------------------------------------


def test_drift_threshold_trigger() -> None:
    below = compute_drift(1, 10, threshold=0.5)
    assert below.drift == 0.1 and not below.triggered
    above = compute_drift(6, 10, threshold=0.5)
    assert above.drift == 0.6 and above.triggered
    # A zero-doc corpus never divides by zero.
    assert compute_drift(0, 0, threshold=0.5).drift == 0.0


def test_affected_set_grows_transitive_neighborhood() -> None:
    mentions = [
        MentionView("m1", "doc-a", "International Business Machines", "ORG", 1.0),
        MentionView("m2", "doc-b", "IBM Corporation", "ORG", 1.0),  # shares "ibm"? no; shares nothing
        MentionView("m3", "doc-c", "Acme", "ORG", 1.0),
    ]
    # Seed with doc-a: m1's tokens include "international","business","machines".
    affected = compute_affected_set(mentions, ["doc-a"])
    assert "m1" in affected.mention_ids
    # m3 (Acme) shares no token with the seed, so it is NOT pulled in.
    assert "m3" not in affected.mention_ids


def test_affected_set_full_is_whole_corpus() -> None:
    mentions = [
        MentionView("m1", "doc-a", "Acme", "ORG", 1.0),
        MentionView("m2", "doc-b", "Globex", "ORG", 1.0),
    ]
    affected = compute_affected_set(mentions, [], full=True)
    assert affected.is_full
    assert affected.mention_ids == {"m1", "m2"}


# -- Source-level exclusion (the deletion re-run seam, S11) -------------------


def test_source_excludes_deleted_documents(tmp_path: Path) -> None:
    from latence_core.stages.source import LocalFolderSource, _document_id

    docs = tmp_path / "docs"
    docs.mkdir()
    a = docs / "a.txt"
    a.write_text("alpha\n", encoding="utf-8")
    (docs / "b.txt").write_text("beta\n", encoding="utf-8")
    a_id = _document_id(a.read_bytes())

    storage = Storage.for_uri(f"file://{docs}")
    # Without exclusions, both documents are produced.
    src = LocalFolderSource({"path": f"file://{docs}", "extensions": ["txt"]})
    all_ids = {pi.provenance.document_id for pi in src.produce(storage)}
    assert a_id in all_ids and len(all_ids) == 2

    # Excluding a's content id at Source drops it (the Retraction/Purge re-run seam).
    src2 = LocalFolderSource(
        {"path": f"file://{docs}", "extensions": ["txt"], "exclude_document_ids": [a_id]}
    )
    kept = {pi.provenance.document_id for pi in src2.produce(storage)}
    assert a_id not in kept and len(kept) == 1

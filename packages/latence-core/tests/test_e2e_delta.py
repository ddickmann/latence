"""End-to-end S11 incremental deltas through the DeltaRunner.

Covers every S11 acceptance criterion over a real Source->Parse->Chunk->Entity->Relation->
Disambiguation->Graph->Export pipeline driven by the in-core deterministic Providers:

* adding a bridging file MERGES two prior clusters, touching only the affected set (merge-on-add);
* retracting a file tombstones its records, excludes them, retains them for audit, and SPLITS a
  merge justified solely by it (split-on-delete);
* purging a file physically removes its derived records — nothing derived survives;
* a drift threshold triggers a full Reconciliation, reported in the churn;
* re-running the same delta is idempotent (no new Version); a prior Version stays inspectable;
* the delta churn lands in the Quality Report.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from latence_core.contracts import DeletionMode, DeltaKind, DisambiguationRecord, Record
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.state.corpus_store import CorpusStore
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: delta-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}', sensitivity: internal, extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
    config:
      labels:
        ORG: {{terms: [Acme, International Business Machines, IBM]}}
        PERSON: {{terms: [Alice, Bob]}}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      relations:
        works_for: {{head: PERSON, tail: ORG, triggers: [works at, joined]}}
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [graph]
"""
    )


def _doc_id(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record_reach_ids(record: object) -> set[str]:
    # The source documents a committed record draws on — the same reach signal expunge filters
    # by (Evidence.document_ids ∪ source_document_ids ∪ provenance.document_id), read via the
    # in-core reach helper so the test asserts on exactly what the retract/expunge uses.
    from latence_core.state.graph_patcher import _record_reach

    return set(_record_reach(record))  # type: ignore[arg-type]


def _entity_sizes(records: list[Record]) -> dict[str, int]:
    return {
        r.entity.canonical_text: r.entity.mention_count
        for r in records
        if isinstance(r, DisambiguationRecord) and r.entity is not None
    }


def test_delta_add_merges_and_reconciliation_reported(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)

    # V0: one document. Bootstrap -> full reconciliation (drift 1.0).
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    r0 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    assert r0.delta is not None
    assert r0.delta.to_version == 0
    assert r0.delta.reconciliation_triggered  # bootstrap drift crosses threshold
    assert r0.delta.documents_added == 1
    assert cs.head_version() == 0

    # V1: add a bridging document with "International Business Machines" -> merges with "IBM".
    (docs / "b.txt").write_text(
        "International Business Machines partnered with Acme.\n", encoding="utf-8"
    )
    r1 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    assert r1.delta is not None
    assert r1.delta.to_version == 1
    assert r1.delta.documents_added == 1
    assert r1.delta.entities_merged >= 1  # merge-on-add caught
    sizes = _entity_sizes(cs.read_records(1))
    assert sizes.get("IBM", 0) >= 2  # IBM absorbed International Business Machines across docs


def test_add_touches_only_the_affected_set(tmp_path: Path) -> None:
    # Adding a file must update the corpus touching ONLY the affected set: an entity from an
    # untouched, unrelated document is carried forward VERBATIM (same record bytes), not
    # re-derived — the ADR-0018 "recompute only the affected set" property.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    # V0: two unrelated documents (Acme; Alice+IBM).
    (docs / "acme.txt").write_text("Acme is a company.\n", encoding="utf-8")
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    v0 = {r.record_id: r.model_dump_json() for r in cs.read_records(0)}

    # V1: add a bridging doc that merges IBM — Acme's cluster (2 docs) is in the affected region,
    # but a genuinely untouched entity's serialized bytes must be preserved across the delta.
    (docs / "b.txt").write_text(
        "International Business Machines partnered with Globex.\n", encoding="utf-8"
    )
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    v1 = {r.record_id: r.model_dump_json() for r in cs.read_records(1)}
    # Alice (a PERSON in doc-a, untouched by the IBM/Globex bridge) is carried forward unchanged.
    alice_ids = [rid for rid in v0 if "alice" in rid.lower()]
    assert alice_ids
    for rid in alice_ids:
        assert rid in v1
        assert v0[rid] == v1[rid]  # byte-identical: not re-derived


def test_surgical_merge_on_add_and_split_on_delete_below_drift(tmp_path: Path) -> None:
    # Prove the affected-set SURGICAL path end-to-end — NOT the full-Reconciliation fallback. The
    # corpus is sized with enough unrelated documents that a single change stays below the 0.5
    # drift threshold, so merge-on-add and split-on-delete are produced by the GraphPatcher splice
    # (reconciliation_triggered is False), not by a full re-resolve. This is AC#1's raison d'être;
    # the smaller fixtures elsewhere cross the threshold and silently run reconciliation instead.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)

    # V0: six unrelated documents (one carrying IBM, and filler docs mentioning the untouched
    # gazetteer entities Bob/Alice/Acme) — a big enough corpus that one change is < 0.5 drift.
    # Bootstrap itself reconciles (drift 1.0), which is expected.
    (docs / "a.txt").write_text("Alice works at IBM.\n", encoding="utf-8")
    (docs / "c.txt").write_text("Bob is here.\n", encoding="utf-8")
    (docs / "d.txt").write_text("Acme is a company.\n", encoding="utf-8")
    (docs / "e.txt").write_text("Bob visited Acme.\n", encoding="utf-8")
    (docs / "g.txt").write_text("Alice is here too.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    assert cs.head_version() == 0

    # V1: add ONE bridging document that merges IBM with International Business Machines.
    # 1 changed / 6 docs = 0.17 drift < 0.5 -> the SURGICAL merge-on-add, no reconciliation.
    (docs / "b.txt").write_text(
        "International Business Machines partnered with Globex.\n", encoding="utf-8"
    )
    r1 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    assert r1.delta is not None
    assert r1.delta.reconciliation_triggered is False  # the surgical path ran, not reconciliation
    assert r1.delta.drift < 0.5
    assert r1.delta.entities_merged >= 1  # merge-on-add via the affected-set splice
    assert _entity_sizes(cs.read_records(1)).get("IBM", 0) >= 2

    # The perf-baseline witness (issue #42): the committed corpus-level recompute was
    # affected-set-scoped, NOT O(corpus). The blocking-neighborhood closure re-resolved at least
    # one record (the merged IBM region) but STRICTLY fewer than the whole current corpus-level
    # set — the untouched Bob/Alice/Acme clusters were carried forward, not recomputed. Would fail
    # if the affected-set wiring were reverted (the metric would default to 0, not >= 1).
    assert r1.delta.affected_set_size >= 1
    assert r1.delta.corpus_record_count > r1.delta.affected_set_size

    # A genuinely untouched entity (Bob, whose reach never touches the IBM/Globex bridge) is
    # carried forward byte-identical across the delta — the surgical splice preserves it; a full
    # recompute would re-derive it. Because reconciliation did NOT run (pinned above), byte-equality
    # here proves the affected-set carry-forward, not a coincidental identical re-derivation.
    v0 = {r.record_id: r.model_dump_json() for r in cs.read_records(0)}
    v1 = {r.record_id: r.model_dump_json() for r in cs.read_records(1)}
    bob_ids = [rid for rid in v0 if "bob" in rid.lower()]
    assert bob_ids
    for rid in bob_ids:
        assert rid in v1
        assert v0[rid] == v1[rid]

    # Retract the bridging document b -> IBM splits back apart. 1 changed / 5 surviving docs = 0.2
    # drift < 0.5 -> the SURGICAL split-on-delete (GraphPatcher retract + partially-affected).
    b_id = _doc_id(docs / "b.txt")
    rr = DeltaRunner().retract(_pipeline(store, docs), run_id="ret", document_ids=[b_id])
    assert rr.delta is not None
    assert rr.delta.reconciliation_triggered is False  # surgical, not reconciliation
    assert rr.delta.drift < 0.5
    assert rr.delta.entities_split >= 1  # split-on-delete via the affected-set retract
    assert _entity_sizes(cs.read_records(cs.head_version() or 0)).get("IBM", 0) == 1


def test_edit_then_revert_restores_content(tmp_path: Path) -> None:
    # Editing a file then reverting it to earlier content must restore that content. Because
    # document ids are content-addressed, an in-place edit retires the old content-id (the detector
    # marks it DELETED) and mints a new one — but that incidental DELETED is NOT a user-requested
    # retraction, so it must NOT permanently blacklist the old content at Source. Regression guard
    # for the edit-then-revert data-loss bug (only explicit retract/purge tombstones are durable).
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)

    # V0: a.txt mentions Acme.
    (docs / "a.txt").write_text("Acme is a company.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    assert "Acme" in _entity_sizes(cs.read_records(cs.head_version() or 0))

    # V1: edit a.txt -> IBM (Acme's content-id is retired; the detector records it DELETED).
    (docs / "a.txt").write_text("IBM is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    v1_sizes = _entity_sizes(cs.read_records(cs.head_version() or 0))
    assert "IBM" in v1_sizes
    assert "Acme" not in v1_sizes

    # V2: revert a.txt back to the ORIGINAL Acme content. The reverted content-id must NOT be
    # blacklisted at Source (that was only an edit, never a retraction) — the corpus reflects the
    # current source folder, so Acme reappears.
    (docs / "a.txt").write_text("Acme is a company.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v2")
    v2_sizes = _entity_sizes(cs.read_records(cs.head_version() or 0))
    assert "Acme" in v2_sizes  # reverted content restored, not silently dropped
    assert "IBM" not in v2_sizes


def test_local_folder_edit_is_added_plus_deleted_never_updated(tmp_path: Path) -> None:
    # Issue #42.3: under the content-addressed source.local_folder, DeltaKind.UPDATED is
    # unreachable — an in-place edit mints a NEW content-id (ADDED) and retires the old one
    # (DELETED), never UPDATED (which requires a content-independent stable id). This pins the
    # documented behavior: it fails if the local-folder Source stopped content-addressing (which
    # would make UPDATED reachable and the DeltaKind/DeltaDetector docstrings stale).
    from latence_core.contracts import DeltaKind
    from latence_core.state.corpus_store import CorpusStore

    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Acme is a company.\n", encoding="utf-8")
    old_id = _doc_id(docs / "a.txt")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    # Edit a.txt in place: same file name, new bytes -> new content-id.
    (docs / "a.txt").write_text("IBM is here.\n", encoding="utf-8")
    new_id = _doc_id(docs / "a.txt")
    assert new_id != old_id  # content-addressing minted a fresh id
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")

    entries = cs.read_version(cs.head_version() or 0).delta.entries
    kinds = {e.document_id: e.kind for e in entries}
    # The edit is classified ADDED (new id) + DELETED (old id) — NOT UPDATED.
    assert kinds.get(new_id) is DeltaKind.ADDED
    assert kinds.get(old_id) is DeltaKind.DELETED
    assert all(e.kind is not DeltaKind.UPDATED for e in entries)


def test_retracted_document_stays_gone_across_reruns(tmp_path: Path) -> None:
    # The dual of the edit-then-revert guard: an EXPLICIT retraction IS durable. A retracted
    # document, whose file is still on disk, must not silently resurrect on the next apply_delta —
    # only intentional tombstones stay excluded at Source.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Acme is a company.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Alice is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    a_id = _doc_id(docs / "a.txt")
    DeltaRunner().retract(_pipeline(store, docs), run_id="ret", document_ids=[a_id])
    assert "Acme" not in _entity_sizes(cs.read_records(cs.head_version() or 0))

    # A subsequent apply_delta over the still-present a.txt must NOT resurrect the retracted Acme.
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v2")
    assert "Acme" not in _entity_sizes(cs.read_records(cs.head_version() or 0))


def test_delta_reruns_are_idempotent(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    # Re-running the same corpus commits no new Version (fingerprint matches head).
    r = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0b")
    assert r.delta is not None
    assert r.delta.to_version == 0
    assert cs.head_version() == 0


def test_retraction_splits_and_retains_for_audit(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    (docs / "b.txt").write_text(
        "International Business Machines partnered with Acme.\n", encoding="utf-8"
    )
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    merged = _entity_sizes(cs.read_records(cs.head_version() or 0))
    assert merged.get("IBM", 0) >= 2

    # Retract the bridging document b -> IBM must split back apart (its bridge is gone).
    b_id = _doc_id(docs / "b.txt")
    rr = DeltaRunner().retract(_pipeline(store, docs), run_id="ret", document_ids=[b_id])
    assert rr.delta is not None
    assert rr.delta.documents_deleted == 1
    assert rr.delta.deletion_mode is DeletionMode.RETRACTION
    assert rr.delta.entities_split >= 1  # split-on-delete
    assert rr.delta.records_retracted >= 1
    assert rr.delta.records_purged == 0  # a retraction purges nothing
    after = _entity_sizes(cs.read_records(cs.head_version() or 0))
    assert after.get("IBM", 0) == 1  # back to a single-document cluster

    # The retracted records are RETAINED for audit: prior Versions stay inspectable.
    v1 = cs.read_records(1)
    assert _entity_sizes(v1).get("IBM", 0) >= 2  # the pre-retraction merged state survives in v1


def test_purge_removes_all_derived_records(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    a_id = _doc_id(docs / "a.txt")
    rp = DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[a_id])
    assert rp.delta is not None
    assert rp.delta.deletion_mode is DeletionMode.PURGE
    assert rp.delta.documents_deleted == 1
    assert rp.delta.records_purged >= 1
    assert rp.delta.records_retracted == 0
    # Nothing derived survives: the live set for the head Version is empty.
    live = cs.read_records(cs.head_version() or 0)
    assert live == []


def test_purge_leaves_no_trace_in_prior_versions_or_source(tmp_path: Path) -> None:
    # Purge is a hard GDPR erasure: verified NOTHING derived from the PURGED document survives
    # (S11 AC) — AND that a co-resident, NON-purged document's records are NOT collaterally erased
    # (expunge_records filters by reach-intersection, issue #42). Beyond the head Version omitting
    # the purged records, the purged document's raw text and derived records must not linger in any
    # prior committed Version's record log, any run checkpoint/export artifact, or the on-disk
    # source file — while the co-resident document's records still survive in that prior Version.
    # (A Retraction, by contrast, RETAINS the deleted records too — asserted elsewhere.)
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    # Two co-resident documents committed in the SAME Version 0. a.txt is purged; keep.txt must
    # survive the purge intact in the prior Version's audit log. Bob (a gazetteer PERSON) shares no
    # canonical surface with the purged Alice/Acme/IBM, so a correct reach-filtered expunge leaves
    # its records untouched.
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    v0_before = cs.read_records(0)
    assert len(v0_before) >= 1  # v0 had derived records before the purge
    keep_id = _doc_id(docs / "keep.txt")
    # The co-resident document's records present in Version 0 BEFORE the purge (by reach).
    keep_before = {
        r.record_id for r in v0_before if keep_id in _record_reach_ids(r)
    }
    assert keep_before, "Bob (keep.txt) must have derived records in v0 before the purge"

    a_id = _doc_id(docs / "a.txt")
    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[a_id])

    # 1. The purged source file is physically deleted; the co-resident one is untouched.
    assert not (docs / "a.txt").exists()
    assert (docs / "keep.txt").exists()
    # 2. The prior Version's record log no longer holds the PURGED document's records...
    v0_after = cs.read_records(0)
    assert all(a_id not in _record_reach_ids(r) for r in v0_after)
    # 2b. ...but the co-resident NON-purged document's records SURVIVE verbatim in that prior
    #     Version — the collateral-erasure guard (expunge is reach-scoped, not a blanket wipe).
    keep_after = {r.record_id for r in v0_after if keep_id in _record_reach_ids(r)}
    assert keep_after == keep_before, "co-resident non-purged records were collaterally erased"
    # 3. No file anywhere under the store tree still holds the purged surfaces (raw text or derived);
    #    the co-resident Globex surface, by contrast, is allowed to remain.
    store_root = tmp_path / "store"
    leaks = [
        p
        for p in store_root.rglob("*")
        if p.is_file()
        and any(
            term in p.read_text(encoding="utf-8", errors="ignore")
            for term in ("Alice", "Acme", "IBM")
        )
    ]
    assert leaks == [], f"purged text survived in: {[str(p) for p in leaks]}"


def test_purge_leaves_no_file_name_or_content_hash_in_the_version_history(tmp_path: Path) -> None:
    # A file name can itself be personal data ("smith-dismissal.txt"), and the Corpus Version
    # manifests carried it: every Version's delta lists its documents with file_name and
    # content_hash, and a purge rewrote only the fingerprint and the record count. The CLI promised
    # that nothing is retained while the name, the id and the hash of the purged document stayed in
    # wal/v*/version.json. After a purge:
    #   * the purged document's file name is in no file under the store;
    #   * no Version lists it as a live document (ADDED / UPDATED / UNCHANGED), so its content hash
    #     is gone with those entries;
    #   * exactly the suppression tombstone remains: DELETED entries carrying the content-addressed
    #     id and nothing else, which is what keeps a Source that still serves the file from
    #     silently re-ingesting it;
    #   * the co-resident document's entries are untouched.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "smith-dismissal.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    purged_id = _doc_id(docs / "smith-dismissal.txt")
    keep_id = _doc_id(docs / "keep.txt")
    v0_entries_before = {e.document_id: e for e in cs.read_version(0).delta.entries}
    assert v0_entries_before[purged_id].file_name == "smith-dismissal.txt"  # the leak's source

    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[purged_id])

    store_root = tmp_path / "store"
    named = [
        str(p.relative_to(store_root))
        for p in store_root.rglob("*")
        if p.is_file() and "smith-dismissal" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert named == [], f"the purged document's file name survived in: {named}"

    head = cs.head_version()
    assert head is not None
    tombstones = 0
    for number in range(head + 1):
        for entry in cs.read_version(number).delta.entries:
            if entry.document_id != purged_id:
                continue
            assert entry.kind is DeltaKind.DELETED, (
                f"Version {number} still lists the purged document as {entry.kind.value}"
            )
            assert entry.file_name == "" and entry.content_hash is None
            tombstones += 1
    assert tombstones >= 1, "the suppression tombstone is what keeps the purge durable"

    # The co-resident document is still listed exactly as before, name and hash included.
    assert {e.document_id: e for e in cs.read_version(0).delta.entries}[keep_id] == (
        v0_entries_before[keep_id]
    )
    # ...and the purge is still durable: a later delta does not bring the document back.
    (docs / "smith-dismissal.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="after")
    assert all(purged_id not in _record_reach_ids(r) for r in cs.head_records())


def test_purge_scrubs_the_manifest_an_abandoned_transaction_staged(tmp_path: Path) -> None:
    # A crash mid-commit leaves a staged version.json beside the staged records.jsonl. It is never
    # read back, but it is a file on disk that names the document, so a purge rewrites it too.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "smith-dismissal.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    purged_id = _doc_id(docs / "smith-dismissal.txt")

    wal = cs._wal  # noqa: SLF001 — reproduces the crash residue the WAL recover() reports
    abandoned_version = 99
    wal.begin(abandoned_version)
    wal.stage_text(abandoned_version, "version.json", cs.read_version(0).model_dump_json())
    staged = tmp_path / "store" / "_latence" / "corpus" / "wal" / "v99" / "version.json"
    assert "smith-dismissal" in staged.read_text(encoding="utf-8")  # residue present pre-purge

    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[purged_id])

    text = staged.read_text(encoding="utf-8")
    assert "smith-dismissal" not in text and purged_id not in text
    assert "keep.txt" in text  # the co-resident document's listing is untouched


def test_purge_neutralises_abandoned_wal_residue_end_to_end(tmp_path: Path) -> None:
    # H-D1 Gap 1 (security S2), end-to-end through the DeltaRunner: an abandoned WAL transaction
    # (a crash mid-commit) leaves a fully-staged wal/v<N>/records.jsonl deriving from doc D, with
    # NO manifest — invisible to read_records but recoverable in cleartext on disk. A GDPR Purge of
    # D must neutralise it. The DoD walk: after purge, walk EVERY records.jsonl under the corpus
    # root (committed AND abandoned) and assert none reaches D, and D's secret surface strings
    # appear in no file on disk. This FAILS against main (the abandoned records.jsonl survives).
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    # (i) Commit a head Version deriving from doc D (a.txt).
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    a_id = _doc_id(docs / "a.txt")
    head_records = cs.read_records(cs.head_version() or 0)
    assert any(a_id in _record_reach_ids(r) for r in head_records)

    # (ii) Stage an ABANDONED transaction whose records reach doc D (crash mid-commit): open a WAL
    # txn, stage a records.jsonl derived from D, never write the manifest. Use a high version
    # number so it is a distinct crashed attempt the upcoming purge (which commits head+1) never
    # reaches — the residue must survive the purge's own commit and still be swept.
    wal = cs._wal  # noqa: SLF001 — reproduces the crash residue the WAL recover() reports
    abandoned_version = 99
    wal.begin(abandoned_version)
    wal.stage_text(abandoned_version, "records.jsonl", cs._encode_records(head_records))  # noqa: SLF001
    assert wal.abandoned_versions() == (abandoned_version,)
    corpus_root = tmp_path / "store"
    abandoned = corpus_root / "_latence" / "corpus" / "wal" / f"v{abandoned_version}" / "records.jsonl"
    assert "Acme" in abandoned.read_text(encoding="utf-8")  # residue present pre-purge

    # (iii) Purge doc D.
    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[a_id])

    # DoD 1 — walk EVERY records.jsonl under the corpus root: none may reach doc D.
    for path in corpus_root.rglob("records.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert a_id not in line, f"purged doc reached in {path}"
    # DoD 2 — D's secret surface strings appear in NO file anywhere on disk.
    leaks = [
        p
        for p in corpus_root.rglob("*")
        if p.is_file()
        and any(
            term in p.read_text(encoding="utf-8", errors="ignore")
            for term in ("Alice", "Acme", "IBM")
        )
    ]
    assert leaks == [], f"purged text survived in: {[str(p) for p in leaks]}"
    # No-collateral still holds: the co-resident Bob (keep.txt) is untouched by the abandoned sweep.
    assert (docs / "keep.txt").exists()


def test_purge_then_reapply_identical_delta_is_a_noop_end_to_end(tmp_path: Path) -> None:
    # H-D1 Gap 2 (correctness S3), end-to-end: commit, purge D, then re-apply the IDENTICAL delta
    # over the post-purge live set. With the head manifest re-fingerprinted, re-running yields the
    # existing head Version (a no-op), not a spurious duplicate. This FAILS against main (a
    # duplicate Version is minted because the head fingerprint is stale). Also asserts every
    # rewritten Version's stored record_count / fingerprint match its true post-purge records.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    a_id = _doc_id(docs / "a.txt")
    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[a_id])
    head_after_purge = cs.head_version()
    assert head_after_purge is not None

    # Every rewritten Version's stored metadata matches its true post-purge records.
    for version in range(head_after_purge + 1):
        stored = cs.read_version(version)
        records = cs.read_records(version)
        assert stored.record_count == len(records), f"stale record_count on v{version}"
        assert stored.fingerprint == CorpusStore._fingerprint(records), (  # noqa: SLF001
            f"stale fingerprint on v{version}"
        )

    # Re-apply the identical delta over the (unchanged) post-purge source: a documented no-op.
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="rerun")
    assert cs.head_version() == head_after_purge, "a spurious duplicate Version was minted"


def test_delta_churn_lands_in_quality_report(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    # The persisted Quality Report carries the DeltaChurn (S11 AC).
    report_uri = f"{store}/_latence/runs/v0/quality-report.json"
    from latence_core.quality import QualityReport

    persisted = QualityReport.model_validate_json(Storage.for_uri(store).read_text(report_uri))
    assert persisted.delta is not None
    assert persisted.delta.to_version == 0
    assert persisted.delta.documents_added == 1
    # The human-readable rendering includes the delta churn section.
    assert "Delta churn" in persisted.render()


def test_delta_run_is_deterministic(tmp_path: Path) -> None:
    # The G1 determinism criterion: identical input + config over the same corpus yields an
    # identical Corpus Version fingerprint. Re-running the same delta must reproduce the exact
    # committed fingerprint (and, being idempotent, commit no new Version).
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    cs = CorpusStore(Storage.for_uri(store), store)
    fp_first = cs.read_version(0).fingerprint
    # A second run over the identical corpus reproduces the same fingerprint (no drift).
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0b")
    assert cs.head_version() == 0  # idempotent, no new Version
    assert cs.read_version(0).fingerprint == fp_first

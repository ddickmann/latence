"""Delta chaos + property hardening for the S11 incremental-delta engine (#76, Hardening bar).

The end-to-end tests in ``test_e2e_delta.py`` pin specific hand-built delta scenarios; this suite
drives the *same* real ``DeltaRunner`` over **randomized, seeded** interleaved add / retract / purge
sequences and asserts the S11 guarantees hold after every step of every sequence — the chaos half
of the #76 hardening bar (ADR-0018):

* **idempotency** — re-applying the current corpus state commits NO new Corpus Version (the
  fingerprint matches head), whatever sequence of operations preceded it;
* **durable retraction** — a document explicitly retracted stays gone across every subsequent
  ``apply_delta`` even though its file is still on disk (only intentional tombstones are durable);
* **purge no-collateral** — purging a document erases everything derived from it (head + prior
  Versions + on-disk source + run artifacts) while leaving every co-resident document's records
  intact — the reach-scoped expunge, never a blanket wipe (issue #42);
* **purge idempotency** — purging an already-purged / absent document is a safe no-op;
* **reconciliation-under-drift** — the churn's ``reconciliation_triggered`` equals ``drift >=
  threshold`` exactly, and a full Reconciliation's affected set is the whole corpus;
* **WAL-recovery fuzz** — over randomized commit/abandon sequences, an abandoned (crashed
  mid-commit) transaction never advances the head and always leaves the last committed Version
  intact and inspectable.

Deterministic and dependency-free: randomness comes from a seeded :class:`random.Random`, so the
whole suite is reproducible (no ``hypothesis`` dep — ADR-0016 keeps core lean).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from latence_core.contracts import DisambiguationRecord, Record
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.state.corpus_store import CorpusStore
from latence_core.state.write_ahead_log import WriteAheadLog
from latence_core.storage import Storage

# A small vocabulary of gazetteer-known surfaces, so extraction is deterministic and non-empty.
_SENTENCES = [
    "Alice works at Acme.",
    "Bob is here.",
    "IBM is a company.",
    "International Business Machines partnered with Acme.",
    "Acme is a company.",
    "Alice visited IBM.",
    "Bob works at Acme.",
    "Globex is here.",
]


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: delta-chaos
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
        ORG: {{terms: [Acme, International Business Machines, IBM, Globex]}}
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
    from latence_core.state.graph_patcher import _record_reach

    return set(_record_reach(record))  # type: ignore[arg-type]


def _entity_surfaces(records: list[Record]) -> set[str]:
    return {
        r.entity.canonical_text
        for r in records
        if isinstance(r, DisambiguationRecord) and r.entity is not None
    }


# ---------------------------------------------------------------------------
# Interleaved add / retract / purge chaos
# ---------------------------------------------------------------------------


def test_interleaved_add_retract_purge_preserves_invariants(tmp_path: Path) -> None:
    # A seeded random walk of add / retract / purge operations against a real DeltaRunner. After
    # EVERY step the S11 invariants must hold: idempotency (a re-apply commits nothing), durable
    # retraction (a retracted doc never comes back), and purge no-collateral (nothing derived from
    # a purged doc lingers, co-resident docs untouched). Runs several independent seeded walks.
    for seed in range(12):
        rng = random.Random(seed)
        run_dir = tmp_path / f"seed-{seed}"
        docs = run_dir / "docs"
        docs.mkdir(parents=True)
        store = f"file://{run_dir}/store"
        cs = CorpusStore(Storage.for_uri(store), store)

        # doc_key -> (on-disk path, content-id). A "retracted"/"purged" doc leaves the live set.
        present: dict[str, Path] = {}
        retracted_ids: set[str] = set()
        purged_ids: set[str] = set()
        counter = 0

        def _write_doc(kdocs: Path, kname: str, kcounter: int) -> Path:
            # Unique content per doc (a distinct filler token) so content-ids never collide.
            sentence = _SENTENCES[kcounter % len(_SENTENCES)]
            path = kdocs / f"{kname}.txt"
            path.write_text(f"{sentence} token{kcounter}\n", encoding="utf-8")
            return path

        # Seed with at least one document so the first op has something to work with.
        first = _write_doc(docs, "d0", counter)
        present["d0"] = first
        counter += 1
        DeltaRunner().apply_delta(_pipeline(store, docs), run_id=f"s{seed}-init")

        for step in range(1, rng.randint(6, 12) + 1):
            op = rng.choice(["add", "add", "retract", "purge", "noop"])
            run_id = f"s{seed}-{step}"

            if op == "add":
                name = f"d{counter}"
                path = _write_doc(docs, name, counter)
                present[name] = path
                counter += 1
                DeltaRunner().apply_delta(_pipeline(store, docs), run_id=run_id)

            elif op == "retract" and present:
                name = rng.choice(sorted(present))
                path = present.pop(name)
                doc_id = _doc_id(path)
                retracted_ids.add(doc_id)
                DeltaRunner().retract(
                    _pipeline(store, docs), run_id=run_id, document_ids=[doc_id]
                )
                # Durable retraction: the retracted content-id is gone from the live head set.
                assert all(
                    doc_id not in _record_reach_ids(r)
                    for r in cs.read_records(cs.head_version() or 0)
                ), (seed, step, "retracted doc lingers in live set")

            elif op == "purge" and present:
                name = rng.choice(sorted(present))
                path = present.pop(name)
                doc_id = _doc_id(path)
                # Snapshot every co-resident doc's live records BEFORE the purge.
                live_before = cs.read_records(cs.head_version() or 0)
                coresident_before = {
                    r.record_id
                    for r in live_before
                    if doc_id not in _record_reach_ids(r)
                }
                DeltaRunner().purge(
                    _pipeline(store, docs), run_id=run_id, document_ids=[doc_id]
                )
                purged_ids.add(doc_id)
                # No-collateral: the purged doc's file is gone; co-resident live records survive.
                assert not path.exists(), (seed, step, "purged source file survived")
                live_after = cs.read_records(cs.head_version() or 0)
                assert all(
                    doc_id not in _record_reach_ids(r) for r in live_after
                ), (seed, step, "purged doc lingers in live set")
                coresident_after = {
                    r.record_id
                    for r in live_after
                    if doc_id not in _record_reach_ids(r)
                }
                # Every co-resident record still present is a subset check (a co-resident doc may
                # also have been re-resolved, but none may be dropped BECAUSE of the purge unless
                # it too reached the purged doc). Records that reached ONLY co-resident docs stay.
                assert coresident_before <= coresident_after | {
                    r.record_id for r in live_after
                }, (seed, step, "purge collaterally dropped a co-resident record")

            else:  # noop -> assert idempotency: re-applying commits no new Version.
                head_before = cs.head_version()
                DeltaRunner().apply_delta(_pipeline(store, docs), run_id=f"{run_id}-idem")
                assert cs.head_version() == head_before, (
                    seed,
                    step,
                    "a no-op re-apply minted a new Corpus Version",
                )

            # Global invariant after every step: no retracted/purged content-id is ever in the
            # live set — durable tombstones hold across the whole walk.
            live = cs.read_records(cs.head_version() or 0)
            gone = retracted_ids | purged_ids
            assert all(
                not (_record_reach_ids(r) & gone) for r in live
            ), (seed, step, "a retracted/purged doc resurrected")


def test_retracted_document_stays_gone_across_many_reruns(tmp_path: Path) -> None:
    # Focused durable-retraction property: retract a doc, then hammer apply_delta N times over the
    # still-present file. It must never resurrect (only intentional tombstones are durable).
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    (docs / "gone.txt").write_text("Acme is a company.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    gone_id = _doc_id(docs / "gone.txt")
    DeltaRunner().retract(_pipeline(store, docs), run_id="ret", document_ids=[gone_id])
    assert "Acme" not in _entity_surfaces(cs.read_records(cs.head_version() or 0))

    for i in range(6):
        DeltaRunner().apply_delta(_pipeline(store, docs), run_id=f"rerun-{i}")
        surfaces = _entity_surfaces(cs.read_records(cs.head_version() or 0))
        assert "Acme" not in surfaces, f"retracted Acme resurrected on rerun {i}"
        assert "Bob" in surfaces, "co-resident Bob wrongly dropped"


# ---------------------------------------------------------------------------
# Purge idempotency + no-collateral (reach-scoped)
# ---------------------------------------------------------------------------


def test_purge_is_idempotent_and_absent_purge_is_a_noop(tmp_path: Path) -> None:
    # Purging the same document twice, and purging an absent/unknown id, must both be safe no-ops:
    # the second purge changes nothing (the doc already left the corpus) and never raises.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    a_id = _doc_id(docs / "a.txt")
    DeltaRunner().purge(_pipeline(store, docs), run_id="pur1", document_ids=[a_id])
    head_after_first = cs.head_version()
    live_after_first = {r.record_id for r in cs.read_records(head_after_first or 0)}

    # Purge the SAME (now-absent) id again: a no-op, no new live Version content, never raises.
    DeltaRunner().purge(_pipeline(store, docs), run_id="pur2", document_ids=[a_id])
    live_after_second = {r.record_id for r in cs.read_records(cs.head_version() or 0)}
    assert live_after_second == live_after_first, "re-purging changed the live set"

    # Purge a completely unknown id: also a no-op that leaves Bob intact.
    DeltaRunner().purge(
        _pipeline(store, docs), run_id="pur3", document_ids=["sha256:deadbeef"]
    )
    assert "Bob" in _entity_surfaces(cs.read_records(cs.head_version() or 0))


def test_purge_no_collateral_across_prior_versions_and_source(tmp_path: Path) -> None:
    # Reach-scoped erasure over a multi-version history: after purging one document, NO prior
    # committed Version's record log, run artifact, or on-disk file retains the purged surfaces —
    # while the co-resident document's records survive verbatim in every prior Version.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    # Build extra history so the expunge must scan MORE than the head Version.
    (docs / "c.txt").write_text("Globex is here.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")

    keep_id = _doc_id(docs / "keep.txt")
    keep_before = {
        r.record_id for r in cs.read_records(0) if keep_id in _record_reach_ids(r)
    }
    assert keep_before, "Bob must have derived records in v0 before the purge"

    a_id = _doc_id(docs / "a.txt")
    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[a_id])

    # The purged source file is gone; the co-resident file remains.
    assert not (docs / "a.txt").exists()
    assert (docs / "keep.txt").exists()

    # Every committed Version's record log is free of the purged doc's reach...
    wal = WriteAheadLog(Storage.for_uri(store), f"{store.rstrip('/')}/_latence/corpus/wal")
    for version in wal.committed_versions():
        recs = cs.read_records(version)
        assert all(a_id not in _record_reach_ids(r) for r in recs), version

    # ...but Bob's records survive verbatim in the prior Version 0 audit log.
    keep_after = {
        r.record_id for r in cs.read_records(0) if keep_id in _record_reach_ids(r)
    }
    assert keep_after == keep_before, "co-resident non-purged records were collaterally erased"

    # No file anywhere under the store still holds the purged surfaces.
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


# ---------------------------------------------------------------------------
# Reconciliation-under-drift property
# ---------------------------------------------------------------------------


def test_reconciliation_triggered_matches_drift_threshold(tmp_path: Path) -> None:
    # The churn's reconciliation flag must equal drift >= threshold EXACTLY, across a range of
    # configured thresholds and corpus sizes. A full reconciliation's affected set is the whole
    # corpus; an incremental delta's is a strict subset.
    for seed, threshold in enumerate([0.0, 0.25, 0.5, 0.9, 1.5]):
        run_dir = tmp_path / f"thr-{seed}"
        docs = run_dir / "docs"
        docs.mkdir(parents=True)
        store = f"file://{run_dir}/store"
        cs = CorpusStore(Storage.for_uri(store), store)
        # A corpus of several unrelated documents.
        for i in range(5):
            (docs / f"d{i}.txt").write_text(
                f"{_SENTENCES[i % len(_SENTENCES)]} tok{i}\n", encoding="utf-8"
            )
        DeltaRunner(drift_threshold=threshold).apply_delta(
            _pipeline(store, docs), run_id=f"thr{seed}-v0"
        )
        # Add one document -> a small drift.
        (docs / "new.txt").write_text("Globex partnered with Acme. tokNEW\n", encoding="utf-8")
        r = DeltaRunner(drift_threshold=threshold).apply_delta(
            _pipeline(store, docs), run_id=f"thr{seed}-v1"
        )
        assert r.delta is not None
        expected = r.delta.drift >= threshold
        assert r.delta.reconciliation_triggered is expected, (threshold, r.delta.drift)
        if r.delta.reconciliation_triggered:
            # A full reconciliation recomputes the whole corpus-level set.
            assert r.delta.affected_set_size == r.delta.corpus_record_count
        elif r.delta.corpus_record_count > 0:
            # An incremental delta touches at most the whole corpus (usually strictly fewer).
            assert r.delta.affected_set_size <= r.delta.corpus_record_count
        # Regardless: the committed head fingerprint is stable on an immediate re-apply.
        head = cs.head_version()
        DeltaRunner(drift_threshold=threshold).apply_delta(
            _pipeline(store, docs), run_id=f"thr{seed}-idem"
        )
        assert cs.head_version() == head, "reconciliation/idempotency disagreement"


# ---------------------------------------------------------------------------
# WAL-recovery fuzz
# ---------------------------------------------------------------------------


def test_wal_recovery_fuzz_abandoned_never_advances_head(tmp_path: Path) -> None:
    # A seeded fuzz over commit/abandon sequences: the WAL's recovered head is ALWAYS the highest
    # fully-committed version, an abandoned (staged-but-uncommitted) transaction is NEVER promoted,
    # and the last committed version stays inspectable — the S11 "a failed run leaves Version N
    # intact" guarantee under randomized interleavings.
    for seed in range(40):
        rng = random.Random(seed)
        root = f"file://{tmp_path}/wal-{seed}"
        wal = WriteAheadLog(Storage.for_uri(root), root)
        highest_committed: int | None = None
        abandoned_versions: set[int] = set()
        for version in range(rng.randint(3, 10)):
            wal.begin(version)
            wal.stage_text(version, "records.jsonl", f"v{version}-data\n")
            # Randomly either commit this version or abandon it (crash before the manifest write).
            if rng.random() < 0.6:
                uri = wal.staged_uri(version, "records.jsonl")
                wal.commit(version, files={"records": uri}, metadata={"v": str(version)})
                highest_committed = version
            else:
                abandoned_versions.add(version)  # left staged, no manifest

        rec = wal.recover()
        # The recovered head is exactly the highest committed version (or None if none committed).
        assert rec.committed == highest_committed, (seed, rec.committed, highest_committed)
        # Every abandoned transaction is reported abandoned and NOT committed.
        assert set(rec.abandoned) == abandoned_versions, (seed, rec.abandoned)
        for v in abandoned_versions:
            assert not wal.is_committed(v), (seed, v)
        # The committed head, if any, is still inspectable (its manifest reads back).
        if highest_committed is not None:
            manifest = wal.read_manifest(highest_committed)
            assert manifest["committed"] is True
            assert manifest["version"] == highest_committed


def test_wal_abandoned_transaction_leaves_prior_committed_records_intact(tmp_path: Path) -> None:
    # The concrete data-integrity half of WAL recovery: after committing v0 then crashing mid-v1
    # (stage without commit), v0's committed payload is byte-intact and readable — the abandoned
    # v1 staging never corrupts the live head. Fuzzed over several payloads.
    for seed in range(20):
        rng = random.Random(seed + 100)
        root = f"file://{tmp_path}/wal-intact-{seed}"
        storage = Storage.for_uri(root)
        wal = WriteAheadLog(storage, root)
        payload = f"committed-v0-{rng.randint(0, 9999)}\n"
        wal.begin(0)
        uri0 = wal.stage_text(0, "records.jsonl", payload)
        wal.commit(0, files={"records": uri0}, metadata={})
        # Crash mid-v1: begin + stage, no commit.
        wal.begin(1)
        wal.stage_text(1, "records.jsonl", "torn-partial-v1\n")
        rec = wal.recover()
        assert rec.committed == 0 and rec.abandoned == (1,), seed
        # v0's committed payload is byte-intact.
        assert storage.read_text(wal.staged_uri(0, "records.jsonl")) == payload, seed

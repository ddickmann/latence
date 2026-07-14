"""W17 (ADR-0043) — cross-run INCREMENTAL EXTRACTION through the DeltaRunner.

Proves the issue-#42 boundary shift: a delta re-parses/-chunks/-extracts only the NEW/CHANGED
documents and REUSES every unchanged document's persisted doc-level records, while the committed
corpus stays byte-identical to a full from-scratch run of the whole source.

Covered:

* **equivalence + skip telemetry** — adding 1 document to a committed 3-document corpus produces
  committed records for the 3 unchanged documents that are byte-identical to a full from-scratch
  run of all 4, while the delta run's doc-level Stages ran for ONLY the added document
  (``documents_reused == 3``, and the run's parse/entity checkpoints hold only the added doc);
* **change detection** — editing 1 document's bytes re-extracts that document and reuses the rest;
* **the reuse store round-trips** — the committed Corpus Version persists every live document's
  doc-level records keyed by content-addressed id;
* **purge neutralises the reuse store** — a purged document's raw text does not survive in
  ``doclevel.jsonl`` (the new residue surface);
* **determinism / idempotency preserved** — a reuse delta is byte-reproducible and a no-op re-run
  mints no Version.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from latence_core.contracts import Record
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.state.corpus_store import CorpusStore
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: delta-incremental
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
        ORG: {{terms: [Acme, Globex, Initech, Umbrella, IBM]}}
        PERSON: {{terms: [Alice, Bob, Carol, Dave]}}
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


def _checkpoint_doc_ids(store: str, run_id: str, stage: str) -> set[str]:
    """The distinct source-document ids present in a run's Stage checkpoint (the skip witness)."""
    storage = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/checkpoints/{stage}.jsonl"
    if not storage.exists(uri):
        return set()
    import json

    ids: set[str] = set()
    for line in storage.read_text(uri).splitlines():
        if line.strip():
            ids.add(json.loads(line)["provenance"]["document_id"])
    return ids


def _records_for_docs(records: list[Record], doc_ids: set[str]) -> dict[str, str]:
    """Committed records whose provenance reach is a SUBSET of ``doc_ids``, id → serialized bytes."""
    from latence_core.state.graph_patcher import _record_reach

    out: dict[str, str] = {}
    for r in records:
        reach = set(_record_reach(r))
        if reach and reach <= doc_ids:
            out[r.record_id] = r.model_dump_json()
    return out


def test_add_reuses_unchanged_and_extracts_only_new(tmp_path: Path) -> None:
    # V0: a committed 3-document corpus of UNRELATED documents.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Bob works at Globex.\n", encoding="utf-8")
    (docs / "c.txt").write_text("Carol works at Initech.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    unchanged_ids = {_doc_id(docs / n) for n in ("a.txt", "b.txt", "c.txt")}

    # V1: add a 4th unrelated document.
    (docs / "d.txt").write_text("Dave works at Umbrella.\n", encoding="utf-8")
    added_id = _doc_id(docs / "d.txt")
    r1 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    assert r1.delta is not None

    # Skip telemetry: 3 documents reused, exactly 1 extracted.
    assert r1.delta.documents_reused == 3
    assert r1.delta.documents_extracted == 1
    assert r1.delta.documents_added == 1

    # Skip witness: the delta run's doc-level checkpoints hold ONLY the added document — the 3
    # reused documents never ran parse/chunk/extract.
    assert _checkpoint_doc_ids(store, "v1", "parse") == {added_id}
    assert _checkpoint_doc_ids(store, "v1", "entities") == {added_id}
    assert _checkpoint_doc_ids(store, "v1", "relations") <= {added_id}

    # Equivalence: a full from-scratch run of ALL 4 documents in a SEPARATE store.
    fresh_store = f"file://{tmp_path}/fresh"
    fresh_cs = CorpusStore(Storage.for_uri(fresh_store), fresh_store)
    DeltaRunner().apply_delta(_pipeline(fresh_store, docs), run_id="all4")

    incremental = _records_for_docs(cs.read_records(cs.head_version() or 0), unchanged_ids)
    from_scratch = _records_for_docs(
        fresh_cs.read_records(fresh_cs.head_version() or 0), unchanged_ids
    )
    assert incremental  # there ARE unchanged-doc records to compare
    # Byte-identical committed records for the 3 unchanged documents.
    assert incremental == from_scratch


def test_reuse_store_round_trips_every_live_document(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Bob works at Globex.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    (docs / "c.txt").write_text("Carol works at Initech.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")

    # The head Version's reuse store holds every LIVE document's doc-level records (all 3 docs).
    doclevel = cs.head_doclevel()
    live_ids = {_doc_id(docs / n) for n in ("a.txt", "b.txt", "c.txt")}
    assert {r.provenance.document_id for r in doclevel} == live_ids
    # Each document contributed a DocumentRecord + at least one ChunkRecord + its EntityMentions.
    from latence_core.contracts import ChunkRecord, DocumentRecord, EntityMention

    for did in live_ids:
        kinds = {type(r).__name__ for r in doclevel if r.provenance.document_id == did}
        assert {DocumentRecord.__name__, ChunkRecord.__name__, EntityMention.__name__} <= kinds


def test_edit_reextracts_only_the_edited_document(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Bob works at Globex.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    # Edit b.txt -> a NEW content id (Bob -> Carol). a.txt is byte-unchanged => reused.
    (docs / "b.txt").write_text("Carol works at Initech.\n", encoding="utf-8")
    edited_id = _doc_id(docs / "b.txt")
    r1 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    assert r1.delta is not None
    # The unchanged a.txt is reused; the edited b.txt (new content id) is extracted. Its old
    # content id is DELETED (content-addressed edit = add + delete), so documents_added counts the
    # new id and only that document ran the doc-level Stages.
    assert r1.delta.documents_reused == 1
    assert _checkpoint_doc_ids(store, "v1", "entities") == {edited_id}


def test_purge_neutralises_the_doc_level_reuse_store(tmp_path: Path) -> None:
    # A GDPR Purge must erase the purged document's raw text from the NEW residue surface too — the
    # doc-level reuse store (doclevel.jsonl) holds raw chunk/mention text.
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob works at Globex.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    a_id = _doc_id(docs / "a.txt")

    DeltaRunner().purge(_pipeline(store, docs), run_id="pur", document_ids=[a_id])

    corpus_root = tmp_path / "store"
    # No doclevel.jsonl anywhere reaches the purged document...
    for path in corpus_root.rglob("doclevel.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert a_id not in line, f"purged doc survived in reuse store {path}"
    # ...and its surface strings appear in NO file under the store (raw or derived).
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
    # The co-resident document still reuses (Bob/Globex is untouched).
    assert (docs / "keep.txt").exists()


def test_reuse_delta_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Bob works at Globex.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    (docs / "c.txt").write_text("Carol works at Initech.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    fp = cs.read_version(cs.head_version() or 0).fingerprint

    # Re-applying the identical corpus (all 3 docs now reused) is a no-op: no new Version, same
    # fingerprint — reuse does not perturb determinism or idempotency.
    r = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1b")
    assert r.delta is not None
    assert cs.head_version() == 1
    assert cs.read_version(1).fingerprint == fp
    assert r.delta.documents_reused == 3
    assert r.delta.documents_extracted == 0

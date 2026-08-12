"""ADR-0061 — incremental ingest over a CONTEXT-ENRICHMENT pipeline == full rebuild.

The ticket-03 CI-scale equivalence gate. Before ADR-0061 a pipeline containing a
``context_enrichment`` Stage fell back to full re-extraction on every delta
(``DeltaRunner._reuse_supported`` refused reuse), so "incremental" was incremental in name only on
the flagship profile. Now the reuse overlay reaches the enrichment Stage's per-carrier chunk
stream — merged document-interleaved in source order — and every chunk's header is re-derived
from the CURRENT assembled graph each run.

Covered:

* **add → update → remove churn == full rebuild, byte-for-byte** — after a base build, an add
  delta, an edit delta and a file-removal delta, the FINAL delta run's exported corpus
  (``records.jsonl``) and its assembled-graph checkpoint are byte-identical to a from-scratch
  batch run of the same final corpus: same rows, same order (the order-preserving merge), same
  headers, same graph;
* **reuse actually engages** — the deltas reuse the unchanged documents (``documents_reused``
  witness + doc-level checkpoints hold only the changed documents), so the equivalence is not the
  trivial full-re-extraction one;
* **no export double-count** — the enrichment Stage re-emits the reused chunks, so the Export
  must NOT materialize the overlay again (every exported ``record_id`` appears exactly once);
* **churn never silently re-induces** — on an induced-schema (per-chunk, offline lexicon
  inducer) enrichment pipeline, a delta's induction checkpoint holds ONLY the changed document's
  chunks: unchanged documents are excluded at Source and their induced chunks ride the reuse
  store (the ticket-03 non-negotiable, previously violated by the full-extraction fallback).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path, *, induce: bool = False) -> Pipeline:
    induce_stage = """
  - name: induce
    capability: schema_induction
    provider: label_inducer.lexicon
    depends_on: [chunk]
    config:
      candidates:
        entity_types:
          person: [Alice, Bob, Carol, Dave]
          organization: [Acme, Globex, Initech, Umbrella]
        relation_types:
          works_for: [works at, joined]
"""
    chunk_parent = "induce" if induce else "chunk"
    return Pipeline.from_yaml_text(
        f"""
name: delta-enrichment-incremental
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
{induce_stage if induce else ""}
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [{chunk_parent}]
    config:
      labels:
        ORG: {{terms: [Acme, Globex, Initech, Umbrella]}}
        PERSON: {{terms: [Alice, Bob, Carol, Dave]}}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [{chunk_parent}, entities]
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
  - name: enrich
    capability: context_enrichment
    provider: context.kg_header
    depends_on: [{chunk_parent}, graph]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [enrich]
"""
    )


def _doc_id(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_doc_ids(store: str, run_id: str, stage: str) -> set[str]:
    storage = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/checkpoints/{stage}.jsonl"
    if not storage.exists(uri):
        return set()
    ids: set[str] = set()
    for line in storage.read_text(uri).splitlines():
        if line.strip():
            ids.add(json.loads(line)["provenance"]["document_id"])
    return ids


def _read(store: str, run_id: str, rel: str) -> str:
    return Storage.for_uri(store).read_text(f"{store}/_latence/runs/{run_id}/{rel}")


def _seed_docs(docs: Path) -> None:
    docs.mkdir()
    (docs / "a.txt").write_text("Alice works at Acme. Acme is in town.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Bob works at Globex. Alice joined Globex.\n", encoding="utf-8")
    (docs / "c.txt").write_text("Carol works at Initech.\n", encoding="utf-8")


def test_add_update_remove_churn_matches_full_rebuild_byte_for_byte(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _seed_docs(docs)
    store = f"file://{tmp_path}/store"

    # Base build (v0), then three churn operations through the REAL delta path.
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    # ADD: a new document arrives.
    (docs / "d.txt").write_text("Dave works at Umbrella. Dave joined Acme.\n", encoding="utf-8")
    added_id = _doc_id(docs / "d.txt")
    r_add = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    assert r_add.delta is not None
    assert r_add.delta.documents_reused == 3  # reuse ENGAGES on an enrichment pipeline now
    assert r_add.delta.documents_added == 1
    assert _checkpoint_doc_ids(store, "v1", "parse") == {added_id}
    assert _checkpoint_doc_ids(store, "v1", "entities") == {added_id}

    # UPDATE: an existing document's bytes change.
    (docs / "b.txt").write_text("Bob works at Initech now. Bob joined Initech.\n", encoding="utf-8")
    edited_id = _doc_id(docs / "b.txt")
    r_upd = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v2")
    assert r_upd.delta is not None
    assert r_upd.delta.documents_reused == 3  # a, c, d unchanged
    assert _checkpoint_doc_ids(store, "v2", "entities") == {edited_id}

    # REMOVE: a document disappears from the source (the incremental removal path).
    (docs / "c.txt").unlink()
    r_del = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v3")
    assert r_del.delta is not None
    assert r_del.delta.documents_deleted == 1
    assert r_del.delta.documents_reused == 3  # a, b', d — survivors reused, nothing re-extracted
    assert r_del.delta.documents_extracted == 0

    # FULL REBUILD of the same final corpus (a, b', d) — a plain batch run, fresh store.
    fresh_store = f"file://{tmp_path}/fresh"
    LocalRunner().run(_pipeline(fresh_store, docs), run_id="rebuild")

    # The gate: byte-identical exported corpus (rows, order, headers) and byte-identical
    # assembled graph — the incremental store's exported retrieval state IS the rebuild's.
    incremental_corpus = _read(store, "v3", "export/records.jsonl")
    rebuild_corpus = _read(fresh_store, "rebuild", "export/records.jsonl")
    assert incremental_corpus == rebuild_corpus
    incremental_graph = _read(store, "v3", "checkpoints/graph.jsonl")
    rebuild_graph = _read(fresh_store, "rebuild", "checkpoints/graph.jsonl")
    assert incremental_graph == rebuild_graph

    # The equivalence is not vacuous: the final corpus has all 3 documents' chunks, and at least
    # one reused document's chunk carries a KG-derived context header recomputed this run.
    rows = [json.loads(line) for line in incremental_corpus.splitlines() if line.strip()]
    assert {r["provenance"]["document_id"] for r in rows} == {
        _doc_id(docs / "a.txt"),
        edited_id,
        added_id,
    }
    reused_doc_rows = [r for r in rows if r["provenance"]["document_id"] == added_id]
    assert any(r.get("context_header") for r in reused_doc_rows)


def test_export_materializes_each_reused_record_exactly_once(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _seed_docs(docs)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    (docs / "d.txt").write_text("Dave works at Umbrella.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")

    # The enrichment Stage re-emits the reused chunks (fresh ∪ overlay), so the Export must NOT
    # fold the overlay in again: every exported record_id appears exactly once.
    corpus = _read(store, "v1", "export/records.jsonl")
    ids = [json.loads(line)["record_id"] for line in corpus.splitlines() if line.strip()]
    assert len(ids) == len(set(ids))
    # ...and the reused documents' chunks ARE in the export (the overlay reached enrichment).
    docs_in_export = {
        json.loads(line)["provenance"]["document_id"]
        for line in corpus.splitlines()
        if line.strip()
    }
    assert _doc_id(docs / "a.txt") in docs_in_export


def test_churn_on_an_induced_schema_pipeline_never_reinduces_unchanged_documents(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    _seed_docs(docs)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs, induce=True), run_id="v0")
    # All 3 documents were induced on the base build.
    assert len(_checkpoint_doc_ids(store, "v0", "induce")) == 3

    (docs / "d.txt").write_text("Dave works at Umbrella.\n", encoding="utf-8")
    added_id = _doc_id(docs / "d.txt")
    r = DeltaRunner().apply_delta(_pipeline(store, docs, induce=True), run_id="v1")
    assert r.delta is not None
    assert r.delta.documents_reused == 3

    # The ticket-03 non-negotiable: churn must not re-trigger induction on unchanged documents.
    # The induction Stage is doc-level, so its delta checkpoint holds ONLY the added document's
    # chunks — the unchanged documents' induced chunks ride the reuse store instead.
    assert _checkpoint_doc_ids(store, "v1", "induce") == {added_id}

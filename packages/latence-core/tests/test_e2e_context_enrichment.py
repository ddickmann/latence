"""W13 context enrichment end-to-end: the Stage is genuinely OPTIONAL + contract-preserving.

Runs a full CPU spine (source→…→graph) with and without the ``context_enrichment`` Stage on a tiny
offline corpus, and proves the ADR-0039 guarantees at the run level:

* the Stage is OPTIONAL — dropping it leaves the upstream chunk checkpoint byte-identical and the
  Quality Report's ``context_enrichment`` section absent (the additive-field guarantee);
* the header hits the EMBEDDING input only — the exported corpus ``content`` is byte-identical with
  and without the Stage (the LLM window never grows);
* the Quality Report gains the counts-only ``chunks_enriched`` / ``hub_entities_capped`` roll-up.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

_TEXT = "Alice works at Acme. Acme partnered with Globex.\n"


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(_TEXT, encoding="utf-8")
    return docs


def _pipeline(store: str, docs: Path, *, with_enrich: bool) -> Pipeline:
    corpus_dep = "enrich" if with_enrich else "content_screen"
    enrich_stage = (
        """
  - name: enrich
    capability: context_enrichment
    provider: context.kg_header
    depends_on: [content_screen, graph]
    config: {max_neighbors_per_entity: 4, max_triples: 6}
"""
        if with_enrich
        else ""
    )
    return Pipeline.from_yaml_text(
        f"""
name: ctx-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', sensitivity: public, extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config: {{max_tokens: 256, overlap_tokens: 0, min_tokens: 2}}
  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [content_screen]
    config:
      labels:
        ORG: {{terms: [Acme, Globex]}}
        PERSON: {{terms: [Alice]}}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [content_screen, entities]
    config:
      window_chars: 200
      relations:
        works_for: {{head: PERSON, tail: ORG, triggers: [works at]}}
        partner_of: {{head: ORG, tail: ORG, triggers: [partnered with]}}
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config: {{corpus_id: ctx-e2e}}
{enrich_stage}  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [{corpus_dep}]
    config:
      basename: corpus
      context_columns: true
      embedder: {{provider: embedding.hashing, config: {{dimension: 16}}}}
"""
    )


def _corpus_rows(storage: Storage, base: str) -> list[dict[str, Any]]:
    text = storage.read_text(f"{base}/export/corpus.jsonl")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_optional_stage_content_byte_identical_and_report_section(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    s = Storage.for_uri(store)

    LocalRunner().run(_pipeline(store, docs, with_enrich=False), run_id="no-ctx")
    LocalRunner().run(_pipeline(store, docs, with_enrich=True), run_id="with-ctx")
    base_no = f"{store}/_latence/runs/no-ctx"
    base_with = f"{store}/_latence/runs/with-ctx"

    # 1. The chunk stream feeding the graph is byte-identical with and without the Stage — the
    #    Stage never mutates upstream (offsets/provenance/content untouched).
    assert s.read_text(f"{base_no}/checkpoints/content_screen.jsonl") == s.read_text(
        f"{base_with}/checkpoints/content_screen.jsonl"
    )

    rows_no = _corpus_rows(s, base_no)
    rows_with = _corpus_rows(s, base_with)
    assert rows_no and rows_with

    # 2. The exported chunk CONTENT is byte-identical with and without enrichment (embedding-only).
    assert [r["content"] for r in rows_no] == [r["content"] for r in rows_with]

    # 3. Without the Stage, every chunk's context_header is null; with it, at least one is populated.
    assert all(r.get("context_header") is None for r in rows_no)
    assert any(r.get("context_header") is not None for r in rows_with)
    populated = next(r for r in rows_with if r.get("context_header") is not None)
    assert "Acme" in populated["context_header"]["entities"]

    # 4. The Quality Report gains the counts-only enrichment roll-up only WITH the Stage.
    report_no = QualityReport.model_validate_json(s.read_text(f"{base_no}/quality-report.json"))
    report_with = QualityReport.model_validate_json(
        s.read_text(f"{base_with}/quality-report.json")
    )
    assert report_no.context_enrichment is None
    assert report_with.context_enrichment is not None
    assert report_with.context_enrichment.chunks_enriched >= 1


def test_context_enrichment_requires_both_chunk_and_graph_wiring(tmp_path: Path) -> None:
    """A context_enrichment Stage wired to only the graph (no chunk parent) fails loudly."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    # Rewire enrich to depend on graph ONLY — the chunk side is missing.
    pipeline = _pipeline(store, docs, with_enrich=True)
    for stage in pipeline.stages:
        if stage.name == "enrich":
            stage.depends_on = ["graph"]
    import pytest
    from latence_core.errors import ContractError

    with pytest.raises(ContractError, match="no ChunkRecords"):
        LocalRunner().run(pipeline, run_id="bad-wiring")

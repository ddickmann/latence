"""End-to-end S8 Disambiguation through the local Runner.

Source -> Parse -> Chunk -> Entity -> Relation -> Disambiguation -> Export, using the
in-core deterministic ``disambiguation.cascade``. Asserts the exported records are
DisambiguationRecords (ENTITY-scope canonical entities + RELATION-scope normalized
relations), that cross-document mentions merge to one canonical entity with Evidence and an
audited merge log, that a configured KB links with graceful fallback, that the relation is
normalized + inverse-swapped onto the canonical entities, that the Quality Report's
DisambiguationQuality carries the merge/link statistics (the S8 AC), that a below-policy
merge is logged not applied, and that the run is deterministic and resumes cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.contracts import DisambiguationRecord, DisambiguationScope
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: disambiguation-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
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
        ORG:
          terms: [Acme Corporation, Acme, International Business Machines, IBM]
        PERSON:
          terms: [Alice, Bob]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      window_chars: 200
      relations:
        works_for:
          head: PERSON
          tail: ORG
          triggers: [works at, employed by, joined]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
    config:
      kb:
        - kb_id: Q1
          name: Acme
          entity_type: ORG
      kb_threshold: 0.5
      inverses:
        employs: works_for
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [disambiguate]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    # Two documents; "IBM" and "International Business Machines" merge by acronym across docs;
    # "Acme" appears in both. Alice works at Acme (a works_for relation).
    (docs / "memo.txt").write_text(
        "Alice works at Acme. International Business Machines is a partner.\n",
        encoding="utf-8",
    )
    (docs / "note.txt").write_text(
        "Bob joined Acme last year. IBM announced results. "
        "Acme Corporation filed the report.\n",
        encoding="utf-8",
    )
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_disambiguation_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s8e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s8e2e"

    # 1. The export holds typed DisambiguationRecord rows.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [DisambiguationRecord.model_validate(r) for r in rows]
    assert records
    entities = [r.entity for r in records if r.scope is DisambiguationScope.ENTITY and r.entity]
    relations = [
        r.relation for r in records if r.scope is DisambiguationScope.RELATION and r.relation
    ]

    # 2. IBM + International Business Machines merged (acronym), across documents.
    by_text = {e.canonical_text: e for e in entities}
    ibm = next(
        e for e in entities if "ibm" in e.canonical_text.lower()
        or "international" in e.canonical_text.lower()
    )
    assert ibm.mention_count >= 2
    assert len(ibm.source_document_ids) == 2  # cross-document
    applied = [d for d in ibm.merge_decisions if d.applied]
    assert applied and applied[0].strategy.value == "acronym"
    # Evidence links the merge back to source mentions.
    assert ibm.evidence.mention_ids

    # 3. Acme linked to the configured KB (graceful fallback would leave kb_id None).
    acme = by_text.get("Acme")
    assert acme is not None
    assert acme.kb_id == "Q1"

    # 4. The works_for relation is normalized onto canonical entities (Alice -> Acme).
    assert any(
        r.label == "works_for" and r.head_text == "Alice" and r.tail_text == "Acme"
        for r in relations
    )

    # 5. Quality Report: DisambiguationQuality present with merge/link stats (S8 AC).
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.disambiguation is not None
    dq = persisted.disambiguation
    assert dq.mentions_in >= 4
    assert dq.entities_out >= 1
    assert dq.entities_merged >= 1
    assert dq.merges_by_strategy.get("acronym", 0) >= 1
    assert dq.entities_linked >= 1          # Acme -> Q1
    assert dq.relations_normalized >= 1
    # "Acme" vs "Acme Corporation" proposes a substring merge at ratio 0.5, below the 0.82
    # policy: it must be LOGGED (applied=False) and counted, not silently dropped — the audit
    # half of "no silent over-merge" (S8 AC), surfaced all the way into the Quality Report.
    assert dq.merges_below_policy >= 1
    below = [
        d
        for e in entities
        for d in e.merge_decisions
        if not d.applied and d.strategy.value == "substring"
    ]
    assert below, "a below-policy substring decision must survive onto an emitted entity"
    assert report.disambiguation is not None
    assert report.disambiguation.entities_out == dq.entities_out
    assert report.disambiguation.merges_below_policy == dq.merges_below_policy

    # 6. Contract completeness holds over the DisambiguationRecords.
    assert persisted.contracts_complete


def test_disambiguation_slice_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    out_a = s.read_text(f"{store}/_latence/runs/det-a/export/records.jsonl")
    out_b = s.read_text(f"{store}/_latence/runs/det-b/export/records.jsonl")
    assert out_a == out_b


def test_disambiguation_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/resume/checkpoints/disambiguate.jsonl"
    lines = [x for x in s.read_text(ckpt).splitlines() if x.strip()]
    assert lines
    scopes = {DisambiguationRecord.model_validate_json(line).scope for line in lines}
    assert DisambiguationScope.ENTITY in scopes


def test_disambiguation_requires_mentions_wiring(tmp_path: Path) -> None:
    # A Disambiguation Stage wired only to Parse (no Entity-Extraction) must error clearly.
    store = f"file://{tmp_path}/store"
    docs = _docs(tmp_path)
    pipeline = Pipeline.from_yaml_text(
        f"""
name: bad-disambig
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [disambiguate]
"""
    )
    try:
        LocalRunner().run(pipeline, run_id="bad")
    except TypeError as exc:
        assert "disambiguation" in str(exc)
        assert "EntityMention" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a TypeError for the mis-wired Disambiguation Stage")

"""End-to-end S5 Relation Extraction through the local Runner.

Two shapes:

1. **Separate Stages** — Source -> Parse -> Chunk -> Entity Extraction -> Relation
   Extraction -> Export, using the in-core deterministic ``entity.gazetteer`` and
   ``relation.pattern``. Asserts the exported records include typed RelationMentions
   whose covering span resolves to the correct source page (the S5 round-trip), that
   the Relation Extraction Stage's multi-input wiring (chunks + mentions) works, that
   the Quality Report's RelationQuality carries per-type counts + the confidence
   distribution, that the run is deterministic and resumes cleanly.

2. **Fused Stage** — Source -> Parse -> Chunk -> (fused Entity+Relation) -> Export,
   using a fake fused Provider injected through the registry. Asserts one fused Stage
   satisfies S4+S5 in one pass (both mentions AND relations exported), the Runner
   routes both downstream WITHOUT a separate Entity-Extraction Stage (no
   double-extraction), and the report carries both entity and relation quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latence_core.capability import FusedExtraction, ProviderRegistry
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    Provenance,
    RelationMention,
)
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _shift(
    chunk: ChunkRecord, doc_start: int, doc_end: int, idx: PageOffsetIndex | None
) -> Provenance:
    """Provenance for a span at document offsets, with its page resolved via the map."""
    update: dict[str, int] = {"char_start": doc_start, "char_end": doc_end}
    if idx is not None:
        update["page_start"] = idx.page_for_offset(doc_start)
        update["page_end"] = idx.page_for_offset(max(doc_end - 1, doc_start))
    return chunk.provenance.model_copy(update=update)


_ENTITY_YAML = """
      labels:
        PERSON:
          terms: [Alice, Bob]
          confidence: 0.8
        ORG:
          terms: [Acme Corp, Globex]
          confidence: 0.95
"""

_RELATION_YAML = """
      relations:
        works_for:
          head: PERSON
          tail: ORG
          triggers: [works for, from]
          confidence: 0.9
"""


def _separate_pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: relation-slice
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
    config:{_ENTITY_YAML}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:{_RELATION_YAML}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [relations]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works for Acme Corp. Bob from Globex replied.\n", encoding="utf-8"
    )
    # An empty document: handled without error (no mentions, no relations).
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_relation_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_separate_pipeline(store, docs), run_id="s5e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s5e2e"

    # 1. The export holds typed RelationMention records.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    relations = [RelationMention.model_validate(r) for r in rows]
    assert relations
    by_pair = {(r.head_text, r.tail_text): r for r in relations}
    assert ("Alice", "Acme Corp") in by_pair
    assert ("Bob", "Globex") in by_pair
    assert by_pair[("Alice", "Acme Corp")].label == "works_for"

    # 2. The S5 round-trip: a relation's covering span resolves to the right page.
    from latence_core.contracts import DocumentRecord

    parse_ckpt = f"{base}/checkpoints/parse.jsonl"
    docs_by_id = {
        d.record_id: d
        for d in (
            DocumentRecord.model_validate_json(line)
            for line in s.read_text(parse_ckpt).splitlines()
            if line.strip()
        )
    }
    for r in relations:
        parent = docs_by_id[r.document_record_id]
        assert parent.page_map is not None
        idx = PageOffsetIndex(parent.page_map)
        assert r.provenance.char_start is not None
        assert idx.page_for_offset(r.provenance.char_start) == r.provenance.page_start
        # Head/tail refs point at real exported/known mention ids in the same doc.
        assert r.head_mention_id.startswith(r.document_record_id)
        assert r.classification.sensitivity == "internal"

    # 3. Quality Report: RelationQuality present with per-type counts + confidence.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.relations is not None
    rq = persisted.relations
    assert rq.relations_total == len(relations)
    assert rq.per_type.get("works_for") == len(relations)
    assert rq.offset_resolving is True
    assert rq.documents_with_relations == 1
    assert rq.confidence.count == len(relations)
    # The report is consistent with what was returned.
    assert report.relations is not None
    assert report.relations.relations_total == persisted.relations.relations_total


def test_relation_slice_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _separate_pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    out_a = s.read_text(f"{store}/_latence/runs/det-a/export/records.jsonl")
    out_b = s.read_text(f"{store}/_latence/runs/det-b/export/records.jsonl")
    assert out_a == out_b


def test_relation_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _separate_pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    # Every Stage (incl. relation extraction) resumes from its checkpoint.
    assert all(m.from_checkpoint for m in report2.stages)
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/resume/checkpoints/relations.jsonl"
    lines = [x for x in s.read_text(ckpt).splitlines() if x.strip()]
    assert lines
    for line in lines:
        r = RelationMention.model_validate_json(line)
        assert r.provenance.char_start is not None
        assert r.provenance.page_start is not None


# -- Fused Provider: one Stage satisfies S4 + S5 in one pass (ADR-0013) --------


class _FakeFusedProvider:
    """A deterministic in-test Fused Provider (stands in for gliner-relex).

    Fulfils FusedEntityRelationExtractor: over each chunk it emits one PERSON and one
    ORG mention and a works_for relation between them, with document-shifted offsets —
    exercising the Runner's fused routing (both carriers from one Stage) without torch.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def extract_fused(self, chunks: Any) -> FusedExtraction:
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        for chunk in chunks:
            if not isinstance(chunk, ChunkRecord) or not chunk.content.strip():
                continue
            out = self._extract_one(chunk)
            if out is None:
                continue
            head, tail, relation = out
            mentions.extend([head, tail])
            relations.append(relation)
        return FusedExtraction(mentions=mentions, relations=relations)

    @staticmethod
    def _extract_one(
        chunk: ChunkRecord,
    ) -> tuple[EntityMention, EntityMention, RelationMention] | None:
        text = chunk.content
        a = text.find("Alice")
        o = text.find("Acme Corp")
        if a < 0 or o < 0:
            return None
        base = chunk.provenance.char_start or 0
        idx = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
        head = EntityMention(
            record_id=f"{chunk.record_id}#entity-0",
            document_record_id=chunk.document_record_id,
            chunk_record_id=chunk.record_id,
            index=0,
            label="PERSON",
            text="Alice",
            confidence=0.9,
            provenance=_shift(chunk, base + a, base + a + len("Alice"), idx),
            classification=chunk.classification,
        )
        tail = EntityMention(
            record_id=f"{chunk.record_id}#entity-1",
            document_record_id=chunk.document_record_id,
            chunk_record_id=chunk.record_id,
            index=1,
            label="ORG",
            text="Acme Corp",
            confidence=0.9,
            provenance=_shift(chunk, base + o, base + o + len("Acme Corp"), idx),
            classification=chunk.classification,
        )
        span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
        span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
        relation = RelationMention(
            record_id=f"{chunk.document_record_id}#relation-0",
            document_record_id=chunk.document_record_id,
            index=0,
            label="works_for",
            head_mention_id=head.record_id,
            tail_mention_id=tail.record_id,
            head_text=head.text,
            tail_text=tail.text,
            confidence=0.88,
            provenance=_shift(chunk, span_start, span_end, idx),
            classification=chunk.classification,
        )
        return head, tail, relation


class _InjectedRegistry(ProviderRegistry):
    """A registry that resolves one extra fused Provider by name, else delegates."""

    def load(self, name: str) -> type:
        if name == "relation.fused_fake":
            return _FakeFusedProvider
        return super().load(name)


def _fused_pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: fused-slice
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
  - name: extract
    capability: fused_entity_relation
    provider: relation.fused_fake
    depends_on: [chunk]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [extract]
"""
    )


def test_fused_provider_satisfies_s4_and_s5_in_one_pass(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_InjectedRegistry())
    report = runner.run(_fused_pipeline(store, docs), run_id="fused")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/fused"

    # Only ONE extraction Stage ran (no separate entity_extraction) — the fused Stage.
    stage_caps = {m.capability for m in report.stages}
    assert "fused_entity_relation" in stage_caps
    assert "entity_extraction" not in stage_caps  # no double-extraction

    # The export holds BOTH mentions and relations from the one fused Stage.
    rows = [
        json.loads(line)
        for line in s.read_text(f"{base}/export/records.jsonl").splitlines()
        if line.strip()
    ]
    mentions = [r for r in rows if "head_mention_id" not in r]
    relations = [r for r in rows if "head_mention_id" in r]
    assert mentions, "fused Stage must export entity mentions (S4 half)"
    assert relations, "fused Stage must export relations (S5 half)"
    rel = RelationMention.model_validate(relations[0])
    assert rel.label == "works_for"
    # The relation's head/tail refs point at mentions the same fused Stage produced.
    mention_ids = {m["record_id"] for m in mentions}
    assert rel.head_mention_id in mention_ids
    assert rel.tail_mention_id in mention_ids

    # The Quality Report carries BOTH entity and relation quality from one Stage.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.entities is not None
    assert persisted.entities.mentions_total == len(mentions)
    assert persisted.relations is not None
    assert persisted.relations.relations_total == len(relations)
    assert persisted.relations.per_type.get("works_for") == len(relations)


def test_fused_stage_resumes_mixed_checkpoint(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_InjectedRegistry())
    runner.run(_fused_pipeline(store, docs), run_id="fused-resume")
    report2 = runner.run(_fused_pipeline(store, docs), run_id="fused-resume")
    # Every Stage resumes; the fused Stage's MIXED checkpoint round-trips to both
    # EntityMentions and RelationMentions.
    assert all(m.from_checkpoint for m in report2.stages)
    assert report2.entities is not None and report2.entities.mentions_total > 0
    assert report2.relations is not None and report2.relations.relations_total > 0


def _colliding_prov() -> Provenance:
    return Provenance(
        source_uri="mem://d",
        file_name="d.txt",
        file_type="txt",
        file_size=64,
        document_id="d1",
        char_start=0,
        char_end=15,
        page_start=1,
        page_end=1,
    )


def test_mixed_checkpoint_reader_does_not_misclassify_head_mention_id_content(
    tmp_path: Path,
) -> None:
    """Regression: an EntityMention whose text/label is literally ``head_mention_id``
    must round-trip through the mixed-checkpoint reader as an EntityMention.

    The discriminator keys on the ``head_mention_id`` JSON *key*, not a substring of
    the raw line, so untrusted mention content that happens to equal the relation
    contract's field name can no longer be misclassified as a RelationMention (which
    previously crashed the resumed run with a pydantic ValidationError).
    """
    cls = Classification(language="en")
    text_collision = EntityMention(
        record_id="e-text",
        document_record_id="d1",
        chunk_record_id="c1",
        index=0,
        label="TOKEN",
        text="head_mention_id",
        confidence=0.9,
        provenance=_colliding_prov(),
        classification=cls,
    )
    label_collision = text_collision.model_copy(
        update={"record_id": "e-label", "label": "head_mention_id", "text": "Alice"}
    )
    relation = RelationMention(
        record_id="r1",
        document_record_id="d1",
        index=0,
        label="works_for",
        head_mention_id="e-text",
        tail_mention_id="e-label",
        head_text="head_mention_id",
        tail_text="Alice",
        confidence=0.88,
        provenance=_colliding_prov(),
        classification=cls,
    )

    payload = "\n".join(
        r.model_dump_json() for r in (text_collision, label_collision, relation)
    )
    records = LocalRunner._read_mixed_lines(payload)

    assert [type(r) for r in records] == [
        EntityMention,
        EntityMention,
        RelationMention,
    ]
    assert records[0].text == "head_mention_id"  # type: ignore[union-attr]
    assert records[1].label == "head_mention_id"  # type: ignore[union-attr]
    assert records[2].head_mention_id == "e-text"  # type: ignore[union-attr]

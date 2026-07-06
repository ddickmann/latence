"""End-to-end S4 Entity Extraction through the local Runner.

Source -> Parse -> Chunk -> Entity Extraction -> Export over a real folder of text
documents. Asserts the exported records are typed EntityMentions whose offsets
resolve back to the correct source page (the S4 round-trip via PageOffsetIndex),
that the Quality Report's EntityQuality section carries per-type counts + the
confidence distribution, that an empty document is handled without error, and that
the run resumes cleanly from checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.contracts import EntityMention
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

_LABELS_YAML = """
      labels:
        ORG:
          terms: [Acme Corp, Globex]
          confidence: 0.95
        PERSON:
          terms: [Alice, Bob]
          confidence: 0.8
"""


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: entity-slice
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
    config:{_LABELS_YAML}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice from Acme Corp emailed Bob about the Globex deal.\n", encoding="utf-8"
    )
    # An empty document: must be handled without error (S4 AC), producing no mentions.
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_entity_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s4e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s4e2e"

    # 1. The export holds typed EntityMention records (the S4 corpus is mentions).
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    mentions = [EntityMention.model_validate(r) for r in rows]
    by_text = {m.text: m for m in mentions}
    assert set(by_text) == {"Alice", "Acme Corp", "Bob", "Globex"}
    assert by_text["Acme Corp"].label == "ORG"
    assert by_text["Alice"].label == "PERSON"

    # 2. The S4 round-trip: a mention's document offset resolves to the right source
    # page. Rebuild the parent document's page map from the Parse checkpoint and
    # confirm every mention's char span resolves to its recorded page.
    parse_ckpt = f"{base}/checkpoints/parse.jsonl"
    from latence_core.contracts import DocumentRecord

    docs_by_id = {
        d.record_id: d
        for d in (
            DocumentRecord.model_validate_json(line)
            for line in s.read_text(parse_ckpt).splitlines()
            if line.strip()
        )
    }
    for m in mentions:
        parent = docs_by_id[m.document_record_id]
        assert parent.page_map is not None
        idx = PageOffsetIndex(parent.page_map)
        assert m.provenance.char_start is not None
        resolved_page = idx.page_for_offset(m.provenance.char_start)
        assert resolved_page == m.provenance.page_start
        # The recorded surface string matches the parent document at that span.
        assert parent.content[m.provenance.char_start : m.provenance.char_end] == m.text
        # Classification inherited from the document (via the chunk), unchanged.
        assert m.classification.sensitivity == "internal"

    # 3. Quality Report: EntityQuality present with per-type counts + confidence dist.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.entities is not None
    eq = persisted.entities
    assert eq.mentions_total == 4
    assert eq.per_type == {"ORG": 2, "PERSON": 2}
    assert eq.offset_resolving is True
    assert eq.mentions_offset_resolving == 4
    # Confidence distribution: 2 ORG @0.95 (top bin) + 2 PERSON @0.8 (bin 8).
    assert eq.confidence.count == 4
    assert eq.confidence.bins[9] == 2
    assert eq.confidence.bins[8] == 2
    assert 0.8 <= eq.confidence.mean <= 0.95
    # The empty doc produced no mentions but did not crash the run.
    assert eq.chunks_with_mentions == 1
    # Returned report is consistent with what was persisted.
    assert report.entities is not None
    assert report.entities.mentions_total == persisted.entities.mentions_total


def test_entity_slice_resolves_offsets_through_markdown_markup(tmp_path: Path) -> None:
    # A markdown document whose entity mentions all sit AFTER stripped markup (a heading
    # and a bold run). End to end, every exported mention's span must index the ORIGINAL
    # markdown exactly — the S4 coordinate fix (char_start + local would be left-shifted).
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(
        "# Memo\n\n**Reminder:** Alice from Acme Corp will meet Bob about Globex.\n",
        encoding="utf-8",
    )
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_pipeline(store, docs), run_id="s4md")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s4md"

    rows = [
        json.loads(line)
        for line in s.read_text(f"{base}/export/records.jsonl").splitlines()
        if line.strip()
    ]
    mentions = [EntityMention.model_validate(r) for r in rows]
    by_text = {m.text: m for m in mentions}
    assert {"Alice", "Acme Corp", "Bob", "Globex"} <= set(by_text)

    from latence_core.contracts import DocumentRecord

    docs_by_id = {
        d.record_id: d
        for d in (
            DocumentRecord.model_validate_json(line)
            for line in s.read_text(f"{base}/checkpoints/parse.jsonl").splitlines()
            if line.strip()
        )
    }
    for m in mentions:
        parent = docs_by_id[m.document_record_id]
        assert m.provenance.char_start is not None and m.provenance.char_end is not None
        # The exact original slice at the recorded span equals the surface string.
        assert parent.content[m.provenance.char_start : m.provenance.char_end] == m.text
        # And the mention's own page resolves from its TRUE offset.
        assert parent.page_map is not None
        idx = PageOffsetIndex(parent.page_map)
        assert idx.page_for_offset(m.provenance.char_start) == m.provenance.page_start
    # "Alice" is not left-shifted onto the stripped "**Reminder:**" prefix.
    alice = by_text["Alice"]
    assert (
        docs_by_id[alice.document_record_id]
        .content[alice.provenance.char_start :]
        .startswith("Alice")
    )


def test_entity_slice_is_deterministic(tmp_path: Path) -> None:
    # Two runs over the SAME input + config (identical source_uri) into two run ids
    # produce byte-identical exported mentions (hash-stable, seeded Baseline bar).
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    out_a = s.read_text(f"{store}/_latence/runs/det-a/export/records.jsonl")
    out_b = s.read_text(f"{store}/_latence/runs/det-b/export/records.jsonl")
    assert out_a == out_b


def test_entity_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    # Every Stage (incl. entity extraction) resumes from its checkpoint.
    assert all(m.from_checkpoint for m in report2.stages)
    # The resumed entity checkpoint round-trips to valid EntityMentions.
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/resume/checkpoints/entities.jsonl"
    lines = [x for x in s.read_text(ckpt).splitlines() if x.strip()]
    assert lines
    for line in lines:
        m = EntityMention.model_validate_json(line)
        assert m.provenance.char_start is not None
        assert m.provenance.page_start is not None

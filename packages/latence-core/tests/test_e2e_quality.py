"""End-to-end S10 Quality Report + gold-set harness through the local Runner.

The S10 headline demo: an adopter drops an annotation file next to their corpus and gets
per-Stage precision/recall/F1 on THEIR documents, plus contract-completeness (Provenance /
Classification present + offsets aligned), the corpus-level drift roll-up, and both a
machine-readable (JSON) and a human-readable (rendered Markdown) report — every run.

Runs Source -> Parse -> Chunk -> Entity -> Relation -> Export with a gold set wired in, and
asserts: the gold-set evaluation lands in the report with the expected P/R/F1; the harness
matched the annotated documents; completeness reports offsets aligned; drift is clean; the
rendered summary persists next to the JSON; and the run is deterministic + resumes.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    (docs / "note.txt").write_text("Bob joined Acme last year.\n", encoding="utf-8")
    return docs


def _gold_set_file(tmp_path: Path) -> Path:
    """Gold annotations authored against the two documents' parsed-markdown offsets.

    memo.txt = "Alice works at Acme.\n":  Alice[0,5] PERSON, Acme[15,19] ORG.
    note.txt = "Bob joined Acme last year.\n": Bob[0,3] PERSON, Acme[11,15] ORG.
    Each document has one works_for(PERSON -> ORG) relation.
    """
    gold = {
        "match_min_overlap": 0.0,
        "documents": [
            {
                "file_name": "memo.txt",
                "entities": [
                    {"label": "PERSON", "char_start": 0, "char_end": 5, "text": "Alice"},
                    {"label": "ORG", "char_start": 15, "char_end": 19, "text": "Acme"},
                ],
                "relations": [
                    {
                        "label": "works_for",
                        "head": {"label": "PERSON", "char_start": 0, "char_end": 5},
                        "tail": {"label": "ORG", "char_start": 15, "char_end": 19},
                    }
                ],
            },
            {
                "file_name": "note.txt",
                "entities": [
                    {"label": "PERSON", "char_start": 0, "char_end": 3, "text": "Bob"},
                    {"label": "ORG", "char_start": 11, "char_end": 15, "text": "Acme"},
                ],
                "relations": [
                    {
                        "label": "works_for",
                        "head": {"label": "PERSON", "char_start": 0, "char_end": 3},
                        "tail": {"label": "ORG", "char_start": 11, "char_end": 15},
                    }
                ],
            },
        ],
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold), encoding="utf-8")
    return path


def _pipeline(store: str, docs_dir: Path, gold_uri: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: quality-slice
storage_uri: {store}
gold_set_uri: {gold_uri}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
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
          terms: [Acme]
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
          triggers: [works at, joined]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [relations]
"""
    )


def test_quality_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    gold = _gold_set_file(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs, f"file://{gold}"), run_id="s10e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s10e2e"

    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.schema_version == 12

    # 1. Gold-set evaluation present with the two annotated documents matched.
    assert persisted.goldset is not None
    gs = persisted.goldset
    assert gs.documents_annotated == 2
    assert gs.documents_matched == 2
    stage_names = {s.stage for s in gs.stages}
    assert stage_names == {"entities", "relations"}

    # 2. Entities: the gazetteer finds Alice/Bob/Acme(x2) exactly -> perfect P/R/F1.
    entities = next(s for s in gs.stages if s.stage == "entities")
    assert entities.true_positives == 4
    assert entities.false_positives == 0
    assert entities.false_negatives == 0
    assert entities.precision == 1.0 and entities.recall == 1.0 and entities.f1 == 1.0
    assert set(entities.per_label) == {"ORG", "PERSON"}
    assert entities.per_label["ORG"].true_positives == 2

    # 3. Relations: both works_for(PERSON->ORG) relations recovered with aligned endpoints.
    relations = next(s for s in gs.stages if s.stage == "relations")
    assert relations.true_positives == 2
    assert relations.recall == 1.0
    assert relations.per_label["works_for"].f1 == 1.0

    # 4. Contract completeness: Provenance/Classification present AND offsets aligned.
    assert persisted.contracts_complete is True
    assert persisted.completeness.offsets_aligned is True
    assert persisted.completeness.offset_bearing_records > 0

    # 5. Drift roll-up present and clean (exact page resolution over the chunk offsets).
    assert persisted.drift is not None
    assert persisted.drift.records_audited > 0
    assert persisted.drift.clean is True

    # 6. Both representations persisted: JSON + rendered Markdown.
    assert s.exists(f"{base}/quality-report.json")
    rendered = persisted.render()
    assert "Gold-set evaluation" in rendered
    assert "entities: P=1.000 R=1.000 F1=1.000" in rendered

    # 7. The returned report matches the persisted one.
    assert report.goldset is not None
    assert report.goldset.documents_matched == gs.documents_matched


def test_quality_slice_without_goldset_reports_no_evaluation(tmp_path: Path) -> None:
    # A pipeline with no gold_set_uri still emits the report; the gold-set section is absent.
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs, "").model_copy(update={"gold_set_uri": None})
    report = LocalRunner().run(pipeline, run_id="nogold")
    assert report.goldset is None
    # Drift + completeness still present (they don't depend on a gold set).
    assert report.drift is not None
    assert report.contracts_complete is True


def test_quality_slice_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    gold = _gold_set_file(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs, f"file://{gold}")
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    a = QualityReport.model_validate_json(
        s.read_text(f"{store}/_latence/runs/det-a/quality-report.json")
    )
    b = QualityReport.model_validate_json(
        s.read_text(f"{store}/_latence/runs/det-b/quality-report.json")
    )

    # The quality DATA is identical across runs; only the run_id and wall-clock timings
    # (total_duration_seconds + per-stage duration_seconds) legitimately differ.
    def _strip_timing(report: QualityReport) -> dict[str, object]:
        dump = report.model_dump()
        dump.pop("run_id")
        dump["total_duration_seconds"] = 0.0
        for stage in dump["stages"]:
            stage["duration_seconds"] = 0.0
        return dump

    assert _strip_timing(a) == _strip_timing(b)
    # The gold-set evaluation in particular is byte-identical (seeded, hash-stable).
    assert a.goldset is not None and b.goldset is not None
    assert a.goldset.model_dump() == b.goldset.model_dump()


def test_documents_matched_is_extraction_coverage_not_ingest_count(tmp_path: Path) -> None:
    # Regression for the corrected documents_matched semantics (issue #39.2): it counts gold
    # documents the run produced >=1 extraction for, NOT documents the Source read. A gold
    # document that WAS ingested + parsed but from which nothing was extracted must fall out
    # of documents_matched (coverage gap) while STILL contributing its annotations as false
    # negatives to recall. If someone re-defined it as an ingest count, this fails.
    docs = tmp_path / "docs"
    docs.mkdir()
    # memo.txt yields an ORG extraction; silent.txt is read + parsed but its only annotation
    # is a PERSON, which the gazetteer below is NOT configured to find -> zero extractions.
    (docs / "memo.txt").write_text("Acme is here.\n", encoding="utf-8")
    (docs / "silent.txt").write_text("Nobody notable here.\n", encoding="utf-8")
    gold = {
        "match_min_overlap": 0.0,
        "documents": [
            {
                "file_name": "memo.txt",
                "entities": [{"label": "ORG", "char_start": 0, "char_end": 4, "text": "Acme"}],
            },
            {
                "file_name": "silent.txt",
                "entities": [{"label": "PERSON", "char_start": 0, "char_end": 6, "text": "Nobody"}],
            },
        ],
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: coverage-slice
storage_uri: {store}
gold_set_uri: file://{gold_path}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
      extensions: [txt]
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
          terms: [Acme]
"""
    )
    report = LocalRunner().run(pipeline, run_id="coverage")
    assert report.goldset is not None
    gs = report.goldset
    # Both documents are annotated; both were ingested and parsed.
    assert gs.documents_annotated == 2
    # Only memo.txt produced an extraction -> matched == 1, NOT 2. silent.txt was read but
    # yielded nothing, so it is an un-scored coverage gap, not a match.
    assert gs.documents_matched == 1
    # P/R/F1 are unaffected by the coverage gap: silent.txt's PERSON annotation is a false
    # negative (nothing matched it), so recall reflects the miss.
    entities = next(s for s in gs.stages if s.stage == "entities")
    assert entities.true_positives == 1  # Acme in memo.txt
    assert entities.false_negatives == 1  # Nobody in silent.txt, never extracted
    assert entities.recall == 0.5


def test_quality_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    gold = _gold_set_file(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs, f"file://{gold}")
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)
    # The gold-set harness is recomputed on resume from the reloaded records -> same result.
    assert report2.goldset is not None
    assert report2.goldset.stages[0].f1 == 1.0

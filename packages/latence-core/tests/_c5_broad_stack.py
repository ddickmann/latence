"""The broad stack behind the C5 byte-identity fixture (shared by the test and its generator).

Kept beside the test rather than inside it so the checked-in fixture
(``tests/data/quality_report_c5_fixture.json``) can be regenerated from the SAME pipeline
definition the assertion runs — a fixture built from a drifted copy of the stack would prove
nothing. Regenerate with::

    PYTHONHASHSEED=0 uv run python packages/latence-core/tests/_c5_broad_stack.py

Only do that deliberately: the fixture's whole job is to fail when the report's bytes move.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner

RUN_ID = "c5"
FIXTURE = Path(__file__).parent / "data" / "quality_report_c5_fixture.json"


def docs(tmp_path: Path) -> Path:
    """Three tiny documents that between them populate every report section."""
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "memo.txt").write_text(
        "Alice works at Acme. Contact alice@acme.example for details.\n",
        encoding="utf-8",
    )
    (directory / "note.txt").write_text(
        "Bob joined International Business Machines. IBM partnered with Acme.\n",
        encoding="utf-8",
    )
    (directory / "brief.txt").write_text(
        "Globex partnered with Acme. Carol works at Globex.\n",
        encoding="utf-8",
    )
    return directory


def gold_set(tmp_path: Path) -> Path:
    """Gold annotations over ``memo.txt`` so the gold-set section is populated too."""
    gold = {
        "match_min_overlap": 0.0,
        "documents": [
            {
                "file_name": "memo.txt",
                "entities": [
                    {"label": "person", "char_start": 0, "char_end": 5, "text": "Alice"},
                    {"label": "org", "char_start": 15, "char_end": 19, "text": "Acme"},
                ],
                "relations": [
                    {
                        "label": "works_for",
                        "head": {"label": "person", "char_start": 0, "char_end": 5},
                        "tail": {"label": "org", "char_start": 15, "char_end": 19},
                    }
                ],
            }
        ],
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold), encoding="utf-8")
    return path


def broad_pipeline(store: str, docs_dir: Path, gold_uri: str) -> Pipeline:
    """One stack wiring every in-core Capability that contributes a Quality-Report section.

    source -> intake_screen -> parse -> chunk -> content_screen -> {entities, relations} ->
    consolidate -> disambiguate -> graph -> complete -> enrich -> export, plus redaction and
    profiling branches and a gold set. Deliberately broad: a byte-identity fixture is only
    meaningful if it covers the sections the fold has to reproduce.
    """
    return Pipeline.from_yaml_text(
        f"""
name: c5-aggregators
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
  - name: intake_screen
    capability: intake_screening
    provider: screening.intake_signature
    depends_on: [source]
    config:
      max_bytes: 2097152
      max_zip_ratio: 50.0
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [intake_screen]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
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
        org:
          terms: [Acme, Globex]
        organization:
          terms: [International Business Machines, IBM]
        person:
          terms: [Alice, Bob, Carol]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [content_screen, entities]
    config:
      window_chars: 200
      relations:
        works_for:
          head: person
          tail: org
          triggers: [works at, joined]
        partner_of:
          head: org
          tail: org
          triggers: [partnered with]
  - name: consolidate
    capability: type_consolidation
    provider: type_consolidation.cascade
    depends_on: [entities, relations]
    config:
      aliases:
        organization: org
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [content_screen]
    config:
      salt: c5-secret-salt
  - name: profiling
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, entities]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [consolidate]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: c5
  - name: complete
    capability: graph_completion
    provider: graph_completion.reference
    depends_on: [graph]
    config:
      corpus_id: c5
      transitive_labels: [partner_of]
      symmetric_labels: [partner_of]
  - name: enrich
    capability: context_enrichment
    provider: context.kg_header
    depends_on: [redact, graph]
    config:
      max_neighbors_per_entity: 1
      max_triples: 6
      min_entity_confidence: 0.0
      min_edge_confidence: 0.0
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [enrich]
    config:
      basename: corpus
"""
    )


def timing_free(report: QualityReport) -> str:
    """The report's JSON with wall-clock durations zeroed — everything else byte-for-byte.

    Durations are the only fields a re-run legitimately changes, so they are canonicalised
    (never dropped: a missing key would hide a shape change) and the remaining bytes compared
    exactly.
    """
    canonical = report.model_copy(deep=True)
    canonical.total_duration_seconds = 0.0
    for stage in canonical.stages:
        stage.duration_seconds = 0.0
    return canonical.to_json()


def run_broad(tmp_path: Path) -> QualityReport:
    """Run the broad stack under ``tmp_path`` and return its Quality Report."""
    directory = docs(tmp_path)
    gold = gold_set(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = broad_pipeline(store, directory, f"file://{gold}")
    return LocalRunner().run(pipeline, run_id=RUN_ID)


def _regenerate() -> None:  # pragma: no cover - developer entry point
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        report = run_broad(Path(tmp))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(timing_free(report), encoding="utf-8")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":  # pragma: no cover - developer entry point
    _regenerate()

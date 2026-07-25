"""End-to-end T3 type canonicalization through the local Runner (ADR-0054).

Source -> Parse -> Chunk -> Entity -> Relation -> **Type Consolidation** -> Disambiguation ->
Graph -> Export, on a corpus whose documents induce DRIFTED type names for one real type — ``org``
in one document, ``organization`` in another, ``company`` in a third, exactly what per-chunk label
induction produces once no corpus-wide schema is forced upstream.

The anti-false-green construction is ``test_without_the_stage_the_drift_reaches_the_graph`` next to
``test_the_drift_collapses_to_one_canonical_type_in_the_graph``: the SAME corpus and the SAME
downstream stages, differing only in whether the consolidation Stage is wired. The first is the
pre-change behaviour (three node types in the KG) and is the regression witness this lane exists to
close; the second is the fix. Both assert on the assembled graph, never on a config flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from latence_core.contracts import GraphRecord, GraphScope, TypeKind, TypeVocabulary
from latence_core.errors import ContractError
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

# One real type, three induced names. The gazetteer is configured per document family so each
# document's mentions carry a different surface for the same type — the drift per-chunk induction
# creates, reproduced deterministically without a model.
_LABELS = """
      labels:
        org:
          terms: [Acme Corporation, Acme]
        organization:
          terms: [International Business Machines, IBM]
        company:
          terms: [Globex]
        person:
          terms: [Alice, Bob]
"""

_CONSOLIDATE_STAGE = """
  - name: consolidate_types
    capability: type_consolidation
    provider: type_consolidation.cascade
    depends_on: [entities, relations]
    config:
      aliases:
        company: organization
"""


def _pipeline(store: str, docs_dir: Path, *, consolidate: bool) -> Pipeline:
    """The same pipeline with and without the consolidation Stage — the only difference."""
    disambiguate_parents = "[consolidate_types]" if consolidate else "[entities, relations]"
    return Pipeline.from_yaml_text(
        f"""
name: type-consolidation-slice
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
    config:{_LABELS}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      window_chars: 200
      relations:
        works_for:
          head: person
          tail: org
          triggers: [works at, employed by, joined]
        Works_For:
          head: person
          tail: organization
          triggers: [works at, employed by, joined]
{_CONSOLIDATE_STAGE if consolidate else ""}
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: {disambiguate_parents}
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: t3
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [graph]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Alice works at Acme Corporation.\n", encoding="utf-8")
    (docs / "b.txt").write_text(
        "Bob joined International Business Machines last year. IBM announced results.\n",
        encoding="utf-8",
    )
    (docs / "c.txt").write_text("Globex filed the report.\n", encoding="utf-8")
    return docs


def _node_types(store: str, run_id: str) -> set[str]:
    """The entity types the assembled knowledge graph actually carries."""
    s = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/export/records.jsonl"
    rows = [json.loads(line) for line in s.read_text(uri).splitlines() if line.strip()]
    nodes = [GraphRecord.model_validate(r) for r in rows]
    return {
        n.node.label
        for n in nodes
        if n.scope is GraphScope.NODE and n.node is not None and n.node.label != "person"
    }


def _report(store: str, run_id: str) -> QualityReport:
    s = Storage.for_uri(store)
    return QualityReport.model_validate_json(
        s.read_text(f"{store}/_latence/runs/{run_id}/quality-report.json")
    )


# -- the regression: the control arm, then the fix ---------------------------


def test_without_the_stage_the_drift_reaches_the_graph(tmp_path: Path) -> None:
    """PRE-CHANGE BEHAVIOUR (the witness): three induced names ⇒ three KG node types.

    Nothing between extraction and graph assembly canonicalized the type vocabulary, so a corpus
    that induced ``org``/``organization``/``company`` for one real type built a graph with three.
    This is the regression per-chunk induction would have shipped; the next test is the fix.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-drift"
    LocalRunner().run(_pipeline(store, docs, consolidate=False), run_id="drift")
    assert _node_types(store, "drift") == {"org", "organization", "company"}
    assert _report(store, "drift").type_consolidation is None


def test_the_drift_collapses_to_one_canonical_type_in_the_graph(tmp_path: Path) -> None:
    """THE FIX: the same corpus, with the Stage wired, yields ONE canonical node type."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-clean"
    LocalRunner().run(_pipeline(store, docs, consolidate=True), run_id="clean")
    assert _node_types(store, "clean") == {"organization"}


def test_the_relation_vocabulary_collapses_too(tmp_path: Path) -> None:
    """``works_for`` and ``Works_For`` are one edge type, not two."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-rel"
    LocalRunner().run(_pipeline(store, docs, consolidate=True), run_id="rel")
    s = Storage.for_uri(store)
    rows = [
        json.loads(line)
        for line in s.read_text(f"{store}/_latence/runs/rel/export/records.jsonl").splitlines()
        if line.strip()
    ]
    edges = [GraphRecord.model_validate(r) for r in rows]
    labels = {
        e.edge.label for e in edges if e.scope is GraphScope.EDGE and e.edge is not None
    }
    assert len(labels) <= 1, f"the drifted relation labels must collapse, got {labels}"


# -- auditability: the operator must SEE the consolidation -------------------


def test_the_quality_report_shows_the_collapse_and_the_alias_map(tmp_path: Path) -> None:
    """"N raw induced types -> M canonical types", plus every rewrite that produced it."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-report"
    LocalRunner().run(_pipeline(store, docs, consolidate=True), run_id="rep")
    section = _report(store, "rep").type_consolidation
    assert section is not None
    assert section.raw_entity_types == 4  # org, organization, company, person
    assert section.canonical_entity_types == 2  # organization, person
    assert section.entity_alias_map == {"company": "organization", "org": "organization"}
    assert section.mentions_remapped > 0


def test_the_rendered_report_names_the_consolidation_for_a_human(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-render"
    LocalRunner().run(_pipeline(store, docs, consolidate=True), run_id="rend")
    rendered = _report(store, "rend").render()
    assert "## Type consolidation" in rendered
    assert "4 raw -> 2 canonical" in rendered
    assert "org -> organization" in rendered


def test_the_raw_induced_label_reaches_the_checkpoint(tmp_path: Path) -> None:
    """The audit trail is on the durable record, not only in memory (ADR-0017)."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-raw"
    LocalRunner().run(_pipeline(store, docs, consolidate=True), run_id="raw")
    s = Storage.for_uri(store)
    rows = [
        json.loads(line)
        for line in s.read_text(
            f"{store}/_latence/runs/raw/checkpoints/consolidate_types.jsonl"
        ).splitlines()
        if line.strip()
    ]
    remapped = [r for r in rows if r.get("raw_label") and "head_mention_id" not in r]
    assert remapped, "every rewritten mention must carry the type it was induced as"
    assert {r["raw_label"] for r in remapped} == {"org", "company"}
    assert all(r["label"] == "organization" for r in remapped)
    # Unrewritten records carry no raw_label, so the field means "this was remapped".
    assert any(r.get("raw_label") is None for r in rows)


# -- the vocabulary is a durable, mergeable artifact (T4) --------------------


def test_the_canonical_vocabulary_is_persisted_beside_the_run(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-vocab"
    LocalRunner().run(_pipeline(store, docs, consolidate=True), run_id="voc")
    s = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/voc/type-vocabulary/consolidate_types.json"
    assert s.exists(uri)
    vocabulary = TypeVocabulary.model_validate_json(s.read_text(uri))
    assert vocabulary.alias_map(TypeKind.ENTITY) == {
        "company": "organization",
        "org": "organization",
    }
    assert vocabulary.canonical_for(TypeKind.ENTITY, "org") == "organization"
    # A type the vocabulary never saw passes through unchanged — the remap is TOTAL, so a delta
    # run's brand-new type is never dropped nor mapped into a wrong cluster.
    assert vocabulary.canonical_for(TypeKind.ENTITY, "statute") == "statute"


def test_the_runner_reads_the_vocabulary_back(tmp_path: Path) -> None:
    """The seam an incremental delta run seeds from."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-readback"
    runner = LocalRunner()
    pipeline = _pipeline(store, docs, consolidate=True)
    runner.run(pipeline, run_id="rb")
    s = Storage.for_uri(store)
    assert runner.read_type_vocabulary(s, pipeline, "rb", "consolidate_types") is not None
    assert runner.read_type_vocabulary(s, pipeline, "rb", "no_such_stage") is None


# -- determinism + wiring failures -------------------------------------------


def test_two_runs_of_the_same_corpus_produce_an_identical_vocabulary(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    first_store = f"file://{tmp_path}/store-d1"
    second_store = f"file://{tmp_path}/store-d2"
    LocalRunner().run(_pipeline(first_store, docs, consolidate=True), run_id="d")
    LocalRunner().run(_pipeline(second_store, docs, consolidate=True), run_id="d")
    first = Storage.for_uri(first_store).read_text(
        f"{first_store}/_latence/runs/d/type-vocabulary/consolidate_types.json"
    )
    second = Storage.for_uri(second_store).read_text(
        f"{second_store}/_latence/runs/d/type-vocabulary/consolidate_types.json"
    )
    assert first == second


def test_a_consolidation_stage_wired_to_the_wrong_parent_fails_loudly(tmp_path: Path) -> None:
    """No silent pass-through: a mis-wired Stage is a typed error, not drifted types + a clean report."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store-miswired"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: miswired
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
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
  - name: consolidate_types
    capability: type_consolidation
    provider: type_consolidation.cascade
    depends_on: [chunk]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [consolidate_types]
"""
    )
    with pytest.raises(ContractError, match="received no EntityMentions"):
        LocalRunner().run(pipeline, run_id="mis")

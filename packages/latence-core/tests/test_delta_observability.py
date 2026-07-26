"""A delta run is observable — the same seams ``latence process`` has (C3, defect c).

``DeltaRunner.__init__`` constructed a bare ``LocalRunner(registry)`` and exposed no way to reach
either observability seam, so ``latence delta`` — the DAILY enterprise path — **structurally
could not** emit per-Stage spans or OpenLineage events while ``latence process`` could. Not a
configuration gap: there was no parameter. Both seams are now constructor arguments, defaulting to
``None`` (off), so an un-wired delta run stays byte-identical.

This pins the structural fix. The CLI's three ``DeltaRunner()`` call sites still need the same
``_lineage_emitter_from_env()`` the ``process`` command already passes — see the lane report.
"""

from __future__ import annotations

from pathlib import Path

from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.tracing import CollectingSpanSink


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: delta-observability
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
        ORG: {{terms: [Acme]}}
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Acme shipped the order.\n", encoding="utf-8")
    return docs


def test_delta_run_emits_per_stage_spans_when_a_sink_is_wired(tmp_path: Path) -> None:
    """`latence delta` can now be traced. Pre-fix there was no parameter to pass a sink through."""
    docs = _docs(tmp_path)
    sink = CollectingSpanSink()

    DeltaRunner(span_sink=sink).apply_delta(
        _pipeline(f"file://{tmp_path}/store", docs), run_id="d0"
    )

    begun = {s.stage for s in sink.spans if s.event == "stage_begin"}
    ended = {s.stage for s in sink.spans if s.event == "stage_end"}
    assert {"source", "parse", "chunk", "entities", "disambiguate"} <= begun
    assert begun == ended, "every Stage that began must have a matching end span"


def test_delta_run_without_a_sink_is_unchanged(tmp_path: Path) -> None:
    """Off by default: a plain delta run behaves exactly as before (both seams ``None``)."""
    docs = _docs(tmp_path)
    report = DeltaRunner().apply_delta(
        _pipeline(f"file://{tmp_path}/store", docs), run_id="d0"
    )
    assert report.delta is not None
    assert report.delta.to_version == 0

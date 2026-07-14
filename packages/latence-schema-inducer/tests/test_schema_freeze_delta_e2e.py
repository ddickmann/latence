"""W17 (ADR-0043) — the FROZEN induced schema across an incremental delta, end to end.

The coupling issue #42 exposes: when ``schema_induction`` runs at ``granularity: corpus`` it induces
ONE shared vocabulary from ALL chunks, so adding a document could shift the induced labels and
invalidate the unchanged documents' extraction. W17 FREEZES the induced schema in the Corpus
Version and reuses it by default, so:

1. the FIRST ``apply_delta`` induces the corpus schema (LLM calls) and freezes it;
2. a subsequent ``apply_delta`` that adds a document makes NO induction LLM call — the frozen schema
   is stamped on the new document's chunks, so it extracts under the SAME labels as the reused ones;
3. ``--reinduce`` re-runs induction over the full corpus (LLM calls again) — the escape hatch.

Driven by the faithful offline ``openai`` mock (no network), through the real DeltaRunner + Runner.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import _fake_openai as fake
from latence_core.contracts import ChunkRecord, InducedLabels
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.state.corpus_store import CorpusStore
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA_REPLY = json.dumps(
    {"entity_types": ["Company", "Person"], "relation_types": ["employs"], "pii_types": ["email"]}
)


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: schema-freeze
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
  - name: induce
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [chunk]
    config: {{granularity: corpus, api_key: test-key}}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [induce]
"""
    )


def _induced_labels_in_run(store: str, run_id: str) -> list[InducedLabels]:
    """Every non-None induced_labels carrier in a run's induce-Stage checkpoint."""
    storage = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/checkpoints/induce.jsonl"
    if not storage.exists(uri):
        return []
    out: list[InducedLabels] = []
    for line in storage.read_text(uri).splitlines():
        if line.strip():
            chunk = ChunkRecord.model_validate_json(line)
            if chunk.induced_labels is not None:
                out.append(chunk.induced_labels)
    return out


def test_schema_is_frozen_then_reused_then_reinduced(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    cs = CorpusStore(Storage.for_uri(store), store)
    (docs / "a.txt").write_text("Acme employs Alice.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Globex employs Bob.\n", encoding="utf-8")

    # V0: first run induces the corpus schema (LLM is called) and freezes it in the Version.
    with fake.install([fake.reply(_SCHEMA_REPLY), fake.reply(_SCHEMA_REPLY)]) as script:
        DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
        assert script.calls, "the first run must call the inducer LLM"
    frozen = cs.head_schema()
    assert frozen is not None
    assert set(frozen.entity_types) == {"company", "person"}

    # V1: add a document. The frozen schema is reused — NO induction LLM call is made, and the new
    # document's chunk still carries the SAME frozen labels (extraction stays under one vocabulary).
    (docs / "c.txt").write_text("Initech employs Carol.\n", encoding="utf-8")
    with fake.install([fake.reply(_SCHEMA_REPLY)]) as script:
        r1 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
        assert script.calls == [], "a frozen-schema delta must make NO induction LLM call"
    assert r1.delta is not None
    assert r1.delta.documents_reused == 2
    assert r1.delta.documents_extracted == 1
    # The added document's freshly-run chunk carries the frozen labels.
    fresh_labels = _induced_labels_in_run(store, "v1")
    assert fresh_labels, "the added document's chunk was induced-stamped"
    assert all(set(lbl.entity_types) == {"company", "person"} for lbl in fresh_labels)
    # The frozen schema is unchanged in the new Version.
    assert cs.head_schema() == frozen

    # --reinduce: re-run induction over the FULL current corpus (LLM called again) — the escape
    # hatch that lets the vocabulary move with the data.
    with fake.install(
        [fake.reply(_SCHEMA_REPLY), fake.reply(_SCHEMA_REPLY), fake.reply(_SCHEMA_REPLY)]
    ) as script:
        r2 = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v2", reinduce=True)
        assert script.calls, "--reinduce must re-run the inducer LLM over the full corpus"
    assert r2.delta is not None
    assert r2.delta.documents_reused == 0
    assert r2.delta.documents_extracted == 3

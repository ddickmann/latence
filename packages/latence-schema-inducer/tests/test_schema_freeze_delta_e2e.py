"""Induction across an incremental delta, end to end (ADR-0043 as amended by T4).

**Two induction models, two different cross-run guarantees.**

``granularity: corpus`` (the legacy, still-selectable path) induces ONE shared vocabulary from ALL
chunks, so adding a document could shift the induced labels and invalidate the unchanged documents'
extraction — the issue-#42 coupling. W17 breaks it by FREEZING that single schema in the Corpus
Version:

1. the FIRST ``apply_delta`` induces the corpus schema (LLM calls) and freezes it;
2. a subsequent ``apply_delta`` that adds a document makes NO induction LLM call — the frozen schema
   is stamped on the new document's chunks, so it extracts under the SAME labels as the reused ones;
3. ``--reinduce`` re-runs induction over the full corpus (LLM calls again) — the escape hatch.

``granularity: chunk`` (the default since T1, and what ``latence setup`` emits) has **no such
coupling to break**: a chunk's labels are a function of its own text alone. T4's model is therefore
NOT to freeze anything but to induce the AFFECTED CHUNKS ONLY — the unchanged documents are excluded
at Source, so their chunks are never re-induced and their labels are carried on their reused chunk
records. ``test_a_per_chunk_delta_induces_only_the_added_document_s_chunks`` is the witness: the
delta makes exactly one LLM call per NEW chunk and none for the corpus, and the new chunk keeps its
OWN induced labels rather than the base run's. Cross-run type consistency for this path is earned
downstream, by the persisted canonical type vocabulary (ADR-0054, exercised in
``latence-core/tests/test_e2e_delta_type_vocabulary.py``).

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
# What the added document's OWN chunk induces — deliberately disjoint from the base corpus's
# schema, so "the new chunk kept its own labels" cannot be confused with "it got the frozen ones".
_DELTA_REPLY = json.dumps(
    {"entity_types": ["Statute"], "relation_types": ["supersedes"], "pii_types": []}
)


def _pipeline(store: str, docs_dir: Path, *, granularity: str = "corpus") -> Pipeline:
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
    config: {{granularity: {granularity}, api_key: test-key}}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [induce]
"""
    )


def _induced_labels_in_run(
    store: str, run_id: str, *, stage: str = "induce"
) -> list[InducedLabels]:
    """Every non-None induced_labels carrier in a run's induce-Stage checkpoint."""
    storage = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/checkpoints/{stage}.jsonl"
    if not storage.exists(uri):
        return []
    out: list[InducedLabels] = []
    for line in storage.read_text(uri).splitlines():
        if line.strip():
            chunk = ChunkRecord.model_validate_json(line)
            if chunk.induced_labels is not None:
                out.append(chunk.induced_labels)
    return out


def test_a_per_chunk_delta_induces_only_the_added_documents_chunks(tmp_path: Path) -> None:
    """T4: per-chunk induction on a delta touches ONLY the new/changed chunks.

    The affected-set logic that already excludes an unchanged document at Source (ADR-0029/0043)
    is what bounds induction too — ``induce`` is a doc-level Stage, so a document that never runs
    is never induced. The assertion is on the LLM CALL COUNT (the cost that matters) and on the
    added chunk's own labels, not on any config flag: exactly one call, for the one new chunk,
    and the chunk keeps the schema IT induced rather than the base corpus's.

    Pre-change, the delta stamped the parent Version's frozen corpus schema onto the new chunk and
    made ZERO calls — cheap, but it re-imposed the corpus union on a chunk-level run, which is the
    broadcast the redesign removes.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Acme employs Alice.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Globex employs Bob.\n", encoding="utf-8")

    with fake.install([fake.reply(_SCHEMA_REPLY), fake.reply(_SCHEMA_REPLY)]) as script:
        DeltaRunner().apply_delta(_pipeline(store, docs, granularity="chunk"), run_id="v0")
        base_calls = len(script.calls)
    assert base_calls == 2, "the base run induces once per chunk (one chunk per document here)"
    # Per-chunk induction is not a corpus schema: nothing is frozen in the Corpus Version.
    cs = CorpusStore(Storage.for_uri(store), store)
    assert cs.head_schema() is None

    (docs / "c.txt").write_text("Initech supersedes the statute.\n", encoding="utf-8")
    with fake.install([fake.reply(_DELTA_REPLY)]) as script:
        report = DeltaRunner().apply_delta(_pipeline(store, docs, granularity="chunk"), "v1")
        delta_calls = len(script.calls)
    assert delta_calls == 1, "only the ADDED document's chunk is induced — never the whole corpus"
    assert report.delta is not None
    assert report.delta.documents_reused == 2
    assert report.delta.documents_extracted == 1
    # The added chunk carries ITS OWN induced labels, not the base corpus's vocabulary.
    fresh = _induced_labels_in_run(store, "v1")
    assert fresh, "the added document's chunk was induced"
    assert all(set(lbl.entity_types) == {"statute"} for lbl in fresh), (
        "a per-chunk delta must not stamp the base run's schema on the new chunk"
    )


def _mixed_granularity_pipeline(store: str, docs_dir: Path) -> Pipeline:
    """A pipeline holding BOTH induction models — the case that separates them behaviourally."""
    return Pipeline.from_yaml_text(
        f"""
name: mixed-granularity
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
  - name: induce_corpus
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [chunk]
    config: {{granularity: corpus, api_key: test-key}}
  - name: induce_chunk
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [induce_corpus]
    config: {{granularity: chunk, api_key: test-key}}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [induce_chunk]
"""
    )


def test_a_frozen_corpus_schema_is_not_stamped_on_a_per_chunk_induction_stage(
    tmp_path: Path,
) -> None:
    """T4: the freeze is scoped to the coupling it exists to break, not to the Stage KIND.

    The frozen schema is injected into every ``schema_induction`` Stage's config by capability
    alone, so in a pipeline holding both models the PER-CHUNK Stage was handed the parent Version's
    corpus union and stamped it verbatim — silently re-imposing the broadcast the redesign removes,
    on the one path that is supposed to be chunk-local. Scoped by granularity, the per-chunk Stage
    induces the new chunk normally and the chunk keeps its OWN labels.

    Pre-change this asserted ``0`` induction calls and ``{'company', 'person'}`` on the new chunk.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Acme employs Alice.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Globex employs Bob.\n", encoding="utf-8")
    with fake.install([fake.reply(_SCHEMA_REPLY)] * 4):
        DeltaRunner().apply_delta(_mixed_granularity_pipeline(store, docs), run_id="v0")
    assert CorpusStore(Storage.for_uri(store), store).head_schema() is not None

    (docs / "c.txt").write_text("Initech supersedes the statute.\n", encoding="utf-8")
    with fake.install([fake.reply(_DELTA_REPLY)]) as script:
        DeltaRunner().apply_delta(_mixed_granularity_pipeline(store, docs), run_id="v1")
        calls = len(script.calls)
    assert calls == 1, "the corpus Stage is frozen (0 calls); the per-chunk Stage still induces"
    fresh = _induced_labels_in_run(store, "v1", stage="induce_chunk")
    assert fresh
    assert all(set(lbl.entity_types) == {"statute"} for lbl in fresh), (
        "a per-chunk Stage must not be handed the frozen CORPUS schema"
    )


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

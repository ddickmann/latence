"""T4 — incremental deltas reconcile with per-chunk induction via a PERSISTED, EXTENDED vocabulary.

ADR-0043 originally kept a delta consistent with its base run by FREEZING the one induced corpus
schema. Per-chunk induction (T1) produces no such schema, so the cross-run artifact is now the
canonical TYPE vocabulary (ADR-0054): the Corpus Version persists it, and each delta EXTENDS it —
a drifted type that matches an existing cluster ALIASES into it, a genuinely new type APPENDS a
canonical entry, and no existing canonical label ever churns.

The corpus is built so the two models disagree loudly. The base corpus is dominated by
``organization``; the delta's documents are dominated by ``org``. Canonicalizing the delta from
scratch would therefore RE-ELECT ``org`` as the canonical label and silently retype every node the
committed KG already holds — the exact churn T4 forbids. Every test below asserts on the committed
vocabulary and on the assembled graph's node types, never on a config flag.

``test_reinduce_recanonicalizes_from_scratch`` is the deliberate control arm: with ``--reinduce``
the seed is dropped and the rename DOES happen, which is what "full rebuild" means — so the tests
prove seeding is what holds the type space stable, not some incidental property of the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.contracts import GraphRecord, GraphScope, TypeKind
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.state.corpus_store import CorpusStore
from latence_core.storage import Storage

# One real type under two surfaces, plus a type the base corpus never sees. The gazetteer is
# deterministic and model-free, so the "induced" drift is reproducible byte for byte.
_LABELS = """
      labels:
        organization:
          terms: [Acme Corporation, International Business Machines]
        org:
          terms: [Globex, Initech, Umbrella]
        statute:
          terms: [GDPR]
        person:
          terms: [Alice, Bob, Carol, Dave]
"""


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: delta-type-vocabulary
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
    config:{_LABELS}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      relations:
        works_for: {{head: person, tail: organization, triggers: [works at, joined]}}
  - name: consolidate_types
    capability: type_consolidation
    provider: type_consolidation.cascade
    depends_on: [entities, relations]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [consolidate_types]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config: {{corpus_id: t4}}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [graph]
"""
    )


def _base_corpus(tmp_path: Path) -> Path:
    """Two documents whose only organization surface is ``organization`` (the salience winner)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Alice works at Acme Corporation.\n", encoding="utf-8")
    (docs / "b.txt").write_text(
        "Bob joined International Business Machines this year.\n", encoding="utf-8"
    )
    return docs


def _add_delta_documents(docs: Path) -> None:
    """Three ``org`` mentions + one brand-new ``statute`` type — the delta's own vocabulary.

    ``org`` now OUTWEIGHS ``organization`` corpus-wide (3 vs 2), so a from-scratch canonical
    election over the union would rename the type.
    """
    (docs / "c.txt").write_text("Carol works at Globex. Initech filed too.\n", encoding="utf-8")
    (docs / "d.txt").write_text(
        "Umbrella disclosed the GDPR assessment.\n", encoding="utf-8"
    )


def _entity_node_types(store: str, run_id: str) -> set[str]:
    s = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/export/records.jsonl"
    rows = [json.loads(line) for line in s.read_text(uri).splitlines() if line.strip()]
    nodes = [GraphRecord.model_validate(r) for r in rows]
    return {
        n.node.label
        for n in nodes
        if n.scope is GraphScope.NODE and n.node is not None and n.node.label != "person"
    }


def _canonical_entity_types(store: str) -> dict[str, list[str]]:
    """``{canonical entity type: aliases}`` as committed in the head Corpus Version."""
    cs = CorpusStore(Storage.for_uri(store), store)
    vocabulary = cs.head_type_vocabulary()
    assert vocabulary is not None, "the Corpus Version must persist the canonical type vocabulary"
    return {t.canonical: list(t.aliases) for t in vocabulary.of_kind(TypeKind.ENTITY)}


def _report(store: str, run_id: str) -> QualityReport:
    s = Storage.for_uri(store)
    return QualityReport.model_validate_json(
        s.read_text(f"{store}/_latence/runs/{run_id}/quality-report.json")
    )


# -- the vocabulary survives the run and the delta EXTENDS it ----------------


def test_the_corpus_version_persists_the_canonical_type_vocabulary(tmp_path: Path) -> None:
    """Without persistence a delta has nothing to canonicalize against (the T4 precondition)."""
    docs = _base_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    assert _canonical_entity_types(store) == {"organization": [], "person": []}


def test_a_delta_aliases_new_types_into_the_committed_vocabulary_without_renaming_it(
    tmp_path: Path,
) -> None:
    """THE HEADLINE: ``org`` outvotes ``organization`` on the delta, and still does not rename it.

    Pre-change (no persisted vocabulary, no seed) the delta re-canonicalizes the union from
    scratch, ``org`` wins the salience vote 3-2, and every organization node in the committed KG
    is silently retyped. Seeding forbids that: ``org`` joins ``organization`` as an alias.
    """
    docs = _base_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    _add_delta_documents(docs)
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")

    canonical = _canonical_entity_types(store)
    assert "organization" in canonical, "an existing canonical type must not be renamed by a delta"
    assert "org" not in canonical, "the delta's dominant surface must ALIAS, not take over"
    assert "org" in canonical["organization"]
    # ...and the graph the delta assembled carries the stable name, not the re-elected one.
    assert _entity_node_types(store, "v1") == {"organization", "statute"}


def test_a_genuinely_new_type_appends_an_entry_and_lands_on_the_delta_churn(
    tmp_path: Path,
) -> None:
    """A type the base corpus never saw grows the vocabulary — visibly, on the Quality Report."""
    docs = _base_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    _add_delta_documents(docs)
    report = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")

    assert "statute" in _canonical_entity_types(store)
    churn = report.delta
    assert churn is not None
    assert churn.canonical_types_added == 1, "only ``statute`` is a genuinely new canonical type"
    assert churn.type_aliases_added >= 1, "``org`` aliased INTO the existing cluster"
    assert churn.canonical_types_total == len(_canonical_entity_types(store)) + 1  # +relation
    rendered = report.render()
    assert "canonical types: +1 new" in rendered


def test_applying_the_same_delta_twice_yields_the_same_canonical_vocabulary(
    tmp_path: Path,
) -> None:
    """Idempotency: re-applying an already-applied delta must not move the type space."""
    docs = _base_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    _add_delta_documents(docs)
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    cs = CorpusStore(Storage.for_uri(store), store)
    after_first = cs.head_type_vocabulary()

    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1-again")
    assert cs.head_type_vocabulary() == after_first


def test_a_second_delta_keeps_growing_the_vocabulary_monotonically(tmp_path: Path) -> None:
    """The type space only ever grows: nothing an earlier delta committed is dropped or renamed."""
    docs = _base_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    _add_delta_documents(docs)
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    first = _canonical_entity_types(store)

    (docs / "e.txt").write_text("Dave works at Umbrella.\n", encoding="utf-8")
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v2")
    second = _canonical_entity_types(store)
    assert set(first) <= set(second), "a canonical type must never disappear across deltas"
    for canonical, aliases in first.items():
        assert set(aliases) <= set(second[canonical]), "an alias must never be un-mapped"


def test_an_untouched_document_stays_unchanged_and_the_affected_set_stays_bounded(
    tmp_path: Path,
) -> None:
    """A Type-Consolidation Stage must not make every reused document look EDITED.

    It is the one corpus-level Stage that re-emits its input carrier, so its checkpoint spans the
    REUSED documents' mentions as well as the fresh ones. Feeding those into the change-detection
    manifest — which already carries the reused records — doubled every reused document's derived
    texts and flipped its content hash, so ``documents_unchanged`` was 0, every untouched document
    was classified UPDATED, and the affected-set recompute widened to the whole corpus: ADR-0018's
    "recompute only the affected set" bound, silently undone for the enterprise-SOTA stack.

    The witnesses are the churn's own classification and the affected-set/corpus ratio.
    """
    docs = _base_corpus(tmp_path)
    (docs / "x.txt").write_text("Alice works at Acme Corporation again.\n", encoding="utf-8")
    (docs / "y.txt").write_text(
        "Bob joined International Business Machines in March.\n", encoding="utf-8"
    )
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")

    (docs / "c.txt").write_text("Carol works at Globex.\n", encoding="utf-8")
    report = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1")
    churn = report.delta
    assert churn is not None
    assert churn.documents_added == 1
    assert churn.documents_updated == 0, "an untouched document must not be classified as edited"
    assert churn.documents_unchanged == 4
    assert not churn.reconciliation_triggered
    assert churn.affected_set_size < churn.corpus_record_count, (
        "an incremental delta must re-resolve less than the whole corpus"
    )


def test_reinduce_recanonicalizes_from_scratch(tmp_path: Path) -> None:
    """THE CONTROL ARM: ``--reinduce`` drops the seed, so the dominant surface DOES take over.

    This is what "re-canonicalize from scratch (a full rebuild)" means, and it is the proof that
    the stability above comes from seeding rather than from the corpus happening to agree.
    """
    docs = _base_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v0")
    _add_delta_documents(docs)
    report = DeltaRunner().apply_delta(_pipeline(store, docs), run_id="v1", reinduce=True)

    canonical = _canonical_entity_types(store)
    assert "org" in canonical, "a from-scratch rebuild re-elects the corpus's dominant surface"
    assert "organization" in canonical["org"]
    assert report.delta is not None
    assert report.delta.documents_reused == 0, "a full rebuild reuses nothing"

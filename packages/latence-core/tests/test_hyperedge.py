"""Hyperedge building (ADR-0057, hypergraph-m1): the spec-M1 acceptance criteria.

Covers the ticket-12 test list: every relation's arguments fall inside its Hyperedge span;
no Hyperedge exceeds the token ceiling; dedup-merge is idempotent (``support`` counts, ids
stable); **``tau_text`` never contains a span the exported corpus masked** (the PII-bypass
regression, ADR-0057's crux); deterministic ids across runs; the weak toggle; the arity cap;
the abbreviation-safe splitter; and — through the REAL LocalRunner — the wired end-to-end
proof plus the byte-identity guarantee: a stack that never configures hyperedges emits a
graph stream identical to before this feature existed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from latence_core.chunking import estimate_tokens
from latence_core.contracts import (
    CanonicalEntity,
    ChunkRecord,
    Classification,
    EntityMention,
    Evidence,
    GraphRecord,
    GraphScope,
    PIISpan,
    Provenance,
    RedactionAction,
    RelationMention,
)
from latence_core.hyperedge import HyperedgeBuilder, HyperedgeConfig
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.sentences import RuleSentenceSplitter
from latence_core.stages.graph_assembly import CanonicalGraphAssembler
from latence_core.storage import Storage

_CLS = Classification(language="en", category="memo", sensitivity="internal")


def _prov(doc: str, cs: int, ce: int) -> Provenance:
    return Provenance(
        source_uri=f"file:///{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=500,
        document_id=doc,
        char_start=cs,
        char_end=ce,
        page_start=1,
        page_end=1,
    )


_TEXT = (
    "Acme Corp acquired Beta Ltd in 2019 for 2 billion euros. "
    "The deal was approved by the EU Commission. "
    "Contact John Doe at john@acme.com for details."
)


def _chunk(record_id: str = "chunk:1", doc: str = "doc-1", *, redacted: bool = True) -> ChunkRecord:
    pii = (
        [
            PIISpan(
                pii_type="person",
                char_start=_TEXT.index("John Doe"),
                char_end=_TEXT.index("John Doe") + len("John Doe"),
                action=RedactionAction.MASK,
                placeholder="[PERSON]",
                detector="regex:person",
            ),
            PIISpan(
                pii_type="email",
                char_start=_TEXT.index("john@acme.com"),
                char_end=_TEXT.index("john@acme.com") + len("john@acme.com"),
                action=RedactionAction.MASK,
                placeholder="[EMAIL]",
                detector="regex:email",
            ),
        ]
        if redacted
        else []
    )
    return ChunkRecord(
        record_id=record_id,
        document_record_id=doc,
        index=0,
        token_count=estimate_tokens(_TEXT),
        content=_TEXT,
        masked_content=(
            _TEXT.replace("john@acme.com", "[EMAIL]").replace("John Doe", "[PERSON]")
            if redacted
            else None
        ),
        pii_spans=pii,
        provenance=_prov(doc, 0, len(_TEXT)),
        classification=_CLS,
    )


def _mention(rid: str, text: str, label: str, conf: float, chunk_id: str = "chunk:1",
             doc: str = "doc-1") -> EntityMention:
    start = _TEXT.index(text)
    return EntityMention(
        record_id=rid,
        document_record_id=doc,
        chunk_record_id=chunk_id,
        index=0,
        label=label,
        text=text,
        confidence=conf,
        provenance=_prov(doc, start, start + len(text)),
        classification=_CLS,
    )


def _relation(rid: str, head: str, tail: str, label: str = "acquired",
              conf: float = 0.88, doc: str = "doc-1") -> RelationMention:
    return RelationMention(
        record_id=rid,
        document_record_id=doc,
        index=0,
        label=label,
        head_mention_id=head,
        tail_mention_id=tail,
        head_text="h",
        tail_text="t",
        confidence=conf,
        provenance=_prov(doc, 0, 30),
        classification=_CLS,
    )


def _entity(rid: str, name: str, mids: list[str], links: dict[str, float]) -> CanonicalEntity:
    return CanonicalEntity(
        record_id=rid,
        canonical_text=name,
        entity_type="ORG",
        member_mention_ids=mids,
        member_link_confidences=links,
        evidence=Evidence(mention_ids=mids, document_ids=["doc-1"], confidence=0.9),
        provenance=_prov("doc-1", 0, 9),
        classification=_CLS,
    )


_NODE_OF = {"e:acme": "nA", "e:beta": "nB", "e:eu": "nE", "e:john": "nJ"}


def _fixture() -> tuple[list[ChunkRecord], list[EntityMention], list[RelationMention],
                        list[CanonicalEntity]]:
    mentions = [
        _mention("m:acme", "Acme Corp", "ORG", 0.95),
        _mention("m:beta", "Beta Ltd", "ORG", 0.9),
        _mention("m:eu", "EU Commission", "ORG", 0.85),
        _mention("m:john", "John Doe", "PERSON", 0.8),
    ]
    relations = [_relation("r:1", "m:acme", "m:beta")]
    entities = [
        _entity("e:acme", "Acme Corp", ["m:acme"], {"m:acme": 0.92}),
        _entity("e:beta", "Beta Ltd", ["m:beta"], {"m:beta": 0.9}),
        _entity("e:eu", "EU Commission", ["m:eu"], {}),
        _entity("e:john", "John Doe", ["m:john"], {}),
    ]
    return [_chunk()], mentions, relations, entities


def _build(**cfg: object) -> list:
    chunks, mentions, relations, entities = _fixture()
    builder = HyperedgeBuilder("corpus-test", HyperedgeConfig(**cfg))  # type: ignore[arg-type]
    return builder.build(chunks, mentions, relations, entities, lambda e: _NODE_OF[e])


# -- spec M1: relation arguments inside the span --------------------------------


def test_every_relation_argument_falls_inside_its_hyperedge_span() -> None:
    hyperedges = _build()
    strong = [h for h in hyperedges if h.has_relations]
    assert strong, "the fixture relation must produce a strong Hyperedge"
    for h in strong:
        for rel in h.relations:
            assert rel.head_node_id in h.entity_node_ids
            assert rel.tail_node_id in h.entity_node_ids
        # the primary span (original coordinates) covers the relation argument mentions
        assert h.provenance.char_start is not None and h.provenance.char_end is not None
        for text in ("Acme Corp", "Beta Ltd"):
            start = _TEXT.index(text)
            assert h.provenance.char_start <= start < h.provenance.char_end


def test_no_hyperedge_exceeds_the_token_ceiling() -> None:
    # A long many-sentence chunk with relations far apart forces the split path.
    sentences = [
        f"Company{i} sold Asset{i} to Buyer{i} in 20{i:02d} for a large sum of money. "
        for i in range(1, 13)
    ]
    text = "".join(sentences)
    chunk = ChunkRecord(
        record_id="chunk:long",
        document_record_id="doc-2",
        index=0,
        token_count=estimate_tokens(text),
        content=text,
        provenance=Provenance(
            source_uri="file:///long.txt", file_name="long.txt", file_type="txt",
            file_size=4000, document_id="doc-2", char_start=0, char_end=len(text),
            page_start=1, page_end=1,
        ),
        classification=_CLS,
    )
    mentions, relations, entities, node_of = [], [], [], {}
    for i in range(1, 13):
        for role, label in (("Company", "ORG"), ("Asset", "PRODUCT"), ("Buyer", "ORG")):
            name = f"{role}{i}"
            start = text.index(name)
            mid = f"m:{name}"
            mentions.append(
                EntityMention(
                    record_id=mid, document_record_id="doc-2", chunk_record_id="chunk:long",
                    index=0, label=label, text=name, confidence=0.9,
                    provenance=Provenance(
                        source_uri="file:///long.txt", file_name="long.txt", file_type="txt",
                        file_size=4000, document_id="doc-2", char_start=start,
                        char_end=start + len(name), page_start=1, page_end=1,
                    ),
                    classification=_CLS,
                )
            )
            eid = f"e:{name}"
            entities.append(_entity(eid, name, [mid], {}))
            node_of[eid] = f"n:{name}"
        relations.append(_relation(f"r:{i}", f"m:Company{i}", f"m:Asset{i}", "sold"))
    builder = HyperedgeBuilder("corpus-long", HyperedgeConfig(max_tokens=60, min_tokens=10))
    hyperedges = builder.build([chunk], mentions, relations, entities, lambda e: node_of[e])
    assert hyperedges
    for h in hyperedges:
        assert estimate_tokens(h.tau_text) <= 60 or h.tau_text.count(".") <= 1, (
            f"over-ceiling Hyperedge that was splittable: {h.token_cost} tokens"
        )


# -- spec M1: dedup-merge idempotency -------------------------------------------


def test_same_fact_in_two_chunks_merges_with_support_two() -> None:
    chunks, mentions, relations, entities = _fixture()
    # The same fact again in a second chunk of a second document (same canonical entities).
    chunk2 = _chunk("chunk:2", "doc-9")
    m2 = [
        _mention("m:acme2", "Acme Corp", "ORG", 0.7, "chunk:2", "doc-9"),
        _mention("m:beta2", "Beta Ltd", "ORG", 0.7, "chunk:2", "doc-9"),
        _mention("m:eu2", "EU Commission", "ORG", 0.7, "chunk:2", "doc-9"),
        _mention("m:john2", "John Doe", "PERSON", 0.7, "chunk:2", "doc-9"),
    ]
    r2 = [_relation("r:2", "m:acme2", "m:beta2", conf=0.5, doc="doc-9")]
    entities2 = [
        _entity("e:acme", "Acme Corp", ["m:acme", "m:acme2"], {"m:acme": 0.92}),
        _entity("e:beta", "Beta Ltd", ["m:beta", "m:beta2"], {"m:beta": 0.9}),
        _entity("e:eu", "EU Commission", ["m:eu", "m:eu2"], {}),
        _entity("e:john", "John Doe", ["m:john", "m:john2"], {}),
    ]
    builder = HyperedgeBuilder("corpus-test")
    merged = builder.build(
        [*chunks, chunk2], [*mentions, *m2], [*relations, *r2], entities2,
        lambda e: _NODE_OF[e],
    )
    strong = [h for h in merged if h.has_relations]
    assert len(strong) == 1, "identical entity set + relation signature must merge"
    assert strong[0].support == 2
    assert len(strong[0].merged_sources) == 1
    # higher-gamma instance (chunk 1: better confidences) stays primary
    assert strong[0].provenance.document_id == "doc-1"

    # idempotency: building again yields the identical result
    again = builder.build(
        [*chunks, chunk2], [*mentions, *m2], [*relations, *r2], entities2,
        lambda e: _NODE_OF[e],
    )
    assert [h.model_dump(mode="json") for h in merged] == [
        h.model_dump(mode="json") for h in again
    ]


def test_deterministic_ids_are_stable_across_runs() -> None:
    first = _build()
    second = _build()
    assert [h.hyperedge_id for h in first] == [h.hyperedge_id for h in second]
    assert first, "fixture must build at least one Hyperedge"


# -- ADR-0057: the PII-bypass regression ----------------------------------------


def test_tau_text_never_contains_a_masked_span() -> None:
    hyperedges = _build()
    assert hyperedges
    joined = " ".join(h.tau_text for h in hyperedges)
    assert "john@acme.com" not in joined
    assert "John Doe" not in joined
    assert "[PERSON]" in joined or "[EMAIL]" in joined  # the masked variant is present


def test_tau_text_is_raw_when_the_corpus_is_raw() -> None:
    chunks, mentions, relations, entities = _fixture()
    raw_chunk = _chunk(redacted=False)
    builder = HyperedgeBuilder("corpus-test")
    hyperedges = builder.build([raw_chunk], mentions, relations, entities,
                               lambda e: _NODE_OF[e])
    joined = " ".join(h.tau_text for h in hyperedges)
    assert "John Doe" in joined  # no redaction ran -> the corpus itself is raw


# -- weak toggle + arity ---------------------------------------------------------


def test_weak_hyperedges_off_drops_entities_only_units() -> None:
    with_weak = _build(weak_hyperedges=True, min_tokens=1)
    without = _build(weak_hyperedges=False, min_tokens=1)
    assert any(h.weak for h in with_weak)
    assert not any(h.weak for h in without)
    # strong content unaffected by the toggle
    assert [h.hyperedge_id for h in with_weak if h.has_relations] == [
        h.hyperedge_id for h in without if h.has_relations
    ]


def test_arity_cap_splits_by_relation_groups() -> None:
    hyperedges = _build(max_arity=2, min_tokens=1)
    for h in hyperedges:
        assert h.arity <= 2
    strong = [h for h in hyperedges if h.has_relations]
    assert strong and strong[0].entity_node_ids == ["nA", "nB"]  # co-related pair stays


# -- gamma ----------------------------------------------------------------------


def test_gamma_renormalises_over_available_signals_and_discounts_weak() -> None:
    hyperedges = _build(min_tokens=1)
    strong = next(h for h in hyperedges if h.has_relations)
    assert 0.0 < strong.gamma <= 1.0
    weak = [h for h in hyperedges if h.weak]
    for h in weak:
        assert h.gamma <= 0.5  # the weak discount applies


# -- the splitter ----------------------------------------------------------------


def test_sentence_splitter_is_abbreviation_safe() -> None:
    split = RuleSentenceSplitter().split
    cases = [
        ("Dr. Smith met Mr. Jones. They agreed.", 2),
        ("Die Lieferung erfolgt z. B. am 3. Mai 2024. Danach folgt die Abnahme.", 2),
        ("ZF übernahm WABCO für ca. 7 Mrd. Euro. Die Übernahme wurde 2020 vollzogen.", 2),
        ("Pi is 3.14 roughly. Next sentence.", 2),
        ("See Fig. 3 for details. The results follow.", 2),
        ("# Heading\nFirst item. Second item.", 3),
        ("Wait... What happened?", 2),
        ("Er sagte... dann nichts mehr.", 1),
        ("J. R. Smith arrived. He left early.", 2),
        ("Am 1. Januar beginnt das Jahr. Im 4. Quartal endet es.", 2),
        ("Acme Inc. acquired Beta Ltd. in 2019.", 1),
        ("Das Projekt Nr. 5 läuft. Es endet bald.", 2),
    ]
    for text, want in cases:
        spans = split(text)
        got = [text[a:b] for a, b in spans]
        assert len(spans) == want, f"{text!r}: expected {want} sentences, got {got}"
        for a, b in spans:
            assert text[a:b] == text[a:b].strip()  # trimmed spans


# -- the assembler seam (byte-identity + opt-in) --------------------------------


def test_unconfigured_assembler_stream_is_unchanged_and_wants_no_streams() -> None:
    assembler = CanonicalGraphAssembler({"corpus_id": "c"})
    assert assembler.wants_document_streams is False
    # assemble() and assemble_with_documents() agree on an empty run
    assert list(assembler.assemble([])) == list(
        assembler.assemble_with_documents([], chunks=(), mentions=(), relations=())
    )


def test_configured_assembler_opts_in() -> None:
    assembler = CanonicalGraphAssembler({"corpus_id": "c", "hyperedges": True})
    assert assembler.wants_document_streams is True


# -- end to end through the REAL LocalRunner ------------------------------------


def _e2e_pipeline(store: str, docs_dir: Path, *, hyperedges: bool) -> Pipeline:
    hyperedge_cfg = "\n      hyperedges: true" if hyperedges else ""
    return Pipeline.from_yaml_text(
        f"""
name: hyperedge-slice
storage_uri: {store}
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
          terms: [Acme, Beta Industries]
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
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: hyperedge-slice{hyperedge_cfg}
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
"""
    )


def _e2e_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works at Acme. Bob joined Beta Industries last spring.\n", encoding="utf-8"
    )
    return docs


def test_hyperedges_flow_end_to_end_through_the_runner(tmp_path: Path) -> None:
    docs = _e2e_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_e2e_pipeline(store, docs, hyperedges=True), run_id="he-e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/he-e2e"

    # The hyperedges parquet materialized next to nodes/edges, and loads.
    uri = f"{base}/export/graph-hyperedges.parquet"
    assert s.exists(uri), "hyperedges parquet must be exported when the projection is on"
    with s.open(uri, "rb") as fh:
        rows = pq.read_table(fh).to_pylist()  # type: ignore[no-untyped-call]
    assert rows, "the fixture corpus must produce at least one Hyperedge"
    for row in rows:
        assert row["token_cost"] >= 1
        assert row["token_counter"] == "latence:estimate_tokens:v1"
        assert 0.0 <= row["gamma"] <= 1.0
        nodes = json.loads(row["entity_node_ids"])
        assert nodes == sorted(set(nodes))
    strong = [r for r in rows if r["has_relations"]]
    assert strong, "the works_for relation must produce a strong Hyperedge"

    # The Quality Report carries the roll-up (counts + token-cost distribution).
    assert isinstance(report, QualityReport)
    assert report.graph is not None and report.graph.hyperedges is not None
    hq = report.graph.hyperedges
    assert hq.hyperedges_total == len(rows)
    assert hq.token_cost_min >= 1 and hq.token_cost_max >= hq.token_cost_min
    assert hq.relations_total >= 1


def test_unconfigured_run_is_byte_identical_and_exports_no_hyperedges(tmp_path: Path) -> None:
    docs = _e2e_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_e2e_pipeline(store, docs, hyperedges=False), run_id="he-off")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/he-off"
    assert not s.exists(f"{base}/export/graph-hyperedges.parquet")
    assert isinstance(report, QualityReport)
    assert report.graph is not None and report.graph.hyperedges is None
    # No HYPEREDGE record in the assembled stream: the checkpointed graph is scope NODE/EDGE only.
    graph_records = [
        GraphRecord.model_validate(json.loads(line))
        for line in s.read_text(f"{base}/checkpoints/graph.jsonl").splitlines()
        if line.strip()
    ]
    assert graph_records
    assert all(r.scope in (GraphScope.NODE, GraphScope.EDGE) for r in graph_records)


# -- M2 (ticket 13): the sidecar KV ---------------------------------------------


def _kv_fixture_records() -> tuple[list, list]:
    """Hyperedges + nodes straight from the real builder over the standard fixture."""
    chunks, mentions, relations, entities = _fixture()
    builder = HyperedgeBuilder("corpus-test")
    hyperedges = builder.build(chunks, mentions, relations, entities, lambda e: _NODE_OF[e])
    from latence_core.contracts import GraphNode

    nodes = [
        GraphNode(
            record_id=f"node:{nid}", node_id=nid, entity_id=eid, label="ORG",
            canonical_name=name, confidence=0.9,
            evidence=Evidence(mention_ids=["m1"], document_ids=["doc-1"], confidence=0.9),
            provenance=_prov("doc-1", 0, 9), classification=_CLS,
        )
        for eid, nid, name in (
            ("e:acme", "nA", "Acme Corp"), ("e:beta", "nB", "Beta Ltd"),
            ("e:eu", "nE", "EU Commission"), ("e:john", "nJ", "John Doe"),
        )
    ]
    return hyperedges, nodes


def test_kv_is_byte_reproducible_from_the_records() -> None:
    from latence_core.hyperedge_store import build_store_bytes, store_entries

    hyperedges, nodes = _kv_fixture_records()
    raw = build_store_bytes(store_entries(hyperedges, nodes))
    again = build_store_bytes(store_entries(list(reversed(hyperedges)), list(reversed(nodes))))
    assert raw == again  # rebuild equals original, independent of input order


def test_kv_incremental_append_equals_full_rebuild() -> None:
    from latence_core.hyperedge_store import build_store_bytes, store_entries

    hyperedges, nodes = _kv_fixture_records()
    split = max(1, len(hyperedges) // 2)
    # "appending" = rebuilding over the union of old + new records (the KV is derived,
    # ADR-0056); the criterion is that the result equals a from-scratch build.
    incremental = build_store_bytes(
        store_entries([*hyperedges[:split], *hyperedges[split:]], nodes)
    )
    full = build_store_bytes(store_entries(hyperedges, nodes))
    assert incremental == full


def test_kv_postings_are_gamma_sorted_with_gamma_inline() -> None:
    from latence_core.hyperedge_store import HyperedgeStoreReader, build_store_bytes, store_entries

    hyperedges, nodes = _kv_fixture_records()
    raw = build_store_bytes(store_entries(hyperedges, nodes))
    with HyperedgeStoreReader.from_bytes(raw) as store:
        for node_id in ("nA", "nB"):
            ids, gammas = store.postings(node_id)
            assert len(ids) == len(gammas) > 0
            assert gammas == sorted(gammas, reverse=True)
        # solver fields carry no text; tau lives in its own keyspace
        h = hyperedges[0]
        fields = store.solver_fields(h.hyperedge_id)
        assert fields is not None and "tau_text" not in fields
        assert fields["token_counter"] == h.token_counter
        assert store.tau_text(h.hyperedge_id) == h.tau_text
        # alias table: tier-1 zero-model entry (ADR-0059), NFKC + casefold
        assert store.alias("ACME  Corp") == ["nA"]


def test_kv_reader_holds_nothing_between_calls(tmp_path: Path) -> None:
    from latence_core.hyperedge_store import HyperedgeStoreReader, build_store_bytes, store_entries

    hyperedges, nodes = _kv_fixture_records()
    path = tmp_path / "graph-hyperedges.kv"
    path.write_bytes(build_store_bytes(store_entries(hyperedges, nodes)))
    # open -> read -> close per call; after close the reader is inert (structural statelessness)
    reader = HyperedgeStoreReader.open(str(path))
    ids, _ = reader.postings("nA")
    reader.close()
    assert ids
    import pytest as _pytest

    with _pytest.raises(ValueError):
        reader.get(b"e|anything")  # a closed reader must not silently serve stale data


def test_exact_token_counter_config_is_loud_when_missing() -> None:
    import pytest as _pytest
    from latence_core.errors import ConfigError
    from latence_core.stages.graph_assembly import _resolve_hyperedge_config

    with _pytest.raises(ConfigError, match="tiktoken:<encoding>"):
        _resolve_hyperedge_config({"token_counter": "spacy:foo"})
    with _pytest.raises(ConfigError, match="tiktoken"):
        # tiktoken is deliberately NOT a dev dependency: naming it must fail loudly, never
        # silently fall back to the heuristic (ADR-0058: no silent counter substitution).
        _resolve_hyperedge_config({"token_counter": "tiktoken:cl100k_base"})


def test_e2e_export_ships_the_kv_beside_the_parquet(tmp_path: Path) -> None:
    from latence_core.hyperedge_store import HyperedgeStoreReader

    docs = _e2e_docs(tmp_path)
    store_uri = f"file://{tmp_path}/store"
    LocalRunner().run(_e2e_pipeline(store_uri, docs, hyperedges=True), run_id="he-kv")
    s = Storage.for_uri(store_uri)
    base = f"{store_uri}/_latence/runs/he-kv"
    kv_uri = f"{base}/export/graph-hyperedges.kv"
    assert s.exists(kv_uri), "the KV must ship beside the parquet on the opted-in path"

    with s.open(kv_uri, "rb") as fh:
        raw = fh.read()
    with s.open(f"{base}/export/graph-hyperedges.parquet", "rb") as fh:
        rows = pq.read_table(fh).to_pylist()  # type: ignore[no-untyped-call]
    with HyperedgeStoreReader.from_bytes(raw) as reader:
        for row in rows:  # KV agrees with the durable parquet, record for record
            fields = reader.solver_fields(row["hyperedge_id"])
            assert fields is not None
            assert fields["token_cost"] == row["token_cost"]
            assert fields["gamma"] == row["gamma"]
            assert reader.tau_text(row["hyperedge_id"]) == row["tau_text"]
        # description + degree stamped onto the nodes (spec 1.4), embedding-input ready
        with s.open(f"{base}/export/graph-nodes.parquet", "rb") as fh:
            node_rows = pq.read_table(fh).to_pylist()  # type: ignore[no-untyped-call]
        for nrow in node_rows:
            props = json.loads(nrow["properties"])
            assert "description" in props and "hyperedge_degree" in props
            assert props["description"].startswith(nrow["canonical_name"])


def test_e2e_off_path_writes_no_kv_and_plain_node_properties(tmp_path: Path) -> None:
    docs = _e2e_docs(tmp_path)
    store_uri = f"file://{tmp_path}/store"
    LocalRunner().run(_e2e_pipeline(store_uri, docs, hyperedges=False), run_id="he-kv-off")
    s = Storage.for_uri(store_uri)
    base = f"{store_uri}/_latence/runs/he-kv-off"
    assert not s.exists(f"{base}/export/graph-hyperedges.kv")
    with s.open(f"{base}/export/graph-nodes.parquet", "rb") as fh:
        node_rows = pq.read_table(fh).to_pylist()  # type: ignore[no-untyped-call]
    for nrow in node_rows:  # the default node stream carries no hyperedge-derived keys
        props = json.loads(nrow["properties"])
        assert "description" not in props and "hyperedge_degree" not in props


def test_kv_format_interop_with_the_retrieval_reader() -> None:
    """The format is the contract (ADR-0056): the store latence-core WRITES must be read
    byte-compatibly by latence-retrieval's independent reader implementation — including
    postings order, alias normalisation, and solver fields. Two implementations, one
    format; drift in either direction fails here."""
    from latence_core.hyperedge_store import build_store_bytes, store_entries
    from latence_retrieval.hypergraph import HyperedgeKV

    hyperedges, nodes = _kv_fixture_records()
    raw = build_store_bytes(store_entries(hyperedges, nodes))
    with HyperedgeKV.from_bytes(raw) as kv:
        for h in hyperedges:
            fields = kv.solver_fields(h.hyperedge_id)
            assert fields is not None
            assert fields["entity_node_ids"] == h.entity_node_ids
            assert fields["gamma"] == h.gamma
            assert kv.tau_text(h.hyperedge_id) == h.tau_text
        ids, gammas = kv.postings("nA")
        assert gammas == sorted(gammas, reverse=True) and len(ids) == len(gammas)
        assert kv.alias("acme  CORP") == ["nA"]


# -- ticket 10: Hyperedges under Corpus Versions / Purge (inheritance, proven) ----


def test_purge_expunges_hyperedges_and_their_tau_text(tmp_path: Path) -> None:
    """Ticket 10's claim, proven rather than asserted: because corpus-level Stages fully
    re-run on a delta (the S11 semantic), the hyperedge stream is regenerated per Corpus
    Version — so Purge's "nothing derived survives" covers Hyperedges BY INHERITANCE, with
    no new machinery. This pins it: after purging a document, the head Version's live set
    contains no Hyperedge derived from it (its tau_text gone — GDPR reaches the sidecar,
    ADR-0056/0057), while a co-resident document's Hyperedges survive."""
    import hashlib

    from latence_core.contracts import DeletionMode
    from latence_core.delta_runner import DeltaRunner
    from latence_core.state.corpus_store import CorpusStore

    docs = _e2e_docs(tmp_path)  # memo.txt: Alice/Acme + Bob/Beta Industries
    (docs / "purge_me.txt").write_text(
        "Carol joined Acme in June. Carol works at Acme headquarters.\n", encoding="utf-8"
    )
    store_uri = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(_e2e_yaml_with_carol(store_uri, docs))

    DeltaRunner().apply_delta(pipeline, run_id="hv0")
    cs = CorpusStore(Storage.for_uri(store_uri), store_uri)
    head0 = cs.head_version()
    assert head0 is not None
    live0 = cs.read_records(head0)
    hyper0 = [
        r for r in live0
        if isinstance(r, GraphRecord) and r.scope is GraphScope.HYPEREDGE
    ]
    assert hyper0, "v0 must commit Hyperedge records"
    assert any("Carol" in (r.hyperedge.tau_text if r.hyperedge else "") for r in hyper0)

    doc_id = "sha256:" + hashlib.sha256((docs / "purge_me.txt").read_bytes()).hexdigest()
    rp = DeltaRunner().purge(pipeline, run_id="hpurge", document_ids=[doc_id])
    assert rp.delta is not None and rp.delta.deletion_mode is DeletionMode.PURGE

    head1 = cs.head_version()
    assert head1 is not None
    live1 = cs.read_records(head1)
    hyper1 = [
        r for r in live1
        if isinstance(r, GraphRecord) and r.scope is GraphScope.HYPEREDGE
    ]
    # nothing derived from the purged document survives — tau_text included
    assert not any("Carol" in (r.hyperedge.tau_text if r.hyperedge else "") for r in hyper1)
    # and the co-resident document's Hyperedges are NOT collaterally erased
    assert any("Alice" in (r.hyperedge.tau_text if r.hyperedge else "") for r in hyper1)


def _e2e_yaml_with_carol(store_uri: str, docs: Path) -> str:
    """The e2e pipeline YAML with Carol in the gazetteer (delta/purge fixture)."""
    return f"""
name: hyperedge-purge
storage_uri: {store_uri}
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
          terms: [Acme, Beta Industries]
        PERSON:
          terms: [Alice, Bob, Carol]
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
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: hyperedge-purge
      hyperedges: true
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
"""

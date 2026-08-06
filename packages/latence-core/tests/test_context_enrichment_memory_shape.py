# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The memory SHAPE of :meth:`ContextHeaderEnricher.enrich`, pinned structurally (no RSS).

Why this file exists: the S3 campaign's UDA dataset (220 129 chunks / 2 603 887 EntityMentions /
516 512 canonical entities) was OOM-killed in **Context Enrichment** inside a container with a hard
29 GB cgroup cap (``memory.limit_in_bytes`` 30999998464; the cgroup ``oom_kill`` counter went
3 → 4). Same class as the Hyperedge OOM one Stage earlier (``test_hyperedge_memory_shape``): the
cause was shape, not scale. The Stage grouped the run's mentions per chunk as **pydantic records**
— a measured ~3.35 kB per :class:`~latence_core.contracts.EntityMention`, each dragging its own
``Provenance`` + ``Classification`` submodel, so 8.1 GiB of mentions before a single header was
built — even though the header build reads three fields off a mention and keeps only a
per-(chunk, entity) MAXIMUM of one of them.

The properties that fix pins, asserted structurally so they cannot flake:

1. **No input record is retained.** ``EntityMention`` / ``GraphRecord`` / ``GraphNode`` /
   ``GraphEdge`` are projected (or folded) on arrival and dropped; by the time the first chunk is
   emitted, ZERO of them are alive. This is the 8.1 GiB term.
2. **Mentions are folded, not stored.** The retained per-chunk state is one entry per
   (chunk, canonical entity), not one per mention — a float in a dict rather than a record, and
   bounded by a count the corpus's 516 512 distinct entities cap rather than by its 2.6 M mentions.
3. **Nothing accumulates across the chunk loop.** Every live count at the LAST chunk of a
   many-chunk run equals its value at the FIRST — headers are yielded, never collected, and the
   mention->node join (the one mention-sized table) is dropped before the loop begins.

Everything is measured by counting live objects (``gc.get_objects()`` filtered by exact type and by
this test's own id prefixes), sampled at two known points: the first chunk and the last. Counting
objects is deterministic under CPython refcounting; a peak-RSS threshold would not be, which is why
none is used. ``enrich`` is a generator, so the sampling needs no seam: the test simply looks
between yields, while the enricher's frame is suspended.

``_SLACK`` absorbs the references the interpreter legitimately holds across a loop iteration (the
previous iteration's locals are not rebound until the next statement executes). It is far smaller
than the accumulation this test would catch: at 40 chunks the "accumulating" reading is 1000 live
mentions, the bounded reading is 0.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    ContextHeader,
    EntityMention,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from latence_core.stages.context_enrichment import (
    ContextHeaderEnricher,
    _ChunkEntities,
    _EdgeMeta,
    _Neighbor,
    _NodeView,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_CLS = Classification(language="en", sensitivity="internal")
_CHUNKS = 40
_ENTS = 3  # canonical entities local to each chunk
_MENTIONS_PER_ENT = 8  # mentions of each of them, in that chunk
#: Per chunk: every local entity's mentions, plus one mention of the corpus-wide hub entity.
_MENTIONS_PER_CHUNK = _ENTS * _MENTIONS_PER_ENT + 1
#: Per chunk, the number of DISTINCT canonical entities those mentions fold down to.
_ENTITIES_PER_CHUNK = _ENTS + 1
_TOTAL_MENTIONS = _CHUNKS * _MENTIONS_PER_CHUNK
_NODES = _CHUNKS * _ENTS + 2  # + the two hubs
_EDGES = _CHUNKS * 4
_SLACK = 4  # loop-carried references; see the module docstring

_HUB_A = "ctxnode:hubA"
_HUB_B = "ctxnode:hubB"
#: A key that exists ONLY in the mention->node join — the probe for "that table was dropped".
_JOIN_SENTINEL = "ctxmention:0000:0:0"


def _prov(doc: str, cs: int, ce: int) -> Provenance:
    return Provenance(
        source_uri=f"file:///{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=4096,
        document_id=doc,
        char_start=cs,
        char_end=ce,
        page_start=1,
        page_end=1,
    )


def _chunk_id(ci: int) -> str:
    return f"ctxchunk:{ci:04d}"


def _node_id(ci: int, ei: int) -> str:
    return f"ctxnode:{ci:04d}:{ei}"


def _mention_id(ci: int, ei: int, k: int) -> str:
    return f"ctxmention:{ci:04d}:{ei}:{k}"


def _hub_mention_id(ci: int) -> str:
    return f"ctxmention:{ci:04d}:hub"


def _chunks() -> Iterator[ChunkRecord]:
    for ci in range(_CHUNKS):
        doc = f"ctxdoc:{ci // 4:03d}"
        content = f"Chunk {ci} discusses several entities and their relations at length. "
        yield ChunkRecord(
            record_id=_chunk_id(ci),
            document_record_id=doc,
            index=ci % 4,
            content=content,
            token_count=len(content) // 4,
            provenance=_prov(doc, 1000 * ci, 1000 * ci + len(content)),
            classification=_CLS,
        )


def _mentions() -> Iterator[EntityMention]:
    """Every mention, built lazily and never held by the test — so "zero alive" is meaningful."""
    for ci in range(_CHUNKS):
        doc = f"ctxdoc:{ci // 4:03d}"
        index = 0
        for ei in range(_ENTS):
            for k in range(_MENTIONS_PER_ENT):
                yield EntityMention(
                    record_id=_mention_id(ci, ei, k),
                    document_record_id=doc,
                    chunk_record_id=_chunk_id(ci),
                    index=index,
                    label="ORG",
                    text=f"Entity {ci}-{ei}",
                    confidence=round(0.30 + 0.06 * ((ci + ei + k) % 11), 3),
                    provenance=_prov(doc, 1000 * ci + 10 * index, 1000 * ci + 10 * index + 6),
                    classification=_CLS,
                )
                index += 1
        yield EntityMention(
            record_id=_hub_mention_id(ci),
            document_record_id=doc,
            chunk_record_id=_chunk_id(ci),
            index=index,
            label="ORG",
            text="Hub A",
            confidence=round(0.40 + 0.05 * (ci % 9), 3),
            provenance=_prov(doc, 1000 * ci + 900, 1000 * ci + 905),
            classification=_CLS,
        )


def _node(
    node_id: str, name: str, members: list[str], confidence: float, centrality: float
) -> GraphNode:
    doc = "ctxdoc:000"
    return GraphNode(
        record_id=f"ctxentity:{node_id}",
        node_id=node_id,
        entity_id=f"ctxentity:{node_id}",
        label="ORG",
        canonical_name=name,
        member_mention_ids=members,
        source_document_ids=[doc],
        confidence=confidence,
        # ADR-0051 graph_features stamps, so _NodeView's centrality/community path is exercised.
        # A seed-independent spread (``hash`` would make the fixture PYTHONHASHSEED-dependent).
        properties={"centrality": centrality, "community": f"c{len(node_id) % 3}"},
        evidence=Evidence(mention_ids=members[:1], confidence=0.9),
        provenance=_prov(doc, 0, 10),
        classification=_CLS,
    )


def _graph() -> Iterator[GraphRecord]:
    """Nodes then edges, exactly as the assembler emits them — built lazily, never held."""
    doc = "ctxdoc:000"
    for ci in range(_CHUNKS):
        for ei in range(_ENTS):
            node = _node(
                _node_id(ci, ei),
                f"Entity {ci}-{ei}",
                [_mention_id(ci, ei, k) for k in range(_MENTIONS_PER_ENT)],
                round(0.35 + 0.05 * ((ci + ei) % 12), 3),
                round(0.01 * ((ci * 3 + ei) % 97), 4),
            )
            yield GraphRecord(
                record_id=f"ctxgraph:node:{node.node_id}",
                scope=GraphScope.NODE,
                node=node,
                provenance=_prov(doc, 0, 10),
                classification=_CLS,
            )
    # Hub A is a member of every chunk (so its adjacency blows past the per-entity cap and the
    # hub-entity guard bites); Hub B has no mentions at all, so it only ever appears as a
    # neighbor endpoint.
    for hub, name, members in (
        (_HUB_A, "Hub A", [_hub_mention_id(ci) for ci in range(_CHUNKS)]),
        (_HUB_B, "Hub B", []),
    ):
        node = _node(hub, name, members, 0.88, 0.5)
        yield GraphRecord(
            record_id=f"ctxgraph:node:{hub}",
            scope=GraphScope.NODE,
            node=node,
            provenance=_prov(doc, 0, 10),
            classification=_CLS,
        )
    for ci in range(_CHUNKS):
        pairs = (
            ("a", _node_id(ci, 0), _node_id(ci, 1), "works_for"),
            ("b", _node_id(ci, 1), _node_id(ci, 2), "located_in"),
            ("c", _node_id(ci, 0), _HUB_A, "member_of"),
            ("d", _node_id(ci, 2), _HUB_B, "reports_to"),
        )
        for slot, src, tgt, label in pairs:
            edge_id = f"ctxedge:{ci:04d}:{slot}"
            edge = GraphEdge(
                record_id=f"ctxrelation:{edge_id}",
                edge_id=edge_id,
                relation_id=f"ctxrelation:{edge_id}",
                source_node_id=src,
                target_node_id=tgt,
                label=label,
                confidence=round(0.25 + 0.05 * ((ci + ord(slot)) % 14), 3),
                evidence=Evidence(mention_ids=[_mention_id(ci, 0, 0)], confidence=0.8),
                provenance=_prov(doc, 0, 10),
                classification=_CLS,
            )
            yield GraphRecord(
                record_id=f"ctxgraph:edge:{edge_id}",
                scope=GraphScope.EDGE,
                edge=edge,
                provenance=_prov(doc, 0, 10),
                classification=_CLS,
            )


def _live_counts() -> dict[str, int]:
    """Live instances of each relevant type, restricted to THIS test's records."""
    gc.collect()
    counts = dict.fromkeys(
        (
            "ChunkRecord",
            "EntityMention",
            "GraphRecord",
            "GraphNode",
            "GraphEdge",
            "node_view",
            "edge_meta",
            "chunk_entities",
            "folded_entries",
            "neighbor",
        ),
        0,
    )
    for obj in gc.get_objects():
        cls = type(obj)
        if cls is ChunkRecord:
            if obj.record_id.startswith("ctxchunk:"):
                counts["ChunkRecord"] += 1
        elif cls is EntityMention:
            if obj.record_id.startswith("ctxmention:"):
                counts["EntityMention"] += 1
        elif cls is GraphRecord:
            if obj.record_id.startswith("ctxgraph:"):
                counts["GraphRecord"] += 1
        elif cls is GraphNode:
            if obj.node_id.startswith("ctxnode:"):
                counts["GraphNode"] += 1
        elif cls is GraphEdge:
            if obj.edge_id.startswith("ctxedge:"):
                counts["GraphEdge"] += 1
        elif cls is _NodeView:
            if obj.node_id.startswith("ctxnode:"):
                counts["node_view"] += 1
        elif cls is _EdgeMeta:
            if obj.edge_id.startswith("ctxedge:"):
                counts["edge_meta"] += 1
        elif cls is _ChunkEntities:
            if obj.chunk_record_id.startswith("ctxchunk:"):
                counts["chunk_entities"] += 1
                counts["folded_entries"] += len(obj.best_mention_confidence)
        elif cls is _Neighbor and obj.edge_id.startswith("ctxedge:"):
            counts["neighbor"] += 1
    return counts


def _mention_keyed_locals(gen: Iterator[ChunkRecord]) -> list[str]:
    """Names in the suspended ``enrich`` frame still bound to a dict keyed by MENTION ids.

    ``gc.get_objects()`` cannot see the mention->node join: CPython *untracks* a dict whose keys
    and values are all atomic (a ``dict[str, str]`` qualifies), so it vanishes from the GC's object
    list even while fully alive — a probe over ``gc`` would pass no matter what. ``enrich`` is a
    generator, so its frame is right here while it is suspended at a yield: read the locals
    directly. Matching on the VALUE (any local holding a mention-keyed dict) rather than on the
    name also catches the table surviving under an alias.
    """
    frame = getattr(gen, "gi_frame", None)
    if frame is None:
        return []
    return [
        name
        for name, value in dict(frame.f_locals).items()
        if isinstance(value, dict) and _JOIN_SENTINEL in value
    ]


def _sampled_run(
    enricher: ContextHeaderEnricher,
) -> tuple[dict[int, dict[str, int]], dict[int, list[str]], int]:
    """Drive enrich() to exhaustion, sampling live counts at the first and the last chunk.

    The output chunks are deliberately NOT collected: holding them would put this test's own
    references into every count it then asserts on.
    """
    samples: dict[int, dict[str, int]] = {}
    join_locals: dict[int, list[str]] = {}
    emitted = 0
    gen = enricher.enrich(_chunks(), _mentions(), _graph())
    for i, chunk in enumerate(gen):
        assert isinstance(chunk, ChunkRecord)
        emitted += 1
        if i in (0, _CHUNKS - 1):
            join_locals[i] = _mention_keyed_locals(gen)
            samples[i] = _live_counts()
    return samples, join_locals, emitted


def test_enrich_retains_no_input_records_and_nothing_accumulates() -> None:
    """The three pinned properties, sampled from inside the run at the first and last chunk."""
    enricher = ContextHeaderEnricher({})
    samples, join_locals, emitted = _sampled_run(enricher)

    assert emitted == _CHUNKS, "the enricher must thread the chunk stream 1:1"
    assert set(samples) == {0, _CHUNKS - 1}, f"probe misfired; sampled {sorted(samples)}"
    assert enricher.hub_entities_capped > 0, (
        "the fixture must exercise the hub-entity cap for the probe to be over real work"
    )
    first, last = samples[0], samples[_CHUNKS - 1]

    # (1) No pydantic input record survives the projection/fold — the 8.1 GiB term at UDA scale.
    for when, sample in (("first chunk", first), ("last chunk", last)):
        for kind in ("EntityMention", "GraphRecord", "GraphNode", "GraphEdge"):
            assert sample[kind] == 0, (
                f"{sample[kind]} input {kind}(s) were still alive at the {when}. enrich() must "
                "project every graph record and fold every mention on arrival and drop the "
                "record; holding them is exactly what OOM-killed the UDA run under the 29 GB cap."
            )
        assert sample["ChunkRecord"] <= _SLACK, (
            f"{sample['ChunkRecord']} ChunkRecords were alive at the {when}; the enricher must "
            "not retain the chunk stream it threads through."
        )
        assert sample["edge_meta"] == 0, (
            "edge projections outlived _index_graph; they are consumed into the adjacency and "
            "must be released with it."
        )
    for when, names in (("first chunk", join_locals[0]), ("last chunk", join_locals[_CHUNKS - 1])):
        assert names == [], (
            f"the mention->node join was still alive at the {when}, bound to {names}. It is the "
            f"one mention-sized table in the Stage ({_TOTAL_MENTIONS} entries here, 2.6 M at UDA "
            "scale) and nothing past the fold reads it — it must be dropped before the loop."
        )

    # (2) Mentions are FOLDED, not stored: the retained state is one entry per (chunk, entity).
    assert first["chunk_entities"] == _CHUNKS, (
        f"{first['chunk_entities']} chunk-entity folds exist for {_CHUNKS} chunks; the fixture is "
        "no longer exercising a many-chunk run."
    )
    assert first["folded_entries"] == _CHUNKS * _ENTITIES_PER_CHUNK, (
        f"{first['folded_entries']} folded entries for {_TOTAL_MENTIONS} mentions; the fold must "
        f"collapse a chunk's {_MENTIONS_PER_CHUNK} mentions to its {_ENTITIES_PER_CHUNK} distinct "
        "canonical entities, not keep per-mention state."
    )
    assert first["folded_entries"] < _TOTAL_MENTIONS, "the fold is not collapsing anything"

    # (3) Nothing accumulates across the chunk loop: every count is flat first -> last.
    for kind in ("node_view", "chunk_entities", "folded_entries", "neighbor"):
        assert last[kind] == first[kind], (
            f"{kind} grew from {first[kind]} to {last[kind]} between the first and the last of "
            f"{_CHUNKS} chunks — per-chunk state is accumulating across the run."
        )
    assert first["node_view"] == _NODES
    assert first["neighbor"] == 2 * _EDGES  # each edge is seen from BOTH endpoints


def test_headers_are_not_retained_by_the_enricher() -> None:
    """A yielded header is the caller's; the enricher keeps none (it would be a per-chunk leak)."""
    enricher = ContextHeaderEnricher({})
    live: list[int] = []
    for i, chunk in enumerate(enricher.enrich(_chunks(), _mentions(), _graph())):
        del chunk  # the caller drops it immediately: only enricher-held headers can survive
        if i in (0, _CHUNKS - 1):
            gc.collect()
            live.append(sum(1 for o in gc.get_objects() if type(o) is ContextHeader))
    assert live[1] <= live[0] + _SLACK, (
        f"live ContextHeaders grew {live[0]} -> {live[1]} across {_CHUNKS} chunks; enrich() is "
        "collecting the headers it yields instead of handing them off."
    )


def test_streaming_enrich_is_byte_identical_to_a_list_fed_enrich() -> None:
    """Bounded memory must not have cost a single byte of output.

    Same corpus, once through one-shot generators (what the Runner hands the Stage for the mention
    side) and once through fully materialised lists — the serialized chunks must be identical,
    including order, every header's entity/type/node-id alignment, its neighbor triples and caps.
    """
    for config in (
        {},
        {"max_neighbors_per_entity": 0, "max_triples": 0},
        {"max_neighbors_per_entity": 2, "max_triples": 3},
        {"min_entity_confidence": 0.6, "min_edge_confidence": 0.5},
    ):
        streamed_enricher = ContextHeaderEnricher(config)
        streamed = [
            c.model_dump_json()
            for c in streamed_enricher.enrich(_chunks(), _mentions(), _graph())
        ]
        listed_enricher = ContextHeaderEnricher(config)
        listed = [
            c.model_dump_json()
            for c in listed_enricher.enrich(list(_chunks()), list(_mentions()), list(_graph()))
        ]
        assert streamed == listed, f"output diverged for config {config}"
        assert streamed
        assert streamed_enricher.hub_entities_capped == listed_enricher.hub_entities_capped


class _OneShot:
    """Raises on a second pass — a generator would silently read an EMPTY corpus instead.

    Mirrors ``test_hyperedge_memory_shape._OneShot``: the Runner hands Context Enrichment a lazy,
    one-shot generator over the run's EntityMentions, so a Stage that iterated an input twice would
    drop every header with no error at all.
    """

    def __init__(self, records: list[Any]) -> None:
        self._records = records
        self.passes = 0

    def __iter__(self) -> Iterator[Any]:
        self.passes += 1
        if self.passes > 1:
            msg = (
                "an input stream was iterated more than once; the Runner hands Context Enrichment "
                "a one-shot generator for the mention stream, so a second pass reads an EMPTY "
                "corpus"
            )
            raise AssertionError(msg)
        yield from self._records


def test_each_input_stream_is_consumed_exactly_once() -> None:
    """Every stream is single-pass — the property streaming the inputs at all depends on."""
    streams = {
        "chunks": _OneShot(list(_chunks())),
        "mentions": _OneShot(list(_mentions())),
        "graph": _OneShot(list(_graph())),
    }
    enriched = list(
        ContextHeaderEnricher({}).enrich(
            streams["chunks"], streams["mentions"], streams["graph"]
        )
    )
    assert len(enriched) == _CHUNKS
    assert all(c.context_header is not None for c in enriched)
    assert all(s.passes == 1 for s in streams.values())


@pytest.mark.parametrize("copies", [2, 3])
def test_a_repeated_chunk_id_keeps_its_header(copies: int) -> None:
    """Why the folded per-chunk map is NOT popped as each chunk is emitted.

    Unlike the Hyperedge builder — which dedups its chunk views by ``record_id`` — this Provider
    threads the chunk stream 1:1 and cannot dedup, and a Pipeline may wire it to more than one
    chunk-carrying parent (Chunk *and* Redaction both re-emit the same ``record_id``, ADR-0042).
    Popping the map would silently strip the header off every repeat; the map is the folded
    aggregate, so keeping it whole is bounded anyway.
    """
    repeated = [c for _ in range(copies) for c in _chunks()]
    enriched = [
        c.model_dump_json()
        for c in ContextHeaderEnricher({}).enrich(repeated, _mentions(), _graph())
    ]
    assert len(enriched) == copies * _CHUNKS
    single = enriched[:_CHUNKS]
    for i in range(1, copies):
        assert enriched[i * _CHUNKS : (i + 1) * _CHUNKS] == single, (
            "a repeated chunk id lost its context_header — the per-chunk fold was released "
            "before the stream was exhausted."
        )

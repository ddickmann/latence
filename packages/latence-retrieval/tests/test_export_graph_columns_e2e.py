"""R4 regression: the framework's OWN export must feed the graph-augmented-retrieval seam.

ADR-0051's graph features (``kg_node_centrality`` / ``kg_node_community``) and join keys
(``kg_node_ids`` / ``neighbor_node_ids``) are computed at index time and stamped onto each chunk's
:class:`~latence_core.contracts.ContextHeader`. The query-time consumers (``expand`` /
``multihop`` / ``pack``) read them off ``Candidate.metadata`` under fixed default keys. A store
copies every non-text corpus column into that metadata verbatim. So the whole seam is only alive if
the Export surfaces the header under the consumers' EXACT default column names.

This test walks the framework's own path end-to-end — export a chunk whose header carries the
ADR-0051 graph features, map the exported Parquet columns into ``Candidate.metadata`` exactly as
``QdrantBackend._to_candidate`` does, then run the default ``expand`` consumer — and asserts the
graph bonuses actually land. On the pre-fix export (which emitted an opaque ``context_header`` blob
and a mis-named ``neighbor_entity_ids`` column, and NO ``kg_node_ids`` / ``kg_node_community`` /
``kg_node_centrality`` columns) the candidate metadata carries none of the consumer keys, every
graph bonus is ``0.0``, and the augmented list is byte-identical to first-stage — so this test
fails. The assertions are on observable output (which scores changed, which candidate rose), never
on the column names alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
from latence_core.contracts import ChunkRecord, Classification, ContextHeader, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from latence_retrieval import Candidate, Expander

if TYPE_CHECKING:
    from pathlib import Path

_TEXT_KEY = "content"


def _prov(n: int) -> Provenance:
    return Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=0,
        char_end=max(1, n),
        page_start=1,
        page_end=1,
    )


def _chunk(record_id: str, text: str, header: ContextHeader | None) -> ChunkRecord:
    return ChunkRecord(
        record_id=record_id,
        provenance=_prov(len(text)),
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=0,
        content=text,
        token_count=max(1, len(text.split())),
        context_header=header,
    )


def _candidates_from_export(uris: list[str], storage: Storage, scores: dict[str, float]) -> list[Candidate]:
    """Read the exported records.parquet and build candidates as ``QdrantBackend._to_candidate`` does.

    A store copies every non-text payload column into ``Candidate.metadata`` verbatim (the text
    column becomes ``text``). This mirrors that faithfully: it makes NO assumption about which graph
    columns exist — it copies whatever the Export actually wrote, so a pre-fix export simply yields
    candidates whose metadata lacks the consumer keys.
    """
    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    candidates: list[Candidate] = []
    for row in table.to_pylist():
        rid = row["record_id"]
        metadata: dict[str, Any] = {k: v for k, v in row.items() if k != _TEXT_KEY}
        candidates.append(
            Candidate(id=rid, score=scores[rid], text=str(row.get(_TEXT_KEY) or ""), metadata=metadata)
        )
    return candidates


def test_exported_graph_columns_drive_the_expand_seam_off_the_frameworks_own_corpus(
    tmp_path: Path,
) -> None:
    """The default ``expand`` consumer reads ADR-0051 graph features off the framework's own export.

    Two chunks share a community (``c-acme``) and one owns a graph-hub entity (high centrality). The
    seed is the top-scored ``s``; candidate ``c2`` is a 1-hop neighbour of ``s`` AND shares its
    community AND is a hub. With community/centrality weights, ``c2``'s graph bonus must lift it —
    which is only possible if the export surfaced ``kg_node_ids`` / ``neighbor_node_ids`` /
    ``kg_node_community`` / ``kg_node_centrality`` under the consumer's default keys.
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)

    seed_header = ContextHeader(
        entities=["Acme"],
        entity_types=["ORG"],
        kg_node_ids=["n-acme"],
        neighbor_node_ids=["n-berlin"],
        kg_node_community=["c-acme"],
        kg_node_centrality=[0.9],
    )
    # c2 owns n-berlin (a KG edge from the seed's reach), shares community c-acme, and is a hub.
    c2_header = ContextHeader(
        entities=["Berlin"],
        entity_types=["LOC"],
        kg_node_ids=["n-berlin"],
        neighbor_node_ids=[],
        kg_node_community=["c-acme"],
        kg_node_centrality=[0.8],
    )
    # c3 is a higher-scored stranger with NO graph link/community/centrality to the seed.
    c3_header = ContextHeader(
        entities=["Zeta"],
        entity_types=["ORG"],
        kg_node_ids=["n-zeta"],
        neighbor_node_ids=[],
        kg_node_community=["c-other"],
        kg_node_centrality=[0.0],
    )

    records = [
        _chunk("s", "Acme is based in Berlin.", seed_header),
        _chunk("c2", "Berlin is a city.", c2_header),
        _chunk("c3", "Zeta is unrelated.", c3_header),
    ]
    first_stage = {"s": 1.0, "c2": 0.5, "c3": 0.6}

    export = JsonlParquetExport({"basename": "records", "context_columns": True})
    uris = export.export(records, storage, store)
    candidates = _candidates_from_export(uris, storage, first_stage)

    # Sanity: the store round-tripped the ADR-0051 keys the consumer reads (pre-fix: absent).
    c2_meta = next(c.metadata for c in candidates if c.id == "c2")
    assert c2_meta["kg_node_ids"] == ["n-berlin"]
    assert c2_meta["kg_node_community"] == ["c-acme"]
    assert c2_meta["kg_node_centrality"] == [0.8]

    expander = Expander(
        {"seed_top_k": 1, "hop_weight": 0.5, "community_weight": 1.0, "centrality_weight": 1.0}
    )
    augmented = expander.process("q", candidates)
    scored = {c.id: c.score for c in augmented}

    # c2 earned hop(0.5) + community(1.0) + centrality(1.0*0.8) = 2.3 on top of its 0.5 first-stage.
    assert scored["c2"] == 0.5 + 0.5 + 1.0 + 1.0 * 0.8
    # The stranger c3 earned no graph bonus at all — its score is unchanged from first-stage.
    assert scored["c3"] == 0.6
    # The graph signal genuinely reordered the list: c2 (linked hub) now outranks the c3 stranger,
    # which is the OPPOSITE of the first-stage order (0.6 > 0.5) — the seam is alive, not a no-op.
    assert augmented[0].id == "s"
    assert [c.id for c in augmented] == ["s", "c2", "c3"]


def test_absent_graph_features_leave_the_expand_seam_a_graceful_no_op(tmp_path: Path) -> None:
    """A chunk exported without the ADR-0051 graph-feature enrichment writes empty graph columns.

    The store still round-trips the (empty) columns, and the community/centrality signals contribute
    nothing — the additive / graceful-when-absent discipline, byte-identical to the unenriched path.
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    header = ContextHeader(
        entities=["Acme"],
        entity_types=["ORG"],
        kg_node_ids=["n-acme"],
        neighbor_node_ids=["n-berlin"],
        # graph_features OFF: no community / centrality
    )
    records = [
        _chunk("s", "Acme is based in Berlin.", header),
        _chunk("c2", "Berlin is a city.", _chunk_header_berlin()),
    ]
    first_stage = {"s": 1.0, "c2": 0.5}
    export = JsonlParquetExport({"basename": "records", "context_columns": True})
    uris = export.export(records, storage, store)
    candidates = _candidates_from_export(uris, storage, first_stage)

    c2_meta = next(c.metadata for c in candidates if c.id == "c2")
    assert c2_meta["kg_node_community"] == []
    assert c2_meta["kg_node_centrality"] == []

    # community/centrality weights contribute nothing (empty columns); only the 1-hop edge remains.
    expander = Expander(
        {"seed_top_k": 1, "hop_weight": 0.5, "community_weight": 1.0, "centrality_weight": 1.0}
    )
    scored = {c.id: c.score for c in expander.process("q", candidates)}
    assert scored["c2"] == 0.5 + 0.5  # only the hop bonus, no community/centrality
    assert scored["s"] == 1.0


def _chunk_header_berlin() -> ContextHeader:
    return ContextHeader(
        entities=["Berlin"],
        entity_types=["LOC"],
        kg_node_ids=["n-berlin"],
        neighbor_node_ids=[],
    )

"""The console must stay usable on a run that is bigger than the fixtures (audit R4).

Every other console fixture holds 1–5 chunks, so peak memory and page size are indistinguishable
between a streaming reader and one that materialises the whole run. These tests build a run whose
checkpoints are an order of magnitude larger than the memory budget the console is allowed to
spend, and assert the budget directly:

* opening **one** chunk must not cost the whole run (``RunArtifacts.chunk`` is a targeted,
  streaming lookup, not a full scan);
* a list view must emit a **bounded** page, however many records the run holds.

The budgets are measured with :mod:`tracemalloc` (Python allocations, deterministic) rather than
RSS, so they do not flake on an allocator's whim.
"""

from __future__ import annotations

import json
import tracemalloc
from typing import TYPE_CHECKING

import pytest
from latence_console.artifacts import MAX_LINE_BYTES, MAX_SCAN_RECORDS, RunArtifacts
from latence_console.server import ConsoleApp
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
    RiskMarker,
    ScreeningCheckpoint,
    ScreeningFinding,
    ScreeningVerdict,
)
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

# Big enough that a whole-run reader blows every budget below by more than an order of magnitude,
# small enough that writing it stays under a second.
CHUNK_COUNT = 4000
CHUNK_CHARS = 800

DOC_ID = "sha256:" + "b" * 58
RUN_ID = "big"

# One chunk page must cost no more than this. The run's checkpoints are ~16 MB; a reader that
# materialises them peaks two orders of magnitude above this line.
CHUNK_PAGE_PEAK_BYTES = 6 * 1024 * 1024

# A list page must be bounded regardless of run size. 4000 unpaginated rows are ~1 MB.
LIST_PAGE_MAX_BYTES = 256 * 1024


def _classification() -> Classification:
    return Classification(language="en", category="report", sensitivity="internal")


def _provenance(index: int) -> Provenance:
    return Provenance(
        source_uri="file:///corpus/big.md",
        file_name="big.md",
        file_type="md",
        file_size=CHUNK_COUNT * CHUNK_CHARS,
        document_id=DOC_ID,
        page_start=1,
        page_end=1,
        char_start=index * CHUNK_CHARS,
        char_end=(index + 1) * CHUNK_CHARS,
    )


def _chunk(index: int, *, redacted: bool) -> ChunkRecord:
    text = f"chunk {index} " + "x" * CHUNK_CHARS
    return ChunkRecord(
        record_id=f"{DOC_ID}#chunk-{index}",
        provenance=_provenance(index),
        classification=_classification(),
        document_record_id=DOC_ID,
        index=index,
        content=text,
        token_count=CHUNK_CHARS // 4,
        masked_content=f"[MASKED {index}] " + "x" * CHUNK_CHARS if redacted else None,
    )


@pytest.fixture(scope="module")
def big_storage(tmp_path_factory: pytest.TempPathFactory) -> tuple[Storage, str]:
    """A run whose chunk + redact checkpoints together are ~16 MB of JSONL."""
    root: Path = tmp_path_factory.mktemp("bigcorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/{RUN_ID}"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {
                    "name": "big",
                    "stages": [
                        {"name": "chunk", "depends_on": []},
                        {"name": "redact", "depends_on": ["chunk"]},
                    ],
                },
                "run_id": RUN_ID,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    for stage, redacted in (("chunk", False), ("redact", True)):
        payload = "".join(
            _chunk(i, redacted=redacted).model_dump_json() + "\n" for i in range(CHUNK_COUNT)
        )
        storage.write_atomic(f"{run_dir}/checkpoints/{stage}.jsonl", payload.encode("utf-8"))
    return storage, storage_uri


@pytest.fixture(scope="module")
def big_app(big_storage: tuple[Storage, str]) -> ConsoleApp:
    storage, storage_uri = big_storage
    return ConsoleApp.over_storage(storage, storage_uri)


def _peak_bytes(call: object) -> tuple[object, int]:
    """Run ``call`` and report what it returned plus its peak Python allocation."""
    assert callable(call)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = call()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_opening_one_chunk_does_not_materialise_the_run(big_app: ConsoleApp) -> None:
    """The chunk inspector is a targeted lookup: one 6 KB page must not cost 16 MB of records."""
    target = f"/run/{RUN_ID}/chunk?id={DOC_ID}%23chunk-7"
    big_app.respond("GET", target)  # warm caches so the measurement is the read, not the imports

    response, peak = _peak_bytes(lambda: big_app.respond("GET", target))

    assert isinstance(response, object)
    status = response.status  # type: ignore[attr-defined]
    body = response.body  # type: ignore[attr-defined]
    assert status == 200
    assert b"chunk-7" in body
    assert peak < CHUNK_PAGE_PEAK_BYTES, (
        f"opening one chunk peaked at {peak / 1024 / 1024:.1f} MB — the whole run was "
        f"materialised (budget {CHUNK_PAGE_PEAK_BYTES / 1024 / 1024:.0f} MB)"
    )


def test_chunk_lookup_shows_the_most_downstream_record(big_app: ConsoleApp) -> None:
    """A targeted lookup must still win from the DOWNSTREAM checkpoint (ADR-0042/0044)."""
    response = big_app.respond("GET", f"/run/{RUN_ID}/chunk?id={DOC_ID}%23chunk-7")

    assert b"[MASKED 7]" in response.body or b"MASKED" in response.body
    assert response.status == 200


def test_chunk_list_page_is_bounded(big_app: ConsoleApp) -> None:
    """4000 chunks must not become one unpaginated HTML page."""
    response = big_app.respond("GET", f"/run/{RUN_ID}/chunks")

    assert response.status == 200
    assert len(response.body) < LIST_PAGE_MAX_BYTES, (
        f"the chunk list rendered {len(response.body) / 1024:.0f} KB in one page — "
        "every row was emitted with no pagination"
    )
    # The pager must actually be there and reachable, or "bounded" would just mean "truncated".
    assert b"page=2" in response.body


def test_chunk_list_pager_moves(big_app: ConsoleApp) -> None:
    """Page 2 shows different rows than page 1, and links back."""
    first = big_app.respond("GET", f"/run/{RUN_ID}/chunks")
    second = big_app.respond("GET", f"/run/{RUN_ID}/chunks?page=2")

    assert second.status == 200
    assert first.body != second.body
    assert len(second.body) < LIST_PAGE_MAX_BYTES
    assert b"page=1" in second.body


def test_truncated_list_is_stated_and_stays_redacted(big_app: ConsoleApp) -> None:
    """A capped list says so, and every row it does show is the post-Redaction version.

    The cap is only safe because the scan walks the checkpoints most-downstream first. Reading
    forwards and stopping at the cap would have filled the map from the *pre*-Redaction
    ``chunk.jsonl`` and surfaced unmasked text (ADR-0042/0044).
    """
    response = big_app.respond("GET", f"/run/{RUN_ID}/chunks")

    assert b"Partial view" in response.body
    assert b">redacted<" in response.body
    body = response.body.decode("utf-8")
    rows = body.count("/chunk?id=")
    assert rows > 0
    # Every row on the page carries the "redacted" pill: none came from the upstream checkpoint.
    assert body.count(">redacted<") == rows


def test_read_jsonl_never_reads_a_whole_file(
    big_storage: tuple[Storage, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader streams: ``Storage.read_text`` is not on the checkpoint path at all."""
    storage, storage_uri = big_storage

    def _forbidden(self: Storage, uri: str, encoding: str = "utf-8") -> str:
        if "/checkpoints/" in uri:
            msg = f"read_text called on a checkpoint: {uri}"
            raise AssertionError(msg)
        return Storage.read_bytes(self, uri).decode(encoding)

    monkeypatch.setattr(Storage, "read_text", _forbidden)

    artifacts = RunArtifacts(storage, f"{storage_uri}/_latence/runs/{RUN_ID}")
    chunk = artifacts.chunk(f"{DOC_ID}#chunk-11")

    assert chunk is not None
    assert chunk.index == 11


# -- Evidence beyond the scan cap (audit R5) ---------------------------------
#
# The list views' scan is capped at MAX_SCAN_RECORDS; the KG views always scan (they need the
# node/edge maps). A mention the cap dropped is still ON DISK, and the run's Evidence still
# cites it — so the graph must resolve it, not label it "not in this run's checkpoints".
# Claiming a checkpointed record was never checkpointed is a false provenance claim
# (ADR-0021/0031).

MENTION_COUNT = MAX_SCAN_RECORDS + 500
BEYOND_CAP = MENTION_COUNT - 1
KG_RUN_ID = "kgbig"
KG_NODE_A = "node-a"
KG_NODE_B = "node-b"
KG_EDGE = "edge-a-b"
CITED_MENTION = f"{DOC_ID}#mention-{BEYOND_CAP}"
CITED_TEXT = f"Beyond-cap Entity {BEYOND_CAP}"


def _mention_record(index: int) -> EntityMention:
    return EntityMention(
        record_id=f"{DOC_ID}#mention-{index}",
        provenance=_provenance(index),
        classification=_classification(),
        document_record_id=DOC_ID,
        chunk_record_id=f"{DOC_ID}#chunk-{index}",
        index=0,
        label="ORG",
        text=f"Beyond-cap Entity {index}",
        confidence=0.9,
    )


def _kg_records() -> list[GraphRecord]:
    provenance = _provenance(0)
    evidence = Evidence(
        mention_ids=[CITED_MENTION],
        document_ids=[DOC_ID],
        snippet="a relation justified by a mention past the scan cap",
        confidence=0.8,
    )
    nodes = [
        GraphNode(
            record_id=node_id,
            provenance=provenance,
            classification=_classification(),
            node_id=node_id,
            entity_id=f"entity-{node_id}",
            label="ORG",
            canonical_name=name,
            member_mention_ids=[CITED_MENTION],
            source_document_ids=[DOC_ID],
            confidence=0.9,
            evidence=evidence,
        )
        for node_id, name in ((KG_NODE_A, "Alpha"), (KG_NODE_B, "Beta"))
    ]
    edge = GraphEdge(
        record_id=KG_EDGE,
        provenance=provenance,
        classification=_classification(),
        edge_id=KG_EDGE,
        relation_id="relation-a-b",
        source_node_id=KG_NODE_A,
        target_node_id=KG_NODE_B,
        label="partners_with",
        confidence=0.8,
        properties={},
        evidence=evidence,
    )
    return [
        *(
            GraphRecord(
                record_id=f"{node.node_id}#record",
                provenance=provenance,
                classification=_classification(),
                scope=GraphScope.NODE,
                node=node,
            )
            for node in nodes
        ),
        GraphRecord(
            record_id=f"{KG_EDGE}#record",
            provenance=provenance,
            classification=_classification(),
            scope=GraphScope.EDGE,
            edge=edge,
        ),
    ]


@pytest.fixture(scope="module")
def kg_storage(tmp_path_factory: pytest.TempPathFactory) -> tuple[Storage, str]:
    """A run whose entity checkpoint holds more mentions than the scan cap can keep."""
    root: Path = tmp_path_factory.mktemp("kgcorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/{KG_RUN_ID}"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {
                    "name": "kgbig",
                    "stages": [
                        {"name": "entities", "depends_on": []},
                        {"name": "graph", "depends_on": ["entities"]},
                    ],
                },
                "run_id": KG_RUN_ID,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    storage.write_atomic(
        f"{run_dir}/checkpoints/entities.jsonl",
        "".join(
            _mention_record(i).model_dump_json() + "\n" for i in range(MENTION_COUNT)
        ).encode("utf-8"),
    )
    storage.write_atomic(
        f"{run_dir}/checkpoints/graph.jsonl",
        "".join(r.model_dump_json() + "\n" for r in _kg_records()).encode("utf-8"),
    )
    return storage, storage_uri


@pytest.fixture(scope="module")
def kg_app(kg_storage: tuple[Storage, str]) -> ConsoleApp:
    storage, storage_uri = kg_storage
    return ConsoleApp.over_storage(storage, storage_uri)


def test_graph_resolves_evidence_beyond_the_scan_cap(kg_app: ConsoleApp) -> None:
    """A cited mention the scan's cap dropped is still resolved — it IS in the checkpoints.

    The graph view scans first (it needs the node/edge maps), and the entity checkpoint holds
    more mentions than the cap keeps. Answering the Evidence lookup from that truncated map
    reported a checkpointed mention as absent: a false provenance claim (ADR-0021/0031).
    """
    response = kg_app.respond("GET", f"/run/{KG_RUN_ID}/graph")

    assert response.status == 200
    body = response.body.decode("utf-8")
    assert "mention not in this run&#x27;s checkpoints" not in body
    assert "not in this run" not in body
    assert CITED_TEXT in body


def test_entity_page_resolves_evidence_beyond_the_scan_cap(kg_app: ConsoleApp) -> None:
    """The same holds on the per-entity page, which also always scans."""
    response = kg_app.respond("GET", f"/run/{KG_RUN_ID}/entity?id={KG_NODE_A}")

    assert response.status == 200
    body = response.body.decode("utf-8")
    assert "not in this run" not in body
    assert CITED_TEXT in body


def test_lookup_after_a_truncated_scan_still_finds_the_record(
    kg_storage: tuple[Storage, str],
) -> None:
    """``mention()`` must not become a *worse* lookup once a list view has scanned."""
    storage, storage_uri = kg_storage
    artifacts = RunArtifacts(storage, f"{storage_uri}/_latence/runs/{KG_RUN_ID}")

    fresh = artifacts.mention(CITED_MENTION)
    artifacts.graph_edges()  # forces the capped scan
    assert artifacts.truncated()
    after_scan = artifacts.mention(CITED_MENTION)

    assert fresh is not None
    assert after_scan is not None
    assert after_scan.text == CITED_TEXT


def test_absent_mention_is_still_reported_as_unresolved(
    kg_storage: tuple[Storage, str],
) -> None:
    """The fallback must not invent a record: an id the run never wrote stays ``None``."""
    storage, storage_uri = kg_storage
    artifacts = RunArtifacts(storage, f"{storage_uri}/_latence/runs/{KG_RUN_ID}")
    artifacts.graph_edges()

    assert artifacts.mention(f"{DOC_ID}#mention-999999") is None


def test_entity_page_states_a_partial_view(kg_app: ConsoleApp) -> None:
    """The per-entity page's relations come from the capped scan, so it must say so too."""
    response = kg_app.respond("GET", f"/run/{KG_RUN_ID}/entity?id={KG_NODE_A}")

    assert response.status == 200
    assert b"Partial view" in response.body


# -- an empty list must not assert emptiness of the run (audit R5) ------------
#
# The scan keeps at most MAX_SCAN_RECORDS graph records. A run whose graph is bigger than that
# has entities whose every edge sits beyond the cap — and the entity page rendered
# "Relations (0)" over the flat sentence "No relation in this graph touches this entity" for
# exactly those. That is an affirmative claim of isolation the graph checkpoints contradict:
# the operator auditing why the KG connected two entities is told the entity is isolated.

BIG_GRAPH_RUN_ID = "kghuge"
BIG_GRAPH_NODES = MAX_SCAN_RECORDS + 501
BIG_GRAPH_EDGES = MAX_SCAN_RECORDS + 500


def _plain_evidence() -> Evidence:
    return Evidence(
        mention_ids=[], document_ids=[DOC_ID], snippet="a relation past the cap", confidence=0.8
    )


def _big_graph_records() -> list[GraphRecord]:
    provenance = _provenance(0)
    evidence = _plain_evidence()
    records: list[GraphRecord] = []
    for i in range(BIG_GRAPH_NODES):
        node = GraphNode(
            record_id=f"n{i}",
            provenance=provenance,
            classification=_classification(),
            node_id=f"n{i}",
            entity_id=f"entity-{i}",
            label="ORG",
            canonical_name=f"Node {i:05d}",
            member_mention_ids=[],
            source_document_ids=[DOC_ID],
            confidence=0.9,
            evidence=evidence,
        )
        records.append(
            GraphRecord(
                record_id=f"n{i}#record",
                provenance=provenance,
                classification=_classification(),
                scope=GraphScope.NODE,
                node=node,
            )
        )
    for i in range(BIG_GRAPH_EDGES):
        edge = GraphEdge(
            record_id=f"e{i}",
            provenance=provenance,
            classification=_classification(),
            edge_id=f"e{i}",
            relation_id=f"relation-{i}",
            source_node_id=f"n{i}",
            target_node_id=f"n{i + 1}",
            label=f"rel_{i:05d}",
            confidence=0.8,
            properties={},
            evidence=evidence,
        )
        records.append(
            GraphRecord(
                record_id=f"e{i}#record",
                provenance=provenance,
                classification=_classification(),
                scope=GraphScope.EDGE,
                edge=edge,
            )
        )
    return records


@pytest.fixture(scope="module")
def big_graph_storage(tmp_path_factory: pytest.TempPathFactory) -> tuple[Storage, str]:
    """A run whose graph checkpoint holds more nodes AND edges than the scan cap can keep."""
    root: Path = tmp_path_factory.mktemp("kghugecorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/{BIG_GRAPH_RUN_ID}"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {"name": "kghuge", "stages": [{"name": "graph", "depends_on": []}]},
                "run_id": BIG_GRAPH_RUN_ID,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    storage.write_atomic(
        f"{run_dir}/checkpoints/graph.jsonl",
        "".join(r.model_dump_json() + "\n" for r in _big_graph_records()).encode("utf-8"),
    )
    return storage, storage_uri


@pytest.fixture(scope="module")
def big_graph_app(big_graph_storage: tuple[Storage, str]) -> ConsoleApp:
    storage, storage_uri = big_graph_storage
    return ConsoleApp.over_storage(storage, storage_uri)


def _edges_on_disk_touching(storage: Storage, run_dir: str, node_id: str) -> list[str]:
    """Every edge id in the run's graph checkpoint with ``node_id`` as an endpoint.

    Read straight off Storage, bypassing the console entirely: ADR-0017 makes the files the
    source of truth, so the ground the page is judged against must come from the files, never
    from the reader under test.
    """
    found = []
    with storage.open(f"{run_dir}/checkpoints/graph.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            edge = record.get("edge")
            if edge and node_id in (edge["source_node_id"], edge["target_node_id"]):
                found.append(edge["edge_id"])
    return found


def test_entity_page_never_claims_isolation_on_a_capped_read(
    big_graph_app: ConsoleApp,
    big_graph_storage: tuple[Storage, str],
) -> None:
    """An entity whose edges the cap dropped must not be reported as touched by no relation.

    ``e{i}`` connects ``n{i}`` to ``n{i+1}``, so every node here is an endpoint of at least one
    edge on Storage — asserted against the checkpoint file itself, not assumed. Rendering the
    flat sentence "No relation in this graph touches this entity" for such a node states as
    fact the opposite of what the files say (ADR-0017: the files are the source of truth;
    ADR-0021/0031: provenance is true).
    """
    storage, storage_uri = big_graph_storage
    run_dir = f"{storage_uri}/_latence/runs/{BIG_GRAPH_RUN_ID}"
    # A node deep enough that both its edges sit past the cap the scan stops at.
    node_id = f"n{BIG_GRAPH_EDGES - 1}"
    on_disk = _edges_on_disk_touching(storage, run_dir, node_id)
    assert on_disk, f"fixture is wrong: {node_id} has no edge on Storage to be lied about"

    body = big_graph_app.respond(
        "GET", f"/run/{BIG_GRAPH_RUN_ID}/entity?id={node_id}"
    ).body.decode("utf-8")

    assert "Relations (0" in body, (
        f"the cap no longer drops {node_id}'s edges ({on_disk}) — the fixture must outgrow "
        "MAX_SCAN_RECORDS for this regression to mean anything"
    )
    assert "Partial view" in body, f"{node_id} lost its edges to the cap without saying so"
    assert "No relation in this graph touches this entity." not in body, (
        f"the entity page told the reader {node_id} is isolated, but the run's graph "
        f"checkpoint holds an edge with {node_id} as an endpoint"
    )
    # The weaker, true statement is made in its place — the page still says it found nothing.
    assert "was among the relations this view read" in body
    # And the tally is not left standing as a bare count of the whole run.
    assert "Relations (0)" not in body
    assert "Relations (0 read so far)" in body


SMALL_RUN_ID = "kgsmall"


@pytest.fixture(scope="module")
def small_kg_app(tmp_path_factory: pytest.TempPathFactory) -> ConsoleApp:
    """The same two-node, one-edge graph, on a run that fits entirely under the cap."""
    root: Path = tmp_path_factory.mktemp("kgsmallcorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/{SMALL_RUN_ID}"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {
                    "name": "kgsmall",
                    "stages": [
                        {"name": "entities", "depends_on": []},
                        {"name": "graph", "depends_on": ["entities"]},
                    ],
                },
                "run_id": SMALL_RUN_ID,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    storage.write_atomic(
        f"{run_dir}/checkpoints/entities.jsonl",
        (_mention_record(BEYOND_CAP).model_dump_json() + "\n").encode("utf-8"),
    )
    storage.write_atomic(
        f"{run_dir}/checkpoints/graph.jsonl",
        "".join(r.model_dump_json() + "\n" for r in _kg_records()).encode("utf-8"),
    )
    return ConsoleApp.over_storage(storage, storage_uri)


def test_entity_page_speaks_plainly_when_the_read_was_complete(
    small_kg_app: ConsoleApp,
) -> None:
    """The qualification is scoped to a capped read; a whole read still speaks plainly.

    Guards against "fixing" the false claim by hedging everywhere — a page that hedges even when
    it read the whole run teaches the reader to ignore the hedge, which costs the fix its value.
    ``node-b`` is the target of ``edge-a-b`` and this run holds four records in total.
    """
    response = small_kg_app.respond("GET", f"/run/{SMALL_RUN_ID}/entity?id={KG_NODE_B}")
    body = response.body.decode("utf-8")

    assert response.status == 200
    assert "Partial view" not in body
    assert "Relations (1)" in body
    assert "read so far" not in body


def test_complete_graph_with_no_edges_still_states_it_plainly(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A run that really assembled nothing must keep saying so — the true claim survives."""
    root: Path = tmp_path_factory.mktemp("kgemptycorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/empty"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {"name": "empty", "stages": [{"name": "graph", "depends_on": []}]},
                "run_id": "empty",
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    storage.write_atomic(f"{run_dir}/checkpoints/graph.jsonl", b"")
    app = ConsoleApp.over_storage(storage, storage_uri)

    body = app.respond("GET", "/run/empty/graph").body.decode("utf-8")

    assert "This run assembled no graph nodes." in body
    assert "This run assembled no graph edges." in body
    assert "Partial view" not in body


# -- the flagged-risk-marker list on a capped read (audit R6) -----------------

FLAG_RUN_ID = "flagcap"
#: Past MAX_SCAN_RECORDS, so the scan stops before the flagged chunk at the tail.
FLAG_CHUNK_COUNT = MAX_SCAN_RECORDS + 500
FLAG_DOC_ID = "sha256:" + "c" * 58
FLAG_CATEGORY = "prompt_injection"


def _flag_marker() -> RiskMarker:
    return RiskMarker(
        category=FLAG_CATEGORY,
        detector="screening.content_keyword",
        score=0.93,
        reason="matched injection pattern 'ignore previous instructions'",
    )


def _flag_provenance(index: int) -> Provenance:
    return Provenance(
        source_uri="file:///corpus/flagged.md",
        file_name="flagged.md",
        file_type="md",
        file_size=FLAG_CHUNK_COUNT * 64,
        document_id=FLAG_DOC_ID,
        page_start=1,
        page_end=1,
        char_start=index * 64,
        char_end=(index + 1) * 64,
    )


def _flag_chunk(index: int, *, marked: bool) -> ChunkRecord:
    return ChunkRecord(
        record_id=f"{FLAG_DOC_ID}#chunk-{index}",
        provenance=_flag_provenance(index),
        classification=_classification(),
        document_record_id=FLAG_DOC_ID,
        index=index,
        content=f"chunk {index} " + "y" * 40,
        token_count=12,
        risk_markers=[_flag_marker()] if marked else [],
    )


@pytest.fixture(scope="module")
def flag_storage(tmp_path_factory: pytest.TempPathFactory) -> tuple[Storage, str]:
    """A run bigger than the cap whose LAST chunk carries a risk marker into the corpus."""
    root: Path = tmp_path_factory.mktemp("flagcapcorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/{FLAG_RUN_ID}"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {"name": "flagcap", "stages": [{"name": "chunk", "depends_on": []}]},
                "run_id": FLAG_RUN_ID,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    last = FLAG_CHUNK_COUNT - 1
    storage.write_atomic(
        f"{run_dir}/checkpoints/chunk.jsonl",
        "".join(
            _flag_chunk(i, marked=(i == last)).model_dump_json() + "\n"
            for i in range(FLAG_CHUNK_COUNT)
        ).encode("utf-8"),
    )
    # The sidecar the Screener wrote when it raised the marker: the same run, saying so twice.
    storage.write_atomic(
        f"{run_dir}/screening/content_screen.jsonl",
        (
            ScreeningFinding(
                checkpoint=ScreeningCheckpoint.CONTENT,
                verdict=ScreeningVerdict.FLAG,
                document_id=FLAG_DOC_ID,
                detector="screening.content_keyword",
                category=FLAG_CATEGORY,
                reason=_flag_marker().reason,
                marker=_flag_marker(),
            ).model_dump_json()
            + "\n"
        ).encode("utf-8"),
    )
    return storage, storage_uri


def _markers_on_disk(storage: Storage, run_dir: str) -> list[str]:
    """Every chunk id in the run's chunk checkpoint that carries a risk marker.

    Read straight off Storage, bypassing the console: ADR-0017 makes the files the source of
    truth, so the ground the page is judged against must come from the files, never from the
    reader under test.
    """
    found = []
    with storage.open(f"{run_dir}/checkpoints/chunk.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("risk_markers"):
                found.append(record["record_id"])
    return found


def test_quarantine_page_never_claims_no_risk_marker_on_a_capped_read(
    flag_storage: tuple[Storage, str],
) -> None:
    """A marker the cap dropped must not be rendered as "the corpus carries none".

    The flagged list is derived from ``RunArtifacts.chunks()``, which is the capped scan, so a
    marker past MAX_SCAN_RECORDS is simply absent from the list the page renders. Saying
    "No record in this run's corpus carries a risk marker." there states as fact the opposite of
    what the run's own chunk checkpoint says — and this is the RAG-safety surface, the page a
    consumer reads to decide what to exclude (ADR-0017; ADR-0021/0031).
    """
    storage, storage_uri = flag_storage
    run_dir = f"{storage_uri}/_latence/runs/{FLAG_RUN_ID}"
    on_disk = _markers_on_disk(storage, run_dir)
    assert on_disk, "fixture is wrong: no chunk on Storage carries a marker to be lied about"

    app = ConsoleApp.over_storage(storage, storage_uri)
    body = app.respond("GET", f"/run/{FLAG_RUN_ID}/quarantine").body.decode("utf-8")

    assert "Flagged — kept, with a risk marker in the corpus (0" in body, (
        f"the cap no longer drops the flagged chunk ({on_disk}) — the fixture must outgrow "
        "MAX_SCAN_RECORDS for this regression to mean anything"
    )
    assert "Partial view" in body, "the run lost records to the cap without saying so"
    # The page HTML-escapes its text, so the claim is matched in the form it actually renders.
    assert "No record in this run&#x27;s corpus carries a risk marker." not in body, (
        "the quarantine page told the reader the corpus carries no risk marker, but the run's "
        f"chunk checkpoint holds {on_disk}"
    )
    # The weaker, true statement is made in its place — the page still says it found nothing.
    assert "No risk marker was among the records this view read" in body
    # And the tally is not left standing as a bare count of the whole run.
    assert "Flagged — kept, with a risk marker in the corpus (0)" not in body
    assert "Flagged — kept, with a risk marker in the corpus (0 read so far)" in body
    # The held-back half is NOT capped (the quarantine sidecars are read whole), so it must keep
    # speaking plainly — hedging a complete read teaches the reader to ignore the hedge.
    assert "Quarantined — removed from the pipeline (0)" in body
    assert "Screening held nothing back in this run." in body


def test_quarantine_page_speaks_plainly_when_the_read_was_complete(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A run that really carries no marker must keep saying so — the true claim survives."""
    root: Path = tmp_path_factory.mktemp("flagsmallcorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/flagsmall"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {"name": "flagsmall", "stages": [{"name": "chunk", "depends_on": []}]},
                "run_id": "flagsmall",
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    storage.write_atomic(
        f"{run_dir}/checkpoints/chunk.jsonl",
        (_flag_chunk(0, marked=False).model_dump_json() + "\n").encode("utf-8"),
    )
    app = ConsoleApp.over_storage(storage, storage_uri)

    body = app.respond("GET", "/run/flagsmall/quarantine").body.decode("utf-8")

    assert "Partial view" not in body
    assert "Flagged — kept, with a risk marker in the corpus (0)" in body
    assert "No record in this run&#x27;s corpus carries a risk marker." in body
    assert "read so far" not in body


# -- An over-long line is bounded WHILE it is read (audit R6) -----------------
#
# ``MAX_LINE_BYTES`` documents a promise: the reader will not accumulate a line bigger than the
# cap. Iterating the file object (``for raw in handle``) broke that promise in the only way that
# matters — the line was materialised in full and only *then* measured. A real
# ``checkpoints/source.jsonl`` line carries one whole ingested document inline, so a run over a
# large PDF puts a single hundred-megabyte line in front of every browse request.

#: An over-long line an order of magnitude past MAX_LINE_BYTES, so a reader that allocates it
#: before rejecting it is unmistakable against the budget below.
OVERLONG_LINE_BYTES = 64 * 1024 * 1024

#: Peak the targeted lookup is allowed while walking past that line. The enforced bound is
#: MAX_LINE_BYTES + one read block (~17 MB); a reader that materialises the line pays 64 MB+.
OVERLONG_READ_PEAK_BYTES = 24 * 1024 * 1024


@pytest.fixture(scope="module")
def torn_line_app(tmp_path_factory: pytest.TempPathFactory) -> ConsoleApp:
    """A run whose ``chunk.jsonl`` opens with one 64 MB line, then a real record."""
    root: Path = tmp_path_factory.mktemp("torncorpus")
    storage_uri = f"file://{root}/corpus"
    storage = Storage.for_uri(storage_uri)
    run_dir = f"{storage_uri}/_latence/runs/torn"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {
                "pipeline": {"name": "torn", "stages": [{"name": "chunk", "depends_on": []}]},
                "run_id": "torn",
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    overlong = b'{"content":"' + b"x" * OVERLONG_LINE_BYTES + b'"}\n'
    good = (_chunk(3, redacted=False).model_dump_json() + "\n").encode("utf-8")
    storage.write_atomic(f"{run_dir}/checkpoints/chunk.jsonl", overlong + good + overlong)
    return ConsoleApp.over_storage(storage, storage_uri)


def test_overlong_line_is_never_accumulated(torn_line_app: ConsoleApp) -> None:
    """The cap must bound the read, not merely describe it after the fact."""
    target = f"/run/torn/chunk?id={DOC_ID}%23chunk-3"
    torn_line_app.respond("GET", target)  # warm caches so the measurement is the read

    response, peak = _peak_bytes(lambda: torn_line_app.respond("GET", target))

    assert response.status == 200  # type: ignore[attr-defined]
    assert b"chunk 3" in response.body  # type: ignore[attr-defined]
    assert peak < OVERLONG_READ_PEAK_BYTES, (
        f"reading past a {OVERLONG_LINE_BYTES / 1024 / 1024:.0f} MB line peaked at "
        f"{peak / 1024 / 1024:.1f} MB — the line was materialised before MAX_LINE_BYTES "
        f"rejected it (budget {OVERLONG_READ_PEAK_BYTES / 1024 / 1024:.0f} MB)"
    )


def test_records_around_an_overlong_line_still_render(torn_line_app: ConsoleApp) -> None:
    """Abandoning a line must skip exactly that line — the records on either side survive."""
    body = torn_line_app.respond("GET", "/run/torn/chunks").body.decode("utf-8")

    assert "chunk-3" in body
    assert "x" * (MAX_LINE_BYTES // 1024) not in body

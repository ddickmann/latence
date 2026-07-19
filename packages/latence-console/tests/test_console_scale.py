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
from latence_console.artifacts import MAX_SCAN_RECORDS, RunArtifacts
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

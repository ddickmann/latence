"""W13 context enrichment, the Export half: header into the VECTOR only + the two metadata columns.

The crux (ADR-0039): the compact :class:`~latence_core.contracts.ContextHeader` inflates the EMBEDDING
input, never the stored chunk text — the LLM window stays byte-for-byte the clean chunk. And, opt-in,
two Parquet columns (``context_header`` JSON + ``neighbor_entity_ids`` native list) surface the KG
join keys for a downstream retriever, without touching the default schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    ContextHeader,
    Provenance,
)
from latence_core.embedding import HashingEmbedder
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

_DIMENSION = 32


def _prov(n: int) -> Provenance:
    return Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=0,
        char_end=n,
        page_start=1,
        page_end=1,
    )


def _header() -> ContextHeader:
    return ContextHeader(
        entities=["Acme", "Alice"],
        entity_types=["ORG", "PERSON"],
        neighbor_triples=["Alice —works_for→ Acme", "Acme —located_in→ Berlin"],
        kg_node_ids=["n-acme", "n-alice"],
        neighbor_node_ids=["n-berlin"],
    )


def _chunk(text: str, record_id: str = "c1", *, header: ContextHeader | None = None) -> ChunkRecord:
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


def _embedder_spec() -> dict[str, object]:
    return {"provider": "embedding.hashing", "config": {"dimension": _DIMENSION}}


def test_header_hits_the_embedding_input_but_stored_content_is_byte_unchanged(tmp_path: Path) -> None:
    """The vector embeds header+content; the exported ``content`` is byte-identical to no-header."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    text = "Alice works for Acme."
    export = JsonlParquetExport({"basename": "corpus", "embedder": _embedder_spec()})

    uris = export.export([_chunk(text, header=_header())], storage, store)
    jsonl_uri = next(u for u in uris if u.endswith(".jsonl"))
    row = json.loads(storage.read_text(jsonl_uri).splitlines()[0])

    # The STORED chunk text is byte-unchanged — the LLM window never grew (the crux).
    assert row["content"] == text

    # The attached vector is the embedder over header_text + "\n" + content — NOT over content alone.
    header_text = (
        "Entities: Acme (ORG), Alice (PERSON) | "
        "Relations: Alice —works_for→ Acme; Acme —located_in→ Berlin"
    )
    embedder = HashingEmbedder({"dimension": _DIMENSION})
    expected_with_header = next(embedder.embed([f"{header_text}\n{text}"]))
    plain = next(embedder.embed([text]))
    assert row["embedding"] == expected_with_header
    assert row["embedding"] != plain  # the header genuinely changed the vector


def test_no_header_embeds_plain_content_byte_identical(tmp_path: Path) -> None:
    """A chunk with context_header=None embeds exactly its own text (the additive no-op guarantee)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    text = "Alice works for Acme."
    export = JsonlParquetExport({"basename": "corpus", "embedder": _embedder_spec()})
    uris = export.export([_chunk(text, header=None)], storage, store)
    row = json.loads(storage.read_text(next(u for u in uris if u.endswith(".jsonl"))).splitlines()[0])
    assert row["content"] == text
    assert row["embedding"] == next(HashingEmbedder({"dimension": _DIMENSION}).embed([text]))


def test_context_columns_present_and_populated_when_opted_in(tmp_path: Path) -> None:
    """With ``context_columns: true`` the two metadata columns are written + populated."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus", "context_columns": True})
    uris = export.export([_chunk("Alice works for Acme.", header=_header())], storage, store)

    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert "context_header" in table.column_names
    assert "neighbor_entity_ids" in table.column_names
    # context_header is the header as a JSON string; neighbor_entity_ids the native list of keys.
    header_json = table.column("context_header")[0].as_py()
    assert json.loads(header_json)["entities"] == ["Acme", "Alice"]
    assert table.column("neighbor_entity_ids")[0].as_py() == ["n-berlin"]


def test_context_columns_absent_by_default_schema_byte_identical(tmp_path: Path) -> None:
    """Without the opt-in the Parquet schema is byte-identical to today (genuinely optional)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus"})  # no context_columns
    uris = export.export([_chunk("Alice works for Acme.", header=_header())], storage, store)
    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert "context_header" not in table.column_names
    assert "neighbor_entity_ids" not in table.column_names
    assert set(table.column_names) == {
        "record_id",
        "schema_version",
        "provenance",
        "classification",
        "content",
        "media_type",
    }


def test_unenriched_row_gets_empty_context_columns(tmp_path: Path) -> None:
    """An opted-in export of a header-less chunk writes empty context columns (no crash)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus", "context_columns": True})
    uris = export.export([_chunk("plain text", header=None)], storage, store)
    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert table.column("context_header")[0].as_py() == ""
    assert table.column("neighbor_entity_ids")[0].as_py() == []

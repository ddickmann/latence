"""W13 context enrichment, the Export half: header into the VECTOR only + the two metadata columns.

The crux (ADR-0039): the compact :class:`~latence_core.contracts.ContextHeader` inflates the EMBEDDING
input, never the stored chunk text — the LLM window stays byte-for-byte the clean chunk. And, opt-in,
Parquet columns (``context_header`` JSON + the ``kg_node_ids`` / ``neighbor_node_ids`` /
``kg_node_community`` / ``kg_node_centrality`` native lists, named for the consumers' default metadata
keys) surface the KG projection for a downstream retriever, without touching the default schema.
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
    """With ``context_columns: true`` the KG metadata columns are written + populated.

    The columns are named for the exact default metadata keys the ``expand`` / ``multihop`` / ``pack``
    consumers read (``kg_node_ids`` / ``neighbor_node_ids`` / ``kg_node_community`` /
    ``kg_node_centrality``), so a store that copies columns into candidate metadata makes the graph
    seam alive off the framework's own corpus (ADR-0051).
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus", "context_columns": True})
    uris = export.export([_chunk("Alice works for Acme.", header=_header())], storage, store)

    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert "context_header" in table.column_names
    # The graph columns are named for the consumers' default keys (not an opaque header blob).
    assert {"kg_node_ids", "neighbor_node_ids", "kg_node_community", "kg_node_centrality"} <= set(
        table.column_names
    )
    # context_header is the full header as a JSON string; the graph columns are native lists.
    header_json = table.column("context_header")[0].as_py()
    assert json.loads(header_json)["entities"] == ["Acme", "Alice"]
    assert table.column("kg_node_ids")[0].as_py() == ["n-acme", "n-alice"]
    assert table.column("neighbor_node_ids")[0].as_py() == ["n-berlin"]


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
    assert {"kg_node_ids", "neighbor_node_ids", "kg_node_community", "kg_node_centrality"}.isdisjoint(
        table.column_names
    )
    assert set(table.column_names) == {
        "record_id",
        "schema_version",
        "provenance",
        "classification",
        "content",
        "media_type",
    }


def _redacted_chunk(
    *, content: str, masked_content: str, header: ContextHeader
) -> ChunkRecord:
    """A chunk the Redaction Stage handled: raw ``content`` + its PII-masked ``masked_content``."""
    return ChunkRecord(
        record_id="c-red",
        provenance=_prov(len(content)),
        classification=Classification(language="en", sensitivity="confidential"),
        document_record_id="d1",
        index=0,
        content=content,
        masked_content=masked_content,
        token_count=max(1, len(content.split())),
        context_header=header,
    )


def test_redacted_header_never_re_leaks_masked_pii_into_inspectable_cells(tmp_path: Path) -> None:
    """R5 regression (ADR-0053): the EXPORTED ContextHeader must not carry a name redaction removed.

    Under the default ``extract_on='unmasked'`` posture the header's entity canonical names + rendered
    triples are the REAL, pre-redaction names. The header is embedding-input for the dense vector only
    (safe), but it is ALSO written into two INSPECTABLE sinks — the JSONL ``context_header`` field and
    the opt-in Parquet ``context_header`` column. Writing the raw names there re-leaks exactly what the
    corpus redaction stripped from ``content`` (REDACTION-BY-DEFAULT). This asserts the masked name is
    gone from BOTH inspectable sinks, a surviving non-PII entity stays, AND the dense vector is
    unchanged (it still sees the full header — vector-only is safe). FAILS on the pre-fix code (the raw
    ``Jane Confidential`` reaches both cells).
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    # NER ran on the RAW text, so the header names 'Jane Confidential'; the Redaction Stage masked
    # that person name out of the corpus text (masked_content), leaving the ORG 'Acme'.
    header = ContextHeader(
        entities=["Acme", "Jane Confidential"],
        entity_types=["ORG", "PERSON"],
        neighbor_triples=["Jane Confidential —works_for→ Acme", "Acme —located_in→ Berlin"],
        kg_node_ids=["n-acme", "n-jane"],
        neighbor_node_ids=["n-berlin"],
    )
    chunk = _redacted_chunk(
        content="Jane Confidential works at Acme.",
        masked_content="[NAME] works at Acme.",
        header=header,
    )
    export = JsonlParquetExport(
        {"basename": "corpus", "context_columns": True, "embedder": _embedder_spec()}
    )
    uris = export.export([chunk], storage, store)

    # (1) JSONL: the whole line (content + the context_header field) is clean of the masked name.
    jsonl_uri = next(u for u in uris if u.endswith(".jsonl"))
    jsonl_text = storage.read_text(jsonl_uri)
    assert "Jane Confidential" not in jsonl_text
    row = json.loads(jsonl_text.splitlines()[0])
    assert row["context_header"]["entities"] == ["Acme"]  # the PERSON dropped, the ORG kept
    assert row["context_header"]["neighbor_triples"] == []  # both triples name a non-surviving endpoint

    # (2) Parquet: no cell carries the masked name; the ``context_header`` column keeps only 'Acme'.
    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    all_cells = " ".join(str(v) for col in table.columns for v in col.to_pylist())
    assert "Jane Confidential" not in all_cells
    header_json = json.loads(table.column("context_header")[0].as_py())
    assert header_json["entities"] == ["Acme"]
    assert "Acme" in table.column("context_header")[0].as_py()
    # The surviving entity keeps its aligned join key; the masked entity's is dropped with it.
    assert table.column("kg_node_ids")[0].as_py() == ["n-acme"]

    # (3) The dense vector is UNCHANGED — it still embeds the FULL header (vector-only is safe), so the
    # export-time redaction did not weaken the retrieval signal.
    full_header_text = (
        "Entities: Acme (ORG), Jane Confidential (PERSON) | "
        "Relations: Jane Confidential —works_for→ Acme; Acme —located_in→ Berlin"
    )
    expected = next(
        HashingEmbedder({"dimension": _DIMENSION}).embed([f"{full_header_text}\n[NAME] works at Acme."])
    )
    assert row["embedding"] == expected


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
    assert table.column("kg_node_ids")[0].as_py() == []
    assert table.column("neighbor_node_ids")[0].as_py() == []
    assert table.column("kg_node_community")[0].as_py() == []
    assert table.column("kg_node_centrality")[0].as_py() == []

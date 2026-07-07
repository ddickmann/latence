"""The Embedder seam is wired end to end through Export (#37, regression).

Before #37 the Embedder was a dead seam: the ``Embedder`` Protocol + ``HashingEmbedder`` +
``embedding.hashing`` entry point existed, but no ``CapabilityKind`` and no Runner/Export
wiring, so no pipeline could route corpus text through it to attach vectors. #37 completes the
opt-in Export augmentation (ADR-0017): a corpus Export names its Embedder in
``config['embedder']`` and every exported row gets a fixed-width ``embedding`` vector.

These tests exercise the *wired path*:

* the in-core ``embedding.hashing`` reference resolves from the plugin registry through the
  corpus Export and attaches one vector per row (JSONL + Parquet), of the Embedder's dimension;
* embeddings are OPTIONAL — no ``embedder`` key ⇒ no vectors, and the text is still exported;
* the vector embeds the PII-handled ``masked_content`` of a RedactionRecord (never the raw text);
* a full LocalRunner run with an ``embedder`` on the corpus Export lands vectors on disk;
* a mis-named / non-Embedder Provider fails loudly at wiring time, not silently.

Each would fail if the Export-embedder wiring were reverted.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    Provenance,
    RedactionRecord,
)
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.runner import LocalRunner
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

_DIMENSION = 32


def _chunk(text: str, record_id: str = "c1", index: int = 0) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=index,
        content=text,
        token_count=max(1, len(text.split())),
    )


def _redaction(masked: str, record_id: str = "r1") -> RedactionRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=0,
        char_end=len(masked),
        page_start=1,
        page_end=1,
    )
    return RedactionRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        masked_content=masked,
    )


def _embedder_spec() -> dict[str, object]:
    return {"provider": "embedding.hashing", "config": {"dimension": _DIMENSION}}


def test_corpus_export_attaches_embedding_per_row(tmp_path: Path) -> None:
    """A corpus Export with an ``embedder`` attaches one dimension-wide vector per row."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus", "embedder": _embedder_spec()})

    records = [_chunk("alpha beta gamma", "c1", 0), _chunk("delta epsilon", "c2", 1)]
    uris = export.export(records, storage, store)

    jsonl_uri = next(u for u in uris if u.endswith(".jsonl"))
    rows = [json.loads(line) for line in storage.read_text(jsonl_uri).splitlines() if line.strip()]
    assert len(rows) == 2
    for row in rows:
        assert "embedding" in row, "corpus row is missing the opt-in embedding vector"
        assert len(row["embedding"]) == _DIMENSION
        assert all(isinstance(v, float) for v in row["embedding"])

    # Parquet carries a native list<double> embedding column of the same width.
    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert "embedding" in table.column_names
    first = table.column("embedding")[0].as_py()
    assert len(first) == _DIMENSION


def test_embeddings_are_optional_no_embedder_no_vectors(tmp_path: Path) -> None:
    """Without an ``embedder`` key the corpus exports text but NO embedding (ADR-0017)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus"})

    uris = export.export([_chunk("alpha beta")], storage, store)
    jsonl_uri = next(u for u in uris if u.endswith(".jsonl"))
    rows = [json.loads(line) for line in storage.read_text(jsonl_uri).splitlines() if line.strip()]
    assert rows and "embedding" not in rows[0]
    # The corpus text is still exported, so a vector DB can ingest and embed later.
    assert rows[0]["content"] == "alpha beta"

    parquet_uri = next(u for u in uris if u.endswith(".parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert "embedding" not in table.column_names


def test_embedding_covers_the_pii_handled_masked_text(tmp_path: Path) -> None:
    """A RedactionRecord's vector embeds its ``masked_content`` (the PII-handled text)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport({"basename": "corpus", "embedder": _embedder_spec()})

    masked = "Contact [EMAIL] for details."
    uris = export.export([_redaction(masked)], storage, store)
    jsonl_uri = next(u for u in uris if u.endswith(".jsonl"))
    row = json.loads(storage.read_text(jsonl_uri).splitlines()[0])
    assert "embedding" in row and len(row["embedding"]) == _DIMENSION

    # The attached vector is exactly the hashing embedder over masked_content — proving the
    # export embeds the PII-handled variant, not empty text or the raw content.
    from latence_core.embedding import HashingEmbedder

    expected = next(HashingEmbedder({"dimension": _DIMENSION}).embed([masked]))
    assert row["embedding"] == expected
    # And it is NOT the zero (empty-text) vector — masked_content was actually embedded.
    assert any(v != 0.0 for v in row["embedding"])


def test_runner_end_to_end_lands_embeddings_on_disk(tmp_path: Path) -> None:
    """A LocalRunner run with an ``embedder`` on the corpus Export writes vectors to disk."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    store = f"file://{tmp_path}/state"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: embed-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: public
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
    config:
      basename: corpus
      embedder:
        provider: embedding.hashing
        config: {{dimension: {_DIMENSION}}}
"""
    )
    LocalRunner().run(pipeline, run_id="embed-e2e")

    storage = Storage.for_uri(store)
    jsonl_uri = f"{store}/_latence/runs/embed-e2e/export/corpus.jsonl"
    assert storage.exists(jsonl_uri)
    rows = [json.loads(line) for line in storage.read_text(jsonl_uri).splitlines() if line.strip()]
    assert rows, "no corpus rows exported"
    for row in rows:
        assert len(row["embedding"]) == _DIMENSION


def test_embedding_capability_kind_exists() -> None:
    """The seam is a first-class Capability (the #37 'no CapabilityKind' gap)."""
    assert CapabilityKind.EMBEDDING.value == "embedding"


def test_standalone_embedding_stage_is_rejected_with_guidance(tmp_path: Path) -> None:
    """A bare `embedding` DAG Stage is a wiring error pointing at the Export augmentation."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    store = f"file://{tmp_path}/state"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: embed-bad
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: public
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: embed
    capability: embedding
    provider: embedding.hashing
    depends_on: [parse]
"""
    )
    with pytest.raises(ValueError, match="opt-in Export augmentation"):
        LocalRunner().run(pipeline, run_id="embed-bad")


def test_misnamed_embedder_provider_fails_loudly(tmp_path: Path) -> None:
    """A non-Embedder Provider named as the Export embedder raises, not silently no-ops."""
    with pytest.raises((KeyError, TypeError)):
        JsonlParquetExport(
            {"embedder": {"provider": "export.jsonl_parquet", "config": {}}}
        )
    with pytest.raises(KeyError):
        JsonlParquetExport({"embedder": {"provider": "embedding.nonexistent", "config": {}}})

"""The optional BM25 term-statistics artifact emission, tested THROUGH the Export (#164).

The exporter's observable contract, driven via ``JsonlParquetExport.export`` (never past it):

* with no ``bm25`` block the corpus is **byte-identical** to a plain export and neither BM25 file is
  written (the load-bearing "off by default / additive" guarantee, ADR-0017/0049);
* wired, it emits the two documented files — ``bm25-stats.json`` (corpus term statistics) and
  ``bm25-postings.parquet`` (per-record forward postings) — with the documented shapes;
* the term statistics **match a reference tokenization** (the AC): df / tf / doc_len / avgdl / idf
  computed independently from the same ``tokenizer.regex`` agree;
* the statistics are over the redacted corpus text (``masked_content``), never re-leaking PII;
* the whole artifact is deterministic (a re-export is byte-identical);
* a mis-named (non-Tokenizer) provider fails loudly, never silently emitting an empty artifact.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
from latence_core.bm25 import Bm25Accumulator, RegexWordTokenizer
from latence_core.contracts import ChunkRecord, Classification, Provenance, RedactionRecord
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path


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


def _bm25_config() -> dict[str, object]:
    return {"basename": "corpus", "bm25": {"provider": "tokenizer.regex", "config": {}}}


def _read_stats(storage: Storage, uris: list[str]) -> dict[str, Any]:
    stats_uri = next(u for u in uris if u.endswith("bm25-stats.json"))
    loaded: dict[str, Any] = json.loads(storage.read_text(stats_uri))
    return loaded


def _read_postings(storage: Storage, uris: list[str]) -> Any:
    uri = next(u for u in uris if u.endswith("bm25-postings.parquet"))
    with storage.open(uri, "rb") as fh:
        return pq.read_table(fh)  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Off by default: byte-identical, no files (ADR-0017/0049)
# ---------------------------------------------------------------------------


def test_no_bm25_block_is_byte_identical_and_writes_no_files(tmp_path: Path) -> None:
    records = [_chunk("alpha beta", "c1", 0), _chunk("gamma delta", "c2", 1)]

    plain_store = f"file://{tmp_path}/plain"
    plain_storage = Storage.for_uri(plain_store)
    plain_uris = JsonlParquetExport({"basename": "corpus"}).export(records, plain_storage, plain_store)

    off_store = f"file://{tmp_path}/off"
    off_storage = Storage.for_uri(off_store)
    off_uris = JsonlParquetExport({"basename": "corpus"}).export(
        [_chunk("alpha beta", "c1", 0), _chunk("gamma delta", "c2", 1)], off_storage, off_store
    )

    for plain_uri, off_uri in zip(sorted(plain_uris), sorted(off_uris), strict=True):
        assert plain_storage.read_bytes(plain_uri) == off_storage.read_bytes(off_uri)
    assert not any(u.rsplit("/", 1)[-1].startswith("bm25-") for u in off_uris)
    assert not off_storage.exists(f"{off_store}/bm25-stats.json")
    assert not off_storage.exists(f"{off_store}/bm25-postings.parquet")


# ---------------------------------------------------------------------------
# Wired: the two documented files, with the documented shapes
# ---------------------------------------------------------------------------


def test_bm25_emits_stats_and_postings_files(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    records = [_chunk("alpha beta alpha", "c1", 0), _chunk("gamma", "c2", 1)]
    uris = JsonlParquetExport(_bm25_config()).export(records, storage, store)

    assert any(u.endswith("bm25-stats.json") for u in uris)
    assert any(u.endswith("bm25-postings.parquet") for u in uris)

    stats = _read_stats(storage, uris)
    assert stats["artifact"] == "latence.bm25.termstats"
    assert stats["version"] == 1
    assert stats["tokenizer"] == "tokenizer.regex"
    assert stats["doc_count"] == 2
    assert stats["total_tokens"] == 4  # "alpha beta alpha" (3) + "gamma" (1)
    assert stats["avgdl"] == 2.0
    terms = {t["term"]: t for t in stats["terms"]}
    assert terms["alpha"]["df"] == 1 and terms["gamma"]["df"] == 1
    assert [t["term"] for t in stats["terms"]] == sorted(terms)  # ascending

    table = _read_postings(storage, uris)
    assert table.column_names == ["record_id", "doc_len", "terms", "term_frequencies"]
    assert table.column("record_id").to_pylist() == ["c1", "c2"]
    assert table.column("doc_len").to_pylist() == [3, 1]
    # c1's "alpha" has tf 2, "beta" tf 1 (terms ascending, tfs parallel).
    assert table.column("terms")[0].as_py() == ["alpha", "beta"]
    assert table.column("term_frequencies")[0].as_py() == [2, 1]


def test_stats_match_an_independent_reference_tokenization(tmp_path: Path) -> None:
    """The AC: the emitted term statistics equal those from the same reference tokenizer, computed
    independently of the Export."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    texts = {"c1": "The quick brown fox.", "c2": "The fox jumps; the FOX runs."}
    records = [_chunk(texts["c1"], "c1", 0), _chunk(texts["c2"], "c2", 1)]
    uris = JsonlParquetExport(_bm25_config()).export(records, storage, store)

    reference = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    reference.add_batch([("c1", texts["c1"]), ("c2", texts["c2"])])
    expected = reference.stats().to_artifact()

    assert _read_stats(storage, uris) == expected


def test_bm25_signal_is_over_redacted_text(tmp_path: Path) -> None:
    """The statistics are over masked_content, never the raw text (no PII re-leak, ADR-0053)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    masked = "Contact EMAIL today today"
    uris = JsonlParquetExport(_bm25_config()).export([_redaction(masked)], storage, store)

    reference = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    reference.add("r1", masked)
    assert _read_stats(storage, uris) == reference.stats().to_artifact()
    # The masked token is present; a raw-PII token that the corpus removed would not be.
    terms = {t["term"] for t in _read_stats(storage, uris)["terms"]}
    assert "email" in terms and "today" in terms


def test_empty_corpus_writes_valid_empty_artifact(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    uris = JsonlParquetExport(_bm25_config()).export([], storage, store)

    stats = _read_stats(storage, uris)
    assert stats["doc_count"] == 0 and stats["avgdl"] == 0 and stats["terms"] == []
    table = _read_postings(storage, uris)
    assert table.num_rows == 0
    assert table.column_names == ["record_id", "doc_len", "terms", "term_frequencies"]


# ---------------------------------------------------------------------------
# Determinism + additivity with the other signals
# ---------------------------------------------------------------------------


def test_bm25_export_is_deterministic(tmp_path: Path) -> None:
    records = [_chunk("alpha beta alpha", "c1", 0), _chunk("gamma delta", "c2", 1)]
    outputs: list[dict[str, bytes]] = []
    for run in ("a", "b"):
        store = f"file://{tmp_path}/{run}"
        storage = Storage.for_uri(store)
        uris = JsonlParquetExport(_bm25_config()).export(list(records), storage, store)
        outputs.append({u.rsplit("/", 1)[-1]: storage.read_bytes(u) for u in uris})
    assert outputs[0] == outputs[1] and "bm25-stats.json" in outputs[0]


def test_bm25_is_additive_with_the_corpus_columns(tmp_path: Path) -> None:
    """Wiring BM25 leaves records.jsonl/records.parquet byte-identical to a plain export."""
    records = [_chunk("alpha beta", "c1", 0), _chunk("gamma", "c2", 1)]

    plain_store = f"file://{tmp_path}/plain"
    plain_storage = Storage.for_uri(plain_store)
    plain_uris = JsonlParquetExport({"basename": "corpus"}).export(records, plain_storage, plain_store)

    bm_store = f"file://{tmp_path}/bm"
    bm_storage = Storage.for_uri(bm_store)
    bm_uris = JsonlParquetExport(_bm25_config()).export(
        [_chunk("alpha beta", "c1", 0), _chunk("gamma", "c2", 1)], bm_storage, bm_store
    )

    for name in ("corpus.jsonl", "corpus.parquet"):
        plain_uri = next(u for u in plain_uris if u.endswith(name))
        bm_uri = next(u for u in bm_uris if u.endswith(name))
        assert plain_storage.read_bytes(plain_uri) == bm_storage.read_bytes(bm_uri)


# ---------------------------------------------------------------------------
# Wiring invariant bites (loud failure, never a silently-empty artifact)
# ---------------------------------------------------------------------------


def test_misnamed_bm25_provider_fails_loudly() -> None:
    """A non-Tokenizer provider named as bm25 raises (TypeError); an unknown name raises KeyError."""
    with pytest.raises(TypeError):
        JsonlParquetExport({"bm25": {"provider": "embedding.hashing", "config": {}}})
    with pytest.raises(KeyError):
        JsonlParquetExport({"bm25": {"provider": "tokenizer.nonexistent", "config": {}}})


def test_whitespace_tokenizer_selects_a_different_analysis(tmp_path: Path) -> None:
    """Swapping the Tokenizer provider changes the artifact — the seam is real, through the Export."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    config = {"basename": "corpus", "bm25": {"provider": "tokenizer.whitespace", "config": {}}}
    uris = JsonlParquetExport(config).export([_chunk("foo, foo bar.", "c1")], storage, store)
    stats = _read_stats(storage, uris)
    assert stats["tokenizer"] == "tokenizer.whitespace"
    assert {t["term"] for t in stats["terms"]} == {"foo,", "foo", "bar."}

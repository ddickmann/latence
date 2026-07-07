"""Memory scaling of the incremental Export, validated at reduced scale (ADR-0033 / #63).

The whole point of the streamed Export is that its peak transient is O(batch), not O(corpus):
the old ``from_pylist(whole)`` built one Python list + one Arrow table sized to the entire
corpus (the peak G1 Lane 6 measured). This test proves the bound at reduced scale by feeding
the records as a *generator* (so the input is never pre-materialized) and measuring peak
allocation with ``tracemalloc`` for a corpus and a 4x-larger corpus: a whole-corpus buffer
would grow peak ~4x, whereas the streamed path holds only a bounded batch, so peak stays close
to flat. The real 100k/1M peak-RSS numbers still need a dedicated perf rig (flagged in #63 /
ADR-0033) — this is the deterministic, in-suite scaling assertion, not the perf-rig number.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator
from typing import TYPE_CHECKING

import latence_core.stages.export as export_mod
import pytest
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path


def _chunk_stream(n: int) -> Iterator[ChunkRecord]:
    """Yield ``n`` chunks lazily — the corpus is never held as a whole list."""
    for i in range(n):
        text = f"chunk number {i} with a little body text to give each row some weight"
        prov = Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=100,
            document_id=f"sha256:d{i}",
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=1,
        )
        yield ChunkRecord(
            record_id=f"c{i}",
            document_record_id=f"d{i}",
            index=0,
            provenance=prov,
            classification=Classification(language="en", sensitivity="public"),
            content=text,
            token_count=len(text.split()),
        )


def _peak_export_bytes(n: int, out: str, storage: Storage) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    JsonlParquetExport().export(_chunk_stream(n), storage, out)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_export_peak_memory_is_bounded_not_corpus_linear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a small flush batch so the batch cost is a small constant relative to the corpus,
    # and use corpora that are large multiples of it — so a whole-corpus buffer would dominate
    # peak (~4x growth) while the streamed path's peak is the constant per-batch working set.
    monkeypatch.setattr(export_mod, "_EXPORT_BATCH_ROWS", 200)
    storage = Storage.for_uri(f"file://{tmp_path}")
    small = 2000
    large = small * 4

    peak_small = _peak_export_bytes(small, f"file://{tmp_path}/small", storage)
    peak_large = _peak_export_bytes(large, f"file://{tmp_path}/large", storage)

    # A whole-corpus buffer would make peak grow ~linearly with the corpus (≈4x here). The
    # streamed path holds only a bounded batch, so peak must stay well under linear growth.
    # Generous margin (2x, not the 4x a linear buffer would hit) keeps the assertion robust to
    # allocator noise + the ParquetWriter's own bookkeeping while still failing hard if the
    # Export reverts to buffering the whole corpus.
    assert peak_large < peak_small * 2, (
        f"Export peak grew near-linearly with the corpus "
        f"(small={peak_small} bytes, large(4x)={peak_large} bytes): the O(batch) bound broke."
    )

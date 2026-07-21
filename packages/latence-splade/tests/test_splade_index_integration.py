"""Integration: the core SPLADE-index driver drives the REAL ``SpladeCompositionEmbedder`` (§4.2).

The dogfood finding (tutorial §4.2) required that ``latence retrieval splade-index`` generate REAL
SPLADE sparse vectors via :class:`~latence_splade.provider.SpladeCompositionEmbedder` — not a stub.
The CLI wiring + driver contract are pinned in ``latence-core``'s ``test_splade_index.py`` against a
Protocol fake; this test closes the loop by running the core driver
(:func:`latence_core.splade_index.build_splade_index`) over the ACTUAL Provider the CLI resolves,
with the provider's own faithful fake *model* seam (so the composition runs for real, offline, with
no ``naver/splade-v3`` weights download in CI). Production swaps only the model seam for the real
``transformers`` checkpoint — the driver → provider → composition path this exercises is identical.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from latence_core.splade_index import build_splade_index
from latence_splade.provider import SpladeCompositionEmbedder

_FAKE_VOCAB = 64
_TOKEN = re.compile(r"[a-z0-9]+")


def _tid(token: str) -> int:
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _FAKE_VOCAB


class _FakeSpladeModel:
    """The faithful offline model seam (same contract as ``TransformersSpladeModel``)."""

    @property
    def vocab_size(self) -> int:
        return _FAKE_VOCAB

    def term_weights(self, texts: Any) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        for text in texts:
            terms: dict[int, float] = {}
            for token in _TOKEN.findall(text.lower()):
                terms[_tid(token)] = terms.get(_tid(token), 0.0) + 1.0
            out.append(terms)
        return out

    def term_ids(self, term: str) -> list[int]:
        seen: set[int] = set()
        ids: list[int] = []
        for token in _TOKEN.findall(term.lower()):
            idx = _tid(token)
            if idx not in seen:
                seen.add(idx)
                ids.append(idx)
        return ids


def _write_export(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    export = tmp_path / "export"
    export.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"record_id": [r for r, _ in rows], "content": [t for _, t in rows]}
    )
    pq.write_table(table, export / "records.parquet")  # type: ignore[no-untyped-call]
    return export


def test_driver_indexes_a_corpus_through_the_real_splade_provider(tmp_path: Path) -> None:
    export = _write_export(
        tmp_path,
        [
            ("sha256:a", "transport header fields"),
            ("sha256:b", "status code semantics"),
            ("sha256:c", "   "),  # whitespace-only → empty composed vector → counted, not written
        ],
    )
    out = tmp_path / "export" / "splade"

    embedder = SpladeCompositionEmbedder({"model": "naver/splade-v3"}, model=_FakeSpladeModel())
    result = build_splade_index(
        export_dir=str(export),
        out_dir=str(out),
        embedder=embedder,
        model_id="naver/splade-v3",
        weights_license="CC-BY-NC-SA-4.0",
    )

    assert result.chunks_total == 3
    assert result.chunks_indexed == 2
    assert result.chunks_empty == 1
    assert result.vocab_size == _FAKE_VOCAB
    assert result.license_verified is False
    assert result.operator_accepted is True

    index_rows = pq.read_table(out / "index.parquet").to_pylist()  # type: ignore[no-untyped-call]
    assert {row["record_id"] for row in index_rows} == {"sha256:a", "sha256:b"}
    for row in index_rows:
        assert row["indices"] == sorted(row["indices"])
        assert len(row["indices"]) == len(row["values"]) >= 1

"""Regression: `latence retrieval splade-index` builds a real per-chunk SPLADE sparse index (§4.2).

The dogfood finding at tutorial §4.2 was that the ``latence retrieval splade-index`` entry point did
not exist at all — the ``latence`` CLI had no ``retrieval`` command group — so no REAL SPLADE-v3
sparse vectors could be generated for a corpus. These tests pin the built command + driver:

* the driver (:func:`latence_core.splade_index.build_splade_index`) reads an Export's
  ``records.parquet``, encodes every chunk through an injected SparseEmbedder, writes only the
  non-empty vectors to ``index.parquet``, counts the empty one, and records the honest license
  manifest;
* the CLI ``latence retrieval splade-index`` exists, REFUSES without
  ``--accept-noncommercial-license``, and on the happy path resolves the ``sparse.splade`` Provider
  through the registry and writes the artifact.

They use a faithful fake SparseEmbedder (the Protocol's offline second adapter) so the read/emit/
license logic is exercised without weights — the CLI resolves the *real* provider through the
registry, which is monkeypatched here to the fake class so no checkpoint downloads in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from latence_core.capability import ProviderRegistry, SparseVector
from latence_core.cli import app
from latence_core.splade_index import SpladeIndexError, build_splade_index
from typer.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

runner = CliRunner()


class _FakeSparseEmbedder:
    """A faithful offline SparseEmbedder: one deterministic term per lower-cased word token.

    Faithful to the contract the real ``sparse.splade`` Provider satisfies: one
    :class:`SparseVector` per input text in order, canonical ascending indices, and an EMPTY text
    composes to the empty vector (no surviving terms) — exactly the empty-vector case the index must
    count-but-not-write. Constructed with a single positional config dict, mirroring
    ``SpladeCompositionEmbedder.__init__(config)`` so the CLI's ``embedder_cls({"model": ...})`` call
    works against it unchanged.
    """

    _VOCAB = 128

    def __init__(self, config: dict[str, object] | None = None) -> None:
        self._config = config or {}

    @property
    def vocab_size(self) -> int:
        return self._VOCAB

    def encode_sparse(self, texts: Iterable[str]) -> Iterator[SparseVector]:
        for text in texts:
            terms: dict[int, float] = {}
            for token in text.lower().split():
                idx = (hash(token) & 0x7FFFFFFF) % self._VOCAB
                terms[idx] = terms.get(idx, 0.0) + 1.0
            yield SparseVector.from_terms(terms)


def _write_export(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    """Write a real ``records.parquet`` (``record_id`` + ``content``) and return its directory."""
    export = tmp_path / "export"
    export.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "record_id": [rid for rid, _ in rows],
            "content": [text for _, text in rows],
        }
    )
    pq.write_table(table, export / "records.parquet")  # type: ignore[no-untyped-call]
    return export


def _read_index(index_path: Path) -> list[dict[str, object]]:
    rows = pq.read_table(index_path).to_pylist()  # type: ignore[no-untyped-call]
    return cast("list[dict[str, object]]", rows)


def test_build_splade_index_writes_per_chunk_vectors_and_manifest(tmp_path: Path) -> None:
    export = _write_export(
        tmp_path,
        [
            ("sha256:a", "transport header fields"),
            ("sha256:b", "status code semantics"),
            ("sha256:c", ""),  # empty content → empty sparse vector → counted, not written
        ],
    )
    out = tmp_path / "export" / "splade"

    result = build_splade_index(
        export_dir=str(export),
        out_dir=str(out),
        embedder=_FakeSparseEmbedder(),
        model_id="naver/splade-v3",
        weights_license="CC-BY-NC-SA-4.0",
    )

    # Counts are honest: every chunk is either indexed or counted empty, never dropped.
    assert result.chunks_total == 3
    assert result.chunks_indexed == 2
    assert result.chunks_empty == 1

    index_rows = _read_index(out / "index.parquet")
    assert {row["record_id"] for row in index_rows} == {"sha256:a", "sha256:b"}
    for row in index_rows:
        indices = cast("list[int]", row["indices"])
        values = cast("list[float]", row["values"])
        assert indices == sorted(indices), "sparse indices must be canonical ascending"
        assert len(indices) == len(values)
        assert len(indices) >= 1

    # Manifest carries the honest ADR-0012 license provenance the tutorial §4.2 look-for demands.
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"] == "naver/splade-v3"
    assert manifest["weights_license"] == "CC-BY-NC-SA-4.0"
    assert manifest["license_verified"] is False
    assert manifest["operator_accepted"] is True
    assert manifest["chunks_indexed"] == 2
    assert manifest["chunks_empty"] == 1
    assert manifest["vocab_size"] == _FakeSparseEmbedder._VOCAB


def test_build_splade_index_refuses_a_missing_export(tmp_path: Path) -> None:
    with pytest.raises(SpladeIndexError, match="No Export corpus"):
        build_splade_index(
            export_dir=str(tmp_path / "nope"),
            out_dir=str(tmp_path / "out"),
            embedder=_FakeSparseEmbedder(),
            model_id="naver/splade-v3",
            weights_license="CC-BY-NC-SA-4.0",
        )


def test_splade_index_cli_refuses_without_the_license_flag(tmp_path: Path) -> None:
    export = _write_export(tmp_path, [("sha256:a", "alpha")])
    out = tmp_path / "splade"

    result = runner.invoke(
        app,
        [
            "retrieval",
            "splade-index",
            "--export",
            str(export),
            "--out",
            str(out),
            "--model",
            "naver/splade-v3",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "accept-noncommercial-license" in result.output
    # The refusal is BEFORE any work: no artifact is written.
    assert not out.exists()


def test_splade_index_cli_happy_path_builds_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_export(
        tmp_path, [("sha256:a", "transport header fields"), ("sha256:b", "status codes")]
    )
    out = tmp_path / "export" / "splade"

    # The CLI resolves the REAL ``sparse.splade`` Provider through the registry; monkeypatch that
    # resolution to the faithful fake class so the wiring is exercised end-to-end without weights.
    monkeypatch.setattr(ProviderRegistry, "load", lambda self, name: _FakeSparseEmbedder)

    result = runner.invoke(
        app,
        [
            "retrieval",
            "splade-index",
            "--export",
            str(export),
            "--out",
            str(out),
            "--model",
            "naver/splade-v3",
            "--accept-noncommercial-license",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2/2 chunks indexed" in result.output
    assert "weights_license=CC-BY-NC-SA-4.0" in result.output
    assert "license_verified=false" in result.output
    assert "operator_accepted=true" in result.output
    assert (out / "index.parquet").exists()
    assert (out / "manifest.json").exists()
    assert len(_read_index(out / "index.parquet")) == 2

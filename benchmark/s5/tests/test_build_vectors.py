# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``build_vectors`` — schema match against a real S3 export, resumability, idempotence.

The heavy part (a 1.2 GB Granite checkpoint) is injected as an ``embedder_factory``, so these tests
exercise the artifact-production logic — schema, ordering, additivity, checkpointing — on a
hand-built miniature export in a tmp dir. The stand-in is a *test double for the model only*: it
never touches a real export, and the two live exports are built with the real embedder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from benchmark.s5 import build_vectors as bv
from benchmark.s5.corpus import CONTENT_ONLY_VECTORS, load_export, read_vector_sidecar
from latence_core.bm25 import Bm25Accumulator, RegexWordTokenizer
from latence_core.stages.export import JsonlParquetExport

#: A real S3 export, when this checkout has one (they are gitignored — dataset-derived data stays
#: local), so the schema assertion can be made against the artifact it must match rather than only
#: against the contract that produced it.
S3_EXPORT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "sota-corpus"
    / "s3-exports"
    / "ohr_bench"
    / "corpus.parquet"
)

TEXTS = [
    "Windows Media Encoder",
    "Windows Media Encoder (WME) is a discontinued freeware media encoder from Microsoft.",
    "The download page reports that it is not supported on Windows 7.",
    "Cinema of Cameroon: the first film shot in the country was made in 1926.",
    "Müller & Söhne, für die Größe des Marktes, taxed at 12%.",
    "",
    "A seventh chunk, deliberately short.",
    "An eighth chunk so the corpus does not divide evenly into shards.",
    "Nine.",
]


class StubEmbedder:
    """A deterministic 768-dim stand-in for the Granite checkpoint (tests only).

    Vectors are a sha256-seeded unit vector per text, so they are reproducible, distinct, and
    unit-norm exactly like the real embedder's — which is all the artifact-shape assertions need.
    Counts the texts it saw so the resumability tests can prove work was skipped.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts: list[str]) -> Any:
        self.seen.extend(texts)
        out = np.zeros((len(texts), bv.EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(bv.EMBED_DIM).astype(np.float32)
            out[row] = vector / np.linalg.norm(vector)
        return out


class ExplodingEmbedder(StubEmbedder):
    """Embeds ``limit`` texts, then raises — the interrupted-run stand-in."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit

    def encode(self, texts: list[str]) -> Any:
        if len(self.seen) >= self._limit:
            msg = "boom"
            raise RuntimeError(msg)
        return super().encode(texts)


def _header(index: int) -> dict[str, Any]:
    """A populated W13 ContextHeader — the S3 corpora's shape (200/200 sampled rows carry one)."""
    return {
        "entities": [f"Entity{index}", "Microsoft"],
        "entity_types": ["PRODUCT", "ORG"],
        "neighbor_triples": [f"Entity{index} —made_by→ Microsoft"],
    }


def _row(index: int, text: str, *, with_header: bool = False) -> dict[str, Any]:
    """One export-shaped JSONL row (the fields ``_corpus_text`` / the base schema read)."""
    record_id = f"sha256:{index:064x}#chunk-{index}"
    return {
        "record_id": record_id,
        "schema_version": 20,
        "provenance": {
            "char_start": index * 100,
            "char_end": index * 100 + len(text),
            "file_name": f"p{index:04d}.txt",
            "schema_version": 20,
        },
        "classification": {"category": None, "language": "en", "sensitivity": "internal"},
        "content": text,
        "masked_content": None,
        "context_header": _header(index) if with_header else None,
        "media_type": "text/markdown",
    }


def _write_export(root: Path, rows: list[dict[str, Any]]) -> Path:
    """Materialise ``rows`` as the two files an M6 export has: ``corpus.jsonl`` + the base table."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    schema = JsonlParquetExport._parquet_schema(False)  # noqa: SLF001
    table = pa.Table.from_pylist(
        [
            {
                "record_id": row["record_id"],
                "schema_version": row["schema_version"],
                "provenance": json.dumps(row["provenance"], sort_keys=True),
                "classification": json.dumps(row["classification"], sort_keys=True),
                "content": row["content"],
                "media_type": row["media_type"],
            }
            for row in rows
        ],
        schema=schema,
    )
    pq.write_table(table, root / "corpus.parquet")  # type: ignore[no-untyped-call]
    return root


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """A miniature export with exactly the two files the M6 exports have."""
    return _write_export(tmp_path / "export", [_row(i, t) for i, t in enumerate(TEXTS)])


@pytest.fixture
def headered_export_dir(tmp_path: Path) -> Path:
    """The S3 corpora's shape: every row carries a populated ContextHeader.

    The plain fixture mirrors the reused M6 exports (``context_header: null``), where the two text
    variants are byte-identical — so it cannot tell the two code paths apart, and every assertion
    about the ablation has to be made here instead.
    """
    return _write_export(
        tmp_path / "s3-export", [_row(i, t, with_header=True) for i, t in enumerate(TEXTS)]
    )


def _plan(export_dir: Path, **overrides: Any) -> bv.BuildPlan:
    defaults: dict[str, Any] = {
        "export_dir": export_dir,
        "device": "cpu",
        "batch_size": 4,
        "shard_rows": 4,
        "write_jsonl": False,
        "force": False,
        "verify_only": False,
        "spot_checks": 3,
    }
    defaults.update(overrides)
    return bv.BuildPlan(**defaults)


def _build(export_dir: Path, embedder: StubEmbedder, **overrides: Any) -> dict[str, Any]:
    return bv.build(
        _plan(export_dir, **overrides),
        embedder_factory=lambda: embedder,
        log=lambda _message: None,
    )


# -- schema ------------------------------------------------------------------------------------


def test_produced_schema_is_the_export_contract_plus_embedding(export_dir: Path) -> None:
    """The rewritten table is the original schema with the Export's own ``embedding`` field."""
    original = pq.read_schema(export_dir / "corpus.parquet")  # type: ignore[no-untyped-call]
    _build(export_dir, StubEmbedder())
    produced = pq.read_schema(export_dir / "corpus.parquet")  # type: ignore[no-untyped-call]
    assert produced.names == [*original.names, "embedding"]
    assert produced.field("embedding").type == pa.list_(pa.float64())


@pytest.mark.skipif(not S3_EXPORT.is_file(), reason="no local S3 export to compare against")
def test_schema_matches_a_real_s3_export_for_the_shared_columns(export_dir: Path) -> None:
    """Every column this tool produces has the S3 exports' name, order and Arrow type."""
    _build(export_dir, StubEmbedder())
    produced = pq.read_schema(export_dir / "corpus.parquet")  # type: ignore[no-untyped-call]
    reference = pq.read_schema(S3_EXPORT)  # type: ignore[no-untyped-call]
    shared = [name for name in produced.names if name in reference.names]
    assert shared == produced.names, "produced a column the S3 exports do not have"
    assert shared == [name for name in reference.names if name in produced.names], "column reorder"
    for name in shared:
        assert produced.field(name).type == reference.field(name).type, name


def test_bm25_postings_schema_is_the_exports_own(export_dir: Path) -> None:
    _build(export_dir, StubEmbedder())
    produced = pq.read_schema(export_dir / "bm25-postings.parquet")  # type: ignore[no-untyped-call]
    assert produced == JsonlParquetExport._bm25_postings_schema()  # noqa: SLF001


# -- additivity --------------------------------------------------------------------------------


def test_pre_existing_columns_and_row_order_survive(export_dir: Path) -> None:
    """The six base columns, the ids and the row order are carried through untouched."""
    before = pq.read_table(export_dir / "corpus.parquet")  # type: ignore[no-untyped-call]
    _build(export_dir, StubEmbedder())
    backup = export_dir / "corpus.parquet.pre-vectors.bak"
    assert backup.is_file(), "the original must be kept"
    after = pq.read_table(export_dir / "corpus.parquet")  # type: ignore[no-untyped-call]
    assert after.num_rows == before.num_rows
    for name in before.schema.names:
        assert after.column(name).equals(before.column(name)), name
    assert pq.read_table(backup).equals(before)  # type: ignore[no-untyped-call]


def test_corpus_jsonl_is_untouched_by_default_and_gains_vectors_on_opt_in(
    export_dir: Path,
) -> None:
    jsonl = export_dir / "corpus.jsonl"
    original = jsonl.read_bytes()
    _build(export_dir, StubEmbedder())
    assert jsonl.read_bytes() == original, "the default must not rewrite corpus.jsonl"

    _build(export_dir, StubEmbedder(), write_jsonl=True, force=True)
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(TEXTS)
    assert all(len(row["embedding"]) == bv.EMBED_DIM for row in rows)
    matrix = pq.read_table(export_dir / "corpus.parquet").column("embedding").to_pylist()  # type: ignore[no-untyped-call]
    assert rows[3]["embedding"] == pytest.approx(matrix[3])


# -- BM25 --------------------------------------------------------------------------------------


def test_bm25_artifacts_equal_the_frameworks_own_accumulator(export_dir: Path) -> None:
    """The emitted stats are byte-for-byte what ``Bm25Accumulator`` produces over the same text."""
    _build(export_dir, StubEmbedder())
    expected_acc = Bm25Accumulator(RegexWordTokenizer(), bv.BM25_TOKENIZER)
    for text in TEXTS:
        expected_acc.add("x", text)
    expected = expected_acc.stats().to_artifact()
    produced = json.loads((export_dir / "bm25-stats.json").read_text(encoding="utf-8"))
    assert produced["terms"] == expected["terms"]
    assert produced["avgdl"] == expected["avgdl"]
    assert produced["doc_count"] == len(TEXTS)
    assert produced["tokenizer"] == bv.BM25_TOKENIZER

    postings = pq.read_table(export_dir / "bm25-postings.parquet")  # type: ignore[no-untyped-call]
    ids = pq.read_table(export_dir / "corpus.parquet", columns=["record_id"]).column("record_id")  # type: ignore[no-untyped-call]
    assert postings.column("record_id").equals(ids)


# -- resumability + idempotence ------------------------------------------------------------------


def test_resumes_after_an_interrupted_embed(export_dir: Path) -> None:
    """A crashed run leaves complete shards behind; the re-run embeds only what is missing."""
    with pytest.raises(RuntimeError, match="boom"):
        _build(export_dir, ExplodingEmbedder(limit=8))
    work = export_dir / bv.WORK_DIRNAME
    assert sorted(p.name for p in work.glob("shard-*.npy")) == [
        "shard-000000.npy",
        "shard-000001.npy",
    ]
    assert not (export_dir / "bm25-stats.json").exists(), "no artifact from a failed run"

    resumed = StubEmbedder()
    report = _build(export_dir, resumed, spot_checks=0)
    assert len(resumed.seen) == 1, "only the one un-checkpointed row may be re-embedded"
    assert report["rows"] == len(TEXTS)


def test_second_run_is_a_no_op(export_dir: Path) -> None:
    """Re-running a complete export embeds nothing and leaves every artifact byte-identical."""
    _build(export_dir, StubEmbedder())
    fingerprint = {
        name: hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
        for name in ("corpus.parquet", "corpus.jsonl", "bm25-postings.parquet", "bm25-stats.json")
    }
    second = StubEmbedder()
    _build(export_dir, second, spot_checks=0)
    assert second.seen == []
    for name, digest in fingerprint.items():
        assert hashlib.sha256((export_dir / name).read_bytes()).hexdigest() == digest, name


def test_force_rebuild_reproduces_the_same_vectors(export_dir: Path) -> None:
    """``--force`` discards the checkpoints and re-embeds — to the same, deterministic result."""
    _build(export_dir, StubEmbedder())
    first = pq.read_table(export_dir / "corpus.parquet").column("embedding").to_pylist()  # type: ignore[no-untyped-call]
    rebuilt = StubEmbedder()
    _build(export_dir, rebuilt, force=True, spot_checks=0)
    assert len(rebuilt.seen) == len(TEXTS)
    second = pq.read_table(export_dir / "corpus.parquet").column("embedding").to_pylist()  # type: ignore[no-untyped-call]
    assert second == first


def test_a_moved_record_id_invalidates_its_shard(export_dir: Path) -> None:
    """A checkpoint is reused only while it still covers exactly the same ids, in order."""
    with pytest.raises(RuntimeError, match="boom"):
        _build(export_dir, ExplodingEmbedder(limit=4))
    lines = (export_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    rows[0], rows[1] = rows[1], rows[0]
    (export_dir / "corpus.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    resumed = StubEmbedder()
    with pytest.raises(ValueError, match="disagree on row order"):
        _build(export_dir, resumed)
    assert len(resumed.seen) == len(TEXTS), "the reordered shard must be recomputed"


# -- verification ---------------------------------------------------------------------------------


def test_verify_reports_the_numbers_and_rank1_self_retrieval(export_dir: Path) -> None:
    report = _build(export_dir, StubEmbedder())
    assert report["rows"] == len(TEXTS)
    assert report["embedding_dim"] == bv.EMBED_DIM
    assert report["all_rows_768"] is True
    assert report["nan_or_inf"] == 0
    assert report["norm_min"] == pytest.approx(1.0, abs=1e-5)
    assert report["rows_unchanged"] is True
    assert report["bm25"]["postings_cover_corpus"] is True
    assert report["bm25"]["doc_count_matches"] is True
    assert report["self_retrieval"]["rank1_hits"] == report["self_retrieval"]["samples"]


def test_verify_only_does_not_rebuild(export_dir: Path) -> None:
    _build(export_dir, StubEmbedder())
    digest = hashlib.sha256((export_dir / "corpus.parquet").read_bytes()).hexdigest()
    watcher = StubEmbedder()
    report = bv.build(
        _plan(export_dir, verify_only=True, spot_checks=2),
        embedder_factory=lambda: watcher,
        log=lambda _message: None,
    )
    assert report["rows"] == len(TEXTS)
    assert len(watcher.seen) == 2, "verification re-embeds only its spot-check samples"
    assert hashlib.sha256((export_dir / "corpus.parquet").read_bytes()).hexdigest() == digest


def test_missing_export_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus.jsonl"):
        bv.build(_plan(tmp_path), embedder_factory=StubEmbedder, log=lambda _m: None)


def test_resolve_device_honours_an_explicit_choice() -> None:
    assert bv.resolve_device("cpu") == "cpu"
    assert bv.resolve_device("cuda") == "cuda"
    assert bv.resolve_device("auto") in {"cpu", "mps", "cuda"}


# -- the content-only vector sidecar ---------------------------------------------------------------


def _corpus_rows(export_dir: Path) -> list[dict[str, Any]]:
    return list(bv.iter_rows(export_dir / "corpus.jsonl"))


def _ids(export_dir: Path) -> list[str]:
    return [str(row["record_id"]) for row in _corpus_rows(export_dir)]


def test_default_build_writes_no_sidecar(export_dir: Path) -> None:
    """The default path is unchanged: corpus.parquet gains the column, no extra file appears."""
    _build(export_dir, StubEmbedder())
    assert not (export_dir / CONTENT_ONLY_VECTORS).exists()
    assert "embedding" in pq.read_schema(export_dir / "corpus.parquet").names  # type: ignore[no-untyped-call]
    assert not (export_dir / f"{bv.WORK_DIRNAME}-{bv.CONTENT_VARIANT}").exists()


def test_sidecar_round_trips_the_exact_vectors(headered_export_dir: Path) -> None:
    """write → load gives back the same matrix, in the same row order, under the same ids."""
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(
        headered_export_dir,
        StubEmbedder(),
        with_context_header=False,
        vectors_out=sidecar,
    )
    expected = StubEmbedder().encode(
        [row["content"] for row in _corpus_rows(headered_export_dir)]
    )
    ids, matrix = read_vector_sidecar(sidecar)
    assert ids == _ids(headered_export_dir)
    assert matrix.dtype == np.float32
    assert matrix.shape == (len(TEXTS), bv.EMBED_DIM)
    assert np.array_equal(matrix, expected)


def test_the_sidecar_carries_only_ids_and_vectors(headered_export_dir: Path) -> None:
    """No content duplication — the sidecar is one column's worth of data, nothing else."""
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)
    schema = pq.read_schema(sidecar)  # type: ignore[no-untyped-call]
    assert schema.names == ["record_id", "embedding"]
    assert schema.field("embedding").type == pa.list_(pa.float32(), bv.EMBED_DIM)
    metadata = schema.metadata or {}
    assert metadata[b"latence.vectors.variant"] == bv.CONTENT_VARIANT.encode()
    assert metadata[b"latence.vectors.dim"] == str(bv.EMBED_DIM).encode()


def test_a_sidecar_build_leaves_the_enriched_vectors_and_every_other_artifact_untouched(
    headered_export_dir: Path,
) -> None:
    """The blocker this exists for: the S3 exports' header-enriched column must survive intact."""
    _build(headered_export_dir, StubEmbedder())  # the S3 state: enriched vectors already in place
    enriched = pq.read_table(headered_export_dir / "corpus.parquet")  # type: ignore[no-untyped-call]
    fingerprint = {
        name: hashlib.sha256((headered_export_dir / name).read_bytes()).hexdigest()
        for name in ("corpus.parquet", "corpus.jsonl", "bm25-postings.parquet", "bm25-stats.json")
    }

    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    report = _build(
        headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar
    )

    for name, digest in fingerprint.items():
        got = hashlib.sha256((headered_export_dir / name).read_bytes()).hexdigest()
        assert got == digest, f"{name} was rewritten by a --vectors-out build"
    assert report["vectors"] == str(sidecar)
    assert report["variant"] == bv.CONTENT_VARIANT
    # …and the sidecar is the OTHER variant, not a second copy of the enriched vectors.
    _, content_only = read_vector_sidecar(sidecar, expected_ids=_ids(headered_export_dir))
    header_vectors = np.asarray(enriched.column("embedding").to_pylist(), dtype=np.float32)
    assert not np.allclose(content_only, header_vectors)


def test_a_sidecar_build_on_a_bare_export_touches_nothing_else(export_dir: Path) -> None:
    """No corpus rewrite, no backup, no BM25: --vectors-out produces exactly one file."""
    before = (export_dir / "corpus.parquet").read_bytes()
    sidecar = export_dir / CONTENT_ONLY_VECTORS
    report = _build(export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)
    assert (export_dir / "corpus.parquet").read_bytes() == before
    assert not (export_dir / f"corpus.parquet{bv.BACKUP_SUFFIX}").exists()
    assert not (export_dir / "bm25-stats.json").exists()
    assert "bm25" not in report, "a sidecar build must not claim a BM25 verdict it did not produce"
    assert report["rows"] == len(TEXTS)
    assert report["self_retrieval"]["rank1_hits"] == report["self_retrieval"]["samples"]


def test_a_second_sidecar_build_is_a_no_op(headered_export_dir: Path) -> None:
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    second = StubEmbedder()
    _build(
        headered_export_dir,
        second,
        with_context_header=False,
        vectors_out=sidecar,
        spot_checks=0,
    )
    assert second.seen == []
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == digest


def test_the_two_variants_do_not_share_checkpoints(headered_export_dir: Path) -> None:
    """Identical ids, different text: a shared shard store would silently cross-feed vectors."""
    _build(headered_export_dir, StubEmbedder())
    content = StubEmbedder()
    _build(
        headered_export_dir,
        content,
        with_context_header=False,
        vectors_out=headered_export_dir / CONTENT_ONLY_VECTORS,
        spot_checks=0,
    )
    assert content.seen == list(TEXTS), "the content-only variant reused header checkpoints"
    assert (headered_export_dir / bv.WORK_DIRNAME).is_dir()
    assert (headered_export_dir / f"{bv.WORK_DIRNAME}-{bv.CONTENT_VARIANT}").is_dir()


def test_the_spot_check_embeds_the_variant_it_verifies(headered_export_dir: Path) -> None:
    """A content-only sidecar verified as if it were header-enriched must NOT pass."""
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)
    matching = bv.verify(
        headered_export_dir,
        spot_checks=3,
        embedder_factory=StubEmbedder,
        log=lambda _m: None,
        with_context_header=False,
        vectors=sidecar,
    )
    assert matching["self_retrieval"]["rank1_hits"] == matching["self_retrieval"]["samples"]
    crossed = bv.verify(
        headered_export_dir,
        spot_checks=3,
        embedder_factory=StubEmbedder,
        log=lambda _m: None,
        with_context_header=True,
        vectors=sidecar,
    )
    assert crossed["self_retrieval"]["rank1_hits"] < crossed["self_retrieval"]["samples"]


def test_read_vector_sidecar_refuses_a_reordered_record_id(headered_export_dir: Path) -> None:
    """A misaligned vector matrix produces confident nonsense, so loading it must raise."""
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)
    shuffled = _ids(headered_export_dir)
    shuffled[2], shuffled[5] = shuffled[5], shuffled[2]
    with pytest.raises(ValueError, match="row 2 .* refusing to use a misaligned"):
        read_vector_sidecar(sidecar, expected_ids=shuffled)


def test_read_vector_sidecar_refuses_a_length_mismatch(headered_export_dir: Path) -> None:
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)
    with pytest.raises(ValueError, match=f"covers {len(TEXTS)} rows but the corpus has 3"):
        read_vector_sidecar(sidecar, expected_ids=_ids(headered_export_dir)[:3])


def test_read_vector_sidecar_refuses_a_table_that_is_not_one(export_dir: Path) -> None:
    with pytest.raises(ValueError, match="not a vector sidecar"):
        read_vector_sidecar(export_dir / "corpus.parquet")


def test_load_export_swaps_in_the_sidecar_as_its_dense_matrix(headered_export_dir: Path) -> None:
    """The harness seam: one export, either vector source, and they are genuinely different."""
    _build(headered_export_dir, StubEmbedder())  # enriched vectors into corpus.parquet
    sidecar = headered_export_dir / CONTENT_ONLY_VECTORS
    _build(headered_export_dir, StubEmbedder(), with_context_header=False, vectors_out=sidecar)

    enriched = load_export(headered_export_dir, "t")
    swapped = load_export(headered_export_dir, "t", vectors=sidecar)
    assert [row.record_id for row in swapped.rows] == [row.record_id for row in enriched.rows]
    assert enriched.embeddings is not None
    assert swapped.embeddings is not None
    assert np.array_equal(swapped.embeddings, read_vector_sidecar(sidecar)[1])
    assert not np.allclose(swapped.embeddings, enriched.embeddings)


def test_load_export_refuses_a_sidecar_built_for_another_corpus(
    headered_export_dir: Path, export_dir: Path
) -> None:
    """Two exports, two id spaces: pointing one at the other's vectors must raise, not rank."""
    other = export_dir / CONTENT_ONLY_VECTORS
    _build(export_dir, StubEmbedder(), with_context_header=False, vectors_out=other)
    _write_export(  # give the two exports different ids so the mismatch is a real one
        headered_export_dir,
        [_row(i + 100, text, with_header=True) for i, text in enumerate(TEXTS)],
    )
    with pytest.raises(ValueError, match="refusing to use a misaligned vector matrix"):
        load_export(headered_export_dir, "t", vectors=other)


# -- the CLI -----------------------------------------------------------------------------------


def test_parse_args_maps_vectors_out_and_the_variant(tmp_path: Path) -> None:
    plan = bv.parse_args(
        [
            "--export",
            str(tmp_path),
            "--no-context-header",
            "--vectors-out",
            str(tmp_path / CONTENT_ONLY_VECTORS),
        ]
    )
    assert plan.with_context_header is False
    assert plan.variant == bv.CONTENT_VARIANT
    assert plan.vectors_out == (tmp_path / CONTENT_ONLY_VECTORS).resolve()
    assert bv.work_dirname(plan.variant) == f"{bv.WORK_DIRNAME}-{bv.CONTENT_VARIANT}"

    default = bv.parse_args(["--export", str(tmp_path)])
    assert default.vectors_out is None
    assert default.variant == bv.HEADER_VARIANT
    assert bv.work_dirname(default.variant) == bv.WORK_DIRNAME


def test_vectors_out_and_jsonl_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        bv.parse_args(
            ["--export", str(tmp_path), "--jsonl", "--vectors-out", str(tmp_path / "v.parquet")]
        )

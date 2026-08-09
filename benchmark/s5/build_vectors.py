# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``build_vectors.py`` — add the missing dense + BM25 artifacts to an already-exported corpus.

The two reused M6 exports (``benchmark/official-data/musique-full/export`` and
``.../hotpot-full/export``) were produced before the S3 pipeline wired an ``embedder`` and a
``bm25`` tokenizer into ``export.jsonl_parquet``, so their ``corpus.parquet`` carries only the six
base columns and the directory has no ``bm25-postings.parquet`` / ``bm25-stats.json``. The S5
retrieval matrix refuses those legs (``legs.missing_inputs``) rather than faking a zero, and no run
checkpoint survives to re-run the pipeline from. This CLI closes the gap the only honest way left:
it regenerates **exactly the artifacts the Export stage would have written**, from the corpus text
already on disk.

**Why this is a reproduction, not a re-implementation.** Every choice is taken from the framework
itself rather than guessed:

* the embedded text is :func:`latence_core.stages.export._corpus_text` with ``unmasked=True`` — the
  same function, the same W13 context-header prepend rule, the same ``unsafe_unmasked_corpus``
  posture the S3 pipeline yaml sets (the M6 rows carry ``masked_content: null`` and
  ``context_header: null``, so the text is each row's ``content``, which is what makes this
  reconstruction exact);
* the model is :data:`benchmark.s5.embed.GRANITE_MODEL`
  (``ibm-granite/granite-embedding-311m-multilingual-r2``) with CLS pooling + L2 normalisation and
  **no** instruction prefix — the pipeline's ``embedding.sentence_transformers`` provider settings,
  reproduced on plain ``transformers`` exactly the way the query side already does it
  (:mod:`benchmark.s5.embed`), so corpus and query vectors land in one space;
* the BM25 artifacts come from the framework's own :class:`latence_core.bm25.Bm25Accumulator`
  driven by the registry-resolved ``tokenizer.regex`` Tokenizer, written with the Export's own
  :meth:`~latence_core.stages.export.JsonlParquetExport._bm25_postings_schema`. Nothing about BM25
  is hand-rolled here;
* the Parquet ``embedding`` column type comes from the Export's own
  :meth:`~latence_core.stages.export.JsonlParquetExport._parquet_schema`, so the produced table's
  schema is the S3 exports' schema for every shared column.

**Additive, never destructive.** The existing six columns, their values, the row order and the ids
are carried through untouched; ``corpus.parquet`` is backed up to ``corpus.parquet.pre-vectors.bak``
before it is replaced, and the pre-existing columns are compared batch-by-batch against that backup
afterwards — a mismatch aborts before the new file is published.

**corpus.jsonl.** The S5 harness reads vectors from ``corpus.parquet`` only
(:func:`benchmark.s5.corpus.load_export`), and materialising 768 floats per row as JSON text would
grow the two files from 324 MB to roughly 4 GB for no reader. So the default leaves ``corpus.jsonl``
byte-identical, and the (S3-shaped) JSONL ``embedding`` key is an explicit ``--jsonl`` opt-in.

**Resumable + idempotent.** Vectors are checkpointed as fixed-size shards under
``<export>/.build-vectors/``; each shard records the sha256 of the record ids it covers, so a re-run
re-embeds only the shards that are missing or whose ids moved, and a completed export re-runs as a
no-op unless ``--force`` is given.

Usage::

    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.build_vectors \\
        --export benchmark/official-data/musique-full/export --device auto
    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.build_vectors \\
        --export benchmark/official-data/musique-full/export --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5.embed import _MAX_TOKENS as MAX_TOKENS  # noqa: E402
from benchmark.s5.embed import GRANITE_MODEL  # noqa: E402
from latence_core.bm25 import Bm25Accumulator  # noqa: E402
from latence_core.capability import ProviderRegistry  # noqa: E402
from latence_core.pipeline import CapabilityKind  # noqa: E402
from latence_core.stages.export import (  # noqa: E402
    JsonlParquetExport,
    _corpus_content,
    _corpus_text,
)
from latence_core.subprovider import (  # noqa: E402
    build_sub_provider,
    owner_label,
    sub_provider_ref,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

#: The BM25 Tokenizer the S3 ``export_corpus`` stage names (``bm25: {provider: tokenizer.regex}``).
BM25_TOKENIZER = "tokenizer.regex"
#: The width the S3 exports carry (``dimension: 768`` in the pipeline yaml, Granite r2 311m).
EMBED_DIM = 768
#: Checkpoint directory, inside the export (already covered by the ``official-data/*/export/``
#: gitignore rule, so no vectors can leak into the repo).
WORK_DIRNAME = ".build-vectors"
#: The untouched original, kept next to the rewritten table so the additive claim stays checkable.
BACKUP_SUFFIX = ".pre-vectors.bak"
#: Rows per checkpoint shard (the resume granularity) and per emitted Parquet row group.
DEFAULT_SHARD_ROWS = 8192
#: Texts per forward pass. 256 measured fastest on this corpus (mean ~49 tokens/chunk).
DEFAULT_BATCH_SIZE = 256
#: The Export's own JSONL/row-group flush granularity, mirrored for the BM25 postings writer.
_POSTINGS_BATCH_ROWS = 1024


# -- device + embedder ------------------------------------------------------------------------


def resolve_device(requested: str) -> str:
    """Map ``auto|cpu|mps|cuda`` to a device that actually exists on this host.

    ``auto`` prefers CUDA, then Apple MPS, then CPU — the same precedence the framework's own
    device seam uses, widened with MPS because this harness also runs on the maintainer's laptop.
    An explicitly requested device is honoured as given (a typo must fail loudly in torch, not be
    silently downgraded to a slower run the operator did not ask for).
    """
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GraniteCorpusEmbedder:
    """Corpus-side vectors from the export's own embedding model — CLS pooled, L2 normalised.

    The corpus twin of :class:`benchmark.s5.embed.GraniteQueryEmbedder`: same checkpoint, same
    pooling, same normalisation, same absence of an instruction prefix, so a query vector and a
    chunk vector are comparable by plain dot product. It differs only in the two things a 237k-row
    corpus needs and a single query does not:

    * a **device seam** (``auto|cpu|mps|cuda``) — the S3 export ran on ``cuda``; this rebuild runs
      wherever it is invoked, and the device is reported so the run is honest about it;
    * **length-sorted batching** — texts are sorted by token length inside each call, encoded, then
      restored to input order. This is what ``sentence-transformers`` itself does in ``encode``, so
      it also matches the S3 export's batching; measured here it doubles throughput by not padding
      a 5-token chunk out to a 96-token one.

    Deterministic for a fixed device: ``eval()`` mode, ``float32``, no dropout, no sampling.
    """

    def __init__(
        self,
        model_id: str = GRANITE_MODEL,
        *,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._model_id = model_id
        self._device = resolve_device(device)
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def device(self) -> str:
        """The resolved torch device this embedder runs on."""
        return self._device

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModel.from_pretrained(self._model_id, dtype=torch.float32).eval()
            self._model = model.to(self._device)
        return self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> Any:  # noqa: ANN401 - numpy is untyped here
        """``(len(texts), 768)`` float32 unit vectors, in input order."""
        import torch

        tokenizer, model = self._load()
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        lengths = [len(text) for text in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for start in range(0, len(order), self._batch_size):
            positions = order[start : start + self._batch_size]
            batch = [texts[i] for i in positions]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_tokens,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state[:, 0]
                vectors = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            out[positions] = vectors.cpu().numpy()
        return out


# -- corpus streaming -------------------------------------------------------------------------


def iter_rows(jsonl: Path) -> Iterator[dict[str, Any]]:
    """Stream ``corpus.jsonl`` one decoded row at a time (the export's own line-per-record form)."""
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def batched(rows: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield ``rows`` in bounded lists of ``size`` — the O(batch) discipline the Export uses."""
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def ids_digest(record_ids: Sequence[str]) -> str:
    """sha256 over the shard's record ids — the checkpoint's proof it still covers these rows."""
    digest = hashlib.sha256()
    for record_id in record_ids:
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# -- checkpointed vector store ----------------------------------------------------------------


class ShardStore:
    """Resumable on-disk vector checkpoints: one ``.npy`` per shard plus a manifest.

    The manifest records, per shard index, the sha256 of the record ids it covers and its row
    count. A re-run recomputes those two facts from ``corpus.jsonl`` and skips any shard that still
    matches — so an interrupted 50-minute embed resumes at the shard it died in, and a completed one
    re-runs as a pure verification pass. A shard whose ids moved (a re-chunked or reordered corpus)
    fails its digest and is recomputed rather than silently reused, which is the property that makes
    the checkpoint safe to keep next to the data.
    """

    def __init__(self, work_dir: Path, *, model_id: str, dim: int) -> None:
        self._dir = work_dir
        self._model_id = model_id
        self._dim = dim
        self._manifest_path = work_dir / "manifest.json"
        self._shards: dict[str, dict[str, Any]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._manifest_path.is_file():
            return
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if data.get("model") != self._model_id or int(data.get("dim", 0)) != self._dim:
            # A different model/width was checkpointed here: its vectors are not ours to reuse.
            self._shards = {}
            return
        self._shards = dict(data.get("shards") or {})

    def _save_manifest(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {"model": self._model_id, "dim": self._dim, "shards": self._shards}
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._manifest_path)

    def path(self, index: int) -> Path:
        """The ``.npy`` file holding shard ``index``'s vectors."""
        return self._dir / f"shard-{index:06d}.npy"

    def has(self, index: int, digest: str, rows: int) -> bool:
        """True when shard ``index`` is already embedded for exactly these ``rows`` ids."""
        entry = self._shards.get(str(index))
        if entry is None or entry.get("digest") != digest or int(entry.get("rows", -1)) != rows:
            return False
        return self.path(index).is_file()

    def write(self, index: int, digest: str, vectors: Any) -> None:  # noqa: ANN401 - numpy
        """Persist a shard's vectors, then record it — so a crash never leaves a claimed-but-absent
        shard."""
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path(index).with_suffix(".npy.tmp")
        # Through a handle, not a path: ``np.save`` re-appends ``.npy`` to a path that lacks it.
        with tmp.open("wb") as handle:
            np.save(handle, vectors.astype(np.float32, copy=False))
        os.replace(tmp, self.path(index))
        self._shards[str(index)] = {"digest": digest, "rows": int(vectors.shape[0])}
        self._save_manifest()

    def read(self, index: int) -> Any:  # noqa: ANN401 - numpy
        """Load shard ``index``'s ``(rows, dim)`` float32 vectors."""
        return np.load(self.path(index))

    def count(self) -> int:
        """Total rows currently checkpointed."""
        return sum(int(entry.get("rows", 0)) for entry in self._shards.values())

    def clear(self) -> None:
        """Drop every checkpoint (``--force``)."""
        if self._dir.is_dir():
            shutil.rmtree(self._dir)
        self._shards = {}


class VectorFeed:
    """Reads the checkpointed shards back as one continuous, memory-bounded row stream."""

    def __init__(self, store: ShardStore, shard_count: int) -> None:
        self._store = store
        self._shard_count = shard_count
        self._index = 0
        self._buffer: Any = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def take(self, rows: int) -> Any:  # noqa: ANN401 - numpy
        """The next ``rows`` vectors in corpus order."""
        while self._buffer.shape[0] < rows and self._index < self._shard_count:
            self._buffer = np.concatenate([self._buffer, self._store.read(self._index)], axis=0)
            self._index += 1
        if self._buffer.shape[0] < rows:
            have = self._buffer.shape[0]
            msg = f"vector checkpoint exhausted: needed {rows} rows, have {have}."
            raise ValueError(msg)
        head, self._buffer = self._buffer[:rows], self._buffer[rows:]
        return head


# -- the three artifact writers -----------------------------------------------------------------


def embedding_array(vectors: Any) -> pa.Array:  # noqa: ANN401 - numpy
    """A ``list<double>`` Arrow array over ``(n, dim)`` vectors — the Export's column shape.

    Widened to ``float64`` exactly like the Export does (its Embedder yields Python ``float``s into
    a ``pa.list_(pa.float64())`` column); the values are the model's float32 output, so the widening
    is exact and the column reads back byte-comparable with an S3 export's.
    """
    rows, dim = vectors.shape
    offsets = pa.array(np.arange(rows + 1, dtype=np.int32) * dim, type=pa.int32())
    values = pa.array(np.ascontiguousarray(vectors, dtype=np.float64).reshape(-1))
    return pa.ListArray.from_arrays(offsets, values)


def corpus_schema(existing: pa.Schema) -> pa.Schema:
    """``existing`` + the Export's own ``embedding`` field, appended in the S3 column position."""
    embed_field = JsonlParquetExport._parquet_schema(True).field("embedding")  # noqa: SLF001
    return pa.schema([*existing, embed_field])


def write_corpus_parquet(export_dir: Path, store: ShardStore, shard_count: int) -> int:
    """Rewrite ``corpus.parquet`` with the ``embedding`` column appended; return the row count.

    Streams the original table row group by row group, appends the matching vectors, and writes to
    a temp sibling. Before publishing it: the original is backed up (once, never overwritten) and
    every pre-existing column of the new file is compared against that backup, batch by batch. Row
    ids are additionally checked against ``corpus.jsonl`` in lockstep, so a table whose order
    diverged from the JSONL the vectors were computed from aborts instead of mis-pairing vectors
    with rows.
    """
    parquet_path = export_dir / "corpus.parquet"
    backup = parquet_path.with_suffix(parquet_path.suffix + BACKUP_SUFFIX)
    if not backup.is_file():
        shutil.copy2(parquet_path, backup)

    source = pq.ParquetFile(backup)  # type: ignore[no-untyped-call]
    base_schema = source.schema_arrow
    if "embedding" in base_schema.names:  # pragma: no cover - guarded by the caller
        msg = "corpus.parquet backup already carries an 'embedding' column; refusing to double-add."
        raise ValueError(msg)
    out_schema = corpus_schema(base_schema)
    feed = VectorFeed(store, shard_count)
    jsonl_ids = (row["record_id"] for row in iter_rows(export_dir / "corpus.jsonl"))

    tmp = parquet_path.with_suffix(".parquet.tmp")
    written = 0
    writer = pq.ParquetWriter(tmp, out_schema)  # type: ignore[no-untyped-call]
    try:
        for batch in source.iter_batches(batch_size=DEFAULT_SHARD_ROWS):  # type: ignore[no-untyped-call]
            vectors = feed.take(batch.num_rows)
            for record_id in batch.column("record_id").to_pylist():
                expected = next(jsonl_ids, None)
                if expected != record_id:
                    msg = (
                        "corpus.parquet and corpus.jsonl disagree on row order "
                        f"({record_id!r} vs {expected!r}); refusing to attach vectors."
                    )
                    raise ValueError(msg)
            arrays = [*batch.columns, embedding_array(vectors)]
            writer.write_table(  # type: ignore[no-untyped-call]
                pa.Table.from_arrays(arrays, schema=out_schema)
            )
            written += batch.num_rows
    finally:
        writer.close()  # type: ignore[no-untyped-call]

    _assert_base_columns_unchanged(backup, tmp, base_schema.names)
    os.replace(tmp, parquet_path)
    return written


def _assert_base_columns_unchanged(backup: Path, candidate: Path, columns: Sequence[str]) -> None:
    """Abort unless every pre-existing column of ``candidate`` equals ``backup``'s, value for value.

    Compared batch by batch (the tables are up to 250 MB of text) with Arrow's own ``equals``, which
    is a data-level comparison — so a rewritten cell, a reordered row or a dropped id is caught
    before the new file replaces the M6 baseline.
    """
    old = pq.ParquetFile(backup)  # type: ignore[no-untyped-call]
    new = pq.ParquetFile(candidate)  # type: ignore[no-untyped-call]
    if old.metadata.num_rows != new.metadata.num_rows:
        msg = f"row count changed: {old.metadata.num_rows} -> {new.metadata.num_rows}."
        raise ValueError(msg)
    old_batches = old.iter_batches(batch_size=DEFAULT_SHARD_ROWS, columns=list(columns))  # type: ignore[no-untyped-call]
    new_batches = new.iter_batches(batch_size=DEFAULT_SHARD_ROWS, columns=list(columns))  # type: ignore[no-untyped-call]
    for old_batch, new_batch in zip(old_batches, new_batches, strict=True):
        for column in columns:
            if not old_batch.column(column).equals(new_batch.column(column)):
                msg = f"pre-existing column {column!r} changed during the rewrite; aborting."
                raise ValueError(msg)


def write_bm25(export_dir: Path) -> dict[str, Any]:
    """Emit ``bm25-postings.parquet`` + ``bm25-stats.json`` with the framework's own builder.

    The Tokenizer is resolved through the plugin registry from the ``tokenizer.regex`` name the S3
    pipeline yaml configures, the fold is :class:`latence_core.bm25.Bm25Accumulator`, the postings
    rows are :meth:`~latence_core.bm25.Bm25Posting.as_row` under the Export's own postings schema,
    and the stats file is :meth:`~latence_core.bm25.Bm25Stats.to_artifact` — so the artifact is the
    Export's, not a look-alike. Statistics are folded over
    :func:`~latence_core.stages.export._corpus_content` (``unmasked=True``, matching the S3 config):
    the STORED corpus text, never the embedding-only context header.
    """
    ref = sub_provider_ref(
        {"provider": BM25_TOKENIZER}, capability=CapabilityKind.EXPORT, key="bm25"
    )
    assert ref is not None  # noqa: S101 - a literal block always parses
    tokenizer = build_sub_provider(
        ref, registry=ProviderRegistry(), owner=owner_label(CapabilityKind.EXPORT)
    )
    accumulator = Bm25Accumulator(tokenizer, ref.provider)
    schema = JsonlParquetExport._bm25_postings_schema()  # noqa: SLF001

    postings_tmp = export_dir / "bm25-postings.parquet.tmp"
    writer = pq.ParquetWriter(postings_tmp, schema)  # type: ignore[no-untyped-call]
    try:
        for batch in batched(iter_rows(export_dir / "corpus.jsonl"), _POSTINGS_BATCH_ROWS):
            postings = accumulator.add_batch(
                (row["record_id"], _corpus_content(row, unmasked=True)) for row in batch
            )
            writer.write_table(  # type: ignore[no-untyped-call]
                pa.Table.from_pylist([p.as_row() for p in postings], schema=schema)
            )
    finally:
        writer.close()  # type: ignore[no-untyped-call]
    os.replace(postings_tmp, export_dir / "bm25-postings.parquet")

    artifact = accumulator.stats().to_artifact()
    stats_tmp = export_dir / "bm25-stats.json.tmp"
    stats_tmp.write_text(
        json.dumps(artifact, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(stats_tmp, export_dir / "bm25-stats.json")
    return artifact


def write_jsonl_embeddings(export_dir: Path, store: ShardStore, shard_count: int) -> int:
    """Opt-in (``--jsonl``): add the ``embedding`` key to every ``corpus.jsonl`` row.

    Off by default — the S5 harness reads vectors from Parquet only, and this multiplies the two
    files' size by roughly twelve. When asked for, the line format is the Export's: one sorted-key,
    non-ASCII-escaped JSON object per line, written to a temp sibling and renamed, so the original
    survives a failure mid-file.
    """
    jsonl = export_dir / "corpus.jsonl"
    feed = VectorFeed(store, shard_count)
    tmp = jsonl.with_suffix(".jsonl.tmp")
    written = 0
    with tmp.open("w", encoding="utf-8") as out:
        for batch in batched(iter_rows(jsonl), DEFAULT_SHARD_ROWS):
            vectors = feed.take(len(batch))
            for row, vector in zip(batch, vectors, strict=True):
                row["embedding"] = [float(x) for x in vector]
                out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                written += 1
    os.replace(tmp, jsonl)
    return written


# -- the build --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildPlan:
    """One resolved build request."""

    export_dir: Path
    device: str
    batch_size: int
    shard_rows: int
    write_jsonl: bool
    force: bool
    verify_only: bool
    spot_checks: int
    #: Embed content+ContextHeader (the S3 rule) or content only (dense baseline).
    with_context_header: bool = True


def is_complete(export_dir: Path) -> bool:
    """True when the export already carries all three artifacts this CLI produces."""
    parquet = export_dir / "corpus.parquet"
    if not parquet.is_file():
        return False
    has_embedding = "embedding" in pq.read_schema(parquet).names  # type: ignore[no-untyped-call]
    return (
        has_embedding
        and (export_dir / "bm25-postings.parquet").is_file()
        and (export_dir / "bm25-stats.json").is_file()
    )


def embed_corpus(
    plan: BuildPlan,
    store: ShardStore,
    embedder_factory: Callable[[], Any],
    *,
    log: Callable[[str], None],
) -> int:
    """Embed every shard that is not already checkpointed; return the shard count.

    The embedder is built lazily by ``embedder_factory`` and only when a shard actually needs
    computing, so a fully-resumed run never pays for a 1.2 GB checkpoint load — and the tests inject
    a deterministic stand-in for the model without any fake vector reaching a real export.
    """
    embedder: Any = None
    shard_count = 0
    done = 0
    started = time.monotonic()
    for index, batch in enumerate(
        batched(iter_rows(plan.export_dir / "corpus.jsonl"), plan.shard_rows)
    ):
        shard_count = index + 1
        record_ids = [str(row["record_id"]) for row in batch]
        digest = ids_digest(record_ids)
        if store.has(index, digest, len(batch)):
            continue
        if embedder is None:
            embedder = embedder_factory()
        texts = [
            corpus_text_for(row, with_context_header=plan.with_context_header)
            for row in batch
        ]
        vectors = embedder.encode(texts)
        if vectors.shape != (len(batch), EMBED_DIM):
            msg = f"embedder returned {vectors.shape} for a shard of {len(batch)} rows."
            raise ValueError(msg)
        store.write(index, digest, vectors)
        done += len(batch)
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed else 0.0
        log(f"  shard {index}: {done} rows embedded, {rate:.1f} rows/s")
    return shard_count


def build(
    plan: BuildPlan,
    *,
    embedder_factory: Callable[[], Any] | None = None,
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
) -> dict[str, Any]:
    """Bring one export directory up to the S3 artifact set; return the verification report."""
    export_dir = plan.export_dir
    for required in ("corpus.jsonl", "corpus.parquet"):
        if not (export_dir / required).is_file():
            msg = f"{export_dir} has no {required}; this augments an export, it cannot make one."
            raise FileNotFoundError(msg)

    if plan.verify_only:
        return verify(
            export_dir,
            spot_checks=plan.spot_checks,
            device=plan.device,
            log=log,
            embedder_factory=embedder_factory,
        )

    store = ShardStore(export_dir / WORK_DIRNAME, model_id=GRANITE_MODEL, dim=EMBED_DIM)
    if plan.force:
        store.clear()
    elif is_complete(export_dir):
        log(f"{export_dir}: artifacts already present; nothing to do (--force rebuilds)")
        return verify(
            export_dir,
            spot_checks=plan.spot_checks,
            device=plan.device,
            log=log,
            embedder_factory=embedder_factory,
        )

    factory = embedder_factory or (
        lambda: GraniteCorpusEmbedder(device=plan.device, batch_size=plan.batch_size)
    )
    started = time.monotonic()
    log(f"{export_dir}: embedding (device={resolve_device(plan.device)})")
    shard_count = embed_corpus(plan, store, factory, log=log)
    log(f"  vectors checkpointed: {store.count()} rows in {shard_count} shards")

    if "embedding" not in pq.read_schema(export_dir / "corpus.parquet").names:  # type: ignore[no-untyped-call]
        rows = write_corpus_parquet(export_dir, store, shard_count)
        log(f"  corpus.parquet rewritten with the embedding column ({rows} rows)")
    else:
        log("  corpus.parquet already carries the embedding column")

    artifact = write_bm25(export_dir)
    log(f"  bm25 artifacts written (vocab {len(artifact['terms'])}, avgdl {artifact['avgdl']})")

    if plan.write_jsonl:
        rows = write_jsonl_embeddings(export_dir, store, shard_count)
        log(f"  corpus.jsonl rewritten with the embedding key ({rows} rows)")

    report = verify(
        export_dir,
        spot_checks=plan.spot_checks,
        device=plan.device,
        log=log,
        embedder_factory=embedder_factory,
    )
    report["wall_seconds"] = round(time.monotonic() - started, 1)
    report["device"] = resolve_device(plan.device)
    return report


# -- verification -----------------------------------------------------------------------------


def corpus_text_for(row: dict[str, Any], *, with_context_header: bool) -> str:
    """The text to embed for ``row`` — the framework's own selectors, never a local copy.

    ``with_context_header=True`` is :func:`_corpus_text`: the W13/ADR-0039 rule that PREPENDS
    the row's rendered :class:`ContextHeader` (its KG entities + neighbour triples) to the
    content, which is what the S3 exports embedded. ``False`` is the bare
    :func:`_corpus_content` — the content-only DENSE BASELINE.

    Why the flag exists (measured 2026-08-09): the S3 corpora carry populated context headers
    (200/200 sampled multihop_rag rows) while the reused M6 exports carry ``context_header:
    null`` on every row. Embedding both with the header rule would silently compare
    KG-enriched vectors against plain ones across datasets. Building BOTH variants turns that
    confound into an explicit ablation axis: every dataset gets a content-only baseline, and
    header enrichment becomes a measurable leg rather than a hidden advantage.
    """
    if with_context_header:
        return _corpus_text(row, unmasked=True)
    return _corpus_content(row, unmasked=True)


def read_matrix(parquet: Path) -> tuple[list[str], Any, set[int]]:
    """``(record_ids, float32 matrix, the set of per-row vector widths)`` from a corpus table.

    Read row group by row group and stacked into one pre-sized ``float32`` array, because the
    largest export here is 185,215 × 768: materialising that column as Python lists first (the
    convenient one-liner) costs several GB of transient, while this costs the 568 MB the matrix
    itself needs. The width **set** is collected on the way through, so "every row is 768-wide" is
    an observation about every row rather than an inference from the first one.
    """
    source = pq.ParquetFile(parquet)  # type: ignore[no-untyped-call]
    total = int(source.metadata.num_rows)
    record_ids: list[str] = []
    widths: set[int] = set()
    matrix = np.zeros((total, EMBED_DIM), dtype=np.float32)
    filled = 0
    for batch in source.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=DEFAULT_SHARD_ROWS, columns=["record_id", "embedding"]
    ):
        record_ids.extend(str(value) for value in batch.column("record_id").to_pylist())
        column = batch.column("embedding")
        offsets = column.offsets.to_numpy()
        widths.update(int(w) for w in np.unique(np.diff(offsets)))
        values = column.flatten().to_numpy(zero_copy_only=False)
        rows = batch.num_rows
        if widths == {EMBED_DIM}:
            matrix[filled : filled + rows] = values.reshape(rows, EMBED_DIM)
        filled += rows
    return record_ids, matrix, widths


def verify(
    export_dir: Path,
    *,
    spot_checks: int = 5,
    device: str = "auto",
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
    embedder_factory: Callable[[], Any] | None = None,
    with_context_header: bool = True,
) -> dict[str, Any]:
    """Check the produced artifacts and return the numbers, never a bare "ok".

    Asserts the additive contract (row count unchanged against the backup, pre-existing columns
    intact), the vector contract (768 wide, finite, unit norm) and the BM25 contract (postings row
    per corpus row, sane vocabulary + average document length). Then it runs the same sanity check
    :mod:`benchmark.s5.embed` uses on the S3 exports: a sampled chunk's own text, re-embedded, must
    retrieve that chunk at rank 1 by cosine over the whole matrix. ``spot_checks=0`` skips only that
    last step (it is the one that loads the model).
    """
    parquet = export_dir / "corpus.parquet"
    backup = parquet.with_suffix(parquet.suffix + BACKUP_SUFFIX)
    record_ids, matrix, widths = read_matrix(parquet)
    norms = np.linalg.norm(matrix, axis=1)
    report: dict[str, Any] = {
        "export": str(export_dir),
        "rows": len(record_ids),
        "embedding_dim": int(matrix.shape[1]),
        "all_rows_768": bool(widths == {EMBED_DIM} and matrix.shape[0] == len(record_ids)),
        "nan_or_inf": int(np.count_nonzero(~np.isfinite(matrix))),
        "norm_min": float(norms.min()) if len(norms) else 0.0,
        "norm_max": float(norms.max()) if len(norms) else 0.0,
    }
    if backup.is_file():
        report["rows_before"] = int(pq.ParquetFile(backup).metadata.num_rows)  # type: ignore[no-untyped-call]
        report["rows_unchanged"] = bool(report["rows_before"] == report["rows"])

    stats = json.loads((export_dir / "bm25-stats.json").read_text(encoding="utf-8"))
    postings = pq.read_table(export_dir / "bm25-postings.parquet", columns=["record_id"])  # type: ignore[no-untyped-call]
    report["bm25"] = {
        "artifact": stats["artifact"],
        "tokenizer": stats["tokenizer"],
        "doc_count": stats["doc_count"],
        "vocab_size": len(stats["terms"]),
        "avgdl": stats["avgdl"],
        "total_tokens": stats["total_tokens"],
        "postings_rows": postings.num_rows,
        "postings_cover_corpus": bool(postings.num_rows == len(record_ids)),
        "doc_count_matches": bool(stats["doc_count"] == len(record_ids)),
    }

    if spot_checks > 0 and record_ids:
        report["self_retrieval"] = _self_retrieval(
            export_dir, record_ids, matrix, spot_checks, device, embedder_factory, log,
            with_context_header=with_context_header,
        )
    return report


def _self_retrieval(
    export_dir: Path,
    record_ids: Sequence[str],
    matrix: Any,  # noqa: ANN401 - numpy
    samples: int,
    device: str,
    embedder_factory: Callable[[], Any] | None,
    log: Callable[[str], None],
    *,
    with_context_header: bool = True,
) -> dict[str, Any]:
    """Re-embed ``samples`` chunks' own text and confirm each retrieves itself at rank 1.

    Evenly spaced positions (not a random sample) so the check is reproducible across runs and
    covers the whole corpus rather than clustering at one end. Cosine is a dot product because every
    stored vector is unit length.
    """
    positions = sorted({int(i) for i in np.linspace(0, len(record_ids) - 1, samples)})
    wanted = {record_ids[p] for p in positions}
    texts: dict[str, str] = {}
    for row in iter_rows(export_dir / "corpus.jsonl"):
        if row["record_id"] in wanted:
            texts[str(row["record_id"])] = corpus_text_for(
                row, with_context_header=with_context_header
            )
            if len(texts) == len(wanted):
                break
    factory = embedder_factory or (lambda: GraniteCorpusEmbedder(device=device))
    embedder = factory()
    probes = embedder.encode([texts[record_ids[p]] for p in positions])
    scores = probes @ matrix.T
    top = np.argmax(scores, axis=1)
    hits = int(sum(1 for probe, position in enumerate(positions) if int(top[probe]) == position))
    self_cos = [float(scores[probe, position]) for probe, position in enumerate(positions)]
    log(f"  self-retrieval rank-1: {hits}/{len(positions)}")
    return {
        "samples": len(positions),
        "rank1_hits": hits,
        "min_self_cosine": round(min(self_cos), 6) if self_cos else 0.0,
    }


# -- CLI --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> BuildPlan:
    """Parse the CLI into a :class:`BuildPlan`."""
    parser = argparse.ArgumentParser(
        description="Add the S3-shaped embedding column + BM25 artifacts to an existing export"
    )
    parser.add_argument("--export", type=Path, required=True, help="an export/ directory")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument(
        "--jsonl", action="store_true", help="also add the embedding key to corpus.jsonl"
    )
    parser.add_argument("--force", action="store_true", help="discard checkpoints and rebuild")
    parser.add_argument("--verify-only", action="store_true", help="report on existing artifacts")
    parser.add_argument("--spot-checks", type=int, default=5, help="self-retrieval samples (0=off)")
    parser.add_argument(
        "--no-context-header",
        action="store_true",
        help="embed CONTENT ONLY (the dense baseline). Default embeds the W13 "
        "ContextHeader prepend, matching how the S3 corpora were exported.",
    )
    args = parser.parse_args(argv)
    return BuildPlan(
        export_dir=args.export.resolve(),
        device=str(args.device),
        batch_size=int(args.batch_size),
        shard_rows=int(args.shard_rows),
        write_jsonl=bool(args.jsonl),
        force=bool(args.force),
        verify_only=bool(args.verify_only),
        spot_checks=int(args.spot_checks),
        with_context_header=not bool(args.no_context_header),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: build (or verify) one export and print the report as JSON on stdout."""
    plan = parse_args(argv)
    report = build(plan)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BM25_TOKENIZER",
    "EMBED_DIM",
    "BuildPlan",
    "GraniteCorpusEmbedder",
    "ShardStore",
    "build",
    "corpus_schema",
    "embed_corpus",
    "read_matrix",
    "resolve_device",
    "verify",
    "write_bm25",
    "write_corpus_parquet",
    "write_jsonl_embeddings",
]

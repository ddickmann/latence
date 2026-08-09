# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The loaded S3 export: chunk rows, the dense matrix, the KG indices — and the gold ChunkIndex.

One :class:`Export` per dataset, built once per run and shared by every leg, because the legs
are a *cumulative ladder*: re-reading a 640 MB vector table six times would make the ablation's
own latency numbers meaningless. What it holds is exactly the emitted artifacts, nothing derived
beyond the indices the S3 stage did not persist:

* the chunk rows (id, text, provenance, ``kg_node_ids``) in export order,
* the dense matrix (``float32``, L2-normalised at export, so cosine is one matmul),
* ``file_name → chunk ids`` and node/hyperedge → chunk maps,
* the :class:`~benchmark.s5.goldmap.ChunkIndex` implementation the gold mapping resolves against.

It implements ``ChunkIndex`` structurally, so :func:`benchmark.s5.goldmap.resolve` stays a pure
function over a five-chunk hand-written index in tests and over 220,129 real rows here.

**Two vector sources, never mixed silently.** ``corpus.parquet``'s ``embedding`` column holds the
vectors the S3 Export wrote, which embed *content + the W13 ContextHeader prepend* (the row's KG
entities and neighbour triples). The content-only baseline cannot go in that column — the Export's
writer refuses to double-add it and the enriched vectors are the ones already on disk — so it lives
in a **sidecar** table (:data:`CONTENT_ONLY_VECTORS`, written by ``build_vectors --vectors-out``)
holding nothing but ``record_id`` + ``embedding``. :func:`read_vector_sidecar` is the only way in,
and it RAISES unless the sidecar's ids are the corpus's ids in the corpus's order: a vector matrix
that is off by one row does not fail, it returns confident nonsense.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from benchmark.s5.pagemap import DocumentPages, load_dataset_pages

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ChunkRow:
    """One exported chunk: the id retrieval ranks, its text, and where it came from."""

    record_id: str
    content: str
    file_name: str
    char_start: int
    char_end: int
    node_ids: tuple[str, ...]


@dataclass
class Export:
    """A dataset's loaded S3 export — the shared substrate every leg reads."""

    dataset: str
    root: Path
    rows: list[ChunkRow]
    embeddings: Any | None
    pages: dict[str, DocumentPages]
    _by_file: dict[str, list[int]] = field(default_factory=dict)
    _by_id: dict[str, int] = field(default_factory=dict)
    _by_node: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for index, row in enumerate(self.rows):
            self._by_file.setdefault(row.file_name, []).append(index)
            self._by_id[row.record_id] = index
            for node in row.node_ids:
                self._by_node.setdefault(node, []).append(index)

    # -- ChunkIndex (benchmark.s5.goldmap) ----------------------------------------------------

    def chunks_in_file(self, file_name: str) -> Sequence[str]:
        """Every chunk id emitted from ``file_name``, in export order."""
        return [self.rows[i].record_id for i in self._by_file.get(file_name, ())]

    def chunks_on_page(self, file_name: str, page: int) -> Sequence[str]:
        """Chunk ids whose provenance char span overlaps 1-based ``page`` of ``file_name``.

        Falls back to **nothing** (not to the whole file) when the document has no reconstructed
        page map: an unresolvable page reference must show up in the resolution rate, because
        silently widening it to document level is precisely the inflation this harness exists to
        avoid on the single-hop control datasets.
        """
        doc = self.pages.get(Path(file_name).stem)
        if doc is None:
            return []
        span = doc.span(page)
        if span is None:
            return []
        start, end = span
        return [
            self.rows[i].record_id
            for i in self._by_file.get(file_name, ())
            if self.rows[i].char_start < end and self.rows[i].char_end > start
        ]

    def has_page_map(self, file_name: str) -> bool:
        """Whether ``file_name``'s page char spans were reconstructed for this run.

        ``False`` covers both "this dataset has no S2 ``pages/`` artifacts" and "this document's
        reconstruction did not reproduce the assembled markdown, so :mod:`benchmark.s5.pagemap`
        omitted it". Either way page-level gold is *unavailable* here rather than *wrong*, which
        is the distinction :func:`benchmark.s5.goldmap.resolve_ref` needs to decide between a
        coarser document-level reference and an honest miss.
        """
        return Path(file_name).stem in self.pages

    # -- retrieval-side views -----------------------------------------------------------------

    def index_of(self, record_id: str) -> int | None:
        """Row position of ``record_id`` (``None`` when it is not in this export)."""
        return self._by_id.get(record_id)

    def rows_for_nodes(self, node_ids: Sequence[str]) -> list[int]:
        """Row positions of every chunk mentioning any of ``node_ids``, deduplicated + sorted."""
        hit: set[int] = set()
        for node in node_ids:
            hit.update(self._by_node.get(node, ()))
        return sorted(hit)

    @property
    def files(self) -> frozenset[str]:
        """Every source file name present in the export."""
        return frozenset(self._by_file)


_EMBEDDING_COLUMN = "embedding"
_RECORD_ID_COLUMN = "record_id"

#: The conventional file name for the CONTENT-ONLY vector sidecar inside an export directory.
#: ``build_vectors --vectors-out <export>/corpus-vectors-content-only.parquet`` writes it and the
#: ``dense_content_only`` ablation leg reads it; naming it here (rather than at each call site)
#: is what lets :func:`benchmark.s5.legs.missing_inputs` refuse the leg by file name.
CONTENT_ONLY_VECTORS = "corpus-vectors-content-only.parquet"

#: Rows per read batch — the sidecar is up to 297k × 768 float32 (~0.9 GB), so it is streamed
#: into one pre-sized matrix rather than materialised as Python lists first.
_SIDECAR_BATCH_ROWS = 8192


def read_vector_sidecar(
    path: Path, *, expected_ids: Sequence[str] | None = None
) -> tuple[list[str], Any]:
    """``(record_ids, float32 (rows, dim) matrix)`` from a ``--vectors-out`` sidecar table.

    The sidecar carries exactly two columns — ``record_id`` and a fixed-width ``float32``
    ``embedding`` — in the corpus's row order, and no content: it is an *alternative value* for
    one column of ``corpus.parquet``, not a second copy of the corpus.

    ``expected_ids`` is the alignment contract and it is checked, not assumed. A sidecar built
    from a different (re-chunked, re-ordered, or simply other) corpus would still load as a
    perfectly shaped matrix and every rank it produced would be wrong-but-plausible, so a
    disagreement RAISES here, naming the first position that differs. Callers that genuinely have
    no corpus to check against pass ``None`` and own that decision explicitly.
    """
    source = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
    names = set(source.schema_arrow.names)
    missing = [c for c in (_RECORD_ID_COLUMN, _EMBEDDING_COLUMN) if c not in names]
    if missing:
        msg = f"{path}: not a vector sidecar — missing column(s) {missing}."
        raise ValueError(msg)

    total = int(source.metadata.num_rows)
    record_ids: list[str] = []
    matrix: Any = None
    filled = 0
    for batch in source.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=_SIDECAR_BATCH_ROWS, columns=[_RECORD_ID_COLUMN, _EMBEDDING_COLUMN]
    ):
        rows = int(batch.num_rows)
        if not rows:
            continue
        values = batch.column(_EMBEDDING_COLUMN).flatten().to_numpy(zero_copy_only=False)
        if values.size % rows:
            msg = f"{path}: ragged embedding rows ({values.size} values over {rows} rows)."
            raise ValueError(msg)
        width = values.size // rows
        if matrix is None:
            matrix = np.zeros((total, width), dtype=np.float32)
        elif width != matrix.shape[1]:
            msg = f"{path}: embedding width changes mid-file ({matrix.shape[1]} -> {width})."
            raise ValueError(msg)
        matrix[filled : filled + rows] = values.reshape(rows, width)
        record_ids.extend(str(value) for value in batch.column(_RECORD_ID_COLUMN).to_pylist())
        filled += rows
    if matrix is None:
        matrix = np.zeros((0, 0), dtype=np.float32)
    if expected_ids is not None:
        _assert_ids_aligned(path, record_ids, list(expected_ids))
    return record_ids, matrix


def _assert_ids_aligned(path: Path, found: Sequence[str], expected: Sequence[str]) -> None:
    """Raise unless ``found`` is ``expected``, same ids in the same order — never a warning."""
    if len(found) != len(expected):
        msg = (
            f"{path}: vector sidecar covers {len(found)} rows but the corpus has "
            f"{len(expected)}; refusing to use a misaligned vector matrix."
        )
        raise ValueError(msg)
    for position, (got, want) in enumerate(zip(found, expected, strict=True)):
        if got != want:
            msg = (
                f"{path}: vector sidecar row {position} is {got!r} but the corpus row there is "
                f"{want!r}; refusing to use a misaligned vector matrix."
            )
            raise ValueError(msg)


def load_export(
    root: Path,
    dataset: str,
    *,
    s2_root: Path | None = None,
    with_embeddings: bool = True,
    page_docs: frozenset[str] | None = None,
    vectors: Path | None = None,
) -> Export:
    """Read ``root``'s ``corpus.parquet`` into an :class:`Export` (+ the page maps when present).

    The embedding column is read only when a leg needs it (``with_embeddings``) — the vectors are
    the bulk of the table and a bm25-only re-run should not pay for them. Vectors are cast to
    ``float32``: the export stores ``double`` but the values are the embedder's fp16 output, so
    the cast is exact and halves the resident matrix. ``page_docs`` limits the page-span
    reconstruction to the documents this run's queries actually reference — a 50-query UDA smoke
    touches a handful of 1,917 documents, and reassembling all of them would dominate the run.

    ``vectors`` REPLACES the dense matrix with a :func:`read_vector_sidecar` table (the
    content-only ablation variant), leaving every other column read from ``corpus.parquet``
    exactly as before. The embedding column is then not read at all — there is no path where both
    sources end up in one :class:`Export`, because a leg must be able to say which vectors
    produced its number.
    """
    # The reused M6 exports (musique/hotpot) predate the context/embedding columns, so the
    # optional ones are selected only when the schema actually has them — a leg that needs a
    # missing column is refused upstream by `legs.missing_inputs`, never silently zero-filled.
    available = set(pq.read_schema(root / "corpus.parquet").names)  # type: ignore[no-untyped-call]
    optional = [c for c in ("kg_node_ids", _EMBEDDING_COLUMN) if c in available]
    if (not with_embeddings or vectors is not None) and _EMBEDDING_COLUMN in optional:
        optional.remove(_EMBEDDING_COLUMN)
    table = pq.read_table(  # type: ignore[no-untyped-call]
        root / "corpus.parquet", columns=["record_id", "content", "provenance", *optional]
    )
    provenances = table.column("provenance").to_pylist()
    record_ids = table.column("record_id").to_pylist()
    contents = table.column("content").to_pylist()
    node_lists = (
        table.column("kg_node_ids").to_pylist()
        if "kg_node_ids" in table.column_names
        else [()] * len(record_ids)
    )
    rows: list[ChunkRow] = []
    for record_id, content, raw_provenance, nodes in zip(
        record_ids, contents, provenances, node_lists, strict=True
    ):
        provenance = json.loads(raw_provenance)
        rows.append(
            ChunkRow(
                record_id=str(record_id),
                content=str(content or ""),
                file_name=str(provenance.get("file_name", "")),
                char_start=int(provenance.get("char_start", 0)),
                char_end=int(provenance.get("char_end", 0)),
                node_ids=tuple(str(n) for n in (nodes or ())),
            )
        )
    embeddings = None
    if vectors is not None:
        embeddings = read_vector_sidecar(
            vectors, expected_ids=[row.record_id for row in rows]
        )[1]
    elif _EMBEDDING_COLUMN in table.column_names:
        embeddings = np.asarray(
            table.column(_EMBEDDING_COLUMN).to_numpy(zero_copy_only=False).tolist(),
            dtype=np.float32,
        )
    pages: dict[str, DocumentPages] = {}
    if s2_root is not None:
        pages = load_dataset_pages(s2_root, dataset, only=page_docs)
    return Export(dataset=dataset, root=root, rows=rows, embeddings=embeddings, pages=pages)


def hyperedge_chunk_map(hyperedges_parquet: Path) -> dict[str, str]:
    """``hyperedge_id → primary chunk record id`` from the emitted hyperedge parquet.

    The KV's solver fields deliberately carry no provenance (ADR-0056: the parquet is the
    durable, inspectable form), so the hypergraph and CoE legs resolve their selected hyperedges
    back to rankable chunk ids exactly the way the M6 harness does.
    """
    table = pq.read_table(hyperedges_parquet, columns=["hyperedge_id", "properties"])  # type: ignore[no-untyped-call]
    out: dict[str, str] = {}
    for hyperedge_id, raw in zip(
        table.column("hyperedge_id").to_pylist(),
        table.column("properties").to_pylist(),
        strict=True,
    ):
        properties = json.loads(raw) if raw else {}
        chunk = properties.get("primary_chunk_record_id")
        if chunk:
            out[str(hyperedge_id)] = str(chunk)
    return out


def node_alias_map(nodes_parquet: Path) -> dict[str, str]:
    """``canonical surface → node id`` for the graph leg's
    :class:`~latence_retrieval.graph_retrieval.AliasQuerySeeder`.

    The emitted node table is the alias source the framework itself publishes; a surface claimed
    by several nodes keeps the **first in id order** so the seeding is deterministic.
    """
    table = pq.read_table(nodes_parquet, columns=["node_id", "canonical_name"])  # type: ignore[no-untyped-call]
    out: dict[str, str] = {}
    for node_id, name in sorted(
        zip(
            table.column("node_id").to_pylist(),
            table.column("canonical_name").to_pylist(),
            strict=True,
        )
    ):
        surface = str(name or "").strip()
        if surface and surface.lower() not in out:
            out[surface.lower()] = str(node_id)
    return out


__all__ = [
    "CONTENT_ONLY_VECTORS",
    "ChunkRow",
    "Export",
    "hyperedge_chunk_map",
    "load_export",
    "node_alias_map",
    "read_vector_sidecar",
]

"""JSONL + Parquet Export Provider (ADR-0006, S1 spine).

Materializes records in both built-in serializations from the one contract:
JSONL (streaming, human-diffable, checkpoint-friendly) and Parquet/Arrow
(lakehouse handoff — Databricks/Snowflake read Stage outputs natively). Both
are written from the same Pydantic dumps, so they never drift.

**Opt-in embeddings (ADR-0017).** Embeddings are optional: the corpus exports its
text with full Provenance/Classification whether or not an Embedder is wired. When the
Stage ``config`` names an Embedder Provider under
``embedder`` (``{"provider": ..., "config": {...}}``), this Provider resolves that
Embedder from the plugin registry and attaches a fixed-width ``embedding`` vector to
every exported row (JSONL key + Parquet column), embedding the record's **corpus
text** — the ``masked_content`` of a RedactionRecord (the PII-handled variant) or the
``content`` of a Chunk/Document. The in-core ``embedding.hashing`` reference makes the
seam exercisable end to end offline; a served model is the pluggable
``embedding.endpoint`` package, swapped by one config line (the framework names the
seam, not the model).

**The PII-masking guarantee and its one explicit opt-out.** By default the corpus text this
Provider materializes (the ``content`` cell, the embedder input) is the PII-handled
``masked_content`` whenever the Redaction Stage produced one — the raw text never reaches the
exported corpus. That default is unconditional and unchanged. The single, self-incriminating way
out is the ``unsafe_unmasked_corpus`` config key (default ``False``): when set, the corpus carries
the RAW ``content`` while ``pii_spans`` / ``masked_content`` are still written as metadata (the
detection evidence is kept, only the substitution is skipped), and the Export logs a WARNING
naming the risk.
It exists for BENCHMARKING, where placeholder-substituted text destroys the measurement
(measured on the S3 campaign: 82% of 2WikiMultiHopQA / MultiHop-RAG rows carried placeholders,
including article titles) — never for a production corpus.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from latence_core.capability import Embedder, ProviderRegistry
from latence_core.contracts import SCHEMA_VERSION
from latence_core.iceberg import build_iceberg_output
from latence_core.pipeline import CapabilityKind
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)
from latence_core.subprovider import owner_label, resolve_sub_provider

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import Record
    from latence_core.storage import Storage

# Row-group / JSONL flush granularity for the incremental Export (ADR-0033 / #63). The
# Export no longer materialises the whole corpus as one Python list + one Arrow table
# (``pa.Table.from_pylist(flat)``) — the single biggest transient G1 measured, coincident
# with peak RSS. Instead it streams the input in bounded batches: each batch becomes one
# Parquet row group (via a ``pq.ParquetWriter``) and one JSONL flush, so the Export's peak
# memory is O(batch), not O(corpus). Byte-for-byte identical output to the buffered path
# (same per-row dumps, same sorted-key JSONL, same flat Parquet schema + values) — only the
# transient footprint changes.
_EXPORT_BATCH_ROWS = 1024

# The message prefix this Stage's nested-block errors carry ("Export config 'embedder' …"), read
# from the one slot declaration so it cannot drift from what the offline checker reports.
_OWNER = owner_label(CapabilityKind.EXPORT)

_LOG = logging.getLogger(__name__)

# The one-per-export WARNING the ``unsafe_unmasked_corpus`` opt-out emits (never at import or
# wiring time — it names what THIS export is about to write). Kept as a constant so the operator
# banner, the docstrings and the regression test all read the same words.
_UNMASKED_CORPUS_WARNING = (
    "Export config 'unsafe_unmasked_corpus' is ON: exporting UNMASKED corpus text; PII detections "
    "are recorded in pii_spans but NOT substituted — benchmark/offline use only."
)


def _corpus_content(row: dict[str, Any], *, unmasked: bool = False) -> str:
    """The PII-safe corpus text materialized for a row (``content`` column + JSONL).

    A redacted :class:`~latence_core.contracts.ChunkRecord` (W16, ADR-0042) carries the un-redacted
    text in ``content`` and the PII-handled variant in ``masked_content``; a deprecated
    :class:`~latence_core.contracts.RedactionRecord` carries only ``masked_content``. When
    ``masked_content`` is present and non-``None``, it IS the exported corpus text — so the RAG
    corpus never materializes the raw, un-redacted ``content`` (the PII-safety guarantee: the
    stack-validate rag-scan reads every exported cell). A row with no ``masked_content`` (no
    Redaction Stage, or a non-chunk carrier) exports its own ``content`` unchanged — byte-identical
    to a run without Redaction (the additive-field guarantee).

    **The one opt-out (``unmasked=True``).** Bound at the call site from the Export's
    ``unsafe_unmasked_corpus`` config key (default ``False`` ⇒ everything above holds verbatim).
    When ``True`` this returns the row's RAW ``content`` (falling back to ``masked_content`` only
    when the carrier has no ``content`` at all, e.g. a deprecated RedactionRecord), so the corpus
    text is the un-substituted original. The row still carries ``masked_content`` and ``pii_spans``
    as metadata — the detection evidence is never discarded, only the substitution is skipped — and
    the Export logs a WARNING once per export. BENCHMARKING only: placeholders destroy the
    measurement (82% of the S3 2WikiMultiHopQA / MultiHop-RAG rows carried them, titles included).
    """
    if unmasked:
        content = row.get("content")
        if content is not None:
            return str(content)
    masked = row.get("masked_content")
    if masked is not None:
        return str(masked)
    return str(row.get("content") or "")


class JsonlParquetExport:
    """Export Provider writing ``records.jsonl`` and ``records.parquet``.

    **PII-safety guarantee (default) + its one explicit opt-out.** Whenever the Redaction Stage
    produced a ``masked_content``, THAT is the corpus text this Provider materializes — the
    ``content`` cell and the Embedder input — so the exported corpus never materializes the raw,
    un-redacted text (:func:`_corpus_content`). The one documented way out is
    ``unsafe_unmasked_corpus`` below; with it absent/False every byte of the output is exactly
    what it was before the flag existed.

    Config:
        basename: str  — output file stem (default 'records').
        unsafe_unmasked_corpus: bool  — OPT-OUT of the PII-masking guarantee above (default
            **False**; benchmark/offline use only). Named to be self-incriminating: it is meant to
            be impossible to read ``unsafe_unmasked_corpus: true`` in a stack YAML, a diff or a
            grep and not know what was traded away. When True, every corpus-text surface carries
            the row's RAW ``content`` instead of ``masked_content``, and the Export logs a WARNING
            once per export naming the risk. ``pii_spans`` and ``masked_content`` are STILL written
            to every exported row — the detection evidence is preserved, only the substitution is
            skipped — so the export remains auditable and a consumer can re-apply masking. It
            exists because placeholder-substituted text invalidates benchmarking: the S3
            campaign measured 82% of the exported 2WikiMultiHopQA / MultiHop-RAG rows carrying
            placeholders, article titles included, which makes any cross-dataset comparison
            meaningless. Absent/False ⇒ behaviour is byte-identical to a build without this key.
        embedder: dict | None  — OPTIONAL opt-in Embedder augmentation (ADR-0017). A
            ``{"provider": "<entry-point name>", "config": {...}}`` block naming an
            :class:`~latence_core.capability.Embedder` Provider (e.g. the in-core
            ``embedding.hashing`` reference, or the pluggable ``embedding.endpoint``).
            When present, every exported row gets an ``embedding`` vector of the
            Embedder's fixed ``dimension``. Absent ⇒ no embeddings (the corpus text is
            still exported with full Provenance/Classification).
        iceberg: dict | None  — OPTIONAL opt-in Iceberg-table output (#184, ADR-0017), OFF by
            default. A ``{"uri": "<Iceberg REST catalog>", "namespace": "latence", "table":
            "records", "mode": "append" | "overwrite", ...}`` block. When present, the Export
            *additionally* registers this run's corpus as an Apache Iceberg table in that REST
            catalog, so the same data is directly governable in Unity/Polaris/Glue/Lakekeeper with
            no glue code. The registered table mirrors the Parquet consumer contract exactly (its
            schema IS the Parquet schema, including the optional ``embedding`` column when this
            Export emits it) and is version-stamped with the same contract ``schema_version`` the
            Parquet column carries. The plain Parquet + JSONL deliverable remains the source of
            truth and is **byte-identical** whether or not this block is present — the export
            directory itself is untouched; the only additions are an immutable,
            content-addressed data file inside the TARGET TABLE's own warehouse location
            (reported in the output URIs, and deliberately not in the ephemeral
            ``_latence/runs/<run_id>/`` tree that Purge and run-retention delete) and the catalog
            registration. Delta semantics are the ``mode`` choice, documented
            honestly in :mod:`latence_core.iceberg`: ``append`` adds a snapshot (right for a delta
            run; a repeated FULL export duplicates rows), ``overwrite`` replaces the table contents
            in one atomic commit (right for a full re-export). Absent ⇒ nothing is registered,
            ``pyiceberg`` is never imported, and no extra file is written.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The JSONL/Parquet Export materialises the RAG-ready corpus (pyarrow, Apache-2.0).
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = config or {}
        self._basename = str(config.get("basename", "records"))
        # The one opt-out of the PII-masking guarantee (default False ⇒ masked, byte-identical to
        # a build without this key). Read once here; threaded to every corpus-text call site as an
        # explicit argument, never module/global state, so the posture of each call is greppable.
        self._unsafe_unmasked_corpus = bool(config.get("unsafe_unmasked_corpus", False))
        registry = registry or ProviderRegistry()
        self._embedder: Embedder | None = self._resolve("embedder", config, registry)
        # #184 (ADR-0017): the OPTIONAL Iceberg-table output. ``None`` unless an ``iceberg``
        # block is configured — off by default, so the default Export never touches pyiceberg
        # and its files stay byte-identical. Parsed HERE (at wiring time) so a malformed catalog
        # config fails before the pipeline runs, not after it produced its deliverable.
        self._iceberg = build_iceberg_output(config.get("iceberg"), default_table=self._basename)

    @staticmethod
    def _resolve(key: str, config: dict[str, Any], registry: ProviderRegistry) -> Any | None:  # noqa: ANN401 - the concrete type is the slot's declared Protocol
        """Resolve the opt-in Embedder from its nested ``{"provider", "config"}`` block.

        The block's shape, its Provider resolution and its capability check all live in
        :mod:`latence_core.subprovider`, which is also what the offline ``latence stack check``,
        the device-skip pre-flight and the envspec walker read — so the Protocol the slot demands
        is declared ONCE (``EXPORT`` → ``embedder``) instead of being re-stated at every call
        site. Failure modes
        are unchanged: a malformed shape is a :class:`~latence_core.errors.ConfigError`, a Provider
        that does not satisfy the slot's Protocol a :class:`~latence_core.errors.ContractError`,
        both at wiring time and never a silently-absent embedding (ADR-0034).
        """
        return resolve_sub_provider(
            config.get(key),
            capability=CapabilityKind.EXPORT,
            key=key,
            registry=registry,
            owner=_OWNER,
        )

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        """Materialize records to ``<basename>.jsonl`` + ``<basename>.parquet``, streaming.

        Incremental (ADR-0033 / #63): the input is consumed in bounded batches of
        ``_EXPORT_BATCH_ROWS``; each batch is dumped, (optionally) embedded, JSONL-appended,
        and written as one Parquet row group — so the Export's peak memory is O(batch), not the
        old O(corpus) ``from_pylist(whole)`` transient (peak RSS coincided with it). Every file of
        the deliverable rides ONE :meth:`Storage.atomic_writers` block, so a crash mid-Export
        leaves the previous files or the complete new set, never a torn write and never a
        half-published set — and, unlike the two-call ``temp_uri``/``finalize_atomic`` protocol it
        replaces, never a fleet of orphaned ``.tmp`` siblings either: an exception between opening
        the streams and publishing them used to close the handles and abandon every
        temp file, once per failed attempt. The bytes are identical to the buffered path: same
        per-row Pydantic JSON dumps, same sorted-key JSONL line order + trailing-newline rule,
        same flat Parquet schema + values.

        When ``unsafe_unmasked_corpus`` is configured, the corpus text written here is the RAW
        ``content`` (``pii_spans`` / ``masked_content`` still ride along as metadata) and a single
        WARNING is emitted per export call — once, before the first byte, never per row.
        """
        if self._unsafe_unmasked_corpus:
            _LOG.warning(_UNMASKED_CORPUS_WARNING)
        embed = self._embedder is not None
        schema = self._parquet_schema(embed)

        out_dir = out_dir.rstrip("/")
        jsonl_uri = f"{out_dir}/{self._basename}.jsonl"
        parquet_uri = f"{out_dir}/{self._basename}.parquet"
        # The deliverable is ONE file set, so it rides ONE atomic publication: Storage opens each
        # stream on its own temp sibling (creating ``out_dir`` on the way — a fresh run has no
        # export/ yet), publishes nothing until every stream has closed cleanly, and sweeps every
        # temp file if anything raises in between. The export owns none of that.
        stream_uris = [jsonl_uri, parquet_uri]

        wrote_any = False
        # #184: the record schema_versions this run actually carried — stamped onto the optional
        # Iceberg table so it never claims a contract version its own rows do not have. Folded from
        # the rows already in hand (O(distinct versions), normally one), never a second pass.
        record_versions: set[int] = set()
        # Streamed writers over the one atomic publication. The JSONL is byte-stable: one
        # sorted-key object per line, and the trailing newline is written per line (an N-line file
        # ends in exactly one "\n"; an empty file is empty). The Parquet writer is
        # handed its Storage handle directly (pyarrow streams row groups into any binary
        # file-like), so it goes through the same Storage seam. The inner ExitStack holds only the
        # pyarrow writer close and unwinds FIRST (rightmost context exits first), so the footer
        # is flushed before ``atomic_writers`` closes the handles and publishes.
        with (
            storage.atomic_writers(stream_uris) as streams,
            contextlib.ExitStack() as stack,
        ):
            jsonl_fh = streams[0]
            # pyarrow ships no type stubs; the writer calls are dynamically typed.
            writer = pq.ParquetWriter(streams[1], schema)  # type: ignore[no-untyped-call]
            stack.callback(writer.close)

            for batch_rows in self._batched(
                records, _EXPORT_BATCH_ROWS, unmasked=self._unsafe_unmasked_corpus
            ):
                record_versions.update(int(row["schema_version"]) for row in batch_rows)
                self._attach_embeddings(batch_rows)
                for row in batch_rows:
                    line = json.dumps(row, sort_keys=True, ensure_ascii=False)
                    jsonl_fh.write((line + "\n").encode("utf-8"))
                record_batch = self._record_batch(batch_rows, schema, embed)
                writer.write_table(pa.Table.from_batches([record_batch], schema))  # type: ignore[no-untyped-call]
                wrote_any = True
            if not wrote_any:
                # No rows: still write the (empty) schema-only Parquet so a reader gets the
                # column set, matching the buffered path's empty ``pa.table(columns)``.
                writer.write_table(schema.empty_table())  # type: ignore[no-untyped-call]

        # Past the block above every stream is published (or none of them is, and this line was
        # never reached).
        out_uris = [jsonl_uri, parquet_uri]
        if self._iceberg is not None:
            # #184: LAST, and only after the Parquet deliverable is atomically published — the
            # catalog is registered against a file that is already complete and immutable, so a
            # crash before this point leaves the normal deliverable untouched and simply no table.
            out_uris.append(
                self._iceberg.register(
                    storage=storage,
                    parquet_uri=parquet_uri,
                    basename=self._basename,
                    arrow_schema=schema,
                    contract_version=SCHEMA_VERSION,
                    record_versions=record_versions,
                )
            )
        return out_uris

    def _attach_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Attach the opt-in dense ``embedding`` per row in-place (ADR-0017).

        The Embedder sees :func:`_corpus_content` — the PII-safe corpus text this export
        materializes (``masked_content`` when present, else ``content``) — so the vector is
        consistent with the exported text. It honours the Export's ``unsafe_unmasked_corpus``
        opt-out (default False ⇒ the redacted text): when it is on, the vector is computed over
        the RAW corpus text, matching what this export writes. Computed once per batch so the
        transient stays O(batch); with no Embedder wired the rows are untouched (byte-identical
        export).
        """
        if self._embedder is None:
            return
        texts = [_corpus_content(row, unmasked=self._unsafe_unmasked_corpus) for row in rows]
        vectors = list(self._embedder.embed(texts))
        self._require_one_per_row(vectors, rows, "Embedder")
        for row, vector in zip(rows, vectors, strict=True):
            row["embedding"] = vector

    @staticmethod
    def _require_one_per_row(
        produced: list[Any], rows: list[dict[str, Any]], capability: str
    ) -> None:
        """Enforce the one-output-per-input-row contract an Embedder promises."""
        if len(produced) != len(rows):  # pragma: no cover - defensive: Embedder contract
            msg = (
                f"{capability} returned {len(produced)} outputs for {len(rows)} records; "
                f"the contract yields one output per input text, in order."
            )
            raise ValueError(msg)

    @staticmethod
    def _parquet_schema(embed: bool) -> pa.Schema:
        """The flat Parquet schema (ADR-0006): nested contract fields as JSON strings.

        Fixing the schema up front lets the ``ParquetWriter`` stream row-group batches into one
        file (and lets an all-empty corpus still write the column set). The optional embedding
        is a native ``list<double>`` column, added ONLY when an Embedder is wired, so the default
        schema carries no vector column (ADR-0017).
        """
        fields = [
            pa.field("record_id", pa.string()),
            pa.field("schema_version", pa.int64()),
            pa.field("provenance", pa.string()),
            pa.field("classification", pa.string()),
            pa.field("content", pa.string()),
            pa.field("media_type", pa.string()),
        ]
        if embed:
            fields.append(pa.field("embedding", pa.list_(pa.float64())))
        return pa.schema(fields)

    @staticmethod
    def _record_batch(
        rows: list[dict[str, Any]], schema: pa.Schema, embed: bool
    ) -> pa.RecordBatch:
        """One Arrow RecordBatch (a Parquet row group) for a batch of dumped rows.

        Same flat projection as the buffered path: nested Provenance/Classification serialized
        as sorted-key JSON strings, content/media_type verbatim, optional embedding native
        (:meth:`_attach_embeddings` set it on the row).
        """
        flat = [
            {
                "record_id": row["record_id"],
                "schema_version": row["schema_version"],
                "provenance": json.dumps(row["provenance"], sort_keys=True),
                "classification": json.dumps(row["classification"], sort_keys=True),
                "content": row.get("content", ""),
                "media_type": row.get("media_type", ""),
                **({"embedding": row["embedding"]} if embed else {}),
            }
            for row in rows
        ]
        return pa.RecordBatch.from_pylist(flat, schema=schema)

    @staticmethod
    def _batched(
        records: Iterable[Record], size: int, *, unmasked: bool = False
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield the records dumped to JSON-able rows in bounded batches of ``size``.

        The single place the corpus is pulled through: each record is ``model_dump``ed and
        buffered only until the batch fills, so at most ``size`` rows are live at once — the
        O(batch) bound that replaces the old whole-corpus list.

        ``unmasked`` is the Export's ``unsafe_unmasked_corpus`` opt-out (default False ⇒ the
        PII-safe substitution below runs exactly as before): when on, it is SKIPPED so the row
        keeps its raw ``content``, while ``masked_content`` and ``pii_spans`` still ride the dumped
        row as metadata — the detection evidence survives, only the substitution does not.
        """
        batch: list[dict[str, Any]] = []
        for record in records:
            row = record.model_dump(mode="json")
            # W16 (ADR-0042): materialize the PII-safe corpus text as the exported ``content``, so a
            # redacted chunk's raw ``content`` never reaches the corpus (neither the JSONL row nor
            # Parquet ``content`` column — the rag-scan reads every exported cell). Only substitutes
            # when the Redaction Stage set ``masked_content``; a chunk with no Redaction Stage
            # (``masked_content`` None) exports its ``content`` unchanged (byte-identical to today).
            if not unmasked and "content" in row and row.get("masked_content") is not None:
                row["content"] = row["masked_content"]
            batch.append(row)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

"""JSONL + Parquet Export Provider (ADR-0006, S1 spine).

Materializes records in both built-in serializations from the one contract:
JSONL (streaming, human-diffable, checkpoint-friendly) and Parquet/Arrow
(lakehouse handoff — Databricks/Snowflake read Stage outputs natively). Both
are written from the same Pydantic dumps, so they never drift.

**Opt-in embeddings (ADR-0017).** Embeddings are optional: the corpus exports its
text with full Provenance/Classification whether or not an Embedder is wired, so any
vector DB can ingest it. When the Stage ``config`` names an Embedder Provider under
``embedder`` (``{"provider": ..., "config": {...}}``), this Provider resolves that
Embedder from the plugin registry and attaches a fixed-width ``embedding`` vector to
every exported row (JSONL key + Parquet column), embedding the record's **corpus
text** — the ``masked_content`` of a RedactionRecord (the PII-handled variant) or the
``content`` of a Chunk/Document. The in-core ``embedding.hashing`` reference makes the
seam exercisable end to end offline; a served model is the pluggable
``embedding.endpoint`` package, swapped by one config line (the framework names the
seam, not the model).
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from latence_core.bm25 import Bm25Accumulator
from latence_core.capability import (
    ComposingSparseEmbedder,
    Embedder,
    FdeConverter,
    MultiVectorEmbedder,
    ProviderRegistry,
    SparseEmbedder,
    SparseSignals,
    Tokenizer,
)
from latence_core.contracts import SCHEMA_VERSION
from latence_core.errors import ConfigError, ContractError
from latence_core.iceberg import build_iceberg_output
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.capability import SparseVector
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


def _context_header_text(header: dict[str, Any]) -> str:
    """Render a chunk's :class:`~latence_core.contracts.ContextHeader` into a compact embedding
    prefix.

    W13 (ADR-0039): the OPTIONAL ``context_enrichment`` Stage stamps a chunk with a compact header
    (its canonical entities + top-k KG-neighbor triples); the Export prepends THIS rendered text to
    the EMBEDDING input only, so the coherence inflates the vector, never the stored chunk. Rendered
    deterministically (the header's collections are already sorted + capped): an ``Entities:``
    clause
    (``name (type)``) and a ``Relations:`` clause (the rendered ``A —label→ B`` triples). An empty
    header (no entities and no triples) renders ``""`` so a chunk with nothing to add embeds exactly
    its own text (byte-identical to no header).
    """
    entities = header.get("entities") or []
    entity_types = header.get("entity_types") or []
    triples = header.get("neighbor_triples") or []
    parts: list[str] = []
    if entities:
        rendered = ", ".join(
            f"{name} ({etype})" if etype else str(name)
            for name, etype in zip(entities, entity_types, strict=False)
        )
        parts.append(f"Entities: {rendered}")
    if triples:
        parts.append("Relations: " + "; ".join(str(t) for t in triples))
    return " | ".join(parts)


def _corpus_content(row: dict[str, Any]) -> str:
    """The PII-safe corpus text materialized for a row (``content`` column + JSONL).

    A redacted :class:`~latence_core.contracts.ChunkRecord` (W16, ADR-0042) carries the un-redacted
    text in ``content`` and the PII-handled variant in ``masked_content``; a deprecated
    :class:`~latence_core.contracts.RedactionRecord` carries only ``masked_content``. When
    ``masked_content`` is present and non-``None``, it IS the exported corpus text — so the RAG
    corpus never materializes the raw, un-redacted ``content`` (the PII-safety guarantee: the
    stack-validate rag-scan reads every exported cell). A row with no ``masked_content`` (no
    Redaction Stage, or a non-chunk carrier) exports its own ``content`` unchanged — byte-identical
    to a run without Redaction (the additive-field guarantee).
    """
    masked = row.get("masked_content")
    if masked is not None:
        return str(masked)
    return str(row.get("content") or "")


def _corpus_text(row: dict[str, Any]) -> str:
    """The corpus text an Embedder embeds for a row (the PII-safe text + optional W13 header).

    The base is the PII-safe :func:`_corpus_content` (``masked_content`` when present, else
    ``content``), so the vector is consistent with the exported PII-handled text (never the raw,
    un-redacted text).

    W13 (ADR-0039): when the row carries a :class:`~latence_core.contracts.ContextHeader`
    (``context_enrichment`` Stage upstream), its compact rendering is PREPENDED to the embedding
    input ONLY — ``header_text + "\\n" + text`` — so the KG coherence goes into the VECTOR while the
    stored corpus text stays byte-unchanged (the LLM window never grows). A row with no header (or
    an empty one) embeds exactly its own corpus text, byte-identical to today.
    """
    base = _corpus_content(row)
    header = row.get("context_header")
    if isinstance(header, dict):
        header_text = _context_header_text(header)
        if header_text:
            return f"{header_text}\n{base}"
    return base


# The redaction-survival token space (ADR-0053): lower-cased ``\w+`` runs. An entity surface is
# tested as a whole word/phrase against the redacted corpus text, NEVER as a bare substring — so a
# masked name whose letters happen to sit inside a surviving word (``"ann"`` inside ``"annual"``) is
# NOT counted as surviving. The same gate the SPLADE composition Provider applies to its injected
# entity terms (``latence_splade``): one REDACTION-BY-DEFAULT rule, expressed the same way on both
# inspectable sinks.
_WORD = re.compile(r"\w+", re.UNICODE)

# The Context Enricher renders every neighbor triple as ``{src} —{label}→ {tgt}`` (the
# ``context_enrichment`` arrow constants). These markers recover the two ENDPOINT canonical names —
# the corpus surfaces the redaction gate checks — without treating the relation label (not a corpus
# surface) as a name.
_TRIPLE_ARROW_L = " —"
_TRIPLE_ARROW_R = "→ "


def _word_tokens(text: str) -> list[str]:
    """Lower-cased word tokens of ``text`` (the redaction-survival token space, ADR-0053)."""
    return _WORD.findall(text.casefold())


def _occurs_as_phrase(needle: list[str], haystack: list[str]) -> bool:
    """True iff ``needle`` occurs as a contiguous run of word tokens in ``haystack``.

    Word-boundary survival, not substring: a single-token surface must match a whole surviving
    token (``["ann"]`` does NOT survive inside ``["annual"]``), and a multi-word surface must
    survive as an adjacent run. An empty ``needle`` (a punctuation-only surface) never survives; the
    check mirrors the SPLADE provider's redaction gate so the two inspectable sinks apply the
    identical REDACTION-BY-DEFAULT rule.
    """
    n = len(needle)
    if n == 0:
        return False
    return any(haystack[start : start + n] == needle for start in range(len(haystack) - n + 1))


def _triple_endpoint_surfaces(triple: str) -> list[str]:
    """The two endpoint canonical names in a rendered ``A —label→ B`` neighbor triple.

    Split on the exact ``context_enrichment`` arrow markers to recover the SRC and TGT surfaces (the
    corpus names the redaction gate checks) — never the relation label. A string missing either
    marker is unparseable (a hand-built or foreign triple): it is returned whole so the gate treats
    it as one surface and, failing to find it, drops the triple by default (REDACTION-BY-DEFAULT).
    """
    src, sep_l, rest = triple.partition(_TRIPLE_ARROW_L)
    _label, sep_r, tgt = rest.partition(_TRIPLE_ARROW_R)
    if not sep_l or not sep_r:
        return [triple]
    return [src, tgt]


def _redacted_context_header(header: dict[str, Any], redacted_text: str) -> dict[str, Any]:
    """The EXPORTED ContextHeader with PII the corpus redaction removed stripped out (ADR-0053).

    W13 (ADR-0039) builds the header from entities the NER extracted over RAW text (the default
    ``extract_on='unmasked'`` posture), so a person entity's ``canonical_name`` is the REAL name —
    the same one the PII Redaction Stage masked out of ``masked_content``. The header is
    embedding-input for the DENSE vector only (safe — a vector exposes no readable term), but it is
    ALSO written into two INSPECTABLE surfaces (the JSONL ``context_header`` field + the opt-in
    Parquet ``context_header`` column / its ``entities``). Writing the raw names there re-leaks
    exactly what redaction removed (REDACTION-BY-DEFAULT, ADR-0053; the W13 embedding-input-ONLY
    invariant).

    So — mirroring the SPLADE composition's redaction gate (:mod:`latence_splade`) — a NAME is kept
    in the exported header only if its surface still occurs, as a whole word/phrase, in the redacted
    corpus text; a name that no longer occurs was masked and is dropped:

    * ``entities`` (and the aligned ``entity_types`` / ``entity_confidences`` / ``kg_node_ids`` /
      ``kg_node_centrality`` / ``kg_node_community``) — the entry is dropped **together** when the
      entity surface did not survive, so the aligned arrays stay consistent (ADR-0039/0051
      alignment).
    * ``neighbor_triples`` — a rendered ``A —label→ B`` triple is dropped unless BOTH endpoint names
      survive; a masked endpoint (or an external neighbor whose name the local redacted text cannot
      vouch for) is redacted by default. The pseudonymous ``neighbor_node_ids`` (sha256 join keys)
      are RETAINED, so the query-time KG-expansion seam still works off content-addressed ids that
      carry no readable name.

    The safe, non-readable fields (the sha256 ``kg_node_ids`` / ``neighbor_node_ids`` /
    ``kg_node_community`` join keys, the ``kg_node_centrality`` floats, the ``entity_types`` labels)
    carry no canonical name, so they are not a re-leak surface. Invoked only when the row was
    actually redacted (``masked_content`` present); an un-redacted corpus removed nothing to
    re-leak, so its header is exported byte-identical (:func:`_redact_exported_headers`).
    """
    haystack = _word_tokens(redacted_text)

    entities = header.get("entities") or []
    entity_types = header.get("entity_types") or []
    entity_confidences = header.get("entity_confidences") or []
    kg_node_ids = header.get("kg_node_ids") or []
    centrality = header.get("kg_node_centrality") or []
    community = header.get("kg_node_community") or []
    # ADR-0051: graph features are all-or-nothing (empty, or aligned 1:1 with kg_node_ids). Carry
    # them into the survivors only when present, so an unenriched header stays empty (byte-stable).
    has_features = bool(centrality) and bool(community)

    kept_entities: list[Any] = []
    kept_types: list[Any] = []
    kept_confidences: list[Any] = []
    kept_ids: list[Any] = []
    kept_centrality: list[Any] = []
    kept_community: list[Any] = []
    for idx, name in enumerate(entities):
        if not _occurs_as_phrase(_word_tokens(str(name)), haystack):
            continue
        kept_entities.append(name)
        if idx < len(entity_types):
            kept_types.append(entity_types[idx])
        if idx < len(entity_confidences):
            kept_confidences.append(entity_confidences[idx])
        if idx < len(kg_node_ids):
            kept_ids.append(kg_node_ids[idx])
        if has_features and idx < len(centrality):
            kept_centrality.append(centrality[idx])
        if has_features and idx < len(community):
            kept_community.append(community[idx])

    triples = header.get("neighbor_triples") or []
    kept_triples = [
        triple
        for triple in triples
        if all(
            _occurs_as_phrase(_word_tokens(surface), haystack)
            for surface in _triple_endpoint_surfaces(str(triple))
        )
    ]

    redacted = dict(header)
    redacted["entities"] = kept_entities
    redacted["entity_types"] = kept_types
    redacted["entity_confidences"] = kept_confidences
    redacted["kg_node_ids"] = kept_ids
    redacted["kg_node_centrality"] = kept_centrality
    redacted["kg_node_community"] = kept_community
    redacted["neighbor_triples"] = kept_triples
    return redacted


def _context_columns_for_row(row: dict[str, Any]) -> dict[str, Any]:
    """The W13/ADR-0051 Parquet context columns for a dumped row (ADR-0039).

    A store copies every non-text corpus column into its candidate metadata verbatim
    (e.g. ``QdrantBackend._to_candidate``), and the query-time graph consumers — ``expand``
    (:class:`~latence_retrieval.expand.Expander`), ``multihop`` and ``pack`` — read the KG
    projection off ``Candidate.metadata`` **under fixed default keys**. So the columns are named for
    those exact keys, so the graph-augmented-retrieval seam is alive end-to-end on the framework's
    own corpus (ADR-0051's default 1-hop path): naming them anything else leaves the computed graph
    features dead weight the consumers can never read.

    * ``context_header`` — the chunk's :class:`~latence_core.contracts.ContextHeader` as a
      sorted-key JSON string (``""`` when the row has no header), kept for full, opaque fidelity.
    * ``kg_node_ids`` — the chunk's own entity node ids (``expand``/``multihop`` ``node_ids_key``).
    * ``neighbor_node_ids`` — the node ids the chunk's triples reach (``expand``'s
      ``neighbor_ids_key``, the query-time 1-hop expansion keys).
    * ``kg_node_community`` — per-entity community id (``expand``'s ``community_key``, ADR-0051).
    * ``kg_node_centrality`` — per-entity centrality float (``expand``'s ``centrality_key`` and
      ``pack``'s centrality value-signal, ADR-0051).

    Each native list is ``[]`` when the header is absent (a redaction/document row, or a chunk the
    ``context_enrichment`` Stage left unenriched) or when the ADR-0051 graph-feature enrichment was
    OFF (``kg_node_centrality`` / ``kg_node_community`` empty) — the additive / graceful-when-absent
    discipline the consumers rely on. Query-time expansion is still the retriever's job (ADR-0017);
    W13/ADR-0051 only emit the keys it reads.
    """
    header = row.get("context_header")
    if isinstance(header, dict):
        return {
            "context_header": json.dumps(header, sort_keys=True),
            "kg_node_ids": list(header.get("kg_node_ids") or []),
            "neighbor_node_ids": list(header.get("neighbor_node_ids") or []),
            "kg_node_community": list(header.get("kg_node_community") or []),
            "kg_node_centrality": [float(v) for v in (header.get("kg_node_centrality") or [])],
        }
    return {
        "context_header": "",
        "kg_node_ids": [],
        "neighbor_node_ids": [],
        "kg_node_community": [],
        "kg_node_centrality": [],
    }


def _sparse_metadata_for_row(row: dict[str, Any]) -> dict[str, str]:
    """The chunk's selected-metadata map offered to a composing SparseEmbedder (ADR-0053).

    Exposes the row's :class:`~latence_core.contracts.Classification` fields (``category``,
    ``sensitivity``, ``language``) as ``{field: value}`` so an operator's ``metadata_fields`` config
    (e.g. ``["category"]``) selects which are injected as metadata terms. Only non-empty values are
    offered; the composing embedder picks the configured subset, so a field the operator did not
    select is never injected (an unconfigured metadata map composes to no metadata leg).
    """
    metadata: dict[str, str] = {}
    classification = row.get("classification")
    if isinstance(classification, dict):
        for key in ("category", "sensitivity", "language"):
            value = classification.get(key)
            if value:
                metadata[key] = str(value)
    return metadata


def _sparse_signals_for_row(row: dict[str, Any]) -> SparseSignals:
    """Build the row's :class:`~latence_core.capability.SparseSignals` for index-time composition.

    ADR-0053's index-time composition needs the chunk's structured signals, not just its text. Those
    are already projected onto the row:

    * **entities** — the chunk's canonical entity surfaces from its
      :class:`~latence_core.contracts.ContextHeader` (built by the ``context_enrichment`` Stage),
      each offered at its **real** aligned ``entity_confidences`` value — the admission confidence
      the enrichment gate recorded — NOT a hardcoded ``1.0``. Carrying the real confidence is what
      lets a composing embedder's documented ``confidence_threshold`` exclude a weak-NER entity from
      the emitted sparse vector at index time (ADR-0053 / #162 AC-04: weak NER never pollutes the
      sparse signal); stamping every surface at ``1.0`` would make that gate a silent no-op. A
      legacy header carrying no ``entity_confidences`` (or a shorter array) falls back to ``1.0``
      for the unaligned surfaces, preserving the pre-field behaviour. The composing embedder's
      redaction gate then additionally drops any surface that no longer occurs in the redacted
      corpus text, so
      a masked PII surface is never re-injected (REDACTION-BY-DEFAULT, ADR-0053). A row with no
      header carries no entity signal.
    * **metadata** — the row's selected :class:`~latence_core.contracts.Classification` fields
      (:func:`_sparse_metadata_for_row`).

    A row with neither is :data:`~latence_core.capability.EMPTY_SPARSE_SIGNALS`, composing to
    exactly the model leg — identical to the text-only path.
    """
    entities: tuple[tuple[str, float], ...] = ()
    header = row.get("context_header")
    if isinstance(header, dict):
        surfaces = header.get("entities") or []
        confidences = header.get("entity_confidences") or []
        entities = tuple(
            # Real aligned confidence per surface; a legacy/short array falls back to 1.0 (the
            # pre-field behaviour) only for the surfaces it does not cover.
            (str(surface), float(confidences[idx]) if idx < len(confidences) else 1.0)
            for idx, surface in enumerate(surfaces)
        )
    return SparseSignals(entities=entities, metadata=_sparse_metadata_for_row(row))


class JsonlParquetExport:
    """Export Provider writing ``records.jsonl`` and ``records.parquet``.

    Config:
        basename: str  — output file stem (default 'records').
        embedder: dict | None  — OPTIONAL opt-in Embedder augmentation (ADR-0017). A
            ``{"provider": "<entry-point name>", "config": {...}}`` block naming an
            :class:`~latence_core.capability.Embedder` Provider (e.g. the in-core
            ``embedding.hashing`` reference, or the pluggable ``embedding.endpoint``).
            When present, every exported row gets an ``embedding`` vector of the
            Embedder's fixed ``dimension``. Absent ⇒ no embeddings (the corpus text is
            still exported with full Provenance/Classification, so any vector DB can
            ingest it and embed later).
        context_columns: bool  — OPTIONAL opt-in W13/ADR-0051 projection (ADR-0039). When True, the
            KG projection is written as extra Parquet columns so a downstream retriever/vector-DB
            can filter/hybrid-BM25/rerank on it — named for the **exact default metadata keys** the
            ``expand`` / ``multihop`` / ``pack`` consumers read, so a store that copies columns into
            candidate metadata makes the graph-augmented-retrieval seam alive end-to-end (ADR-0051):
            ``context_header`` (the chunk's :class:`~latence_core.contracts.ContextHeader` as a JSON
            string, ``""`` when absent), ``kg_node_ids`` + ``neighbor_node_ids`` (native
            ``list<string>`` — the chunk's own entity node ids and its 1-hop expansion keys), and
            the ADR-0051 graph features ``kg_node_community`` (native ``list<string>``) +
            ``kg_node_centrality`` (native ``list<double>``). Absent/False ⇒ the columns are NOT
            written and the Parquet schema is byte-identical to today (the genuinely-optional
            guarantee — a stack without the ``context_enrichment`` Stage never sets it). The JSONL
            always carries the full ``context_header`` (or ``null``) via the record dump, exactly
            like every other additive field, so no flag is needed there.
        sparse_embedder: dict | None  — OPTIONAL opt-in
            :class:`~latence_core.capability.SparseEmbedder` (ADR-0049/0050/0053). Same
            ``{"provider": ..., "config": {...}}`` shape as ``embedder``.
            When present, every exported row gets parallel ``sparse_indices`` (native
            ``list<int64>``) + ``sparse_values`` (native ``list<double>``) columns — the shape a
            store's sparse index ingests — generated over the row's redacted corpus text. Absent ⇒
            the columns are NOT written (schema byte-identical to today).
        multivector_embedder: dict | None  — OPTIONAL opt-in
            :class:`~latence_core.capability.MultiVectorEmbedder` (ADR-0050, experimental, off by
            default). Same block shape. It is the *source* for both the FDE column and the raw
            multi-vector sidecar; naming it alone does NOT add a column to ``records.parquet`` (the
            raw multi-vectors are heavy — they only reach the sidecar when ``keep_multivectors`` is
            set, and feed the FDE column when ``fde_converter`` is set).
        fde_converter: dict | None  — OPTIONAL opt-in :class:`~latence_core.capability.FdeConverter`
            (ADR-0050, experimental). Requires ``multivector_embedder``. When present, every row
            gets an ``fde_embedding`` (native ``list<double>``) column — the multi-vector to one
            dense vector intended for the customer's *dense* index. Absent ⇒ no ``fde_embedding``.
        keep_multivectors: bool  — OPTIONAL opt-in (ADR-0050, experimental, default False). Requires
            ``multivector_embedder``. When True, the raw per-row multi-vectors are written to a
            **sidecar** ``multivectors.parquet`` (``record_id`` + ``list<list<double>>``), NOT into
            ``records.parquet``. Absent/False ⇒ no sidecar file is written. The toggle is the
            ADR-0050 "FDE-only vs FDE + keep multi-vectors" choice: ``fde_converter`` alone emits
            the dense column; adding ``keep_multivectors`` also persists the raw vectors for later
            exact MaxSim rerank.

        bm25: dict | None  — OPTIONAL opt-in BM25 term-statistics artifact (#164, ADR-0048/0049).
            A ``{"provider": ..., "config": {...}}`` block naming a
            :class:`~latence_core.capability.Tokenizer` Provider (e.g. the in-core
            ``tokenizer.regex`` reference or ``tokenizer.whitespace``). When present, the Export
            emits TWO additive sidecar files a store loads to drive its own first-stage BM25 search
            (ADR-0048: an artifact, **never** a first-stage index here): ``bm25-stats.json`` (the
            corpus term statistics — ``doc_count``, ``avgdl``, ``total_tokens``, and per-term
            ``df`` + precomputed Robertson idf) and ``bm25-postings.parquet`` (per-record forward
            postings: ``record_id``, ``doc_len``, parallel ``terms`` / ``term_frequencies``). The
            statistics are computed over the **stored** redacted corpus text (``masked_content``
            when present, and NOT the embedding-only W13 header — the artifact must match what a
            store indexes, and never re-leak PII, ADR-0044/0053). Absent ⇒ neither file is written
            and the corpus is byte-identical.

        iceberg: dict | None  — OPTIONAL opt-in Iceberg-table output (#184, ADR-0017), OFF by
            default. A ``{"uri": "<Iceberg REST catalog>", "namespace": "latence", "table":
            "records", "mode": "append" | "overwrite", ...}`` block. When present, the Export
            *additionally* registers this run's corpus as an Apache Iceberg table in that REST
            catalog, so the same data is directly governable in Unity/Polaris/Glue/Lakekeeper with
            no glue code. The registered table mirrors the Parquet consumer contract exactly (its
            schema IS the Parquet schema, including whichever optional signal columns this Export
            emits) and is version-stamped with the same contract ``schema_version`` the Parquet
            column carries. The plain Parquet + JSONL deliverable remains the source of truth and is
            **byte-identical** whether or not this block is present; the only additions are an
            immutable, content-addressed data file under ``<out_dir>/iceberg-data/`` (in the output
            URIs) and the catalog registration. Delta semantics are the ``mode`` choice, documented
            honestly in :mod:`latence_core.iceberg`: ``append`` adds a snapshot (right for a delta
            run; a repeated FULL export duplicates rows), ``overwrite`` replaces the table contents
            in one atomic commit (right for a full re-export). Absent ⇒ nothing is registered,
            ``pyiceberg`` is never imported, and no extra file is written.

    Every new signal column/sidecar is **additive**: with none of ``sparse_embedder`` /
    ``multivector_embedder`` / ``fde_converter`` / ``keep_multivectors`` / ``bm25`` wired,
    ``records.jsonl`` + ``records.parquet`` are byte-identical to today and no sidecar is written
    (ADR-0017/0049).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        The JSONL/Parquet Export materialises the RAG-ready corpus (pyarrow, Apache-2.0).
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012 — the
        package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call
        cost, so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
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
        # One shared registry for every opt-in signal generator this Export resolves, so the same
        # roster (and cache) backs the embedder + the retrieval-signal generators.
        registry = registry or ProviderRegistry()
        self._embedder = self._resolve_signal(
            config.get("embedder"), registry, Embedder, "embedder"
        )
        # W13 (ADR-0039): opt-in Parquet projection of the ContextHeader (byte-identical schema when
        # absent — the genuinely-optional guarantee).
        self._context_columns = bool(config.get("context_columns", False))
        # ADR-0049/0050: the three opt-in index-time retrieval-signal generators. Each is additive —
        # absent ⇒ no column/sidecar, and the export is byte-identical to today.
        self._sparse = self._resolve_signal(
            config.get("sparse_embedder"), registry, SparseEmbedder, "sparse_embedder"
        )
        self._multivector = self._resolve_signal(
            config.get("multivector_embedder"),
            registry,
            MultiVectorEmbedder,
            "multivector_embedder",
        )
        self._fde = self._resolve_signal(
            config.get("fde_converter"), registry, FdeConverter, "fde_converter"
        )
        self._keep_multivectors = bool(config.get("keep_multivectors", False))
        # #164 (ADR-0048/0049): the opt-in BM25 term-statistics exporter. Its varying seam is a
        # Tokenizer; the presence of the ``bm25`` block turns it on (additive — absent ⇒ no files,
        # byte-identical export). The tokenizer's provider NAME is stamped into the artifact so it
        # is self-describing (a store must analyse queries the same way).
        bm25_spec = config.get("bm25")
        self._bm25 = self._resolve_signal(bm25_spec, registry, Tokenizer, "bm25")
        self._bm25_tokenizer_name = (
            str(bm25_spec["provider"]) if isinstance(bm25_spec, dict) else ""
        )
        # #184 (ADR-0017): the OPTIONAL Iceberg-table output. ``None`` unless an ``iceberg``
        # block is configured — off by default, so the default Export never touches pyiceberg
        # and its files stay byte-identical. Parsed HERE (at wiring time) so a malformed catalog
        # config fails before the pipeline runs, not after it produced its deliverable.
        self._iceberg = build_iceberg_output(config.get("iceberg"), default_table=self._basename)
        # Wiring invariants: the multi-vector generator is the sole source of both the FDE column
        # and the sidecar, so either sink without it is a config error caught loudly at wiring time
        # (never a silently-empty column/sidecar).
        if self._fde is not None and self._multivector is None:
            msg = (
                "Export config 'fde_converter' requires 'multivector_embedder' (the FDE column is "
                "the multi-vector reduced to one dense vector — there is nothing to convert "
                "without it)."
            )
            raise ConfigError(msg)
        if self._keep_multivectors and self._multivector is None:
            msg = (
                "Export config 'keep_multivectors' requires 'multivector_embedder' (there are no "
                "multi-vectors to persist to the sidecar without it)."
            )
            raise ConfigError(msg)

    @staticmethod
    def _resolve_signal(
        spec: Any,
        registry: ProviderRegistry,
        protocol: type,
        key: str,
    ) -> Any | None:
        """Resolve one opt-in signal generator from a ``{"provider", "config"}`` config block.

        The single resolver shared by the dense Embedder and the three retrieval-signal generators
        (ADR-0049): ``spec`` is ``None`` (the signal is off) or a ``{"provider", "config"}`` map.
        The named Provider is loaded from the ``latence.providers`` registry and built, then checked
        against ``protocol`` (:class:`Embedder` / :class:`SparseEmbedder` /
        :class:`MultiVectorEmbedder` / :class:`FdeConverter`) so a mis-named Provider fails **loudly
        at wiring time** — a :class:`~latence_core.errors.ContractError` (a ``TypeError``) — rather
        than silently emitting no signal. A malformed ``spec`` shape is a
        :class:`~latence_core.errors.ConfigError`.
        """
        if spec is None:
            return None
        if not isinstance(spec, dict):
            msg = (
                f"Export config {key!r} must be a mapping "
                "{'provider': <name>, 'config': {...}} or absent; "
                f"got {type(spec).__name__}."
            )
            raise ConfigError(msg)
        provider_name = spec.get("provider")
        if not isinstance(provider_name, str) or not provider_name:
            msg = f"Export config {key!r} requires a non-empty string 'provider' entry-point name."
            raise ConfigError(msg)
        provider_config = spec.get("config", {})
        if not isinstance(provider_config, dict):
            msg = f"Export config '{key}.config' must be a mapping if present."
            raise ConfigError(msg)
        provider_cls = registry.load(provider_name)
        provider = provider_cls(provider_config)
        if not isinstance(provider, protocol):
            msg = (
                f"Provider {provider_name!r} named in Export config {key!r} does not satisfy the "
                f"{protocol.__name__} capability."
            )
            raise ContractError(msg)
        return provider

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        """Materialize records to ``<basename>.jsonl`` + ``<basename>.parquet``, streaming.

        Incremental (ADR-0033 / #63): the input is consumed in bounded batches of
        ``_EXPORT_BATCH_ROWS``; each batch is dumped, (optionally) embedded, JSONL-appended,
        and written as one Parquet row group — so the Export's peak memory is O(batch), not the
        old O(corpus) ``from_pylist(whole)`` transient (peak RSS coincided with it). Both files
        are written to sibling temp paths and atomically published (:meth:`Storage.temp_uri` +
        :meth:`Storage.finalize_atomic`), so a crash mid-Export leaves the previous file or the
        complete new one, never a torn write — the same WAL guarantee as the buffered path. The
        bytes are identical to the buffered path: same per-row Pydantic JSON dumps, same
        sorted-key JSONL line order + trailing-newline rule, same flat Parquet schema + values.
        """
        embed = self._embedder is not None
        sparse = self._sparse is not None
        fde = self._fde is not None
        sidecar = self._keep_multivectors
        schema = self._parquet_schema(embed, self._context_columns, sparse=sparse, fde=fde)

        out_dir = out_dir.rstrip("/")
        # Streaming opens bypass write_bytes' implicit parent-makedirs, so ensure the export
        # directory exists before opening the temp files (a fresh run has no export/ yet).
        storage.makedirs(out_dir)
        jsonl_uri = f"{out_dir}/{self._basename}.jsonl"
        parquet_uri = f"{out_dir}/{self._basename}.parquet"
        jsonl_tmp = storage.temp_uri(jsonl_uri)
        parquet_tmp = storage.temp_uri(parquet_uri)
        # ADR-0050: raw multi-vectors ride a SIDECAR (heavy, different access pattern), never
        # ``records.parquet``. Written only when ``keep_multivectors`` is set.
        sidecar_uri = f"{out_dir}/multivectors.parquet"
        sidecar_schema = self._sidecar_schema() if sidecar else None
        sidecar_tmp = storage.temp_uri(sidecar_uri) if sidecar else None
        # #164: the BM25 forward postings stream into a sidecar parquet (one row group per batch, so
        # the per-document postings stay O(batch)); the corpus df/avgdl fold into the accumulator
        # (O(vocab)) and become ``bm25-stats.json`` at finalize. Both only when ``bm25`` is wired.
        bm25 = self._bm25 is not None
        bm25_acc = (
            Bm25Accumulator(self._bm25, self._bm25_tokenizer_name)
            if self._bm25 is not None
            else None
        )
        postings_uri = f"{out_dir}/bm25-postings.parquet"
        stats_uri = f"{out_dir}/bm25-stats.json"
        postings_schema = self._bm25_postings_schema() if bm25 else None
        postings_tmp = storage.temp_uri(postings_uri) if bm25 else None

        wrote_any = False
        # #184: the record schema_versions this run actually carried — stamped onto the optional
        # Iceberg table so it never claims a contract version its own rows do not have. Folded from
        # the rows already in hand (O(distinct versions), normally one), never a second pass.
        record_versions: set[int] = set()
        # Streamed writers, all to temp paths, atomically published on success. The JSONL is
        # byte-stable: one sorted-key object per line, and the trailing newline is written per line
        # (an N-line file ends in exactly one "\n"; an empty file is empty). The Parquet + sidecar
        # writers are handed the temp file handle directly (pyarrow streams row groups into any
        # binary file-like), so they go through the same Storage seam. An ExitStack registers each
        # writer's close BEFORE its file handle closes (LIFO), so the pyarrow footer is flushed
        # before the handle shuts — the same ordering the old ``finally: writer.close()`` gave.
        with contextlib.ExitStack() as stack:
            jsonl_fh = stack.enter_context(storage.open(jsonl_tmp, "wb"))
            parquet_fh = stack.enter_context(storage.open(parquet_tmp, "wb"))
            # pyarrow ships no type stubs; the writer calls are dynamically typed.
            writer = pq.ParquetWriter(parquet_fh, schema)  # type: ignore[no-untyped-call]
            stack.callback(writer.close)
            sidecar_writer = None
            if sidecar and sidecar_schema is not None and sidecar_tmp is not None:
                sidecar_fh = stack.enter_context(storage.open(sidecar_tmp, "wb"))
                sidecar_writer = pq.ParquetWriter(sidecar_fh, sidecar_schema)  # type: ignore[no-untyped-call]
                stack.callback(sidecar_writer.close)
            postings_writer = None
            if bm25 and postings_schema is not None and postings_tmp is not None:
                postings_fh = stack.enter_context(storage.open(postings_tmp, "wb"))
                postings_writer = pq.ParquetWriter(postings_fh, postings_schema)  # type: ignore[no-untyped-call]
                stack.callback(postings_writer.close)

            for batch_rows in self._batched(records, _EXPORT_BATCH_ROWS):
                record_versions.update(int(row["schema_version"]) for row in batch_rows)
                multivectors = self._attach_signals(batch_rows)
                # ADR-0053: the dense-vector + SPLADE signals have now consumed the RAW header (the
                # vector is allowed it — non-inspectable). Before the header reaches the INSPECTABLE
                # sinks (the JSONL ``context_header`` field below + the Parquet ``context_header``
                # column via ``_record_batch``), strip out any PII the corpus redaction removed, so
                # a redacted person name is never re-materialised into an inspectable corpus cell.
                self._redact_exported_headers(batch_rows)
                for row in batch_rows:
                    line = json.dumps(row, sort_keys=True, ensure_ascii=False)
                    jsonl_fh.write((line + "\n").encode("utf-8"))
                record_batch = self._record_batch(
                    batch_rows, schema, embed, self._context_columns, sparse=sparse, fde=fde
                )
                writer.write_table(pa.Table.from_batches([record_batch], schema))  # type: ignore[no-untyped-call]
                if sidecar_writer is not None and sidecar_schema is not None:
                    sidecar_batch = self._sidecar_batch(batch_rows, multivectors, sidecar_schema)
                    writer_table = pa.Table.from_batches([sidecar_batch], sidecar_schema)
                    sidecar_writer.write_table(writer_table)  # type: ignore[no-untyped-call]
                if postings_writer is not None and postings_schema is not None and bm25_acc:
                    postings_batch = self._bm25_postings_batch(
                        batch_rows, bm25_acc, postings_schema
                    )
                    postings_writer.write_table(  # type: ignore[no-untyped-call]
                        pa.Table.from_batches([postings_batch], postings_schema)
                    )
                wrote_any = True
            if not wrote_any:
                # No rows: still write the (empty) schema-only Parquet so a reader gets the
                # column set, matching the buffered path's empty ``pa.table(columns)``.
                writer.write_table(schema.empty_table())  # type: ignore[no-untyped-call]
                if sidecar_writer is not None and sidecar_schema is not None:
                    sidecar_writer.write_table(sidecar_schema.empty_table())  # type: ignore[no-untyped-call]
                if postings_writer is not None and postings_schema is not None:
                    postings_writer.write_table(postings_schema.empty_table())  # type: ignore[no-untyped-call]

        storage.finalize_atomic(jsonl_tmp, jsonl_uri)
        storage.finalize_atomic(parquet_tmp, parquet_uri)
        out_uris = [jsonl_uri, parquet_uri]
        if sidecar and sidecar_tmp is not None:
            storage.finalize_atomic(sidecar_tmp, sidecar_uri)
            out_uris.append(sidecar_uri)
        if bm25 and bm25_acc is not None and postings_tmp is not None:
            # The postings parquet is flushed (its writer closed with the ExitStack above); now
            # publish it, then write the corpus stats JSON from the completed fold. Both atomically.
            storage.finalize_atomic(postings_tmp, postings_uri)
            stats_tmp = storage.temp_uri(stats_uri)
            stats_json = json.dumps(
                bm25_acc.stats().to_artifact(), sort_keys=True, ensure_ascii=False
            )
            storage.write_text(stats_tmp, stats_json)
            storage.finalize_atomic(stats_tmp, stats_uri)
            out_uris.extend([postings_uri, stats_uri])
        if self._iceberg is not None:
            # #184: LAST, and only after the Parquet deliverable is atomically published — the
            # catalog is registered against a file that is already complete and immutable, so a
            # crash before this point leaves the normal deliverable untouched and simply no table.
            out_uris.append(
                self._iceberg.register(
                    storage=storage,
                    parquet_uri=parquet_uri,
                    out_dir=out_dir,
                    basename=self._basename,
                    arrow_schema=schema,
                    contract_version=SCHEMA_VERSION,
                    record_versions=record_versions,
                )
            )
        return out_uris

    def _attach_signals(self, rows: list[dict[str, Any]]) -> list[list[list[float]]] | None:
        """Attach the opt-in dense/sparse/FDE signals per row in-place; return the multi-vectors.

        The signals split on the W13 context header (ADR-0039), which is *embedding-input for the
        DENSE vector ONLY* — never a stored, inspectable surface (ADR-0053, REDACTION-BY-DEFAULT):

        * the dense :class:`Embedder` → ``embedding`` (ADR-0017), over :func:`_corpus_text`
          (redacted content **+** the rendered W13 header): the header inflates the vector, which
          exposes no readable term, so no PII is re-leaked;
        * the :class:`SparseEmbedder` → ``sparse_indices`` + ``sparse_values`` (ADR-0050/0053), and
        * the :class:`MultiVectorEmbedder` → the per-row multi-vectors that feed the
          :class:`FdeConverter` → ``fde_embedding`` and (when ``keep_multivectors``) the sidecar —
          both over :func:`_corpus_content` (redacted content, **NO header**), exactly like the BM25
          postings path (:func:`_bm25_postings_batch`). Their emitted columns are ingested into an
          *inspectable* index (sparse term ids; raw per-token vectors), so prepending the header —
          whose KG entity surfaces are the REAL, un-redacted names under the default
          ``extract_on='unmasked'`` posture — would re-leak the PII the corpus stripped. Feeding
          them the redacted content means the signal matches exactly what a store indexes.

        Each variant is computed once per batch so the transient stays O(batch). Returns the batch's
        multi-vectors (aligned to ``rows``) so the caller can write the sidecar, or ``None`` when no
        multi-vector generator is wired. Rows are mutated in place with the additive keys; a row
        NEVER carries the raw multi-vectors (they ride the sidecar only). With no generator wired
        the rows are untouched (byte-identical export — ADR-0017/0049).
        """
        if self._embedder is None and self._sparse is None and self._multivector is None:
            return None
        # Two text views: the DENSE embedder sees the W13 header (vector-only, ADR-0039); every
        # INSPECTABLE index-time signal (sparse, multi-vector) sees the redacted content only,
        # matching what a store indexes — so the artifact never re-leaks the header PII.
        content_texts = [_corpus_content(row) for row in rows]
        if self._embedder is not None:
            embed_texts = [_corpus_text(row) for row in rows]
            vectors = list(self._embedder.embed(embed_texts))
            self._require_one_per_row(vectors, rows, "Embedder")
            for row, vector in zip(rows, vectors, strict=True):
                row["embedding"] = vector
        if self._sparse is not None:
            sparse_vectors = list(self._encode_sparse(content_texts, rows))
            self._require_one_per_row(sparse_vectors, rows, "SparseEmbedder")
            for row, sv in zip(rows, sparse_vectors, strict=True):
                row["sparse_indices"] = list(sv.indices)
                row["sparse_values"] = list(sv.values)
        multivectors: list[list[list[float]]] | None = None
        if self._multivector is not None:
            multivectors = list(self._multivector.encode_multivector(content_texts))
            self._require_one_per_row(multivectors, rows, "MultiVectorEmbedder")
            if self._fde is not None:
                fde_vectors = list(self._fde.convert(multivectors))
                self._require_one_per_row(fde_vectors, rows, "FdeConverter")
                for row, fv in zip(rows, fde_vectors, strict=True):
                    row["fde_embedding"] = list(fv)
        return multivectors

    @staticmethod
    def _redact_exported_headers(rows: list[dict[str, Any]]) -> None:
        """Strip PII the corpus redaction removed back out of every row's EXPORTED ContextHeader.

        The final REDACTION-BY-DEFAULT gate on the header (ADR-0053), applied in place per batch
        AFTER :meth:`_attach_signals` consumed the RAW header for the dense-vector embedding and
        the SPLADE composition (both non-inspectable — the vector exposes no readable term). It runs
        BEFORE the header reaches the two INSPECTABLE sinks — the JSONL ``context_header`` field and
        the opt-in Parquet ``context_header`` column (:func:`_context_columns_for_row`) — so those
        cells never re-materialise a redacted person name (or a relation triple containing one).

        Only rows that were actually redacted (``masked_content`` present) are gated: an un-redacted
        row removed nothing, so its header is left byte-identical to today (the additive /
        byte-identical-when-unwired guarantee). A header-less row is untouched
        (:func:`_redacted_context_header`).
        """
        for row in rows:
            header = row.get("context_header")
            if not isinstance(header, dict):
                continue
            if row.get("masked_content") is None:
                # No Redaction Stage ran for this row: nothing was removed, so nothing to re-leak —
                # export the header raw (byte-identical to a run without this gate).
                continue
            row["context_header"] = _redacted_context_header(header, _corpus_content(row))

    def _encode_sparse(
        self, content_texts: list[str], rows: list[dict[str, Any]]
    ) -> Iterator[SparseVector]:
        """Encode the batch's sparse vectors, applying the ADR-0053 composition when supported.

        The seam split: a :class:`~latence_core.capability.ComposingSparseEmbedder` (the real SPLADE
        composition Provider) also injects each chunk's confidence-gated entity surfaces + selected
        metadata, so the emitted corpus vector is the FULL ADR-0053 composition — symmetric with the
        query-side encoder (one composition, two clocks). It is fed the redacted ``content_texts``
        and each row's :func:`_sparse_signals_for_row`. A plain
        :class:`~latence_core.capability.SparseEmbedder` (e.g. the deterministic ``sparse.hashing``
        reference) has no structured-signal seam, so it is driven text-only — byte-identically to
        today. Both emit over the redacted corpus text, so neither re-leaks the PII the corpus
        removed.
        """
        sparse = self._sparse
        assert sparse is not None  # noqa: S101 - _attach_signals only calls this when wired
        if isinstance(sparse, ComposingSparseEmbedder):
            signals = [_sparse_signals_for_row(row) for row in rows]
            return sparse.encode_sparse_composed(content_texts, signals)
        return iter(sparse.encode_sparse(content_texts))

    @staticmethod
    def _require_one_per_row(
        produced: list[Any], rows: list[dict[str, Any]], capability: str
    ) -> None:
        """Enforce the one-output-per-input-row contract every signal generator promises."""
        if len(produced) != len(rows):  # pragma: no cover - defensive: signal-generator contract
            msg = (
                f"{capability} returned {len(produced)} outputs for {len(rows)} records; "
                f"the contract yields one output per input text, in order."
            )
            raise ValueError(msg)

    @staticmethod
    def _parquet_schema(
        embed: bool,
        context_columns: bool = False,
        *,
        sparse: bool = False,
        fde: bool = False,
    ) -> pa.Schema:
        """The flat Parquet schema (ADR-0006): nested contract fields as JSON strings.

        Fixing the schema up front lets the ``ParquetWriter`` stream row-group batches into one
        file (and lets an all-empty corpus still write the column set). The optional embedding
        is a native ``list<double>`` column so a vector DB reads it directly. The W13/ADR-0051
        context columns (``context_header`` JSON + the ``kg_node_ids`` / ``neighbor_node_ids`` /
        ``kg_node_community`` native ``list<string>`` + ``kg_node_centrality`` native
        ``list<double>`` — named for the consumers' default metadata keys) are added ONLY when opted
        in, so the default schema is byte-identical to today (ADR-0039/0051). The
        retrieval-signal columns (ADR-0050) — SPLADE ``sparse_indices`` (native ``list<int64>``) +
        ``sparse_values`` (native ``list<double>``), and the MUVERA ``fde_embedding`` (native
        ``list<double>``) — are likewise added ONLY when their generator is wired, so the default
        schema is byte-identical to today (ADR-0017/0049).
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
        if sparse:
            fields.append(pa.field("sparse_indices", pa.list_(pa.int64())))
            fields.append(pa.field("sparse_values", pa.list_(pa.float64())))
        if fde:
            fields.append(pa.field("fde_embedding", pa.list_(pa.float64())))
        if context_columns:
            fields.append(pa.field("context_header", pa.string()))
            fields.append(pa.field("kg_node_ids", pa.list_(pa.string())))
            fields.append(pa.field("neighbor_node_ids", pa.list_(pa.string())))
            fields.append(pa.field("kg_node_community", pa.list_(pa.string())))
            fields.append(pa.field("kg_node_centrality", pa.list_(pa.float64())))
        return pa.schema(fields)

    @staticmethod
    def _sidecar_schema() -> pa.Schema:
        """The multi-vector sidecar schema (ADR-0050): ``record_id`` + ``list<list<double>>``.

        The raw multi-vectors are heavy and have a different access pattern than the corpus, so
        they ride ``multivectors.parquet`` keyed by ``record_id`` rather than ``records.parquet``.
        """
        return pa.schema(
            [
                pa.field("record_id", pa.string()),
                pa.field("multivector", pa.list_(pa.list_(pa.float64()))),
            ]
        )

    @staticmethod
    def _record_batch(
        rows: list[dict[str, Any]],
        schema: pa.Schema,
        embed: bool,
        context_columns: bool = False,
        *,
        sparse: bool = False,
        fde: bool = False,
    ) -> pa.RecordBatch:
        """One Arrow RecordBatch (a Parquet row group) for a batch of dumped rows.

        Same flat projection as the buffered path: nested Provenance/Classification serialized
        as sorted-key JSON strings, content/media_type verbatim, optional embedding native. When
        the W13/ADR-0051 context columns are opted in, ``context_header`` is the chunk's
        ContextHeader as a sorted-key JSON string (``""`` when absent) and the graph columns
        (``kg_node_ids`` / ``neighbor_node_ids`` / ``kg_node_community`` / ``kg_node_centrality``)
        carry the header's KG projection under the consumers' default metadata keys (``[]`` when
        absent).
        The retrieval-signal columns (``sparse_indices``/``sparse_values``, ``fde_embedding``) are
        included only when their generator is wired (:meth:`_attach_signals` set them on the row).
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
                **(
                    {
                        "sparse_indices": row["sparse_indices"],
                        "sparse_values": row["sparse_values"],
                    }
                    if sparse
                    else {}
                ),
                **({"fde_embedding": row["fde_embedding"]} if fde else {}),
                **(_context_columns_for_row(row) if context_columns else {}),
            }
            for row in rows
        ]
        return pa.RecordBatch.from_pylist(flat, schema=schema)

    @staticmethod
    def _sidecar_batch(
        rows: list[dict[str, Any]],
        multivectors: list[list[list[float]]] | None,
        schema: pa.Schema,
    ) -> pa.RecordBatch:
        """One sidecar RecordBatch: each row's ``record_id`` + its raw multi-vector (ADR-0050).

        ``multivectors`` is aligned to ``rows`` (both come from the same batch); it is never
        ``None`` here because the sidecar is written only when a MultiVectorEmbedder is wired.
        """
        assert multivectors is not None  # noqa: S101 - the sidecar only runs when multi-vectors set
        flat = [
            {"record_id": row["record_id"], "multivector": mv}
            for row, mv in zip(rows, multivectors, strict=True)
        ]
        return pa.RecordBatch.from_pylist(flat, schema=schema)

    @staticmethod
    def _bm25_postings_schema() -> pa.Schema:
        """The ``bm25-postings.parquet`` schema (#164): per-record forward BM25 postings.

        ``record_id`` + ``doc_len`` + parallel ``terms`` (native ``list<string>``) / term
        ``term_frequencies`` (native ``list<int64>``), the canonical shape a store inverts into its
        BM25 index (the corpus-level df/avgdl ride the companion ``bm25-stats.json``).
        """
        return pa.schema(
            [
                pa.field("record_id", pa.string()),
                pa.field("doc_len", pa.int64()),
                pa.field("terms", pa.list_(pa.string())),
                pa.field("term_frequencies", pa.list_(pa.int64())),
            ]
        )

    @staticmethod
    def _bm25_postings_batch(
        rows: list[dict[str, Any]],
        accumulator: Bm25Accumulator,
        schema: pa.Schema,
    ) -> pa.RecordBatch:
        """Fold a batch into the BM25 accumulator; return its postings as one Arrow RecordBatch.

        The BM25 statistics are generated over each row's **stored** redacted corpus text
        (:func:`_corpus_content` — ``masked_content`` when present, so the artifact never re-leaks
        PII the corpus removed, and NOT the embedding-only W13 header, so the postings match exactly
        what a store indexes). The accumulator updates the corpus df/avgdl fold (O(vocab)) and
        returns one canonical :class:`~latence_core.bm25.Bm25Posting` per row, streamed here as a
        row group so the per-document postings stay O(batch).
        """
        postings = accumulator.add_batch(
            (row["record_id"], _corpus_content(row)) for row in rows
        )
        return pa.RecordBatch.from_pylist([p.as_row() for p in postings], schema=schema)

    @staticmethod
    def _batched(records: Iterable[Record], size: int) -> Iterator[list[dict[str, Any]]]:
        """Yield the records dumped to JSON-able rows in bounded batches of ``size``.

        The single place the corpus is pulled through: each record is ``model_dump``ed and
        buffered only until the batch fills, so at most ``size`` rows are live at once — the
        O(batch) bound that replaces the old whole-corpus list.
        """
        batch: list[dict[str, Any]] = []
        for record in records:
            row = record.model_dump(mode="json")
            # W16 (ADR-0042): materialize the PII-safe corpus text as the exported ``content``, so a
            # redacted chunk's raw ``content`` never reaches the corpus (neither the JSONL row nor
            # Parquet ``content`` column — the rag-scan reads every exported cell). Only substitutes
            # when the Redaction Stage set ``masked_content``; a chunk with no Redaction Stage
            # (``masked_content`` None) exports its ``content`` unchanged (byte-identical to today).
            if "content" in row and row.get("masked_content") is not None:
                row["content"] = row["masked_content"]
            batch.append(row)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

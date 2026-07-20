"""Index-time retrieval-signal generation + additive Export emission (#160).

The seam + emission plumbing tested THROUGH the interface (the Export Provider's `export`, the
three signal-generator Capabilities, and the reference emitters), never past it:

* the additive columns/sidecar appear ONLY when their generator is wired, and carry the right
  shapes (parallel `sparse_indices`/`sparse_values`; `fde_embedding` dense; a `multivectors.parquet`
  sidecar keyed by `record_id`);
* with NO generator wired the corpus is **byte-identical** to a plain export (ADR-0017/0049) — the
  load-bearing "additive" guarantee;
* the signals are generated over the redacted corpus text (`masked_content`), so they never re-leak
  PII the corpus removed (ADR-0053);
* the reference emitters are deterministic (a seeded run is byte-reproducible);
* the wiring invariants bite: `fde_converter`/`keep_multivectors` without a `multivector_embedder`,
  and a mis-named Provider, fail loudly (never a silently-empty column).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
from latence_core.capability import ProviderRegistry, SparseVector
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    ContextHeader,
    Provenance,
    RedactionRecord,
)
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


def _read_jsonl(storage: Storage, uris: list[str]) -> list[dict[str, Any]]:
    jsonl_uri = next(u for u in uris if u.endswith(".jsonl"))
    return [json.loads(line) for line in storage.read_text(jsonl_uri).splitlines() if line.strip()]


def _read_parquet_columns(storage: Storage, uris: list[str], stem: str = "corpus") -> list[str]:
    parquet_uri = next(u for u in uris if u.endswith(f"{stem}.parquet"))
    with storage.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]
    return list(table.column_names)


# ---------------------------------------------------------------------------
# Byte-identical default: no generator wired ⇒ corpus unchanged (ADR-0017/0049)
# ---------------------------------------------------------------------------


def test_no_generator_wired_is_byte_identical(tmp_path: Path) -> None:
    """With none of the signal generators wired the export bytes match a plain export exactly."""
    records = [_chunk("alpha beta", "c1", 0), _chunk("gamma delta", "c2", 1)]

    plain_store = f"file://{tmp_path}/plain"
    plain_storage = Storage.for_uri(plain_store)
    plain_uris = JsonlParquetExport({"basename": "corpus"}).export(records, plain_storage, plain_store)

    # An Export constructed with the signal keys ABSENT (the default) must produce identical bytes.
    signal_store = f"file://{tmp_path}/signal"
    signal_storage = Storage.for_uri(signal_store)
    signal_uris = JsonlParquetExport({"basename": "corpus"}).export(
        [_chunk("alpha beta", "c1", 0), _chunk("gamma delta", "c2", 1)], signal_storage, signal_store
    )

    for plain_uri, signal_uri in zip(sorted(plain_uris), sorted(signal_uris), strict=True):
        assert plain_storage.read_bytes(plain_uri) == signal_storage.read_bytes(signal_uri)
    # And no sidecar file was written.
    assert not any(u.endswith("multivectors.parquet") for u in signal_uris)


def test_sparse_columns_absent_by_default(tmp_path: Path) -> None:
    """A plain corpus export carries none of the retrieval-signal columns."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    uris = JsonlParquetExport({"basename": "corpus"}).export([_chunk("alpha")], storage, store)
    cols = _read_parquet_columns(storage, uris)
    for absent in ("sparse_indices", "sparse_values", "fde_embedding"):
        assert absent not in cols
    rows = _read_jsonl(storage, uris)
    assert all(
        k not in rows[0] for k in ("sparse_indices", "sparse_values", "fde_embedding", "multivector")
    )


# ---------------------------------------------------------------------------
# Sparse emission (ADR-0050/0053)
# ---------------------------------------------------------------------------


def test_sparse_embedder_emits_parallel_columns(tmp_path: Path) -> None:
    """A wired SparseEmbedder attaches parallel sparse_indices/sparse_values per row (JSONL+Parquet)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport(
        {"basename": "corpus", "sparse_embedder": {"provider": "sparse.hashing", "config": {}}}
    )
    uris = export.export([_chunk("alpha beta alpha", "c1"), _chunk("", "c2")], storage, store)

    rows = _read_jsonl(storage, uris)
    assert len(rows) == 2
    for row in rows:
        assert "sparse_indices" in row and "sparse_values" in row
        assert len(row["sparse_indices"]) == len(row["sparse_values"])
        # canonical ascending, unique indices
        assert row["sparse_indices"] == sorted(set(row["sparse_indices"]))
    # "alpha beta alpha" has terms; the empty chunk is the empty sparse vector.
    assert rows[0]["sparse_indices"] and rows[1]["sparse_indices"] == []

    cols = _read_parquet_columns(storage, uris)
    assert "sparse_indices" in cols and "sparse_values" in cols
    with storage.open(next(u for u in uris if u.endswith("corpus.parquet")), "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]
    # The Parquet sparse_indices is a native int list, sparse_values a native double list.
    assert all(isinstance(i, int) for i in table.column("sparse_indices")[0].as_py())


def test_sparse_signal_is_over_redacted_text(tmp_path: Path) -> None:
    """The sparse signal is generated over masked_content, never the raw text (ADR-0053)."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    export = JsonlParquetExport(
        {"basename": "corpus", "sparse_embedder": {"provider": "sparse.hashing", "config": {}}}
    )
    masked = "Contact [EMAIL] today"
    uris = export.export([_redaction(masked)], storage, store)
    row = _read_jsonl(storage, uris)[0]

    from latence_core.signals import SparseHashingEmbedder

    expected = next(SparseHashingEmbedder({}).encode_sparse([masked]))
    assert row["sparse_indices"] == expected.indices
    assert row["sparse_values"] == expected.values


def _redacted_chunk_with_header(
    *, masked: str, header: ContextHeader, record_id: str = "c1"
) -> ChunkRecord:
    """A W16-redacted chunk (masked_content exported) that also carries a W13 context header.

    ``content`` keeps the raw text and the header keeps the raw KG entity surfaces — the DEFAULT
    ``extract_on='unmasked'`` posture — while ``masked_content`` is the PII-stripped corpus text.
    """
    return ChunkRecord(
        record_id=record_id,
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=100,
            document_id="sha256:d1",
            char_start=0,
            char_end=len(masked),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=0,
        content="Bruce Wayne was admitted on Tuesday",
        masked_content=masked,
        token_count=6,
        context_header=header,
    )


def test_sparse_signal_never_encodes_the_w13_header_pii(tmp_path: Path) -> None:
    """R1-invariants-0: the sparse signal encodes ONLY the redacted content, never the W13 header.

    Under the default ``extract_on='unmasked'`` posture the context header carries the REAL,
    un-redacted KG entity surface ("Bruce Wayne"). Prepending it to the sparse embedder's input (the
    old ``_corpus_text`` feed) leaked term ids for ``bruce``/``wayne`` into the inspectable sparse
    index — tokens absent from the stored/redacted corpus text (ADR-0053). The sparse signal must be
    byte-identical to encoding ``masked_content`` alone, exactly like the BM25 postings path.
    """
    from latence_core.signals import SparseHashingEmbedder

    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    masked = "Patient [REDACTED] was admitted on Tuesday"
    header = ContextHeader(
        entities=["Bruce Wayne", "Dr Thompkins"],
        entity_types=["PERSON", "PERSON"],
        neighbor_triples=["Bruce Wayne —treated_by→ Dr Thompkins"],
        kg_node_ids=["n-bruce", "n-thompkins"],
        neighbor_node_ids=["n-thompkins"],
    )
    # A wide vocab so distinct tokens never share a term id — the leak, if any, is unambiguous.
    sparse_config = {"vocab_size": 1 << 20}
    export = JsonlParquetExport(
        {
            "basename": "corpus",
            "sparse_embedder": {"provider": "sparse.hashing", "config": sparse_config},
        }
    )
    uris = export.export(
        [_redacted_chunk_with_header(masked=masked, header=header)], storage, store
    )
    row = _read_jsonl(storage, uris)[0]

    # Strong equality: the emitted sparse vector IS the encoding of the redacted content alone,
    # with NO header prefix — the same input a store's BM25/sparse index would see.
    embedder = SparseHashingEmbedder(sparse_config)
    expected = next(embedder.encode_sparse([masked]))
    assert row["sparse_indices"] == expected.indices
    assert row["sparse_values"] == expected.values

    # And crisply: not one header-only PII term id leaked into the inspectable sparse vector. The
    # surfaces "bruce"/"wayne"/"thompkins" are in the header but NOT in the redacted corpus text.
    leaked = next(embedder.encode_sparse(["bruce wayne thompkins"]))
    assert leaked.indices, "guard: the PII surfaces must tokenize to real term ids"
    assert not (set(leaked.indices) & set(row["sparse_indices"])), (
        "W13 header PII term ids leaked into the sparse index (REDACTION-BY-DEFAULT, ADR-0053)"
    )


# ---------------------------------------------------------------------------
# FDE emission + multi-vector sidecar (ADR-0050)
# ---------------------------------------------------------------------------


def _mv_fde_config() -> dict[str, object]:
    return {
        "basename": "corpus",
        "multivector_embedder": {"provider": "multivector.hashing", "config": {"dimension": 8}},
        "fde_converter": {"provider": "fde.reference", "config": {"input_dimension": 8, "buckets": 4}},
    }


def test_fde_column_emitted_but_no_sidecar_by_default(tmp_path: Path) -> None:
    """`fde_converter` emits the fde_embedding dense column; the raw multi-vectors stay OUT of it."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    uris = JsonlParquetExport(_mv_fde_config()).export([_chunk("alpha beta gamma")], storage, store)

    rows = _read_jsonl(storage, uris)
    assert "fde_embedding" in rows[0]
    assert len(rows[0]["fde_embedding"]) == 4 * 8  # buckets * input_dimension
    # The heavy raw multi-vectors are NOT inlined into the corpus row.
    assert "multivector" not in rows[0]
    # No sidecar unless keep_multivectors.
    assert not any(u.endswith("multivectors.parquet") for u in uris)
    assert not storage.exists(f"{store}/multivectors.parquet")


def test_keep_multivectors_writes_sidecar(tmp_path: Path) -> None:
    """`keep_multivectors` writes a sidecar keyed by record_id with the raw list<list<double>>."""
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    config = {**_mv_fde_config(), "keep_multivectors": True}
    records = [_chunk("alpha beta", "c1"), _chunk("gamma", "c2")]
    uris = JsonlParquetExport(config).export(records, storage, store)

    sidecar_uri = next(u for u in uris if u.endswith("multivectors.parquet"))
    with storage.open(sidecar_uri, "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]
    assert table.column_names == ["record_id", "multivector"]
    ids = table.column("record_id").to_pylist()
    assert ids == ["c1", "c2"]
    # "alpha beta" → 2 token vectors of width 8; "gamma" → 1.
    mv0 = table.column("multivector")[0].as_py()
    assert len(mv0) == 2 and all(len(tok) == 8 for tok in mv0)
    assert len(table.column("multivector")[1].as_py()) == 1
    # The raw multi-vectors are NOT in records.parquet.
    assert "multivector" not in _read_parquet_columns(storage, uris)


# ---------------------------------------------------------------------------
# Determinism through the interface
# ---------------------------------------------------------------------------


def test_signal_export_is_deterministic(tmp_path: Path) -> None:
    """Two exports of the same records with the same generators are byte-identical (all artifacts)."""
    config = {**_mv_fde_config(), "keep_multivectors": True,
              "sparse_embedder": {"provider": "sparse.hashing", "config": {}}}
    records = [_chunk("alpha beta alpha", "c1"), _chunk("gamma delta", "c2")]

    outputs: list[dict[str, bytes]] = []
    for run in ("a", "b"):
        store = f"file://{tmp_path}/{run}"
        storage = Storage.for_uri(store)
        uris = JsonlParquetExport(dict(config)).export(list(records), storage, store)
        outputs.append({u.rsplit("/", 1)[-1]: storage.read_bytes(u) for u in uris})
    assert outputs[0] == outputs[1] and outputs[0]


# ---------------------------------------------------------------------------
# Wiring invariants bite (loud failure, never a silently-empty column)
# ---------------------------------------------------------------------------


def test_fde_without_multivector_is_a_config_error() -> None:
    """`fde_converter` without `multivector_embedder` fails loudly at wiring time."""
    with pytest.raises(ValueError, match="requires 'multivector_embedder'"):
        JsonlParquetExport(
            {"fde_converter": {"provider": "fde.reference", "config": {"input_dimension": 8}}}
        )


def test_keep_multivectors_without_multivector_is_a_config_error() -> None:
    """`keep_multivectors` without `multivector_embedder` fails loudly."""
    with pytest.raises(ValueError, match="requires 'multivector_embedder'"):
        JsonlParquetExport({"keep_multivectors": True})


def test_misnamed_sparse_provider_fails_loudly() -> None:
    """A non-SparseEmbedder Provider named as sparse_embedder raises (TypeError), never no-ops."""
    with pytest.raises(TypeError):
        JsonlParquetExport(
            {"sparse_embedder": {"provider": "embedding.hashing", "config": {}}}
        )
    with pytest.raises(KeyError):
        JsonlParquetExport(
            {"sparse_embedder": {"provider": "sparse.nonexistent", "config": {}}}
        )


class _LazyBackendSparseEmbedder:
    """A SparseEmbedder shaped exactly like ``sparse.splade``: ``vocab_size`` loads the backend.

    ``vocab_size`` is the ONLY member whose real implementation touches the heavy backend, and it
    is a **property** — so any ``hasattr``-based capability check invokes it. This stands in for
    ``latence_splade.SpladeSparseEmbedder``, whose ``vocab_size`` calls
    ``AutoModelForMaskedLM.from_pretrained(...)``.
    """

    reads = 0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    @property
    def vocab_size(self) -> int:
        type(self).reads += 1
        msg = "the heavy backend is required to run this provider"
        raise ImportError(msg)

    def encode_sparse(self, texts: Any) -> Any:
        for _ in texts:
            yield SparseVector(indices=[1], values=[1.0])


class _NoVocabSizeSparseEmbedder:
    """Carries ``encode_sparse`` but NOT ``vocab_size`` — must still be rejected."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def encode_sparse(self, texts: Any) -> Any:
        for _ in texts:
            yield SparseVector(indices=[1], values=[1.0])


def test_wiring_a_signal_provider_never_invokes_its_capability_property() -> None:
    """Audit R6: constructing an Export must not READ a resolved Provider's Protocol property.

    ``Export._resolve_signal`` used plain ``isinstance(provider, SparseEmbedder)``. CPython
    implements a ``runtime_checkable`` Protocol check with ``hasattr`` over the Protocol's members,
    and ``SparseEmbedder`` declares ``vocab_size`` as a **property** — so the capability check
    *called* it. For the committed ``deploy/env/cpu-text`` stack that meant
    ``JsonlParquetExport.__init__`` pulled a Hugging Face checkpoint (network!) during what
    ADR-0016 promises is a cheap, offline, config-only construction, and made the R5
    environment-stack guard silently skip the whole Export behind its ``ImportError`` carve-out.

    On the old code this FAILS with that exact ImportError escaping ``__init__``.
    """
    _LazyBackendSparseEmbedder.reads = 0
    export = JsonlParquetExport(
        {"basename": "corpus", "sparse_embedder": {"provider": "sparse.lazy", "config": {}}},
        registry=_registry_with("sparse.lazy", _LazyBackendSparseEmbedder),
    )
    assert export is not None
    assert _LazyBackendSparseEmbedder.reads == 0, (
        "wiring read the Provider's 'vocab_size' property — the capability check is invoking "
        "members instead of merely resolving them"
    )


def test_a_provider_missing_the_capability_property_is_still_rejected() -> None:
    """The non-invoking check is not a weaker check: an absent property still fails loudly."""
    with pytest.raises(TypeError, match="SparseEmbedder"):
        JsonlParquetExport(
            {"sparse_embedder": {"provider": "sparse.novocab", "config": {}}},
            registry=_registry_with("sparse.novocab", _NoVocabSizeSparseEmbedder),
        )


def test_sparse_vector_enforces_canonical_invariants() -> None:
    """The SparseVector contract rejects malformed vectors (length/order) at construction."""
    with pytest.raises(ValueError, match="length mismatch"):
        SparseVector(indices=[1, 2], values=[0.5])
    with pytest.raises(ValueError, match="strictly ascending"):
        SparseVector(indices=[2, 1], values=[0.5, 0.5])
    # from_terms sorts + drops zeros into the canonical form.
    sv = SparseVector.from_terms({5: 1.0, 1: 2.0, 3: 0.0})
    assert sv.indices == [1, 5] and sv.values == [2.0, 1.0]


# ---------------------------------------------------------------------------
# R1: the index-time SPLADE composition (ADR-0053) is applied at export, so the
# emitted corpus sparse vector carries the boosted entity + selected-metadata terms
# — symmetric with the query-side encoder (one composition, two clocks).
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
_FAKE_VOCAB = 1 << 20


def _fake_tid(token: str) -> int:
    """A byte-stable vocab id for a token (sha256, not the salted builtin hash)."""
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % _FAKE_VOCAB


class _FakeComposingSparseEmbedder:
    """A faithful, torch-free ADR-0053 composing ``SparseEmbedder`` for the Export seam.

    Second adapter of the composing seam (the real one is ``latence-splade``'s
    ``SpladeCompositionEmbedder``): it MIRRORS the real observable behaviour —

    * ``encode_sparse(texts)`` is the **model leg only** (term frequencies over the fixed vocab), the
      honest text-only path a plain ``SparseEmbedder`` offers;
    * ``encode_sparse_composed(texts, signals)`` unions the model leg with the confidence-gated,
      redaction-gated entity surfaces (a surface below ``confidence_threshold`` OR absent from the
      redacted text is dropped) and the operator-selected ``metadata_fields`` values, each boosted —
      the ADR-0053 composition.

    It holds no state and names no model; the test drives it THROUGH the Export interface.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._entity_boost = float(cfg.get("entity_boost", 1.0))
        self._metadata_boost = float(cfg.get("metadata_boost", 1.0))
        self._confidence_threshold = float(cfg.get("confidence_threshold", 0.5))
        self._metadata_fields = [str(f) for f in (cfg.get("metadata_fields") or [])]

    @property
    def vocab_size(self) -> int:
        return _FAKE_VOCAB

    @staticmethod
    def _model_terms(text: str) -> dict[int, float]:
        terms: dict[int, float] = {}
        for token in _TOKEN.findall(text.lower()):
            idx = _fake_tid(token)
            terms[idx] = terms.get(idx, 0.0) + 1.0
        return terms

    def encode_sparse(self, texts: Any) -> Any:
        for text in texts:
            yield SparseVector.from_terms(self._model_terms(text))

    def encode_sparse_composed(self, texts: Any, signals: Any) -> Any:
        text_list = list(texts)
        signal_list = list(signals)
        assert len(text_list) == len(signal_list)
        for text, sig in zip(text_list, signal_list, strict=True):
            terms = self._model_terms(text)
            haystack = text.lower()
            for surface, confidence in sig.entities:
                if confidence < self._confidence_threshold:
                    continue
                if surface.lower() not in haystack:  # the redaction gate
                    continue
                for token in _TOKEN.findall(surface.lower()):
                    idx = _fake_tid(token)
                    terms[idx] = terms.get(idx, 0.0) + self._entity_boost
            for field_name in self._metadata_fields:
                value = sig.metadata.get(field_name)
                if not value:
                    continue
                for token in _TOKEN.findall(str(value).lower()):
                    idx = _fake_tid(token)
                    terms[idx] = terms.get(idx, 0.0) + self._metadata_boost
            yield SparseVector.from_terms(terms)


def _registry_with(name: str, provider_cls: type) -> ProviderRegistry:
    """A real ProviderRegistry with one provider pre-loaded under ``name`` (no entry-point needed)."""
    registry = ProviderRegistry()
    registry._loaded[name] = provider_cls  # noqa: SLF001 - test seam: preload a provider class
    return registry


def _chunk_with_entities_and_category(
    *, content: str, masked: str, entities: list[str], category: str
) -> ChunkRecord:
    """A redacted chunk carrying a W13 header (its canonical entities) + a classification category."""
    header = ContextHeader(
        entities=entities,
        entity_types=["ROLE"] * len(entities),
    )
    return ChunkRecord(
        record_id="c1",
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=100,
            document_id="sha256:d1",
            char_start=0,
            char_end=len(masked),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="en", category=category, sensitivity="public"),
        document_record_id="d1",
        index=0,
        content=content,
        masked_content=masked,
        token_count=max(1, len(masked.split())),
        context_header=header,
    )


def test_index_time_sparse_composition_injects_entity_and_metadata_terms(tmp_path: Path) -> None:
    """R1: a composing SparseEmbedder's emitted corpus vector carries the ADR-0053 composition.

    Regression for the audit defect: the Export → SparseEmbedder seam was text-only, so the emitted
    ``sparse_indices``/``sparse_values`` were plain ``SPLADE(text)`` — the confidence-gated entity
    terms and the selected metadata terms the query side injects existed ONLY in the query vector, so
    a hybrid/sparse search on an injected term matched zero documents. This asserts the OBSERVABLE
    emitted vector, not a flag: the injected metadata term id is present in the corpus vector and
    absent from the text-only encoding; the entity term id is boosted; a redacted entity surface is
    dropped. It FAILS on the old (text-only) code and passes on the composed seam.
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    # "wayne" is a header entity that was REDACTED out of the corpus text (present in raw content,
    # gone from masked_content) — the redaction gate must drop it. "plaintiff" survives redaction.
    content = "the plaintiff filed a motion at wayne manor"
    masked = "the plaintiff filed a motion"
    chunk = _chunk_with_entities_and_category(
        content=content, masked=masked, entities=["plaintiff", "wayne"], category="litigation"
    )
    sparse_cfg = {"entity_boost": 5.0, "metadata_boost": 7.0, "metadata_fields": ["category"]}
    export = JsonlParquetExport(
        {
            "basename": "corpus",
            "sparse_embedder": {"provider": "sparse.fakecomposing", "config": sparse_cfg},
        },
        registry=_registry_with("sparse.fakecomposing", _FakeComposingSparseEmbedder),
    )
    uris = export.export([chunk], storage, store)
    row = _read_jsonl(storage, uris)[0]
    emitted = dict(zip(row["sparse_indices"], row["sparse_values"], strict=True))

    # The text-only baseline: what the old seam emitted (model leg over the redacted text).
    baseline = next(_FakeComposingSparseEmbedder(sparse_cfg).encode_sparse([masked]))
    baseline_map = dict(zip(baseline.indices, baseline.values, strict=True))

    # 1. The selected metadata term ("litigation") is injected — an id ABSENT from the text-only
    #    encoding (it does not occur in the corpus text), so it can only be there via composition.
    litigation_id = _fake_tid("litigation")
    assert litigation_id not in baseline_map, "guard: the metadata term is not in the corpus text"
    assert emitted.get(litigation_id) == 7.0, (
        "the selected metadata term was NOT injected at index time (ADR-0053 composition inert)"
    )

    # 2. The surviving entity ("plaintiff") is boosted: model weight (1) + entity_boost (5).
    plaintiff_id = _fake_tid("plaintiff")
    assert baseline_map.get(plaintiff_id) == 1.0
    assert emitted.get(plaintiff_id) == 6.0, "the entity surface was not boosted at index time"

    # 3. REDACTION-BY-DEFAULT: the redacted-out entity ("wayne") is NOT injected.
    assert _fake_tid("wayne") not in emitted, (
        "a redacted entity surface leaked into the sparse index (REDACTION-BY-DEFAULT, ADR-0053)"
    )

    # 4. Field selection: an unselected classification field ("public" sensitivity) is not injected.
    assert _fake_tid("public") not in emitted, "an unselected metadata field was injected"


def _chunk_with_entity_confidences(
    *, text: str, entities: list[str], confidences: list[float]
) -> ChunkRecord:
    """A chunk whose ContextHeader carries per-entity ``entity_confidences`` (ADR-0053).

    ``masked_content == content == text`` so every entity surface survives redaction — the confidence
    gate, not the redaction gate, is the only thing that can exclude a term at index time.
    """
    header = ContextHeader(
        entities=entities,
        entity_types=["ROLE"] * len(entities),
        entity_confidences=confidences,
    )
    return ChunkRecord(
        record_id="c1",
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=100,
            document_id="sha256:d1",
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="en", category="litigation", sensitivity="public"),
        document_record_id="d1",
        index=0,
        content=text,
        masked_content=text,
        token_count=max(1, len(text.split())),
        context_header=header,
    )


def test_index_time_confidence_threshold_excludes_weak_entity(tmp_path: Path) -> None:
    """R5: a header entity below the composing embedder's ``confidence_threshold`` is NOT emitted.

    Regression for the audit defect: ``_sparse_signals_for_row`` discarded the real per-entity
    confidence and stamped every header surface at a hardcoded ``1.0``, so the composing embedder's
    documented ``confidence_threshold`` never excluded a weak-NER entity from the emitted corpus
    sparse vector — the gate was a silent no-op on the index-time path (ADR-0053 / #162 AC-04). This
    drives REAL varied confidences through the Export → SparseEmbedder wiring and asserts the
    OBSERVABLE emitted vector: the strong entity is injected, the weak one is gated out. Both surfaces
    survive redaction (they occur in the corpus text), so ONLY the confidence gate can drop the weak
    one. It FAILS on the pre-fix hardcoded-``1.0`` path (the weak term is present) and passes on the
    fix.
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    text = "the plaintiff met wayne at the hall"
    chunk = _chunk_with_entity_confidences(
        text=text,
        entities=["plaintiff", "wayne"],
        confidences=[0.95, 0.2],  # plaintiff strong, wayne a weak NER hit
    )
    sparse_cfg = {"entity_boost": 5.0, "confidence_threshold": 0.9}
    export = JsonlParquetExport(
        {
            "basename": "corpus",
            "sparse_embedder": {"provider": "sparse.fakecomposing", "config": sparse_cfg},
        },
        registry=_registry_with("sparse.fakecomposing", _FakeComposingSparseEmbedder),
    )
    uris = export.export([chunk], storage, store)
    row = _read_jsonl(storage, uris)[0]
    emitted = dict(zip(row["sparse_indices"], row["sparse_values"], strict=True))

    # Guard: both surfaces occur in the corpus text, so neither is dropped by the redaction gate —
    # the confidence gate is the sole discriminator.
    assert "plaintiff" in text and "wayne" in text

    # The STRONG entity (0.95 >= 0.9) is injected: model weight (1) + entity_boost (5).
    plaintiff_id = _fake_tid("plaintiff")
    assert emitted.get(plaintiff_id) == 6.0, (
        "a strong-confidence entity was not injected at index time (ADR-0053 composition inert)"
    )

    # The WEAK entity (0.2 < 0.9) is gated out entirely — it appears only at its model-leg frequency
    # (1.0, from occurring in the text), NEVER boosted. On the pre-fix code it was stamped at 1.0 and
    # boosted to 6.0, so this assertion catches the silent no-op.
    wayne_id = _fake_tid("wayne")
    assert emitted.get(wayne_id) == 1.0, (
        "a weak-NER entity below confidence_threshold was injected — the gate is a no-op "
        "(ADR-0053 / #162 AC-04: weak NER must never pollute the sparse signal)"
    )


def test_index_time_sparse_composition_equals_text_only_when_no_signals(tmp_path: Path) -> None:
    """A composing embedder with NO structured signals on the row composes to exactly the model leg.

    A chunk with no context header and no selected metadata carries EMPTY_SPARSE_SIGNALS, so the
    composed path yields the identical vector the text-only path would — the additive guarantee holds
    even when a composing embedder is wired.
    """
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    # No header, no category ⇒ no entity signal, no selected metadata.
    export = JsonlParquetExport(
        {
            "basename": "corpus",
            "sparse_embedder": {
                "provider": "sparse.fakecomposing",
                "config": {"entity_boost": 5.0, "metadata_boost": 7.0, "metadata_fields": ["category"]},
            },
        },
        registry=_registry_with("sparse.fakecomposing", _FakeComposingSparseEmbedder),
    )
    uris = export.export([_chunk("alpha beta alpha")], storage, store)
    row = _read_jsonl(storage, uris)[0]
    expected = next(_FakeComposingSparseEmbedder({}).encode_sparse(["alpha beta alpha"]))
    assert row["sparse_indices"] == expected.indices
    assert row["sparse_values"] == expected.values

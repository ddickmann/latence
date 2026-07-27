"""P3-F2 — ``AdapterBase`` unit tests (ADR-0036 §1): the four cross-cutting helpers.

Deterministic, seeded, no I/O and no model download. Covers device resolution
(cpu/either/gpu-skip + the ``LATENCE_CUDA=1`` seam + a bad device string), batching
(input-order output for batch + non-batch providers, ``batch_size`` validation, no
whole-corpus buffering), the native-exception → typed-taxonomy guard (RuntimeError →
ProviderError, opted-in bad-config → ConfigError, framework LatenceError passthrough,
**no PII/record content** in the raised message), and the Provenance/offset carry helpers
(reuse the OffsetIndex/PageIndexResolver seam, ADR-0031).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from latence_core import AdapterBase, PageIndexResolver, ProviderProfile
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSlice,
    PageSpan,
    Provenance,
)
from latence_core.errors import (
    ConfigError,
    ContractError,
    LatenceError,
    PageSliceMissingError,
    ProviderError,
    error_category,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# -- fixtures: tiny adapters declaring different profiles ----------------------


def _profile(**overrides: object) -> ProviderProfile:
    base: dict[str, object] = {
        "compute": "cpu",
        "memory_mb": 16,
        "model_id": None,
        "license": "Apache-2.0",
        "license_verified": True,
        "license_code": "Apache-2.0",
        "license_source": "LICENSE",
        "license_verified_on": "2026-07-06",
        "deterministic": True,
        "batch": False,
        "cost_per_1k": None,
    }
    base.update(overrides)
    # H-B1: a model-bearing profile records the weights SPDX separately + cites a model card.
    if base.get("model_id") is not None and "license_weights" not in overrides:
        base["license_weights"] = "Apache-2.0"
        base["license_source"] = "https://huggingface.co/" + str(base["model_id"])
    return ProviderProfile(**base)  # type: ignore[arg-type]


class _CpuAdapter(AdapterBase):
    @classmethod
    def profile(cls) -> ProviderProfile:
        return _profile(compute="cpu")


class _EitherBatchAdapter(AdapterBase):
    @classmethod
    def profile(cls) -> ProviderProfile:
        return _profile(compute="either", batch=True)


class _GpuAdapter(AdapterBase):
    @classmethod
    def profile(cls) -> ProviderProfile:
        return _profile(compute="gpu", model_id="vendor/gpu", batch=True)


class _NoProfileAdapter(AdapterBase):
    """Declares no profile — the device seam treats it as ``either``; batching off."""


# -- 1. device resolution ------------------------------------------------------


def test_cpu_adapter_resolves_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a = _CpuAdapter({"device": "auto"})
    assert a.device.device == "cpu"
    assert a.device.skipped is False


def test_gpu_adapter_skipped_on_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a = _GpuAdapter({"device": "auto"})
    # The skip DeviceDecision — never a fabricated device.
    assert a.device.skipped is True
    assert a.device.device is None
    assert "GPU" in a.device.reason or "CUDA" in a.device.reason


def test_gpu_adapter_resolves_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # The LATENCE_CUDA=1 seam is exercised without a real GPU.
    monkeypatch.setenv("LATENCE_CUDA", "1")
    a = _GpuAdapter({"device": "auto"})
    assert a.device.skipped is False
    assert a.device.device == "cuda"


def test_no_profile_adapter_follows_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a = _NoProfileAdapter({"device": "cpu"})
    assert a.device.device == "cpu"
    assert a.device.skipped is False


def test_bad_device_string_is_config_error() -> None:
    with pytest.raises(ConfigError, match="invalid device config"):
        _CpuAdapter({"device": "tpu"})


# -- 2. batching ---------------------------------------------------------------


def _double(batch: list[int]) -> list[int]:
    return [x * 2 for x in batch]


def test_batching_preserves_input_order_batch_capable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a = _EitherBatchAdapter({"batch_size": 3})
    out = list(a.batched(range(10), _double))
    assert out == [x * 2 for x in range(10)]  # deterministic input order


def test_batching_non_batch_provider_degrades_to_one_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a = _CpuAdapter({})  # batch=False profile
    seen_sizes: list[int] = []

    def _run(batch: list[int]) -> list[int]:
        seen_sizes.append(len(batch))
        return batch

    out = list(a.batched(range(5), _run))
    assert out == list(range(5))
    assert seen_sizes == [1, 1, 1, 1, 1]  # one-item calls, non-batch


def test_batch_size_validation_rejects_zero() -> None:
    a = _EitherBatchAdapter({"batch_size": 0})
    with pytest.raises(ConfigError, match="batch_size must be >= 1"):
        list(a.batched(range(3), _double))


def test_batch_size_validation_rejects_non_int() -> None:
    a = _EitherBatchAdapter({"batch_size": "big"})
    with pytest.raises(ConfigError, match="batch_size must be an int"):
        list(a.batched(range(3), _double))


def test_batching_does_not_buffer_whole_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    # Streaming-friendly: the input iterable is consumed at most one batch ahead. We prove
    # it by counting how far a generator input is advanced before the first output arrives.
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a = _EitherBatchAdapter({"batch_size": 2})
    consumed = 0

    def _gen() -> Iterable[int]:
        nonlocal consumed
        for i in range(1000):
            consumed += 1
            yield i

    gen_out = a.batched(_gen(), _double)
    first = next(gen_out)
    assert first == 0
    # Only the first batch (2 items) was pulled to produce the first output — not all 1000.
    assert consumed == 2


# -- 3. native-exception → typed taxonomy (ADR-0034) ---------------------------


class _FailingAdapter(AdapterBase):
    config_exceptions = (KeyError,)

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _profile()

    def boom_runtime(self, secret_input: str) -> None:
        with self.guard("extract"):
            # A native lib that echoes the offending input into its error message.
            msg = f"native crash on record content: {secret_input}"
            raise RuntimeError(msg)

    def boom_config(self) -> None:
        with self.guard("load"):
            raise KeyError("missing_model_key")

    def boom_framework(self) -> None:
        with self.guard("extract"):
            raise ContractError("a framework contract violation")


def test_native_runtime_error_maps_to_provider_error() -> None:
    a = _FailingAdapter({})
    with pytest.raises(ProviderError) as excinfo:
        a.boom_runtime("PII: john.doe@example.com SSN 123-45-6789")
    err = excinfo.value
    assert error_category(err) == "provider"
    assert "extract" in str(err)


def test_guard_does_not_leak_record_content_or_pii() -> None:
    a = _FailingAdapter({})
    pii = "PII: john.doe@example.com SSN 123-45-6789"
    with pytest.raises(ProviderError) as excinfo:
        a.boom_runtime(pii)
    message = str(excinfo.value)
    # The raised message carries the provider + op + native TYPE name only — never the
    # native exception's message (which echoed the PII-bearing input).
    assert "john.doe@example.com" not in message
    assert "123-45-6789" not in message
    assert "RuntimeError" in message


def test_opted_in_native_exception_maps_to_config_error() -> None:
    a = _FailingAdapter({})
    with pytest.raises(ConfigError) as excinfo:
        a.boom_config()
    assert error_category(excinfo.value) == "config"
    # The native KeyError's message ("missing_model_key") must not leak either.
    assert "missing_model_key" not in str(excinfo.value)
    assert "KeyError" in str(excinfo.value)


def test_framework_latence_error_passes_through_unwrapped() -> None:
    a = _FailingAdapter({})
    # A framework LatenceError raised inside the guard is NOT re-wrapped as a ProviderError.
    with pytest.raises(ContractError) as excinfo:
        a.boom_framework()
    assert isinstance(excinfo.value, LatenceError)
    assert error_category(excinfo.value) == "contract"


# -- 4. Provenance / offset carry helpers (ADR-0031) ---------------------------


def _chunk(*, content: str, char_start: int, offset_map: OffsetMap | None) -> ChunkRecord:
    prov = Provenance(
        source_uri="file://x/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        file_size=100,
        document_id="doc-1",
        char_start=char_start,
        char_end=char_start + len(content),
        page_start=1,
        page_end=1,
    )
    cls = Classification(language="en", sensitivity="internal")
    page_map = PageMap(pages=[PageSpan(page_number=1, char_start=0, char_end=200)], total_chars=200)
    return ChunkRecord(
        record_id="chunk-1",
        provenance=prov,
        classification=cls,
        document_record_id="doc-1",
        index=0,
        content=content,
        token_count=len(content.split()),
        page_slice=PageSlice.for_span(page_map, char_start, char_start + len(content)),
        offset_map=offset_map,
    )


def test_carry_provenance_narrows_offsets_and_keeps_classification() -> None:
    chunk = _chunk(content="Acme Corp ships", char_start=10, offset_map=None)
    prov, cls = AdapterBase.carry_provenance(
        chunk, char_start=15, char_end=20, page_start=2, page_end=2
    )
    # Source lineage preserved; only the offset/page fields narrowed.
    assert prov.document_id == "doc-1"
    assert prov.source_uri == "file://x/doc.txt"
    assert prov.char_start == 15
    assert prov.char_end == 20
    assert prov.page_start == 2
    # Classification inherited unchanged.
    assert cls.language == "en"
    assert cls.sensitivity == "internal"


def test_carry_provenance_no_overrides_is_pure_carry() -> None:
    chunk = _chunk(content="text", char_start=5, offset_map=None)
    prov, cls = AdapterBase.carry_provenance(chunk)
    assert prov is chunk.provenance
    assert cls is chunk.classification


def test_rebase_offsets_identity_without_offset_map() -> None:
    # No offset map → the exact char_start + local shift (the identity path).
    start, end = AdapterBase.rebase_offsets(
        3, 7, chunk_char_start=100, offset_map=None
    )
    assert (start, end) == (103, 107)


def test_rebase_offsets_uses_offset_index_seam() -> None:
    # An offset map with a stripped gap: local offsets after the gap map to a higher
    # original offset. Segment (0, 50): local 0 → doc 50; segment (4, 60): local 4 → doc 60
    # (six original chars were stripped between local 3 and 4). Reuses OffsetIndex (ADR-0031).
    omap = OffsetMap(
        segments=[OffsetSegment(local_start=0, doc_start=50), OffsetSegment(local_start=4, doc_start=60)],
        length=10,
    )
    # local [5, 8) resolves through the second segment: doc_start 60 + (5-4)=61 .. last char
    # (7) → 60 + (7-4)=63, +1 exclusive end = 64.
    start, end = AdapterBase.rebase_offsets(5, 8, chunk_char_start=0, offset_map=omap)
    assert start == 61
    assert end == 64


def _two_page_map() -> PageMap:
    return PageMap(
        pages=[
            PageSpan(page_number=1, char_start=0, char_end=50),
            PageSpan(page_number=2, char_start=50, char_end=120),
        ],
        total_chars=120,
    )


def _paged_chunk(
    *,
    record_id: str,
    index: int,
    char_start: int,
    char_end: int,
    page_start: int,
    page_end: int,
    carries_pages: bool,
) -> ChunkRecord:
    """A chunk of ``doc-rec-1`` in the v19 shape: its OWN spans, cut from the two-page map.

    ``carries_pages=False`` models the only remaining unresolvable case — a Chunker that
    stamped no page spans at all — which the seam must surface, not paper over.
    """
    return ChunkRecord(
        record_id=record_id,
        provenance=Provenance(
            source_uri="file://x/doc.pdf",
            file_name="doc.pdf",
            file_type="pdf",
            file_size=120,
            document_id="doc-1",
            char_start=char_start,
            char_end=char_end,
            page_start=page_start,
            page_end=page_end,
        ),
        classification=Classification(language="en", sensitivity="internal"),
        document_record_id="doc-rec-1",
        index=index,
        content="x" * (char_end - char_start),
        token_count=1,
        page_slice=(
            PageSlice.for_span(_two_page_map(), char_start, char_end) if carries_pages else None
        ),
    )


def test_resolve_pages_uses_page_index_seam() -> None:
    chunk = _paged_chunk(
        record_id="chunk-0",
        index=0,
        char_start=0,
        char_end=120,
        page_start=1,
        page_end=2,
        carries_pages=True,
    )
    # A span at [60, 70) falls on page 2 via the PageIndexResolver seam (ADR-0031).
    ps, pe = AdapterBase.resolve_pages(chunk, 60, 70, resolver=PageIndexResolver())
    assert ps == 2
    assert pe == 2


def test_resolve_pages_resolves_a_later_chunk_that_arrives_alone() -> None:
    """A chunk cites its OWN page with no sibling ever handed to the resolver (v19).

    The intent this preserves is the one v18 could only satisfy by discipline: chunk 1
    straddles the page break, so inheriting its ``(1, 2)`` range is a plausible-looking
    WRONG page (audit R2). Under v18 the map rode chunk 0 and the resolver had to have
    ``observe``d it first; here chunk 0 is never constructed at all and the answer is still
    exact, because the property is the RECORD's, not the stream's.
    """
    later = _paged_chunk(
        record_id="chunk-1",
        index=1,
        char_start=40,
        char_end=80,
        page_start=1,  # the chunk straddles the page boundary: inherited range is (1, 2)
        page_end=2,
        carries_pages=True,
    )
    resolver = PageIndexResolver()
    ps, pe = AdapterBase.resolve_pages(later, 60, 70, resolver=resolver)
    assert (ps, pe) == (2, 2)


def test_resolve_pages_raises_when_the_chunk_carries_no_page_spans() -> None:
    # A Chunker that stamped no spans at all (its document never parsed a page map) is a
    # producer defect. It surfaces as a typed error rather than degrading to the chunk's
    # inherited range, which would emit a page that looks resolved and is not (ADR-0034).
    chunk = _paged_chunk(
        record_id="chunk-1",
        index=1,
        char_start=40,
        char_end=80,
        page_start=3,
        page_end=4,
        carries_pages=False,
    )
    with pytest.raises(PageSliceMissingError) as excinfo:
        AdapterBase.resolve_pages(chunk, 60, 70, resolver=PageIndexResolver())
    assert excinfo.value.record_id == "chunk-1"
    # The identifier, never the content (the no-PII discipline of the error taxonomy).
    assert "xxx" not in str(excinfo.value)


# -- optionality: an AdapterBase subclass still satisfies a Capability structurally --


def test_adapter_is_not_a_capability_layer() -> None:
    # AdapterBase names no Capability method — it is convenience, not a protocol. A subclass
    # implements its Capability method itself (proven end-to-end in test_e2e_adapter_base).
    assert not hasattr(AdapterBase, "extract")
    assert not hasattr(AdapterBase, "parse")
    # A subclass declaring no profile has device access but is non-batch by default.
    a = _NoProfileAdapter({})
    assert a.declared_profile() is None

"""``fused_entity_relation.gliner25`` — the contract, the offsets, and the traps.

Every test drives the Provider against the faithful ``gliner2`` 2.0.0 fake in ``conftest``, which
reproduces the real nested envelopes AND the real degradations (bare strings without
``include_spans``, an empty-but-well-formed relation result when the head is inactive, a
``TypeError`` on an unknown load kwarg). The claim these tests support is "the Provider parses,
grounds and batches the real API correctly"; they say **nothing** about extraction quality, which
needs the real checkpoint on real hardware and has not been measured.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from _gliner25_fake import chunk, install_gliner2
from latence_core.contracts import EntityMention, InducedLabels, PageMap, PageSpan, RelationMention
from latence_core.errors import ConfigError, ProviderError
from latence_core.providers.perf import DEFAULT_GLINER_MAX_LEN
from latence_gliner25 import Gliner25FusedExtractor
from latence_gliner25._backbone import (
    CHUNKER_TOKEN_EXPANSION,
    DEFAULT_MAX_LEN,
    MAX_SUPPORTED_MAX_LEN,
    SAFE_CHUNK_BUDGET,
    SAFE_MAX_LEN,
)

TEXT = "Alice works for Acme Corp in Berlin."
#: "Alice" = [0, 5); "Acme Corp" = [16, 25).
_ENTITY_ROWS: list[dict[str, Any]] = [
    {"start": 0, "end": 5, "label": "person", "confidence": 0.91},
    {"start": 16, "end": 25, "label": "organization", "confidence": 0.88},
]
_RELATION_ROWS: list[dict[str, Any]] = [
    {
        "relation": "works for",
        "head": {"text": "Alice", "confidence": 0.9, "start": 0, "end": 5},
        "tail": {"text": "Acme Corp", "confidence": 0.8, "start": 16, "end": 25},
    }
]

_CONFIG: dict[str, Any] = {
    "labels": ["person", "organization"],
    "relation_labels": ["works for"],
    "device": "cpu",
}


def _extract(provider: Gliner25FusedExtractor, chunks: list[Any]) -> tuple[
    list[EntityMention], list[RelationMention]
]:
    fused = provider.extract_fused(chunks)
    return fused.mentions, fused.relations


# -- the contract -----------------------------------------------------------------------------


def test_the_real_nested_shapes_produce_grounded_mentions_and_relations() -> None:
    """The happy path: nested per-label / per-relation dicts → mentions + a resolved relation."""
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS, relations=_RELATION_ROWS):
        mentions, relations = _extract(provider, [chunk(TEXT)])

    assert [(m.label, m.text) for m in mentions] == [
        ("person", "Alice"),
        ("organization", "Acme Corp"),
    ]
    assert [(m.provenance.char_start, m.provenance.char_end) for m in mentions] == [(0, 5), (16, 25)]
    assert len(relations) == 1
    rel = relations[0]
    assert (rel.label, rel.head_text, rel.tail_text) == ("works for", "Alice", "Acme Corp")
    assert rel.head_mention_id == mentions[0].record_id
    assert rel.tail_mention_id == mentions[1].record_id
    # The covering span is the min/max of the resolved endpoints, so it inherits their correction.
    assert (rel.provenance.char_start, rel.provenance.char_end) == (0, 25)
    # The weakest-link rule: min(head 0.9, tail 0.8).
    assert rel.confidence == pytest.approx(0.8)


def test_a_chunk_offset_is_rebased_onto_the_original_document() -> None:
    """Chunk-local model offsets must become TRUE original-document offsets (ADR-0031)."""
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS, relations=[]):
        mentions, _ = _extract(provider, [chunk(TEXT, char_start=1000)])
    assert [(m.provenance.char_start, m.provenance.char_end) for m in mentions] == [
        (1000, 1005),
        (1016, 1025),
    ]


def test_a_mention_resolves_its_own_page_not_the_chunks_inherited_range() -> None:
    """A sub-document carrier cites the page ITS OWN offset resolves to (C1 / ADR-0031)."""
    page_map = PageMap(
        pages=[
            PageSpan(page_number=1, char_start=0, char_end=20),
            PageSpan(page_number=2, char_start=20, char_end=len(TEXT)),
        ],
        total_chars=len(TEXT),
    )
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS, relations=_RELATION_ROWS):
        mentions, relations = _extract(provider, [chunk(TEXT, page_map=page_map)])
    # "Alice" [0,5) is on page 1; "Acme Corp" [16,25) STRADDLES the page-20 break.
    assert (mentions[0].provenance.page_start, mentions[0].provenance.page_end) == (1, 1)
    assert (mentions[1].provenance.page_start, mentions[1].provenance.page_end) == (1, 2)
    # The relation's covering span is answered from its endpoints' own pages, not by a lookup.
    assert (relations[0].provenance.page_start, relations[0].provenance.page_end) == (1, 2)


def test_a_relation_whose_endpoint_is_not_an_emitted_mention_is_dropped() -> None:
    """No un-groundable record leaks: an endpoint with no matching mention drops the relation."""
    orphan = [
        {
            "relation": "works for",
            "head": {"text": "Alice", "confidence": 0.9, "start": 0, "end": 5},
            "tail": {"text": "Nowhere", "confidence": 0.8, "start": 30, "end": 34},
        }
    ]
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS, relations=orphan):
        _, relations = _extract(provider, [chunk(TEXT)])
    assert relations == []


def test_a_degenerate_zero_length_span_is_skipped_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty surface cannot ground a mention — it is dropped, but never silently (ADR-0034)."""
    rows = [*_ENTITY_ROWS, {"start": 7, "end": 7, "label": "person", "confidence": 0.9}]
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with caplog.at_level(logging.INFO), install_gliner2(entities=rows, relations=[]):
        mentions, _ = _extract(provider, [chunk(TEXT)])
    assert len(mentions) == 2
    assert "degenerate" in caplog.text


# -- the anti-false-green anchors -------------------------------------------------------------


def test_without_include_spans_the_real_api_returns_strings_and_nothing_can_be_grounded() -> None:
    """The trap ``include_spans=True`` exists to avoid, proven against the faithful fake.

    ``_format_spans`` returns bare STRINGS when neither ``include_spans`` nor
    ``include_confidence`` is set. This test calls the fake the way a Provider that forgot the flag
    would, and asserts that the resulting rows carry no offsets at all — i.e. that the fake really
    can disagree with a Provider, and that the flag is load-bearing rather than decorative.
    """
    with install_gliner2(entities=_ENTITY_ROWS, relations=_RELATION_ROWS) as fake:
        model = fake()
        without = model.batch_extract_entities([TEXT], ["person"])[0]["entities"]
        with_spans = model.batch_extract_entities(
            [TEXT], ["person"], include_spans=True, include_confidence=True
        )[0]["entities"]
    assert without == {"person": ["Alice"]}
    assert with_spans["person"][0]["start"] == 0


def test_a_flat_list_mis_parse_would_recover_nothing() -> None:
    """The shape is NESTED. A Provider reading it as a flat list finds zero spans — proven here."""
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        raw = fake().batch_extract_entities(
            [TEXT], ["person", "organization"], include_spans=True, include_confidence=True
        )[0]
    assert isinstance(raw["entities"], dict)
    # What a flat-list reader would get: iterating a dict yields its KEYS, which carry no offsets.
    assert [row for row in raw["entities"] if isinstance(row, dict)] == []


def test_an_unexpected_output_shape_is_a_typed_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shape that is not the pinned one is loud, never a quiet empty result."""
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        monkeypatch.setattr(
            fake, "batch_extract_entities", lambda *a, **k: [{"entities": ["Alice"]}]
        )
        with pytest.raises(ProviderError) as excinfo:
            _extract(provider, [chunk(TEXT)])
    assert "unexpected batch_extract_entities output shape" in str(excinfo.value)


def test_an_inactive_relation_head_is_warned_about_not_inferred(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``enable_relations=False`` yields a well-formed EMPTY result — so it is said out loud."""
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with (
        caplog.at_level(logging.WARNING),
        install_gliner2(
            entities=_ENTITY_ROWS, relations=_RELATION_ROWS, enable_relations=False
        ),
    ):
        mentions, relations = _extract(provider, [chunk(TEXT)])
    assert len(mentions) == 2
    assert relations == []  # no exception, no warning from the library — hence ours
    assert "enable_relations=False" in caplog.text


# -- batching, windows and the padded-activation envelope --------------------------------------


def test_emission_is_invariant_to_batch_size_and_bucketing() -> None:
    """Bucketing and micro-batching must be byte-identical to a per-chunk reference."""
    chunks = [
        chunk(TEXT, record_id=f"doc1#chunk-{i}", index=i, char_start=i * 100) for i in range(5)
    ]
    # Give one chunk its OWN induced label set so it lands in a different bucket.
    chunks[2] = chunk(
        TEXT,
        record_id="doc1#chunk-2",
        index=2,
        char_start=200,
        induced_labels=InducedLabels(
            model_id="test-inducer", entity_types=["person"], relation_types=["works for"]
        ),
    )
    outputs = []
    for batch_size in (1, 2, 16):
        provider = Gliner25FusedExtractor({**_CONFIG, "batch_size": batch_size})
        with install_gliner2(entities=_ENTITY_ROWS, relations=_RELATION_ROWS):
            mentions, relations = _extract(provider, list(chunks))
        outputs.append(
            (
                [m.model_dump(mode="json") for m in mentions],
                [r.model_dump(mode="json") for r in relations],
            )
        )
    assert outputs[0] == outputs[1] == outputs[2]


def test_raising_max_len_does_not_widen_the_padded_forward() -> None:
    """The OOM-you-shipped guard: ``max_texts x max_len`` must not grow when the window grows.

    The padded activation of ONE forward is ``max_texts x max_len`` positions, because gliner2 pads
    every text in a micro-batch to the batch's longest sequence. The 768-window default implies an
    envelope; a 4x-wider window must buy its positions back out of the text cap, not add them.
    """
    reference = Gliner25FusedExtractor({**_CONFIG, "max_len": DEFAULT_GLINER_MAX_LEN})
    envelope = reference._microbatch_max_texts * DEFAULT_GLINER_MAX_LEN  # noqa: SLF001 - the subject
    for window in (DEFAULT_GLINER_MAX_LEN, DEFAULT_MAX_LEN, MAX_SUPPORTED_MAX_LEN):
        provider = Gliner25FusedExtractor({**_CONFIG, "max_len": window})
        max_texts = provider._microbatch_max_texts  # noqa: SLF001 - the bound is the subject
        # The bound is exactly "as many texts as the envelope holds, but never zero". At the 4096
        # ceiling one text alone already exceeds it, and a batch of one is the floor — the reactive
        # CUDA-OOM halving cannot go below it either, and `iter_token_budget_microbatches` always
        # admits the first text of a micro-batch for the same reason.
        assert max_texts == max(1, envelope // window)
        assert max_texts * window <= max(envelope, window)
    # And the headline claim, concretely: the wider default window does NOT keep the 768 text cap.
    wide = Gliner25FusedExtractor({**_CONFIG, "max_len": DEFAULT_MAX_LEN})
    assert wide._microbatch_max_texts < reference._microbatch_max_texts  # noqa: SLF001 - the subject


def test_an_explicit_microbatch_answer_outranks_the_derivation() -> None:
    """An operator with a measurement beats a derivation, in both directions."""
    provider = Gliner25FusedExtractor(
        {**_CONFIG, "max_len": MAX_SUPPORTED_MAX_LEN, "microbatch_max_texts": 8}
    )
    assert provider._microbatch_max_texts == 8  # noqa: SLF001 - the bound is the subject


def test_the_default_window_exploits_the_longer_checkpoint_and_is_threaded_natively() -> None:
    """The default is NOT the inherited 768, and it reaches the native call."""
    assert DEFAULT_MAX_LEN > DEFAULT_GLINER_MAX_LEN
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        _extract(provider, [chunk(TEXT)])
    assert fake.last_max_len == DEFAULT_MAX_LEN


def test_the_micro_batch_cap_bounds_every_native_call() -> None:
    """No single forward may exceed ``microbatch_max_texts`` texts, whatever the batch size."""
    chunks = [
        chunk(TEXT, record_id=f"doc1#chunk-{i}", index=i, char_start=i * 100) for i in range(9)
    ]
    provider = Gliner25FusedExtractor({**_CONFIG, "batch_size": 16, "microbatch_max_texts": 2})
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        _extract(provider, chunks)
    assert fake.call_sizes
    assert max(fake.call_sizes) <= 2


# -- the load seam ----------------------------------------------------------------------------


def test_the_device_config_is_honoured_and_the_perf_knobs_key_off_it() -> None:
    """#191: ``device: cpu`` must NOT resolve fp16 or move the model to cuda."""
    provider = Gliner25FusedExtractor({**_CONFIG, "device": "cpu"})
    with install_gliner2(entities=[]) as fake:
        provider._load_model()  # noqa: SLF001 - the load is the subject
    assert fake.last_kwargs == {}  # both knobs resolved False on cpu ⇒ a plain load
    assert fake.moves == []


def test_an_invalid_device_is_a_config_error_before_any_download() -> None:
    with pytest.raises(ConfigError):
        Gliner25FusedExtractor({**_CONFIG, "device": "tpu"})


def test_an_optimized_load_fault_degrades_to_a_plain_load() -> None:
    """A quantize/compile fault costs throughput, never a crash (the shared perf-seam guard)."""
    provider = Gliner25FusedExtractor({**_CONFIG, "quantize": True, "compile": True})
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        fake.raise_on_optimized_load = True
        mentions, _ = _extract(provider, [chunk(TEXT)])
    assert len(mentions) == 2
    assert fake.last_kwargs == {}  # the plain retry is what succeeded


def test_a_lazy_compile_fault_at_inference_reloads_uncompiled_once() -> None:
    """``torch.compile`` fails on the FIRST forward; the latch re-loads and the run continues."""
    provider = Gliner25FusedExtractor({**_CONFIG, "compile": True})
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        fake.compiled_forward_raises = True
        mentions, _ = _extract(provider, [chunk(TEXT)])
    assert len(mentions) == 2
    # Two loads, and the second asked for no optimisation at all: with `compile` forced off and
    # `quantize` already False on cpu, the shared perf seam calls the loader PLAIN (no kwargs).
    assert fake.load_kwargs_history == [{"quantize": False, "compile": True}, {}]


def test_a_cuda_oom_halves_the_micro_batch_and_still_emits_every_chunk() -> None:
    """Output is batch-invariant, so halving is byte-identical — no chunk is ever dropped."""
    chunks = [
        chunk(TEXT, record_id=f"doc1#chunk-{i}", index=i, char_start=i * 100) for i in range(4)
    ]
    provider = Gliner25FusedExtractor({**_CONFIG, "batch_size": 4, "microbatch_max_texts": 4})
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        fake.oom_above = 2
        mentions, _ = _extract(provider, chunks)
    assert len(mentions) == 8  # 2 per chunk, all four chunks


def test_a_native_fault_is_a_typed_error_carrying_no_chunk_text() -> None:
    """C2/C6: the message names the op and the native type only — never the input."""
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        fake.raise_on_extract = True
        with pytest.raises(ProviderError) as excinfo:
            _extract(provider, [chunk(TEXT)])
    message = str(excinfo.value)
    assert "batch_extract_entities" in message
    assert "Alice" not in message


def test_the_provider_never_passes_a_load_kwarg_the_real_library_rejects() -> None:
    """``split_load_kwargs`` raises TypeError for anything outside its option set — so must we.

    The fake enforces the real allow-list, so this is a genuine check that the shared perf seam's
    ``quantize``/``compile`` are inside it and that nothing else is smuggled in.
    """
    provider = Gliner25FusedExtractor({**_CONFIG, "quantize": True, "compile": False})
    with install_gliner2(entities=[]) as fake:
        provider._load_model()  # noqa: SLF001 - the load is the subject
    assert set(fake.last_kwargs) <= {"quantize", "compile"}


# -- config validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"relation_labels": ["works for"]},
        {"labels": ["person"]},
        {"labels": [], "relation_labels": ["works for"]},
    ],
)
def test_missing_label_sets_are_rejected_at_construction(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="labels"):
        Gliner25FusedExtractor(bad)


def test_an_invalid_label_floor_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        Gliner25FusedExtractor({**_CONFIG, "label_floor": "sometimes"})


def test_the_profile_declares_verified_weights_and_code_separately() -> None:
    """C3 / ADR-0012: both licences recorded, with a source and a date."""
    profile = Gliner25FusedExtractor.profile()
    assert profile.license_verified is True
    assert profile.license_code == "Apache-2.0"
    assert profile.license_weights == "Apache-2.0"
    assert profile.license_source is not None
    assert "huggingface.co/fastino/gliner2.5-multi-v1" in profile.license_source
    assert profile.license_verified_on == "2026-08-25"
    assert profile.deterministic is False


# -- the measured silent-truncation window (pod, RTX 2000 Ada, 2026-08-25) ----------------------


def test_a_chunk_the_model_cannot_read_to_the_end_is_REFUSED_not_truncated() -> None:
    """The measured fail-open, closed. gliner2 truncates a long input silently; this must not.

    A chunk whose own ``token_count`` reaches past the model window at the pod-measured 1.2x
    chunker→mdeberta expansion cannot be read to the end. The Provider refuses rather than
    forwarding it, because a partial result here is indistinguishable from a complete one and the
    tail's entities would be missed by nothing that reports it (ADR-0021: segmentation is the Chunk
    Stage's job, so the answer is a loud error naming the budget knob, not a quiet re-chunk).
    """
    provider = Gliner25FusedExtractor({**_CONFIG, "max_len": 1280})
    oversized = chunk(TEXT)
    oversized = oversized.model_copy(update={"token_count": 4000})
    with install_gliner2(entities=_ENTITY_ROWS), pytest.raises(ProviderError) as excinfo:
        _extract(provider, [oversized])
    message = str(excinfo.value)
    assert "max_tokens" in message  # names the knob to lower
    assert "1280" in message


def test_a_chunk_that_fits_is_not_refused() -> None:
    """The negative control: the guard must not fire on the budget the wizard actually emits.

    ``model_window_for(1024) == 1280`` by construction, so a 1024-chunker-token chunk is exactly at
    the boundary and must pass. A guard that refuses the default pipeline is a guard someone
    deletes.
    """
    provider = Gliner25FusedExtractor({**_CONFIG, "max_len": 1280})
    fits = chunk(TEXT).model_copy(update={"token_count": 1024})
    with install_gliner2(entities=_ENTITY_ROWS):
        mentions, _ = _extract(provider, [fits])
    assert len(mentions) == 2


def test_the_fake_really_can_truncate_silently_so_the_guard_is_what_saves_us() -> None:
    """The anti-false-green half: prove the fake models the failure the guard prevents.

    With the guard bypassed (a chunk that reports no token count, so the Provider has no evidence
    to refuse on) and the fake set to stop reading at 10 chars, the call SUCCEEDS and returns a
    well-formed result holding only the entity inside the window. That is the shape of a real
    over-long call: no exception, no warning, a plausible answer, and 50% of the entities gone.
    """
    provider = Gliner25FusedExtractor({**_CONFIG, "max_len": 1280})
    unknown_length = chunk(TEXT).model_copy(update={"token_count": 0})
    with install_gliner2(entities=_ENTITY_ROWS) as fake:
        fake.truncates_beyond_chars = 10  # "Alice" [0,5) survives, "Acme Corp" [16,25) does not
        mentions, _ = _extract(provider, [unknown_length])
    assert [m.text for m in mentions] == ["Alice"]


def test_the_default_window_is_the_MEASURED_safe_one_not_the_advertised_one() -> None:
    """1280, not 4096 — and the two constants must not be confused for each other."""
    assert DEFAULT_MAX_LEN == SAFE_MAX_LEN
    assert MAX_SUPPORTED_MAX_LEN == 4096
    assert SAFE_MAX_LEN < MAX_SUPPORTED_MAX_LEN
    provider = Gliner25FusedExtractor(dict(_CONFIG))
    assert provider._max_len == SAFE_MAX_LEN  # noqa: SLF001 - the default is the subject


def test_a_window_above_the_measured_one_is_allowed_on_extraction_but_WARNED(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Losing entities is a recall trade an operator can see and choose — so warn, do not refuse.

    The warning has to carry the measurement, or it teaches nothing.
    """
    with caplog.at_level(logging.WARNING):
        provider = Gliner25FusedExtractor({**_CONFIG, "max_len": 3072})
    assert provider._max_len == 3072  # noqa: SLF001 - the accepted value is the subject
    assert "MEASURED-safe window" in caplog.text
    assert "NON-MONOTONICALLY" in caplog.text


def test_the_window_is_DERIVED_from_the_decided_1024_budget_not_typed() -> None:
    """1024 is an operator decision; the window is the framework's own derivation of it.

    Asserted three ways, because each catches a different way this could rot: the budget is the
    one ``chunk.page`` defaults to, the window is what ``model_window_for`` makes of it, and the
    guard's expansion is the same constant the derivation used — so a wizard-emitted stack can
    never refuse its own chunks.
    """
    from latence_core.setup.recipe import MDEBERTA_TOKEN_EXPANSION, model_window_for
    from latence_core.stages.chunk import PAGE_CHUNK_MAX_TOKENS

    assert SAFE_CHUNK_BUDGET == 1024 == PAGE_CHUNK_MAX_TOKENS
    assert model_window_for(SAFE_CHUNK_BUDGET) == SAFE_MAX_LEN
    assert CHUNKER_TOKEN_EXPANSION == MDEBERTA_TOKEN_EXPANSION

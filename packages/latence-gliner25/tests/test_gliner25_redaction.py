"""``redaction.gliner25`` — zero-shot PII on the general GLiNER 2.5 checkpoint.

What these tests establish: the Provider parses the real per-label-dict shape, grounds its spans,
masks through the shared redaction seam, and refuses the two configurations that would make it
claim more than it can do. What they deliberately do NOT establish is PII **recall** — there is no
GLiNER 2.5 PII checkpoint, this Provider asks a general model zero-shot, and no measurement of
that exists yet. A fake cannot measure recall, and pretending otherwise is the exact false green
this repository refuses.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from _gliner25_fake import chunk, install_gliner2
from latence_core.contracts import InducedLabels, PageMap, PageSpan
from latence_core.errors import ConfigError, ProviderError
from latence_gliner25 import Gliner25PIIRedactor
from latence_gliner25._backbone import (
    DEFAULT_MAX_LEN,
    DEFAULT_MODEL,
    MAX_SUPPORTED_MAX_LEN,
    SAFE_MAX_LEN,
)

TEXT = "Alice works at Acme. Contact alice@example.com about it."
#: "Alice" = [0, 5); "alice@example.com" = [29, 46).
_PII_ROWS: list[dict[str, Any]] = [
    {"start": 0, "end": 5, "label": "person", "confidence": 0.95},
    {"start": 29, "end": 46, "label": "email", "confidence": 0.99},
]

_CONFIG: dict[str, Any] = {"pii_labels": ["person", "email"], "device": "cpu"}


def test_pii_is_detected_grounded_and_masked_through_the_shared_seam() -> None:
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    with install_gliner2(entities=_PII_ROWS):
        out = list(provider.redact([chunk(TEXT)]))

    assert len(out) == 1
    redacted = out[0]
    assert redacted.content == TEXT  # the clean content is NEVER mutated
    assert redacted.masked_content is not None
    assert "alice@example.com" not in redacted.masked_content
    types = sorted(span.pii_type for span in redacted.pii_spans)
    assert types == ["email", "person"]
    assert all(span.detector.startswith("gliner25:") for span in redacted.pii_spans)


def test_the_raw_pii_value_is_never_persisted_on_a_span() -> None:
    """S6: the surface string is transient — it derives a surrogate and is not stored."""
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    with install_gliner2(entities=_PII_ROWS):
        redacted = next(iter(provider.redact([chunk(TEXT)])))
    dumped = redacted.model_dump_json(exclude={"content"})
    assert "alice@example.com" not in dumped


def test_a_span_resolves_its_own_source_page() -> None:
    page_map = PageMap(
        pages=[
            PageSpan(page_number=1, char_start=0, char_end=25),
            PageSpan(page_number=2, char_start=25, char_end=len(TEXT)),
        ],
        total_chars=len(TEXT),
    )
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    with install_gliner2(entities=_PII_ROWS):
        redacted = next(iter(provider.redact([chunk(TEXT, page_map=page_map)])))
    by_type = {span.pii_type: span for span in redacted.pii_spans}
    assert by_type["person"].page_start == 1
    assert by_type["email"].page_start == 2


def test_induced_pii_types_union_onto_the_config_floor_never_replace_it() -> None:
    """The floor is UNIVERSAL: induction may only EXTEND the label set (ADR-0038 / ADR-0044)."""
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    rows = [*_PII_ROWS, {"start": 15, "end": 19, "label": "iban", "confidence": 0.9}]
    induced = InducedLabels(model_id="test-inducer", pii_types=["iban"])
    with install_gliner2(entities=rows):
        redacted = next(iter(provider.redact([chunk(TEXT, induced_labels=induced)])))
    types = sorted(span.pii_type for span in redacted.pii_spans)
    assert types == ["email", "iban", "person"]


def test_a_flat_list_mis_parse_would_recover_nothing() -> None:
    """The anti-false-green anchor: the shape is a per-label DICT, not the gliner flat list."""
    with install_gliner2(entities=_PII_ROWS) as fake:
        raw = fake().batch_extract_entities(
            [TEXT], ["person", "email"], include_spans=True, include_confidence=True
        )[0]
    assert isinstance(raw["entities"], dict)
    assert [row for row in raw["entities"] if isinstance(row, dict)] == []


def test_an_unexpected_output_shape_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    with install_gliner2(entities=_PII_ROWS) as fake:
        monkeypatch.setattr(fake, "batch_extract_entities", lambda *a, **k: [{"entities": []}])
        with pytest.raises(ValueError, match="unexpected batch_extract_entities output shape"):
            list(provider.redact([chunk(TEXT)]))


# -- the two refusals that keep the claim honest -----------------------------------------------


def test_guardrails_is_refused_on_the_default_checkpoint() -> None:
    """There is no GLiNER 2.5 guardrails checkpoint, so the switch must not silently do nothing.

    A security control that reports nothing while looking enabled is worse than one that is off.
    The error has to name the two things that DO work, or it is just an obstacle.
    """
    with pytest.raises(ConfigError) as excinfo:
        Gliner25PIIRedactor({**_CONFIG, "guardrails": True})
    message = str(excinfo.value)
    assert DEFAULT_MODEL in message
    assert "screening.content_keyword" in message
    assert "redaction.gliner2" in message


def test_guardrails_is_accepted_with_a_warning_on_an_operator_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Someone who fine-tuned their own 2.5 guardrails head knows more than this package does."""
    with caplog.at_level(logging.WARNING):
        Gliner25PIIRedactor(
            {**_CONFIG, "guardrails": True, "model": "acme/gliner25-guardrails-internal"}
        )
    assert "guardrails is ON" in caplog.text


def test_a_non_bool_guardrails_value_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        Gliner25PIIRedactor({**_CONFIG, "guardrails": "yes"})


def test_the_safety_floor_cannot_be_subsetted_per_chunk() -> None:
    """``label_floor`` is refused rather than silently ignored — the operator must know."""
    with pytest.raises(ConfigError) as excinfo:
        Gliner25PIIRedactor({**_CONFIG, "label_floor": "per_chunk"})
    assert "universal" in str(excinfo.value)


def test_label_floor_universal_is_accepted_as_an_explicit_no_op() -> None:
    Gliner25PIIRedactor({**_CONFIG, "label_floor": "universal"})


def test_a_span_checkpoint_including_the_dedicated_pii_one_is_refused() -> None:
    """The 1.x PII checkpoints are ``span`` models — this Provider cannot and must not load them."""
    provider = Gliner25PIIRedactor(
        {**_CONFIG, "model": "fastino/GLiNER2-Guardrails-PII-Multi"}
    )
    with install_gliner2(), pytest.raises(ProviderError) as excinfo:
        provider._load_model()  # noqa: SLF001 - the load is the gate under test
    assert "redaction.gliner2" in str(excinfo.value)


# -- window, batching and the load seam ---------------------------------------------------------


def test_the_window_default_is_threaded_natively() -> None:
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    with install_gliner2(entities=_PII_ROWS) as fake:
        list(provider.redact([chunk(TEXT)]))
    assert fake.last_max_len == DEFAULT_MAX_LEN


def test_emission_is_invariant_to_batch_size() -> None:
    chunks = [
        chunk(TEXT, record_id=f"doc1#chunk-{i}", index=i, char_start=i * 100) for i in range(5)
    ]
    outputs = []
    for batch_size in (1, 2, 8):
        provider = Gliner25PIIRedactor({**_CONFIG, "batch_size": batch_size})
        with install_gliner2(entities=_PII_ROWS):
            outputs.append(
                [c.model_dump(mode="json") for c in provider.redact(list(chunks))]
            )
    assert outputs[0] == outputs[1] == outputs[2]


def test_a_cuda_oom_halves_and_still_masks_every_chunk() -> None:
    """No chunk may be dropped or emitted un-scanned — the ADR-0044 floor depends on it."""
    chunks = [
        chunk(TEXT, record_id=f"doc1#chunk-{i}", index=i, char_start=i * 100) for i in range(4)
    ]
    provider = Gliner25PIIRedactor(
        {**_CONFIG, "batch_size": 4, "microbatch_max_texts": 4}
    )
    with install_gliner2(entities=_PII_ROWS) as fake:
        fake.oom_above = 2
        out = list(provider.redact(chunks))
    assert len(out) == 4
    assert all(c.masked_content is not None for c in out)


def test_a_native_fault_is_typed_and_carries_no_pii() -> None:
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    with install_gliner2(entities=_PII_ROWS) as fake:
        fake.raise_on_extract = True
        with pytest.raises(ProviderError) as excinfo:
            list(provider.redact([chunk(TEXT)]))
    message = str(excinfo.value)
    assert "alice@example.com" not in message
    assert "Alice" not in message


def test_the_profile_states_plainly_that_no_2_5_pii_checkpoint_exists() -> None:
    """The overclaim guard: the descriptor must say what this Provider is, not what it resembles."""
    profile = Gliner25PIIRedactor.profile()
    assert profile.perf_note is not None
    note = profile.perf_note.lower()
    assert "zero-shot" in note
    assert "no 2.5 pii" in note
    assert "no parity" in note
    assert profile.license_verified is True
    assert profile.license_weights == "Apache-2.0"


# -- the fail-open the measurement found, closed (pod, RTX 2000 Ada, 2026-08-25) ----------------


def test_a_window_above_the_measured_one_is_REFUSED_for_redaction() -> None:
    """For a Redactor an over-wide window is not a recall trade — it is unmasked PII, silently.

    The card advertises 4096; a real run showed extraction degrading silently and non-monotonically
    past ~1850 tokens (90 SSNs in a 7.5k-token document → 32 spans covering the first 35%). A
    Redaction Stage in that regime emits "redacted" chunks that still contain PII with nothing in
    the run reporting it. The refusal must name the safe window AND the budget knob, or it is only
    an obstacle.
    """
    with pytest.raises(ConfigError) as excinfo:
        Gliner25PIIRedactor({**_CONFIG, "max_len": MAX_SUPPORTED_MAX_LEN})
    message = str(excinfo.value)
    assert str(SAFE_MAX_LEN) in message
    assert "max_tokens" in message
    assert "fail-open" in message


def test_the_wider_window_is_accepted_with_a_warning_on_an_operator_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The measurement describes ONE checkpoint; an operator with their own evidence outranks it."""
    with caplog.at_level(logging.WARNING):
        provider = Gliner25PIIRedactor(
            {**_CONFIG, "max_len": 2048, "model": "acme/gliner25-internal"}
        )
    assert provider._max_len == 2048  # noqa: SLF001 - the accepted value is the subject
    assert "on your evidence" in caplog.text


def test_an_over_long_chunk_is_REFUSED_rather_than_emitted_as_masked() -> None:
    """THE fail-open test. An un-readable chunk must never come back looking redacted."""
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    oversized = chunk(TEXT).model_copy(update={"token_count": 4000})
    with install_gliner2(entities=_PII_ROWS), pytest.raises(ProviderError) as excinfo:
        list(provider.redact([oversized]))
    assert "truncates a longer input SILENTLY" in str(excinfo.value)


def test_the_fake_proves_the_failure_the_guard_prevents() -> None:
    """Anti-false-green: with no evidence to refuse on, the silent partial mask really happens.

    ``token_count == 0`` gives the guard nothing to act on (it refuses on evidence, never on its
    absence), and the fake stops reading at 10 chars. The result is a chunk that reports
    ``masked_content`` set and a clean-looking span list — while the email at [29, 46) is still
    there in the clear. That is the shape of a real over-long redaction call, and it is why the
    window bound and the chunk guard both exist.
    """
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    unknown_length = chunk(TEXT).model_copy(update={"token_count": 0})
    with install_gliner2(entities=_PII_ROWS) as fake:
        fake.truncates_beyond_chars = 10
        redacted = next(iter(provider.redact([unknown_length])))
    assert redacted.masked_content is not None
    assert [s.pii_type for s in redacted.pii_spans] == ["person"]
    assert "alice@example.com" in redacted.masked_content  # the tail's PII, unmasked


def test_the_redactor_default_window_is_the_measured_safe_one() -> None:
    provider = Gliner25PIIRedactor(dict(_CONFIG))
    assert provider._max_len == SAFE_MAX_LEN  # noqa: SLF001 - the default is the subject

"""W18 (ADR-0044) — the UNIVERSAL FINANCIAL-PII redaction floor is non-bypassable.

Before W18, only ``redaction.hybrid_rule`` recognised the high-harm financial subset
(``ssn``/``credit_card``/``iban``); the three learned Providers inherited only their model's
coverage, so a model that missed one leaked it into the exported corpus. W18 routes a deterministic,
zero-dep floor through the SINGLE shared seam every Provider calls
(:func:`~latence_core.redaction_policy.finalize_redacted_chunk`), so the guarantee cannot be
forgotten per-Provider again and cannot be bypassed by model/backend choice.

These tests prove the floor at the choke point directly (a Provider that detected NOTHING still yields
``masked_content`` with the financial types masked), that it honours the operator's explicit
``skip`` / policy action, and that the floor patterns stay byte-identical to the hybrid_rule built-ins
(the drift guard). The per-Provider chunk paths (gliner / gliner2 / presidio) carry their own floor
test in each package's suite.
"""

from __future__ import annotations

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    Provenance,
    RedactionAction,
)
from latence_core.redaction_policy import (
    FINANCIAL_FLOOR_PATTERNS,
    RedactionPolicy,
    finalize_redacted_chunk,
    floor_candidates,
    plan_chunk_redaction,
)
from latence_core.stages.redaction import _BUILTIN_PATTERNS, HybridPIIRedactor

# One of each floor type, plus a non-floor entity that must survive untouched.
_CARD = "4111 1111 1111 1111"
_IBAN = "DE89370400440532013000"
_SSN = "123-45-6789"
_TEXT = f"Acme invoice: card {_CARD}, iban {_IBAN}, ssn {_SSN} for Bob."


def _chunk(text: str, *, sensitivity: str = "public") -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=len(text),
        document_id="sha256:d1",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id="d1#chunk:0",
        document_record_id="d1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=text,
        token_count=len(text.split()),
        offset_map=OffsetMap(segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text)),
    )


def _plan(chunk: ChunkRecord, policy: RedactionPolicy) -> object:
    # detector_types EMPTY + sensitive_set EMPTY models the pre-W18 gliner-family call: a backend
    # whose configured coverage is nothing. needs_detection stays True (not skipped, not near-empty).
    return plan_chunk_redaction(
        chunk, policy, detector_types=set(), sensitive_set=set(), min_chars=1
    )


# -- the choke point: a backend that detected NOTHING still floors financial PII ----------------


def test_financial_floor_masks_when_backend_detects_nothing() -> None:
    """``accepted == []`` (any Provider whose model found nothing) → the floor masks card/iban/ssn."""
    chunk = _chunk(_TEXT)
    policy = RedactionPolicy(policies=None, default_action=RedactionAction.MASK)
    plan = _plan(chunk, policy)

    rec = finalize_redacted_chunk(chunk, plan, accepted=[], salt="s")  # type: ignore[arg-type]

    assert rec.masked_content is not None
    # Every floor value is gone from the exported text — the leak the floor closes.
    assert _CARD not in rec.masked_content
    assert _IBAN not in rec.masked_content
    assert _SSN not in rec.masked_content
    # The masked-out spans are the floor's, and only the floor's (the backend produced none).
    assert {s.pii_type for s in rec.pii_spans} == {"credit_card", "iban", "ssn"}
    assert all(s.detector.startswith("floor:") for s in rec.pii_spans)
    # The non-PII entity survives untouched.
    assert "Bob" in rec.masked_content


def test_floor_content_is_byte_unchanged() -> None:
    """The floor masks only ``masked_content`` — the clean ``content`` stays intact (extraction)."""
    chunk = _chunk(_TEXT)
    policy = RedactionPolicy(policies=None, default_action=RedactionAction.MASK)
    rec = finalize_redacted_chunk(chunk, _plan(chunk, policy), accepted=[], salt="s")  # type: ignore[arg-type]
    assert rec.content == _TEXT


def test_floor_honours_policy_action() -> None:
    """A floor span is handled with the chunk's resolved action (here HASH), not a hardcoded mask."""
    chunk = _chunk(_TEXT)
    policy = RedactionPolicy(policies=None, default_action=RedactionAction.HASH)
    rec = finalize_redacted_chunk(chunk, _plan(chunk, policy), accepted=[], salt="s")  # type: ignore[arg-type]
    assert rec.pii_spans
    assert all(s.action is RedactionAction.HASH for s in rec.pii_spans)
    assert _IBAN not in (rec.masked_content or "")


def test_floor_respects_explicit_skip() -> None:
    """Operator intent is preserved: an explicit ``skip:true`` still skips (no floor forced on)."""
    chunk = _chunk(_TEXT, sensitivity="confidential")
    policy = RedactionPolicy({"confidential": {"skip": True}}, RedactionAction.MASK)
    plan = plan_chunk_redaction(
        chunk, policy, detector_types=set(), sensitive_set=set(), min_chars=1
    )
    rec = finalize_redacted_chunk(chunk, plan, accepted=[], salt="s")
    assert rec.pii_spans == []
    assert rec.masked_content == _TEXT  # nothing masked — the operator chose to skip


# -- non-bypassable through the in-core hybrid_rule Provider, even when narrowed -----------------


def test_hybrid_rule_floor_survives_a_types_narrowing() -> None:
    """A ``types:['email']`` policy narrows hybrid_rule's OWN detection away from financial — but the
    floor still masks the IBAN/card/SSN (the guarantee is independent of the policy allow-list)."""
    cfg = {"policies": {"public": {"action": "mask", "types": ["email"]}}}
    rec = next(iter(HybridPIIRedactor(cfg).redact([_chunk(_TEXT)])))
    assert rec.masked_content is not None
    assert _CARD not in rec.masked_content
    assert _IBAN not in rec.masked_content
    assert _SSN not in rec.masked_content


def test_hybrid_rule_full_coverage_is_byte_identical_to_the_floor() -> None:
    """hybrid_rule with all detectors ALREADY masks these types; the floor overlaps exactly and the
    Provider wins every tie, so the masked text is byte-identical to the pre-W18 hybrid output."""
    rec = next(iter(HybridPIIRedactor({}).redact([_chunk(_TEXT)])))
    # No floor-provenance spans leak through when the Provider covered them (tie → Provider wins).
    assert not any(s.detector.startswith("floor:") for s in rec.pii_spans)
    assert rec.masked_content is not None
    for value in (_CARD, _IBAN, _SSN):
        assert value not in rec.masked_content


# -- drift guard: the floor patterns ARE the hybrid_rule built-ins ------------------------------


def test_floor_patterns_match_hybrid_rule_builtins() -> None:
    """The floor never disagrees with the in-core rule Redactor: same regex per shared type."""
    for name, (_pii_type, regex) in FINANCIAL_FLOOR_PATTERNS.items():
        assert _BUILTIN_PATTERNS[name] == regex


def test_floor_candidates_empty_on_clean_text() -> None:
    """A chunk with no financial PII yields no floor spans (the byte-identical common case)."""
    assert floor_candidates("The quarterly report is attached for Bob to review.") == []

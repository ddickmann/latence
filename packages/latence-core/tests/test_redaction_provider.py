"""HybridPIIRedactor: deterministic hybrid-rule PII Redaction (S6, in-core reference).

Unit tests for the in-core reference Provider: happy-path masked variant + structured
PII spans (type/offset/action/placeholder) that resolve to the correct source page,
each RedactionAction (mask/replace/hash/tag), the sensitivity-keyed policy (the
metadata-aware core of S6), the config gazetteer + custom patterns, determinism, the
"raw PII never persisted" invariant, config validation, and the clean / skip / empty
degradations.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    Provenance,
    RedactionAction,
)
from latence_core.stages.redaction import HybridPIIRedactor


def _doc(
    text: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
    record_id: str = "d1",
) -> DocumentRecord:
    """A parsed DocumentRecord — Redaction scans this (original markdown), not chunks."""
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:d1",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return DocumentRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=text,
        page_map=page_map,
    )


_TEXT = "Contact alice@acme.com or call 555-123-4567 today."


# -- happy path ---------------------------------------------------------------


def test_detects_and_masks_builtin_pii() -> None:
    doc = _doc(_TEXT)
    rec = list(HybridPIIRedactor({"detectors": ["email", "phone"]}).redact([doc]))[0]
    # Both the email and the phone are masked in the text variant.
    assert "alice@acme.com" not in rec.masked_content
    assert "555-123-4567" not in rec.masked_content
    assert "[EMAIL]" in rec.masked_content
    assert "[PHONE]" in rec.masked_content
    types = {s.pii_type for s in rec.pii_spans}
    assert types == {"email", "phone"}
    # Structured span: type, document offset, action, placeholder, detector.
    email = next(s for s in rec.pii_spans if s.pii_type == "email")
    assert email.detector == "regex:email"
    assert email.action is RedactionAction.MASK
    assert email.placeholder == "[EMAIL]"
    assert _TEXT[email.char_start : email.char_end] == "alice@acme.com"


def test_span_offsets_are_document_coordinates() -> None:
    # PII deep in the document: the span offset is the exact original-document offset,
    # so the slice round-trips back to the raw value (the coordinate system spans use).
    prefix = "# Heading\n\nSome intro paragraph before the sensitive line.\n\n"
    doc = _doc(prefix + _TEXT)
    rec = list(HybridPIIRedactor({"detectors": ["email"]}).redact([doc]))[0]
    email = rec.pii_spans[0]
    assert email.char_start == len(prefix) + _TEXT.index("alice@acme.com")
    assert (prefix + _TEXT)[email.char_start : email.char_end] == "alice@acme.com"


def test_clean_document_yields_empty_spans_and_identical_text() -> None:
    doc = _doc("Nothing sensitive in this line at all.")
    rec = list(HybridPIIRedactor({"detectors": ["email", "phone", "ssn"]}).redact([doc]))[0]
    assert rec.pii_count == 0
    assert rec.masked_content == "Nothing sensitive in this line at all."


def test_empty_document_is_not_an_error() -> None:
    doc = _doc("")
    rec = list(HybridPIIRedactor({"detectors": ["email"]}).redact([doc]))[0]
    assert rec.pii_count == 0
    assert rec.masked_content == ""


# -- actions ------------------------------------------------------------------


def test_replace_action_pseudonymises_deterministically() -> None:
    doc = _doc(_TEXT)
    cfg = {"detectors": ["email"], "default_action": "replace"}
    rec = list(HybridPIIRedactor(cfg).redact([doc]))[0]
    span = rec.pii_spans[0]
    assert span.action is RedactionAction.REPLACE
    assert span.placeholder.startswith("EMAIL_")
    assert "alice@acme.com" not in rec.masked_content
    assert span.placeholder in rec.masked_content


def test_hash_action_is_stable_and_hides_value() -> None:
    doc = _doc(_TEXT)
    cfg = {"detectors": ["email"], "default_action": "hash", "salt": "s1"}
    rec = list(HybridPIIRedactor(cfg).redact([doc]))[0]
    span = rec.pii_spans[0]
    assert span.action is RedactionAction.HASH
    assert span.placeholder.startswith("<EMAIL:")
    assert "alice" not in span.placeholder


def test_tag_action_leaves_text_intact_but_records_span() -> None:
    doc = _doc(_TEXT)
    cfg = {"detectors": ["email"], "default_action": "tag"}
    rec = list(HybridPIIRedactor(cfg).redact([doc]))[0]
    # Detect-only: the surface text is unchanged, but the span is recorded.
    assert rec.masked_content == _TEXT
    assert rec.pii_count == 1
    assert rec.pii_spans[0].action is RedactionAction.TAG
    assert rec.pii_spans[0].placeholder == ""


# -- metadata-aware policy (keyed off Classification.sensitivity) --------------


def test_policy_keys_off_sensitivity() -> None:
    cfg = {
        "detectors": ["email"],
        "policies": {
            "public": {"action": "mask"},
            "internal": {"action": "tag"},
        },
    }
    provider = HybridPIIRedactor(cfg)
    public = list(provider.redact([_doc(_TEXT, sensitivity="public")]))[0]
    internal = list(provider.redact([_doc(_TEXT, sensitivity="internal")]))[0]
    # Same text, different disposition driven purely by Classification.sensitivity.
    assert public.policy == "public"
    assert "[EMAIL]" in public.masked_content
    assert internal.policy == "internal"
    assert internal.masked_content == _TEXT  # tagged, not masked


def test_policy_skip_disables_redaction_for_a_class() -> None:
    cfg = {"detectors": ["email"], "policies": {"confidential": {"skip": True}}}
    rec = list(HybridPIIRedactor(cfg).redact([_doc(_TEXT, sensitivity="confidential")]))[0]
    assert rec.pii_count == 0
    assert rec.masked_content == _TEXT


def test_policy_types_allowlist_restricts_detection() -> None:
    # Only 'email' is acted on for this sensitivity, even though phone is detectable.
    cfg = {
        "detectors": ["email", "phone"],
        "policies": {"public": {"action": "mask", "types": ["email"]}},
    }
    rec = list(HybridPIIRedactor(cfg).redact([_doc(_TEXT, sensitivity="public")]))[0]
    assert {s.pii_type for s in rec.pii_spans} == {"email"}
    assert "555-123-4567" in rec.masked_content  # phone left intact


def test_default_policy_fallback() -> None:
    # A document whose sensitivity has no explicit policy uses the 'default' entry.
    cfg = {"detectors": ["email"], "policies": {"default": {"action": "hash"}}}
    rec = list(HybridPIIRedactor(cfg).redact([_doc(_TEXT, sensitivity="weird")]))[0]
    assert rec.policy == "default"
    assert rec.pii_spans[0].action is RedactionAction.HASH


# -- config-supplied recognisers ---------------------------------------------


def test_gazetteer_and_custom_patterns() -> None:
    text = "Employee Alice, ref A-99, emailed the team."
    doc = _doc(text)
    cfg = {
        "detectors": [],  # no built-ins
        "gazetteer": {"name": ["Alice"]},
        "patterns": {"ref": [r"A-\d+"]},
    }
    rec = list(HybridPIIRedactor(cfg).redact([doc]))[0]
    types = {s.pii_type for s in rec.pii_spans}
    assert types == {"name", "ref"}
    name = next(s for s in rec.pii_spans if s.pii_type == "name")
    assert name.detector == "gazetteer:name"
    assert "[NAME]" in rec.masked_content
    assert "[REF]" in rec.masked_content


# -- page resolution (round-trip) --------------------------------------------


def test_span_page_resolves_through_page_map() -> None:
    # Two-page document; the PII sits on page 2. The span must cite page 2.
    assembly = PageMap.from_page_texts(["First page has nothing.", "Reach alice@acme.com here."])
    text = assembly.text
    doc = _doc(text, page_map=assembly.page_map)
    rec = list(HybridPIIRedactor({"detectors": ["email"]}).redact([doc]))[0]
    span = rec.pii_spans[0]
    assert span.page_start == 2
    assert span.page_end == 2


def test_markup_document_offsets_and_page_are_faithful() -> None:
    # REGRESSION: Redaction must scan the ORIGINAL parsed markdown, not a
    # reconstruction from markup-stripped chunks. A document whose text carries
    # markdown markup (headings, bold, a link) before the PII, split across two
    # pages, must still record the email's EXACT original offset and page. (The
    # earlier chunk-reconstruction placed stripped content at original offsets,
    # yielding a fabricated coordinate system: wrong char span and wrong page.)
    page1 = "# Confidential Report Heading\n\n**Overview** of the [system](http://ex.com)."
    page2 = "Reach the admin at bob@corp.com immediately."
    assembly = PageMap.from_page_texts([page1, page2])
    text = assembly.text
    doc = _doc(text, page_map=assembly.page_map)
    rec = list(HybridPIIRedactor({"detectors": ["email"]}).redact([doc]))[0]
    assert len(rec.pii_spans) == 1
    span = rec.pii_spans[0]
    # The offset is the true original-markdown offset (round-trips to the value).
    assert text[span.char_start : span.char_end] == "bob@corp.com"
    assert span.char_start == text.index("bob@corp.com")
    # The email lives on page 2 — it must resolve there, not to page 1.
    assert span.page_start == 2
    assert span.page_end == 2
    # The masked variant is the faithful full document with only the email replaced.
    assert rec.masked_content == text.replace("bob@corp.com", "[EMAIL]")


# -- determinism --------------------------------------------------------------


def test_deterministic_output() -> None:
    doc = _doc(_TEXT)
    cfg = {"detectors": ["email", "phone"], "default_action": "replace", "salt": "x"}
    a = list(HybridPIIRedactor(cfg).redact([doc]))[0].model_dump_json()
    b = list(HybridPIIRedactor(cfg).redact([doc]))[0].model_dump_json()
    assert a == b


def test_raw_pii_never_stored_on_span() -> None:
    doc = _doc(_TEXT)
    rec = list(HybridPIIRedactor({"detectors": ["email"]}).redact([doc]))[0]
    dumped = rec.pii_spans[0].model_dump_json()
    assert "alice@acme.com" not in dumped


# -- multi-document ordering --------------------------------------------------


def test_one_record_per_document_in_first_seen_order() -> None:
    d1 = _doc("mail a@b.com", record_id="d1")
    d2 = _doc("mail c@d.com", record_id="d2")
    recs = list(HybridPIIRedactor({"detectors": ["email"]}).redact([d1, d2]))
    assert [r.document_record_id for r in recs] == ["d1", "d2"]
    assert all(r.pii_count == 1 for r in recs)


def test_parse_error_document_is_skipped() -> None:
    # A PARSE_ERROR document has no usable content; Redaction skips it (mirrors Chunk),
    # emitting no RedactionRecord for it, while a healthy sibling is still redacted.
    broken = DocumentRecord(
        record_id="bad",
        provenance=_doc("").provenance,
        classification=Classification(language="en", sensitivity="public"),
        content="",
        disposition=Disposition.PARSE_ERROR,
        error="corrupt file",
    )
    good = _doc("mail a@b.com", record_id="good")
    recs = list(HybridPIIRedactor({"detectors": ["email"]}).redact([broken, good]))
    assert [r.document_record_id for r in recs] == ["good"]


# -- config validation --------------------------------------------------------


def test_unknown_builtin_detector_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown built-in PII detector"):
        HybridPIIRedactor({"detectors": ["passport"]})


def test_unknown_action_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown redaction action"):
        HybridPIIRedactor({"default_action": "delete"})


def test_bad_policy_shape_rejected() -> None:
    with pytest.raises(TypeError):
        HybridPIIRedactor({"policies": {"public": "mask"}})


def test_bad_gazetteer_shape_rejected() -> None:
    with pytest.raises(TypeError):
        HybridPIIRedactor({"gazetteer": ["Alice"]})

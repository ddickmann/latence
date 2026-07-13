"""HybridPIIRedactor: deterministic hybrid-rule PII Redaction (S6, in-core reference).

W16 (ADR-0042): Redaction is a **chunk→chunk transform** — the Redactor consumes ChunkRecords and
yields the SAME chunks with ``masked_content`` + ``pii_spans`` populated (the clean ``content`` stays
UNMASKED). Each chunk is ≤ the model window, so PII is masked in full with NO truncation. Unit tests
for the in-core reference Provider: the masked variant + structured chunk-local PII spans
(type/offset/action/placeholder) that resolve to the correct source page, each RedactionAction
(mask/replace/hash/tag), the sensitivity-keyed policy, the config gazetteer + custom patterns,
determinism, the "raw PII never persisted" invariant, config validation, and the clean/skip/empty
degradations.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
    RedactionAction,
)
from latence_core.stages.chunk import MarkdownChunker
from latence_core.stages.redaction import HybridPIIRedactor


def _chunk(
    text: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
    record_id: str = "d1#chunk:0",
    document_record_id: str = "d1",
    index: int = 0,
) -> ChunkRecord:
    """A ChunkRecord — Redaction scans this chunk's ``content`` (W16, ADR-0042).

    Spans are chunk-local (offsets into ``content``); an identity ``offset_map`` + optional
    ``page_map`` let a span still resolve to its source page (the shared chunk page-resolution seam).
    """
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
    return ChunkRecord(
        record_id=record_id,
        document_record_id=document_record_id,
        index=index,
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=text,
        token_count=max(len(text.split()), 0),
        page_map=page_map,
        offset_map=OffsetMap(segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text))
        if text
        else None,
    )


_TEXT = "Contact alice@acme.com or call 555-123-4567 today."


# -- happy path ---------------------------------------------------------------


def test_detects_and_masks_builtin_pii() -> None:
    chunk = _chunk(_TEXT)
    rec = next(iter(HybridPIIRedactor({"detectors": ["email", "phone"]}).redact([chunk])))
    # The clean content is UNCHANGED (extraction still sees the entities).
    assert rec.content == _TEXT
    # Both the email and the phone are masked in masked_content.
    assert rec.masked_content is not None
    assert "alice@acme.com" not in rec.masked_content
    assert "555-123-4567" not in rec.masked_content
    assert "[EMAIL]" in rec.masked_content
    assert "[PHONE]" in rec.masked_content
    types = {s.pii_type for s in rec.pii_spans}
    assert types == {"email", "phone"}
    # Structured span: type, CHUNK-LOCAL offset, action, placeholder, detector.
    email = next(s for s in rec.pii_spans if s.pii_type == "email")
    assert email.detector == "regex:email"
    assert email.action is RedactionAction.MASK
    assert email.placeholder == "[EMAIL]"
    assert _TEXT[email.char_start : email.char_end] == "alice@acme.com"


def test_span_offsets_are_chunk_content_coordinates() -> None:
    # PII deep in the chunk: the span offset is the exact chunk-content offset,
    # so the slice round-trips back to the raw value (the coordinate masked_content uses).
    prefix = "# Heading\n\nSome intro paragraph before the sensitive line.\n\n"
    chunk = _chunk(prefix + _TEXT)
    rec = next(iter(HybridPIIRedactor({"detectors": ["email"]}).redact([chunk])))
    email = rec.pii_spans[0]
    assert email.char_start == len(prefix) + _TEXT.index("alice@acme.com")
    assert (prefix + _TEXT)[email.char_start : email.char_end] == "alice@acme.com"


def test_clean_chunk_yields_empty_spans_and_identical_text() -> None:
    chunk = _chunk("Nothing sensitive in this line at all.")
    rec = next(iter(HybridPIIRedactor({"detectors": ["email", "phone", "ssn"]}).redact([chunk])))
    assert rec.pii_spans == []
    assert rec.masked_content == "Nothing sensitive in this line at all."
    assert rec.content == "Nothing sensitive in this line at all."


def test_empty_chunk_is_not_an_error() -> None:
    chunk = _chunk("")
    rec = next(iter(HybridPIIRedactor({"detectors": ["email"]}).redact([chunk])))
    assert rec.pii_spans == []
    assert rec.masked_content == ""


# -- actions ------------------------------------------------------------------


def test_replace_action_pseudonymises_deterministically() -> None:
    chunk = _chunk(_TEXT)
    cfg = {"detectors": ["email"], "default_action": "replace"}
    rec = next(iter(HybridPIIRedactor(cfg).redact([chunk])))
    span = rec.pii_spans[0]
    assert span.action is RedactionAction.REPLACE
    assert span.placeholder.startswith("EMAIL_")
    assert rec.masked_content is not None
    assert "alice@acme.com" not in rec.masked_content
    assert span.placeholder in rec.masked_content


def test_hash_action_is_stable_and_hides_value() -> None:
    chunk = _chunk(_TEXT)
    cfg = {"detectors": ["email"], "default_action": "hash", "salt": "s1"}
    rec = next(iter(HybridPIIRedactor(cfg).redact([chunk])))
    span = rec.pii_spans[0]
    assert span.action is RedactionAction.HASH
    assert span.placeholder.startswith("<EMAIL:")
    assert "alice" not in span.placeholder


def test_tag_action_leaves_text_intact_but_records_span() -> None:
    chunk = _chunk(_TEXT)
    cfg = {"detectors": ["email"], "default_action": "tag"}
    rec = next(iter(HybridPIIRedactor(cfg).redact([chunk])))
    # Detect-only: the surface text is unchanged, but the span is recorded.
    assert rec.masked_content == _TEXT
    assert len(rec.pii_spans) == 1
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
    public = next(iter(provider.redact([_chunk(_TEXT, sensitivity="public")])))
    internal = next(iter(provider.redact([_chunk(_TEXT, sensitivity="internal")])))
    # Same text, different disposition driven purely by Classification.sensitivity.
    assert public.masked_content is not None and "[EMAIL]" in public.masked_content
    assert internal.masked_content == _TEXT  # tagged, not masked


def test_policy_skip_disables_redaction_for_a_class() -> None:
    cfg = {"detectors": ["email"], "policies": {"confidential": {"skip": True}}}
    rec = next(iter(HybridPIIRedactor(cfg).redact([_chunk(_TEXT, sensitivity="confidential")])))
    assert rec.pii_spans == []
    assert rec.masked_content == _TEXT


def test_policy_types_allowlist_restricts_detection() -> None:
    # Only 'email' is acted on for this sensitivity, even though phone is detectable.
    cfg = {
        "detectors": ["email", "phone"],
        "policies": {"public": {"action": "mask", "types": ["email"]}},
    }
    rec = next(iter(HybridPIIRedactor(cfg).redact([_chunk(_TEXT, sensitivity="public")])))
    assert {s.pii_type for s in rec.pii_spans} == {"email"}
    assert rec.masked_content is not None and "555-123-4567" in rec.masked_content  # phone intact


def test_default_policy_fallback() -> None:
    # A chunk whose sensitivity has no explicit policy uses the 'default' entry.
    cfg = {"detectors": ["email"], "policies": {"default": {"action": "hash"}}}
    rec = next(iter(HybridPIIRedactor(cfg).redact([_chunk(_TEXT, sensitivity="weird")])))
    assert rec.pii_spans[0].action is RedactionAction.HASH


# -- config-supplied recognisers ---------------------------------------------


def test_gazetteer_and_custom_patterns() -> None:
    text = "Employee Alice, ref A-99, emailed the team."
    chunk = _chunk(text)
    cfg = {
        "detectors": [],  # no built-ins
        "gazetteer": {"name": ["Alice"]},
        "patterns": {"ref": [r"A-\d+"]},
    }
    rec = next(iter(HybridPIIRedactor(cfg).redact([chunk])))
    types = {s.pii_type for s in rec.pii_spans}
    assert types == {"name", "ref"}
    name = next(s for s in rec.pii_spans if s.pii_type == "name")
    assert name.detector == "gazetteer:name"
    assert rec.masked_content is not None
    assert "[NAME]" in rec.masked_content
    assert "[REF]" in rec.masked_content


# -- page resolution (round-trip through offset_map + page_map) ----------------


def test_span_page_resolves_through_page_map() -> None:
    # Two-page document; the PII sits on page 2. Chunk the whole document so the chunk carries the
    # real offset_map + page_map; the chunk-local span must still resolve to page 2.
    assembly = PageMap.from_page_texts(["First page has nothing.", "Reach alice@acme.com here."])
    chunk = _chunk(assembly.text, page_map=assembly.page_map)
    rec = next(iter(HybridPIIRedactor({"detectors": ["email"]}).redact([chunk])))
    span = rec.pii_spans[0]
    assert span.page_start == 2
    assert span.page_end == 2


def test_markup_stripped_chunk_offsets_and_page_are_faithful() -> None:
    # W16 (ADR-0042) the chunk-level analogue of the old ORIGINAL-markdown regression: a REAL chunk
    # (markup-stripped ``content`` + its offset_map) whose PII sits on page 2. The chunk-local span
    # offset round-trips to the value in ``content``; the page resolves through offset_map→page_map
    # to the ORIGINAL page 2 (never the naive page 1).
    from latence_core.contracts import DocumentRecord

    page1 = "# Confidential Report Heading\n\n**Overview** of the [system](http://ex.com)."
    page2 = "Reach the admin at bob@corp.com immediately."
    assembly = PageMap.from_page_texts([page1, page2])
    doc = DocumentRecord(
        record_id="d1",
        provenance=Provenance(
            source_uri="file:///c/a.md",
            file_name="a.md",
            file_type="md",
            file_size=len(assembly.text),
            document_id="sha256:d1",
            char_start=0,
            char_end=len(assembly.text),
        ),
        classification=Classification(language="en", sensitivity="public"),
        content=assembly.text,
        page_map=assembly.page_map,
    )
    chunks = list(MarkdownChunker({}).chunk([doc]))
    out = list(HybridPIIRedactor({"detectors": ["email"]}).redact(chunks))
    spans = [(c, s) for c in out for s in c.pii_spans]
    assert len(spans) == 1
    chunk, span = spans[0]
    # The offset is chunk-local into the (markup-stripped) content and round-trips to the value.
    assert chunk.content[span.char_start : span.char_end] == "bob@corp.com"
    # The email lives on page 2 — it must resolve there, not to page 1.
    assert span.page_start == 2
    assert span.page_end == 2
    # The masked variant is the faithful chunk content with only the email replaced.
    assert chunk.masked_content == chunk.content.replace("bob@corp.com", "[EMAIL]")


# -- determinism --------------------------------------------------------------


def test_deterministic_output() -> None:
    chunk = _chunk(_TEXT)
    cfg = {"detectors": ["email", "phone"], "default_action": "replace", "salt": "x"}
    a = next(iter(HybridPIIRedactor(cfg).redact([chunk]))).model_dump_json()
    b = next(iter(HybridPIIRedactor(cfg).redact([chunk]))).model_dump_json()
    assert a == b


def test_raw_pii_never_stored_on_span() -> None:
    chunk = _chunk(_TEXT)
    rec = next(iter(HybridPIIRedactor({"detectors": ["email"]}).redact([chunk])))
    dumped = rec.pii_spans[0].model_dump_json()
    assert "alice@acme.com" not in dumped


# -- multi-chunk ordering -----------------------------------------------------


def test_chunks_threaded_one_to_one_in_first_seen_order() -> None:
    c1 = _chunk("mail a@b.com", record_id="d1#chunk:0", document_record_id="d1", index=0)
    c2 = _chunk("mail c@d.com", record_id="d2#chunk:0", document_record_id="d2", index=0)
    recs = list(HybridPIIRedactor({"detectors": ["email"]}).redact([c1, c2]))
    assert [r.record_id for r in recs] == ["d1#chunk:0", "d2#chunk:0"]
    assert all(len(r.pii_spans) == 1 for r in recs)
    # Content is unchanged; only masked_content carries the redaction.
    assert [r.content for r in recs] == ["mail a@b.com", "mail c@d.com"]


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

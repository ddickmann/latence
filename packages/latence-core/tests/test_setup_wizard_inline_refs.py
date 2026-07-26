"""The enterprise-sota wizard WIRES the deterministic INLINE-CITATION cross-reference extractor.

Regression for the dogfood measured gap. The header-only ``fused_entity_relation.header_refs``
Provider recovers an RFC's ``Obsoletes:``/``Updates:`` HEADER edges, but the overwhelming majority
of a real RFC corpus's gold cross-references are INLINE body citations ("RFC 7230", "[RFC9110]")
scattered through the document body — which a body chunk (carrying no self id) makes structurally
invisible to the header-only extractor (measured ~14% reference recall). The
``fused_entity_relation.inline_refs`` Provider recovers exactly those, but only if the wizard WIRES
it. These tests pin that wiring:

* the wizard SNIFFS an RFC-shaped corpus and returns the ready ``inline_refs`` config
  (:func:`detect_inline_family`), and returns ``None`` for a corpus with no such headers;
* the emitted enterprise-sota pipeline carries an ``inline_refs`` extraction Stage that feeds
  Disambiguation ALONGSIDE the learned ``extract`` and the ``header_refs`` Stages — and, with no
  family detected, carries no such Stage (the off-by-default, byte-identical-wiring invariant);
* the EMITTED config, handed to the REAL Provider (no fake), recovers the actual inline body
  citations from body chunks that carry no self declaration — the whole point of the feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from _setup_stack import try_stage
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.setup import SetupAnswers, build_stack_config, detect_inline_family
from latence_core.setup.wizard import PROFILE_LITE, PROFILE_SOTA
from latence_core.stages.inline_refs import InlineReferenceExtractor

if TYPE_CHECKING:
    from pathlib import Path

_RFC_FRONT_MATTER = (
    "Internet Engineering Task Force (IETF)                      R. Fielding, Ed.\n"
    "Request for Comments: 9110\n"
    "Obsoletes: 7230\n"
    "Category: Standards Track\n"
    "\n"
    "                          HTTP Semantics\n"
)
_RFC_BODY = (
    "As described in RFC 3986, a URI reference is resolved against a base. "
    "See also [RFC5234] for the ABNF core rules used throughout this document.\n"
)


def _rfc_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "corpus"
    src.mkdir(exist_ok=True)
    (src / "rfc9110.txt").write_text(_RFC_FRONT_MATTER + _RFC_BODY, encoding="utf-8")
    return src


def _plain_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "plain"
    src.mkdir(exist_ok=True)
    (src / "note.txt").write_text(
        "A memo about quarterly revenue. No cross-reference headers here.\n", encoding="utf-8"
    )
    return src


def _answers(
    src: Path, tmp_path: Path, *, profile: str, inline_refs: dict[str, Any] | None
) -> SetupAnswers:
    return SetupAnswers(
        source=f"file://{src}",
        storage=f"file://{tmp_path}/out",
        parser="parser.plaintext",
        device="auto",
        pii=True,
        extract_on="unmasked",
        profile=profile,
        extensions=("txt",),
        inline_refs=inline_refs,
    )


# Stage lookups come from the shared `_setup_stack` helper (C7): five setup test modules used
# to re-implement this same render-parse-search one-liner. Where a test is about ONE Stage's
# config rather than the whole document, prefer asserting the contribution's block directly —
# see `test_setup_recipe.py`.

# --- detection -------------------------------------------------------------


def test_detect_inline_family_recognises_rfc_headers(tmp_path: Path) -> None:
    """An RFC-shaped corpus is sniffed as the RFC family and yields a ready inline_refs config."""
    config = detect_inline_family(f"file://{_rfc_corpus(tmp_path)}")
    assert config is not None
    assert config["id_template"] == "RFC {id}"
    assert config["relation_label"] == "references"
    assert "citation_pattern" in config
    assert "source_id_pattern" in config  # the provenance-file-name fallback resolver


def test_detect_inline_family_ignores_a_headerless_corpus(tmp_path: Path) -> None:
    """A corpus with no recognised metadata headers detects nothing (no spurious wiring)."""
    assert detect_inline_family(f"file://{_plain_corpus(tmp_path)}") is None


def test_detect_inline_family_is_none_for_a_missing_source(tmp_path: Path) -> None:
    """A non-existent source degrades to None (never raises), like the other scan helpers."""
    assert detect_inline_family(f"file://{tmp_path}/does-not-exist") is None


# --- wiring ----------------------------------------------------------------


def test_sota_pipeline_wires_inline_refs_stage(tmp_path: Path) -> None:
    """With a detected RFC family, the emitted enterprise-sota pipeline carries an inline_refs
    extraction Stage that reads the induced chunk stream and feeds Disambiguation alongside extract.
    """
    config = detect_inline_family(f"file://{_rfc_corpus(tmp_path)}")
    assert config is not None
    text = build_stack_config(
        _answers(_rfc_corpus(tmp_path), tmp_path, profile=PROFILE_SOTA, inline_refs=config)
    )
    stage = try_stage(text, "inline_refs")
    assert stage is not None, "the enterprise-sota pipeline must wire an inline_refs Stage"
    assert stage.provider == "fused_entity_relation.inline_refs"
    assert stage.depends_on == ["induce"]

    # Both extractors feed the resolution phase. Since ADR-0054 they feed it through the
    # type-consolidation Stage, which canonicalizes the drifted induced TYPE vocabulary before
    # entity resolution reads it — so that Stage gathers the extractors and `disambiguate` reads it.
    consolidate = try_stage(text, "consolidate_types")
    assert consolidate is not None
    assert "extract" in consolidate.depends_on
    assert "inline_refs" in consolidate.depends_on

    disambiguate = try_stage(text, "disambiguate")
    assert disambiguate is not None
    assert disambiguate.depends_on == ["consolidate_types"]


def test_sota_pipeline_omits_inline_refs_when_no_family(tmp_path: Path) -> None:
    """No detected family ⇒ NO inline_refs Stage and disambiguate depends on extract alone (the
    off-by-default, byte-identical-wiring invariant)."""
    text = build_stack_config(
        _answers(_plain_corpus(tmp_path), tmp_path, profile=PROFILE_SOTA, inline_refs=None)
    )
    assert try_stage(text, "inline_refs") is None
    assert try_stage(text, "consolidate_types").depends_on == ["extract"]
    assert try_stage(text, "disambiguate").depends_on == ["consolidate_types"]


def test_lite_profile_never_wires_inline_refs(tmp_path: Path) -> None:
    """The lite profile carries no inline_refs Stage (the feature is enterprise-sota only)."""
    text = build_stack_config(
        _answers(_rfc_corpus(tmp_path), tmp_path, profile=PROFILE_LITE, inline_refs=None)
    )
    assert try_stage(text, "inline_refs") is None


# --- end-to-end: the EMITTED config recovers real inline edges (no fake) ----


def _chunk(text: str, *, index: int, char_start: int) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///corpus-rfc/rfc9110.txt",
        file_name="rfc9110.txt",
        file_type="txt",
        file_size=len(text),
        document_id="sha256:rfc9110",
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=f"rfc9110#chunk-{index}",
        document_record_id="rfc9110",
        index=index,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
    )


def test_emitted_config_recovers_inline_body_citations(tmp_path: Path) -> None:
    """The inline_refs config the wizard emits, driven through the REAL Provider over a front-matter
    chunk + a BODY chunk that has no self id, recovers the inline citations as references edges and
    skips the header Obsoletes id (RFC 7230) — proving the wiring is not fake green."""
    config = detect_inline_family(f"file://{_rfc_corpus(tmp_path)}")
    assert config is not None
    text = build_stack_config(
        _answers(_rfc_corpus(tmp_path), tmp_path, profile=PROFILE_SOTA, inline_refs=config)
    )
    emitted = try_stage(text, "inline_refs").config

    provider = InlineReferenceExtractor(dict(emitted))
    chunks = [
        _chunk(_RFC_FRONT_MATTER, index=0, char_start=0),
        _chunk(_RFC_BODY, index=1, char_start=len(_RFC_FRONT_MATTER)),  # no self id here
    ]
    edges = {(r.head_text, r.label, r.tail_text) for r in provider.extract_fused(chunks).relations}
    assert edges == {
        ("RFC 9110", "references", "RFC 3986"),
        ("RFC 9110", "references", "RFC 5234"),
    }

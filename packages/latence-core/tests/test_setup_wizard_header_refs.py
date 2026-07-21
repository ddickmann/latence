"""The enterprise-sota wizard WIRES the deterministic structured-header cross-reference extractor.

Regression for dogfood-3-inspect. The in-core, zero-dep ``fused_entity_relation.header_refs``
Provider recovers an RFC corpus's explicit ``Obsoletes:``/``Updates:`` header edges — the ones the
entity-anchored learned extractor structurally cannot see, and the ones the tutorial's §3.1
correctness bar compares against the independent ``_refs.jsonl`` graph. That Provider was built,
registered, and conformance-tested, but the wizard-generated enterprise-sota pipeline never emitted
a Stage for it, so the RFC-retrieval-proof run recovered ZERO such edges (induced-vs-independent
overlap = 0). These tests pin the missing wiring:

* the wizard SNIFFS an RFC-shaped corpus and returns the ready ``header_refs`` config
  (:func:`detect_header_family`), and returns ``None`` for a corpus with no such headers;
* the emitted enterprise-sota pipeline carries a ``header_refs`` extraction Stage that feeds
  Disambiguation ALONGSIDE the learned ``extract`` Stage — and, with no family detected, carries no
  such Stage and the disambiguate wiring is byte-identical to before (the off-by-default invariant);
* the EMITTED config, handed to the REAL Provider (no fake), recovers the actual RFC header edges
  with endpoints rendered as ``RFC <n>`` — the exact shape the benchmark's overlap chart parses.

Pre-fix (no ``header_refs`` field, no builder wiring) the pipeline has no such Stage, so
:func:`test_sota_pipeline_wires_header_refs_stage` and the end-to-end recovery test fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.pipeline import Pipeline
from latence_core.setup import SetupAnswers, build_stack_config, detect_header_family
from latence_core.setup.wizard import PROFILE_LITE, PROFILE_SOTA
from latence_core.stages.header_refs import HeaderReferenceExtractor

if TYPE_CHECKING:
    from pathlib import Path

# The RFC front-matter shape build_corpus.py mines for the independent _refs.jsonl gold graph: a
# "Request for Comments: N" self declaration plus bare-number Obsoletes:/Updates: header lines.
_RFC_FRONT_MATTER = (
    "Internet Engineering Task Force (IETF)                      R. Fielding, Ed.\n"
    "Request for Comments: 9110\n"
    "STD: 97\n"
    "Obsoletes: 2818, 7230, 7231\n"
    "Updates: 3864\n"
    "Category: Standards Track\n"
    "\n"
    "                          HTTP Semantics\n"
)


def _rfc_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "corpus"
    src.mkdir(exist_ok=True)
    (src / "rfc9110.txt").write_text(_RFC_FRONT_MATTER, encoding="utf-8")
    (src / "rfc9111.txt").write_text(
        "Request for Comments: 9111\nObsoletes: 7234\n\n   HTTP Caching\n", encoding="utf-8"
    )
    return src


def _plain_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "plain"
    src.mkdir(exist_ok=True)
    (src / "note.txt").write_text(
        "A memo about quarterly revenue. No cross-reference headers here.\n", encoding="utf-8"
    )
    return src


def _answers(src: Path, tmp_path: Path, *, profile: str, header_refs: dict[str, Any] | None) -> SetupAnswers:
    return SetupAnswers(
        source=f"file://{src}",
        storage=f"file://{tmp_path}/out",
        parser="parser.plaintext",
        device="auto",
        pii=True,
        extract_on="unmasked",
        profile=profile,
        extensions=("txt",),
        header_refs=header_refs,
    )


def _stage(text: str, name: str) -> Any:
    return next((s for s in Pipeline.from_yaml_text(text).stages if s.name == name), None)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detect_header_family_recognises_rfc_headers(tmp_path: Path) -> None:
    """An RFC-shaped corpus is sniffed as the RFC family and yields a ready header_refs config."""
    config = detect_header_family(f"file://{_rfc_corpus(tmp_path)}")
    assert config is not None
    assert config["id_template"] == "RFC {id}"
    relations = config["relations"]
    assert isinstance(relations, dict)
    assert set(relations) == {"obsoletes", "updates"}


def test_detect_header_family_ignores_a_headerless_corpus(tmp_path: Path) -> None:
    """A corpus with no recognised metadata headers detects nothing (no spurious wiring)."""
    assert detect_header_family(f"file://{_plain_corpus(tmp_path)}") is None


def test_detect_header_family_is_none_for_a_missing_source(tmp_path: Path) -> None:
    """A non-existent source degrades to None (never raises), like the other scan helpers."""
    assert detect_header_family(f"file://{tmp_path}/does-not-exist") is None


# ---------------------------------------------------------------------------
# wiring (the regression: this Stage was missing entirely)
# ---------------------------------------------------------------------------


def test_sota_pipeline_wires_header_refs_stage(tmp_path: Path) -> None:
    """With a detected RFC family, the emitted enterprise-sota pipeline carries a header_refs
    extraction Stage that reads the induced chunk stream and feeds Disambiguation alongside extract.
    """
    config = detect_header_family(f"file://{_rfc_corpus(tmp_path)}")
    assert config is not None
    text = build_stack_config(_answers(_rfc_corpus(tmp_path), tmp_path, profile=PROFILE_SOTA, header_refs=config))

    header_stage = _stage(text, "header_refs")
    assert header_stage is not None, "the enterprise-sota pipeline must wire a header_refs Stage"
    assert header_stage.provider == "fused_entity_relation.header_refs"
    assert header_stage.depends_on == ["induce"]

    disambiguate = _stage(text, "disambiguate")
    assert disambiguate is not None
    # Both extractors feed Disambiguation (the Runner concatenates parents' records by carrier type).
    assert "extract" in disambiguate.depends_on
    assert "header_refs" in disambiguate.depends_on


def test_sota_pipeline_omits_header_refs_when_no_family(tmp_path: Path) -> None:
    """No detected family ⇒ NO header_refs Stage and disambiguate depends on extract alone (the
    off-by-default, byte-identical-wiring invariant)."""
    text = build_stack_config(_answers(_plain_corpus(tmp_path), tmp_path, profile=PROFILE_SOTA, header_refs=None))
    assert _stage(text, "header_refs") is None
    assert _stage(text, "disambiguate").depends_on == ["extract"]


def test_lite_profile_never_wires_header_refs(tmp_path: Path) -> None:
    """The lite profile carries no header_refs Stage (the feature is enterprise-sota only)."""
    text = build_stack_config(_answers(_rfc_corpus(tmp_path), tmp_path, profile=PROFILE_LITE, header_refs=None))
    assert _stage(text, "header_refs") is None


# ---------------------------------------------------------------------------
# end-to-end: the EMITTED config recovers real edges (no fake)
# ---------------------------------------------------------------------------


def _chunk(text: str) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///corpus-rfc/rfc9110.txt",
        file_name="rfc9110.txt",
        file_type="txt",
        file_size=len(text),
        document_id="sha256:rfc9110",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id="rfc9110#chunk-0",
        document_record_id="rfc9110",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
    )


def test_emitted_config_recovers_real_rfc_edges(tmp_path: Path) -> None:
    """The header_refs config the wizard emits, driven through the REAL Provider over an RFC chunk,
    recovers the obsoletes/updates edges with ``RFC <n>`` endpoints — the exact tuples the
    benchmark's induced-vs-independent overlap chart parses (proving the wiring is not fake green)."""
    config = detect_header_family(f"file://{_rfc_corpus(tmp_path)}")
    assert config is not None
    text = build_stack_config(_answers(_rfc_corpus(tmp_path), tmp_path, profile=PROFILE_SOTA, header_refs=config))
    emitted = _stage(text, "header_refs").config

    provider = HeaderReferenceExtractor(dict(emitted))
    fused = provider.extract_fused([_chunk(_RFC_FRONT_MATTER)])
    edges = {(r.head_text, r.label, r.tail_text) for r in fused.relations}
    assert edges == {
        ("RFC 9110", "obsoletes", "RFC 2818"),
        ("RFC 9110", "obsoletes", "RFC 7230"),
        ("RFC 9110", "obsoletes", "RFC 7231"),
        ("RFC 9110", "updates", "RFC 3864"),
    }

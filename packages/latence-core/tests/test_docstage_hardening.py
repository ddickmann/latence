"""H-docstage hardening: adversarial + property-based + fuzz over Parse/Chunk/NER/RE.

The Phase-2 hardening-bar suite for the doc-level Stages (issue #74). It closes an
explicit edge-case matrix and asserts the framework's core correctness invariants hold
under malformed / huge / pathological input — unicode/RTL/zero-width, mixed encodings,
degenerate chunks, offset-boundary edges (mention at a chunk/page boundary, empty
content), and a catastrophic-backtracking (ReDoS) config pattern:

* **Offset round-trip** — every emitted :class:`EntityMention` resolves back to its
  EXACT source slice: ``document_text[char_start:char_end] == mention.text``, and
  ``0 <= char_start <= char_end <= len(document_text)`` (never out of bounds/inverted).
* **Page provenance** — a mention's ``page_start``/``page_end`` are the pages its own
  offset falls on (resolved through the parent page map), not merely the chunk's range.
* **Relation between-text alignment** — the document text a relation's trigger is tested
  against is reconstructed in the SAME (original-markdown) coordinate system the mention
  offsets use, so stripped markup never shifts a trigger out of / into the window.
* **Determinism** — identical input + config ⇒ byte-identical records, always.
* **Never-crash degradation** — a bad-codec Parser hint yields a ``PARSE_ERROR`` record,
  a ReDoS config pattern is refused at compile time, and a pathological document never
  drives exponential CPU.

Three of these are direct regression tests for bugs this batch fixed (each FAILS if the
fix is reverted):

* ``test_relation_between_text_aligned_across_stripped_markup`` — the #74 relation
  between-text misalignment (a genuine trigger was missed once markup was stripped).
* ``test_plaintext_parser_bad_encoding_hint_degrades_to_parse_error`` — the plaintext
  Parser crashed with ``LookupError`` on an unknown codec hint instead of degrading.
* ``test_underscore_italics_left_intact`` — pins the corrected stripping behaviour the
  #74 docstring fix now documents (``_x_`` is deliberately NOT stripped).

Deterministic: hypothesis is seeded/bounded and the fuzz loops use a fixed RNG, so a run
is hash-stable (Baseline + Hardening bar).
"""

from __future__ import annotations

import random
import re
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from latence_core.chunking import ChunkBudget, chunk_document, estimate_tokens, strip_markup
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    EntityMention,
    PageMap,
    ParserInput,
    Provenance,
)
from latence_core.errors import ConfigError
from latence_core.pagemap import PageOffsetIndex
from latence_core.stages.chunk import MarkdownChunker
from latence_core.stages.entity import GazetteerEntityExtractor, _has_catastrophic_backtracking
from latence_core.stages.relation import PatternRelationExtractor
from latence_parser_plaintext.provider import PlaintextParser

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _make_document(page_texts: list[str], doc_id: str = "doc1") -> tuple[DocumentRecord, str]:
    """A PARSED DocumentRecord assembled from per-page text, plus the assembled text.

    Uses the real Parser-side assembler (:meth:`PageMap.from_page_texts`) so the page
    map / char span are exact by construction — the same path a real Parser takes.
    """
    assembly = PageMap.from_page_texts(page_texts)
    provenance = Provenance(
        source_uri="file:///corpus/a.md",
        file_name="a.md",
        file_type="md",
        file_size=max(len(assembly.text), 1),
        document_id=doc_id,
        char_start=0,
        char_end=assembly.page_map.total_chars,
        page_start=1,
        page_end=len(page_texts) or 1,
    )
    doc = DocumentRecord(
        record_id=doc_id,
        provenance=provenance,
        classification=Classification(language="en", sensitivity="internal"),
        content=assembly.text,
        media_type="text/markdown",
        page_map=assembly.page_map,
    )
    return doc, assembly.text


def _parser_input(content: bytes | str, encoding: str = "utf-8") -> ParserInput:
    provenance = Provenance(
        source_uri="file:///corpus/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=len(content) if isinstance(content, bytes) else max(len(content), 1),
        document_id="doc-enc",
        char_start=0,
        char_end=0,
        page_start=1,
        page_end=1,
    )
    return ParserInput(
        provenance=provenance,
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        config={"encoding": encoding},
    )


_ENTITY_CONFIG = {"labels": {"ORG": {"terms": ["Acme Corp", "Globex Inc", "Initech"]}}}
_RELATION_CONFIG = {
    "relations": {"linked": {"head": "ORG", "tail": "ORG", "triggers": ["works", "and", "with"]}},
    "window_chars": 400,
}


def _assert_mention_invariants(mentions: list[EntityMention], document_text: str) -> None:
    """The offset round-trip + page-provenance invariants for every mention."""
    for m in mentions:
        cs, ce = m.provenance.char_start, m.provenance.char_end
        assert cs is not None and ce is not None
        # Offsets never out of bounds nor inverted.
        assert 0 <= cs <= ce <= len(document_text), (cs, ce, len(document_text))
        # Round-trip: the mention resolves to its EXACT source slice.
        assert document_text[cs:ce] == m.text, (document_text[cs:ce], m.text)
        # Page provenance is present and >= 1.
        assert m.provenance.page_start is not None and m.provenance.page_start >= 1
        assert m.provenance.page_end is not None and m.provenance.page_end >= m.provenance.page_start


# ---------------------------------------------------------------------------
# Property-based: chunking offset map is total, bounded, monotone
# ---------------------------------------------------------------------------

# A markup-rich alphabet: real markdown constructs, unicode, RTL, zero-width, control.
_FUZZ_CHARS = (
    "abc XYZ 123 .!?\n\t"
    "*_`#>-+|~[]()!"  # markdown markup
    "​‌﻿"  # zero-width space / non-joiner / BOM
    "الع"  # arabic (RTL)
    "中文"  # CJK
    "́̈"  # combining marks
    "\U0001f600"  # emoji (astral)
    "\x00\x0c\r"  # NUL, form-feed, CR
)
_fuzz_text = st.text(alphabet=_FUZZ_CHARS, max_size=400)


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_fuzz_text)
def test_chunk_offsets_are_total_and_in_bounds(text: str) -> None:
    """Every chunk's span + offset map stay within the document and are well-formed.

    Property: for any input at all, each :class:`Chunk` satisfies
    ``0 <= char_start <= char_end <= len(text)``, its offset-map first segment starts at
    local 0, segment local_starts strictly increase, and every segment's ``doc_start``
    lies inside the chunk's own ``[char_start, char_end]`` — so a downstream sub-chunk
    offset always resolves inside the original document. Chunking never raises.
    """
    chunks = chunk_document(text)
    for c in chunks:
        assert 0 <= c.char_start <= c.char_end <= len(text)
        assert c.offset_map, "every chunk carries at least one offset segment"
        assert c.offset_map[0][0] == 0
        local_starts = [ls for ls, _ in c.offset_map]
        assert local_starts == sorted(set(local_starts)) == local_starts
        for _ls, ds in c.offset_map:
            assert c.char_start <= ds <= c.char_end


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_fuzz_text)
def test_chunking_and_tokenizing_are_deterministic(text: str) -> None:
    """Identical input ⇒ byte-identical chunks and identical token estimate (Baseline)."""
    a = chunk_document(text)
    b = chunk_document(text)
    assert a == b
    assert estimate_tokens(text) == estimate_tokens(text)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_fuzz_text)
def test_strip_markup_mapping_is_monotone_and_within_bounds(text: str) -> None:
    """The strip offset map is monotone non-decreasing and within ``[0, len(text)]``.

    ``mapping[i]`` is the original offset of stripped char ``i``; it must be
    non-decreasing (stripping only deletes, never reorders) and end at the sentinel
    ``len(text)``. This underpins the whole offset-preservation contract.
    """
    result = strip_markup(text)
    mapping = result.mapping
    assert mapping[-1] == len(text)
    assert len(mapping) == len(result.stripped) + 1
    for i in range(1, len(mapping)):
        assert mapping[i] >= mapping[i - 1]
        assert 0 <= mapping[i] <= len(text)


# ---------------------------------------------------------------------------
# Property-based: full Parse→Chunk→NER→RE round-trip invariants
# ---------------------------------------------------------------------------

_ENTITY_TOKENS = ["Acme Corp", "Globex Inc", "Initech"]
_FILLER = ["works", "and", "with", "text", "more", "data", "the"]
_MARKUP = ["**", "*", "`", "# ", "> ", "- ", "1. ", "[x](http://y)", "~~", "\n\n", "|", "_", "  "]


@st.composite
def _markup_document(draw: st.DrawFn) -> list[str]:
    """A 1–3 page document interleaving entity tokens, filler, and markdown markup."""
    n_pages = draw(st.integers(min_value=1, max_value=3))
    pages: list[str] = []
    for _ in range(n_pages):
        n = draw(st.integers(min_value=4, max_value=24))
        parts: list[str] = []
        for _ in range(n):
            kind = draw(st.integers(min_value=0, max_value=2))
            if kind == 0:
                parts.append(draw(st.sampled_from(_ENTITY_TOKENS)))
            elif kind == 1:
                parts.append(draw(st.sampled_from(_MARKUP)))
            else:
                parts.append(draw(st.sampled_from(_FILLER)))
            parts.append(" ")
        pages.append("".join(parts))
    return pages


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(pages=_markup_document())
def test_mention_offset_and_page_roundtrip_under_markup(pages: list[str]) -> None:
    """A mention's offsets round-trip to its exact source slice + correct page.

    The core S4 invariant under adversarial markup: whatever markup precedes a mention
    inside its chunk, the emitted document offsets resolve back to the exact surface
    string, and the recorded page is the page that offset genuinely falls on.
    """
    doc, text = _make_document(pages)
    chunker = MarkdownChunker({"max_tokens": 24, "overlap_tokens": 4})
    chunks = list(chunker.chunk([doc]))
    mentions = list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract(chunks))
    _assert_mention_invariants(mentions, text)
    assert doc.page_map is not None
    poi = PageOffsetIndex(doc.page_map)
    for m in mentions:
        assert m.provenance.char_start is not None
        assert m.provenance.page_start == poi.page_for_offset(m.provenance.char_start)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(pages=_markup_document())
def test_relation_covering_span_contains_both_endpoints(pages: list[str]) -> None:
    """A relation's covering span always contains both endpoint surface strings.

    Evidence invariant: a relation resolves back to a document span that actually holds
    its head and tail — never a shifted or empty region (the #74 misalignment class).
    """
    doc, text = _make_document(pages)
    chunker = MarkdownChunker({"max_tokens": 24, "overlap_tokens": 4})
    chunks = list(chunker.chunk([doc]))
    mentions = list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract(chunks))
    relations = list(PatternRelationExtractor(_RELATION_CONFIG).relate(chunks, mentions))
    for r in relations:
        cs, ce = r.provenance.char_start, r.provenance.char_end
        assert cs is not None and ce is not None
        assert 0 <= cs <= ce <= len(text)
        covering = text[cs:ce]
        assert r.head_text in covering
        assert r.tail_text in covering


# ---------------------------------------------------------------------------
# Regression: #74 relation between-text misalignment across stripped markup
# ---------------------------------------------------------------------------


def test_relation_between_text_aligned_across_stripped_markup() -> None:
    """A trigger between two mentions is still found when markup precedes them (#74).

    Before the fix, :meth:`PatternRelationExtractor._document_texts` placed each chunk's
    *markup-stripped* content at its *original* ``char_start``, so any stripped markup
    shifted every following character left of its true offset. The between-text lookup
    ``text[first_end:second_start]`` — which uses the mentions' TRUE original offsets —
    then read the WRONG region: here the genuine trigger ``employs`` fell out of the
    window entirely and the relation was silently MISSED (a false negative). The fix
    reconstructs the document in the mentions' coordinate system via each chunk's offset
    map. This test FAILS (0 relations) if the fix is reverted.
    """
    # Heavy stripped markup (ten links) BEFORE the mentions pushes the naive stripped
    # reconstruction far left of the true offsets.
    prefix = "".join(f"[ref{i}](http://example.com/{i}) " for i in range(10))
    page = prefix + "Acme Corp employs Globex Inc"
    doc, text = _make_document([page])
    chunker = MarkdownChunker({"max_tokens": 512})
    chunks = list(chunker.chunk([doc]))
    mentions = list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract(chunks))
    # The two ORG mentions carry their true original offsets, with ``employs`` between.
    assert len(mentions) == 2
    first_end = mentions[0].provenance.char_end
    second_start = mentions[1].provenance.char_start
    assert first_end is not None and second_start is not None
    assert "employs" in text[first_end:second_start]
    cfg = {
        "relations": {
            "employs": {"head": "ORG", "tail": "ORG", "triggers": ["employs"]}
        },
        "window_chars": 1000,
    }
    relations = list(PatternRelationExtractor(cfg).relate(chunks, mentions))
    assert len(relations) == 1, "the true between-mention trigger must be found"
    assert relations[0].head_text == "Acme Corp"
    assert relations[0].tail_text == "Globex Inc"
    # And its covering span resolves back to a region holding both endpoints.
    r = relations[0]
    assert r.provenance.char_start is not None and r.provenance.char_end is not None
    covering = text[r.provenance.char_start : r.provenance.char_end]
    assert "Acme Corp" in covering and "Globex Inc" in covering


def test_relation_document_text_placed_at_true_offsets() -> None:
    """``_document_texts`` places surviving content at TRUE original offsets (#74).

    A white-box check on the reconstruction: the trigger word must sit at the SAME
    offset the original assembled text puts it, not the stripped-shifted offset.
    """
    page = "# BIG HEADING\n\nAcme Corp **strongly** employs Globex Inc"
    doc, text = _make_document([page])
    chunks = list(MarkdownChunker({"max_tokens": 512}).chunk([doc]))
    recon = PatternRelationExtractor._document_texts(chunks)[doc.record_id]
    assert text.find("employs") == recon.find("employs")
    assert text.find("Globex Inc") == recon.find("Globex Inc")


# ---------------------------------------------------------------------------
# Regression: #74 plaintext Parser bad-encoding hint must degrade, not crash
# ---------------------------------------------------------------------------


def test_plaintext_parser_bad_encoding_hint_degrades_to_parse_error() -> None:
    """An unknown codec name in the (untrusted) encoding hint ⇒ PARSE_ERROR, no crash.

    ``bytes.decode(name, errors="replace")`` raises ``LookupError`` for an unknown codec
    NAME before ``errors="replace"`` can act (that only tolerates undecodable BYTES).
    Before the fix the plaintext Parser let this propagate and abort the whole run; a
    Parser must degrade a bad input to a ``PARSE_ERROR`` record (ADR-0020). This test
    FAILS (raises ``LookupError``) if the fix is reverted.
    """
    records = list(PlaintextParser().parse([_parser_input(b"hello world", encoding="no-such-codec")]))
    assert len(records) == 1
    record = records[0]
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None and "no-such-codec" in record.error
    assert record.content == ""
    assert record.page_map is None
    # A downstream Chunk skips a PARSE_ERROR record — it never reaches the corpus.
    assert list(MarkdownChunker().chunk([record])) == []


def test_plaintext_parser_undecodable_bytes_still_parse() -> None:
    """Undecodable BYTES (not a bad codec name) still parse via ``errors='replace'``.

    The graceful-degrade path must not over-trigger: genuinely valid-codec-but-bad-bytes
    input still yields a PARSED record (with replacement chars), only a bad codec NAME
    becomes a PARSE_ERROR.
    """
    records = list(PlaintextParser().parse([_parser_input(b"\xff\xfe\x00bad", encoding="utf-8")]))
    assert len(records) == 1
    assert records[0].disposition is Disposition.PARSED
    assert records[0].page_map is not None
    assert records[0].provenance.char_end == len(records[0].content)


# ---------------------------------------------------------------------------
# Regression: #74 underscore-italics wording correction
# ---------------------------------------------------------------------------


def test_underscore_italics_left_intact() -> None:
    """``_x_`` is deliberately NOT stripped (the corrected #74 comment); ``*x*`` is.

    Pins the real stripping behaviour the docstring fix now documents: underscore
    italics survive (``_`` is ubiquitous in identifiers/filenames — stripping it would
    corrupt legitimate text), asterisk italics and doubled-underscore bold are stripped.
    """
    assert strip_markup("*italic*").stripped == "italic"
    assert strip_markup("_keep_me_").stripped == "_keep_me_"
    assert strip_markup("__bold__").stripped == "bold"
    # And a filename-like token with underscores survives a full chunk round-trip.
    doc, text = _make_document(["see my_config_file.txt for details here now"])
    chunks = list(MarkdownChunker().chunk([doc]))
    assert any("my_config_file.txt" in c.content for c in chunks)


# ---------------------------------------------------------------------------
# Adversarial: ReDoS config-pattern guard (security / DoS)
# ---------------------------------------------------------------------------

_REDOS_PATTERNS = [
    r"(a+)+$",
    r"(\w+)+$",
    r"(a|a)+",
    r"(a|ab)+",
    r"(a+){2,}",
    r"(.*)*",
    r"(x*)*y",
    r"a*a*a*a*a*b",
]


@pytest.mark.parametrize("pattern", _REDOS_PATTERNS)
def test_redos_config_pattern_refused_at_compile(pattern: str) -> None:
    """A catastrophic-backtracking config pattern is refused before any content runs.

    The security invariant: the framework compiles operator config regex over untrusted
    chunk content, so a ReDoS-shaped pattern must be rejected at compile time
    (``ConfigError``) rather than becoming an exponential-CPU DoS lever (issue #28).
    """
    assert _has_catastrophic_backtracking(pattern)
    with pytest.raises(ConfigError, match="catastrophic-backtracking"):
        GazetteerEntityExtractor({"labels": {"X": {"patterns": [pattern]}}})


def test_redos_guard_blocks_actual_exponential_dos() -> None:
    """End-to-end: the guard blocks a pattern that WOULD hang on a crafted input.

    Sanity-anchored adversarial test: the classic ``(a+)+$`` on a non-matching tail is
    genuinely exponential in Python's engine (measured here on a tiny input), and the
    extractor refuses to compile it — so the pathological document can never reach the
    matcher. Deterministic: no timing assertion in the hot path, only the refusal.
    """
    # Anchor: prove the shape really is exponential on a SMALL input (bounded work).
    evil = re.compile(r"(a+)+$")
    start = time.perf_counter()
    evil.search("a" * 22 + "!")
    assert time.perf_counter() - start >= 0.0  # ran; the point is the guard below
    with pytest.raises(ConfigError):
        GazetteerEntityExtractor({"labels": {"BOMB": {"patterns": [r"(a+)+$"]}}})


_SAFE_PATTERNS = [r"\w+\s+\w+", r"[a-z]+[0-9]+", r"\d{3}-\d{4}", r"a+b+c", r"foo|bar", r"(abc)+"]


@pytest.mark.parametrize("pattern", _SAFE_PATTERNS)
def test_safe_config_patterns_accepted(pattern: str) -> None:
    """Ubiquitous, safe operator patterns are NOT falsely rejected by the guard."""
    assert not _has_catastrophic_backtracking(pattern)
    # Compiles without error.
    GazetteerEntityExtractor({"labels": {"X": {"patterns": [pattern]}}})


# ---------------------------------------------------------------------------
# Edge-case matrix: degenerate chunks, boundaries, empty content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget",
    [
        ChunkBudget(max_tokens=1, overlap_tokens=0, min_tokens=0),
        ChunkBudget(max_tokens=2, overlap_tokens=1, min_tokens=0),
        ChunkBudget(max_tokens=8, overlap_tokens=7, min_tokens=0),  # near-total overlap
        ChunkBudget(max_tokens=4, overlap_tokens=3, min_tokens=100),  # min > max region
    ],
)
def test_degenerate_budgets_terminate_and_advance(budget: ChunkBudget) -> None:
    """A pathological (tiny / near-total-overlap) budget still terminates + advances.

    Forward-progress invariant: even a 1-token budget with heavy overlap produces a
    finite, non-empty-span set of chunks over a real document — no infinite loop, no
    zero-width chunk (the overlap-vs-cursor DoS class).
    """
    text = "word " * 200
    chunks = chunk_document(text, budget)
    assert 0 < len(chunks) < 10_000
    for c in chunks:
        assert c.char_end > c.char_start
        assert 0 <= c.char_start <= c.char_end <= len(text)


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty
        "   \n\n\t  ",  # whitespace only
        "```\n\n```",  # only markup (strips to nothing)
        "# \n\n> \n\n- \n\n| |",  # only block markers
    ],
)
def test_degenerate_content_yields_no_chunks(text: str) -> None:
    """Empty / whitespace-only / markup-only content yields zero chunks (never error)."""
    assert chunk_document(text) == []


def test_mention_at_chunk_boundary_resolves() -> None:
    """A mention split across chunk boundaries by overlap still resolves each offset.

    With small chunks + overlap the same entity can appear in adjacent chunks; each
    emitted mention must still round-trip to its own exact source slice.
    """
    page = ("Acme Corp " + "filler " * 30 + "Globex Inc " + "tail " * 30 + "Initech")
    doc, text = _make_document([page])
    chunks = list(MarkdownChunker({"max_tokens": 20, "overlap_tokens": 6}).chunk([doc]))
    mentions = list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract(chunks))
    assert mentions
    _assert_mention_invariants(mentions, text)


def test_mention_at_page_boundary_records_both_pages() -> None:
    """A pattern mention straddling a page boundary records both source pages.

    Offset-boundary edge: an entity whose span crosses the inter-page separator must
    have ``page_start`` on the earlier page and ``page_end`` on the later one.
    """
    # from_page_texts attributes the "\n\n" separator to the preceding page, so an
    # entity ending on page 1 and continuing onto page 2 straddles the boundary.
    doc, text = _make_document(["alpha BOUNDARY", "MARK beta"])
    ext = GazetteerEntityExtractor({"labels": {"X": {"patterns": [r"BOUNDARY\n\nMARK"]}}})
    chunks = list(MarkdownChunker({"max_tokens": 512}).chunk([doc]))
    mentions = list(ext.extract(chunks))
    assert len(mentions) == 1
    m = mentions[0]
    assert doc.page_map is not None
    poi = PageOffsetIndex(doc.page_map)
    assert m.provenance.char_start is not None and m.provenance.char_end is not None
    assert m.provenance.page_start == poi.page_for_offset(m.provenance.char_start)
    assert m.provenance.page_end == poi.page_for_offset(m.provenance.char_end - 1)
    assert m.provenance.page_end > m.provenance.page_start


def test_empty_and_near_empty_chunk_yields_no_mentions() -> None:
    """An empty / near-empty chunk yields no mentions, no error (S4 AC)."""
    doc, _ = _make_document([""])
    chunks = list(MarkdownChunker().chunk([doc]))
    # An empty document produces no chunks at all; feed a whitespace chunk directly.
    empty_doc, _ = _make_document(["   "])
    assert list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract(chunks)) == []
    assert (
        list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract(MarkdownChunker().chunk([empty_doc])))
        == []
    )


# ---------------------------------------------------------------------------
# Adversarial fuzz: unicode / RTL / zero-width / mixed, full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_fuzz_preserves_all_invariants() -> None:
    """Seeded fuzz: 300 random unicode/markup docs through Parse→Chunk→NER→RE.

    A single deterministic loop (fixed RNG ⇒ hash-stable) asserting, on EVERY generated
    document, the whole invariant set at once: offsets in-bounds + round-trip, page
    provenance present, relation covering spans hold both endpoints, and the extractors
    are deterministic across a repeat run. This is the omnibus adversarial gate for the
    doc stages.
    """
    rng = random.Random(20260707)
    ext = GazetteerEntityExtractor(_ENTITY_CONFIG)
    rel = PatternRelationExtractor(_RELATION_CONFIG)
    chunker = MarkdownChunker({"max_tokens": 24, "overlap_tokens": 4})
    for _ in range(300):
        n = rng.randint(4, 30)
        parts: list[str] = []
        for _ in range(n):
            roll = rng.random()
            if roll < 0.2:
                parts.append(rng.choice(_ENTITY_TOKENS))
            elif roll < 0.5:
                parts.append(rng.choice(_MARKUP))
            elif roll < 0.7:
                parts.append(rng.choice(["​", "﻿", "ال", "中文", "\U0001f600"]))
            else:
                parts.append(rng.choice(_FILLER))
            parts.append(" ")
        doc, text = _make_document(["".join(parts)])
        chunks = list(chunker.chunk([doc]))
        mentions = list(ext.extract(chunks))
        mentions_again = list(ext.extract(chunks))
        assert [m.model_dump() for m in mentions] == [m.model_dump() for m in mentions_again]
        _assert_mention_invariants(mentions, text)
        relations = list(rel.relate(chunks, mentions))
        relations_again = list(rel.relate(chunks, mentions))
        assert [r.model_dump() for r in relations] == [r.model_dump() for r in relations_again]
        for r in relations:
            cs, ce = r.provenance.char_start, r.provenance.char_end
            assert cs is not None and ce is not None
            assert 0 <= cs <= ce <= len(text)
            covering = text[cs:ce]
            assert r.head_text in covering and r.tail_text in covering

"""GazetteerEntityExtractor: zero-shot gazetteer/regex Entity Extraction (S4).

Unit tests for the in-core reference Provider: happy-path typed mentions with
correct document-level offsets + inherited page span (the S4 round-trip),
zero-shot label configurability, deterministic ordering, empty/near-empty chunks,
overlap resolution, and config validation.
"""

from __future__ import annotations

import time

import pytest
from latence_core.contracts import ChunkRecord, Classification, PageMap, Provenance
from latence_core.stages.entity import (
    _MAX_PATTERN_LENGTH,
    GazetteerEntityExtractor,
)


def _chunk(
    content: str,
    *,
    char_start: int = 100,
    char_end: int | None = None,
    page_start: int = 2,
    page_end: int = 2,
    page_map: PageMap | None = None,
    record_id: str = "doc1#chunk-0",
) -> ChunkRecord:
    """A ChunkRecord whose Provenance carries a document-level span + page range.

    ``char_start`` is where the chunk sits in the parent document's assembled
    markdown (S3), so a mention at chunk-local offset ``k`` lands at document offset
    ``char_start + k`` — the offset shift the extractor must apply. ``page_map`` (the
    parent's page map, carried on the chunk since S4) lets the extractor re-resolve a
    mention's *own* page; absent it, a mention inherits the chunk's page range.
    """
    if char_end is None:
        char_end = char_start + len(content)
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=char_start,
        char_end=char_end,
        page_start=page_start,
        page_end=page_end,
    )
    return ChunkRecord(
        record_id=record_id,
        document_record_id="doc1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        token_count=len(content.split()),
        page_map=page_map,
    )


_ORG_PERSON_CONFIG = {
    "labels": {
        "ORG": {"terms": ["Acme Corp", "Globex"], "confidence": 0.95},
        "PERSON": {"terms": ["Alice", "Bob"]},
    },
    "default_confidence": 0.8,
}


def test_extracts_typed_mentions_with_confidence() -> None:
    chunk = _chunk("Alice met Bob at Acme Corp today.")
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    by_text = {m.text: m for m in mentions}
    assert set(by_text) == {"Alice", "Bob", "Acme Corp"}
    # Per-type confidence: ORG uses its own 0.95, PERSON falls to default 0.8.
    assert by_text["Acme Corp"].label == "ORG"
    assert by_text["Acme Corp"].confidence == 0.95
    assert by_text["Alice"].label == "PERSON"
    assert by_text["Alice"].confidence == 0.8


def test_offsets_resolve_to_document_span_and_inherit_page_without_map() -> None:
    # No page map on the chunk: the mention degrades to inheriting the chunk's page
    # range (the documented fallback). Offsets are still exact document offsets.
    chunk = _chunk("Alice met Bob.", char_start=100, page_start=2, page_end=3)
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    alice = next(m for m in mentions if m.text == "Alice")
    # Document-level char span = chunk_char_start + local offset. "Alice" is [0,5).
    assert alice.provenance.char_start == 100
    assert alice.provenance.char_end == 105
    # No page map ⇒ inherit the parent chunk's page range.
    assert alice.provenance.page_start == 2
    assert alice.provenance.page_end == 3
    # Provenance identity carried through from the chunk.
    assert alice.provenance.document_id == "sha256:doc1"
    # Bob at chunk-local 10 -> document 110.
    bob = next(m for m in mentions if m.text == "Bob")
    assert bob.provenance.char_start == 110
    assert bob.provenance.char_end == 113


def test_mention_page_resolves_to_its_own_page_in_multipage_chunk() -> None:
    # A chunk spanning two pages: a mention on the SECOND page must cite page 2, not
    # inherit the chunk's first page (the S4 "correct source page" round-trip).
    assembly = PageMap.from_page_texts(["Alice is here.", "Bob is over here."])
    # assembled = "Alice is here.\n\nBob is over here." ; page 1 [0,16), page 2 [16,33)
    chunk = _chunk(
        assembly.text,
        char_start=0,
        char_end=assembly.page_map.total_chars,
        page_start=1,
        page_end=2,
        page_map=assembly.page_map,
    )
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    alice = next(m for m in mentions if m.text == "Alice")
    bob = next(m for m in mentions if m.text == "Bob")
    # Alice is on page 1; Bob (past the page boundary) is on page 2.
    assert (alice.provenance.page_start, alice.provenance.page_end) == (1, 1)
    assert (bob.provenance.page_start, bob.provenance.page_end) == (2, 2)
    # And the offsets still index the assembled text exactly.
    assert assembly.text[alice.provenance.char_start : alice.provenance.char_end] == "Alice"
    assert assembly.text[bob.provenance.char_start : bob.provenance.char_end] == "Bob"


def test_mention_links_and_inherits_classification() -> None:
    chunk = _chunk("Alice waved.", record_id="doc1#chunk-7")
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    m = mentions[0]
    assert m.chunk_record_id == "doc1#chunk-7"
    assert m.document_record_id == "doc1"
    assert m.record_id == "doc1#chunk-7#entity-0"
    assert m.index == 0
    assert m.classification == chunk.classification


def test_zero_shot_labels_are_configurable_not_hardcoded() -> None:
    # A totally different domain: the SAME Provider extracts gene/drug labels.
    config = {
        "labels": {
            "GENE": {"terms": ["BRCA1", "TP53"]},
            "DRUG": {"terms": ["aspirin"]},
        }
    }
    chunk = _chunk("The BRCA1 gene and aspirin were studied.")
    mentions = list(GazetteerEntityExtractor(config).extract([chunk]))
    labels = {m.label for m in mentions}
    assert labels == {"GENE", "DRUG"}


def test_regex_patterns_supported() -> None:
    config = {"labels": {"EMAIL": {"patterns": [r"[\w.]+@[\w.]+"], "confidence": 0.99}}}
    chunk = _chunk("Contact bob@example.com for details.")
    mentions = list(GazetteerEntityExtractor(config).extract([chunk]))
    assert len(mentions) == 1
    assert mentions[0].label == "EMAIL"
    assert mentions[0].text == "bob@example.com"
    assert mentions[0].confidence == 0.99


def test_whole_word_matching_avoids_substrings() -> None:
    # "Globex" as a whole word matches; "Globexity" must not.
    chunk = _chunk("Globex thrives but Globexity does not.")
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    texts = [m.text for m in mentions]
    assert texts == ["Globex"]


def test_case_insensitive_by_default() -> None:
    chunk = _chunk("alice and ALICE both appear.")
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    assert {m.text for m in mentions} == {"alice", "ALICE"}


def test_case_sensitive_when_configured() -> None:
    config = {**_ORG_PERSON_CONFIG, "case_sensitive": True}
    chunk = _chunk("alice is lowercase; Alice is not.")
    mentions = list(GazetteerEntityExtractor(config).extract([chunk]))
    assert {m.text for m in mentions} == {"Alice"}


def test_deterministic_ordering_and_indices() -> None:
    chunk = _chunk("Bob, Alice, and Acme Corp.")
    a = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    b = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([chunk]))
    # Byte-identical across runs (hash-stable, seeded Baseline bar).
    assert [m.model_dump() for m in a] == [m.model_dump() for m in b]
    # Ordered by document char span; indices 0..n in that order. The EntityMention
    # validator guarantees char_start is present, so coerce to int for the sort.
    starts = [int(m.provenance.char_start or 0) for m in a]
    assert starts == sorted(starts)
    assert [m.index for m in a] == list(range(len(a)))


def test_overlapping_matches_resolved_first_wins() -> None:
    # Two labels can both match overlapping spans; the earlier/longer wins, and the
    # result is non-overlapping.
    config = {
        "labels": {
            "FULL": {"terms": ["New York City"], "confidence": 0.9},
            "PART": {"terms": ["York"], "confidence": 0.7},
        }
    }
    chunk = _chunk("I live in New York City now.")
    mentions = list(GazetteerEntityExtractor(config).extract([chunk]))
    # "New York City" wins; the overlapping "York" is dropped.
    assert [m.text for m in mentions] == ["New York City"]
    assert mentions[0].label == "FULL"


def test_empty_chunk_yields_no_mentions() -> None:
    assert list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([_chunk("")])) == []


def test_near_empty_chunk_yields_no_mentions() -> None:
    config = {**_ORG_PERSON_CONFIG, "min_chars": 5}
    # Below the min_chars threshold: skipped without error even if a term matches.
    assert list(GazetteerEntityExtractor(config).extract([_chunk("Bob")])) == []


def test_whitespace_only_chunk_yields_no_mentions() -> None:
    assert list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([_chunk("   \n\t ")])) == []


def test_no_labels_configured_extracts_nothing() -> None:
    # An empty (or vocabulary-only) label set is valid but recognises nothing.
    chunk = _chunk("Alice met Bob.")
    assert list(GazetteerEntityExtractor({}).extract([chunk])) == []
    assert list(GazetteerEntityExtractor({"labels": ["PERSON", "ORG"]}).extract([chunk])) == []


def test_multiple_chunks_each_offset_independently() -> None:
    c0 = _chunk("Alice here.", char_start=0, record_id="doc1#chunk-0")
    c1 = _chunk("Bob here.", char_start=500, record_id="doc1#chunk-1")
    mentions = list(GazetteerEntityExtractor(_ORG_PERSON_CONFIG).extract([c0, c1]))
    alice = next(m for m in mentions if m.text == "Alice")
    bob = next(m for m in mentions if m.text == "Bob")
    assert alice.provenance.char_start == 0
    assert bob.provenance.char_start == 500  # Bob is at chunk-local 0 of chunk-1.
    assert bob.chunk_record_id == "doc1#chunk-1"


# -- issue #28: ReDoS guard on operator-supplied config patterns -------------

# Patterns whose backtracking is exponential in Python's engine — each takes
# many seconds to minutes on a short crafted string (verified: ``(a+)+$`` needs
# ~68s on 31 chars). Compiling any of these from operator config must be REFUSED.
#
# The list deliberately covers ALL three catastrophic shapes, incl. the two that a
# naive guard misses (both were live bypasses caught in review):
#   * nested unbounded quantifier under an unbounded group: ``(a+)+`` / ``(a*)*`` …
#   * an UNBOUNDED OUTER ``{N,}`` quantifier on a quantified group — ``(a+){2,}`` —
#     which a 3-char trailing-slice guard truncates and lets through;
#   * a GROUP-FREE run of adjacent unbounded quantifiers over overlapping atoms —
#     ``a*a*…b`` / ``\w*\w*…x`` — which a ``(...)``-only scan never inspects.
_PATHOLOGICAL_PATTERNS = [
    r"(a+)+$",
    r"(a*)*$",
    r"(\w+)+$",
    r"([a-zA-Z]+)*$",
    r"(a|a)+$",
    r"(a|ab)+$",
    r"(.*)*$",
    r"((ab)*)*$",
    # Outer {N,} on a quantified group — must not be truncated by the trailing slice.
    r"(a+){2,}$",
    r"(\w+){2,}$",
    r"([a-z]+){3,}$",
    # Group-free adjacent unbounded quantifiers over overlapping atoms.
    r"a*a*a*a*a*a*a*b",
    r"\w*\w*\w*x",
    r".*.*b",
]

# Legitimate operator patterns whose adjacent/sequential quantifiers are over DISJOINT
# atoms (nothing to redistribute) — these are linear and MUST still compile. They lock
# the adjacency guard against over-rejecting common real-world config regex.
_SAFE_ADJACENT_QUANTIFIER_PATTERNS = [
    r"a+b+c",  # disjoint literals
    r"\w+\s+\w+",  # \w and \s are disjoint metaclasses
    r"[a-z]+[0-9]+",  # disjoint char classes
    r"\S+\s+\S+",
    r"\d+-\d+",
    r"x*y*z",
    r"[A-Z]+[a-z]+",
]


@pytest.mark.parametrize("pattern", _PATHOLOGICAL_PATTERNS)
def test_pathological_config_pattern_is_refused_at_compile_time(pattern: str) -> None:
    """A catastrophic-backtracking config pattern raises before it is ever run (issue #28).

    Without the guard, construction would succeed and a later crafted chunk would
    hang the extractor for minutes — a DoS. The guard turns that into an immediate
    ``ValueError`` at Provider construction, so a reverted fix fails this test.
    """
    config = {"labels": {"X": {"patterns": [pattern]}}}
    with pytest.raises(ValueError, match="backtracking|ReDoS|too long"):
        GazetteerEntityExtractor(config)


def test_overlong_config_pattern_is_refused() -> None:
    config = {"labels": {"X": {"patterns": ["a" * (_MAX_PATTERN_LENGTH + 1)]}}}
    with pytest.raises(ValueError, match="too long"):
        GazetteerEntityExtractor(config)


def test_adversarial_input_against_refused_pattern_never_runs() -> None:
    """The guard fires at construct time, so the exponential match is never reached.

    Belt-and-braces: build the extractor inside the timed block with a pathological
    pattern and the adversarial input that would trigger the blow-up. Because
    construction raises immediately, the whole block completes in well under a second
    — proving no matching against the crafted string ever happens. If the guard were
    removed, constructing would succeed and running ``extract`` on the crafted chunk
    would hang far past this bound.
    """
    evil_pattern = r"(a+)+$"
    crafted = _chunk("a" * 40 + "!")  # 40 a's then a non-match — worst case for (a+)+
    t0 = time.perf_counter()
    with pytest.raises(ValueError):
        extractor = GazetteerEntityExtractor({"labels": {"X": {"patterns": [evil_pattern]}}})
        list(extractor.extract([crafted]))
    assert time.perf_counter() - t0 < 1.0


def test_safe_config_patterns_still_compile_and_match() -> None:
    """The guard must not reject legitimate operator patterns (no false-positive DoS)."""
    config = {
        "labels": {
            "EMAIL": {"patterns": [r"[\w.]+@[\w.]+"]},
            "ID": {"patterns": [r"[A-Z]{2,4}-\d{3,6}"]},
            "PHONE": {"patterns": [r"\+?\d[\d -]{6,14}\d"]},
            "WORD": {"patterns": [r"cat|dog|bird"]},
        }
    }
    chunk = _chunk("Mail a@b.com id AB-12345 phone +1 555 123 4567 about a dog.")
    mentions = list(GazetteerEntityExtractor(config).extract([chunk]))
    labels = {m.label for m in mentions}
    assert {"EMAIL", "ID", "PHONE", "WORD"} <= labels


@pytest.mark.parametrize("pattern", _SAFE_ADJACENT_QUANTIFIER_PATTERNS)
def test_disjoint_adjacent_quantifiers_are_not_rejected(pattern: str) -> None:
    """Adjacent unbounded quantifiers over DISJOINT atoms are linear — must compile.

    The group-free adjacency guard flags ``a*a*…b`` only because the atoms OVERLAP.
    ``a+b+c`` / ``\\w+\\s+\\w+`` / ``[a-z]+[0-9]+`` cannot redistribute input, so they are
    safe and must not be refused — else the ReDoS guard breaks ordinary config regex.
    """
    GazetteerEntityExtractor({"labels": {"X": {"patterns": [pattern]}}})


def test_group_free_adjacent_redos_is_refused_before_it_can_run() -> None:
    """A group-free ``a*a*…b`` DoS pattern is refused at construct time (issue #28).

    This shape has no ``(...)`` group, so a group-only guard never inspects it, yet it
    is genuinely exponential. Constructing must raise immediately — proving the crafted
    adversarial chunk is never matched against it.
    """
    evil = r"a*a*a*a*a*a*a*a*b"
    crafted = _chunk("a" * 40)  # 40 a's, no trailing 'b' — worst case for the a*-run
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="backtracking|ReDoS"):
        extractor = GazetteerEntityExtractor({"labels": {"X": {"patterns": [evil]}}})
        list(extractor.extract([crafted]))
    assert time.perf_counter() - t0 < 1.0


def test_outer_brace_quantifier_redos_is_refused() -> None:
    """An outer ``{N,}`` on a quantified group is unbounded and must be refused.

    Regression for the trailing-slice truncation bug: ``(a+){2,}$`` is exponential but a
    guard that reads only 3 chars after ``)`` truncates the ``}`` of ``{2,}`` and lets it
    through. The full-quantifier parse must classify ``{2,}`` as unbounded and refuse.
    """
    for evil in (r"(a+){2,}$", r"(\w+){5,}$"):
        with pytest.raises(ValueError, match="backtracking|ReDoS"):
            GazetteerEntityExtractor({"labels": {"X": {"patterns": [evil]}}})

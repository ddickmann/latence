"""Unit tests for the ported chunking engine (18-pattern strip + token budgeting).

Deterministic and seeded-free (the engine has no randomness). Covers the three
ported parts: markup stripping WITH offset preservation, token budgeting into
retrieval-sized pieces, and the offset map back into the ORIGINAL text.
"""

from __future__ import annotations

import time

import pytest
from latence_core.chunking import (
    _WORD_RE,
    ChunkBudget,
    _overlap_start,
    _token_index,
    chunk_document,
    estimate_tokens,
    strip_markup,
)

# -- markup stripping preserves offsets --------------------------------------


def test_strip_removes_markup_but_keeps_text() -> None:
    md = "# Heading\n\nSome **bold** and a [link](http://example.com) here."
    result = strip_markup(md)
    assert "#" not in result.stripped
    assert "**" not in result.stripped
    assert "](http" not in result.stripped
    assert "Heading" in result.stripped
    assert "bold" in result.stripped
    assert "link" in result.stripped


def test_strip_offset_map_points_at_original_characters() -> None:
    """Every stripped character maps back to the identical original character."""
    md = "## Title\n\n- one **two** three\n\n`code` and [x](y)."
    result = strip_markup(md)
    for i, ch in enumerate(result.stripped):
        assert md[result.mapping[i]] == ch, f"offset {i} maps wrong"
    # The sentinel end offset is one past the last original character.
    assert result.mapping[len(result.stripped)] == len(md)


def test_strip_empty_text() -> None:
    result = strip_markup("")
    assert result.stripped == ""
    assert result.mapping == [0]


def test_strip_all_eighteen_pattern_kinds_run() -> None:
    """A doc exercising every construct strips to clean prose with offsets intact."""
    md = (
        "# H1\n"
        "## H2\n\n"
        "Underlined\n===\n\n"
        "> a quote\n\n"
        "- bullet\n"
        "* star\n"
        "1. numbered\n\n"
        "---\n\n"
        "| a | b |\n| --- | --- |\n| c | d |\n\n"
        "**bold** _italic_ ~~strike~~ `code`\n\n"
        "![img](i.png) [ref][1] a [inline](u) [^fn]\n\n"
        "<div>html</div>\n"
        "```py\nprint('x')\n```\n"
    )
    result = strip_markup(md)
    for i, ch in enumerate(result.stripped):
        assert md[result.mapping[i]] == ch
    # markup sigils are gone; human text survives.
    for gone in ("**", "~~", "](", "<div>", "```"):
        assert gone not in result.stripped
    for kept in ("bold", "italic", "strike", "code", "quote", "numbered", "html"):
        assert kept in result.stripped


# -- token estimation --------------------------------------------------------


def test_estimate_tokens_deterministic_and_nonzero() -> None:
    text = "The quarterly revenue report shows strong growth."
    a = estimate_tokens(text)
    b = estimate_tokens(text)
    assert a == b > 0


def test_estimate_tokens_blank_is_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n  ") == 0


def test_estimate_tokens_long_word_costs_more() -> None:
    assert estimate_tokens("supercalifragilisticexpialidocious") > 1


# -- chunk budgeting ---------------------------------------------------------


def test_chunk_empty_and_whitespace_yield_no_chunks() -> None:
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_chunk_short_document_is_one_chunk() -> None:
    text = "A single short sentence that fits well within the budget."
    chunks = chunk_document(text, ChunkBudget(max_tokens=512))
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)
    assert chunks[0].text == text


def test_chunk_splits_long_document_under_budget() -> None:
    text = ("The quarterly report covers revenue and growth. " * 40).strip()
    budget = ChunkBudget(max_tokens=40, overlap_tokens=8, min_tokens=4)
    chunks = chunk_document(text, budget)
    assert len(chunks) > 1
    # Every chunk is at or under the token ceiling (allowing the last remnant).
    for ch in chunks:
        assert ch.token_count <= budget.max_tokens


def test_chunk_offsets_are_in_range_and_ordered() -> None:
    text = ("Sentence number one is here. Sentence two follows. " * 20).strip()
    chunks = chunk_document(text, ChunkBudget(max_tokens=30, overlap_tokens=6, min_tokens=4))
    for ch in chunks:
        assert 0 <= ch.char_start <= ch.char_end <= len(text)
    # Chunks advance through the document (start offsets are non-decreasing).
    starts = [ch.char_start for ch in chunks]
    assert starts == sorted(starts)


def test_chunk_offsets_map_back_into_original_when_no_markup() -> None:
    """With no markup, a chunk's span in the original equals its own text."""
    text = ("Alpha beta gamma delta epsilon zeta eta theta. " * 15).strip()
    chunks = chunk_document(text, ChunkBudget(max_tokens=25, overlap_tokens=0, min_tokens=2))
    for ch in chunks:
        assert text[ch.char_start : ch.char_end] == ch.text


def test_chunk_span_maps_to_original_even_after_markup_strip() -> None:
    """A chunk's offsets index the ORIGINAL markdown, not the stripped text."""
    body = "The revenue figures are strong this quarter and next. " * 8
    text = f"# Financial Report\n\n**Summary:** {body}"
    chunks = chunk_document(text, ChunkBudget(max_tokens=30, overlap_tokens=4, min_tokens=2))
    assert chunks
    for ch in chunks:
        original_slice = text[ch.char_start : ch.char_end]
        # The chunk text is the STRIPPED form; the original slice still contains
        # the human words (the markup was removed from the chunk, not the source).
        first_word = ch.text.split()[0].strip(".,")
        assert first_word in original_slice


def test_chunk_is_deterministic() -> None:
    text = ("Deterministic chunking must be byte-stable across runs. " * 30).strip()
    budget = ChunkBudget(max_tokens=35, overlap_tokens=7, min_tokens=3)
    a = chunk_document(text, budget)
    b = chunk_document(text, budget)
    assert [(c.text, c.char_start, c.char_end, c.token_count) for c in a] == [
        (c.text, c.char_start, c.char_end, c.token_count) for c in b
    ]


def test_chunk_overlap_repeats_previous_tail() -> None:
    text = ("word " * 200).strip()
    budget = ChunkBudget(max_tokens=30, overlap_tokens=10, min_tokens=2)
    chunks = chunk_document(text, budget)
    assert len(chunks) >= 2
    # With overlap, consecutive chunks share source characters (start of chunk n+1
    # is before end of chunk n).
    assert chunks[1].char_start < chunks[0].char_end


def test_chunk_no_overlap_pieces_are_disjoint() -> None:
    text = ("token " * 150).strip()
    budget = ChunkBudget(max_tokens=25, overlap_tokens=0, min_tokens=2)
    chunks = chunk_document(text, budget)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start >= prev.char_end


def test_chunk_short_tail_merges_into_previous() -> None:
    # A document sized to leave a tiny remnant; min_tokens forces the merge.
    text = ("alpha beta gamma delta. " * 10).strip() + " tail"
    budget = ChunkBudget(max_tokens=20, overlap_tokens=0, min_tokens=8)
    chunks = chunk_document(text, budget)
    # No emitted chunk is a sub-min fragment (the tail merged upward).
    if len(chunks) > 1:
        assert estimate_tokens(chunks[-1].text) >= budget.min_tokens


def test_chunk_covers_all_content_no_overlap() -> None:
    """Without overlap, the union of chunk spans covers every original offset."""
    text = ("coverage word here now. " * 20).strip()
    chunks = chunk_document(text, ChunkBudget(max_tokens=25, overlap_tokens=0, min_tokens=2))
    covered = [False] * len(text)
    for ch in chunks:
        for i in range(ch.char_start, ch.char_end):
            covered[i] = True
    # Every non-whitespace character is covered by some chunk.
    for i, c in enumerate(text):
        if not c.isspace():
            assert covered[i], f"char {i} ({c!r}) uncovered"


# -- budget validation -------------------------------------------------------


def test_budget_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ChunkBudget(max_tokens=0)


def test_budget_rejects_overlap_ge_max() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkBudget(max_tokens=10, overlap_tokens=10)


def test_budget_rejects_negative_min() -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        ChunkBudget(max_tokens=100, overlap_tokens=10, min_tokens=-1)


# -- issue #26: O(n^2) chunking-DoS regression -------------------------------


def _overlap_start_rescan(text: str, boundary: int, overlap_tokens: int) -> int:
    """The pre-fix O(n) rescan-from-0 overlap look-back, kept as a correctness oracle.

    If :func:`_overlap_start` ever diverges from this reference, the bisect-based
    rewrite has changed chunk boundaries — the test below asserts they never do.
    """
    if overlap_tokens <= 0:
        return boundary
    matches = list(_WORD_RE.finditer(text, 0, boundary))
    if not matches:
        return boundary
    budget = 0
    start = boundary
    for match in reversed(matches):
        tok = match.group(0)
        budget += 1 + ((len(tok) - 1) // 6 if tok.isalnum() and len(tok) > 6 else 0)
        start = match.start()
        if budget >= overlap_tokens:
            break
    return start


def test_overlap_start_is_byte_identical_to_the_old_rescan() -> None:
    """The bisect rewrite yields the SAME boundary as the O(n) rescan for every cut.

    Chunk boundaries always land at token boundaries (``_best_break`` breaks after a
    delimiter; ``_window_end`` returns ``match.end()``), so the equivalence is tested
    at exactly those positions — plus the document ends and 0 — which is the full set
    the real pipeline ever passes in.
    """
    text = ("The quarterly revenue report; growth, margins! And more. " * 60).strip()
    tokens = _token_index(text)
    boundaries = {0, len(text), *tokens.ends, *tokens.starts}
    for overlap in (0, 1, 5, 10, 64):
        for boundary in sorted(boundaries):
            assert _overlap_start(tokens, boundary, overlap) == _overlap_start_rescan(
                text, boundary, overlap
            ), f"diverged at boundary={boundary}, overlap={overlap}"


def test_chunking_pathological_input_stays_near_linear() -> None:
    """A crafted large plain-text doc must NOT drive chunking into O(n^2) CPU (issue #26).

    Reverting to the rescan-from-0 ``_overlap_start`` makes doubling the input ~4x the
    time; the bounded backward walk keeps it ~2x. Assert the growth ratio stays well
    under quadratic so a reverted fix fails here rather than in production.
    """
    # Short single-token "words" separated by whitespace maximise the token count per
    # char — the worst case for a per-chunk rescan. Small max_tokens ⇒ many chunks.
    unit = "aa "
    budget = ChunkBudget(max_tokens=20, overlap_tokens=8, min_tokens=2)

    def _time(n_chars: int) -> float:
        text = (unit * (n_chars // len(unit))).strip()
        t0 = time.perf_counter()
        chunk_document(text, budget)
        return time.perf_counter() - t0

    small = _time(40_000)
    large = _time(80_000)
    # Quadratic would give ~4x for a 2x input; linear ~2x. Allow generous slack for a
    # noisy CI host but stay decisively sub-quadratic (< 3x). ``small`` is floored so a
    # near-zero fast run cannot make the ratio explode on measurement jitter.
    ratio = large / max(small, 1e-3)
    assert ratio < 3.0, f"chunking scaled {ratio:.1f}x for 2x input — O(n^2) regression"


def test_chunking_large_document_completes_promptly() -> None:
    """A 400k-char in-cap document chunks in well under a second (was ~53s pre-fix)."""
    text = ("word " * 80_000).strip()
    t0 = time.perf_counter()
    chunks = chunk_document(text, ChunkBudget(max_tokens=64, overlap_tokens=16, min_tokens=4))
    elapsed = time.perf_counter() - t0
    assert chunks
    assert elapsed < 5.0, f"400k-char chunking took {elapsed:.1f}s — O(n^2) DoS not fixed"

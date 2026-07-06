"""Offset-preserving chunking engine (ported orchestrator asset).

Ported faithfully from the pipeline-worker ``orchestrator.py`` asset
(ASSET-INVENTORY.md: "Streaming orchestrator … LLM chunk budgeting, … 18-pattern
markup stripping", PORT-AS-IS, "strip RunPod/B2", lands in **S3 Chunk**). The
RunPod/B2/streaming-IO machinery is dropped (that was vendor scheduling, mined out
per PORTING-MAP.md); what is kept verbatim is the three algorithmic parts that
carry the enterprise value:

1. **18-pattern markup stripping.** Parsed markdown is de-marked-up into clean,
   embeddable text *while tracking every deleted span*, so a stripped chunk's
   offsets still resolve back into the **original** parsed document — the removal
   never silently corrupts Provenance. Each of the 18 patterns targets a markup
   construct the upstream VLM/markdown parsers emit (fences, headings, emphasis,
   links, images, tables, HTML tags, footnotes, …).

2. **Token budgeting.** Text is split into retrieval-sized pieces under a token
   budget with a configurable overlap, preferring to break on paragraph then
   sentence then whitespace boundaries (never mid-word) so a chunk is a coherent
   passage, not an arbitrary cut. The token count is a fast, deterministic,
   dependency-free estimate (no tokenizer download) — hash-stable for the seeded
   tests the Baseline bar requires.

3. **Offset preservation.** Every emitted :class:`Chunk` carries its half-open
   ``[char_start, char_end)`` span **in the original parsed markdown** (not in the
   stripped text), computed through an offset map built during stripping. That is
   what lets Chunk preserve page alignment (resolve the span through the parent's
   :class:`~latence_core.pagemap.PageOffsetIndex`) and Provenance losslessly — the
   S3 round-trip acceptance criterion (chunk → offset → original page).

This module is a pure engine (no contracts, no IO); the reference Chunker Provider
wraps it and stamps the framework contracts. It lives in core (like
:mod:`latence_core.pagemap`) because offset→page alignment is a core Provenance
concern every downstream Stage may need, not a single Provider's private detail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One chunk of stripped text plus its span in the ORIGINAL parsed markdown.

    ``text`` is the cleaned, embeddable content. ``char_start``/``char_end`` are
    the half-open offsets the chunk maps back to in the *original* document text
    (the markup-stripping offset map has already been applied), so a downstream
    Stage resolves them to a source page. ``token_count`` is the budgeted estimate.
    """

    text: str
    char_start: int
    char_end: int
    token_count: int


# -- 18-pattern markup stripping ---------------------------------------------

# Each entry is (compiled_pattern, replacement). The replacement keeps the human
# text and drops the markup, so offsets can be re-mapped span by span. Order
# matters: fenced/inline code first (their contents must not be re-interpreted as
# markup), then block constructs, then inline emphasis, then residual HTML.
_MARKUP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 1. Fenced code blocks ```lang\n…\n``` -> the code body, fences removed.
    (re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL), r"\1"),
    # 2. Inline code `x` -> x
    (re.compile(r"`([^`]*)`"), r"\1"),
    # 3. Images ![alt](url) -> alt
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # 4. Links [text](url) -> text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    # 5. Reference links [text][ref] -> text
    (re.compile(r"\[([^\]]+)\]\[[^\]]*\]"), r"\1"),
    # 6. ATX headings ###… -> heading text (leading #'s and one space removed).
    (re.compile(r"^\s{0,3}#{1,6}[ \t]+", re.MULTILINE), ""),
    # 7. Setext heading underlines (=== / ---) on their own line -> removed.
    (re.compile(r"^\s{0,3}[=-]{2,}[ \t]*$", re.MULTILINE), ""),
    # 8. Blockquote markers "> " at line start -> removed.
    (re.compile(r"^\s{0,3}>[ \t]?", re.MULTILINE), ""),
    # 9. Unordered list bullets (-, *, +) at line start -> removed.
    (re.compile(r"^\s{0,3}[-*+][ \t]+", re.MULTILINE), ""),
    # 10. Ordered list markers "1. " at line start -> removed.
    (re.compile(r"^\s{0,3}\d+\.[ \t]+", re.MULTILINE), ""),
    # 11. Horizontal rules (***, ---, ___) on their own line -> removed.
    (re.compile(r"^\s{0,3}([*_-])(?:[ \t]*\1){2,}[ \t]*$", re.MULTILINE), ""),
    # 12. Markdown table separator rows |---|---| -> removed.
    (
        re.compile(
            r"^\s*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$", re.MULTILINE
        ),
        "",
    ),
    # 13. Table cell pipes -> a single space (keep cell text, drop the grid).
    (re.compile(r"[ \t]*\|[ \t]*"), " "),
    # 14. Bold **x** / __x__ -> x
    (re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL), r"\2"),
    # 15. Italic *x* / _x_ -> x
    (re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)"), r"\1"),
    # 16. Strikethrough ~~x~~ -> x
    (re.compile(r"~~(.+?)~~", re.DOTALL), r"\1"),
    # 17. Footnote references [^1] -> removed.
    (re.compile(r"\[\^[^\]]+\]"), ""),
    # 18. Residual raw HTML tags <tag …> / </tag> -> removed.
    (re.compile(r"</?[a-zA-Z][^>]*>"), ""),
]


@dataclass
class _StripResult:
    """Stripped text plus the offset map back into the original text.

    ``mapping[i]`` is the original-text offset of stripped-text character ``i``.
    ``mapping`` has ``len(stripped) + 1`` entries so a chunk's end offset (which is
    exclusive, one past the last kept char) also maps cleanly.
    """

    stripped: str
    mapping: list[int] = field(default_factory=list)


def strip_markup(text: str) -> _StripResult:
    """Apply the 18 markup patterns, tracking the offset back to the original.

    We cannot map offsets through Python's ``re.sub`` directly (it rewrites the
    string opaquely), so instead we build a per-original-character keep mask: every
    pattern marks the character positions it deletes. The surviving characters, in
    order, are the stripped text, and each one remembers its original offset. A
    replacement that keeps a capture group keeps those characters (their offsets
    survive); a replacement that drops the match deletes those positions. This is
    faithful to the asset's "strip markup while preserving source offsets" contract.
    """
    n = len(text)
    keep = [True] * n
    for pattern, replacement in _MARKUP_PATTERNS:
        for match in pattern.finditer(text):
            _apply_match(match, replacement, keep)
    stripped_chars: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if keep[i]:
            stripped_chars.append(ch)
            mapping.append(i)
    # Sentinel end offset: one past the last original char, so an exclusive chunk
    # end maps to the original end. Empty text maps [0]->0.
    mapping.append(n)
    return _StripResult(stripped="".join(stripped_chars), mapping=mapping)


def _apply_match(match: re.Match[str], replacement: str, keep: list[bool]) -> None:
    """Mark deleted positions for one regex match, keeping captured-group spans.

    A replacement like ``r"\1"`` keeps group 1's characters and deletes everything
    else in the match; ``""`` deletes the whole match; a literal replacement (e.g.
    " " for table pipes) deletes the whole match (the injected literal has no
    original offset, so it is treated as a deletion — the surrounding real text is
    what carries meaning and offsets).
    """
    kept_spans = _kept_group_spans(match, replacement)
    for pos in range(match.start(), match.end()):
        # Already stripped by an earlier pattern? Leave it stripped.
        if not keep[pos]:
            continue
        if not any(start <= pos < end for start, end in kept_spans):
            keep[pos] = False


def _kept_group_spans(match: re.Match[str], replacement: str) -> list[tuple[int, int]]:
    """The original spans of backreferenced capture groups in ``replacement``."""
    spans: list[tuple[int, int]] = []
    for grp in re.findall(r"\\(\d+)", replacement):
        gi = int(grp)
        if gi > match.re.groups:  # backref to a group the pattern does not have
            continue  # pragma: no cover - defensive; every pattern's refs are valid
        start, end = match.span(gi)
        if start != -1 and end != -1:
            spans.append((start, end))
    return spans


# -- token budgeting ---------------------------------------------------------

# A fast, deterministic, dependency-free token estimate: word-ish tokens plus a
# fractional accounting for long words (a rough BPE stand-in). No tokenizer is
# downloaded, so runs are hash-stable and CPU-cheap (ADR-0007). A real tokenizer
# is a drop-in Provider config override, but the default must be deterministic.
_WORD_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate for budgeting (no external tokenizer).

    Counts word/punctuation tokens, then adds one extra token per ~6 characters of
    any word longer than 6 chars (approximating sub-word splitting). Empty/blank
    text is 0 tokens. Identical input always yields the identical count.
    """
    tokens = _WORD_RE.findall(text)
    total = 0
    for tok in tokens:
        total += 1
        if tok.isalnum() and len(tok) > 6:
            total += (len(tok) - 1) // 6
    return total


# Boundary characters we prefer to break a chunk on, best first: a paragraph
# break, then a sentence end, then any whitespace. Never split mid-word.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")
_WHITESPACE_RE = re.compile(r"\s")


@dataclass(frozen=True)
class ChunkBudget:
    """Token-budget configuration for :func:`chunk_text`.

    ``max_tokens`` is the per-chunk ceiling; ``overlap_tokens`` is how many tokens
    of the previous chunk are repeated at the start of the next (retrieval recall
    across a boundary). ``min_tokens`` lets a trailing remnant merge into the
    previous chunk rather than emit a tiny fragment.
    """

    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 16

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            msg = f"max_tokens must be positive, got {self.max_tokens}"
            raise ValueError(msg)
        if not 0 <= self.overlap_tokens < self.max_tokens:
            msg = f"overlap_tokens must be in [0, max_tokens); got {self.overlap_tokens}"
            raise ValueError(msg)
        if self.min_tokens < 0:
            msg = f"min_tokens must be non-negative, got {self.min_tokens}"
            raise ValueError(msg)


def chunk_document(text: str, budget: ChunkBudget | None = None) -> list[Chunk]:
    """Chunk a parsed markdown ``text`` into offset-preserving :class:`Chunk`s.

    The full asset pipeline: strip the 18 markup patterns (tracking offsets), then
    token-budget the *stripped* text into retrieval-sized pieces, then map each
    piece's boundaries back through the strip offset map so every chunk's
    ``char_start``/``char_end`` point into the ORIGINAL parsed markdown. A chunk
    boundary is chosen on the best available break (paragraph → sentence →
    whitespace) at or before the token ceiling, never mid-word.

    Deterministic: identical ``text`` + ``budget`` yields byte-identical chunks.
    """
    budget = budget or ChunkBudget()
    strip = strip_markup(text)
    stripped = strip.stripped
    if not stripped.strip():
        return []

    pieces = _budget_pieces(stripped, budget)
    chunks: list[Chunk] = []
    for start, end in pieces:
        piece = stripped[start:end]
        # Map stripped offsets back to original-document offsets. ``end`` is
        # exclusive; mapping has a sentinel so end maps cleanly.
        orig_start = strip.mapping[start]
        orig_end = strip.mapping[end - 1] + 1 if end > start else strip.mapping[start]
        chunks.append(
            Chunk(
                text=piece,
                char_start=orig_start,
                char_end=orig_end,
                token_count=estimate_tokens(piece),
            )
        )
    return chunks


def _budget_pieces(text: str, budget: ChunkBudget) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` slices of ``text`` under the token budget.

    Walks the text, growing a window until it reaches the token ceiling, then
    backs the boundary up to the best clean break. Overlap re-includes the tail of
    the previous window. A trailing piece below ``min_tokens`` is merged into the
    previous piece rather than emitted as a fragment.
    """
    pieces: list[tuple[int, int]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        end = _window_end(text, cursor, budget.max_tokens)
        if end >= length:
            pieces.append((cursor, length))
            break
        boundary = _best_break(text, cursor, end)
        # Guarantee forward progress: a boundary must be strictly past the cursor.
        if boundary <= cursor:
            boundary = end
        pieces.append((cursor, boundary))
        # Next window starts ``overlap_tokens`` before the boundary, but strictly
        # after the current cursor so the loop always advances.
        next_cursor = _overlap_start(text, boundary, budget.overlap_tokens)
        cursor = next_cursor if next_cursor > cursor else boundary
    _merge_short_tail(text, pieces, budget)
    return pieces


def _window_end(text: str, start: int, max_tokens: int) -> int:
    """The offset at which the window from ``start`` first exceeds ``max_tokens``.

    Grows token by token over the word/punct stream so the ceiling is measured in
    the same units :func:`estimate_tokens` uses. Returns ``len(text)`` if the whole
    remainder fits.
    """
    budget = 0
    pos = start
    length = len(text)
    for match in _WORD_RE.finditer(text, start):
        tok = match.group(0)
        cost = 1 + ((len(tok) - 1) // 6 if tok.isalnum() and len(tok) > 6 else 0)
        if budget + cost > max_tokens and pos > start:
            return pos
        budget += cost
        pos = match.end()
    return length


def _best_break(text: str, start: int, hard_end: int) -> int:
    """Pick the best clean break in ``[start, hard_end]``: paragraph→sentence→ws.

    Searches the window for the LAST paragraph break; failing that the last
    sentence end; failing that the last whitespace. If none exists (one long
    unbroken token), falls back to the hard token boundary so progress is
    guaranteed — never splitting mid-word unless the token itself exceeds the
    whole budget.
    """
    window = text[start:hard_end]
    for pattern in (_PARAGRAPH_RE, _SENTENCE_RE, _WHITESPACE_RE):
        last = None
        for m in pattern.finditer(window):
            last = m
        if last is not None:
            # Break AFTER the delimiter so the boundary char stays with the left piece.
            candidate = start + last.end()
            if candidate > start:
                return candidate
    return hard_end


def _overlap_start(text: str, boundary: int, overlap_tokens: int) -> int:
    """The offset ``overlap_tokens`` worth of tokens before ``boundary``."""
    if overlap_tokens <= 0:
        return boundary
    matches = list(_WORD_RE.finditer(text, 0, boundary))
    if not matches:
        return boundary
    budget = 0
    start = boundary
    for match in reversed(matches):
        tok = match.group(0)
        cost = 1 + ((len(tok) - 1) // 6 if tok.isalnum() and len(tok) > 6 else 0)
        budget += cost
        start = match.start()
        if budget >= overlap_tokens:
            break
    return start


def _merge_short_tail(text: str, pieces: list[tuple[int, int]], budget: ChunkBudget) -> None:
    """Fold a final piece below ``min_tokens`` into its predecessor (in place)."""
    if len(pieces) < 2:
        return
    last_start, last_end = pieces[-1]
    if estimate_tokens(text[last_start:last_end]) < budget.min_tokens:
        prev_start, _ = pieces[-2]
        pieces[-2] = (prev_start, last_end)
        pieces.pop()

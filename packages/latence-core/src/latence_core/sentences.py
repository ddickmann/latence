# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Abbreviation-safe sentence segmentation for Hyperedge building (ADR-0057, hypergraph-m1).

Hyperedge segmentation (spec 1.2) needs sentence boundaries over chunk text. The chunker's
own boundary regex (``[.!?]\\s``) is deliberately crude — it only *prefers* a break near a
budget ceiling, so a false boundary costs nothing there. For Hyperedges a false boundary
splits a fact in half, so this module provides a proper splitter: still **zero-dep,
deterministic, CPU-only** (ADR-0007/0016 — no model, no download), but safe against the
boundary traps that actually occur in DE/EN enterprise text:

* abbreviations (``z. B.``, ``Dr.``, ``e.g.``, ``Nr.``, ``etc.``) — never a boundary;
* initials (``J. Smith``) and single-letter tokens — never a boundary;
* decimals (``3.14``) and German ordinal dates (``am 3. Mai``) — never a boundary;
* ellipsis — a boundary only when followed by a sentence-initial character;
* hard line breaks (markdown headings, list items) — always a boundary.

The seam is :class:`SentenceSplitter` (a Protocol): this rule-based reference ships in core
and runs day one; a learned splitter (pysbd, a spaCy senter) is a Provider-package adapter
behind the same seam, opt-in like every model dependency (ADR-0012).

Spans are half-open ``(start, end)`` offsets **into the input text** (the caller's
coordinate system — for Hyperedge building that is the chunk's ``content``), trimmed of
surrounding whitespace, in strictly increasing order, and never overlapping. The same input
always yields the same spans.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

__all__ = ["RuleSentenceSplitter", "SentenceSplitter"]


@runtime_checkable
class SentenceSplitter(Protocol):
    """Splits text into sentence spans (half-open offsets into the input)."""

    def split(self, text: str) -> list[tuple[int, int]]:
        """Return trimmed, non-overlapping, strictly increasing sentence spans."""
        ...


# Abbreviations that end with a period and must never close a sentence. Lower-cased for
# lookup; matched against the token immediately preceding the candidate period. The set is
# deliberately small and high-precision — a missed abbreviation merely over-splits one
# sentence, while a wrong entry would glue real sentences together corpus-wide.
_ABBREVIATIONS = frozenset(
    {
        # English
        "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "dept", "est", "fig", "eq",
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "no", "inc", "ltd", "corp",
        "co", "u.s", "u.k", "p", "pp", "vol", "ed", "eds", "rev",
        # German
        "z.b", "d.h", "u.a", "o.ä", "u.ä", "u.u", "z.t", "s.o", "s.u", "i.d.r",
        "ca", "bzw", "bspw", "evtl", "ggf", "inkl", "exkl", "zzgl", "nr", "abs",
        "art", "str", "tel", "usw", "vgl", "sog", "hrsg", "dipl", "ing", "mio",
        "mrd", "tsd", "abt", "anl", "aufl", "bd", "beil", "betr", "einschl", "erg",
        "geb", "gegr", "jhd", "jh", "kap", "lfd", "max", "min", "mtl", "pos",
        "stk", "tgl", "urspr", "zit", "zzt",
    }
)

# Sentence-terminal punctuation followed by whitespace (or end): the CANDIDATE boundaries.
_CANDIDATE_RE = re.compile(r"[.!?…]+(?=\s|$)")
# A hard break: any newline is a boundary (markdown headings / list items are sentences).
_HARD_BREAK_RE = re.compile(r"\n")
# The word immediately before a period (letters may themselves contain periods: "z.B").
_TAIL_WORD_RE = re.compile(r"([\w][\w.]*)$", re.UNICODE)


class RuleSentenceSplitter:
    """The zero-dep reference :class:`SentenceSplitter` — abbreviation-safe DE/EN rules.

    Deterministic and dependency-free (ADR-0007): safe as a day-one default. A model-based
    splitter is a Provider-package adapter behind the same seam, never an import here.
    """

    def split(self, text: str) -> list[tuple[int, int]]:
        """Split ``text`` into trimmed sentence spans; raise :class:`TypeError` if not str."""
        if not isinstance(text, str):
            msg = f"SentenceSplitter.split expects str, got {type(text).__name__}."
            raise TypeError(msg)
        if not text.strip():
            return []

        boundaries = sorted(
            {m.end() for m in _HARD_BREAK_RE.finditer(text)}
            | {m.end() for m in _CANDIDATE_RE.finditer(text) if self._is_boundary(text, m)}
        )

        spans: list[tuple[int, int]] = []
        start = 0
        for cut in [*boundaries, len(text)]:
            trimmed = self._trim(text, start, cut)
            if trimmed is not None:
                spans.append(trimmed)
            start = cut
        return spans

    def _is_boundary(self, text: str, match: re.Match[str]) -> bool:
        """True when the terminal punctuation at ``match`` really closes a sentence."""
        punct = match.group()
        before = text[: match.start()]

        # Ellipsis (or any multi-dot run): a boundary only before a sentence-initial char.
        if punct not in {".", "!", "?"}:
            return self._next_starts_sentence(text, match.end())

        if punct in {"!", "?"}:
            return True

        tail = _TAIL_WORD_RE.search(before)
        if tail is None:
            return True
        word = tail.group(1)
        bare = word.rstrip(".").lower()

        # Known abbreviation ("z.B." / "Dr." / "e.g.") — the period is part of the token.
        if bare in _ABBREVIATIONS:
            return False
        # A single letter: an initial ("J. Smith") or an enumeration marker ("a.").
        if len(bare) == 1 and bare.isalpha():
            return False
        # A number: a decimal ("3.14" — no space after, so not a candidate anyway), an
        # enumeration ("1." at line start), or a German ordinal date ("am 3. Mai"). A short
        # number closes a sentence only when what FOLLOWS does not continue one (next char
        # lowercase means continuation is impossible for a real boundary in DE/EN).
        if bare.isdigit() and len(bare) <= 2:
            return not self._continues_after_ordinal(text, match.end())
        return True

    @staticmethod
    def _next_starts_sentence(text: str, pos: int) -> bool:
        """True when the first non-space char at/after ``pos`` could open a sentence."""
        for ch in text[pos:]:
            if ch.isspace():
                continue
            return ch.isupper() or ch.isdigit() or ch in "\"'«„([#-–—•"
        return True  # end of text closes the sentence

    @staticmethod
    def _continues_after_ordinal(text: str, pos: int) -> bool:
        """True when the text after a ``<short number>.`` reads as the SAME sentence.

        ``am 3. Mai`` / ``der 2. Versuch`` continue; ``... Ende 2019. Der Vorstand``
        breaks. Continuation = the next word is capitalised-but-known-ordinal-context
        (month names) or lowercase (mid-sentence in both languages).
        """
        rest = text[pos:].lstrip()
        if not rest:
            return False
        word = re.match(r"[\wÄÖÜäöüß]+", rest)
        if word is None:
            return False
        token = word.group()
        if token[0].islower():
            return True
        months = {
            "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
            "September", "Oktober", "November", "Dezember", "Quartal", "Halbjahr",
            "January", "February", "March", "May", "June", "July", "October", "December",
        }
        return token in months

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
        """Trim ``text[start:end]`` to its non-whitespace core; None when blank."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return (start, end) if end > start else None

"""Naive stop-word language auto-detection — the framework's shared heuristic.

Source (S1) and Parse (S2) both need a language guess for Classification, and the
project rule is to reuse a core seam rather than fork it. So the heuristic lives
here once: a deliberately naive stop-word signature match, honest about being a
heuristic rather than pretending to be a model. Deterministic (no randomness, no
hash-order dependence): ties break by the fixed declaration order of the hint
table, so identical text always yields the identical tag.

A real language-ID model is a Provider concern (a future EmbeddingGemma/fastText
Provider); this is the zero-dependency default that keeps ``latence-core`` lean
(ADR-0016) and the demo model-free.
"""

from __future__ import annotations

# Tiny stop-word signatures. Insertion order is the deterministic tie-break, so a
# document that scores equally for two languages resolves to whichever appears
# first here — never a hash-seed-dependent coin flip.
_LANGUAGE_HINTS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "and", "of", "to", "is", "in", "that", "it"}),
    "de": frozenset({"der", "die", "und", "das", "ist", "nicht", "mit", "ein"}),
    "fr": frozenset({"le", "la", "et", "les", "des", "un", "une", "est"}),
    "es": frozenset({"el", "la", "y", "los", "las", "un", "una", "es"}),
}

# The tag for "no signal" — an empty document, or text with no recognised
# stop-words. BCP-47's registered subtag for an undetermined language.
UNKNOWN_LANGUAGE = "und"


def detect_language(text: str) -> str:
    """Best-effort BCP-47-ish language tag for ``text``; ``'und'`` if no signal.

    Scores ``text`` against each language's stop-word signature and returns the
    best match. Ties break by the fixed hint-table order for determinism. Returns
    :data:`UNKNOWN_LANGUAGE` when the text has no alphabetic tokens or matches no
    signature at all.
    """
    tokens = {t for t in text.lower().split() if t.isalpha()}
    if not tokens:
        return UNKNOWN_LANGUAGE
    best_lang = UNKNOWN_LANGUAGE
    best_score = 0
    for lang, hints in _LANGUAGE_HINTS.items():
        score = len(tokens & hints)
        if score > best_score:
            best_score, best_lang = score, lang
    return best_lang

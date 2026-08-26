# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The redaction-survival token space — ONE implementation of the ADR-0053 word-boundary rule.

Two inspectable sinks re-emit entity surfaces alongside a corpus that Redaction has already
masked: the Export Stage (:mod:`latence_core.stages.export`, which decides whether a
ContextHeader entity or a rendered neighbor triple may be exported) and the SPLADE composition
Provider (:mod:`latence_splade`, which decides whether an entity term may be injected into the
sparse vector). Both answer the same question — *did this surface survive redaction?* — and both
must answer it identically, or a name masked in the corpus re-leaks through the other sink.

The rule is **word-boundary survival, never substring**. A surface survives only if its word
tokens occur as a contiguous run in the redacted text: ``"ann"`` does NOT survive inside
``"annual"``, and ``"John Smith"`` survives only as an adjacent ``["john", "smith"]`` run. An
empty surface (punctuation only, no word tokens) never survives — REDACTION-BY-DEFAULT.

Until now that rule was two hand-synchronised copies, and the Export docstring said so out loud:
its check "mirrors the SPLADE provider's redaction gate". Mirroring is what two copies do right up
until the day one of them is edited — and this particular drift would be a *privacy* regression
that both suites, each testing its own copy, would still call green. So the rule lives here, once,
and the two sinks import it. Sharing it is what makes "the two sinks apply the identical rule" a
fact about the code rather than a promise in a docstring.

Dep-light (ADR-0016): stdlib :mod:`re` and nothing else.
"""

from __future__ import annotations

import re

__all__ = ["occurs_as_phrase", "word_tokens"]

#: The token space: lower-cased ``\w+`` runs, Unicode-aware. Chosen to mirror the SPLADE model's
#: lower-cased subword vocabulary, and it is what makes the check word-boundary-aware rather than
#: a substring scan — the distinction that stops a masked name re-leaking because its letters
#: happen to sit inside a surviving word.
_WORD = re.compile(r"\w+", re.UNICODE)


def word_tokens(text: str) -> list[str]:
    """Lower-cased word tokens of ``text`` — the redaction-survival token space (ADR-0053)."""
    return _WORD.findall(text.casefold())


def occurs_as_phrase(needle: list[str], haystack: list[str]) -> bool:
    """True iff ``needle`` occurs as a contiguous run of word tokens in ``haystack``.

    Word-boundary survival, not substring: a single-token surface must match a whole surviving
    token (``["ann"]`` does NOT survive inside ``["annual"]``), and a multi-word surface must
    survive as an adjacent run. An empty ``needle`` (a punctuation-only surface) never survives —
    there is nothing that could have been redacted, and the caller drops it by default.
    """
    n = len(needle)
    if n == 0:
        return False
    return any(haystack[start : start + n] == needle for start in range(len(haystack) - n + 1))

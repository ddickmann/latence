"""Blocking — the shared blocking-key primitive (tokens, keys, transitive closure).

*Blocking* is the concept two subsystems previously shared only through a private
reach-through: a surface string is reduced to its normalised word tokens (its **blocking
keys**), and two items that share a key land in the same *blocking neighborhood* — the
candidate set inside which a resolver is allowed to compare them. The disambiguation
cascade (:mod:`latence_core.disambiguation`) uses the tokens for its overlap/acronym/
substring rungs; the affected-set incremental machinery
(:mod:`latence_core.state.incremental`, :mod:`latence_core.stages.delta`) uses the
**transitive closure** over shared keys to bound what a delta must re-resolve (ADR-0018's
"affected blocks plus immediate neighbors").

This module is the public home for both: :func:`blocking_tokens` /
:func:`blocking_keys` (the tokenizer previously private to ``disambiguation._tokens``)
and :func:`closure` (the neighborhood closure previously embedded in
``compute_affected_set``, which callers could only reach by faking mention records).
Dependency-free and deterministic (pure Unicode table operations + set algebra over
caller-supplied ids), so identical input ⇒ identical output — the Baseline bar.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Word runs (letters/digits, underscore excluded), Unicode-aware so non-Latin scripts
# tokenise too.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def blocking_tokens(text: str) -> list[str]:
    """NFKC-folded, casefolded word tokens — the blocking-key unit.

    Folds through the same NFKC + casefold pipeline as
    :func:`latence_core.disambiguation.normalize_text` so token-level consumers (the
    cascade's acronym/substring/embedding rungs, KB candidate generation, the affected-set
    closure) see compatibility-confusable surfaces identically to the exact strategy —
    otherwise a fullwidth-spelled name would tokenise differently from its ASCII twin and
    evade both the acronym and the overlap rungs. ``_WORD_RE`` matches Unicode word runs
    (letters/digits, underscore excluded), so non-Latin scripts tokenise too. Deterministic.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WORD_RE.findall(folded)


def blocking_keys(text: str) -> set[str]:
    """A surface's blocking keys: its distinct normalised word tokens (shared key ⇒ same block)."""
    return set(blocking_tokens(text))


def closure(items: Iterable[tuple[str, str]], seeds: Iterable[str]) -> set[str]:
    """The transitive closure of ``seeds`` over shared blocking keys — the affected ids.

    ``items`` are ``(item_id, blocking_text)`` pairs — an id plus the surface text whose
    tokens are that item's blocking keys. No mention shape is required: a caller closing
    over records, mentions, or any other identity supplies exactly the two things the
    closure actually reads. ``seeds`` are the item ids the change directly touched.

    The closure grows breadth-first: a member pulls in every item sharing any of its keys,
    and each pulled-in item is itself a frontier, so a bridge two blocks away is caught
    (ADR-0018's "affected blocks plus immediate neighbors"). A seed id absent from
    ``items`` (or with keyless text) closes over nothing beyond itself but stays a member.
    Deterministic: pure set algebra over the supplied ids — no ordering sensitivity.
    """
    keys_by_id: dict[str, set[str]] = {}
    ids_by_key: dict[str, set[str]] = {}
    for item_id, text in items:
        keys = blocking_keys(text)
        keys_by_id[item_id] = keys
        for key in keys:
            ids_by_key.setdefault(key, set()).add(item_id)

    member_ids = set(seeds)
    seen_keys: set[str] = set()
    frontier = set(member_ids)
    while frontier:
        next_frontier: set[str] = set()
        for item_id in frontier:
            for key in keys_by_id.get(item_id, set()):
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                for neighbor in ids_by_key.get(key, set()):
                    if neighbor not in member_ids:
                        member_ids.add(neighbor)
                        next_frontier.add(neighbor)
        frontier = next_frontier
    return member_ids

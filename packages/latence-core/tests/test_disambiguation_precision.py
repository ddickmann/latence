"""Disambiguation PRECISION on a small labeled entity-resolution set (issue #75).

The Hardening-bar quality metric for the Disambiguation Stage: pair-wise merge precision /
recall over a hand-labeled gold set, extending the audited-merge discipline with a *number*
the Stage must beat. The gold labels group each mention by its true entity; the resolver's
clusters induce a predicted grouping; we score the two as a pairwise-linking problem:

* **merge precision** — of the mention pairs the resolver put in the SAME cluster, the
  fraction that are truly the same entity (a low value means over-merging);
* **merge recall** — of the mention pairs that are truly the same entity, the fraction the
  resolver put in the same cluster (a low value means under-merging / over-splitting).

The gold set is intentionally adversarial: it contains a unicode-confusable duplicate (must
merge, tests the NFKC fold), an acronym pair across incompatible types (must NOT merge, tests
the #35 guard), and two genuinely-distinct same-prefix names (must NOT merge). The thresholds
are the recorded Hardening baseline; a regression that starts over-merging (precision drop) or
over-splitting (recall drop) trips the assertion. Deterministic — no randomness, seed-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from latence_core.disambiguation import EntityResolver, MentionInput


@dataclass(frozen=True)
class _Labeled:
    """One labeled mention: its surface + type + the TRUE entity id it belongs to."""

    mention_id: str
    document_id: str
    text: str
    label: str
    gold_entity: str


# A small, hand-labeled entity-resolution gold set. Each ``gold_entity`` is the ground-truth
# canonical entity; mentions sharing it must land in one predicted cluster, and mentions with
# different gold entities must NOT.
_GOLD: list[_Labeled] = [
    # -- Acme Corporation: an ASCII spelling, a fullwidth-confusable spelling, and the acronym
    #    surface, all the same real ORG. The confusable must fold (NFKC); the exact/alias path
    #    covers the rest.
    _Labeled("m1", "d1", "Acme", "ORG", "acme"),
    _Labeled("m2", "d2", "acme", "ORG", "acme"),
    _Labeled("m3", "d3", "ＡＣＭＥ", "ORG", "acme"),  # fullwidth confusable — must merge
    # -- Globex Corporation vs Globex Holdings: same-prefix but genuinely DIFFERENT entities;
    #    must NOT be merged by a greedy substring/overlap rung.
    _Labeled("m4", "d1", "Globex Corporation", "ORG", "globex-corp"),
    _Labeled("m5", "d2", "Globex Holdings", "ORG", "globex-holdings"),
    # -- "IT" (a pronoun) vs "Information Technology" (a field): an acronym match across
    #    incompatible types that must NOT merge (the #35 precision guard).
    _Labeled("m6", "d4", "Information Technology", "FIELD", "info-tech"),
    _Labeled("m7", "d4", "IT", "PRONOUN", "it-pronoun"),
    # -- Two mentions of one person, exact surface, different docs — must merge.
    _Labeled("m8", "d5", "Alice Smith", "PERSON", "alice"),
    _Labeled("m9", "d6", "Alice Smith", "PERSON", "alice"),
]


def _gold_pairs(gold: list[_Labeled]) -> set[frozenset[str]]:
    """The set of mention-id pairs that are truly the same entity (gold same-cluster pairs)."""
    return {
        frozenset((a.mention_id, b.mention_id))
        for a, b in combinations(gold, 2)
        if a.gold_entity == b.gold_entity
    }


def _predicted_pairs(clusters: list[list[str]]) -> set[frozenset[str]]:
    """The set of mention-id pairs the resolver placed in the same predicted cluster."""
    pairs: set[frozenset[str]] = set()
    for members in clusters:
        for a, b in combinations(sorted(members), 2):
            pairs.add(frozenset((a, b)))
    return pairs


def _resolve_gold() -> list[list[str]]:
    """Run the resolver over the gold mentions; return predicted clusters as member-id lists."""
    resolver = EntityResolver(enable_embedding=False)  # no fuzzy embedding rung — precision-first
    resolver.add_all(
        [
            MentionInput(
                mention_id=g.mention_id,
                document_id=g.document_id,
                text=g.text,
                label=g.label,
                confidence=0.9,
            )
            for g in _GOLD
        ]
    )
    clusters, _ = resolver.resolve()
    return [c.member_mention_ids for c in clusters]


def _precision_recall(
    predicted: set[frozenset[str]], gold: set[frozenset[str]]
) -> tuple[float, float]:
    # No merges proposed ⇒ no false merges (precision is vacuously 1.0).
    precision = 1.0 if not predicted else len(predicted & gold) / len(predicted)
    recall = len(predicted & gold) / len(gold) if gold else 1.0
    return precision, recall


def test_merge_precision_recall_baseline() -> None:
    """Pairwise merge precision/recall on the labeled set meets the recorded Hardening baseline.

    With the NFKC confusable fold + the #35 type guard, the resolver should achieve PERFECT
    precision (no wrong merge: Globex-Corp/Globex-Holdings stay apart, IT/Information-Technology
    stay apart) and PERFECT recall (Acme's three surfaces incl. the confusable all merge; Alice
    Smith merges). The baseline is pinned at 1.0/1.0; a regression that over-merges drops
    precision, one that over-splits drops recall.
    """
    predicted = _predicted_pairs(_resolve_gold())
    gold = _gold_pairs(_GOLD)
    precision, recall = _precision_recall(predicted, gold)
    assert precision == 1.0, f"merge precision regressed (over-merge): {precision}"
    assert recall == 1.0, f"merge recall regressed (over-split): {recall}"


def test_confusable_duplicate_is_recalled() -> None:
    """The fullwidth-confusable Acme surface is in the same predicted cluster as ASCII Acme.

    Pins the NFKC contribution to recall specifically: reverting the fold drops m3 out of the
    Acme cluster, and the (m1,m3) gold pair goes unrecalled.
    """
    predicted = _predicted_pairs(_resolve_gold())
    assert frozenset(("m1", "m3")) in predicted


def test_distinct_same_prefix_names_are_not_over_merged() -> None:
    """Globex Corporation and Globex Holdings stay in DIFFERENT clusters (precision guard).

    Pins the precision side: these share the ``globex`` token but are distinct entities; a
    greedy substring/overlap merge would wrongly unite them and tank precision.
    """
    predicted = _predicted_pairs(_resolve_gold())
    assert frozenset(("m4", "m5")) not in predicted

"""S8 disambiguation engine: the ported resolver cascade, Union-Find, KB linking, normaliser.

Unit tests over the pure engine (:mod:`latence_core.disambiguation`): each cascade strategy
fires and merges (or, below policy, is logged not applied — no silent over-merge); Union-Find
picks a deterministic lexicographic-min canonical; GLinker-style KB linking accepts above L0
and falls back below it; the relation normaliser alias/fuzzy-maps labels, type-filters, and
detects inverses with a direction swap. Deterministic and seed-free (no randomness in the
CPU path), so identical input ⇒ identical output.
"""

from __future__ import annotations

from latence_core.disambiguation import (
    EntityResolver,
    KBEntry,
    KnowledgeBase,
    MentionInput,
    RelationInput,
    RelationNormalizer,
    Strategy,
    acronym_of,
    normalize_text,
    token_similarity,
)


def _m(mid: str, doc: str, text: str, label: str = "ORG", conf: float = 0.9) -> MentionInput:
    return MentionInput(mention_id=mid, document_id=doc, text=text, label=label, confidence=conf)


# -- helpers ------------------------------------------------------------------


def test_normalize_and_acronym_and_similarity() -> None:
    assert normalize_text("  Acme   Corp  ") == "acme corp"
    assert acronym_of("International Business Machines") == "ibm"
    assert acronym_of("Acme") == ""  # single word has no acronym
    assert token_similarity("Acme Corp", "Acme Corp") == 1.0
    assert token_similarity("Acme", "Globex") == 0.0


# -- exact / grouping ---------------------------------------------------------


def test_exact_surface_folds_into_one_cluster() -> None:
    resolver = EntityResolver()
    resolver.add_all([_m("m1", "d1", "Acme"), _m("m2", "d2", "acme"), _m("m3", "d1", "ACME")])
    clusters, _ = resolver.resolve()
    assert len(clusters) == 1
    assert clusters[0].member_mention_ids == ["m1", "m2", "m3"]
    assert clusters[0].document_ids == ["d1", "d2"]


# -- acronym ------------------------------------------------------------------


def test_acronym_strategy_merges_and_is_applied() -> None:
    resolver = EntityResolver()
    resolver.add_all(
        [_m("m1", "d1", "International Business Machines"), _m("m2", "d2", "IBM")]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 1
    applied = [a for a in audits if a.applied]
    assert any(a.strategy is Strategy.ACRONYM for a in applied)


# -- precision guards on acronym + substring (issue #35) ----------------------


def test_acronym_type_incompatible_is_blocked_not_merged() -> None:
    # "IT" (SKILL) vs "Information Technology" (FIELD): the acronym fires, but the surfaces
    # carry disjoint types, so the type-compatibility guard blocks the merge. Recorded as a
    # below-guard decision (applied=False), never silently over-merged. Would FAIL (they'd
    # merge into one cluster) if the guard were reverted.
    resolver = EntityResolver()
    resolver.add_all(
        [
            _m("m1", "d1", "Information Technology", "FIELD"),
            _m("m2", "d1", "IT", "SKILL"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2  # NOT merged despite the acronym match
    acr = [a for a in audits if a.strategy is Strategy.ACRONYM]
    assert acr and acr[0].applied is False
    assert "type-incompatible" in acr[0].reason


def test_substring_type_incompatible_is_blocked_not_merged() -> None:
    # "Atlantic Trust" (2 tokens, CONCEPT) is a whole-token substring of "Central Atlantic
    # Trust" (3 tokens, ORG): ratio 2/3 ≥ 0.5 so the SUBSTRING rung fires, but the surfaces
    # carry disjoint types, so the type guard blocks the merge. With a permissive policy the
    # merge WOULD apply were the guard reverted — the adversarial regression.
    resolver = EntityResolver(min_confidence=0.0)
    resolver.add_all(
        [
            _m("m1", "d1", "Central Atlantic Trust", "ORG"),
            _m("m2", "d1", "Atlantic Trust", "CONCEPT"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2  # blocked by the type guard, not merged
    sub = [a for a in audits if a.strategy is Strategy.SUBSTRING]
    assert sub and sub[0].applied is False and "type-incompatible" in sub[0].reason


def test_acronym_same_type_still_merges_cross_document() -> None:
    # The precision guard must NOT block a legitimate same-typed cross-document acronym
    # merge: IBM (ORG) in d2 and International Business Machines (ORG) in d1 still fold to
    # one cluster (type-compatible; shared-document guard is opt-in / off by default).
    resolver = EntityResolver()
    resolver.add_all(
        [
            _m("m1", "d1", "International Business Machines", "ORG"),
            _m("m2", "d2", "IBM", "ORG"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 1
    assert any(a.strategy is Strategy.ACRONYM and a.applied for a in audits)


def test_untyped_surface_is_treated_as_type_compatible() -> None:
    # A surface with an empty label is not blocked wholesale — the type guard only fires when
    # BOTH sides carry disjoint non-empty types. Here both are "" so the acronym merges.
    resolver = EntityResolver()
    resolver.add_all(
        [_m("m1", "d1", "International Business Machines", ""), _m("m2", "d2", "IBM", "")]
    )
    clusters, _ = resolver.resolve()
    assert len(clusters) == 1


def test_shared_document_guard_blocks_cross_document_when_enabled() -> None:
    # With the opt-in shared-document guard enabled, a same-typed acronym whose surfaces never
    # co-occur in one document is blocked (audited, applied=False). Would FAIL (merged) if the
    # guard were not wired.
    resolver = EntityResolver(require_shared_document=True)
    resolver.add_all(
        [
            _m("m1", "d1", "International Business Machines", "ORG"),
            _m("m2", "d2", "IBM", "ORG"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2
    acr = [a for a in audits if a.strategy is Strategy.ACRONYM]
    assert acr and acr[0].applied is False and "no-shared-document" in acr[0].reason


def test_shared_document_guard_allows_same_document_merge() -> None:
    # The same guard permits the merge when the surfaces DO share a document.
    resolver = EntityResolver(require_shared_document=True)
    resolver.add_all(
        [
            _m("m1", "d1", "International Business Machines", "ORG"),
            _m("m2", "d1", "IBM", "ORG"),
        ]
    )
    clusters, _ = resolver.resolve()
    assert len(clusters) == 1


def test_guards_tunable_off_restores_permissive_cascade() -> None:
    # Turning both guards off reproduces the pre-#35 permissive behaviour: a type-incompatible
    # acronym merges at applied confidence (the over-merge the guards exist to prevent).
    resolver = EntityResolver(
        require_type_compatibility=False, require_shared_document=False
    )
    resolver.add_all(
        [
            _m("m1", "d1", "Information Technology", "FIELD"),
            _m("m2", "d2", "IT", "SKILL"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 1  # over-merged, as the un-guarded cascade would
    assert any(a.strategy is Strategy.ACRONYM and a.applied for a in audits)


# -- alias --------------------------------------------------------------------


def test_alias_strategy_merges_configured_synonym() -> None:
    resolver = EntityResolver(aliases={"Big Blue": "IBM"})
    resolver.add_all([_m("m1", "d1", "Big Blue"), _m("m2", "d2", "IBM")])
    clusters, audits = resolver.resolve()
    assert len(clusters) == 1
    assert any(a.strategy is Strategy.ALIAS and a.applied for a in audits)


# -- substring: no silent over-merge -----------------------------------------


def test_substring_below_policy_is_logged_not_applied() -> None:
    # "Acme" (1 token) within "Acme Corporation" (2 tokens) -> ratio 0.5, below the 0.82
    # default policy: recorded but NOT merged (no silent over-merge, S8 AC).
    resolver = EntityResolver()
    resolver.add_all([_m("m1", "d1", "Acme"), _m("m2", "d2", "Acme Corporation")])
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2  # not merged
    sub = [a for a in audits if a.strategy is Strategy.SUBSTRING]
    assert sub and sub[0].applied is False


def test_substring_requires_whole_token_not_char_containment() -> None:
    # The SUBSTRING guard is WHOLE-TOKEN: "Acme" is a token inside "Acme Corporation" but is
    # NOT a substring of "Acmestic" (character containment). A naive `"acme" in "acmestic"`
    # would wrongly fire SUBSTRING and (with a low policy) over-merge. Lower the policy so any
    # SUBSTRING hit WOULD merge, then assert none fires and the two stay separate.
    resolver = EntityResolver(min_confidence=0.0, enable_embedding=False)
    resolver.add_all([_m("m1", "d1", "Acme"), _m("m2", "d2", "Acmestic")])
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2  # never merged — no whole-token containment
    assert not any(a.strategy is Strategy.SUBSTRING for a in audits)


def test_substring_short_fragment_does_not_merge_into_long_name() -> None:
    # A single whole-token fragment ("a") IS contained in "a b c d" but the length ratio
    # (1/4 = 0.25) is below the default min_substring_ratio (0.5), so SUBSTRING declines —
    # `a` does not fold into `a b c d`. Even with a permissive policy the ratio gate holds.
    resolver = EntityResolver(min_confidence=0.0, enable_embedding=False)
    resolver.add_all([_m("m1", "d1", "a"), _m("m2", "d2", "a b c d")])
    clusters, _ = resolver.resolve()
    assert len(clusters) == 2  # ratio gate keeps the tiny fragment separate


def test_substring_above_policy_merges_when_ratio_clears() -> None:
    # Lower the policy so the substring ratio (0.5) clears it: now it merges and is audited.
    resolver = EntityResolver(min_confidence=0.4)
    resolver.add_all([_m("m1", "d1", "Acme"), _m("m2", "d2", "Acme Corporation")])
    clusters, audits = resolver.resolve()
    assert len(clusters) == 1
    assert any(a.strategy is Strategy.SUBSTRING and a.applied for a in audits)


# -- embedding (CPU lexical stand-in) -----------------------------------------


def test_embedding_strategy_can_be_disabled() -> None:
    # Two names sharing 3 of 4 union tokens (Jaccard 0.75) merge under the lexical EMBEDDING
    # stand-in when the policy admits that confidence; disabling EMBEDDING (the strict
    # no-embedder path) keeps them separate regardless of similarity.
    pair = [_m("m1", "d1", "Acme Global Holdings Corp"), _m("m2", "d2", "Acme Global Holdings Inc")]
    on = EntityResolver(embedding_threshold=0.5, min_confidence=0.5)
    on.add_all(list(pair))
    off = EntityResolver(embedding_threshold=0.5, min_confidence=0.5, enable_embedding=False)
    off.add_all(list(pair))
    assert len(on.resolve()[0]) == 1
    assert len(off.resolve()[0]) == 2


# -- Union-Find determinism ---------------------------------------------------


def test_canonical_is_lexicographic_min_and_deterministic() -> None:
    # Transitive merge: alias links all three; the canonical is the lexicographically
    # smallest normalised member, regardless of insertion order.
    resolver = EntityResolver(aliases={"Zeta Corp": "Acme", "Acme": "Alpha"})
    resolver.add_all([_m("m1", "d1", "Zeta Corp"), _m("m2", "d2", "Acme"), _m("m3", "d3", "Alpha")])
    clusters, _ = resolver.resolve()
    assert len(clusters) == 1
    assert clusters[0].canonical_text == "Acme"  # 'acme' < 'alpha' < 'zeta corp'


def test_majority_label_vote() -> None:
    resolver = EntityResolver()
    resolver.add_all(
        [_m("m1", "d1", "Acme", "ORG"), _m("m2", "d2", "acme", "ORG"), _m("m3", "d3", "ACME", "COMPANY")]
    )
    clusters, _ = resolver.resolve()
    assert clusters[0].entity_type == "ORG"  # 2 vs 1


def test_resolve_is_deterministic() -> None:
    mentions = [_m("m1", "d1", "IBM"), _m("m2", "d2", "International Business Machines"),
                _m("m3", "d1", "Acme"), _m("m4", "d2", "Alice", "PERSON")]
    a = EntityResolver()
    a.add_all(mentions)
    b = EntityResolver()
    b.add_all(list(reversed(mentions)))
    ca = [(c.canonical_text, c.member_mention_ids) for c in a.resolve()[0]]
    cb = [(c.canonical_text, c.member_mention_ids) for c in b.resolve()[0]]
    assert ca == cb


# -- KB linking ---------------------------------------------------------------


def test_kb_links_above_threshold_and_falls_back_below() -> None:
    kb = KnowledgeBase(
        [KBEntry("Q1", "Acme Corporation", "ORG", ("Acme",)), KBEntry("Q2", "Globex", "ORG")],
        threshold=0.5,
    )
    hit = kb.link("Acme", "ORG")
    assert hit.linked and hit.kb_id == "Q1"
    miss = kb.link("Umbrella Corp", "ORG")
    assert miss.linked is False and miss.kb_id is None


def test_kb_candidate_below_l0_threshold_falls_back() -> None:
    # A REAL candidate is generated (shares the token "acme") but scores below the L0
    # threshold, so it must fall back to unlinked — exercising the `score < threshold`
    # branch, not the empty-candidate early return. "Acme" (1 token) vs the 4-token entry
    # name is Jaccard 0.25 < 0.5, so the link is rejected but the score is reported.
    kb = KnowledgeBase(
        [KBEntry("Q1", "Acme Corporation International Group", "ORG")],
        threshold=0.5,
    )
    result = kb.link("Acme", "ORG")
    assert result.linked is False
    assert result.kb_id is None
    assert result.score == 0.25  # a real candidate was scored, then rejected by L0


def test_kb_type_filter_rejects_mismatched_type() -> None:
    kb = KnowledgeBase([KBEntry("Q1", "Acme", "ORG")], threshold=0.5)
    assert kb.link("Acme", "PERSON").linked is False


def test_empty_kb_always_falls_back() -> None:
    kb = KnowledgeBase([], threshold=0.5)
    assert kb.link("Acme", "ORG").linked is False


# -- relation normalisation ---------------------------------------------------


def _r(rid: str, label: str, h: str, t: str) -> RelationInput:
    return RelationInput(
        relation_id=rid, document_id="d1", label=label,
        head_mention_id=h, tail_mention_id=t, head_text=h, tail_text=t, confidence=0.8,
    )


def test_label_alias_normalisation() -> None:
    norm = RelationNormalizer(label_aliases={"employed_by": "works_for"})
    out = norm.normalize(_r("r1", "employed_by", "Alice", "Acme"))
    assert out is not None and out.label == "works_for" and out.inverted is False


def test_inverse_detection_swaps_direction() -> None:
    norm = RelationNormalizer(inverses={"employs": "works_for"})
    out = norm.normalize(_r("r1", "employs", "Acme", "Alice"))
    assert out is not None
    assert out.label == "works_for" and out.inverted is True
    assert out.head_mention_id == "Alice" and out.tail_mention_id == "Acme"


def test_type_filter_drops_unallowed_label() -> None:
    norm = RelationNormalizer(allowed_labels={"works_for"})
    assert norm.normalize(_r("r1", "located_in", "Alice", "Paris")) is None
    assert norm.normalize(_r("r2", "works_for", "Alice", "Acme")) is not None


def test_fuzzy_label_match_to_closest_canonical() -> None:
    norm = RelationNormalizer(allowed_labels={"works for company"}, label_threshold=0.5)
    out = norm.normalize(_r("r1", "works for", "Alice", "Acme"))
    assert out is not None and out.label == "works_for_company"

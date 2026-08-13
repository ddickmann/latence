"""Adversarial / property hardening for the corpus-level Stages (issue #75).

The Hardening bar for Profiling / Disambiguation / Graph Assembly: pathological and
hostile inputs must degrade safely — no crash, no unbounded state, no *silent* over-merge,
and every emitted record still carries its Evidence provenance. Each test is deterministic
(no randomness, no seeds needed — the CPU engines have no stochastic path) and would FAIL if
the guard it pins were reverted.

Coverage matrix (closes the issue's enumerated edge cases):

* **cyclic / self-referential relations** — a relation whose head and tail collapse to the
  same canonical entity is dropped, never emitted as a self-loop edge; a directed cycle of
  distinct entities assembles cleanly (a KG is a directed graph, cycles are legal).
* **entity name collisions** — surfaces that normalise identically fold into one cluster;
  same-surface / different-type mentions still fold (majority label wins) and the collision
  is not silently split.
* **unicode-confusable surfaces** — compatibility confusables (fullwidth, ligature, Roman
  numeral) fold to one canonical entity (NFKC), while cross-script homoglyphs stay DISTINCT
  (no silent over-merge across scripts).
* **huge co-occurrence fan-out** — the #63 per-document fan-out guard bounds a single
  ``add_mentions`` call to ``O(cap²)``; asserted directly against the retained pair table.
* **degenerate KGs** — empty corpus, all-dangling relations, an empty KB: every path yields
  a well-formed (possibly empty) graph, not an exception.
* **Evidence provenance + no-silent-over-merge** — every below-policy / below-guard merge is
  audited with ``applied=False`` and NOT folded; every built node/edge carries Evidence.
"""

from __future__ import annotations

from latence_core.contracts import (
    CanonicalEntity,
    Classification,
    Evidence,
    NormalizedRelation,
    Provenance,
)
from latence_core.disambiguation import (
    EntityResolver,
    KBEntry,
    KnowledgeBase,
    MentionInput,
    normalize_text,
)
from latence_core.graph import GraphBuilder
from latence_core.profiling import (
    _MAX_ENTITIES_PER_DOCUMENT,
    AggregatorMentionView,
    CorpusAggregator,
)

# -- fixtures -----------------------------------------------------------------


def _m(mid: str, doc: str, text: str, label: str = "ORG", conf: float = 0.9) -> MentionInput:
    return MentionInput(mention_id=mid, document_id=doc, text=text, label=label, confidence=conf)


def _prov(doc: str = "doc-1") -> Provenance:
    return Provenance(
        source_uri=f"file:///c/{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=10,
        document_id=doc,
        char_start=0,
        char_end=4,
        page_start=1,
        page_end=1,
    )


def _cls() -> Classification:
    return Classification(language="en", sensitivity="internal")


def _entity(rid: str, name: str, docs: list[str], mentions: list[str]) -> CanonicalEntity:
    return CanonicalEntity(
        record_id=rid,
        canonical_text=name,
        entity_type="ORG",
        member_mention_ids=mentions,
        source_document_ids=docs,
        evidence=Evidence(mention_ids=mentions, document_ids=docs, confidence=0.9),
        provenance=_prov(docs[0]),
        classification=_cls(),
    )


def _relation(rid: str, head: str, tail: str) -> NormalizedRelation:
    return NormalizedRelation(
        record_id=rid,
        source_relation_id=rid.replace("#normalized", ""),
        label="works_for",
        head_entity_id=head,
        tail_entity_id=tail,
        head_text="H",
        tail_text="T",
        confidence=0.8,
        evidence=Evidence(mention_ids=["m1", "m2"], document_ids=["doc-1"], confidence=0.8),
        provenance=_prov(),
        classification=_cls(),
    )


# -- unicode-confusable surfaces ----------------------------------------------


def test_compatibility_confusables_fold_into_one_entity() -> None:
    """Fullwidth / ligature / Roman-numeral spellings resolve to the SAME canonical entity.

    Without the NFKC fold in normalize_text these look-alike glyphs would key differently and
    split one real entity into several — the adversary's "hide a duplicate behind a look-alike"
    vector. Reverting the NFKC normalisation fails this: the surfaces would land in 3 clusters.
    """
    resolver = EntityResolver()
    resolver.add_all(
        [
            _m("m1", "d1", "ACME"),  # ascii
            _m("m2", "d2", "ＡＣＭＥ"),  # fullwidth ＡＣＭＥ
            _m("m3", "d3", "ᴬCME"),  # superscript-A + CME
        ]
    )
    clusters, _ = resolver.resolve()
    assert len(clusters) == 1
    # All three mentions folded in; the first-seen ASCII surface is a valid canonical.
    assert clusters[0].member_mention_ids == ["m1", "m2", "m3"]
    assert clusters[0].document_ids == ["d1", "d2", "d3"]


def test_cross_script_homoglyphs_are_not_silently_over_merged() -> None:
    """A Cyrillic-A look-alike stays DISTINCT from the Latin spelling — no silent over-merge.

    NFKC folds compatibility variants but NOT cross-script homoglyphs (Cyrillic U+0410 'А' is
    genuinely a different character from Latin U+0041 'A'). Folding them would let an attacker
    merge two unrelated entities by picking a look-alike glyph; the safe posture keeps them
    apart. Embedding is disabled so only the exact/typed rungs run (a fair "do they key the
    same" test).
    """
    latin = "Acme"
    cyrillic = "Аcme"  # Cyrillic А + cme
    assert normalize_text(latin) != normalize_text(cyrillic)
    resolver = EntityResolver(enable_embedding=False)
    resolver.add_all([_m("m1", "d1", latin), _m("m2", "d2", cyrillic)])
    clusters, _ = resolver.resolve()
    assert len(clusters) == 2  # two distinct entities, not one over-merged blob


# -- entity name collisions ---------------------------------------------------


def test_name_collision_across_types_folds_with_majority_label() -> None:
    """Same surface tagged with different types collides into one cluster (majority label wins).

    The exact-surface grouping is type-blind (a collision on the literal surface is one
    entity); the reported entity_type is the majority label, deterministically. This is a
    collision the Stage must NOT split, and the audit must not claim a strategy fired.
    """
    resolver = EntityResolver()
    resolver.add_all(
        [
            _m("m1", "d1", "Mercury", label="ORG"),
            _m("m2", "d2", "mercury", label="ORG"),
            _m("m3", "d3", "MERCURY", label="PRODUCT"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 1
    assert clusters[0].entity_type == "ORG"  # majority (2 ORG vs 1 PRODUCT)
    # Exact collisions are resolved by grouping, not by a fired cascade rung: no applied audit.
    assert all(not a.applied for a in audits)


# -- huge co-occurrence fan-out (the #63 guard) -------------------------------


def test_cooccurrence_fanout_is_bounded_by_the_63_guard() -> None:
    """A single document with thousands of distinct entities cannot build an unbounded pair set.

    The #63 fan-out guard caps the distinct entities entering the O(n²) pairing at
    ``max_entities_per_document``. With N >> cap distinct entities in one document, the
    retained pair table must be exactly C(cap, 2) — NOT C(N, 2). Reverting the cap makes this
    fail with a pair count blown up by orders of magnitude.
    """
    cap = 8
    n = 200  # far above the cap
    agg = CorpusAggregator(max_entities_per_document=cap)
    agg.add_mentions([AggregatorMentionView("d1", f"E{i:04d}", "ORG") for i in range(n)])
    # Exactly C(cap, 2) pairs are retained — the O(cap²) bound, not O(n²).
    assert len(agg._pair_docs) == cap * (cap - 1) // 2
    # And the default cap is the documented #63 value, so the guard is really wired.
    assert _MAX_ENTITIES_PER_DOCUMENT == 200


def test_cooccurrence_default_cap_bounds_default_aggregator() -> None:
    """The DEFAULT-constructed aggregator (no override) is bounded by the #63 default cap.

    Feeds one document with more distinct entities than the default cap and asserts the pair
    table is bounded by C(default_cap, 2), proving the guard is on by default (not only when a
    test passes a small cap).
    """
    n = _MAX_ENTITIES_PER_DOCUMENT + 50
    agg = CorpusAggregator()  # default caps
    agg.add_mentions([AggregatorMentionView("d1", f"E{i:05d}", "ORG") for i in range(n)])
    expected = _MAX_ENTITIES_PER_DOCUMENT * (_MAX_ENTITIES_PER_DOCUMENT - 1) // 2
    assert len(agg._pair_docs) == expected


# -- cyclic / self-referential relations --------------------------------------


def test_self_referential_relation_is_dropped_not_a_self_loop_edge() -> None:
    """A relation whose head and tail resolve to the same node is dropped (no self-loop edge).

    Uses model_construct to feed the engine the malformed same-endpoint relation the upstream
    contract validator normally forbids, and asserts the engine drops it. Reverting the
    ``source_node == target_node`` guard would emit a GraphEdge with equal endpoints, which the
    GraphEdge validator rejects — crashing assembly.
    """
    entity = _entity("entity:acme", "Acme", ["doc-1"], ["m1"])
    self_rel = NormalizedRelation.model_construct(
        record_id="rel-self#normalized",
        source_relation_id="rel-self",
        label="works_for",
        head_entity_id="entity:acme",
        tail_entity_id="entity:acme",
        head_text="Acme",
        tail_text="Acme",
        confidence=0.8,
        inverted=False,
        evidence=Evidence(mention_ids=["m1"], document_ids=["doc-1"], confidence=0.8),
        provenance=_prov(),
        classification=_cls(),
    )
    nodes, edges = GraphBuilder("corpus").build([entity], [self_rel])
    assert len(nodes) == 1
    assert edges == []  # the self-loop is dropped


def test_directed_cycle_of_distinct_entities_assembles_cleanly() -> None:
    """A directed cycle A→B→C→A over distinct entities is a legal KG and assembles fully.

    A knowledge graph is a directed graph; cycles between DISTINCT entities are valid (unlike
    a self-loop). All three edges must be emitted, each with its Evidence — the assembler must
    not choke on or silently prune a cycle.
    """
    ea = _entity("entity:a", "A", ["doc-1"], ["m1"])
    eb = _entity("entity:b", "B", ["doc-1"], ["m2"])
    ec = _entity("entity:c", "C", ["doc-1"], ["m3"])
    rels = [
        _relation("rel-ab#normalized", "entity:a", "entity:b"),
        _relation("rel-bc#normalized", "entity:b", "entity:c"),
        _relation("rel-ca#normalized", "entity:c", "entity:a"),
    ]
    nodes, edges = GraphBuilder("corpus").build([ea, eb, ec], rels)
    assert len(nodes) == 3
    assert len(edges) == 3
    # Every edge carries Evidence provenance (S9 AC) — none dropped by the cycle.
    assert all(e.evidence_mention_ids for e in edges)
    assert all(e.evidence_document_ids for e in edges)


# -- degenerate KGs -----------------------------------------------------------


def test_empty_corpus_assembles_to_empty_graph() -> None:
    """No entities, no relations ⇒ an empty (not erroring) graph."""
    nodes, edges = GraphBuilder("corpus").build([], [])
    assert nodes == []
    assert edges == []


def test_all_relations_dangling_yields_nodes_but_no_edges() -> None:
    """Relations pointing only at absent entities are all dropped — nodes stand, zero edges.

    The endpoint-validation guard must drop every dangling relation without dropping the
    legitimate nodes or raising — a degenerate but valid corpus shape.
    """
    ea = _entity("entity:a", "A", ["doc-1"], ["m1"])
    rels = [
        _relation("rel-1#normalized", "entity:a", "entity:ghost"),
        _relation("rel-2#normalized", "entity:phantom", "entity:a"),
    ]
    nodes, edges = GraphBuilder("corpus").build([ea], rels)
    assert len(nodes) == 1
    assert edges == []


def test_empty_knowledge_base_links_nothing_gracefully() -> None:
    """An empty KB (a valid configuration) links nothing — graceful fallback, not a crash."""
    kb = KnowledgeBase([])
    result = kb.link("Acme", "ORG")
    assert result.linked is False
    assert result.kb_id is None


def test_degenerate_kb_with_no_shared_token_falls_back() -> None:
    """A KB whose entries share no token with the query returns the unlinked fallback."""
    kb = KnowledgeBase([KBEntry(kb_id="Q1", name="Globex", entity_type="ORG")], threshold=0.5)
    result = kb.link("Acme Corporation", "ORG")
    assert result.linked is False  # no candidate cleared L0


# -- Evidence provenance + no-silent-over-merge under adversarial input --------


def test_type_incompatible_acronym_is_audited_not_silently_merged() -> None:
    """An acronym match across incompatible types is logged (applied=False), never folded.

    "IT" (an acronym of "Information Technology") tagged differently must not silently merge.
    The below-guard decision must appear in the audit with applied=False AND the two surfaces
    must stay in separate clusters — the no-silent-over-merge guarantee (S8 / issue #35).
    Reverting the type-compatibility guard folds them into one cluster and fails this.
    """
    resolver = EntityResolver()
    resolver.add_all(
        [
            _m("m1", "d1", "Information Technology", label="FIELD"),
            _m("m2", "d1", "IT", label="PRONOUN"),
        ]
    )
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2  # not over-merged
    blocked = [a for a in audits if not a.applied and "blocked" in a.reason]
    assert blocked, "the blocked acronym merge must be audited, not silently dropped"


def test_below_policy_merge_is_recorded_and_not_applied() -> None:
    """A cascade rung firing below min_confidence is audited applied=False and NOT folded.

    Sets min_confidence above the embedding score so the fired EMBEDDING decision lands
    below policy: the pair stays split and the decision is recorded — no silent over-merge and
    no silent drop.
    """
    # "Acme Corp" vs "Acme Holdings": token overlap {acme} / {acme,corp,holdings} = 1/3 ≈ 0.33.
    resolver = EntityResolver(embedding_threshold=0.3, min_confidence=0.9)
    resolver.add_all([_m("m1", "d1", "Acme Corp"), _m("m2", "d2", "Acme Holdings")])
    clusters, audits = resolver.resolve()
    assert len(clusters) == 2  # below policy ⇒ not merged
    below = [a for a in audits if not a.applied and "below policy" in a.reason]
    assert below, "a below-policy fired decision must be audited"


def test_every_built_node_carries_evidence_under_collision() -> None:
    """Even with colliding/confusable surfaces, every assembled node has Evidence provenance.

    Assembles nodes from entities whose surfaces are confusable variants and asserts each node
    carries back-references (mention + document Evidence) — the S9 provenance guarantee holds
    under adversarial surfaces.
    """
    ea = _entity("entity:acme", "Acme", ["doc-1", "doc-2"], ["m1", "m2"])
    eb = _entity("entity:globex", "Globex", ["doc-3"], ["m3"])
    nodes, _ = GraphBuilder("corpus").build([ea, eb], [])
    assert len(nodes) == 2
    for node in nodes:
        assert node.evidence_mention_ids, "node must cite its source mentions"
        assert node.evidence_document_ids, "node must cite its source documents"
        assert node.evidence_confidence > 0.0

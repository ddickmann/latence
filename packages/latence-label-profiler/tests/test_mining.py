"""Candidate mining: RAKE-style noun phrases, ReVerb-style relations, structural PII detectors."""

from __future__ import annotations

from latence_label_profiler.mining import mine_corpus
from latence_label_profiler.text import canonical_label, subsequences, tokenize


def test_tokenizer_breaks_phrases_on_stopwords_punctuation_and_bare_numbers() -> None:
    runs = [
        tuple(t.text for t in run)
        for run in tokenize("The purchase order number is 4711, issued by Acme Corporation.").runs
    ]
    assert ("purchase", "order", "number") in runs
    assert ("acme", "corporation") in runs
    # "4711" is an INSTANCE, never a type: a bare number must break the run, not join it.
    assert not any("4711" in run for run in runs)


def test_canonical_label_matches_the_inducer_normal_form() -> None:
    """Our labels must already be fixed points of the downstream sanitizer, not near-misses."""
    from latence_schema_inducer.sanitize import sanitize_label

    for raw in ("Payment_Product", "invoice-number", "  Purchase   Order  ", "E-Mail Address"):
        ours = canonical_label(raw)
        assert ours is not None
        assert sanitize_label(ours) == ours
        assert sanitize_label(raw) == ours


def test_subsequences_are_the_downstream_matcher_semantics() -> None:
    """A candidate covers a surface iff its cue would whole-word match that surface."""
    assert subsequences(("purchase", "order", "number"), max_len=3) == (
        ("purchase",),
        ("purchase", "order"),
        ("purchase", "order", "number"),
        ("order",),
        ("order", "number"),
        ("number",),
    )


def test_mining_groups_head_types_across_distinct_surfaces() -> None:
    texts = [
        "The purchase order number was recorded.",
        "The invoice number was recorded.",
        "The delivery note number was recorded.",
    ]
    mined = mine_corpus(texts)
    entity = mined.slots["entity_types"]
    assert "number" in entity.candidates
    # "number" is the shared head, so it covers all three distinct surfaces.
    covered = entity.candidates["number"].covers
    assert {"purchase order number", "invoice number", "delivery note number"} <= covered


def test_relation_mining_finds_the_span_between_two_noun_phrases() -> None:
    texts = ["The commodity is classified under the tariff heading."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    relations = mined.slots["relation_types"].candidates
    assert any("classified" in label for label in relations)


def test_a_trailing_verb_is_stripped_from_the_phrase_and_handed_to_the_relation() -> None:
    """A verb that happens to be a content word must not ride along inside the entity type.

    ``maps`` is in the closed verb set but is not a stopword, so it does not break the RAKE run:
    without an explicit trim, ``commodity code maps`` becomes a candidate ENTITY type and the
    relation ``maps to`` is never mined at all.
    """
    texts = ["Each commodity code maps to a tariff heading."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    entity = mined.slots["entity_types"].candidates
    assert "commodity code" in entity
    assert not any("maps" in label for label in entity), sorted(entity)
    relations = mined.slots["relation_types"].candidates
    assert any("maps" in label for label in relations), sorted(relations)


def test_a_leading_verb_is_stripped_too() -> None:
    texts = ["Die Ausfuhrgenehmigung wurde von der Zollbehoerde erteilt."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    entity = mined.slots["entity_types"].candidates
    assert "ausfuhrgenehmigung" in entity
    assert not any("wurde" in label for label in entity), sorted(entity)


def test_a_vacuous_relation_phrase_is_rejected() -> None:
    """ReVerb's lexical constraint: a relation of pure copula + preposition says nothing.

    ``is`` covers both ``is classified under`` and ``is required for``, so greedy max-coverage
    would rank it FIRST on mass — and a relation type called "is" is worthless to an extractor.
    """
    texts = [
        "The commodity is classified under the tariff heading.",
        "The export licence is required for the item.",
    ] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    relations = mined.slots["relation_types"].candidates
    assert "is" not in relations, sorted(relations)
    assert "is under" not in relations
    assert any("classified" in label for label in relations), sorted(relations)


def test_an_ed_adjective_before_a_noun_stays_inside_the_noun_phrase() -> None:
    """``combined nomenclature`` is a noun phrase; ``is classified`` is a relation. Same suffix.

    The distinguishing signal is the token BEFORE it: a participle after a copula is a verb, a
    participle after a determiner is an adjective. Without that check, ``combined`` is trimmed out
    of its own noun phrase and mined as a relation type.
    """
    texts = ["A tariff heading in the combined nomenclature."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    entity = mined.slots["entity_types"].candidates
    assert "combined nomenclature" in entity, sorted(entity)
    assert mined.slots["relation_types"].candidates == {}


def test_a_participle_after_a_copula_is_still_trimmed() -> None:
    """The other half of the same rule — it must not have been bought by disabling the trim."""
    texts = ["The commodity is classified under the tariff heading."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    entity = mined.slots["entity_types"].candidates
    assert not any("classified" in label for label in entity), sorted(entity)
    assert any("classified" in label for label in mined.slots["relation_types"].candidates)


def test_an_ing_noun_is_not_treated_as_a_verb() -> None:
    """REGRESSION: with ``-ing`` in the verb-suffix table, ``heading`` was trimmed out of its own
    noun phrase and the nonsense relation ``heading of`` was mined in its place. ``heading``,
    ``building``, ``meeting`` and ``training`` are nouns, and all four are common in exactly the
    corporate corpus this profiler targets.
    """
    texts = ["The tariff heading of the building meeting the training standard."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    entity = mined.slots["entity_types"].candidates
    assert "tariff heading" in entity
    assert "building" in entity
    assert mined.slots["relation_types"].candidates == {}


def test_relation_mining_requires_a_verb_like_token() -> None:
    """A bare preposition between two phrases is not a relation; it must not be mined as one."""
    texts = ["The tariff heading of the commodity code."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    assert mined.slots["relation_types"].candidates == {}


def test_pii_detectors_measure_mass_but_emit_contextual_cues_only() -> None:
    """A detected value is PII; it must never be echoed into an emitted cue list."""
    texts = ["Contact jane.doe@example.com for details."] * 3
    mined = mine_corpus(texts, min_mention_mass=1, min_document_frequency=1)
    pii = mined.slots["pii_types"].candidates
    assert "email address" in pii
    cues = pii["email address"].cues
    assert all("jane.doe@example.com" not in cue for cue in cues)
    assert pii["email address"].detector_only is True


def test_document_frequency_floor_kills_single_document_boilerplate() -> None:
    texts = [
        "confidential internal memorandum about the widget assembly",
        "the widget assembly was shipped",
        "the widget assembly was inspected",
    ]
    mined = mine_corpus(texts, min_document_frequency=2, min_mention_mass=1)
    entity = mined.slots["entity_types"].candidates
    assert "widget assembly" in entity
    assert "confidential internal memorandum" not in entity


def test_mining_never_emits_a_pure_stopword_type() -> None:
    mined = mine_corpus(["the of and to the of and to"] * 5, min_mention_mass=1)
    assert mined.slots["entity_types"].candidates == {}

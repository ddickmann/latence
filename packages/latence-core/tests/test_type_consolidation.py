"""T3 — corpus-wide TYPE-vocabulary canonicalization (ADR-0054), at the Capability's interface.

Every assertion here is about OBSERVABLE BEHAVIOUR through the ``TypeConsolidator`` seam — the
records that come back and the :class:`~latence_core.contracts.TypeVocabulary` that explains them —
never about a config flag or an internal. The engine is exercised only where the Provider cannot
reach it (the semantic rung's Embedder contract).

The lane this covers: per-chunk induction (the extraction redesign) gives each chunk its own,
uncapped label set and therefore produces TYPE DRIFT — chunk A says ``org``, chunk B
``organization``, chunk C ``company``. Nothing in the pipeline canonicalized type labels before
this Stage, so the drift would have reached the knowledge graph as three node types where there is
one. See ``test_e2e_type_consolidation`` for the end-to-end proof through the Runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pytest
from latence_core.capability import Embedder
from latence_core.contracts import (
    Classification,
    EntityMention,
    Provenance,
    RelationMention,
    TypeKind,
    TypeVocabulary,
)
from latence_core.errors import ConfigError
from latence_core.stages.type_consolidation import (
    CascadeTypeConsolidator,
    ExactSurfaceTypeConsolidator,
)


def _provenance(doc: str = "doc-1") -> Provenance:
    return Provenance(
        source_uri=f"file:///c/{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=10,
        document_id=f"sha256:{doc}",
        char_start=0,
        char_end=5,
        page_start=1,
        page_end=1,
    )


def _classification() -> Classification:
    return Classification(language="en", sensitivity="internal")


def _mention(label: str, text: str, *, doc: str = "doc-1") -> EntityMention:
    return EntityMention(
        record_id=f"mention:{doc}:{label}:{text}",
        document_record_id=f"doc:{doc}",
        chunk_record_id=f"chunk:{doc}:0",
        index=0,
        provenance=_provenance(doc),
        classification=_classification(),
        label=label,
        text=text,
        confidence=0.9,
    )


def _relation(label: str, *, doc: str = "doc-1") -> RelationMention:
    return RelationMention(
        record_id=f"relation:{doc}:{label}",
        document_record_id=f"doc:{doc}",
        index=0,
        provenance=_provenance(doc),
        classification=_classification(),
        label=label,
        head_mention_id="m-1",
        tail_mention_id="m-2",
        head_text="Alice",
        tail_text="Acme",
        confidence=0.8,
    )


class _SynonymEmbedder:
    """A faithful offline stand-in for a real STS Embedder over TYPE STRINGS.

    Faithful means it enforces what a real Embedder enforces, and REFUSES what a real one refuses:
    every returned vector has exactly ``dimension`` components, one vector per input, in input
    order — the contract :class:`~latence_core.capability.Embedder` declares. It raises on an
    unknown string rather than returning a plausible-looking vector, so a test that accidentally
    embeds something it did not intend fails loudly instead of passing on a fake's shrug.

    The vectors encode a hand-built synonym geometry: ``org``/``organization``/``company`` sit on
    one axis, ``person`` on another, so the semantic rung has something real to find (``org`` ≡
    ``company`` is a synonym no lexical rule can catch) and something real to leave alone.
    """

    _VECTORS = {
        "org": [1.0, 0.0, 0.0],
        "organization": [0.99, 0.14, 0.0],
        "company": [0.97, 0.24, 0.0],
        "person": [0.0, 1.0, 0.0],
        "individual": [0.0, 0.99, 0.14],
        "invoice number": [0.0, 0.0, 1.0],
    }

    def __init__(self, config: dict[str, object] | None = None) -> None:
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        for text in batch:
            if text not in self._VECTORS:
                msg = (
                    f"_SynonymEmbedder has no vector for {text!r} — "
                    "the test embedded something unintended."
                )
                raise KeyError(msg)
        return iter([list(self._VECTORS[t]) for t in batch])


def test_fake_embedder_satisfies_the_embedder_capability() -> None:
    """The fake must BE an Embedder, or the rung it stands in for is not really exercised."""
    assert isinstance(_SynonymEmbedder(), Embedder)
    vectors = list(_SynonymEmbedder().embed(["org", "person"]))
    assert [len(v) for v in vectors] == [3, 3]


# -- the lane's core claim ---------------------------------------------------


def test_drifted_types_collapse_to_one_canonical_type() -> None:
    """org / organization / company — three induced names for one type — become ONE type.

    THE regression this lane exists for. Before ADR-0054 nothing canonicalized the type
    vocabulary, so per-chunk induction's drift reached the KG verbatim.
    """
    consolidator = CascadeTypeConsolidator(
        {"use_embedding_merge": True, "embedder": {"provider": "test.synonyms"}}
    )
    consolidator._embedder = _SynonymEmbedder()  # noqa: SLF001 - inject the offline stand-in
    result = consolidator.consolidate(
        [
            _mention("org", "Acme"),
            _mention("org", "Acme Corporation"),
            _mention("organization", "IBM", doc="doc-2"),
            _mention("company", "Globex", doc="doc-3"),
            _mention("person", "Alice"),
        ]
    )
    labels = {m.label for m in result.mentions}
    assert labels == {"org", "person"}, "the three org-ish types must collapse to exactly one"
    assert len(result.mentions) == 5, "consolidation relabels records, it never drops them"


def test_the_raw_induced_label_survives_on_every_remapped_record() -> None:
    """Provenance for the remap: an operator can always see which induced type a record came from."""
    result = CascadeTypeConsolidator({}).consolidate(
        [_mention("org", "Acme"), _mention("organization", "IBM"), _mention("person", "Alice")]
    )
    by_text = {m.text: m for m in result.mentions}
    # Equally observed, so the tiebreak elects the fuller surface: ``org`` is rewritten to
    # ``organization``, and the record that changed says so.
    remapped = by_text["Acme"]
    assert remapped.raw_label == "org"
    assert remapped.label == "organization"
    # A record that was NOT rewritten carries no raw_label — the field means "this was remapped",
    # so it never adds noise to a record nothing happened to.
    assert by_text["Alice"].raw_label is None


def test_the_alias_map_is_emitted_for_the_operator() -> None:
    """The vocabulary reports the rewrites actually applied — an empty map means nothing folded."""
    result = CascadeTypeConsolidator({}).consolidate(
        [_mention("org", "Acme"), _mention("organization", "IBM"), _mention("person", "Alice")]
    )
    assert result.vocabulary is not None
    assert result.vocabulary.alias_map(TypeKind.ENTITY) == {"org": "organization"}
    # ``person`` was canonical already, so it is NOT in the map (a rewrite log, not a type list).
    assert "person" not in result.vocabulary.alias_map(TypeKind.ENTITY)


def test_surface_drift_collapses_without_any_semantic_rung() -> None:
    """ORG / org. / Orgs are one type by normalization alone — no model, no config."""
    result = ExactSurfaceTypeConsolidator({}).consolidate(
        [_mention("ORG", "Acme"), _mention("org.", "IBM"), _mention("Orgs", "Globex")]
    )
    assert len({m.label for m in result.mentions}) == 1


def test_relation_types_are_canonicalized_but_never_folded_into_entity_types() -> None:
    """Two vocabularies, one pass: an entity ``org`` must not swallow a relation ``org``."""
    result = CascadeTypeConsolidator({}).consolidate(
        [_mention("organisation", "Acme"), _mention("organisations", "IBM")],
        [_relation("works for"), _relation("Works_For")],
    )
    assert len({m.label for m in result.mentions}) == 1
    assert len({r.label for r in result.relations}) == 1
    assert result.vocabulary is not None
    entity_labels = {t.canonical for t in result.vocabulary.of_kind(TypeKind.ENTITY)}
    relation_labels = {t.canonical for t in result.vocabulary.of_kind(TypeKind.RELATION)}
    assert entity_labels.isdisjoint(relation_labels)


# -- precision: what must NOT be folded --------------------------------------


def test_a_near_prefix_that_is_not_an_abbreviation_is_not_folded() -> None:
    """``product`` and ``production`` share a prefix and are DIFFERENT types.

    The gate that stops this is the whole reason the prefix rung has two thresholds: an
    over-merged TYPE retypes every node of that type, a costlier error than leaving drift.
    """
    result = CascadeTypeConsolidator({}).consolidate(
        [_mention("product", "Widget"), _mention("production", "Line 3")]
    )
    assert {m.label for m in result.mentions} == {"product", "production"}


def test_the_exact_surface_provider_refuses_the_abbreviation_fold() -> None:
    """The second reference Provider's defining behaviour: no fold without an explicit alias."""
    mentions = [_mention("org", "Acme"), _mention("organization", "IBM")]
    assert len({m.label for m in ExactSurfaceTypeConsolidator({}).consolidate(mentions).mentions}) == 2
    # ...and the operator's alias map still works — that is the escape hatch, not a rung.
    aliased = ExactSurfaceTypeConsolidator({"aliases": {"org": "organization"}})
    assert len({m.label for m in aliased.consolidate(mentions).mentions}) == 1


def test_the_exact_surface_provider_cannot_be_configured_back_into_the_cascade() -> None:
    """Its posture is a property of the Provider, not a knob a stack can quietly flip."""
    provider = ExactSurfaceTypeConsolidator(
        {"enable_prefix": True, "enable_acronym": True, "use_embedding_merge": True}
    )
    result = provider.consolidate([_mention("org", "Acme"), _mention("organization", "IBM")])
    assert len({m.label for m in result.mentions}) == 2


def test_an_unrelated_type_is_left_alone_by_the_semantic_rung() -> None:
    """The strict cosine floor keeps ``invoice number`` out of the ``person`` cluster."""
    consolidator = CascadeTypeConsolidator({"use_embedding_merge": True})
    consolidator._embedder = _SynonymEmbedder()  # noqa: SLF001
    result = consolidator.consolidate(
        [_mention("person", "Alice"), _mention("invoice number", "INV-1")]
    )
    assert {m.label for m in result.mentions} == {"person", "invoice number"}


# -- determinism + the off-by-default guarantee ------------------------------


def test_identical_inputs_give_an_identical_vocabulary() -> None:
    """Determinism is part of the contract — a seeded run must be byte-reproducible."""
    mentions = [
        _mention("company", "Globex"),
        _mention("org", "Acme"),
        _mention("organization", "IBM"),
        _mention("person", "Alice"),
    ]
    first = CascadeTypeConsolidator({}).consolidate(mentions)
    second = CascadeTypeConsolidator({}).consolidate(list(reversed(mentions)))
    assert first.vocabulary is not None
    assert second.vocabulary is not None
    assert first.vocabulary.model_dump_json() == second.vocabulary.model_dump_json()


def test_a_run_that_consolidates_nothing_returns_the_records_untouched() -> None:
    """No fold ⇒ byte-identical records, so an inert Stage cannot perturb downstream output."""
    mentions = [_mention("person", "Alice"), _mention("invoice number", "INV-1")]
    result = CascadeTypeConsolidator({}).consolidate(mentions)
    assert [m.model_dump_json() for m in result.mentions] == [
        m.model_dump_json() for m in mentions
    ]
    assert result.vocabulary is not None
    assert result.vocabulary.alias_map(TypeKind.ENTITY) == {}


def test_an_empty_corpus_yields_an_empty_vocabulary_not_a_crash() -> None:
    result = CascadeTypeConsolidator({}).consolidate([], [])
    assert result.mentions == []
    assert result.vocabulary is not None
    assert result.vocabulary.types == []


# -- mergeability for incremental delta runs (T4) ----------------------------


def test_a_delta_run_aliases_a_new_type_into_the_existing_cluster_without_renaming_it() -> None:
    """Append-don't-churn: a later run must not rename a canonical type already in the KG.

    The first run's corpus is dominated by ``organization``, so that wins the salience vote. The
    delta's corpus is dominated by ``org`` — which, re-elected from scratch, would rename the type
    and silently retype every node already assembled. Seeding forbids that.
    """
    first = CascadeTypeConsolidator({}).consolidate(
        [_mention("organization", "IBM"), _mention("organization", "Acme"), _mention("org", "X")]
    )
    assert first.vocabulary is not None
    assert first.vocabulary.of_kind(TypeKind.ENTITY)[0].canonical == "organization"

    delta = CascadeTypeConsolidator({})
    delta.seed_with(first.vocabulary)
    result = delta.consolidate(
        [_mention("org", "Y"), _mention("org", "Z"), _mention("orgs", "W")]
    )
    assert result.vocabulary is not None
    entity_types = result.vocabulary.of_kind(TypeKind.ENTITY)
    assert len(entity_types) == 1
    assert entity_types[0].canonical == "organization", "an existing canonical name must survive"
    assert {m.label for m in result.mentions} == {"organization"}


def test_a_drifted_surface_first_seen_on_the_delta_folds_into_the_committed_cluster() -> None:
    """The T4 case a post-hoc merge cannot reach: no shared label between seed and delta.

    The base corpus only ever wrote ``organization``; the delta only ever writes ``org``. The two
    vocabularies share no label, so folding them AFTER the fact (which can only match on label
    overlap) would append ``org`` as a SECOND canonical entity type and the KG would drift exactly
    where T4 promises it cannot. The seed must therefore take part in the CLUSTERING, not only in
    a post-hoc merge.
    """
    seed = CascadeTypeConsolidator({}).consolidate(
        [_mention("organization", "IBM"), _mention("organization", "Acme")]
    ).vocabulary
    assert seed is not None
    assert seed.of_kind(TypeKind.ENTITY)[0].aliases == [], "the seed never saw ``org``"

    delta = CascadeTypeConsolidator({})
    delta.seed_with(seed)
    result = delta.consolidate([_mention("org", "Globex"), _mention("org", "Initech")])
    assert result.vocabulary is not None
    entity_types = result.vocabulary.of_kind(TypeKind.ENTITY)
    assert [t.canonical for t in entity_types] == ["organization"], (
        "``org`` must alias into the committed cluster, not become a second canonical type"
    )
    assert entity_types[0].aliases == ["org"]
    assert {m.label for m in result.mentions} == {"organization"}


def test_a_committed_cluster_stays_clustered_when_the_rule_that_built_it_is_gone() -> None:
    """A persisted fold is a decision, not a rule application — a later run must not undo it."""
    seed = CascadeTypeConsolidator({"aliases": {"company": "organization"}}).consolidate(
        [_mention("organization", "IBM"), _mention("company", "Acme")]
    ).vocabulary
    assert seed is not None
    assert seed.of_kind(TypeKind.ENTITY)[0].aliases == ["company"]

    # The operator alias is gone from config; without the replay, ``company`` would split back out
    # and every node typed ``organization`` because of it would be retyped.
    delta = CascadeTypeConsolidator({})
    delta.seed_with(seed)
    result = delta.consolidate([_mention("company", "Globex")])
    assert result.vocabulary is not None
    assert [t.canonical for t in result.vocabulary.of_kind(TypeKind.ENTITY)] == ["organization"]
    assert {m.label for m in result.mentions} == {"organization"}


def test_merging_a_genuinely_new_type_appends_it() -> None:
    seed = CascadeTypeConsolidator({}).consolidate([_mention("organization", "IBM")]).vocabulary
    assert seed is not None
    delta = CascadeTypeConsolidator({})
    delta.seed_with(seed)
    result = delta.consolidate([_mention("statute", "GDPR")])
    assert result.vocabulary is not None
    assert {t.canonical for t in result.vocabulary.of_kind(TypeKind.ENTITY)} == {
        "organization",
        "statute",
    }


def test_the_seed_vocabulary_arrives_through_config_not_only_through_seed_with() -> None:
    """T4: the DeltaRunner injects the committed vocabulary as Stage CONFIG.

    The Pipeline is the run's whole input — a seed passed out-of-band would not appear in the
    persisted manifest, so replaying the YAML would canonicalize against a different vocabulary.
    """
    seed = CascadeTypeConsolidator({}).consolidate(
        [_mention("organization", "IBM"), _mention("organization", "Acme")]
    ).vocabulary
    assert seed is not None
    delta = CascadeTypeConsolidator({"seed_vocabulary": seed.model_dump(mode="json")})
    result = delta.consolidate([_mention("org", "Y"), _mention("org", "Z")])
    assert result.vocabulary is not None
    assert [t.canonical for t in result.vocabulary.of_kind(TypeKind.ENTITY)] == ["organization"]
    assert {m.label for m in result.mentions} == {"organization"}


def test_a_malformed_seed_vocabulary_fails_loudly_instead_of_canonicalizing_from_scratch() -> None:
    """ADR-0034: silently dropping the seed would re-elect (and rename) committed type labels."""
    with pytest.raises(ConfigError, match="seed_vocabulary"):
        CascadeTypeConsolidator({"seed_vocabulary": "organization"})
    with pytest.raises(ConfigError, match="seed_vocabulary"):
        CascadeTypeConsolidator({"seed_vocabulary": {"types": [{"kind": "nonsense"}]}})


def test_applying_the_same_delta_twice_reports_the_same_mention_counts() -> None:
    """T4 idempotency: a seeded merge must not accumulate the SAME corpus's mentions twice.

    The consolidation Stage is corpus-level, so on every delta it rebuilds its vocabulary over the
    WHOLE live corpus — the same mentions the seed already counted. Summing the two would report a
    multiple of the true salience on the Quality Report and would keep growing with every delta;
    re-applying an identical delta would report different numbers than the first application.
    """
    corpus = [_mention("organization", "IBM"), _mention("organization", "Acme")]
    first = CascadeTypeConsolidator({}).consolidate(corpus).vocabulary
    assert first is not None
    assert first.of_kind(TypeKind.ENTITY)[0].mention_count == 2

    second = CascadeTypeConsolidator(
        {"seed_vocabulary": first.model_dump(mode="json")}
    ).consolidate(corpus).vocabulary
    assert second is not None
    assert second.of_kind(TypeKind.ENTITY)[0].mention_count == 2, (
        "re-observing the same corpus must not double the count"
    )
    third = CascadeTypeConsolidator(
        {"seed_vocabulary": second.model_dump(mode="json")}
    ).consolidate(corpus).vocabulary
    assert third == second, "the same delta applied again yields the identical vocabulary"


def test_a_vocabulary_round_trips_through_json() -> None:
    """It is persisted as a file between runs, so serialization is part of the contract."""
    built = CascadeTypeConsolidator({}).consolidate(
        [_mention("org", "Acme"), _mention("organization", "IBM")]
    ).vocabulary
    assert built is not None
    restored = TypeVocabulary.model_validate_json(built.model_dump_json())
    assert restored.alias_map(TypeKind.ENTITY) == built.alias_map(TypeKind.ENTITY)


def test_a_label_cannot_belong_to_two_clusters() -> None:
    """The alias->canonical map must be a function, or the remap is ambiguous."""
    from latence_core.contracts import CanonicalType

    with pytest.raises(ValueError, match="more than one cluster"):
        TypeVocabulary(
            types=[
                CanonicalType(kind=TypeKind.ENTITY, canonical="org", aliases=["company"]),
                CanonicalType(kind=TypeKind.ENTITY, canonical="firm", aliases=["company"]),
            ]
        )


# -- typed failure, never a silent fallback (ADR-0034) -----------------------


def test_a_misnamed_embedder_provider_fails_loudly() -> None:
    provider = CascadeTypeConsolidator(
        {"use_embedding_merge": True, "embedder": {"provider": "no.such.embedder"}}
    )
    with pytest.raises(Exception, match="no.such.embedder"):
        provider.consolidate([_mention("org", "Acme"), _mention("company", "Globex")])


def test_a_non_embedder_provider_named_as_the_embedder_fails_loudly() -> None:
    provider = CascadeTypeConsolidator(
        {"use_embedding_merge": True, "embedder": {"provider": "chunk.markdown"}}
    )
    with pytest.raises(ConfigError, match="Embedder capability"):
        provider.consolidate([_mention("org", "Acme"), _mention("company", "Globex")])


def test_a_malformed_alias_map_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="aliases"):
        CascadeTypeConsolidator({"aliases": {"org": 7}})


def test_an_embedder_that_breaks_its_contract_is_rejected() -> None:
    """A short vector list means the labels and vectors no longer correspond — fail, don't guess."""

    class _ShortEmbedder:
        @property
        def dimension(self) -> int:
            return 3

        def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
            list(texts)
            return iter([[1.0, 0.0, 0.0]])

    provider = CascadeTypeConsolidator({"use_embedding_merge": True})
    provider._embedder = _ShortEmbedder()  # noqa: SLF001
    with pytest.raises(ValueError, match="one vector per input"):
        provider.consolidate([_mention("org", "Acme"), _mention("company", "Globex")])

"""The three RED^FM suites end-to-end, against the REAL built-in Providers.

Nothing is stubbed: the dataset root is the real on-disk layout, the Stages are resolved from
the real ``latence.providers`` entry-point registry and instantiated the way the Runner
instantiates them, and the scores are computed from what those Providers actually emitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from latence_benchmark.providers import ProviderUnavailableError
from latence_benchmark.redfm_suites import (
    DisambiguationSuite,
    EntitySuite,
    RelationSuite,
)
from latence_benchmark.suite import StackPin
from redfm_fixtures import redfm_entity, redfm_row

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Two documents sharing three entities under identical surfaces (so a surface-based resolver
# CAN link them), one LOC→LOC gold relation each, and one numeric entity so the typed match
# criterion has something reachable to score.
TEXT_A = "Ada Lovelace worked in London in England for 12 years."
# The leading adverb is deliberate: the pinned capitalisation recogniser tags it, so the run has
# a real false positive and the perfect-recall assertion cannot be mistaken for a free pass.
TEXT_B = "Later, London in England hosted Ada Lovelace."


def _english_rows() -> list[dict[str, Any]]:
    ada_a = redfm_entity(TEXT_A, "Ada Lovelace", "PER", "Q7259")
    london_a = redfm_entity(TEXT_A, "London", "LOC", "Q84")
    england_a = redfm_entity(TEXT_A, "England", "LOC", "Q21")
    years_a = redfm_entity(TEXT_A, "12", "NUMBER", "Q1")
    london_b = redfm_entity(TEXT_B, "London", "LOC", "Q84")
    england_b = redfm_entity(TEXT_B, "England", "LOC", "Q21")
    ada_b = redfm_entity(TEXT_B, "Ada Lovelace", "PER", "Q7259")
    return [
        redfm_row(
            "doc-a",
            TEXT_A,
            [ada_a, london_a, england_a, years_a],
            [(london_a, "location", england_a)],
        ),
        redfm_row(
            "doc-b",
            TEXT_B,
            [london_b, england_b, ada_b],
            [(london_b, "location", england_b)],
        ),
    ]


@pytest.fixture
def dataset(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> Path:
    return write_redfm(tmp_path, {"en": _english_rows()})


# -- Entity Extraction -------------------------------------------------------------------------


def test_the_real_entity_stage_is_scored_against_gold_spans(dataset: Path) -> None:
    result = EntitySuite(languages=("en",)).run(dataset)

    assert result.stage == "entity_extraction"
    assert result.documents_scored == 2
    (english,) = result.metrics_for("language")
    # Seven gold spans across the two documents; the pinned recogniser finds every one.
    assert english.gold == 7
    assert english.recall == 1.0
    # ...and that is not a free pass: the same run has a false positive, because a
    # capitalisation rule also tags the sentence-initial adverb.
    assert english.false_positives > 0


def test_both_match_criteria_are_published_side_by_side(dataset: Path) -> None:
    """A boundaries-only score alone would hide that the pinned Provider barely types entities;
    a typed-only score that is structurally zero could never detect a regression either way."""
    criteria = {m.name: m for m in EntitySuite(languages=("en",)).run(dataset).metrics_for("match")}
    assert set(criteria) == {"span (boundaries only)", "span+type"}
    boundaries = criteria["span (boundaries only)"]
    typed = criteria["span+type"]
    assert typed.gold == boundaries.gold
    # The NUMBER recogniser genuinely agrees with the corpus on one span; the capitalisation
    # recogniser's generic ENTITY label agrees with none.
    assert 0 < typed.true_positives < boundaries.true_positives


def test_the_entity_headline_is_the_language_micro(dataset: Path) -> None:
    result = EntitySuite(languages=("en",)).run(dataset)
    headline = result.headline
    assert headline is not None
    assert headline.dimension == "language"
    assert headline.gold == 7


# -- Relation Extraction -----------------------------------------------------------------------


def test_relation_extraction_is_scored_from_gold_mentions(dataset: Path) -> None:
    """The Stage is handed the corpus's own mentions, so the number is not capped by NER."""
    result = RelationSuite(languages=("en",)).run(dataset)

    assert result.stage == "relation_extraction"
    (english,) = result.metrics_for("language")
    # One in-scope gold triple per document (`location`, LOC → LOC).
    assert english.gold == 2
    assert english.true_positives == 2


def test_out_of_scope_predicates_are_not_charged_as_misses(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """The pin declares five predicates. Counting a sixth as a false negative would charge the
    Stage for a label it was never asked to find, which is not a measurement."""
    ada = redfm_entity(TEXT_A, "Ada Lovelace", "PER", "Q7259")
    london = redfm_entity(TEXT_A, "London", "LOC", "Q84")
    root = write_redfm(
        tmp_path,
        {"en": [redfm_row("doc-a", TEXT_A, [ada, london], [(ada, "spouse", london)])]},
    )
    (english,) = RelationSuite(languages=("en",)).run(root).metrics_for("language")
    assert english.gold == 0
    assert english.false_negatives == 0


def test_per_predicate_slices_cover_every_pinned_rule(dataset: Path) -> None:
    result = RelationSuite(languages=("en",)).run(dataset)
    names = {m.name for m in result.metrics_for("predicate")}
    assert names == set(result.stack_pin.config["relations"])


# -- Disambiguation ----------------------------------------------------------------------------


def test_disambiguation_is_scored_pairwise_against_gold_kb_identity(dataset: Path) -> None:
    result = DisambiguationSuite(languages=("en",)).run(dataset)

    assert result.stage == "disambiguation"
    (english,) = result.metrics_for("language")
    # Three gold pairs: the repeated Ada Lovelace, London and England mentions. The
    # surface-based cascade merges all three, so recall is total.
    assert english.gold == 3
    assert english.recall == 1.0


def test_unlinked_mentions_are_excluded_from_both_sides(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """Two mentions the corpus could not link are not "the same entity". Treating RED^FM's
    literal ``uri == "None"`` as an identity would invent a huge spurious gold cluster."""
    unlinked_a = redfm_entity(TEXT_A, "London", "LOC", "None")
    unlinked_b = redfm_entity(TEXT_B, "London", "LOC", "None")
    root = write_redfm(
        tmp_path,
        {
            "en": [
                redfm_row("doc-a", TEXT_A, [unlinked_a]),
                redfm_row("doc-b", TEXT_B, [unlinked_b]),
            ]
        },
    )
    (english,) = DisambiguationSuite(languages=("en",)).run(root).metrics_for("language")
    assert english.gold == 0
    assert english.false_positives == 0


def test_a_wrong_merge_is_charged_as_a_false_positive(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """Same surface, DIFFERENT gold entity: a surface-based resolver merges them and must be
    charged for it. Without this the suite could not tell over-merging from success."""
    text_a = "Cambridge is in England."
    text_b = "Cambridge is in Massachusetts."
    uk = redfm_entity(text_a, "Cambridge", "LOC", "Q350")
    us = redfm_entity(text_b, "Cambridge", "LOC", "Q49111")
    other_a = redfm_entity(text_a, "England", "LOC", "Q21")
    other_b = redfm_entity(text_b, "Massachusetts", "LOC", "Q771")
    root = write_redfm(
        tmp_path,
        {
            "en": [
                redfm_row("doc-a", text_a, [uk, other_a]),
                redfm_row("doc-b", text_b, [us, other_b]),
            ]
        },
    )
    (english,) = DisambiguationSuite(languages=("en",)).run(root).metrics_for("language")
    assert english.gold == 0
    assert english.false_positives == 1


# -- shared discipline -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suite",
    [EntitySuite(languages=("en",)), RelationSuite(languages=("en",)), DisambiguationSuite(languages=("en",))],
)
def test_two_runs_of_the_same_pin_are_identical(suite: Any, dataset: Path) -> None:
    assert suite.run(dataset).as_dict() == suite.run(dataset).as_dict()


@pytest.mark.parametrize(
    "suite",
    [EntitySuite(languages=("en",)), RelationSuite(languages=("en",)), DisambiguationSuite(languages=("en",))],
)
def test_a_capped_run_is_labelled_as_a_subset(suite: Any, dataset: Path) -> None:
    notes = suite.run(dataset, limit=1).notes
    assert any(note.startswith("SUBSET RUN") for note in notes)


def test_an_uninstalled_provider_is_named_never_silently_substituted(dataset: Path) -> None:
    suite = EntitySuite(
        languages=("en",),
        stack_pin=StackPin(stage="entity_extraction", provider="entity.nonexistent"),
    )
    with pytest.raises(ProviderUnavailableError, match="entity.nonexistent"):
        suite.run(dataset)


def test_the_dataset_licence_travels_with_every_result(dataset: Path) -> None:
    payload = EntitySuite(languages=("en",)).run(dataset).as_dict()
    assert payload["dataset"]["license"]["dataset_license"] == "CC-BY-SA-4.0"
    assert payload["dataset"]["revision"] == "eae75dc9bde14d1172351423bcbb82c5b9a94f66"
    assert payload["headline"]["dimension"] == "language"

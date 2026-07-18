"""The RED^FM reader: it accepts the real shape and refuses everything else."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from latence_benchmark.redfm import REDFM, REDFM_LANGUAGES, load_language
from latence_benchmark.unittests import BenchmarkDataError
from redfm_fixtures import redfm_entity, redfm_row

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

TEXT = "Ada Lovelace worked in London for the Analytical Society."


def _row() -> dict[str, Any]:
    ada = redfm_entity(TEXT, "Ada Lovelace", "PER", "Q7259")
    london = redfm_entity(TEXT, "London", "LOC", "Q84")
    return redfm_row("doc-1", TEXT, [ada, london], [(ada, "location", london)])


def test_a_real_shaped_row_round_trips(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    root = write_redfm(tmp_path, {"en": [_row()]})
    (document,) = load_language(root, "en")

    assert document.docid == "doc-1"
    assert document.text == TEXT
    assert [e.surface for e in document.entities] == ["Ada Lovelace", "London"]
    assert document.entities[0].entity_type == "PER"
    assert document.entities[0].uri == "Q7259"
    (relation,) = document.relations
    assert (relation.predicate, relation.head.surface, relation.tail.surface) == (
        "location",
        "Ada Lovelace",
        "London",
    )


def test_boundaries_that_do_not_slice_the_surface_form_are_refused(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """Every span in the pinned release satisfies text[start:end] == surfaceform.

    A corpus whose text and annotations have drifted apart would otherwise be scored, and the
    resulting number would measure the misalignment rather than the Stage.
    """
    row = _row()
    row["entities"][0]["boundaries"] = [1, 5]
    root = write_redfm(tmp_path, {"en": [row]})
    with pytest.raises(BenchmarkDataError, match="out of alignment"):
        load_language(root, "en")


def test_start_end_keys_are_not_accepted_in_place_of_boundaries(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """The Hub's *parquet* view of RED^FM exposes start/end; the JSONL this harness reads does
    not. Silently accepting the other shape would let a differently-materialised copy score."""
    row = _row()
    entity = row["entities"][0]
    entity["start"], entity["end"] = entity.pop("boundaries")
    root = write_redfm(tmp_path, {"en": [row]})
    with pytest.raises(BenchmarkDataError, match="boundaries"):
        load_language(root, "en")


def test_a_row_missing_a_required_key_is_refused(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    row = _row()
    del row["relations"]
    root = write_redfm(tmp_path, {"en": [row]})
    with pytest.raises(BenchmarkDataError, match="relations"):
        load_language(root, "en")


def test_a_zero_length_annotation_is_not_mentionable(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """The pinned release has two of these. They are read faithfully, but the suites exclude
    them — a zero-length span is not a mention any extractor could emit."""
    row = _row()
    row["entities"].append(
        {"uri": "Q1", "boundaries": [4, 4], "surfaceform": "", "annotator": "Me", "type": "UNK"}
    )
    root = write_redfm(tmp_path, {"en": [row]})
    (document,) = load_language(root, "en")
    assert [e.mentionable for e in document.entities] == [True, True, False]


def test_an_unlinkable_mention_is_not_an_identity(
    tmp_path: Path,
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """RED^FM writes the literal string "None" for a mention it could not link."""
    row = _row()
    row["entities"][1]["uri"] = "None"
    root = write_redfm(tmp_path, {"en": [row]})
    (document,) = load_language(root, "en")
    assert [e.linked for e in document.entities] == [True, False]


def test_a_missing_language_file_names_the_pinned_revision(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkDataError, match=REDFM.revision):
        load_language(tmp_path, "de")


def test_an_unknown_language_lists_the_available_ones(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkDataError, match="Available"):
        load_language(tmp_path, "sv")


def test_the_pinned_dataset_declares_a_german_slice_and_a_verified_licence() -> None:
    """The P1 EU-language rider, and the license gate that let this corpus in at all."""
    assert "de" in REDFM_LANGUAGES
    assert REDFM.license.dataset_license == "CC-BY-SA-4.0"
    assert REDFM.license.code_license == "Apache-2.0"
    assert REDFM.license.commercial_use is True
    assert REDFM.license.verified
    assert REDFM.manifest == tuple(f"data/test.{lang}.jsonl" for lang in REDFM_LANGUAGES)

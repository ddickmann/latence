"""The Redaction suite, the shipped synthetic corpus, and the promise that it holds no real PII.

The Stage under test is the real ``redaction.hybrid_rule`` Provider at its default config —
the one an adopter gets — resolved through the real entry-point registry.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest
from latence_benchmark.corpora.build_synthetic_pii import build_corpus
from latence_benchmark.datasets import DatasetUnavailableError, ShippedCorpusSource, content_digest
from latence_benchmark.metrics import micro, total
from latence_benchmark.redaction_suite import (
    CORPUS_FILE,
    SYNTHETIC_PII,
    RedactionSuite,
    _covers,
    coverage,
    load_corpus,
)
from latence_benchmark.unittests import BenchmarkDataError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def corpus_root() -> Path:
    return ShippedCorpusSource().materialise(SYNTHETIC_PII)


# -- the corpus itself -------------------------------------------------------------------------


def test_the_shipped_corpus_matches_its_pinned_content_digest(corpus_root: Path) -> None:
    """The pin is the whole point: a corpus is a text file anyone can edit, and without this
    check a one-character change would move every published number under an unchanged
    "revision"."""
    assert content_digest(corpus_root, SYNTHETIC_PII) == SYNTHETIC_PII.revision
    assert SYNTHETIC_PII.revision_kind == "content-digest"


def test_an_edited_corpus_is_refused_rather_than_scored(tmp_path: Path) -> None:
    (tmp_path / CORPUS_FILE).write_text(
        (ShippedCorpusSource().materialise(SYNTHETIC_PII) / CORPUS_FILE).read_text(
            encoding="utf-8"
        )
        + '{"doc_id": "extra", "language": "en", "sensitivity": "internal", '
        '"text": "hello", "pii": []}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetUnavailableError, match="content digest"):
        ShippedCorpusSource(tmp_path).materialise(SYNTHETIC_PII)


def test_the_shipped_corpus_is_exactly_what_its_generator_produces(corpus_root: Path) -> None:
    """The corpus's auditability rests on its generator being the source of truth. If the file
    could drift from it, "every value is a reserved example" would be unverifiable."""
    assert (corpus_root / CORPUS_FILE).read_text(encoding="utf-8") == build_corpus()


def test_the_corpus_contains_no_value_from_outside_the_reserved_example_ranges(
    corpus_root: Path,
) -> None:
    """A structural no-real-PII guard, not a promise in a docstring.

    Every e-mail domain must be an RFC 2606 reserved one, every IPv4 must be in a RFC 5737
    TEST-NET block, and every NANP phone number must use the 555 fiction range.
    """
    documents = load_corpus(corpus_root)
    for document in documents:
        for span in document.pii:
            if span.pii_type == "email":
                assert re.search(r"@example\.(com|org|net|de)$", span.value), span.value
            if span.pii_type == "ip":
                assert span.value.startswith(("192.0.2.", "198.51.100.", "203.0.113.")), span.value
            if span.pii_type == "phone" and span.value.startswith("+1"):
                assert "555" in span.value, span.value


def test_the_corpus_has_both_an_english_and_a_german_slice(corpus_root: Path) -> None:
    """The P1 EU-language rider: the privacy number is published per language."""
    languages = {document.language for document in load_corpus(corpus_root)}
    assert languages == {"en", "de"}


def test_the_corpus_plants_pii_the_pinned_provider_cannot_detect(corpus_root: Path) -> None:
    """A gold set containing only what the redactor already catches would report 100% recall
    forever and detect no regression. Names and addresses are the deliberate failable part."""
    types = {span.pii_type for document in load_corpus(corpus_root) for span in document.pii}
    assert {"person_name", "postal_address"} <= types


def test_the_corpus_has_clean_controls(corpus_root: Path) -> None:
    """Without documents that contain no PII at all, precision would be unmeasurable."""
    assert any(not document.pii for document in load_corpus(corpus_root))


def test_a_gold_offset_that_does_not_slice_its_value_is_refused(tmp_path: Path) -> None:
    (tmp_path / CORPUS_FILE).write_text(
        json.dumps(
            {
                "doc_id": "bad",
                "language": "en",
                "sensitivity": "internal",
                "text": "Mail a@example.com now.",
                "pii": [{"pii_type": "email", "value": "a@example.com", "start": 0, "end": 4}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkDataError, match="Regenerate the corpus"):
        load_corpus(tmp_path)


# -- the scoring rule --------------------------------------------------------------------------


def test_coverage_credits_a_wider_mask_but_not_a_partial_one() -> None:
    """The privacy question is "was the value masked", not "did the boundaries agree"."""
    gold = [("d", "ssn", 10, 21)]
    wider = coverage("match", "c", gold=gold, detected=[("d", "phone", 8, 25)])
    partial = coverage("match", "c", gold=gold, detected=[("d", "ssn", 10, 15)])
    assert wider.recall == 1.0
    assert partial.recall == 0.0


def test_a_detection_on_clean_text_is_a_false_positive() -> None:
    score = coverage("match", "c", gold=[], detected=[("d", "phone", 0, 9)])
    assert score.false_positives == 1
    assert score.precision == 0.0


def test_a_boundary_disagreement_is_not_also_charged_as_a_false_alarm() -> None:
    """Charging the same detection twice — once as a miss, once as a false alarm — would
    understate precision on exactly the cases where the value WAS masked."""
    score = coverage("match", "c", gold=[("d", "ssn", 10, 21)], detected=[("d", "ssn", 8, 25)])
    assert (score.true_positives, score.false_positives) == (1, 0)


def test_one_mask_cannot_claim_more_than_one_gold_value() -> None:
    """The mixed-basis hole: crediting a detection once per value it happens to swallow means
    a wider mask is credited without ever being charged, so over-masking costs nothing."""
    gold = [("d", "ssn", 10, 21), ("d", "email", 40, 55), ("d", "phone", 70, 82)]
    one_big_mask = coverage("match", "c", gold=gold, detected=[("d", "person_name", 0, 200)])
    assert one_big_mask.true_positives == 1
    assert one_big_mask.false_negatives == 2
    # ...while a redactor that actually localises the three values is fully credited.
    precise = coverage("match", "c", gold=gold, detected=list(gold))
    assert (precise.true_positives, precise.false_positives, precise.false_negatives) == (3, 0, 0)


def test_masking_the_whole_document_scores_worse_than_the_real_redactor(
    corpus_root: Path,
) -> None:
    """The headline must be able to see over-redaction — the largest failure mode of a redactor.

    A detector that masks every document in full destroys 100% of corpus utility. If it can
    outrank the pinned Provider on the published number, the benchmark backing the framework's
    central privacy claim is measuring nothing.
    """
    documents = load_corpus(corpus_root)
    languages = dict.fromkeys(document.language for document in documents)
    slices = []
    for language in languages:
        in_language = [d for d in documents if d.language == language]
        gold = [(d.doc_id, s.pii_type, s.start, s.end) for d in in_language for s in d.pii]
        detected = [(d.doc_id, "person_name", 0, len(d.text)) for d in in_language]
        # The temptation, stated structurally: every planted value physically IS inside a mask.
        assert all(any(_covers(d, g) for d in detected) for g in gold)
        slices.append(coverage("language", language, gold=gold, detected=detected))
    mask_everything = micro(slices, dimension="language")

    real = RedactionSuite().run(corpus_root).headline
    assert real is not None and real.f1 is not None
    assert mask_everything.f1 is not None
    assert mask_everything.f1 < real.f1


# -- the suite end-to-end ----------------------------------------------------------------------


def test_the_real_redaction_stage_masks_most_of_the_gold_set(corpus_root: Path) -> None:
    result = RedactionSuite().run(corpus_root)
    headline = result.headline
    assert headline is not None
    # The structured types are all caught, so recall is high...
    assert headline.recall is not None
    assert headline.recall > 0.7
    # ...but NOT perfect, because names and addresses are planted and cannot be caught by a
    # rule-based redactor. A benchmark asserting 1.0 here would be measuring itself.
    assert headline.recall < 1.0
    # And it raises no false alarm on the clean controls.
    assert headline.precision == 1.0


def test_person_names_are_reported_as_the_residual_not_hidden(corpus_root: Path) -> None:
    by_type = {m.name: m for m in RedactionSuite().run(corpus_root).metrics_for("pii_type")}
    assert by_type["person_name"].recall == 0.0
    assert by_type["email"].recall == 1.0
    assert by_type["iban"].recall == 1.0
    assert by_type["aws_key"].recall == 1.0


def test_a_detected_type_outside_the_gold_inventory_is_still_charged_as_a_false_positive(
    tmp_path: Path,
) -> None:
    """The `pii_type` dimension is rendered with a bold **MICRO** aggregate, so it must charge
    EVERY detection the run emitted — including one labelled with a type the gold set never
    names. Slicing only over the gold types drops those detections out of the table entirely and
    publishes a precision that no run reproduces.

    The corpus here is one document whose gold set holds a single `email`, in text that also
    contains an RFC 5737 TEST-NET address. The pinned Provider emits both, `ip` is not a gold
    type, and the over-redaction must survive into the aggregate exactly as the `language`
    dimension already reports it.
    """
    value = "erika@example.com"
    text = f"Contact {value} from host 192.0.2.44 today."
    start = text.index(value)
    (tmp_path / CORPUS_FILE).write_text(
        json.dumps(
            {
                "doc_id": "type-fp-1",
                "language": "en",
                "sensitivity": "internal",
                "text": text,
                "pii": [
                    {
                        "pii_type": "email",
                        "value": value,
                        "start": start,
                        "end": start + len(value),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = RedactionSuite().run(tmp_path)

    # What the run really did, per the dimension that was never type-scoped:
    (language,) = result.metrics_for("language")
    assert (language.true_positives, language.false_positives) == (1, 1)

    by_type = {m.name: m for m in result.metrics_for("pii_type")}
    # The out-of-inventory detection gets its own slice: nothing to recall, one false alarm.
    assert "ip" in by_type, "a detected type absent from the gold set vanished from the table"
    assert (by_type["ip"].true_positives, by_type["ip"].false_positives) == (0, 1)
    assert by_type["ip"].precision == 0.0
    # An undefined ratio is reported as undefined, never as a measured 0%.
    assert by_type["ip"].recall is None

    # And the published aggregate agrees with the run instead of flattering it.
    aggregate = total(result.metrics_for("pii_type"), dimension="pii_type")
    assert (aggregate.true_positives, aggregate.false_positives) == (1, 1)
    assert aggregate.precision == 0.5


def test_a_value_masked_under_the_wrong_label_is_visible_as_such(corpus_root: Path) -> None:
    """A date of birth caught by the phone recogniser IS masked — but a policy scoped to
    ``types: ["date_of_birth"]`` would have missed it, so the two criteria must differ."""
    result = RedactionSuite().run(corpus_root)
    criteria = {m.name: m for m in result.metrics_for("match")}
    covered = criteria["coverage (value fully masked)"]
    typed = criteria["exact span+type"]
    assert typed.true_positives < covered.true_positives


def test_the_run_is_deterministic(corpus_root: Path) -> None:
    assert RedactionSuite().run(corpus_root).as_dict() == RedactionSuite().run(corpus_root).as_dict()


def test_the_pin_uses_the_default_comprehensive_detector_set() -> None:
    """An empty config selects every built-in recogniser. Pinning a narrowed `detectors` list
    would publish a number no adopter reproduces."""
    assert RedactionSuite().stack_pin.config == {}
    assert RedactionSuite().stack_pin.provider == "redaction.hybrid_rule"

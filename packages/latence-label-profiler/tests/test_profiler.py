"""The profiler end to end: determinism, the (1 - 1/e) guarantee on a KNOWN-optimal corpus,
the uncertainty the artifact is required to report, and the shape the downstream inducer needs.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any

from latence_label_profiler import CorpusProfile, CorpusProfiler, ProfilerConfig
from latence_label_profiler.coverage import WORST_CASE_RATIO

# A synthetic corpus whose optimal 2-type cover is KNOWN by construction.
#
# Four surfaces, four masses: "invoice number" x5, "order number" x4, "invoice date" x4,
# "order date" x3 -> total mention mass 16. The head types {number, date} cover all 16, as do
# the modifier types {invoice, order}. No single type exceeds 9, so no pair can beat 16 and
# 16 IS the optimum at N = 2. The test brute-forces that rather than trusting this comment.
_KNOWN_CORPUS = (
    ["The invoice number was recorded."] * 5
    + ["The order number was recorded."] * 4
    + ["The invoice date was recorded."] * 4
    + ["The order date was recorded."] * 3
)

_MIXED_CORPUS = [
    "The commodity is classified under the tariff heading of the customs schedule.",
    "Each commodity code maps to a tariff heading in the combined nomenclature.",
    "Contact the compliance officer at jane.doe@example.com for the binding ruling.",
    "The customs declaration references the tariff heading and the commodity code.",
    "The compliance officer reviewed the export licence and the customs declaration.",
    "The binding ruling was issued by the customs authority on the tariff heading.",
    "An export licence is required for the dual use commodity code.",
    "The customs authority published the combined nomenclature update.",
    "Send the export licence application to compliance@example.org before the deadline.",
    "The dual use item was classified under the export control list.",
]


def _profile(texts: list[str], **overrides: Any) -> CorpusProfile:
    return CorpusProfiler(ProfilerConfig(**overrides)).profile(texts)


# -- acceptance 1: determinism ------------------------------------------------


def test_same_corpus_and_seed_produce_byte_identical_output() -> None:
    first = _profile(list(_MIXED_CORPUS), seed=7).to_json()
    second = _profile(list(_MIXED_CORPUS), seed=7).to_json()
    assert first == second
    assert isinstance(first, str)


def test_sampling_is_seeded_and_reproducible() -> None:
    corpus = _MIXED_CORPUS * 20
    a = _profile(corpus, seed=3, sample_size=17).to_json()
    b = _profile(corpus, seed=3, sample_size=17).to_json()
    assert a == b
    c = _profile(corpus, seed=4, sample_size=17)
    assert c.corpus.sample_seed == 4


def test_the_artifact_is_valid_json_and_records_its_own_inputs() -> None:
    profile = _profile(list(_MIXED_CORPUS), seed=1)
    payload = json.loads(profile.to_json())
    assert payload["artifact_version"] == 1
    assert payload["corpus"]["texts_seen"] == len(_MIXED_CORPUS)
    assert payload["corpus"]["sample_seed"] == 1


# -- acceptance 2: greedy is within (1 - 1/e) on a known-optimal corpus -------


def test_greedy_lands_within_one_minus_one_over_e_on_a_known_optimal_corpus() -> None:
    profile = _profile(_KNOWN_CORPUS, seed=0, min_mention_mass=1, min_document_frequency=2)
    problem = profile.problem("entity_types")
    selection = profile.slots["entity_types"]

    budget = 2
    optimum = 0.0
    for combo in combinations(sorted(problem.sets), budget):
        covered: set[str] = set()
        for name in combo:
            covered |= problem.sets[name]
        optimum = max(optimum, sum(problem.weights[e] for e in covered))

    assert optimum == 16.0, "the corpus no longer has the optimum this test was built around"
    greedy_at_2 = selection.result.steps[budget - 1].cumulative
    assert greedy_at_2 >= WORST_CASE_RATIO * optimum
    assert greedy_at_2 == optimum  # greedy is exact here; the bound is the floor, not the claim


def test_the_certificate_is_a_real_bound_and_never_exceeds_the_universe() -> None:
    profile = _profile(list(_MIXED_CORPUS), seed=0, min_mention_mass=1, min_document_frequency=1)
    entity = profile.slots["entity_types"]
    assert entity.result.certificate_bound >= entity.result.value
    assert entity.result.certificate_bound <= entity.result.total_weight
    # The certificate can only be WEAKER than the true optimality ratio (it bounds OPT from above),
    # so it must at minimum dominate the raw coverage fraction.
    assert entity.result.certificate_ratio >= entity.result.coverage_fraction - 1e-9


def test_the_certificate_is_exact_when_greedy_reaches_full_coverage() -> None:
    """The per-corpus certificate earning its keep: 'provably optimal', not 'within 63.2%'."""
    profile = _profile(_KNOWN_CORPUS, seed=0, min_mention_mass=1, min_document_frequency=2)
    entity = profile.slots["entity_types"]
    assert entity.result.coverage_fraction == 1.0
    assert entity.result.certificate_ratio == 1.0
    assert entity.result.certificate_ratio > WORST_CASE_RATIO


# -- acceptance 3: the artifact reports its own uncertainty --------------------


def test_the_artifact_carries_curve_certificate_coverage_and_good_turing() -> None:
    payload = json.loads(
        _profile(
            list(_MIXED_CORPUS), seed=0, min_mention_mass=1, min_document_frequency=1
        ).to_json()
    )
    entity = payload["slots"]["entity_types"]

    assert entity["marginal_gain_curve"], "the Pareto/gain curve is not optional"
    assert entity["coverage_curve"], "C(N) is not optional"
    gains = [point["marginal_gain_fraction"] for point in entity["marginal_gain_curve"]]
    assert gains == sorted(gains, reverse=True)

    cert = entity["certificate"]
    assert cert["worst_case_floor"] == WORST_CASE_RATIO
    assert cert["upper_bound_on_optimum"] >= cert["greedy_value"]
    assert 0.0 < cert["ratio_to_bound"] <= 1.0

    assert 0.0 <= entity["coverage"]["empirical_coverage"] <= 1.0
    assert "zipf" in entity
    assert entity["zipf"]["rank_slope"] is not None
    residual = entity["residual"]
    assert 0.0 <= residual["good_turing_unseen_mass"] <= 1.0
    assert 0.0 <= residual["empirical_uncovered_mass"] <= 1.0
    assert 0.0 <= residual["estimated_future_uncovered_mass"] <= 1.0


def test_a_logarithmic_corpus_is_reported_as_a_budget_decision_not_a_boundary() -> None:
    """The ticket's honesty bar: alpha ~ 1 -> no knee -> say N is a budget, do not imply a knee."""
    # Type names must be ALPHABETIC: the tokenizer treats any token containing a digit as an
    # instance and breaks the phrase on it, so "type001" would mine nothing at all.
    def _name(rank: int) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        return letters[rank // 26 % 26] + letters[rank % 26] + "kind"

    corpus: list[str] = []
    for rank in range(1, 120):
        occurrences = max(1, int(round(600 / rank)))
        corpus.extend([f"the {_name(rank)} was filed"] * occurrences)
    payload = json.loads(
        _profile(corpus, seed=0, min_mention_mass=1, min_document_frequency=1).to_json()
    )
    entity = payload["slots"]["entity_types"]
    assert entity["knee"]["sharp"] is False
    assert entity["knee"]["n_is_a_budget_decision"] is True
    assert "budget decision" in entity["knee"]["interpretation"]
    assert any("budget decision" in note for note in payload["notes"])


def test_the_artifact_reports_n_star_alongside_the_cap() -> None:
    payload = json.loads(
        _profile(
            list(_MIXED_CORPUS), seed=0, min_mention_mass=1, min_document_frequency=1
        ).to_json()
    )
    entity = payload["slots"]["entity_types"]
    assert entity["budget_n"] == 40
    assert "n_star" in entity["knee"]


def test_the_cap_is_never_exceeded() -> None:
    # 300 DISJOINT types: no shared head, so no small set of types can cover the corpus and greedy
    # genuinely runs into the cap instead of exhausting the universe first.
    letters = "abcdefghijklmnopqrstuvwxyz"
    corpus = [
        f"the {letters[i % 26]}{letters[i // 26 % 26]}wid was filed" for i in range(300)
    ] * 2
    profile = _profile(corpus, seed=0, max_types=12, min_mention_mass=1, min_document_frequency=1)
    candidates = profile.lexicon_candidates()
    assert len(candidates["entity_types"]) == 12, "the cap must actually bite in this test"
    for slot in candidates.values():
        assert len(slot) <= 12


# -- the downstream contract --------------------------------------------------


def test_lexicon_candidates_are_exactly_the_inducer_config_shape() -> None:
    from latence_schema_inducer.lexicon import LexiconLabelInducer

    profile = _profile(list(_MIXED_CORPUS), seed=0, min_mention_mass=1, min_document_frequency=1)
    candidates = profile.lexicon_candidates()

    assert set(candidates) <= {"entity_types", "relation_types", "pii_types"}
    for slot, mapping in candidates.items():
        assert isinstance(mapping, dict), slot
        for type_name, cues in mapping.items():
            assert isinstance(type_name, str) and type_name
            assert isinstance(cues, list) and cues
            assert all(isinstance(cue, str) and cue for cue in cues)

    # The decisive one: the REAL provider must accept it without a single config error.
    inducer = LexiconLabelInducer({"candidates": candidates, "granularity": "chunk"})
    assert inducer.induction_calls == 0


def test_the_pool_is_a_corpus_level_pool_not_a_per_chunk_schema() -> None:
    """D5: the profiler emits a POOL; narrowing is the inducer's job, per chunk."""
    from latence_core.contracts import ChunkRecord, Classification, Provenance
    from latence_schema_inducer.lexicon import LexiconLabelInducer

    profile = _profile(list(_MIXED_CORPUS), seed=0, min_mention_mass=1, min_document_frequency=1)
    candidates = profile.lexicon_candidates()
    pool = set(candidates.get("entity_types", {}))
    assert len(pool) >= 3, "the corpus must yield a pool worth narrowing"

    text = "An export licence is required for the dual use commodity code."
    chunk = ChunkRecord(
        record_id="doc-1#chunk-0",
        document_record_id="doc-1",
        index=0,
        content=text,
        token_count=16,
        provenance=Provenance(
            source_uri="mem://t/doc.txt",
            file_name="doc.txt",
            file_type="txt",
            file_size=len(text),
            document_id="sha256:doc-1",
            char_start=0,
            char_end=len(text),
        ),
        classification=Classification(language="en", category="memo", sensitivity="internal"),
    )
    induced = list(LexiconLabelInducer({"candidates": candidates}).induce([chunk]))[0]
    assert induced.induced_labels is not None
    got = set(induced.induced_labels.entity_types)
    assert got <= pool
    assert got < pool, "a chunk that induced the WHOLE pool is the defect D5a describes"


def test_no_pii_value_ever_reaches_the_emitted_candidate_space() -> None:
    corpus = [
        "Write to jane.doe@example.com about DE89370400440532013000 today.",
        "Write to john.roe@example.org about GB33BUKB20201555555555 tomorrow.",
    ] * 3
    profile = _profile(corpus, seed=0, min_mention_mass=1, min_document_frequency=1)
    blob = profile.to_json()
    for secret in (
        "jane.doe@example.com",
        "john.roe@example.org",
        "DE89370400440532013000",
        "GB33BUKB20201555555555",
    ):
        assert secret not in blob
        assert secret.lower() not in blob.lower()


def test_a_detector_only_type_is_flagged_as_such_in_the_artifact() -> None:
    corpus = ["Write to jane.doe@example.com about the shipment."] * 4
    payload = json.loads(
        _profile(corpus, seed=0, min_mention_mass=1, min_document_frequency=1).to_json()
    )
    pii = payload["slots"]["pii_types"]["selection"]
    email = next(item for item in pii if item["label"] == "email address")
    assert email["detector_only"] is True
    assert any("detector" in note for note in payload["notes"])


def test_an_empty_corpus_yields_an_empty_but_honest_artifact() -> None:
    profile = _profile([], seed=0)
    payload = json.loads(profile.to_json())
    assert payload["corpus"]["texts_seen"] == 0
    assert profile.lexicon_candidates() == {}
    assert any("no candidate" in note.lower() for note in payload["notes"])


def test_worst_case_ratio_is_the_real_constant() -> None:
    assert math.isclose(WORST_CASE_RATIO, 0.6321205588285577)

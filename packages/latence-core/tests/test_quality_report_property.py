"""Property hardening for the Quality Report completeness + drift roll-ups (#76, Hardening bar).

The unit tests in ``test_quality_report.py`` pin specific completeness/drift shapes; this suite
pins the *invariants* those derivations must uphold under **randomized, seeded** counts — the
Quality-Report completeness/drift property half of #76 (ADR-0008 first-class artifact):

* **completeness monotonicity** — ``CompletenessCheck.passed`` is True IFF provenance AND
  classification are present on every checked record AND every offset-bearing record is aligned;
  flipping any one of the three counts below its total flips ``passed`` to False;
* **offsets_aligned is exactly the aligned==bearing equality** — never a divide, never an
  off-by-one, for any aligned<=bearing counts;
* **drift.clean is exactly recovered==0** — a single fuzzy recovery makes the roll-up dirty,
  whatever the other counts;
* **render() is deterministic and total** — for any randomly-populated report (with/without the
  optional delta / goldset / drift sections) ``render()`` returns the same string twice and never
  raises, and the headline PASS/FAIL never disagrees with ``completeness.passed``.

Deterministic and dependency-free (seeded :class:`random.Random`, no ``hypothesis`` — ADR-0016).
"""

from __future__ import annotations

import random

from latence_core.contracts import DeletionMode, DeltaChurn
from latence_core.quality import (
    CompletenessCheck,
    DriftDiagnostics,
    GoldSetEvaluation,
    LabelScore,
    QualityReport,
    StageMetrics,
    StageScore,
)

_SEEDS = range(500)


# -- CompletenessCheck property ----------------------------------------------


def test_completeness_passed_is_the_three_way_and() -> None:
    # passed == (provenance complete) AND (classification complete) AND (offsets aligned), for any
    # randomized counts. Cross-checked against an independent boolean recomputation.
    for seed in _SEEDS:
        rng = random.Random(seed)
        checked = rng.randint(0, 20)
        prov = rng.randint(0, checked)
        cls = rng.randint(0, checked)
        bearing = rng.randint(0, 20)
        aligned = rng.randint(0, bearing)
        c = CompletenessCheck(
            records_checked=checked,
            records_with_provenance=prov,
            records_with_classification=cls,
            offset_bearing_records=bearing,
            records_with_aligned_offsets=aligned,
        )
        expected_aligned = aligned == bearing
        expected_passed = prov == checked and cls == checked and expected_aligned
        assert c.offsets_aligned is expected_aligned, seed
        assert c.passed is expected_passed, seed


def test_completeness_any_shortfall_fails() -> None:
    # A complete check that then loses ONE unit on any of the three axes must fail — the guard that
    # the headline can never claim PASS while a contract is unmet.
    for seed in _SEEDS:
        rng = random.Random(seed + 1000)
        checked = rng.randint(1, 20)
        bearing = rng.randint(1, 20)
        full = CompletenessCheck(
            records_checked=checked,
            records_with_provenance=checked,
            records_with_classification=checked,
            offset_bearing_records=bearing,
            records_with_aligned_offsets=bearing,
        )
        assert full.passed is True, seed
        axis = rng.choice(["prov", "cls", "align"])
        if axis == "prov":
            broken = full.model_copy(update={"records_with_provenance": checked - 1})
        elif axis == "cls":
            broken = full.model_copy(update={"records_with_classification": checked - 1})
        else:
            broken = full.model_copy(update={"records_with_aligned_offsets": bearing - 1})
        assert broken.passed is False, (seed, axis)


# -- DriftDiagnostics property -----------------------------------------------


def test_drift_clean_is_exactly_recovered_zero() -> None:
    for seed in _SEEDS:
        rng = random.Random(seed + 2000)
        recovered = rng.randint(0, 10)
        d = DriftDiagnostics(
            resolutions=rng.randint(0, 50),
            recovered=recovered,
            max_drift_chars=rng.randint(0, 100),
            records_audited=rng.randint(0, 20),
        )
        assert d.clean is (recovered == 0), seed


# -- render() determinism + totality -----------------------------------------


def _random_report(rng: random.Random) -> QualityReport:
    checked = rng.randint(0, 6)
    bearing = rng.randint(0, 6)
    completeness = CompletenessCheck(
        records_checked=checked,
        records_with_provenance=rng.randint(0, checked),
        records_with_classification=rng.randint(0, checked),
        offset_bearing_records=bearing,
        records_with_aligned_offsets=rng.randint(0, bearing),
    )
    report = QualityReport(
        pipeline_name=rng.choice(["demo", "corpus", "delta-slice"]),
        run_id=f"run-{rng.randint(0, 999)}",
        document_count=rng.randint(0, 100),
        total_duration_seconds=round(rng.random() * 10, 3),
        stages=[
            StageMetrics(
                name=f"stage{i}",
                capability=rng.choice(["source", "parse", "entity_extraction"]),
                provider="ref",
                records_in=rng.randint(0, 100),
                records_out=rng.randint(0, 100),
                error_count=rng.choice([0, 0, 1]),
                error_category=rng.choice([None, "provider", "contract"]),
            )
            for i in range(rng.randint(0, 4))
        ],
        completeness=completeness,
        contracts_complete=completeness.passed,
    )
    # Optionally attach the drift / goldset / delta sections.
    if rng.random() < 0.5:
        report.drift = DriftDiagnostics(
            resolutions=rng.randint(0, 20),
            recovered=rng.randint(0, 5),
            max_drift_chars=rng.randint(0, 50),
            records_audited=rng.randint(0, 10),
        )
    if rng.random() < 0.5:
        report.goldset = GoldSetEvaluation(
            documents_annotated=rng.randint(0, 10),
            documents_matched=rng.randint(0, 10),
            match_min_overlap=rng.choice([0.0, 0.5, 1.0]),
            stages=[
                StageScore(
                    stage=rng.choice(["entities", "relations"]),
                    true_positives=rng.randint(0, 10),
                    false_positives=rng.randint(0, 10),
                    false_negatives=rng.randint(0, 10),
                    precision=round(rng.random(), 6),
                    recall=round(rng.random(), 6),
                    f1=round(rng.random(), 6),
                    per_label={
                        "ORG": LabelScore(
                            true_positives=rng.randint(0, 5),
                            precision=round(rng.random(), 6),
                            recall=round(rng.random(), 6),
                            f1=round(rng.random(), 6),
                        )
                    },
                )
            ],
        )
    if rng.random() < 0.5:
        report.delta = DeltaChurn(
            from_version=rng.choice([None, 0, 1, 2]),
            to_version=rng.randint(0, 5),
            documents_added=rng.randint(0, 5),
            documents_updated=rng.randint(0, 5),
            documents_deleted=rng.randint(0, 5),
            documents_unchanged=rng.randint(0, 5),
            entities_created=rng.randint(0, 5),
            entities_merged=rng.randint(0, 5),
            entities_split=rng.randint(0, 5),
            edges_added=rng.randint(0, 5),
            edges_retracted=rng.randint(0, 5),
            records_retracted=rng.randint(0, 5),
            records_purged=rng.randint(0, 5),
            affected_set_size=rng.randint(0, 5),
            corpus_record_count=rng.randint(0, 10),
            drift=round(rng.random(), 6),
            reconciliation_triggered=rng.choice([True, False]),
            deletion_mode=rng.choice([DeletionMode.RETRACTION, DeletionMode.PURGE]),
        )
    return report


def test_render_is_deterministic_and_total_over_random_reports() -> None:
    # render() must never raise and must be byte-identical on a re-render, for any randomly
    # populated report — the seeded-run rendering property (S10 AC: deterministic human-readable
    # half). Also: the headline PASS/FAIL always mirrors completeness.passed (they can never
    # disagree, the render() invariant the code relies on).
    for seed in _SEEDS:
        rng = random.Random(seed + 3000)
        report = _random_report(rng)
        first = report.render()
        second = report.render()
        assert first == second, seed
        headline_pass = "contracts complete: PASS" in first
        headline_fail = "contracts complete: FAIL" in first
        assert headline_pass != headline_fail, seed  # exactly one is present
        assert headline_pass is report.completeness.passed, seed
        # A round-trip through JSON preserves the render (the machine + human halves agree).
        restored = QualityReport.model_validate_json(report.to_json())
        assert restored.render() == first, seed


def test_render_sections_present_iff_populated() -> None:
    # The optional sections appear in the render exactly when their model is attached — no phantom
    # "Gold-set"/"Delta churn" heading for a report that never set them.
    for seed in _SEEDS:
        rng = random.Random(seed + 4000)
        report = _random_report(rng)
        rendered = report.render()
        assert ("Gold-set evaluation" in rendered) is (report.goldset is not None), seed
        assert ("Delta churn" in rendered) is (report.delta is not None), seed
        assert ("provenance drift" in rendered) is (report.drift is not None), seed

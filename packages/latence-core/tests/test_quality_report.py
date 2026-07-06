"""Quality Report promotion (S10): completeness offset-alignment, drift, render, versioning.

Unit tests for the S10 additions to the first-class artifact: the enriched
CompletenessCheck (offsets aligned), the DriftDiagnostics roll-up model, the human-readable
render(), the gold-set evaluation shape, and the schema-version bump.
"""

from __future__ import annotations

from latence_core.quality import (
    QUALITY_SCHEMA_VERSION,
    CompletenessCheck,
    DriftDiagnostics,
    GoldSetEvaluation,
    LabelScore,
    QualityReport,
    StageMetrics,
    StageScore,
)


def test_schema_version_is_v12() -> None:
    assert QUALITY_SCHEMA_VERSION == 12
    assert QualityReport(pipeline_name="p", run_id="r").schema_version == 12


def test_completeness_passes_only_when_provenance_classification_and_offsets_align() -> None:
    ok = CompletenessCheck(
        records_checked=3,
        records_with_provenance=3,
        records_with_classification=3,
        offset_bearing_records=2,
        records_with_aligned_offsets=2,
    )
    assert ok.offsets_aligned is True
    assert ok.passed is True


def test_completeness_fails_when_an_offset_is_misaligned() -> None:
    bad = CompletenessCheck(
        records_checked=3,
        records_with_provenance=3,
        records_with_classification=3,
        offset_bearing_records=2,
        records_with_aligned_offsets=1,  # one span not aligned
    )
    assert bad.offsets_aligned is False
    assert bad.passed is False


def test_completeness_fails_when_provenance_missing_even_if_offsets_align() -> None:
    bad = CompletenessCheck(
        records_checked=3,
        records_with_provenance=2,  # one missing
        records_with_classification=3,
        offset_bearing_records=0,
        records_with_aligned_offsets=0,
    )
    assert bad.offsets_aligned is True  # vacuously (nothing to audit)
    assert bad.passed is False


def test_drift_diagnostics_clean_and_dirty() -> None:
    clean = DriftDiagnostics(resolutions=8, recovered=0, max_drift_chars=0, records_audited=4)
    assert clean.clean is True
    dirty = DriftDiagnostics(resolutions=8, recovered=2, max_drift_chars=12, records_audited=4)
    assert dirty.clean is False


def test_stage_score_totals() -> None:
    s = StageScore(
        stage="entities",
        true_positives=6,
        false_positives=2,
        false_negatives=4,
        precision=0.75,
        recall=0.6,
        f1=0.666667,
    )
    assert s.gold_total == 10
    assert s.predicted_total == 8


def test_render_includes_headline_sections() -> None:
    report = QualityReport(
        pipeline_name="demo",
        run_id="run-1",
        document_count=2,
        total_duration_seconds=0.5,
        stages=[
            StageMetrics(name="source", capability="source", provider="source.local_folder"),
            StageMetrics(name="parse", capability="parse", provider="parser.plaintext"),
        ],
        completeness=CompletenessCheck(
            records_checked=2,
            records_with_provenance=2,
            records_with_classification=2,
        ),
        drift=DriftDiagnostics(resolutions=4, recovered=0, records_audited=2),
        goldset=GoldSetEvaluation(
            documents_annotated=2,
            documents_matched=2,
            match_min_overlap=0.0,
            stages=[
                StageScore(
                    stage="entities",
                    true_positives=3,
                    false_positives=1,
                    false_negatives=0,
                    precision=0.75,
                    recall=1.0,
                    f1=0.857143,
                    per_label={"ORG": LabelScore(true_positives=3, precision=1.0, recall=1.0, f1=1.0)},
                )
            ],
        ),
    )
    rendered = report.render()
    assert "# Quality Report — demo (run-1)" in rendered
    assert "documents: 2" in rendered
    assert "contracts complete: PASS" in rendered
    assert "provenance drift: clean" in rendered
    assert "Gold-set evaluation" in rendered
    assert "entities: P=0.750 R=1.000 F1=0.857" in rendered
    assert "ORG:" in rendered
    # Deterministic: same report renders byte-identically.
    assert report.render() == rendered


def test_render_marks_failed_contracts() -> None:
    report = QualityReport(
        pipeline_name="demo",
        run_id="r",
        completeness=CompletenessCheck(
            records_checked=2,
            records_with_provenance=1,
            records_with_classification=2,
        ),
    )
    assert "contracts complete: FAIL" in report.render()

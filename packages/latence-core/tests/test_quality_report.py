"""Quality Report promotion (S10): completeness offset-alignment, drift, render, versioning.

Unit tests for the S10 additions to the first-class artifact: the enriched
CompletenessCheck (offsets aligned), the DriftDiagnostics roll-up model, the human-readable
render(), the gold-set evaluation shape, and the schema-version bump.
"""

from __future__ import annotations

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    PageMap,
    PageSpan,
    Provenance,
)
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
from latence_core.runner import LocalRunner


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


def _chunk_with_page_map(
    *, record_id: str, char_start: int, char_end: int, total_chars: int
) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        file_size=total_chars,
        document_id="sha256:doc",
        page_start=1,
        page_end=1,
        char_start=char_start,
        char_end=char_end,
    )
    page_map = PageMap(
        pages=[PageSpan(page_number=1, char_start=0, char_end=total_chars)],
        total_chars=total_chars,
    )
    return ChunkRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        document_record_id="doc",
        index=0,
        token_count=1,
        page_map=page_map,
    )


def test_drift_diagnostics_audits_only_chunk_records() -> None:
    # Regression for the corrected DriftDiagnostics/_drift_diagnostics scope (issue #39.1):
    # the audit resolves ONLY ChunkRecords with a page map — mentions/relations/redaction
    # spans inherit chunk offsets and are not independently audited. An EntityMention mixed
    # into the record list must not be counted, or the docstring would overclaim again.
    chunk = _chunk_with_page_map(record_id="c1", char_start=0, char_end=10, total_chars=20)
    mention = EntityMention(
        record_id="m1",
        provenance=Provenance(
            source_uri="file:///c/doc.txt",
            file_name="doc.txt",
            file_type="txt",
            file_size=20,
            document_id="sha256:doc",
            page_start=1,
            page_end=1,
            char_start=2,
            char_end=7,
        ),
        classification=Classification(language="en", sensitivity="internal"),
        document_record_id="doc",
        chunk_record_id="c1",
        index=0,
        label="ORG",
        text="Acme",
        confidence=1.0,
    )
    diag = LocalRunner._drift_diagnostics([chunk, mention])
    assert diag is not None
    # Exactly one record audited: the chunk. The mention is ignored despite carrying a span.
    assert diag.records_audited == 1
    assert diag.clean is True


def test_drift_diagnostics_recovers_offset_outside_every_page_span() -> None:
    # A chunk whose end offset falls past the page map's total_chars needs the fuzzy
    # fallback: the audit counts it as a recovery (drift), proving the roll-up is live.
    chunk = _chunk_with_page_map(record_id="c1", char_start=0, char_end=30, total_chars=20)
    diag = LocalRunner._drift_diagnostics([chunk])
    assert diag is not None
    assert diag.records_audited == 1
    assert diag.recovered >= 1
    assert diag.clean is False


def test_drift_diagnostics_none_when_no_chunk_records() -> None:
    # A chunk-free run (only a mention in the list) has nothing to audit -> None, not an
    # all-zero section. Guards the "reports no drift section" contract.
    mention = EntityMention(
        record_id="m1",
        provenance=Provenance(
            source_uri="file:///c/doc.txt",
            file_name="doc.txt",
            file_type="txt",
            file_size=20,
            document_id="sha256:doc",
            page_start=1,
            page_end=1,
            char_start=2,
            char_end=7,
        ),
        classification=Classification(language="en", sensitivity="internal"),
        document_record_id="doc",
        chunk_record_id="c1",
        index=0,
        label="ORG",
        text="Acme",
        confidence=1.0,
    )
    assert LocalRunner._drift_diagnostics([mention]) is None


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

"""Report + run-state builders for the console tests: real artifacts, not stand-ins.

The builders write the SAME two files the local Runner writes under
``<storage_uri>/_latence/runs/<run_id>/`` — ``manifest.json`` and ``quality-report.json`` —
through the real :class:`~latence_core.storage.Storage` seam, and the report is a real
:class:`~latence_core.quality.QualityReport` serialised by its own ``to_json``. Nothing here
is a stand-in for the file format: a fixture that accepted a shape the real report cannot
produce would manufacture false green.
"""

from __future__ import annotations

import json

from latence_core.contracts import (
    DeletionMode,
    DeltaChurn,
    ScreeningCheckpoint,
    ScreeningFinding,
    ScreeningVerdict,
)
from latence_core.quality import (
    ChunkQuality,
    CompletenessCheck,
    GraphQuality,
    ParseQuality,
    QualityReport,
    RedactionQuality,
    ScreeningQuality,
    StageMetrics,
)
from latence_core.storage import Storage


def write_run(
    storage: Storage, storage_uri: str, run_id: str, report: QualityReport | None
) -> None:
    """Write one run's on-disk state exactly where and how the local Runner writes it."""
    run_dir = f"{storage_uri.rstrip('/')}/_latence/runs/{run_id}"
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {"pipeline": {"name": report.pipeline_name if report else "unknown"},
             "run_id": run_id},
            sort_keys=True,
        ).encode("utf-8"),
    )
    if report is not None:
        storage.write_atomic(
            f"{run_dir}/quality-report.json", report.to_json().encode("utf-8")
        )


def healthy_report(run_id: str = "run-0001", pipeline_name: str = "acme-contracts") -> QualityReport:
    """A run where every check holds — the PASSED case."""
    return QualityReport(
        pipeline_name=pipeline_name,
        run_id=run_id,
        document_count=4,
        total_duration_seconds=1.25,
        stages=[
            StageMetrics(
                name="parse", capability="parse", provider="parser.plaintext",
                records_in=4, records_out=4, duration_seconds=0.4,
            ),
            StageMetrics(
                name="graph", capability="graph", provider="graph.builder",
                records_in=4, records_out=1, duration_seconds=0.85,
            ),
        ],
        completeness=CompletenessCheck(
            records_checked=82,
            records_with_provenance=82,
            records_with_classification=82,
            offset_bearing_records=82,
            records_with_aligned_offsets=82,
        ),
        parse=ParseQuality(documents_parsed=4, pages_total=11),
        chunk=ChunkQuality(chunks_total=40, chunks_offset_preserving=40, tokens_total=9000),
        screening=ScreeningQuality(
            intake_screened=7,
            content_screened=40,
            quarantined=1,
            flagged=1,
            passed=45,
            findings=[
                ScreeningFinding(
                    checkpoint=ScreeningCheckpoint.INTAKE,
                    verdict=ScreeningVerdict.QUARANTINE,
                    document_id="sha256:" + "a" * 64,
                    detector="screening.intake_signature",
                    category="zip_bomb",
                    reason="zip compression ratio 966.9x exceeds the 50.0x cap",
                ),
            ],
        ),
        redaction=RedactionQuality(documents_redacted=4, documents_with_pii=0, pii_total=0),
        graph=GraphQuality(
            nodes_total=6, edges_total=33, nodes_linked=2,
            edges_by_label={"works_for": 21, "partner_of": 12},
            edges_with_evidence=33, evidence_coverage=1.0,
        ),
    )


def gapped_report(run_id: str = "run-0002") -> QualityReport:
    """A run that finished but whose contract-completeness did NOT hold — the ATTENTION case."""
    report = healthy_report(run_id=run_id)
    return report.model_copy(
        update={
            "completeness": CompletenessCheck(
                records_checked=82,
                records_with_provenance=70,
                records_with_classification=82,
                offset_bearing_records=82,
                records_with_aligned_offsets=79,
            ),
            "contracts_complete": False,
        }
    )


def failed_report(run_id: str = "run-0003") -> QualityReport:
    """A run with a Stage that raised — the FAILED case."""
    report = healthy_report(run_id=run_id)
    return report.model_copy(
        update={
            "stages": [
                report.stages[0],
                StageMetrics(
                    name="graph", capability="graph", provider="graph.builder",
                    records_in=4, records_out=0, duration_seconds=0.1,
                    error_count=1, error_category="provider",
                ),
            ]
        }
    )


def delta_report(run_id: str = "run-0004") -> QualityReport:
    """An incremental run carrying the what-changed summary."""
    report = healthy_report(run_id=run_id)
    return report.model_copy(
        update={
            "delta": DeltaChurn(
                from_version=3,
                to_version=4,
                documents_added=2,
                documents_updated=1,
                documents_deleted=1,
                documents_unchanged=9,
                documents_extracted=3,
                documents_reused=9,
                edges_added=7,
                edges_retracted=2,
                deletion_mode=DeletionMode.RETRACTION,
            )
        }
    )

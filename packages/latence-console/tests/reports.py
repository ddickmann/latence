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
    ChunkRecord,
    Classification,
    DeletionMode,
    DeltaChurn,
    DocumentRecord,
    EntityMention,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSpan,
    PIISpan,
    Provenance,
    QuarantineRecord,
    RedactionAction,
    RiskMarker,
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


# -- #188: the run-artifact builders (checkpoints + screening sidecars) --------------------
#
# These write the SAME JSONL files the local Runner writes under the run directory
# (``checkpoints/<stage>.jsonl``, ``quarantine/<stage>.jsonl``, ``screening/<stage>.jsonl``),
# serialised by the REAL contracts' own ``model_dump_json``. Nothing here hand-rolls a record
# shape: a line these builders can produce is, by construction, a line the pipeline could
# have produced — and a line the pipeline could not produce cannot be built here either,
# because every contract validates on construction.

# The fixture document: 120 characters of assembled markdown across two pages.
DOC_ID = "sha256:" + "d" * 58
CHUNK_ID = f"{DOC_ID}#chunk-0"
SENSITIVE_CHUNK_ID = f"{DOC_ID}#chunk-1"
MENTION_ID = f"{CHUNK_ID}#entity-0"
DANGLING_MENTION_ID = f"{CHUNK_ID}#entity-99"
NODE_PERSON = "node-person-0001"
NODE_ORG = "node-org-0001"
EDGE_ID = "edge-0001"

# The chunk's markup-stripped text. Markup was stripped INSIDE it, so its offset map is a
# two-run shift, not the identity — which is exactly what makes the inspector's resolved span
# differ from the naive ``char_start + len(content)``.
CHUNK_TEXT = "Ada Lovelace works at Acme Corp in London, UK 1843."
MASKED_CHUNK_TEXT = "[PERSON] works at Acme Corp in London, UK 1843."

_PAGE_MAP = PageMap(
    pages=[
        PageSpan(page_number=1, char_start=0, char_end=50),
        PageSpan(page_number=2, char_start=50, char_end=120),
    ],
    total_chars=120,
)
_OFFSET_MAP = OffsetMap(
    segments=[OffsetSegment(local_start=0, doc_start=10), OffsetSegment(local_start=20, doc_start=60)],
    length=len(CHUNK_TEXT),
)
_CLASSIFICATION = Classification(language="en", category="report", sensitivity="internal")


def _provenance(**overrides: object) -> Provenance:
    base: dict[str, object] = {
        "source_uri": "file:///corpus/memo.md",
        "file_name": "memo.md",
        "file_type": "md",
        "file_size": 120,
        "document_id": DOC_ID,
    }
    base.update(overrides)
    return Provenance.model_validate(base)


_RISK_MARKER = RiskMarker(
    category="prompt_injection",
    detector="screening.content_keyword",
    score=1.0,
    reason="matched injection pattern 'ignore all previous instructions'",
)


def _chunk(*, redacted: bool) -> ChunkRecord:
    """The fixture chunk, before and after the Redaction Stage rewrote it."""
    return ChunkRecord(
        record_id=CHUNK_ID,
        provenance=_provenance(page_start=1, page_end=2, char_start=10, char_end=91),
        classification=_CLASSIFICATION,
        document_record_id=DOC_ID,
        index=0,
        content=CHUNK_TEXT,
        token_count=12,
        page_map=_PAGE_MAP,
        offset_map=_OFFSET_MAP,
        risk_markers=[_RISK_MARKER],
        masked_content=MASKED_CHUNK_TEXT if redacted else None,
        pii_spans=[
            PIISpan(
                pii_type="person",
                detector="regex:person",
                char_start=0,
                char_end=12,
                action=RedactionAction.MASK,
                placeholder="[PERSON]",
                confidence=0.99,
            )
        ]
        if redacted
        else [],
    )


def _sensitive_chunk() -> ChunkRecord:
    """A confidential chunk whose redaction policy ran NO detectors (the no-op floor)."""
    return ChunkRecord(
        record_id=SENSITIVE_CHUNK_ID,
        provenance=_provenance(page_start=2, page_end=2, char_start=91, char_end=120),
        classification=Classification(language="en", category="report", sensitivity="confidential"),
        document_record_id=DOC_ID,
        index=1,
        content="Board pay review, see appendix.",
        token_count=7,
        page_map=_PAGE_MAP,
        redaction_disabled_for_sensitive=True,
    )


def _mention() -> EntityMention:
    return EntityMention(
        record_id=MENTION_ID,
        provenance=_provenance(page_start=2, page_end=2, char_start=62, char_end=71),
        classification=_CLASSIFICATION,
        document_record_id=DOC_ID,
        chunk_record_id=CHUNK_ID,
        index=0,
        label="ORG",
        text="Acme Corp",
        confidence=0.91,
    )


def _graph_records() -> list[GraphRecord]:
    provenance = _provenance(page_start=1, page_end=2, char_start=10, char_end=91)
    evidence = Evidence(
        mention_ids=[MENTION_ID, DANGLING_MENTION_ID],
        document_ids=[DOC_ID],
        snippet="works_for holds between Ada Lovelace and Acme Corp",
        confidence=0.87,
    )
    person = GraphNode(
        record_id=NODE_PERSON,
        provenance=provenance,
        classification=_CLASSIFICATION,
        node_id=NODE_PERSON,
        entity_id="entity-person",
        label="PERSON",
        canonical_name="Ada Lovelace",
        member_mention_ids=[MENTION_ID],
        source_document_ids=[DOC_ID],
        confidence=0.95,
        evidence=evidence,
    )
    org = GraphNode(
        record_id=NODE_ORG,
        provenance=provenance,
        classification=_CLASSIFICATION,
        node_id=NODE_ORG,
        entity_id="entity-org",
        label="ORG",
        canonical_name="Acme Corp",
        member_mention_ids=[MENTION_ID],
        source_document_ids=[DOC_ID],
        kb_id="Q12345",
        confidence=0.9,
        evidence=evidence,
    )
    edge = GraphEdge(
        record_id=EDGE_ID,
        provenance=provenance,
        classification=_CLASSIFICATION,
        edge_id=EDGE_ID,
        relation_id="relation-0001",
        source_node_id=NODE_PERSON,
        target_node_id=NODE_ORG,
        label="works_for",
        confidence=0.83,
        properties={"inverted": False, "is_predicted": False},
        evidence=evidence,
    )
    return [
        GraphRecord(
            record_id=f"{NODE_PERSON}#record",
            provenance=provenance,
            classification=_CLASSIFICATION,
            scope=GraphScope.NODE,
            node=person,
        ),
        GraphRecord(
            record_id=f"{NODE_ORG}#record",
            provenance=provenance,
            classification=_CLASSIFICATION,
            scope=GraphScope.NODE,
            node=org,
        ),
        GraphRecord(
            record_id=f"{EDGE_ID}#record",
            provenance=provenance,
            classification=_CLASSIFICATION,
            scope=GraphScope.EDGE,
            edge=edge,
        ),
    ]


def _quarantine_record() -> QuarantineRecord:
    return QuarantineRecord(
        record_id="sha256:" + "z" * 58,
        provenance=_provenance(
            source_uri="file:///corpus/archive.zip",
            file_name="archive.zip",
            file_type="zip",
            file_size=2169,
            document_id="sha256:" + "z" * 58,
        ),
        classification=Classification(language="und", sensitivity="internal"),
        checkpoint=ScreeningCheckpoint.INTAKE,
        detector="screening.intake_signature",
        category="zip_bomb",
        reason="zip compression ratio 966.9x exceeds the 50.0x cap",
        quarantined_bytes=2169,
    )


_STAGES = [
    {"name": "parse", "depends_on": []},
    {"name": "chunk", "depends_on": ["parse"]},
    {"name": "redact", "depends_on": ["chunk"]},
    {"name": "entities", "depends_on": ["redact"]},
    {"name": "graph", "depends_on": ["entities"]},
]


def write_artifacts(
    storage: Storage,
    storage_uri: str,
    run_id: str,
    *,
    pipeline_name: str = "acme-contracts",
    redact: bool = True,
) -> None:
    """Write one run's checkpoints + screening sidecars exactly where the Runner writes them.

    ``redact=False`` writes the run WITHOUT the Redaction Stage, so a test can assert the
    inspector reports the un-redacted disposition rather than inventing one.
    """
    run_dir = f"{storage_uri.rstrip('/')}/_latence/runs/{run_id}"
    stages = [s for s in _STAGES if redact or s["name"] != "redact"]
    if not redact:
        stages = [
            {"name": s["name"], "depends_on": [d for d in s["depends_on"] if d != "redact"]
             + (["chunk"] if s["name"] == "entities" else [])}
            for s in stages
        ]
    storage.write_atomic(
        f"{run_dir}/manifest.json",
        json.dumps(
            {"pipeline": {"name": pipeline_name, "stages": stages}, "run_id": run_id},
            sort_keys=True,
        ).encode("utf-8"),
    )

    def _jsonl(uri: str, records: list[object]) -> None:
        payload = "".join(r.model_dump_json() + "\n" for r in records)  # type: ignore[attr-defined]
        storage.write_atomic(uri, payload.encode("utf-8"))

    document = DocumentRecord(
        record_id=DOC_ID,
        provenance=_provenance(),
        classification=_CLASSIFICATION,
        content="x" * 120,
        page_map=_PAGE_MAP,
    )
    _jsonl(f"{run_dir}/checkpoints/parse.jsonl", [document])
    _jsonl(
        f"{run_dir}/checkpoints/chunk.jsonl", [_chunk(redacted=False), _sensitive_chunk()]
    )
    if redact:
        # The SAME record_id, rewritten downstream — the console must show THIS one.
        _jsonl(f"{run_dir}/checkpoints/redact.jsonl", [_chunk(redacted=True)])
    _jsonl(f"{run_dir}/checkpoints/entities.jsonl", [_mention()])
    _jsonl(f"{run_dir}/checkpoints/graph.jsonl", _graph_records())
    _jsonl(f"{run_dir}/quarantine/intake_screen.jsonl", [_quarantine_record()])
    _jsonl(
        f"{run_dir}/screening/intake_screen.jsonl",
        [
            ScreeningFinding(
                checkpoint=ScreeningCheckpoint.INTAKE,
                verdict=ScreeningVerdict.QUARANTINE,
                document_id="sha256:" + "z" * 58,
                detector="screening.intake_signature",
                category="zip_bomb",
                reason="zip compression ratio 966.9x exceeds the 50.0x cap",
            ),
            ScreeningFinding(
                checkpoint=ScreeningCheckpoint.CONTENT,
                verdict=ScreeningVerdict.FLAG,
                document_id=DOC_ID,
                detector="screening.content_keyword",
                category="prompt_injection",
                reason=_RISK_MARKER.reason,
                marker=_RISK_MARKER,
            ),
        ],
    )

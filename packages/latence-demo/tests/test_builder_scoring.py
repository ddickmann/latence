"""Regression tests for DemoBuilder._score_dangerous per-file binding (S13 debt #46.2).

``_score_dangerous`` proves each planted dangerous fixture got its expected Screening
disposition. The robust binding is **per file, by ``document_id``** — the content-addressed
id the Source stamps and every ScreeningFinding carries — not by a coincidental
``verdict:category`` pair. These tests pin that: a 'right reason, wrong file' mislabel (the
expected disposition landing on a *different* document) must surface as ``matched=False``.
A revert to pair-keyed matching would let that mislabel pass, turning the adversarial test red.
"""

from __future__ import annotations

from latence_core.contracts import (
    RiskMarker,
    ScreeningCheckpoint,
    ScreeningFinding,
    ScreeningVerdict,
)
from latence_core.quality import QualityReport, ScreeningQuality
from latence_demo.builder import DemoBuilder
from latence_demo.corpus import CorpusManifest


def _manifest() -> CorpusManifest:
    # Two planted fixtures whose expected dispositions differ, plus a stable document_id per file.
    return CorpusManifest(
        documents={"invoice.pdf": 10, "archive.zip": 20},
        dangerous={
            "invoice.pdf": "quarantine:type_spoof",
            "archive.zip": "quarantine:zip_bomb",
        },
        document_ids={
            "invoice.pdf": "sha256:aaaa",
            "archive.zip": "sha256:bbbb",
        },
    )


def _report(findings: list[ScreeningFinding]) -> QualityReport:
    return QualityReport(
        pipeline_name="latence-demo",
        run_id="test",
        screening=ScreeningQuality(findings=findings),
    )


def _quarantine(document_id: str, category: str) -> ScreeningFinding:
    return ScreeningFinding(
        checkpoint=ScreeningCheckpoint.INTAKE,
        verdict=ScreeningVerdict.QUARANTINE,
        document_id=document_id,
        detector="screening.intake_signature",
        category=category,
        reason=f"{category} detected",
    )


def test_right_reason_wrong_file_does_not_match() -> None:
    # The 'type_spoof' quarantine that SHOULD land on invoice.pdf lands on archive.zip instead,
    # and archive.zip's 'zip_bomb' quarantine lands on invoice.pdf. Both dispositions are present
    # in the report (pair-keyed matching would call BOTH files matched), but each is on the WRONG
    # document — a real mislabel the demo must fail on.
    findings = [
        _quarantine("sha256:bbbb", "type_spoof"),  # type_spoof on archive.zip (wrong file)
        _quarantine("sha256:aaaa", "zip_bomb"),  # zip_bomb on invoice.pdf (wrong file)
    ]
    outcomes = DemoBuilder._score_dangerous(_report(findings), _manifest())
    by_name = {o.file_name: o for o in outcomes}
    assert by_name["invoice.pdf"].matched is False
    assert by_name["invoice.pdf"].observed == "quarantine:zip_bomb"
    assert by_name["archive.zip"].matched is False
    assert by_name["archive.zip"].observed == "quarantine:type_spoof"


def test_correct_per_file_binding_matches() -> None:
    # The honest case: each disposition lands on its own file, keyed by document_id.
    findings = [
        _quarantine("sha256:aaaa", "type_spoof"),
        _quarantine("sha256:bbbb", "zip_bomb"),
    ]
    outcomes = DemoBuilder._score_dangerous(_report(findings), _manifest())
    assert all(o.matched for o in outcomes)


def test_missing_finding_is_not_detected() -> None:
    # A dangerous file with no matching finding surfaces as not-detected (failing loudly).
    outcomes = DemoBuilder._score_dangerous(_report([]), _manifest())
    assert all(o.observed == "not-detected" and not o.matched for o in outcomes)


def test_flag_finding_binds_by_document_id() -> None:
    # A content-screen FLAG (prompt injection) also binds by document_id, not by pair.
    manifest = CorpusManifest(
        documents={"support-ticket.md": 30, "memo.md": 40},
        dangerous={"support-ticket.md": "flag:prompt_injection"},
        document_ids={"support-ticket.md": "sha256:cccc", "memo.md": "sha256:dddd"},
    )
    # The flag lands on the WRONG file (memo.md), not support-ticket.md.
    wrong = ScreeningFinding(
        checkpoint=ScreeningCheckpoint.CONTENT,
        verdict=ScreeningVerdict.FLAG,
        document_id="sha256:dddd",
        detector="screening.content_keyword",
        category="prompt_injection",
        reason="injection string quoted",
        marker=RiskMarker(
            category="prompt_injection",
            detector="screening.content_keyword",
            score=0.9,
            reason="injection string quoted",
        ),
    )
    outcomes = DemoBuilder._score_dangerous(_report([wrong]), manifest)
    assert outcomes[0].file_name == "support-ticket.md"
    assert outcomes[0].matched is False
    assert outcomes[0].observed == "not-detected"

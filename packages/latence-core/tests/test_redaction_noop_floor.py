"""H-C1 §4 — the sensitive-document redaction no-op floor.

A policy with ``skip:true``, ``types:[]``, or a ``detectors``/``types`` combination
that resolves to no detectors yields a RedactionRecord with an EMPTY span list that is
indistinguishable, in the raw contract, from "genuinely no PII found". On a
sensitive-classified document (``confidential``/``restricted``/…) that silence is the
"clean bill of health" lie the audit named: a disabled control masquerading as a clean
corpus.

The floor makes it LOUD and first-class: the RedactionRecord carries
``redaction_disabled_for_sensitive=True`` and the Quality Report carries a
``redaction_disabled_for_sensitive`` count, so ``documents_with_pii: 0`` can no longer
hide a ``skip:true`` on a ``confidential`` document. The floor does NOT force redaction
on — an operator may intentionally skip — but the disposition is auditable, not
invisible.

Unit tests cover each way a control resolves to a no-op (skip / empty types / narrowed
detectors) on a sensitive vs a non-sensitive document, the configurable sensitive set,
and that a genuinely-clean sensitive document (control ON, no PII) is NOT flagged. One
end-to-end test drives the signal through the local Runner into the persisted Quality
Report and export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    PageSlice,
    PageSpan,
    Provenance,
)
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.stages.redaction import HybridPIIRedactor
from latence_core.storage import Storage


def _chunk(text: str, record_id: str, sensitivity: str) -> ChunkRecord:
    prov = Provenance(
        source_uri=f"file:///c/{record_id}.txt",
        file_name=f"{record_id}.txt",
        file_type="txt",
        file_size=len(text),
        document_id=f"sha256:{record_id}",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=f"{record_id}#chunk:0",
        document_record_id=record_id,
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=text,
        token_count=len(text.split()),
        # v19: a one-page fixture — the chunk's own page spans are that single page.
        page_slice=PageSlice(
            pages=[PageSpan(page_number=1, char_start=0, char_end=len(text))],
            total_chars=len(text),
        ),
        offset_map=OffsetMap(segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text)),
    )


_SECRET_DOC = "Leaked key ghp_ABCdef1234567890ABCdef1234567890abcd in the ticket."


def _redact(config: dict[str, Any], text: str, sensitivity: str) -> ChunkRecord:
    return next(iter(HybridPIIRedactor(config).redact([_chunk(text, "d", sensitivity)])))


# -- the flag fires when the control ran no detectors on a sensitive doc --------


def test_skip_true_on_confidential_is_flagged() -> None:
    """``skip:true`` on a ``confidential`` doc → empty spans BUT the flag is set."""
    config = {"policies": {"confidential": {"skip": True}}}
    rec = _redact(config, _SECRET_DOC, "confidential")
    assert rec.pii_spans == []
    assert rec.masked_content == _SECRET_DOC  # control off: the secret is untouched
    assert rec.redaction_disabled_for_sensitive is True


def test_empty_types_on_restricted_is_flagged() -> None:
    """A ``types: []`` allow-list runs no detector — flagged on a sensitive doc."""
    config: dict[str, Any] = {"policies": {"restricted": {"types": []}}}
    rec = _redact(config, _SECRET_DOC, "restricted")
    assert rec.pii_spans == []
    assert rec.redaction_disabled_for_sensitive is True


def test_narrowed_detectors_with_disjoint_types_is_flagged() -> None:
    """Detectors ∩ policy-types = ∅ (a real no-op) is flagged on a sensitive doc.

    Only ``email`` is enabled, but the policy restricts to ``ssn`` — so no detector can
    run. That is a disabled control, not "no PII found".
    """
    config = {
        "detectors": ["email"],
        "policies": {"confidential": {"types": ["ssn"]}},
    }
    rec = _redact(config, _SECRET_DOC, "confidential")
    assert rec.pii_spans == []
    assert rec.redaction_disabled_for_sensitive is True


# -- the flag does NOT fire when it shouldn't ---------------------------------


def test_skip_true_on_public_is_not_flagged() -> None:
    """A disabled control on a NON-sensitive doc is not the floor's concern."""
    config = {"policies": {"public": {"skip": True}}}
    rec = _redact(config, _SECRET_DOC, "public")
    assert rec.pii_spans == []
    assert rec.redaction_disabled_for_sensitive is False


def test_active_control_on_confidential_finding_pii_is_not_flagged() -> None:
    """Control ON + PII found on a sensitive doc: masked, and NOT flagged as disabled."""
    config = {"policies": {"confidential": {"action": "mask"}}}
    rec = _redact(config, _SECRET_DOC, "confidential")
    assert rec.pii_spans  # the secret was detected
    assert rec.masked_content is not None
    assert "ghp_ABCdef1234567890ABCdef1234567890abcd" not in rec.masked_content
    assert rec.redaction_disabled_for_sensitive is False


def test_active_control_on_clean_confidential_is_not_flagged() -> None:
    """A genuinely-clean sensitive doc (control ON, no PII) is NOT the disabled-control lie.

    This is the crux: an empty span list here is "nothing to redact", not "control off"
    — the flag must stay False so it stays distinct from the skipped case above.
    """
    config = {"policies": {"confidential": {"action": "mask"}}}
    rec = _redact(config, "The quarterly report is attached for review.", "confidential")
    assert rec.pii_spans == []
    assert rec.redaction_disabled_for_sensitive is False


def test_sensitive_set_is_configurable() -> None:
    """The sensitive set is configurable; a value outside it is not flagged, inside is."""
    config = {
        "sensitive_set": ["secret-tier"],
        "policies": {"confidential": {"skip": True}, "secret-tier": {"skip": True}},
    }
    # ``confidential`` is NOT in the custom sensitive set → not flagged.
    assert _redact(config, _SECRET_DOC, "confidential").redaction_disabled_for_sensitive is False
    # ``secret-tier`` IS → flagged.
    assert _redact(config, _SECRET_DOC, "secret-tier").redaction_disabled_for_sensitive is True


# -- end-to-end: the loud signal reaches the Quality Report --------------------

_YAML = """
      detectors: [email, phone, ssn, secret]
      default_action: mask
      policies:
        confidential:
          skip: true
"""


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: redaction-noop-floor
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: confidential
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:{_YAML}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )


def test_disabled_control_is_loud_in_quality_report_end_to_end(tmp_path: Path) -> None:
    """§4 E2E: a ``skip:true`` on a ``confidential`` corpus is LOUD, not a clean bill.

    The whole corpus is classified ``confidential`` and the policy skips it, so every
    document flows through unredacted (the secret is intact). ``documents_with_pii`` is 0
    — but the run must NOT read as clean: the ``redaction_disabled_for_sensitive`` count
    equals the document count, and the per-record flag is set on the persisted records.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leak.txt").write_text(
        "Committed key ghp_ABCdef1234567890ABCdef1234567890abcd to the repo.\n",
        encoding="utf-8",
    )
    (docs / "memo.txt").write_text("Please email alice@acme.com for access.\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"

    report = LocalRunner().run(_pipeline(store, docs), run_id="floor")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/floor"

    # The Quality Report is LOUD: documents_with_pii is 0 (control off) but the disabled
    # count is non-zero, so 0 can never be mistaken for a clean bill of health.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.redaction is not None
    rq = persisted.redaction
    assert rq.documents_with_pii == 0
    assert rq.documents_redacted == 2
    assert rq.redaction_disabled_for_sensitive == 2
    assert report.redaction is not None
    assert report.redaction.redaction_disabled_for_sensitive == 2

    # Every persisted redacted ChunkRecord carries the per-record flag (W16, ADR-0042).
    rows = [
        json.loads(line)
        for line in s.read_text(f"{base}/export/records.jsonl").splitlines()
        if line.strip()
    ]
    records = [ChunkRecord.model_validate(r) for r in rows]
    assert records
    assert all(r.redaction_disabled_for_sensitive for r in records)
    # The control was genuinely off — the secret is still present (the floor makes this
    # AUDITABLE; it does not force redaction on). With the control off, masked_content == content,
    # so the Export materializes the un-redacted text (the secret) as the corpus text.
    assert any(
        r.masked_content is not None
        and "ghp_ABCdef1234567890ABCdef1234567890abcd" in r.masked_content
        for r in records
    )

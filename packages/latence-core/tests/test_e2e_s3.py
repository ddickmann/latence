"""Real end-to-end pipeline on ``s3://`` object storage (MinIO in CI) — ADR-0009 made true.

H-E1 Part 2. The old ``test_e2e_s3.py`` was named/docstringed "end-to-end S3" but set
``store = file://…`` and ran unconditionally, so the ADR-0009 promise — *the same pipeline runs
on* ``s3://`` *with no code change* — was never exercised on object storage (the ``storage-s3``
MinIO job ran only a 5-byte round-trip). That local-FS file is now honestly
``test_e2e_pipeline_spine.py``; THIS file runs the **same** screening spine (the shared
``_spine_pipeline`` builder) against a real ``s3://`` bucket on MinIO.

It is skipped unless ``LATENCE_S3_ENDPOINT`` is set (so the laptop suite stays hermetic); the
``storage-s3`` CI lane sets it and points at MinIO. The framework code is UNCHANGED between the
two lanes: the Runner builds ``Storage.for_uri(pipeline.storage_uri)`` with no options, and
botocore resolves the MinIO endpoint from the standard ``AWS_ENDPOINT_URL_S3`` environment
variable (set here from ``LATENCE_S3_ENDPOINT``) — that env-only resolution is exactly the
"no code change" ADR-0009 claim.

The e2e exercises the real pipeline invariants ON OBJECT STORAGE:

* intake-Quarantine-before-Parse (the oversized file never reaches Parse),
* offset round-trip (chunk → offset → original source page),
* sensitivity inheritance (``confidential`` flows to every exported chunk),
* the injection risk-marker surviving into the exported corpus,
* an idempotent resume whose output is **byte-identical** (H-E1 Part 1 discipline, on s3://).
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterator

import pytest
from _resume_equality import output_artifacts
from _spine_pipeline import CORPUS, spine_pipeline
from latence_core.contracts import ChunkRecord, PageMap
from latence_core.pagemap import PageOffsetIndex
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

pytestmark = pytest.mark.skipif(
    not os.environ.get("LATENCE_S3_ENDPOINT"),
    reason="set LATENCE_S3_ENDPOINT (and AWS creds) to run the real s3:// end-to-end on MinIO",
)


@pytest.fixture()
def s3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point botocore at MinIO via the STANDARD env vars — no framework code change.

    The framework's ``Storage`` is built with no explicit endpoint/creds (``Storage.for_uri`` in
    the Runner). botocore resolves the endpoint from ``AWS_ENDPOINT_URL_S3`` and creds from the
    ``AWS_*`` chain, so exporting these here is all it takes for the identical pipeline to run on
    ``s3://`` — precisely the ADR-0009 "no code change on object storage" guarantee. Set on the
    process env (not passed as Storage kwargs) so the Runner's own ``Storage.for_uri`` sees them.
    """
    endpoint = os.environ["LATENCE_S3_ENDPOINT"]
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint)
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    # A region is required for the SigV4 signer even against MinIO; default if the CI didn't set it.
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


@pytest.fixture()
def s3_root(s3_env: None) -> Iterator[str]:
    """A fresh, unique ``s3://bucket/<uuid>`` prefix, torn down after the test.

    Requires ``s3fs`` (installed only in the CI s3 job). Uploads nothing yet — the test writes the
    corpus under ``<root>/docs`` and runs the pipeline under ``<root>``.
    """
    pytest.importorskip("s3fs")
    bucket = os.environ.get("LATENCE_S3_BUCKET", "latence-test")
    root = f"s3://{bucket}/{uuid.uuid4().hex}"
    storage = Storage(protocol="s3")
    yield root
    # Best-effort cleanup so the bucket doesn't accumulate across CI runs; cleanup must never
    # fail the test.
    with contextlib.suppress(Exception):  # pragma: no cover - cleanup, not an assertion path
        storage.remove_tree(root)


def _upload_corpus(root: str) -> str:
    """Write the shared corpus under ``<root>/docs`` on s3 and return that ``s3://`` docs URI."""
    storage = Storage.for_uri(root)
    docs_uri = f"{root}/docs"
    for name, text in CORPUS.items():
        storage.write_bytes(f"{docs_uri}/{name}", text.encode("utf-8"))
    return docs_uri


def test_s3_end_to_end_on_object_storage(s3_root: str) -> None:
    docs_uri = _upload_corpus(s3_root)
    store = s3_root
    pipeline = spine_pipeline(store, docs_uri)

    report = LocalRunner().run(pipeline, run_id="s3e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s3e2e"

    # -- Intake Screening ran BEFORE Parse; the oversized file was Quarantined on s3 -----
    intake_ckpt = f"{base}/checkpoints/intake.jsonl"
    passed = [json.loads(x) for x in s.read_text(intake_ckpt).splitlines() if x.strip()]
    assert {d["provenance"]["file_name"] for d in passed} == {"good.md", "inject.txt"}

    quarantined = [
        json.loads(x)
        for x in s.read_text(f"{base}/quarantine/intake.jsonl").splitlines()
        if x.strip()
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["provenance"]["file_name"] == "oversized.txt"
    assert quarantined[0]["category"] == "oversized"
    assert quarantined[0]["reason"]

    # -- Export holds the chunked corpus on s3; nothing derives from the quarantined doc --
    jsonl_uri = f"{base}/export/records.jsonl"
    rows = [json.loads(x) for x in s.read_text(jsonl_uri).splitlines() if x.strip()]
    assert rows, "expected exported chunks on object storage"
    assert "oversized.txt" not in {r["provenance"]["file_name"] for r in rows}
    for row in rows:
        chunk = ChunkRecord.model_validate(row)
        assert chunk.classification.sensitivity == "confidential"  # inherited losslessly on s3
        assert chunk.provenance.char_start is not None
        assert chunk.provenance.page_start is not None

    # -- Offset round-trip on s3: chunk → offset → original source page ------------------
    parse_ckpt = f"{base}/checkpoints/parse.jsonl"
    parsed = {
        json.loads(x)["record_id"]: json.loads(x)
        for x in s.read_text(parse_ckpt).splitlines()
        if x.strip()
    }
    for row in rows:
        parent = parsed[row["document_record_id"]]
        index = PageOffsetIndex(PageMap.model_validate(parent["page_map"]))
        assert row["provenance"]["page_start"] == index.page_for_offset(
            row["provenance"]["char_start"]
        )

    # -- Content Screening flagged the injection; the marker survives to the s3 corpus ----
    inj_flagged = [
        r for r in rows if r.get("risk_markers") and r["provenance"]["file_name"] == "inject.txt"
    ]
    assert inj_flagged
    assert inj_flagged[0]["risk_markers"][0]["category"] == "prompt_injection"
    assert all(not r.get("risk_markers") for r in rows if r["provenance"]["file_name"] == "good.md")

    # -- The Quality Report persisted on s3 carries the dispositions ----------------------
    from latence_core.quality import QualityReport

    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.screening is not None
    assert persisted.screening.quarantined == 1
    assert persisted.screening.intake_screened == 3
    assert persisted.chunk is not None
    assert persisted.chunk.chunks_total == len(rows)
    assert persisted.chunk.offset_preserving is True
    assert report.chunk is not None and report.chunk.chunks_total == persisted.chunk.chunks_total


def test_s3_resume_is_byte_identical_on_object_storage(s3_root: str) -> None:
    """Idempotent resume on ``s3://``: every output artifact is byte-identical (H-E1 Part 1).

    Proves the resume/checkpoint machinery is exact on object storage, not just local disk — a
    resume that dropped/duplicated/reordered a record would change the captured export/checkpoint
    bytes and fail here.
    """
    docs_uri = _upload_corpus(s3_root)
    store = s3_root
    pipeline = spine_pipeline(store, docs_uri)
    s = Storage.for_uri(store)

    LocalRunner().run(pipeline, run_id="r")
    first = output_artifacts(s, store, "r")
    assert first, "the first s3 run must have produced output artifacts"

    report2 = LocalRunner().run(pipeline, run_id="r")
    second = output_artifacts(s, store, "r")

    assert all(m.from_checkpoint for m in report2.stages)
    assert second == first, "resumed s3 output diverged from the first run's"

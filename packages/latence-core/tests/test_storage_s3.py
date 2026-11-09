"""Storage round-trip against an S3-compatible endpoint (an S3 emulator in CI).

Skipped unless ``LATENCE_S3_ENDPOINT`` is set, so the suite stays hermetic on a
laptop while CI proves the same code path works on ``s3://`` with no code change
(ADR-0009). Requires ``s3fs`` (installed only in the CI s3 job).
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LATENCE_S3_ENDPOINT"),
    reason="set LATENCE_S3_ENDPOINT (and AWS creds) to run the S3 test",
)


def _storage_options() -> dict[str, Any]:
    endpoint = os.environ["LATENCE_S3_ENDPOINT"]
    return {
        "key": os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": endpoint},
    }


def test_s3_roundtrip() -> None:
    pytest.importorskip("s3fs")
    from latence_core.storage import Storage

    bucket = os.environ.get("LATENCE_S3_BUCKET", "latence-test")
    s = Storage(protocol="s3", **_storage_options())
    uri = f"s3://{bucket}/{uuid.uuid4().hex}/records.jsonl"
    s.write_atomic(uri, b"line\n")
    assert s.exists(uri)
    assert s.read_bytes(uri) == b"line\n"

"""End-to-end: a SharePoint-rooted corpus through the real LocalRunner, plus the conformance gate.

Two proofs the project bar asks for:

1. **A real pipeline** — ``source.sharepoint`` → Parse → Chunk → Export on
   :class:`~latence_core.runner.LocalRunner`, configured from YAML exactly as an adopter would
   (including the ``transport`` factory reference that selects the offline drive), with the
   deliverable read back off Storage. The assertion is that SharePoint Provenance — the source
   system AND the resolvable document identity — survives every Stage into the exported corpus.
2. **The Provider passes conformance (C1–C6)** as a registered plugin.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage
from latence_source_sharepoint.provider import SharePointSource
from latence_source_sharepoint.testing import DEMO_TOKEN

if TYPE_CHECKING:
    from pathlib import Path

TOKEN_ENV = "LATENCE_SHAREPOINT_DEMO_TOKEN"


def test_sharepoint_provenance_survives_into_the_exported_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, DEMO_TOKEN)
    store = f"file://{tmp_path}/state"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: sharepoint-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.sharepoint
    config:
      drive_id: demo-drive
      sensitivity: internal
      transport: 'latence_source_sharepoint.testing:demo_transport'
      auth:
        mode: env_token
        token_env: {TOKEN_ENV}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
    config:
      basename: corpus
"""
    )
    LocalRunner().run(pipeline, run_id="sharepoint-e2e")

    storage = Storage.for_uri(store)
    jsonl_uri = f"{store}/_latence/runs/sharepoint-e2e/export/corpus.jsonl"
    assert storage.exists(jsonl_uri)
    rows = [json.loads(line) for line in storage.read_text(jsonl_uri).splitlines() if line.strip()]
    assert rows, "no corpus rows exported from the SharePoint drive"
    for row in rows:
        provenance = row["provenance"]
        assert provenance["source_system"] == "sharepoint"
        assert provenance["source_uri"].startswith("sharepoint://demo-drive/")
        assert "itemId=" in provenance["source_uri"]
    assert {row["provenance"]["file_name"] for row in rows} == {
        "nda.md",
        "onboarding.md",
        "security.md",
    }


def test_no_credential_reaches_the_persisted_run_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The framework persists no credential: the token exists only in the environment."""
    monkeypatch.setenv(TOKEN_ENV, DEMO_TOKEN)
    store = f"file://{tmp_path}/state"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: sharepoint-manifest
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.sharepoint
    config:
      drive_id: demo-drive
      transport: 'latence_source_sharepoint.testing:demo_transport'
      auth:
        mode: env_token
        token_env: {TOKEN_ENV}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
"""
    )
    LocalRunner().run(pipeline, run_id="sharepoint-manifest")
    manifest = Storage.for_uri(store).read_text(
        f"{store}/_latence/runs/sharepoint-manifest/manifest.json"
    )
    # The token VALUE lives only in the environment; the manifest records the drive, and core's
    # own scrubber additionally masks the token_env key on the way to disk.
    assert DEMO_TOKEN not in manifest
    assert "demo-drive" in manifest


def test_provider_discovered_via_entry_point() -> None:
    registry = ProviderRegistry()
    assert "source.sharepoint" in registry.names()
    assert registry.load("source.sharepoint") is SharePointSource
    profile = registry.profile("source.sharepoint")
    assert profile is not None
    assert profile.license == "Apache-2.0"
    assert profile.license_verified is True
    assert profile.model_id is None  # a connector ships no weights


def test_provider_passes_conformance_c1_to_c6() -> None:
    """``source.sharepoint`` passes the P3-F3 conformance gate against the faithful Graph fake."""
    verdict = run_conformance(ProviderRegistry(), "source.sharepoint")
    assert verdict.capability == "source"
    assert verdict.passed, [f"{c.check}:{c.status}:{c.detail}" for c in verdict.checks]
    statuses = {c.check: c.status.value for c in verdict.checks}
    assert statuses["C1_valid_records"] == "pass"
    assert statuses["C2_graceful_failure"] == "pass"
    assert statuses["C3_license"] == "pass"
    assert statuses["C4_determinism"] == "pass"  # declared deterministic, and it is
    assert statuses["C5_device"] == "pass"
    assert statuses["C6_no_pii_leak"] == "pass"

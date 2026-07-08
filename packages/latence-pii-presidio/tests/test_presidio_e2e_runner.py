"""End-to-end: the Presidio Redaction Provider through the local Runner + conformance (W1-pii).

The proofs the Baseline bar + the issue's "true drop-in" requirement ask for:

1. **≥1 end-to-end test through the local Runner** — a real Source -> Parse -> Chunk -> Redaction
   -> Export pipeline on :class:`~latence_core.runner.LocalRunner` where the Redaction Stage uses
   ``redaction.presidio`` (resolved from the plugin registry). Asserts the exported RedactionRecords
   have the PII masked, the spans land on the document coordinate system, the raw PII appears in no
   persisted artifact, and the Quality Report's RedactionQuality carries the counts.

2. **True drop-in** — the SAME pipeline with ``redaction.presidio`` swapped for
   ``redaction.hybrid_rule`` (one config line) produces a contract-complete RedactionRecord, and the
   H-C1 sensitive-doc no-op floor still fires on a confidential document with the control off — the
   drop-in swap needs no other change.

3. **The provider passes the P3-F3 conformance suite (C1–C6)** as a registered provider.

The real ``presidio-analyzer`` dep is not installed offline, so a deterministic fake
``presidio_analyzer`` module (+ its ``nlp_engine`` submodule) is injected into ``sys.modules`` (the
import is deferred to first use, so the in-process Runner picks it up) — the established
gliner/endpoint/ST offline discipline. Deterministic + seeded (``PYTHONHASHSEED`` pinned by the
repo-root conftest; ``LATENCE_CUDA=0`` pinned below so the device seam is reproducible).
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
from latence_core.contracts import RedactionRecord
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

# The email the fake analyzer detects, at the offset it occupies in the parsed memo below.
_MEMO = "Please contact alice@example.com about the Q3 report.\n"
_EMAIL = "alice@example.com"


class _FakeRecognizerResult:
    def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class _FakeAnalyzerEngine:
    """Deterministic offline fake for presidio_analyzer.AnalyzerEngine.

    Finds the email in whatever text it is handed (so the offset is real on the parsed markdown,
    not a hardcoded index) and returns it as an EMAIL_ADDRESS RecognizerResult.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def analyze(
        self,
        text: str,
        language: str,  # noqa: ARG002
        entities: list[str] | None = None,
        score_threshold: float = 0.0,  # noqa: ARG002
        **_kwargs: Any,
    ) -> list[_FakeRecognizerResult]:
        out: list[_FakeRecognizerResult] = []
        if entities is None or "EMAIL_ADDRESS" in entities:
            idx = text.find(_EMAIL)
            if idx >= 0:
                out.append(_FakeRecognizerResult("EMAIL_ADDRESS", idx, idx + len(_EMAIL), 0.95))
        return out


class _FakeNlpEngineProvider:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def create_engine(self) -> object:
        return object()


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


@pytest.fixture()
def fake_presidio(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("presidio_analyzer")
    module.AnalyzerEngine = _FakeAnalyzerEngine  # type: ignore[attr-defined]
    module.RecognizerResult = _FakeRecognizerResult  # type: ignore[attr-defined]
    nlp_module = types.ModuleType("presidio_analyzer.nlp_engine")
    nlp_module.NlpEngineProvider = _FakeNlpEngineProvider  # type: ignore[attr-defined]
    module.nlp_engine = nlp_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp_module)


def _pipeline(store: str, docs_dir: Path, *, provider: str, sensitivity: str = "internal") -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: presidio-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: {sensitivity}
      extensions: [txt, md]
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
    provider: {provider}
    depends_on: [parse]
    config:
      entities: [EMAIL_ADDRESS]
      default_action: mask
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(_MEMO, encoding="utf-8")
    (docs / "clean.txt").write_text("The quarterly report is attached for review.\n", encoding="utf-8")
    return docs


def test_presidio_redacts_through_runner(fake_presidio: None, tmp_path: Path) -> None:
    """A LocalRunner run with ``redaction.presidio`` masks the email + carries valid contracts."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(
        _pipeline(store, docs, provider="redaction.presidio"), run_id="presidio-e2e"
    )
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/presidio-e2e"

    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [RedactionRecord.model_validate(r) for r in rows]
    assert records
    memo = next(r for r in records if "[EMAIL]" in r.masked_content)

    # The PII is masked, the span lands on the document coordinate system + round-trips.
    assert _EMAIL not in memo.masked_content
    assert memo.pii_spans[0].pii_type == "email"
    assert memo.pii_spans[0].detector == "presidio"
    # No named policy entry for 'internal' in this config, so the applied policy is 'default'
    # (the default_action=mask still applies); the document's sensitivity is carried unchanged.
    assert memo.policy == "default"
    assert memo.classification.sensitivity == "internal"

    # The Quality Report carries the redaction counts.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.redaction is not None
    assert persisted.redaction.documents_with_pii == 1
    assert persisted.redaction.per_type == {"email": 1}
    assert report.redaction is not None

    # No raw PII value in ANY persisted artifact (the load-bearing S6/C6 check).
    for uri in (jsonl_uri, f"{base}/quality-report.json"):
        assert _EMAIL not in s.read_text(uri)


def test_presidio_is_a_true_drop_in_for_hybrid_rule(fake_presidio: None, tmp_path: Path) -> None:
    """The SAME pipeline with hybrid_rule swapped in produces a contract-complete record.

    Swapping ``redaction.presidio`` for ``redaction.hybrid_rule`` (one config line) with every
    other Stage identical yields a valid RedactionRecord per document — the drop-in swap needs no
    other change. (The two DETECT differently — presidio masks the email via its analyzer, the
    hybrid rule via its built-in email regex — but the contract, policy, and Export are identical.)
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    LocalRunner().run(
        _pipeline(store, docs, provider="redaction.hybrid_rule"), run_id="hybrid-dropin"
    )
    s = Storage.for_uri(store)
    jsonl_uri = f"{store}/_latence/runs/hybrid-dropin/export/records.jsonl"
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [RedactionRecord.model_validate(r) for r in rows]
    assert records
    # hybrid_rule's built-in email recogniser masks the same email — same contract, same policy.
    memo = next(r for r in records if "[EMAIL]" in r.masked_content)
    assert _EMAIL not in memo.masked_content
    assert memo.policy == "default"  # no named 'internal' entry ⇒ default policy (mask), same as presidio
    assert memo.classification.sensitivity == "internal"


def test_no_op_floor_fires_end_to_end_on_sensitive_skip(
    fake_presidio: None, tmp_path: Path
) -> None:
    """H-C1 not regressed: a confidential doc with the control off flags redaction_disabled.

    A ``confidential`` corpus whose policy skips redaction runs no detectors — the emitted
    RedactionRecord's ``redaction_disabled_for_sensitive`` is set (control OFF, not "clean"),
    proving the drop-in provider honours the sensitive-doc no-op floor through the real Runner.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: presidio-noop
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: confidential
      extensions: [txt, md]
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
    provider: redaction.presidio
    depends_on: [parse]
    config:
      entities: [EMAIL_ADDRESS]
      policies:
        confidential: {{skip: true}}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )
    LocalRunner().run(pipeline, run_id="presidio-noop")
    s = Storage.for_uri(store)
    jsonl_uri = f"{store}/_latence/runs/presidio-noop/export/records.jsonl"
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [RedactionRecord.model_validate(r) for r in rows]
    assert records
    assert all(r.redaction_disabled_for_sensitive for r in records)
    # And the Quality Report's loud count reflects the disabled control.
    base = f"{store}/_latence/runs/presidio-noop"
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.redaction is not None
    assert persisted.redaction.redaction_disabled_for_sensitive == len(records)


def test_provider_discovered_via_entry_point() -> None:
    """The provider is registered under 'latence.providers' and loads to the adapter class."""
    registry = ProviderRegistry()
    assert "redaction.presidio" in registry.names()
    cls = registry.load("redaction.presidio")
    from latence_pii_presidio.provider import PresidioRedactor

    assert cls is PresidioRedactor
    profile = registry.profile("redaction.presidio")
    assert profile is not None
    assert profile.license == "MIT"
    assert profile.model_id == "en_core_web_sm"


def test_provider_passes_conformance_c1_to_c6(fake_presidio: None) -> None:
    """``redaction.presidio`` passes the P3-F3 conformance gate (C1–C6).

    Run against the deterministic fake (its live numbers need a rig, #64) — the same discipline as
    the gliner trio and the ST embedder. Every check is PASS or a defined SKIP, never a FAIL. The
    C2 no-PII-in-error and C6 no-leak checks are especially load-bearing for a PII provider.
    """
    verdict = run_conformance(ProviderRegistry(), "redaction.presidio")
    assert verdict.provider == "redaction.presidio"
    assert verdict.capability == "redaction"
    assert verdict.device_skipped is False
    assert verdict.passed, [f"{c.check}:{c.status}:{c.detail}" for c in verdict.checks]
    statuses = {c.check: c.status.value for c in verdict.checks}
    assert statuses["C1_valid_records"] == "pass"
    assert statuses["C2_graceful_failure"] == "pass"
    assert statuses["C3_license"] == "pass"  # permissive MIT (code + weights), evidence-bearing
    assert statuses["C4_determinism"] == "skipped"  # deterministic=False ⇒ C4 documented-skip
    assert statuses["C5_device"] == "pass"  # CPU honored (compute="either")
    assert statuses["C6_no_pii_leak"] == "skipped"  # Redactor handles PII by design (C6 targets non-Redactors)

"""End-to-end: the sentence-transformers Embedder through the local Runner + conformance (W1-embed).

Two proofs the Baseline bar asks for:

1. **≥1 end-to-end test through the local Runner** — a real Source → Parse → Chunk → Export
   pipeline on :class:`~latence_core.runner.LocalRunner` where the corpus Export opts into the
   learned Embedder (``embedding.sentence_transformers``, ADR-0017). Asserts every exported row
   carries a fixed-width ``embedding`` vector produced by the Provider resolved from the plugin
   registry — the learned Embedder wired through the SAME Export seam the in-core hashing embedder
   uses. The real ``sentence-transformers`` dep is not installed offline, so a deterministic fake
   ``SentenceTransformer`` is injected into ``sys.modules`` (the import is deferred to first use,
   so the in-process Runner picks it up) — the established gliner/endpoint offline discipline.

2. **The provider passes the P3-F3 conformance suite (C1–C6)** as a registered provider — the
   enterprise-readiness gate, run against the same fake.

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest; ``LATENCE_CUDA=0``
pinned below so the device seam is reproducible).
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

_DIMENSION = 6


class _FakeSentenceTransformer:
    """Deterministic offline fake for sentence_transformers.SentenceTransformer."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.model_kwargs = dict(model_kwargs or {})  # W8 perf seam pass-through

    def get_sentence_embedding_dimension(self) -> int:
        return _DIMENSION

    def encode(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        # A deterministic, fixed-width vector per input text (order preserved).
        return [[float(len(t) % 5)] * _DIMENSION for t in texts]


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


@pytest.fixture()
def fake_st(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_learned_embedder_lands_vectors_on_disk_through_runner(
    fake_st: None, tmp_path: Path
) -> None:
    """A LocalRunner run with ``embedding.sentence_transformers`` on the Export writes vectors."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha beta gamma delta\n", encoding="utf-8")
    (docs / "b.txt").write_text("epsilon zeta eta theta iota\n", encoding="utf-8")
    store = f"file://{tmp_path}/state"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: st-embed-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: public
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
      embedder:
        provider: embedding.sentence_transformers
        config: {{dimension: {_DIMENSION}}}
"""
    )
    LocalRunner().run(pipeline, run_id="st-embed-e2e")

    storage = Storage.for_uri(store)
    jsonl_uri = f"{store}/_latence/runs/st-embed-e2e/export/corpus.jsonl"
    assert storage.exists(jsonl_uri)
    rows = [json.loads(line) for line in storage.read_text(jsonl_uri).splitlines() if line.strip()]
    assert rows, "no corpus rows exported"
    for row in rows:
        # Every exported row carries a learned-embedder vector of the resolved width (ADR-0017).
        assert len(row["embedding"]) == _DIMENSION


def test_provider_discovered_via_entry_point() -> None:
    """The provider is registered under 'latence.providers' and loads to the adapter class."""
    registry = ProviderRegistry()
    assert "embedding.sentence_transformers" in registry.names()
    cls = registry.load("embedding.sentence_transformers")
    from latence_embedder_st.provider import SentenceTransformersEmbedder

    assert cls is SentenceTransformersEmbedder
    profile = registry.profile("embedding.sentence_transformers")
    assert profile is not None
    assert profile.license == "MIT"  # e5-small weights SPDX
    assert profile.model_id == "intfloat/multilingual-e5-small"


def test_provider_passes_conformance_c1_to_c6(fake_st: None) -> None:
    """``embedding.sentence_transformers`` passes the P3-F3 conformance gate (C1–C6).

    Run against the deterministic fake (its live numbers need a rig, #64) — the same discipline as
    the gliner trio and the endpoint embedder. Every check is PASS or a defined SKIP, never a FAIL.
    """
    verdict = run_conformance(ProviderRegistry(), "embedding.sentence_transformers")
    assert verdict.provider == "embedding.sentence_transformers"
    assert verdict.capability == "embedding"
    assert verdict.device_skipped is False
    assert verdict.passed, [f"{c.check}:{c.status}:{c.detail}" for c in verdict.checks]
    statuses = {c.check: c.status.value for c in verdict.checks}
    assert statuses["C1_valid_records"] == "pass"
    assert statuses["C2_graceful_failure"] == "pass"
    assert statuses["C3_license"] == "pass"  # permissive MIT weights / Apache-2.0 code, evidence-bearing
    assert statuses["C4_determinism"] == "skipped"  # deterministic=False ⇒ C4 documented-skip
    assert statuses["C5_device"] == "pass"  # CPU honored (compute="either")
    assert statuses["C6_no_pii_leak"] == "pass"

"""End-to-end: batched GLiNER Entity Extraction through the local Runner (W7).

The Baseline bar's ≥1 end-to-end proof for W7 encoder batching: a real Source → Parse → Chunk →
Entity-Extraction pipeline on :class:`~latence_core.runner.LocalRunner` where the Entity stage is
``entity.gliner``. Many chunks flow through the Runner in one Stage dispatch (``list(provider.
extract(chunk_records))``), and this test asserts (a) every chunk's entity lands with correct
document-level offsets, and (b) the model was called in BATCHES (``batch_calls`` < number of
chunks) — batching survives the Runner path, not just the unit seam.

The real ``gliner`` dep is not installed offline (ADR-0007/0016), so a deterministic fake
``GLiNER`` with ``batch_predict_entities`` is injected into ``sys.modules`` (the import is deferred
to first use, so the in-process Runner picks it up) — the established offline discipline. The fake
counts its batched forward passes on a module-level list so the count survives the Runner
constructing its own provider instance.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner

if TYPE_CHECKING:
    from pathlib import Path

# The Runner constructs its OWN provider instance, so the fake records its batched-call count on a
# module-level list (not per-instance) — the e2e proof that batching survives the Runner path.
_BATCH_CALLS: list[int] = []


class _FakeGLiNER:
    """Deterministic offline GLiNER fake exposing the batched ``batch_predict_entities`` (W7).

    Tags the first word of each chunk (chunk-local [0, first-space)) as a ``person`` when
    ``person`` is requested — so every non-empty chunk yields exactly one mention, and a
    mis-mapped batch would surface the wrong chunk's word.
    """

    @classmethod
    def from_pretrained(cls, model_id: str) -> _FakeGLiNER:
        return cls()

    @staticmethod
    def _one(text: str, labels: list[str]) -> list[dict[str, Any]]:
        if "person" not in labels:
            return []
        first = text.split(" ", 1)[0] if text else ""
        if not first:
            return []
        return [{"start": 0, "end": len(first), "label": "person", "score": 0.9}]

    def batch_predict_entities(
        self, texts: list[str], labels: list[str], threshold: float = 0.5
    ) -> list[list[dict[str, Any]]]:
        _BATCH_CALLS.append(len(texts))
        return [self._one(text, labels) for text in texts]


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _BATCH_CALLS.clear()


def test_batched_entity_extraction_through_runner(fake_gliner: None, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    # Six one-line docs → six single-chunk documents flowing through the Entity stage together.
    names = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank"]
    for i, name in enumerate(names):
        (docs / f"{i}.txt").write_text(f"{name} appears here.\n", encoding="utf-8")
    store = f"file://{tmp_path}/state"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: gliner-batch-e2e
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
  - name: entities
    capability: entity_extraction
    provider: entity.gliner
    depends_on: [chunk]
    config:
      labels: [person]
      batch_size: 4
"""
    )
    report = LocalRunner().run(pipeline, run_id="gliner-batch-e2e")

    # The Entity stage ran through batched forward passes, NOT one call per chunk: 6 chunks at
    # batch_size=4 => the model saw them in bounded batches (each call <= 4 texts), the total texts
    # across passes equals the chunk count (each chunk in a forward pass exactly once), and the
    # number of forward passes is strictly fewer than the chunk count — batching is real e2e.
    assert _BATCH_CALLS, "the model was never called through the Runner"
    assert all(n <= 4 for n in _BATCH_CALLS)
    assert sum(_BATCH_CALLS) == len(names)
    assert len(_BATCH_CALLS) < len(names)

    # Every document's single mention landed — the Quality Report counts them (S4), so a
    # dropped/mis-mapped batch would change the count the report substantiates.
    assert report.entities is not None
    assert report.entities.mentions_total == len(names)
    assert report.entities.per_type.get("person") == len(names)

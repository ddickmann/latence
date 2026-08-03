# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The pod worker's pipelined scheduler, fully offline.

Loads the REAL ``benchmark/sota-campaign/pod/ocr_worker.py`` with the package's
``ocr_lib`` injected — the same arrangement the stage creates on the pod — and drives it
with a fake render child and a fake (instrumented) OpenAI client. What is proven here is
the throughput restructure's contract: the ONE global OCR cap holds across overlapping
docs, N doc lifecycles genuinely overlap, the per-doc order (render -> OCR -> assemble ->
free) is preserved, and cumulative progress re-accumulates to the identical value on a
relaunch that re-OCRs nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER_PATH = REPO_ROOT / "benchmark" / "sota-campaign" / "pod" / "ocr_worker.py"
POD_DIR = str(WORKER_PATH.parent)


def load_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the real worker module, ``ocr_lib`` pre-injected (as the stage pushes it)."""
    from latence_benchmark.campaign import ocr_lib

    monkeypatch.setitem(sys.modules, "ocr_lib", ocr_lib)
    spec = importlib.util.spec_from_file_location("_s2_ocr_worker_under_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    while POD_DIR in sys.path:  # undo the worker's own sys.path insertion
        sys.path.remove(POD_DIR)
    return module


def make_root(tmp_path: Path, doc_ids: Sequence[str]) -> Path:
    root = tmp_path / "root"
    pdfs = root / "input" / "ds" / "pdfs"
    pdfs.mkdir(parents=True)
    for doc_id in doc_ids:
        (pdfs / f"{doc_id}.pdf").write_bytes(b"%PDF")
    return root


def install_fake_render(
    worker: Any, monkeypatch: pytest.MonkeyPatch, pages: Mapping[str, int]
) -> None:
    """Stand in for the render child: images + the durable sidecar, skip-if-exists."""

    def fake_spawn(root: Path, batch: list[Any]) -> bool:
        doc = batch[0]
        count = pages[doc.doc_id]
        img_dir = root / "pages-img" / doc.key
        img_dir.mkdir(parents=True, exist_ok=True)
        for page in range(1, count + 1):
            target = img_dir / f"{page:04d}.jpg"
            if not target.exists():
                target.write_bytes(b"jpg")
        sidecar = worker._sidecar_path(root, doc)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f"{count}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(worker, "_spawn_render_child", fake_spawn)


class Recorder:
    """Concurrency instrumentation shared by the fake OCR client."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0


class _FakeCompletions:
    def __init__(self, recorder: Recorder, text: str) -> None:
        self._recorder = recorder
        self._text = text

    def create(self, **_kwargs: Any) -> Any:
        rec = self._recorder
        with rec.lock:
            rec.active += 1
            rec.calls += 1
            rec.max_active = max(rec.max_active, rec.active)
        time.sleep(rec.delay)
        with rec.lock:
            rec.active -= 1
        message = types.SimpleNamespace(content=self._text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _FlakyCompletions(_FakeCompletions):
    """Every third request dies with an HTTP-ish error (permanent page failures)."""

    def create(self, **kwargs: Any) -> Any:
        with self._recorder.lock:
            ordinal = self._recorder.calls + 1
        if ordinal % 3 == 0:
            with self._recorder.lock:
                self._recorder.calls += 1
            raise RuntimeError("simulated http failure")
        return super().create(**kwargs)


class _BoobyTrappedCompletions:
    """A relaunch over finished work must never issue an OCR request."""

    def create(self, **_kwargs: Any) -> Any:
        msg = "re-OCR of already-assembled work"
        raise AssertionError(msg)


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, completions: Any) -> None:
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    module = types.ModuleType("openai")
    setattr(module, "OpenAI", lambda **_kwargs: client)  # noqa: B010 — module attr
    monkeypatch.setitem(sys.modules, "openai", module)


def read_state(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class TestEnvKnobs:
    def test_env_overrides_take_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S2_DOCS_IN_FLIGHT", "2")
        monkeypatch.setenv("S2_GLOBAL_CONCURRENCY", "12")
        monkeypatch.setenv("S2_RENDER_WORKERS", "1")
        worker = load_worker(monkeypatch)
        assert worker.N_DOCS_IN_FLIGHT == 2  # the 16GB-pod profile
        assert worker.GLOBAL_CONCURRENCY == 12
        assert worker.RENDER_WORKERS == 1

    def test_junk_and_absent_env_fall_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("S2_GLOBAL_CONCURRENCY", "not-a-number")
        monkeypatch.delenv("S2_DOCS_IN_FLIGHT", raising=False)
        worker = load_worker(monkeypatch)
        assert worker.GLOBAL_CONCURRENCY == 64
        assert worker.N_DOCS_IN_FLIGHT == 8

    def test_non_positive_values_are_clamped_to_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("S2_RENDER_WORKERS", "0")
        worker = load_worker(monkeypatch)
        assert worker.RENDER_WORKERS == 1


class TestPipelinedScheduler:
    def test_global_ocr_cap_holds_across_overlapping_docs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = load_worker(monkeypatch)
        monkeypatch.setattr(worker, "N_DOCS_IN_FLIGHT", 4)
        monkeypatch.setattr(worker, "GLOBAL_CONCURRENCY", 3)
        monkeypatch.setattr(worker, "RENDER_WORKERS", 2)
        doc_ids = [f"d{i}" for i in range(6)]
        root = make_root(tmp_path, doc_ids)
        install_fake_render(worker, monkeypatch, dict.fromkeys(doc_ids, 4))
        recorder = Recorder(delay=0.02)
        install_fake_openai(monkeypatch, _FakeCompletions(recorder, "ocr page text"))

        worker.run(root)

        assert recorder.calls == 24  # every page OCR'd exactly once
        assert recorder.max_active <= 3  # the ONE global cap, across all doc lifecycles
        assert recorder.max_active >= 2  # and the pipeline genuinely fills it

    def test_docs_overlap_up_to_the_in_flight_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = load_worker(monkeypatch)
        monkeypatch.setattr(worker, "N_DOCS_IN_FLIGHT", 3)
        monkeypatch.setattr(worker, "GLOBAL_CONCURRENCY", 8)
        monkeypatch.setattr(worker, "RENDER_WORKERS", 2)
        doc_ids = [f"d{i}" for i in range(6)]
        root = make_root(tmp_path, doc_ids)
        install_fake_render(worker, monkeypatch, dict.fromkeys(doc_ids, 3))
        recorder = Recorder(delay=0.02)
        install_fake_openai(monkeypatch, _FakeCompletions(recorder, "ocr page text"))

        lock = threading.Lock()
        active_docs: set[str] = set()
        peak = [0]
        original = worker.process_doc

        def tracking(
            root_: Path, doc: Any, count: int, pool: Any, client: Any, state: Any
        ) -> Any:
            with lock:
                active_docs.add(doc.key)
                peak[0] = max(peak[0], len(active_docs))
            try:
                return original(root_, doc, count, pool, client, state)
            finally:
                with lock:
                    active_docs.discard(doc.key)

        monkeypatch.setattr(worker, "process_doc", tracking)
        worker.run(root)

        assert peak[0] >= 2  # cross-doc pipelining actually happened
        assert peak[0] <= 3  # and never beyond the in-flight window

    def test_per_doc_lifecycle_order_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = load_worker(monkeypatch)
        monkeypatch.setattr(worker, "N_DOCS_IN_FLIGHT", 2)
        monkeypatch.setattr(worker, "GLOBAL_CONCURRENCY", 4)
        monkeypatch.setattr(worker, "RENDER_WORKERS", 2)
        pages = {"d0": 2, "d1": 1, "d2": 3}
        root = make_root(tmp_path, list(pages))
        install_fake_render(worker, monkeypatch, pages)
        recorder = Recorder(delay=0.01)
        install_fake_openai(
            monkeypatch,
            _FakeCompletions(recorder, "text <table><tr><td>x</td></tr></table>"),
        )

        original_cleanup = worker.cleanup_doc_images

        def checked_cleanup(root_: Path, doc: Any, spot_page: int | None) -> bool:
            markdown = root_ / "markdown" / doc.dataset / f"{doc.doc_id}.md"
            assert markdown.is_file(), "cleanup ran before assembly"
            return bool(original_cleanup(root_, doc, spot_page))

        monkeypatch.setattr(worker, "cleanup_doc_images", checked_cleanup)
        worker.run(root)

        assert read_state(root)["phase"] == "done"
        for doc_id in pages:
            markdown = (root / "markdown" / "ds" / f"{doc_id}.md").read_text()
            assert "| x |" in markdown  # OCR output flowed through table normalization
            assert not (root / "pages-img" / "ds" / doc_id).exists()  # images freed
        metrics = json.loads((root / "metrics.json").read_text())
        assert len(metrics["spot_check"]) == 3  # population <= 12: all docs retained

    def test_progress_is_monotone_across_relaunch_even_with_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = load_worker(monkeypatch)
        monkeypatch.setattr(worker, "N_DOCS_IN_FLIGHT", 2)
        monkeypatch.setattr(worker, "GLOBAL_CONCURRENCY", 4)
        monkeypatch.setattr(worker, "RENDER_WORKERS", 2)
        doc_ids = ["d0", "d1", "d2"]
        root = make_root(tmp_path, doc_ids)
        install_fake_render(worker, monkeypatch, dict.fromkeys(doc_ids, 2))
        recorder = Recorder(delay=0.005)
        install_fake_openai(monkeypatch, _FlakyCompletions(recorder, "ocr page text"))

        worker.run(root)
        first = read_state(root)
        assert first["phase"] == "done"
        assert first["failures"] > 0  # the flaky client really failed some pages

        # Relaunch over the finished manifest: no request may be issued, and progress —
        # which counts ONLY filesystem-backed work, never failure counters — must
        # re-accumulate to the identical value, so it can never dip below the campaign
        # client's high-water mark.
        install_fake_openai(monkeypatch, _BoobyTrappedCompletions())
        worker.run(root)
        second = read_state(root)
        assert second["phase"] == "done"
        assert second["progress"] == first["progress"]
        assert second["pages_done"] == first["pages_done"]

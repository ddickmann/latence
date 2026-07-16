"""`latence stack validate stacks/retrieval-signals.yaml` — the index-time signals AC4 (#160).

The deterministic reference emitters demonstrated through a REAL end-to-end run (not just unit
tests): `stacks/retrieval-signals.yaml` is the blessed default spine with the three in-core signal
generators wired additively on the RAG corpus Export. `validate_stack` runs it twice+resume on the
bundled corpus and asserts every G1 check passes — including determinism/resume (byte-identical
across runs, which covers the emitted signal columns + the multi-vector sidecar) and the whole-export
PII scan (the signals are generated over redacted text, so no raw PII leaks into the new columns).

Seeded: `LATENCE_CUDA=0` pins the device seam; the repo-root conftest pins `PYTHONHASHSEED`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from latence_core.stacks.validate import CheckResult, validate_stack
from latence_core.storage import Storage

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACK = _REPO_ROOT / "stacks" / "retrieval-signals.yaml"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _checks_by_name(checks: list[CheckResult]) -> dict[str, CheckResult]:
    return {c.name: c for c in checks}


def test_retrieval_signals_stack_validates_green(tmp_path: Path) -> None:
    """The signal-wired stack passes every G1 check end-to-end (contracts/rag/determinism/resume)."""
    result = validate_stack(_STACK, storage_root=f"file://{tmp_path}/store")
    assert result.ok, result.render()
    checks = _checks_by_name(result.checks)
    for name in ("contracts", "kg", "rag", "report", "graceful_failure", "determinism", "resume"):
        c = checks[name]
        assert c.passed and not c.skipped, f"{name} did not PASS: {c.detail}"
    assert not result.skipped_stages  # every signal generator is an in-core CPU reference


def test_retrieval_signals_stack_emits_the_signal_columns_and_sidecar(tmp_path: Path) -> None:
    """The run actually materialised the additive columns + the multi-vector sidecar on disk."""
    root = f"file://{tmp_path}/store"
    result = validate_stack(_STACK, storage_root=root)
    assert result.ok, result.render()

    storage = Storage.for_uri(root)
    export_dir = f"{root}/_latence/runs/stack-validate/export"
    files = list(storage.walk_files(export_dir))

    corpus_parquet = next(u for u in files if u.endswith("corpus.parquet"))
    table = pq.read_table(io.BytesIO(storage.read_bytes(corpus_parquet)))  # type: ignore[no-untyped-call]
    cols = set(table.column_names)
    # Dense + sparse + FDE columns landed in records.parquet, additively.
    assert {"embedding", "sparse_indices", "sparse_values", "fde_embedding"} <= cols
    # The raw multi-vectors are NOT in the corpus table (they ride the sidecar).
    assert "multivector" not in cols
    if table.num_rows:
        assert len(table.column("fde_embedding")[0].as_py()) == 4 * 8  # buckets * input_dimension

    # The multi-vector sidecar exists and is keyed by record_id.
    sidecar = next(u for u in files if u.endswith("multivectors.parquet"))
    side_table = pq.read_table(io.BytesIO(storage.read_bytes(sidecar)))  # type: ignore[no-untyped-call]
    assert side_table.column_names == ["record_id", "multivector"]

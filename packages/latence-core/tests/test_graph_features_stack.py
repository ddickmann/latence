"""`latence stack validate stacks/graph-features.yaml` — the ADR-0051 graph-feature AC (#161).

The deterministic reference computers demonstrated through a REAL end-to-end run (not just unit
tests): `stacks/graph-features.yaml` is the blessed default spine with the `graph_features.degree`
computer wired on Graph Assembly and a context-enrichment Stage projecting the per-node features onto
each chunk's context_header. `validate_stack` runs it twice+resume on the bundled corpus and asserts
every G1 check passes — including determinism/resume (byte-identical across runs, which covers the
new graph-nodes.parquet centrality/community columns AND the context_header projection) and the
whole-export PII scan.

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
_STACK = _REPO_ROOT / "stacks" / "graph-features.yaml"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _checks_by_name(checks: list[CheckResult]) -> dict[str, CheckResult]:
    return {c.name: c for c in checks}


def test_graph_features_stack_validates_green(tmp_path: Path) -> None:
    """The feature-wired stack passes every G1 check end-to-end (contracts/kg/rag/determinism)."""
    result = validate_stack(_STACK, storage_root=f"file://{tmp_path}/store")
    assert result.ok, result.render()
    checks = _checks_by_name(result.checks)
    for name in ("contracts", "kg", "rag", "report", "graceful_failure", "determinism", "resume"):
        c = checks[name]
        assert c.passed and not c.skipped, f"{name} did not PASS: {c.detail}"
    assert not result.skipped_stages  # the graph-feature computer is an in-core CPU reference


def test_graph_features_stack_emits_the_node_feature_columns(tmp_path: Path) -> None:
    """The run actually materialised the centrality + community columns on graph-nodes.parquet."""
    root = f"file://{tmp_path}/store"
    result = validate_stack(_STACK, storage_root=root)
    assert result.ok, result.render()

    storage = Storage.for_uri(root)
    export_dir = f"{root}/_latence/runs/stack-validate/export"
    files = list(storage.walk_files(export_dir))

    nodes_parquet = next(u for u in files if u.endswith("graph-nodes.parquet"))
    table = pq.read_table(io.BytesIO(storage.read_bytes(nodes_parquet)))  # type: ignore[no-untyped-call]
    cols = set(table.column_names)
    assert {"centrality", "community"} <= cols
    if table.num_rows:
        # Every node was stamped (the assembler stamps all-or-none), so no null cells.
        assert all(c is not None for c in table.column("centrality").to_pylist())
        assert all(bool(c) for c in table.column("community").to_pylist())

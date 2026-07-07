"""CLI smoke tests for `latence-demo` (S13).

The one-command entrypoint must run the whole demo, print the summary + the viewer path, and
exit non-zero (a built-in smoke test) when a planted dangerous fixture slips through.
"""

from __future__ import annotations

from pathlib import Path

import latence_demo.cli as cli_module
import pytest
from latence_core.capability import ProviderRegistry
from latence_demo.builder import DemoResult
from latence_demo.cli import app
from latence_demo.contracts import (
    DangerousFixtureOutcome,
    DemoRunSummary,
    GraphSummary,
)
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_runs_and_reports(tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    result = runner.invoke(app, ["--storage-uri", store, "--run-id", "cli"])
    assert result.exit_code == 0, result.output
    assert "Latence demo" in result.output
    assert "quarantined" in result.output
    assert "Open the viewer:" in result.output
    # The viewer file the CLI pointed at actually exists.
    assert (tmp_path / "store/_latence/runs/cli/export/index.html").exists()


def test_cli_strict_flag_present(tmp_path: Path) -> None:
    # A clean demo run passes strict mode (every dangerous fixture caught + contracts complete).
    store = f"file://{tmp_path}/store"
    result = runner.invoke(app, ["--storage-uri", store, "--run-id", "cli-strict", "--strict"])
    assert result.exit_code == 0, result.output


# --- Strict-mode failure path (the "smoke test, not just a display" gate, cli.py:69-75). ---
#
# The clean corpus always catches every fixture, so we drive the two failure branches by
# stubbing DemoBuilder to return a summary that reports a slip-through / incomplete contracts.
# This is the demo's key failure mode: a regression that inverts or deletes the strict-mode
# `raise typer.Exit(code=1)` must turn these tests red.


def _summary(*, dangerous_matched: bool, contracts_complete: bool) -> DemoRunSummary:
    return DemoRunSummary(
        pipeline_name="latence-demo",
        run_id="cli-fail",
        storage_uri="file:///tmp/x",
        run_dir="file:///tmp/x/_latence/runs/cli-fail",
        site_uri="file:///tmp/x/_latence/runs/cli-fail/export/index.html",
        documents_ingested=4,
        quarantined=3,
        flagged=1,
        graph=GraphSummary(),
        dangerous=[
            DangerousFixtureOutcome(
                file_name="archive.zip",
                expected="quarantine:zip_bomb",
                observed=("quarantine:zip_bomb" if dangerous_matched else "not-detected"),
                matched=dangerous_matched,
            )
        ],
        contracts_complete=contracts_complete,
    )


class _StubBuilder:
    """Stands in for DemoBuilder; returns a preset summary without running the pipeline."""

    _summary: DemoRunSummary

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self) -> DemoResult:
        return DemoResult(summary=self._summary, report=None, manifest=None)  # type: ignore[arg-type]


def _patch_builder(monkeypatch: object, summary: DemoRunSummary) -> None:
    stub = type("_Stub", (_StubBuilder,), {"_summary": summary})
    monkeypatch.setattr(cli_module, "DemoBuilder", stub)  # type: ignore[attr-defined]


def test_cli_strict_exits_nonzero_when_dangerous_fixture_slips_through(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _patch_builder(monkeypatch, _summary(dangerous_matched=False, contracts_complete=True))
    result = runner.invoke(app, ["--strict"])
    assert result.exit_code == 1, result.output
    assert "planted dangerous fixture was not caught" in result.output


def test_cli_strict_exits_nonzero_when_contracts_incomplete(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _patch_builder(monkeypatch, _summary(dangerous_matched=True, contracts_complete=False))
    result = runner.invoke(app, ["--strict"])
    assert result.exit_code == 1, result.output
    assert "contract-completeness did not hold" in result.output


def test_cli_no_strict_tolerates_slip_through(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Without --strict the same failing summary prints but the process exits 0 (display mode).
    _patch_builder(monkeypatch, _summary(dangerous_matched=False, contracts_complete=False))
    result = runner.invoke(app, ["--no-strict"])
    assert result.exit_code == 0, result.output
    assert "MISMATCH" in result.output


# --- The one-line Provider swap must NOT crash with a raw KeyError (S13 debt #46.1). ---
#
# From a clean clone the CPU demo set has no learned-NER Provider installed, so `latence-demo
# --entity-provider entity.gliner` used to reach the runner and crash with a raw KeyError
# traceback — a poor first impression for the README's headline swap command. The CLI now
# runs a preflight: if the chosen Provider is not in the registry it exits non-zero BEFORE the
# pipeline runs, with a clean message naming the package to install.


def test_cli_swap_to_uninstalled_provider_exits_clean_not_keyerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a clean clone where latence-ner-gliner is NOT installed (in the test env its
    # source IS installed, so we force entity.gliner out of the registry). A revert of the
    # preflight would let this reach the runner and raise a bare KeyError instead.
    # cli imports ProviderRegistry from latence_core.capability; patch that same class object.
    monkeypatch.setattr(
        ProviderRegistry,
        "names",
        lambda self: ["entity.gazetteer", "source.local_folder"],
    )
    store = f"file://{tmp_path}/store"
    result = runner.invoke(
        app, ["--storage-uri", store, "--entity-provider", "entity.gliner"]
    )
    # Clean, CLI-handled non-zero exit — not an uncaught KeyError bubbling out of the runner.
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert not isinstance(result.exception, KeyError)
    assert "entity.gliner" in result.output
    assert "not installed" in result.output
    assert "latence-ner-gliner" in result.output  # names the package to install
    # It failed the preflight — never staged the corpus or reached the runner.
    assert not (tmp_path / "store").exists()


def test_cli_swap_to_unknown_provider_gives_generic_hint(tmp_path: Path) -> None:
    # An unrecognised Provider (no known package) still fails the preflight cleanly, with a
    # generic 'install the package that registers it' hint rather than a raw KeyError.
    store = f"file://{tmp_path}/store"
    result = runner.invoke(
        app, ["--storage-uri", store, "--entity-provider", "entity.does_not_exist"]
    )
    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, KeyError)
    assert "entity.does_not_exist" in result.output
    assert "latence.providers" in result.output  # generic install hint
    assert not (tmp_path / "store").exists()

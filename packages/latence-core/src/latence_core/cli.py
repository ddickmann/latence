"""``latence`` CLI (typer).

* ``latence run <pipeline.yaml>`` — execute a Pipeline on the local Runner (S1).
* ``latence stack validate <stack.yaml>`` — run a named stack end-to-end on the bundled
  corpus and assert the G1-style checks (contracts/KG/RAG/report + determinism + resume +
  graceful-failure + device-honesty), write a signed-style summary, and exit non-zero on
  failure so it gates CI/publish (ADR-0036 §3, P3-F4).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

app = typer.Typer(add_completion=False, help="Latence — AI-ready data pipelines.")

# The ``stack`` sub-command group (``latence stack validate ...``). A Typer sub-app keeps the
# stack tooling namespaced without cluttering the top-level command list.
stack_app = typer.Typer(add_completion=False, help="Full-stack Pipeline configs + validation.")
app.add_typer(stack_app, name="stack")

logger = logging.getLogger("latence")


@app.callback()
def _root() -> None:
    """Latence — turn messy enterprise data into AI-ready data."""
    # Presence of a callback forces a multi-command app so `latence run ...`
    # (rather than a collapsed single-command form) is the invocation.


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def run(
    pipeline_path: Annotated[Path, typer.Argument(help="Path to the pipeline YAML.")],
    run_id: Annotated[str, typer.Option(help="Run id (state/checkpoint namespace).")] = "run-0001",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Execute a Pipeline declared in ``pipeline_path`` on the local Runner."""
    _configure_logging(verbose)
    if not pipeline_path.exists():
        typer.echo(f"Pipeline file not found: {pipeline_path}", err=True)
        raise typer.Exit(code=2)

    # The pipeline YAML itself is read from local disk; its declared storage_uri
    # can point anywhere fsspec supports.
    text = pipeline_path.read_text(encoding="utf-8")
    pipeline = Pipeline.from_yaml_text(text)
    logger.info("loaded pipeline %r with %d stages", pipeline.name, len(pipeline.stages))

    report = LocalRunner().run(pipeline, run_id=run_id)

    logger.info(
        "run complete: %d documents, contracts_complete=%s",
        report.document_count,
        report.contracts_complete,
    )
    storage = Storage.for_uri(pipeline.storage_uri)
    run_dir = f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
    report_uri = f"{run_dir}/quality-report.json"
    # Persist the human-readable rendering alongside the machine-readable JSON (S10: the
    # Quality Report is both machine- and human-readable). stdout stays JSON (pipeable);
    # the rendered summary goes to Storage and to the log.
    storage.write_atomic(f"{run_dir}/quality-report.md", report.render().encode("utf-8"))
    logger.info("quality report:\n%s", report.render())
    typer.echo(report.to_json())
    if not storage.exists(report_uri):  # pragma: no cover - defensive
        typer.echo("warning: quality report was not persisted", err=True)


@stack_app.command("validate")
def stack_validate(
    stack_path: Annotated[
        Path, typer.Argument(help="Path to the stack YAML (e.g. stacks/default.yaml).")
    ],
    storage_root: Annotated[
        str | None,
        typer.Option(
            "--storage-root",
            help="Storage URI the harness owns for this validation (state/exports/corpus). "
            "Defaults to a fresh temp dir so a validation never collides with a real run.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Run a named stack end-to-end on the bundled corpus and assert the G1-style checks.

    Exits 0 when every check passes (a skipped GPU stage on a CPU-only host is a PASS, not a
    failure — the device-honesty posture), non-zero otherwise, so this command gates CI/publish.
    """
    _configure_logging(verbose)
    if not stack_path.exists():
        typer.echo(f"Stack file not found: {stack_path}", err=True)
        raise typer.Exit(code=2)

    # Lazy import so a bare ``latence run`` (or ``import latence_core.cli``) does not pull the
    # harness + its runner/stage graph.
    from latence_core.stacks.validate import _describe_failures, validate_stack

    # A pinned ``--storage-root`` is honored (so an operator can inspect the artifacts after the
    # run); otherwise a dedicated temp dir keeps the harness hermetic + non-colliding by default.
    # The temp dir is torn down only after the checks + summary have been read out.
    if storage_root is not None:
        result = validate_stack(stack_path, storage_root=storage_root)
    else:
        with tempfile.TemporaryDirectory(prefix="latence-stack-validate-") as tmp:
            result = validate_stack(stack_path, storage_root=f"file://{tmp}")

    logger.info("stack validation summary:\n%s", result.render())
    typer.echo(result.render())
    if result.summary_uri is not None:
        typer.echo(f"summary written to: {result.summary_uri}", err=True)

    if not result.ok:
        typer.echo("stack validation FAILED:", err=True)
        typer.echo(_describe_failures(result.checks), err=True)
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - console entry shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

"""``latence`` CLI (typer). S1 exposes one command: ``latence run``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

app = typer.Typer(add_completion=False, help="Latence — AI-ready data pipelines.")

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


def main() -> None:  # pragma: no cover - console entry shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

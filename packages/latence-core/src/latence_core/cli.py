"""``latence`` CLI (typer).

* ``latence run <pipeline.yaml>`` — execute a Pipeline on the local Runner (S1).
* ``latence stack validate <stack.yaml>`` — run a named stack end-to-end on the bundled
  corpus and assert the G1-style checks (contracts/KG/RAG/report + determinism + resume +
  graceful-failure + device-honesty), write a signed-style summary, and exit non-zero on
  failure so it gates CI/publish (ADR-0036 §3, P3-F4).
* ``latence bake-off <matrix.yaml>`` — run a per-Stage provider bake-off: hold every OTHER
  Stage at ``stacks/default.yaml`` and run the SAME bundled corpus through each candidate for
  ONE Stage, writing a signed-style ``MATRIX-RESULTS-<stage>.md`` + a JSON companion with
  per-candidate quality/latency/memory/cost/license/determinism/error-rate (ADR-0036 §4, P3-F5).
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


@app.command("bake-off")
def bake_off(
    matrix_path: Annotated[
        Path, typer.Argument(help="Path to the matrix YAML (e.g. matrix/parse.yaml).")
    ],
    base_stack: Annotated[
        Path | None,
        typer.Option(
            "--base-stack",
            help="Stack whose OTHER stages are held constant. Defaults to the repo's "
            "stacks/default.yaml (resolved relative to the matrix file's repo root).",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Directory to write MATRIX-RESULTS-<stage>.md + .json. Defaults to the matrix "
            "file's parent directory.",
        ),
    ] = None,
    storage_root: Annotated[
        str | None,
        typer.Option(
            "--storage-root",
            help="Storage URI the harness owns for the bake-off (state/exports/corpus). "
            "Defaults to a fresh temp dir so a bake-off never collides with a real run.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Run a per-Stage provider bake-off and write ``MATRIX-RESULTS-<stage>.md`` + a JSON companion.

    Holds every OTHER Stage at the base stack and runs the SAME bundled corpus through each
    for the ONE target Stage (ADR-0036 §4). Degrades per-candidate — a candidate that crashes / is
    device-skipped is a row, never an aborted matrix — so this exits 0 when the harness itself
    ran (the enterprise reads the table; a failed/skipped candidate is a documented row).
    """
    _configure_logging(verbose)
    if not matrix_path.exists():
        typer.echo(f"Matrix file not found: {matrix_path}", err=True)
        raise typer.Exit(code=2)

    # Default base stack: the blessed stacks/default.yaml at the matrix file's repo root
    # (matrix/<stage>.yaml -> repo-root/stacks/default.yaml). An explicit --base-stack overrides.
    resolved_base = base_stack or (matrix_path.resolve().parent.parent / "stacks" / "default.yaml")
    if not resolved_base.exists():
        typer.echo(f"Base stack not found: {resolved_base}", err=True)
        raise typer.Exit(code=2)

    destination = (out_dir or matrix_path.resolve().parent).resolve()

    # Lazy import so a bare ``latence run`` does not pull the bake-off harness + its runner graph.
    from latence_core.bakeoff.harness import run_bakeoff

    if storage_root is not None:
        result = run_bakeoff(
            matrix_path, base_stack_path=resolved_base, storage_root=storage_root
        )
    else:
        with tempfile.TemporaryDirectory(prefix="latence-bakeoff-") as tmp:
            result = run_bakeoff(
                matrix_path, base_stack_path=resolved_base, storage_root=f"file://{tmp}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    md_path = destination / f"MATRIX-RESULTS-{result.stage}.md"
    json_path = destination / f"MATRIX-RESULTS-{result.stage}.json"
    md_path.write_text(result.render(), encoding="utf-8")
    json_path.write_text(result.to_json() + "\n", encoding="utf-8")

    typer.echo(result.render())
    typer.echo(f"results written to: {md_path}", err=True)
    typer.echo(f"machine-readable companion: {json_path}", err=True)


def main() -> None:  # pragma: no cover - console entry shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

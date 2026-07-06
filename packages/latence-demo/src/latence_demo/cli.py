"""``latence-demo`` CLI (S13) — the one-command showable demo entrypoint (ADR-0011).

``latence-demo`` (no args) stages the bundled messy corpus, runs the full-spine Pipeline on
the local Runner CPU-only/offline, writes the self-contained web viewer + the KG + the Quality
Report, and prints a typed run summary. This is exactly what ``docker compose up`` invokes.

Options make the two S13 proofs runnable from the shell:

* ``--entity-provider`` — the one-line Provider swap: rerun the identical Pipeline against a
  different NER Provider (e.g. ``entity.gliner`` local model, ``entity.endpoint`` cloud/vLLM)
  to demonstrate model- and cloud-agnosticism concretely (the swapped Provider's package must
  be installed).
* ``--storage-uri`` / ``--run-id`` — point the demo at any ``fsspec`` Storage (``file://`` by
  default; ``s3://`` etc. work unchanged, the storage-agnostic proof).

The demo exits non-zero if a planted dangerous fixture was NOT caught as intended or if
contract-completeness failed — so the one command is also a smoke test, not just a display.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from latence_demo.builder import DEFAULT_STORAGE_URI, DemoBuilder

app = typer.Typer(add_completion=False, help="Latence — the showable AI-ready-data demo.")

logger = logging.getLogger("latence.demo")


@app.command()
def run(
    storage_uri: Annotated[
        str, typer.Option(help="fsspec Storage root for the corpus + outputs.")
    ] = DEFAULT_STORAGE_URI,
    run_id: Annotated[str, typer.Option(help="Run/checkpoint namespace.")] = "demo",
    entity_provider: Annotated[
        str,
        typer.Option(
            help="Entity-Extraction Provider (the one-line swap): entity.gazetteer (default, "
            "in-core CPU), entity.gliner (local model), entity.endpoint (cloud/vLLM)."
        ),
    ] = "entity.gazetteer",
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Exit non-zero if a dangerous fixture slipped through or contracts failed.",
        ),
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Run the showable demo and print the run summary."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    builder = DemoBuilder(
        storage_uri=storage_uri, run_id=run_id, entity_provider=entity_provider
    )
    result = builder.run()
    summary = result.summary
    typer.echo(summary.render())
    typer.echo(f"Open the viewer: {summary.site_uri}")

    if strict:
        if not summary.dangerous_all_caught:
            typer.echo("FAIL: a planted dangerous fixture was not caught as expected.", err=True)
            raise typer.Exit(code=1)
        if not summary.contracts_complete:
            typer.echo("FAIL: contract-completeness did not hold.", err=True)
            raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - console entry shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

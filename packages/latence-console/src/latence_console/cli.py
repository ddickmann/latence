"""``latence-console`` — one command serves the read-only run console.

    latence-console ./corpus

Points at the same Storage location a run wrote to (a folder, an ``s3://`` prefix, anything
fsspec addresses) and serves a localhost page over the run state and Quality Reports already
there. It writes nothing, opens nothing outward, and stands nothing up that has to be torn
down: stop it with Ctrl-C and the only trace is the files that were there before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from latence_core.storage import Storage

from latence_console.server import ConsoleApp, NotLoopbackError, serve

app = typer.Typer(add_completion=False, help="Latence — the read-only run console.")


def _as_uri(location: str) -> str:
    """Turn a bare path into a ``file://`` URI; leave an explicit scheme untouched."""
    if "://" in location:
        return location.rstrip("/")
    return Path(location).expanduser().resolve().as_uri()


@app.command()
def main(
    location: Annotated[
        str,
        typer.Argument(
            help="Storage location a run wrote to (a folder, or an fsspec URI like s3://bucket/prefix)."
        ),
    ] = ".",
    port: Annotated[int, typer.Option(help="Localhost port to serve on.")] = 8787,
    host: Annotated[str, typer.Option(help="Loopback address to bind (127.0.0.1 or ::1).")] = (
        "127.0.0.1"
    ),
) -> None:
    """Serve the read-only run console over a Storage location."""
    uri = _as_uri(location)
    try:
        storage = Storage.for_uri(uri)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    console = ConsoleApp.over_storage(storage, uri)
    try:
        serve(console, host=host, port=port, on_ready=_announce)
    except NotLoopbackError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        typer.echo(f"Could not serve on {host}:{port}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _announce(url: str) -> None:
    typer.echo(
        "Latence run console — read-only, this machine only.\n"
        f"  open {url}\n"
        "  Ctrl-C to stop."
    )


def run() -> None:  # pragma: no cover - console entry shim
    app()

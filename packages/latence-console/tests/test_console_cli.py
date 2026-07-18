"""The one command (#186 AC 1): ``latence-console <location>``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from latence_console import cli
from typer.testing import CliRunner

runner = CliRunner()


def test_a_bare_path_becomes_a_file_uri(tmp_path: Path) -> None:
    assert cli._as_uri(str(tmp_path)) == tmp_path.resolve().as_uri()


def test_an_explicit_scheme_is_left_alone() -> None:
    assert cli._as_uri("s3://bucket/prefix/") == "s3://bucket/prefix"


def test_the_command_serves_the_given_location(
    populated: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: dict[str, Any] = {}

    def fake_serve(app: Any, host: str = "", port: int = 0, **kwargs: Any) -> None:
        served["uri"] = app.storage_uri
        served["host"] = host
        served["port"] = port
        on_ready = kwargs.get("on_ready")
        if on_ready is not None:
            on_ready(f"http://{host}:{port}/")

    monkeypatch.setattr(cli, "serve", fake_serve)
    result = runner.invoke(cli.app, [populated, "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert served == {"uri": populated, "host": "127.0.0.1", "port": 9999}
    assert "read-only" in result.output
    assert "http://127.0.0.1:9999/" in result.output


def test_a_non_loopback_bind_is_refused_with_a_usage_error(populated: str) -> None:
    result = runner.invoke(cli.app, [populated, "--host", "0.0.0.0", "--port", "0"])
    assert result.exit_code == 2
    assert "loopback" in result.output


def test_a_disallowed_storage_scheme_is_refused(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["http://example.com/corpus"])
    assert result.exit_code == 2
    assert "not permitted" in result.output

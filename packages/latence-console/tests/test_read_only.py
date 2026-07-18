"""The console is strictly read-only and introduces no state (#186 AC 4).

The load-bearing assertion is not "a flag says read-only" — it is that the bytes on Storage
are IDENTICAL after a full browse of every page, and that a mutating request changes nothing
and is refused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from latence_console.server import ConsoleApp, NotLoopbackError, build_server
from latence_core.storage import Storage


def _tree_digest(root: Path) -> str:
    """A digest over every file's path, bytes, and mtime under ``root``."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
        h.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
    return h.hexdigest()


def _browse_everything(app: ConsoleApp) -> list[int]:
    statuses = [app.respond("GET", "/").status]
    for path in ("/run/run-0001", "/run/run-0002", "/run/run-0003", "/run/run-0004",
                 "/run/run-0005", "/run/nope", "/nowhere"):
        statuses.append(app.respond("GET", path).status)
    return statuses


def test_browsing_every_page_leaves_storage_byte_identical(
    populated: str, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    before = _tree_digest(root)
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    _browse_everything(app)
    assert _tree_digest(root) == before


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "post", "TRACE"])
def test_mutating_methods_are_refused_and_write_nothing(
    populated: str, tmp_path: Path, method: str
) -> None:
    root = tmp_path / "corpus"
    before = _tree_digest(root)
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    response = app.respond(method, "/run/run-0001")
    assert response.status == 405
    assert _tree_digest(root) == before


def test_no_new_state_survives_between_requests(populated: str, tmp_path: Path) -> None:
    """A change on disk between two requests is reflected — nothing was cached or indexed."""
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    first = app.respond("GET", "/").body
    assert b"run-0006" not in first

    from reports import healthy_report, write_run  # noqa: PLC0415 - test helper

    storage = Storage.for_uri(populated)
    write_run(storage, populated, "run-0006", healthy_report("run-0006"))
    assert b"run-0006" in app.respond("GET", "/").body


def test_a_crafted_run_id_cannot_escape_the_runs_directory(
    populated: str, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me", encoding="utf-8")
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    for hostile in (
        "/run/../../../../etc/passwd",
        "/run/..%2F..%2Fsecret.txt",
        f"/run/{secret}",
        "/run/.",
        "/run/..",
    ):
        response = app.respond("GET", hostile)
        assert response.status == 404, hostile
        assert b"do not serve me" not in response.body


def test_the_server_refuses_to_bind_off_this_machine(populated: str) -> None:
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    with pytest.raises(NotLoopbackError):
        build_server(app, host="0.0.0.0", port=0)  # noqa: S104 - the point is that it is refused
    with pytest.raises(NotLoopbackError):
        build_server(app, host="192.168.1.10", port=0)


def test_the_server_binds_loopback(populated: str) -> None:
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    httpd = build_server(app, host="127.0.0.1", port=0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()

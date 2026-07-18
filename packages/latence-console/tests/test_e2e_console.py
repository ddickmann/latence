"""End to end: one command serves the console over a corpus the STANDARD DEMO produced.

This is #186 AC 1 taken literally — not a hand-built fixture, but the real
:class:`~latence_demo.builder.DemoBuilder` run (the same code ``latence-demo`` and
``docker compose up`` invoke), then a real socket on loopback, then a real HTTP client
reading the served bytes.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from latence_console import render
from latence_console.server import ConsoleApp, build_server
from latence_core.storage import Storage

pytest.importorskip("latence_demo", reason="the standard demo package is not installed here")

from latence_demo.builder import DemoBuilder  # noqa: E402


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
        h.update(str(path.stat().st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


@pytest.fixture(scope="module")
def demo_corpus(tmp_path_factory: pytest.TempPathFactory) -> str:
    root = tmp_path_factory.mktemp("democonsole")
    uri = f"file://{root}/store"
    DemoBuilder(storage_uri=uri, run_id="demo").run()
    return uri


def test_the_console_serves_the_demo_corpus_over_a_real_socket(demo_corpus: str) -> None:
    root = Path(demo_corpus.removeprefix("file://"))
    before = _tree_digest(root)

    app = ConsoleApp.over_storage(Storage.for_uri(demo_corpus), demo_corpus)
    httpd = build_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/", timeout=10) as resp:  # noqa: S310 - loopback
            assert resp.status == 200
            listing = resp.read().decode("utf-8")
        assert 'href="/run/demo"' in listing
        assert "Passed" in listing

        with urllib.request.urlopen(f"{base}/run/demo", timeout=10) as resp:  # noqa: S310
            assert resp.status == 200
            assert resp.headers["Content-Security-Policy"] == (
                "default-src 'none'; style-src 'unsafe-inline'"
            )
            page = resp.read().decode("utf-8")

        # The demo's own headline facts, rendered as sentences (not JSON).
        assert "can be traced back to the exact page it came from" in page
        assert "knowledge graph holds 6 things connected by 33 relationships" in page
        assert "3 files were held back as unsafe" in page
        assert "zip compression ratio" in page
        assert render.external_references(page) == []

        # A mutating request over the wire is refused, not silently accepted.
        request = urllib.request.Request(f"{base}/run/demo", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=10)  # noqa: S310 - loopback
        assert excinfo.value.code == 405
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)

    # Serving the demo corpus changed nothing on disk.
    assert _tree_digest(root) == before

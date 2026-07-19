"""The console refuses requests that did not address this machine (DNS-rebinding guard).

Binding to loopback is a *network*-layer control; DNS rebinding defeats it because the victim's
own browser makes the request over loopback. The only thing that separates the attack from a
legitimate visit is the ``Host`` header — so these tests drive a **real socket** with raw request
bytes and assert on what the server says to a foreign ``Host``.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import pytest
from latence_console.server import (
    ConsoleApp,
    build_server,
    is_local_host_header,
    is_local_origin_header,
)
from latence_core.storage import Storage


@pytest.fixture
def served(populated: str) -> Iterator[int]:
    """A real console on a real loopback socket; yields the bound port."""
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    httpd = build_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _raw_get(port: int, headers: list[str], target: str = "/") -> tuple[int, bytes]:
    """Send a hand-written request (so we control ``Host``) and return (status, body)."""
    lines = [f"GET {target} HTTP/1.1", *headers, "Connection: close", "", ""]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall("\r\n".join(lines).encode("ascii"))
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, body


def test_a_rebound_foreign_host_is_refused_and_leaks_no_corpus(served: int) -> None:
    """The attack itself: a browser rebound to 127.0.0.1 still sends the attacker's Host."""
    legit, legit_body = _raw_get(served, [f"Host: 127.0.0.1:{served}"])
    assert legit == 200
    assert b"run-0001" in legit_body

    status, body = _raw_get(
        served,
        [f"Host: attacker.example.com:{served}", "Origin: https://evil.example.com"],
    )
    assert status == 403
    assert b"run-0001" not in body
    assert body != legit_body


def test_a_foreign_host_without_a_port_is_refused(served: int) -> None:
    status, body = _raw_get(served, ["Host: attacker.example.com"], target="/run/run-0001")
    assert status == 403
    assert b"run-0001" not in body


def test_loopback_host_on_the_wrong_port_is_refused(served: int) -> None:
    status, _ = _raw_get(served, [f"Host: 127.0.0.1:{served + 1}"])
    assert status == 403


def test_a_missing_host_header_is_refused(served: int) -> None:
    status, body = _raw_get(served, [])
    assert status == 403
    assert b"run-0001" not in body


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost", "[::1]"])
def test_every_loopback_name_on_the_bound_port_is_served(served: int, name: str) -> None:
    status, body = _raw_get(served, [f"Host: {name}:{served}"])
    assert status == 200
    assert b"run-0001" in body


def test_a_cross_site_origin_is_refused_even_with_a_valid_host(served: int) -> None:
    status, _ = _raw_get(
        served, [f"Host: 127.0.0.1:{served}", "Origin: https://evil.example.com"]
    )
    assert status == 403


def test_a_loopback_origin_is_served(served: int) -> None:
    status, _ = _raw_get(
        served, [f"Host: 127.0.0.1:{served}", f"Origin: http://localhost:{served}"]
    )
    assert status == 200


def test_the_refusal_still_carries_the_security_headers(served: int) -> None:
    lines = [
        "GET / HTTP/1.1",
        "Host: attacker.example.com",
        "Connection: close",
        "",
        "",
    ]
    with socket.create_connection(("127.0.0.1", served), timeout=10) as sock:
        sock.sendall("\r\n".join(lines).encode("ascii"))
        raw = b""
        while True:
            data = sock.recv(65536)
            if not data:
                break
            raw += data
    head = raw.partition(b"\r\n\r\n")[0].decode("latin-1")
    assert "Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'" in head
    assert "Cache-Control: no-store" in head
    assert "Referrer-Policy: no-referrer" in head
    assert "X-Content-Type-Options: nosniff" in head


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1:8787", True),
        ("localhost:8787", True),
        ("[::1]:8787", True),
        ("::1", False),  # unbracketed IPv6 with no port cannot name port 8787
        ("127.0.0.2:8787", True),  # the whole 127/8 block is this machine
        ("attacker.example.com:8787", False),
        ("localhost.evil.example.com:8787", False),
        ("evil.example.com", False),
        ("127.0.0.1:9999", False),
        ("127.0.0.1", False),  # implies port 80, not the bound port
        ("", False),
        (None, False),
    ],
)
def test_is_local_host_header(value: str | None, expected: bool) -> None:
    assert is_local_host_header(value, 8787) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("http://127.0.0.1:8787", True),
        ("http://localhost:8787", True),
        ("http://[::1]:8787", True),
        ("null", False),
        ("https://evil.example.com", False),
        ("http://localhost:9999", False),
        ("file://", False),
    ],
)
def test_is_local_origin_header(value: str | None, expected: bool) -> None:
    assert is_local_origin_header(value, 8787) is expected

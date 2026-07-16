"""The ported hardening primitives — bearer-0600, rate limit, loopback/UDS, JSON logs (AC3, #174).

Each is tested through its own small interface: the observable accept/reject of :class:`BearerAuth`
(including the ``0600`` file-mode requirement), the engage/recover of :class:`RateLimiter` under an
injected clock, the loopback-only construction of :class:`LoopbackTransport`, and the structured JSON
record :func:`build_rotating_json_logger` writes.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from latence_retrieval.mcp import (
    BearerAuth,
    LoopbackTransport,
    RateLimiter,
    build_rotating_json_logger,
    log_call,
)

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file-mode semantics")


# --- BearerAuth ----------------------------------------------------------------------------------


def test_authenticate_accepts_the_token_and_rejects_others() -> None:
    auth = BearerAuth("correct-horse")
    assert auth.authenticate("correct-horse") is True
    assert auth.authenticate("wrong") is False
    assert auth.authenticate(None) is False
    assert auth.authenticate("") is False


def test_authenticate_rejects_non_ascii_presented_token_without_raising() -> None:
    # R2: compare_digest accepts str only when both sides are pure ASCII; a raw-str compare raises
    # TypeError on a non-ASCII presented token (fail-open into an internal error). Encoding to bytes
    # fails closed — a clean False.
    auth = BearerAuth("correct-horse")
    assert auth.authenticate("tökén") is False


def test_authenticate_matches_a_non_ascii_configured_token() -> None:
    # A legitimately non-ASCII configured token must authenticate its own value, not brick the
    # server (pre-fix, compare_digest(correct, correct) raised for a non-ASCII token).
    auth = BearerAuth("pässwörd-Ω")
    assert auth.authenticate("pässwörd-Ω") is True
    assert auth.authenticate("pässwörd-x") is False


def test_empty_token_construction_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BearerAuth("")


@_POSIX_ONLY
def test_from_file_requires_owner_only_permissions(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("s3cret\n", encoding="utf-8")
    token_file.chmod(0o600)
    auth = BearerAuth.from_file(token_file)
    assert auth.authenticate("s3cret") is True


@_POSIX_ONLY
def test_from_file_rejects_group_or_other_readable_token(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("s3cret", encoding="utf-8")
    token_file.chmod(0o644)  # group/other readable
    with pytest.raises(ValueError, match="owner-only"):
        BearerAuth.from_file(token_file)


def test_from_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        BearerAuth.from_file(tmp_path / "nope")


def test_from_file_rejects_empty_token(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("   \n", encoding="utf-8")
    if os.name == "posix":
        token_file.chmod(0o600)
    with pytest.raises(ValueError, match="empty"):
        BearerAuth.from_file(token_file, require_owner_only=(os.name == "posix"))


# --- RateLimiter ---------------------------------------------------------------------------------


def test_rate_limiter_engages_then_recovers() -> None:
    now = {"t": 0.0}
    limiter = RateLimiter(capacity=3, window=60.0, now=lambda: now["t"])
    assert [limiter.allow() for _ in range(3)] == [True, True, True]
    assert limiter.allow() is False  # budget exhausted within the window
    now["t"] = 60.0  # a full window later — refilled to capacity
    assert limiter.allow() is True


def test_rate_limiter_refills_proportionally() -> None:
    now = {"t": 0.0}
    limiter = RateLimiter(capacity=60, window=60.0, now=lambda: now["t"])  # 1 token/sec
    for _ in range(60):
        assert limiter.allow() is True
    assert limiter.allow() is False
    now["t"] = 1.0  # one second → one token back
    assert limiter.allow() is True
    assert limiter.allow() is False


def test_rate_limiter_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="capacity"):
        RateLimiter(capacity=0)
    with pytest.raises(ValueError, match="window"):
        RateLimiter(capacity=1, window=0)


# --- LoopbackTransport ---------------------------------------------------------------------------


def test_loopback_defaults_are_valid() -> None:
    t = LoopbackTransport(host="127.0.0.1", port=8765)
    assert t.host == "127.0.0.1"
    assert t.port == 8765


def test_public_host_is_rejected() -> None:
    with pytest.raises(ValueError, match="not loopback"):
        LoopbackTransport(host="0.0.0.0", port=8765)
    with pytest.raises(ValueError, match="not loopback"):
        LoopbackTransport(host="10.0.0.5", port=8765)


def test_unix_domain_socket_is_valid() -> None:
    t = LoopbackTransport(host=None, port=None, uds="/tmp/latence.sock")
    assert t.uds == "/tmp/latence.sock"


def test_uds_and_host_together_is_rejected() -> None:
    with pytest.raises(ValueError, match="not both"):
        LoopbackTransport(host="127.0.0.1", port=8765, uds="/tmp/x.sock")


def test_out_of_range_port_is_rejected() -> None:
    with pytest.raises(ValueError, match="port"):
        LoopbackTransport(host="127.0.0.1", port=70000)


# --- structured rotating JSON logs ---------------------------------------------------------------


def test_rotating_json_logger_writes_a_structured_record(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.log"
    logger = build_rotating_json_logger(log_path, name="test.mcp.rotating")
    log_call(logger, {"tool": "search", "ok": True, "result_count": 2})
    for handler in logger.handlers:
        handler.flush()
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "search"
    assert record["ok"] is True
    assert record["result_count"] == 2
    assert "ts" in record


def test_rotating_handler_is_size_bounded(tmp_path: Path) -> None:
    import logging.handlers

    log_path = tmp_path / "calls.log"
    logger = build_rotating_json_logger(
        log_path, name="test.mcp.rotating2", max_bytes=1024, backup_count=3
    )
    handler = logger.handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == 1024
    assert handler.backupCount == 3


def test_file_mode_bits_constant_matches_0600(tmp_path: Path) -> None:
    # A guard on the intent: 0o600 grants no group/other bits (the requirement from_file enforces).
    assert stat.S_IMODE(0o600) & 0o077 == 0

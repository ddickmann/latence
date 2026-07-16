"""The ported ``memory_engine`` hardening core (ADR-0052): auth, rate-limit, transport, JSON logs.

An agentic-search surface is an exposure, so v1 carries the full ``memory_engine`` posture
(ADR-0052), retargeted off the Apple/colsearch specifics onto the backend seam:

* :class:`BearerAuth` — a ``0600`` bearer-token file + **constant-time** comparison
  (:func:`hmac.compare_digest`), so a stolen filesystem read is the only way to the token and a
  timing side-channel cannot recover it byte-by-byte.
* :class:`RateLimiter` — a monotonic token-bucket, so a runaway agent cannot hammer the customer's
  store; deterministic under an injected clock so a test proves the budget engages.
* :class:`LoopbackTransport` — the v1 bind target, **validated loopback-only or a Unix-domain
  socket** (ADR-0052): a public/wildcard host is rejected at construction, never bound.
* :func:`build_rotating_json_logger` / :func:`log_call` — structured, rotating JSON call records
  for an SRE, emitting **no** secret (never the token) and no raw backend text.

Each is a small, concrete mechanism with exactly one correct implementation — not a seam (there is
no *varying* behaviour to adapt, unlike the two-adapter ``RetrievalBackend``), so they are plain
classes hidden behind the :class:`~latence_retrieval.mcp.server.RetrievalServer`
interface rather than Protocols. The remote transport (OAuth 2.1 / mTLS) is deferred; because the
server's boundary is the pure ``call(tool, args, token)`` method, a later remote face reuses this
same core unchanged (ADR-0052).
"""

from __future__ import annotations

import hmac
import json
import logging
import logging.handlers
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# The exact permission bits a bearer-token file may carry: owner read/write only. Any group/other
# bit (0o077) means another local user could read the token — a real 0600 requirement, enforced.
_MODE_PERMISSION_BITS = 0o777
_GROUP_OTHER_BITS = 0o077


class BearerAuth:
    """Bearer-token authentication from a ``0600`` file, compared in constant time.

    Constructed from an already-loaded token (tests, injection) or via :meth:`from_file`, which
    reads the token from an operator-controlled file and **rejects it unless the file mode is
    owner-only (``0600``)** — a token another local user can read is not a secret.
    :meth:`authenticate` compares
    a presented token with :func:`hmac.compare_digest` (constant-time), so no timing side-channel
    leaks how many leading bytes matched; a missing/empty presented token is simply rejected.

    The token itself is never logged, never echoed in an envelope, and never embedded in an error
    message (the no-leak rule) — an unauthorised call yields only the
    :class:`~latence_retrieval.mcp.envelope.ErrorCode.UNAUTHORIZED` code.
    """

    def __init__(self, token: str) -> None:
        if not isinstance(token, str) or not token:
            msg = "BearerAuth requires a non-empty token string."
            raise ValueError(msg)
        self._token = token

    @classmethod
    def from_file(cls, path: str | Path, *, require_owner_only: bool = True) -> BearerAuth:
        """Load the token from ``path``, requiring owner-only (``0600``) permissions.

        ``require_owner_only`` enforces that no group/other permission bit is set (the POSIX
        ``0600`` contract); it is exposed only so a non-POSIX host (where mode bits are not
        meaningful) can
        opt out — on POSIX it must stay ``True``. The file must be a regular file; a missing file,
        a non-file, a mode looser than ``0600``, or an empty token is a loud ``ValueError`` /
        ``FileNotFoundError`` at construction, never a silently insecure server.
        """
        resolved = Path(path)
        if not resolved.is_file():
            msg = f"BearerAuth token path {str(path)!r} is not a regular file."
            raise ValueError(msg)
        if require_owner_only:
            mode = stat.S_IMODE(resolved.stat().st_mode)
            if mode & _GROUP_OTHER_BITS:
                msg = (
                    f"BearerAuth token file must be owner-only (0600); "
                    f"{oct(mode & _MODE_PERMISSION_BITS)} grants group/other access."
                )
                raise ValueError(msg)
        token = resolved.read_text(encoding="utf-8").strip()
        if not token:
            msg = "BearerAuth token file is empty."
            raise ValueError(msg)
        return cls(token)

    def authenticate(self, presented: str | None) -> bool:
        """Return whether ``presented`` matches the token, compared in constant time.

        The comparison is over **UTF-8-encoded bytes**, not raw ``str``: ``hmac.compare_digest``
        accepts ``str`` only when both operands are pure ASCII and raises ``TypeError`` on any
        non-ASCII character. Encoding both sides to bytes keeps the constant-time guarantee (that is
        exactly what ``compare_digest`` is built for) while failing *closed* — a non-ASCII presented
        token is a clean reject, never an uncaught error — for both an attacker-supplied token and a
        legitimately non-ASCII configured one (which would otherwise brick every call).
        """
        if not presented or not isinstance(presented, str):
            return False
        return hmac.compare_digest(presented.encode("utf-8"), self._token.encode("utf-8"))


@dataclass
class RateLimiter:
    """A monotonic token-bucket: at most ``capacity`` calls per ``window`` seconds.

    ``allow()`` consumes one token if the budget permits and returns whether the call may proceed;
    the bucket refills continuously at ``capacity / window`` tokens per second. The clock is
    injected (``now`` defaults to :func:`time.monotonic`) so a test advances time deterministically
    and proves
    the budget both *engages* (the ``capacity+1``-th call in a window is refused) and *recovers*
    (a call after the window passes is allowed) — behaviour, never a flag.

    Refusal is side-effect-free here: the server maps a refused ``allow()`` to a
    :class:`~latence_retrieval.mcp.envelope.ErrorCode.RATE_LIMITED` envelope and **does not run the
    tool**, so a throttled agent never reaches the customer's store.
    """

    capacity: int
    window: float = 60.0
    now: Callable[[], float] = time.monotonic
    _tokens: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self) -> None:
        cap_ok = isinstance(self.capacity, int) and not isinstance(self.capacity, bool)
        if not cap_ok or self.capacity < 1:
            msg = f"RateLimiter 'capacity' must be a positive integer, got {self.capacity!r}."
            raise ValueError(msg)
        win_ok = isinstance(self.window, (int, float)) and not isinstance(self.window, bool)
        if not win_ok or self.window <= 0:
            msg = f"RateLimiter 'window' must be a positive number, got {self.window!r}."
            raise ValueError(msg)
        self._tokens = float(self.capacity)
        self._last = self.now()

    def allow(self) -> bool:
        """Consume one token if available; return whether the call may proceed."""
        current = self.now()
        elapsed = max(0.0, current - self._last)
        self._last = current
        refill = elapsed * (self.capacity / self.window)
        self._tokens = min(float(self.capacity), self._tokens + refill)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


@dataclass(frozen=True)
class LoopbackTransport:
    """The validated v1 bind target: loopback host+port **or** a Unix-domain socket (ADR-0052).

    Exactly one binding is chosen: a ``host``/``port`` pair whose host must be loopback
    (``127.0.0.1`` / ``::1`` / ``localhost``), or a ``uds`` filesystem socket path. A public or
    wildcard host (``0.0.0.0``, a routable IP, a hostname) is rejected at construction — the v1
    local/loopback guarantee is structural, not a runtime hope. Remote transport (OAuth 2.1 / mTLS)
    is a deferred, separately-authenticated adapter; this dataclass is only the local binding.
    """

    host: str | None = "127.0.0.1"
    port: int | None = None
    uds: str | None = None

    def __post_init__(self) -> None:
        if self.uds is not None:
            if self.host is not None and self.port is not None:
                msg = "LoopbackTransport: set a loopback host/port or a 'uds' path, not both."
                raise ValueError(msg)
            if not isinstance(self.uds, str) or not self.uds:
                msg = f"LoopbackTransport 'uds' must be a non-empty socket path, got {self.uds!r}."
                raise ValueError(msg)
            return
        if self.host not in _LOOPBACK_HOSTS:
            allowed = sorted(_LOOPBACK_HOSTS)
            msg = (
                f"LoopbackTransport host {self.host!r} is not loopback; v1 binds only to {allowed} "
                "or a Unix-domain socket (ADR-0052). Remote transport is deferred."
            )
            raise ValueError(msg)
        if self.port is not None:
            port_ok = isinstance(self.port, int) and not isinstance(self.port, bool)
            if not port_ok or not (1 <= self.port <= 65535):
                msg = f"LoopbackTransport 'port' must be 1..65535 or None, got {self.port!r}."
                raise ValueError(msg)


class _JsonLineFormatter(logging.Formatter):
    """Format a log record's ``extra`` payload as one compact JSON line (a structured log)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "call", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, sort_keys=True, default=str)


def build_rotating_json_logger(
    path: str | Path,
    *,
    name: str = "latence_retrieval.mcp",
    max_bytes: int = 1_048_576,
    backup_count: int = 5,
) -> logging.Logger:
    """A logger writing structured JSON call records to ``path``, size-rotated (ADR-0052).

    Wires a :class:`logging.handlers.RotatingFileHandler` with a one-JSON-line-per-record formatter
    so an SRE gets machine-parseable, bounded, rotating logs. The returned logger is what
    :class:`~latence_retrieval.mcp.server.RetrievalServer` accepts (dependency injection) and feeds
    through :func:`log_call`; a test injects a capturing handler instead and asserts the emitted
    record shape.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        Path(path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(_JsonLineFormatter())
    logger.handlers = [handler]
    return logger


def log_call(logger: logging.Logger, record: dict[str, Any]) -> None:
    """Emit one structured JSON call record — never a secret (no token, no raw backend text).

    ``record`` carries only non-secret observability (tool name, ok flag, error code, duration,
    result count). The caller assembles it; this attaches it as the record's ``call`` payload for
    JSON formatter. Kept a free function (not a method) so the logging sink stays an injected
    dependency the server merely *uses*.
    """
    logger.info("mcp.call", extra={"call": record})


__all__ = [
    "BearerAuth",
    "LoopbackTransport",
    "RateLimiter",
    "build_rotating_json_logger",
    "log_call",
]

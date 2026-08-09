"""Typed error taxonomy — a small, stable exception hierarchy for the framework (#73).

Before this module the framework raised bare builtins (``TypeError`` for a mis-wired
Pipeline, ``ValueError`` for a bad config, ``RuntimeError`` for an endpoint/storage
fault) with no way for an operator — or the Quality Report — to tell a Provider fault
apart from a wiring mistake apart from a Storage failure. The Hardening bar (#73,
DEFINITION-OF-DONE "Observability: … typed error taxonomy") asks for exactly that: a
narrow hierarchy the Runner surfaces in the report + logs, so a failed run says *which
kind* of thing broke.

The hierarchy is deliberately shallow — one root and a handful of categories — and lives
in ``latence-core`` with **zero dependencies** (ADR-0016). It carries no heavy tracing
machinery; that is the separate, optional :mod:`latence_core.tracing` seam.

``LatenceError`` is the root every framework-raised error subclasses, so a caller can
``except LatenceError`` to catch anything the framework itself raised (and let a genuinely
unexpected bug — a real ``KeyError`` in provider code — propagate untyped). Each category
also **multiply-inherits the builtin it replaces** so existing ``except ValueError`` /
``except RuntimeError`` call sites (and the ADR-0009 Storage-scheme guard, the S11 Corpus
Store, the S12 endpoint retry) keep working unchanged — the taxonomy is *additive*, it
narrows the type without breaking the old contract:

* :class:`ContractError` (also a ``TypeError``) — an inter-Stage contract / DAG-wiring
  violation: a Stage wired to the wrong carrier, a missing ``depends_on`` edge, a Provider
  that does not satisfy its Capability. The Pipeline is malformed; retrying will not help.
* :class:`ConfigError` (also a ``ValueError``) — an operator-supplied config is invalid: a
  disallowed Storage scheme, a ReDoS-signature entity pattern, a malformed gold set. Fix
  the config, not the code.
* :class:`ProviderError` (also a ``RuntimeError``) — a Provider (typically an endpoint)
  failed at runtime. ``retryable`` marks a transient fault (a dropped connection, a 5xx)
  the Runner's retry may recover from, vs a terminal one.
* :class:`StorageError` (also a ``RuntimeError``) — a Storage / Corpus-Version operation
  failed or is inconsistent (an uncommitted Version inspected, a torn write).
* :class:`ScreeningQuarantineError` — a *typed disposition*, not a crash: raised only where
  a caller opts to treat a Quarantine as fatal (Screening's normal path never raises — it
  Quarantines and continues, CONTEXT ``Quarantine``). It carries the record id + reason so
  the disposition is inspectable. **Counts/ids only, never raw record content** — the same
  no-PII discipline the Quality Report keeps.

The ``category`` class attribute is the stable, lowercase tag the Quality Report + tracing
spans stamp (``"contract"``, ``"config"``, ``"provider"``, ``"storage"``, ``"screening"``,
or ``"unknown"`` for a non-taxonomy error), so an operator greps one field, not a Python
type name that could be refactored. :func:`error_category` maps *any* caught exception to
that tag — a taxonomy error to its own ``category``, a stray builtin to ``"unknown"`` —
so the Runner can label a failed Stage without importing every provider's error type.
"""

from __future__ import annotations

CATEGORY_UNKNOWN = "unknown"


class LatenceError(Exception):
    """Root of the framework's typed error taxonomy (#73).

    Everything the framework itself raises subclasses this, so ``except LatenceError``
    catches a framework-originated failure while letting a genuinely unexpected bug in
    provider code propagate untyped. The ``category`` class attribute is the stable,
    lowercase tag the Quality Report + tracing spans record.
    """

    category: str = CATEGORY_UNKNOWN


class ContractError(LatenceError, TypeError):
    """An inter-Stage contract or DAG-wiring violation (a malformed Pipeline).

    A Stage received the wrong carrier, a required ``depends_on`` edge is missing, or a
    Provider does not satisfy the Capability its Stage declares. Retrying cannot fix it —
    the Pipeline is wired wrong. Also a ``TypeError`` so the pre-taxonomy ``except
    TypeError`` wiring guards (and their tests) keep catching it.
    """

    category = "contract"


class ConfigError(LatenceError, ValueError):
    """An operator-supplied configuration is invalid.

    A disallowed Storage URI scheme (ADR-0009 allowlist), a ReDoS-signature entity
    pattern (#28), a malformed gold set. Fix the config, not the code. Also a
    ``ValueError`` so the pre-taxonomy ``except ValueError`` config guards keep catching it.
    """

    category = "config"


class ProviderError(LatenceError, RuntimeError):
    """A Provider failed at runtime (typically a remote inference endpoint).

    ``retryable`` marks a transient fault (a dropped connection, a rate-limit, a 5xx) the
    Runner's ported retry may recover from, vs a terminal one (a 4xx, a bad response) that
    fails fast. Also a ``RuntimeError`` so the pre-taxonomy endpoint-retry code path — which
    caught ``RuntimeError`` — keeps working.
    """

    category = "provider"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self._retryable = retryable

    @property
    def retryable(self) -> bool:
        return self._retryable


class StorageError(LatenceError, RuntimeError):
    """A Storage or Corpus-Version operation failed or is inconsistent.

    An uncommitted Corpus Version inspected, a torn write, a read of a missing run artifact.
    Also a ``RuntimeError`` so the S11 Corpus Store's pre-taxonomy ``except RuntimeError``
    call sites keep working.
    """

    category = "storage"


class PageSliceMissingError(ContractError):
    """A chunk was asked for a page and carries no :class:`~latence_core.contracts.PageSlice`.

    Since v19 every chunk is self-describing: the Chunk Stage stamps the page spans
    overlapping that chunk onto the chunk itself, so resolution needs nothing but the record
    in hand. Reaching this error means the record was produced by a Chunker that supplied no
    page spans at all — a wiring/producer defect, which is why it is a
    :class:`ContractError`.

    It is an ERROR rather than a ``None`` on purpose. The mechanism this replaced returned
    ``None`` and let every call site degrade to the chunk's inherited ``[page_start,
    page_end]`` range, which is a *plausible-looking wrong page* — indistinguishable from a
    correct one in the output, and counted only in an aggregate nobody reads. ADR-0034: a
    failure is observable or it is not handled. Carries the chunk's record id (an
    identifier, never content — the same no-PII discipline as
    :class:`ScreeningQuarantineError`).
    """

    def __init__(self, record_id: str) -> None:
        super().__init__(
            f"chunk {record_id!r} carries no page_slice, so the source page of an offset "
            f"inside it cannot be resolved. The Chunk Provider that produced it must stamp "
            f"PageSlice.for_span(document.page_map, chunk.char_start, chunk.char_end)."
        )
        self.record_id = record_id


class PageMapSidecarError(ContractError):
    """A document's page-map sidecar could not be honoured — malformed, stale, or missing.

    The out-of-band handoff (ADR-0060) has one failure that matters: the sidecar and the
    markdown drift apart. Its spans then still *validate* — contiguous, ascending, plausibly
    sized — and still resolve every offset to a page, just the wrong one. That is the same
    silent-plausible-wrong failure :class:`PageSliceMissingError` exists to prevent, one Stage
    earlier, so it gets the same posture: a typed error rather than a best-effort map. The
    absent-but-REQUIRED case (a Parser configured for a corpus the operator declared
    paginated, handed a document with no sidecar) raises the same error, because to a consumer
    the two are the same thing — page boundaries this document was supposed to have and does
    not.

    A Parser catches this and emits a ``PARSE_ERROR``
    :class:`~latence_core.contracts.DocumentRecord` — the document is counted in the Quality
    Report with the reason attached, and the run continues (ADR-0020). It carries no document
    content, only the shape mismatch (the same no-PII discipline as
    :class:`ScreeningQuarantineError`).
    """


class ScreeningQuarantineError(LatenceError):
    """A Quarantine disposition surfaced as a typed error (an *opt-in* fatal Screening).

    Screening's normal path **does not raise** — it Quarantines a record and the run
    continues (CONTEXT ``Quarantine``); this exists for a caller that chooses to treat a
    Quarantine as fatal (a strict batch that must abort on the first dangerous file). It
    carries the quarantined record's id and the reason so the disposition stays inspectable,
    **never the raw record content** — the same no-PII discipline the Quality Report keeps.
    """

    category = "screening"

    def __init__(self, message: str, *, record_id: str, reason: str) -> None:
        super().__init__(message)
        self.record_id = record_id
        self.reason = reason


def error_category(exc: BaseException) -> str:
    """The stable, lowercase category tag for *any* exception (``"unknown"`` off-taxonomy).

    A :class:`LatenceError` maps to its own ``category``; anything else (a stray builtin, a
    provider bug) maps to :data:`CATEGORY_UNKNOWN`. This is what lets the Runner label a
    failed Stage in the Quality Report + tracing span without importing every Provider's
    error type — it reads one field off the caught exception.
    """
    if isinstance(exc, LatenceError):
        return exc.category
    return CATEGORY_UNKNOWN

"""Typed errors the Qdrant sink raises — one named type per thing that can actually go wrong.

Three failure surfaces, kept strictly apart so a caller (and a test) can tell *whose* fault it is:

* :class:`SinkConfigError` — the operator mis-wired the sink (bad collection name, absurd batch
  size, a reserved payload key). Raised at construction, before anything is read or opened.
* :class:`SinkExportError` — the Export cannot be loaded as written (no ``records.parquet``, no
  ``embedding`` column, ragged vector widths, a column colliding with the sink's own bookkeeping
  keys). The pipeline's deliverable is the source of truth (ADR-0017), so the sink refuses to
  invent or pad a vector — it says exactly what is missing.
* :class:`SinkLoadError` (and its :class:`SinkAuthError` specialisation) — the *store* faulted
  during the load. Carries **no** raw store text: no URL, no api-key, no payload, no driver stack.
  A caller catches one named type instead of an opaque ``qdrant_client`` exception, and a test
  asserts the behaviour rather than the third-party exception class.

The shipped :class:`~latence_sink_qdrant.testing.FakeQdrantClient` raises the *same* native-shaped
faults a real Qdrant raises (404 unknown collection, 400 bad point id / wrong vector width, 403 bad
api key, 409 collection exists), so these mappings are exercised offline against a fake that
**rejects what the real store rejects** — never a fake that accepts more than the real thing.
"""

from __future__ import annotations


class SinkError(Exception):
    """Base class for every error the Qdrant sink raises."""


class SinkConfigError(SinkError, ValueError):
    """The sink was constructed with an unusable configuration.

    Raised eagerly at construction — before any file is read and before any connection is opened —
    so a typo in a config file fails in the first second of the command, not after a 40-minute
    upload. It is a :class:`ValueError` as well, matching the framework's convention that a caller
    config mistake stays a ``ValueError`` (only *runtime* faults get their own hierarchy).
    """


class SinkExportError(SinkError):
    """The Export could not be read as a loadable corpus.

    The Export is the source of truth (ADR-0017): the sink projects it, it never repairs it. So a
    missing ``records.parquet``, a corpus exported without an Embedder wired (no ``embedding``
    column — there is no vector to load, and silently loading payload-only points would produce a
    collection that answers no vector query), a row whose vector width differs from the collection's
    dimension, or a column named like the sink's own bookkeeping keys all raise here with an
    actionable message, instead of being papered over.
    """


class SinkLoadError(SinkError):
    """The store faulted during the load (a native ``qdrant_client`` call raised).

    The real-store analogue: the collection was dropped mid-load, the connection died, the server
    rejected a batch. The sink maps that native fault to this typed error carrying no raw store
    text, and — critically — a fault anywhere in the upsert phase **aborts the load before the
    reconciling sweep runs**, so a half-loaded corpus can never trigger a delete of the records that
    were not reached (see :meth:`~latence_sink_qdrant.sink.QdrantSink.load`).
    """


class SinkVerificationError(SinkError):
    """The erasure audit could not be *performed* — so it must not be reported as passed.

    Raised by :meth:`~latence_sink_qdrant.sink.QdrantSink.verify_absent` when the collection it was
    pointed at does not exist. An empty result from that method is the evidence a Retraction/Purge
    runbook ends with, and "the records are gone" and "I inspected nothing" are not the same
    finding: a typo'd collection or a stale URL would otherwise certify records that are still live
    and searchable in the real collection.

    Deliberately **not** a :class:`SinkLoadError`: the load verb's create-if-absent behaviour on a
    missing collection is correct (it is projecting the Export), whereas an audit verb may neither
    create nor assume anything. Like the other store-facing errors it carries no URL, api key or
    driver text — only the collection name and the remedy.
    """


class SinkAuthError(SinkLoadError):
    """The store rejected the credentials (HTTP 401/403 from Qdrant).

    Split out from the generic :class:`SinkLoadError` because it is the one store fault with a
    different remedy — fix the api key, not the data or the collection — and because it is the
    fault most likely to be hit on a first live run. The message names the remedy and never echoes
    the key.
    """


__all__ = [
    "SinkAuthError",
    "SinkConfigError",
    "SinkError",
    "SinkExportError",
    "SinkLoadError",
    "SinkVerificationError",
]

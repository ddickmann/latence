# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""File-identity-keyed memoization of derived read structures — the ADR-0056-amendment pattern.

The one home for a pattern three read paths share (:class:`~latence_retrieval.hypergraph.\
HypergraphIndex`, the parquet :class:`~latence_retrieval.multihop.GraphSource` adapters, and
:class:`~latence_retrieval.graph_retrieval.DuckDBGraphDocumentSource`): a **derived, rebuildable
memoization of an emitted artifact**, keyed by the artifact FILE's identity, never a system of
record. Nothing can be written through it, results are byte-identical with and without it (each
consumer keeps a ``cache=False`` path that preserves its original per-call read verbatim — the
parity oracle), and dropping any entry only costs a rebuild. Statelessness stays behavioral
(ADR-0048): identical inputs yield identical outputs; the cache changes cost, not results.

**The key.** ``(st_dev, st_ino, st_size, st_mtime_ns)`` of the artifact file — its *identity and
version*, not its path — so a rewritten artifact is re-decoded, never served stale, and two paths
to the same file share one entry. A consumer whose decode is parameterised (e.g. configurable
column names over the same parquet) adds those parameters as the ``variant`` part of the key, so
two decodings of one file version never collide.

**The lifetime policy** (the ADR-0056 amendment's cache had none — it grew by one full index per
store version per process, without bound, in any serving process that outlives export refreshes):

1. *Identity invalidation* — any rewrite changes the signature, so the old entry can no longer be
   hit.
2. *Superseded-signature eviction* — inserting a decode of a NEW version of a file drops every
   cached decode of any OTHER version of that same file (same ``(st_dev, st_ino)``, different
   signature, every variant). A long-lived process that refreshes an export in place holds at most
   one version of it.
3. *Bounded LRU* — at most ``max_entries`` values per cache; when full, the least-recently-used
   entry is dropped. The bound is a safety valve for many *distinct* live artifacts; rule 2 is
   what keeps refresh churn from counting against it.

Thread-safety: all bookkeeping runs under a lock; the caller's ``build`` runs outside it, so two
threads racing a cold key may both build. That is a documented cost, not a hazard — the values
are derived and equal, and the last insert simply wins.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

T = TypeVar("T")

#: (st_dev, st_ino, st_size, st_mtime_ns) — one emitted artifact file, one version.
FileSignature = tuple[int, int, int, int]

_DEFAULT_MAX_ENTRIES = 8

__all__ = ["FileIdentityCache", "FileSignature"]


class FileIdentityCache(Generic[T]):
    """A bounded, process-shared cache of derived read structures, keyed by file identity.

    One instance per *kind* of derived structure (module-level at the consumer), shared by every
    reader of that kind in the process. :meth:`get_or_build` is the whole read API: it stats
    ``path``, returns the cached value for the file's current signature (and the caller's
    ``variant``), or builds, stores and returns it — applying the module's lifetime policy
    (identity invalidation, superseded-signature eviction, bounded LRU) on the way.

    A stat failure (missing/unreadable file) propagates to the caller untouched, exactly as the
    consumer's own uncached open would fail — the cache adds no error surface of its own.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            msg = (
                f"FileIdentityCache 'max_entries' must be a positive integer, got {max_entries!r}."
            )
            raise ValueError(msg)
        self._max_entries = max_entries
        self._lock = threading.Lock()
        #: insertion order == recency order (a hit reinserts): first key is the LRU victim.
        self._entries: dict[tuple[FileSignature, Hashable], T] = {}

    def get_or_build(self, path: str, build: Callable[[], T], *, variant: Hashable = ()) -> T:
        """The cached value for ``path``'s current content (+ ``variant``), building on a miss."""
        stat = os.stat(path)
        signature: FileSignature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        key = (signature, variant)
        with self._lock:
            if key in self._entries:
                value = self._entries.pop(key)  # reinsert: most-recently-used
                self._entries[key] = value
                return value
        value = build()  # outside the lock: a slow decode must not block unrelated hits
        with self._lock:
            self._evict_superseded(signature)
            self._entries.pop(key, None)  # a racing build for the same key: last insert wins
            self._entries[key] = value
            while len(self._entries) > self._max_entries:
                del self._entries[next(iter(self._entries))]  # LRU
        return value

    def _evict_superseded(self, signature: FileSignature) -> None:
        """Drop every cached decode of another VERSION of ``signature``'s file (all variants)."""
        file_identity = signature[:2]
        stale = [
            key
            for key in self._entries
            if key[0][:2] == file_identity and key[0] != signature
        ]
        for key in stale:
            del self._entries[key]

    def clear(self) -> None:
        """Drop every entry (tests / long-lived processes reclaiming memory)."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

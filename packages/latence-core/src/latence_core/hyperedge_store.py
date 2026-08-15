# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Hyperedge sidecar's point-lookup KV: derived, rebuildable, read per call (ADR-0056).

ADR-0056 measured the online read path: point lookups are flat in corpus size (~0.5 ms for a
whole query's reads at 1M units) while columnar scans degrade to tens of ms — so the sidecar's
query-time face is a KV, and the parquet stays the durable, inspectable, engine-agnostic form
the KV is **always rebuildable from**. The measurement used LMDB; the *reference*
implementation here is deliberately dependency-free (ADR-0007/0016 — nothing binary in core),
deterministic, and byte-reproducible: a single file of sorted keys read via ``mmap`` + binary
search. An LMDB/RocksDB adapter is a Provider-package option behind the same read interface if
an adopter measures the need; the FORMAT is the contract, the library is not.

Layout (all integers little-endian u64)::

    magic  b"LHKV1\\n"
    n                       number of entries
    key_offsets  [n+1]      into the key blob
    value_offsets[n+1]      into the value blob
    key blob                keys, concatenated, SORTED bytewise
    value blob              values, concatenated

Keyspaces, by prefix (one file, one search structure):

* ``e|<hyperedge_id>`` → the solver fields as sorted-key JSON: ``entity_node_ids``,
  ``token_cost``, ``token_counter``, ``gamma``, ``support``, ``has_relations`` — everything
  selection needs, no text (ADR-0056: the index is built for the solver).
* ``p|<node_id>`` → the node's postings as JSON ``[[hyperedge_ids...], [gammas...]]``,
  **γ-sorted descending with γ inline**, so top-k candidate truncation needs no row fetch.
* ``t|<hyperedge_id>`` → ``tau_text`` (UTF-8, already the corpus redaction variant — the
  writer upstream enforced ADR-0057, this file only carries it).
* ``a|<normalized surface>`` → JSON list of node_ids — the alias table ADR-0059's tier-1
  zero-model entry reads. Today it holds each node's canonical name (normalised); the richer
  glinker alias set joins when alias persistence lands (tracked on the map, not faked here).

Statelessness is structural, as in ``DuckDBGraphSource``: a :class:`HyperedgeStoreReader` is
opened per call, reads, and closes — it holds nothing between queries. Determinism: entries
are sorted by key before writing, values serialize with ``sort_keys=True``, so the same
parquet always rebuilds the byte-identical file (the M2 acceptance criterion).
"""

from __future__ import annotations

import json
import mmap
import struct
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import GraphHyperedge, GraphNode

__all__ = [
    "HyperedgeStoreReader",
    "build_store_bytes",
    "normalize_alias",
    "store_entries",
]

_MAGIC = b"LHKV1\n"
_U64 = struct.Struct("<Q")


def normalize_alias(surface: str) -> str:
    """The alias-table key normalisation: NFKC, casefold, whitespace-collapsed.

    Deterministic and locale-independent — the same surface always maps to the same key on
    every host (the byte-reproducibility bar covers the alias keyspace too).
    """
    return " ".join(unicodedata.normalize("NFKC", surface).casefold().split())


# -- deriving the canonical entry set (export and tests share this) -------------


def store_entries(
    hyperedges: Iterable[GraphHyperedge], nodes: Iterable[GraphNode]
) -> list[tuple[bytes, bytes]]:
    """The canonical KV entry set for a corpus's Hyperedges — sorted, ready to write.

    Pure: same records in, same entries out. The postings keyspace is derived here (not
    stored anywhere else), γ-sorted descending with the hyperedge id as the deterministic
    tiebreak; the alias keyspace maps each node's normalised canonical name to every node id
    sharing that surface (sorted).
    """
    entries: dict[bytes, bytes] = {}
    postings: dict[str, list[tuple[float, str]]] = {}

    for h in sorted(hyperedges, key=lambda x: x.hyperedge_id):
        solver = {
            "entity_node_ids": h.entity_node_ids,
            "gamma": h.gamma,
            "has_relations": h.has_relations,
            "support": h.support,
            "token_cost": h.token_cost,
            "token_counter": h.token_counter,
        }
        entries[b"e|" + h.hyperedge_id.encode()] = json.dumps(
            solver, sort_keys=True, separators=(",", ":")
        ).encode()
        entries[b"t|" + h.hyperedge_id.encode()] = h.tau_text.encode()
        for node_id in h.entity_node_ids:
            postings.setdefault(node_id, []).append((h.gamma, h.hyperedge_id))

    for node_id, items in postings.items():
        items.sort(key=lambda t: (-t[0], t[1]))  # γ desc, id asc — no row fetch for top-k
        payload = [[hid for _, hid in items], [g for g, _ in items]]
        entries[b"p|" + node_id.encode()] = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()

    aliases: dict[str, set[str]] = {}
    for node in sorted(nodes, key=lambda n: n.node_id):
        key = normalize_alias(node.canonical_name)
        if key:
            aliases.setdefault(key, set()).add(node.node_id)
    for surface, node_ids in aliases.items():
        entries[b"a|" + surface.encode()] = json.dumps(
            sorted(node_ids), separators=(",", ":")
        ).encode()

    return sorted(entries.items())


def build_store_bytes(entries: list[tuple[bytes, bytes]]) -> bytes:
    """Serialize sorted ``(key, value)`` entries into the single-file format (byte-stable)."""
    keys = [k for k, _ in entries]
    if keys != sorted(set(keys)):
        msg = "hyperedge store entries must be sorted and unique by key."
        raise ValueError(msg)
    values = [v for _, v in entries]

    key_offsets = [0]
    for k in keys:
        key_offsets.append(key_offsets[-1] + len(k))
    value_offsets = [0]
    for v in values:
        value_offsets.append(value_offsets[-1] + len(v))

    out = bytearray()
    out += _MAGIC
    out += _U64.pack(len(entries))
    for off in key_offsets:
        out += _U64.pack(off)
    for off in value_offsets:
        out += _U64.pack(off)
    out += b"".join(keys)
    out += b"".join(values)
    return bytes(out)


class HyperedgeStoreReader:
    """Per-call reader over the store file: open, look up, close — holds nothing between calls.

    ``mmap``-backed binary search over the sorted key blob: O(log n) per lookup, no
    deserialization of anything not asked for. Use as a context manager scoped to ONE query
    (the ADR-0056 statelessness discipline — structural, not a promise)::

        with HyperedgeStoreReader.open(path) as store:
            ids, gammas = store.postings(node_id)
            fields = store.solver_fields(hyperedge_id)
    """

    def __init__(self, buf: mmap.mmap | memoryview, *, _close: Any = None) -> None:
        self._buf = buf
        self._close = _close
        if bytes(buf[: len(_MAGIC)]) != _MAGIC:
            msg = "not a hyperedge store file (bad magic)."
            raise ValueError(msg)
        pos = len(_MAGIC)
        (self._n,) = _U64.unpack_from(buf, pos)
        pos += 8
        self._key_off_base = pos
        self._val_off_base = pos + 8 * (self._n + 1)
        self._keys_base = self._val_off_base + 8 * (self._n + 1)
        (keys_len,) = _U64.unpack_from(buf, self._key_off_base + 8 * self._n)
        self._values_base = self._keys_base + keys_len

    @classmethod
    def open(cls, path: str) -> HyperedgeStoreReader:
        """Open a store file for one call's reads (mmap, closed with the reader)."""
        fh = open(path, "rb")  # noqa: SIM115 - lifetime is the reader's, closed in close()
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        return cls(buf, _close=(buf, fh))

    @classmethod
    def from_bytes(cls, raw: bytes) -> HyperedgeStoreReader:
        """A reader over in-memory bytes (tests / storage-seam reads)."""
        return cls(memoryview(raw))

    def __enter__(self) -> HyperedgeStoreReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the mapping; the reader holds nothing afterwards."""
        if self._close is not None:
            buf, fh = self._close
            buf.close()
            fh.close()
            self._close = None

    # -- raw lookup ---------------------------------------------------------

    def _key_at(self, i: int) -> bytes:
        (start,) = _U64.unpack_from(self._buf, self._key_off_base + 8 * i)
        (end,) = _U64.unpack_from(self._buf, self._key_off_base + 8 * (i + 1))
        return bytes(self._buf[self._keys_base + start : self._keys_base + end])

    def _value_at(self, i: int) -> bytes:
        (start,) = _U64.unpack_from(self._buf, self._val_off_base + 8 * i)
        (end,) = _U64.unpack_from(self._buf, self._val_off_base + 8 * (i + 1))
        return bytes(self._buf[self._values_base + start : self._values_base + end])

    def get(self, key: bytes) -> bytes | None:
        """Point lookup: the value for ``key``, or None."""
        lo, hi = 0, self._n
        while lo < hi:
            mid = (lo + hi) // 2
            k = self._key_at(mid)
            if k < key:
                lo = mid + 1
            elif k > key:
                hi = mid
            else:
                return self._value_at(mid)
        return None

    def keys(self) -> Iterator[bytes]:
        """All keys in sorted order (inspection/tests; the online path never scans)."""
        for i in range(self._n):
            yield self._key_at(i)

    # -- typed keyspace accessors ------------------------------------------

    def solver_fields(self, hyperedge_id: str) -> dict[str, Any] | None:
        """The ``e|`` record: everything selection needs for one Hyperedge, no text."""
        raw = self.get(b"e|" + hyperedge_id.encode())
        return None if raw is None else dict(json.loads(raw))

    def postings(self, node_id: str) -> tuple[list[str], list[float]]:
        """The ``p|`` record: ``(hyperedge_ids, gammas)``, γ-sorted desc, γ inline."""
        raw = self.get(b"p|" + node_id.encode())
        if raw is None:
            return ([], [])
        ids, gammas = json.loads(raw)
        return (list(ids), [float(g) for g in gammas])

    def tau_text(self, hyperedge_id: str) -> str | None:
        """The ``t|`` record: the span text (corpus redaction variant, ADR-0057)."""
        raw = self.get(b"t|" + hyperedge_id.encode())
        return None if raw is None else raw.decode()

    def alias(self, surface: str) -> list[str]:
        """The ``a|`` record: node_ids whose canonical surface matches (tier-1 entry)."""
        raw = self.get(b"a|" + normalize_alias(surface).encode())
        return [] if raw is None else list(json.loads(raw))

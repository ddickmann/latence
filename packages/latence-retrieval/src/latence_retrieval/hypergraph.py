# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Hypergraph entry + confidence propagation (M3: ADR-0056/0059, ticket-20 verified order).

The query-time half of chain-of-evidence retrieval, over the sidecar KV the pipeline
emitted (``graph-hyperedges.kv``, ADR-0056). Everything here is stateless in the ADR-0048
sense — identical inputs always yield identical outputs and nothing here is a system of
record — and **zero-dep**: the paper-faithful pipeline order makes pure-Python propagation
viable, because the expensive step (propagation) runs only AFTER the candidate universe is
pruned to a small query-aware hypergraph.

The order is load-bearing and measured (wayfinder ticket 20, verified head-to-head against
HyCE-RAG §3.3, arXiv 2607.22597): propagating over the raw expansion drowns bridging
evidence in noise (selection recall 0.15); pruning FIRST by ``s_hyp`` and gating vertices
after propagation doubles recall at 2.7× lower latency. Hence:

    expand (hop-priority, hub-capped)                       [ticket 20: both guards measured]
      → s_hyp = β·s_entry + (1−β)·s_rel, prune to top-K_h   [Eq 16–19]
      → two-stage RWR on the small B_q                      [Eq 26–28]
      → ρ(e) mean-pooled; vertex gate c* ≥ θ_v + coverage   [Eq 29–33]

Entry follows ADR-0059's tiers, zero-model by default: **tier 0** — the caller's own
ANN over ``entity_vecs`` supplies ``ann_node_ids`` (no new inference; the restart mass
spreads over them at weight ``alpha``); **tier 1** — ``alias_surfaces`` resolve against the
KV's NFKC/casefold alias table to the anchored ``V_llm`` set (weight 1.0 — spec 2.1 anchors
restart on entities the query actually names). **Tier 2** (a query NER model producing
better surfaces) plugs into the same two arguments from an opt-in extra; nothing here
imports a model.

``eu_sims`` (hyperedge_id → query similarity from the caller's ``eu_vecs`` ANN) feeds the
``s_rel`` half of the prune; absent, the prune degrades to entry-overlap only (β=1) —
documented, not silent.

**Latency at scale (wayfinder-enterprise-serving ticket 01, measured).** The original
per-call read path re-ran a pure-Python binary search + JSON decode for every postings row
and every solver row a query's expansion touched — at 677k hyperedges that is ~4.7M key
comparisons and ~230k JSON decodes *per query* (seconds). ADR-0056's "resident buys
0.56 ms" ruling was measured over an LMDB C-speed KV at |E_c| ≤ 2000; the shipped zero-dep
reader at real expansion widths is a different regime, so :class:`HypergraphIndex` now
precomputes, once per store FILE per process, exactly the two structures the expansion
scans: the entity→hyperedge postings (int-indexed CSR) and the hyperedge→entity incidence
(+ γ and its global rank). :meth:`HyperedgeKV.open` attaches the shared index
transparently (keyed by file identity — device/inode/size/mtime — so a rewritten store is
re-indexed, never served stale); ``open(path, cache=False)`` keeps the original
scan-per-call path. Both paths produce **byte-identical** signals — pinned by tests and by
the ticket-01 parity run over full real query sets. When ``numpy`` is importable the index
uses vectorised kernels for the same arithmetic; the pure-Python kernels remain the
zero-dep fallback and produce identical results.

The KV reader here is a second, independent implementation of the ADR-0056 store FORMAT
(the format is the contract, not the library): ``latence-retrieval`` depends on pydantic
alone and cannot import ``latence-core``. Cross-implementation interop is pinned by a test
in latence-core (writer there, reader here, byte round-trip).
"""

from __future__ import annotations

import heapq
import json
import mmap
import struct
import unicodedata
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_retrieval.read_cache import FileIdentityCache

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

def _import_numpy() -> Any:
    """The OPTIONAL accelerator: identical results, vectorised kernels (module docstring)."""
    try:
        import numpy
    except ImportError:  # pragma: no cover - exercised in the zero-dep environment
        return None
    return numpy


_np: Any = _import_numpy()

__all__ = [
    "HyperedgeKV",
    "HypergraphConfig",
    "HypergraphIndex",
    "HypergraphSignals",
    "Propagator",
    "RWRPropagator",
    "clear_index_cache",
    "hypergraph_signals",
]

_MAGIC = b"LHKV1\n"
_U64 = struct.Struct("<Q")


def _normalize_alias(surface: str) -> str:
    """Mirror of the writer's alias normalisation (NFKC, casefold, whitespace-collapsed)."""
    return " ".join(unicodedata.normalize("NFKC", surface).casefold().split())


class HyperedgeKV:
    """Per-call reader for the ADR-0056 sidecar store (format-compatible, zero-dep).

    Open → read → close, scoped to ONE query (statelessness is structural)::

        with HyperedgeKV.open(path) as kv:
            ids, gammas = kv.postings(node_id)

    :meth:`open` additionally attaches the process-shared :class:`HypergraphIndex` for the
    file (built on first open, reused while the file is unchanged) so
    :func:`hypergraph_signals` runs its indexed fast path — same results, orders of
    magnitude cheaper at scale. ``open(path, cache=False)`` and :meth:`from_bytes` readers
    carry no index and serve every read from the store bytes.
    """

    def __init__(self, buf: mmap.mmap | memoryview, *, _close: Any = None) -> None:
        self._buf = buf
        self._close = _close
        self._index: HypergraphIndex | None = None
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
    def open(cls, path: str, *, cache: bool = True) -> HyperedgeKV:
        """Open the store file for one call's reads (mmap; closed with the reader).

        ``cache=True`` (the default) attaches the process-shared :class:`HypergraphIndex`
        for this file — built on the first open, invalidated automatically when the file
        changes (the cache key is device/inode/size/mtime). The index is a derived,
        rebuildable read optimisation: results are byte-identical with and without it.
        The cache's lifetime policy (bounded LRU + eviction of a rewritten store's
        superseded index) is :mod:`latence_retrieval.read_cache`'s.
        """
        fh = open(path, "rb")  # noqa: SIM115 - lifetime is the reader's, closed in close()
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        reader = cls(buf, _close=(buf, fh))
        if cache:
            reader._index = _shared_index(path, reader)
        return reader

    @classmethod
    def from_bytes(cls, raw: bytes) -> HyperedgeKV:
        """A reader over in-memory bytes (tests / storage-seam reads)."""
        return cls(memoryview(raw))

    def __enter__(self) -> HyperedgeKV:
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

    @property
    def index(self) -> HypergraphIndex | None:
        """The attached process-shared index, or ``None`` for uncached readers."""
        return self._index

    def _key_at(self, i: int) -> bytes:
        (start,) = _U64.unpack_from(self._buf, self._key_off_base + 8 * i)
        (end,) = _U64.unpack_from(self._buf, self._key_off_base + 8 * (i + 1))
        return bytes(self._buf[self._keys_base + start : self._keys_base + end])

    def _value_at(self, i: int) -> bytes:
        (start,) = _U64.unpack_from(self._buf, self._val_off_base + 8 * i)
        (end,) = _U64.unpack_from(self._buf, self._val_off_base + 8 * (i + 1))
        return bytes(self._buf[self._values_base + start : self._values_base + end])

    def _lower_bound(self, key: bytes) -> int:
        """The first row whose key is ``>= key`` (``self._n`` when none is)."""
        lo, hi = 0, self._n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._key_at(mid) < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _prefix_items(self, prefix: bytes) -> Iterator[tuple[bytes, bytes]]:
        """``(key-suffix, value)`` for every row under ``prefix``, in stored (sorted) order.

        The index builder's scan: keys are stored lexicographically, so one prefix is one
        contiguous row range and the scan is a single forward pass — never per-key search.
        """
        stop_key = prefix[:-1] + bytes([prefix[-1] + 1])
        for row in range(self._lower_bound(prefix), self._lower_bound(stop_key)):
            yield self._key_at(row)[len(prefix) :], self._value_at(row)

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

    def solver_fields(self, hyperedge_id: str) -> dict[str, Any] | None:
        """The solver-facing record: entity set, γ, token_cost (+ counter id), support."""
        raw = self.get(b"e|" + hyperedge_id.encode())
        return None if raw is None else dict(json.loads(raw))

    def postings(self, node_id: str) -> tuple[list[str], list[float]]:
        """``(hyperedge_ids, gammas)`` for a node — γ-sorted desc, γ inline (no row fetch)."""
        raw = self.get(b"p|" + node_id.encode())
        if raw is None:
            return ([], [])
        ids, gammas = json.loads(raw)
        return (list(ids), [float(g) for g in gammas])

    def tau_text(self, hyperedge_id: str) -> str | None:
        """The span text (corpus redaction variant — enforced at the writer, ADR-0057)."""
        raw = self.get(b"t|" + hyperedge_id.encode())
        return None if raw is None else raw.decode()

    def alias(self, surface: str) -> list[str]:
        """Tier-1 zero-model entry: node_ids whose canonical surface matches (ADR-0059)."""
        raw = self.get(b"a|" + _normalize_alias(surface).encode())
        return [] if raw is None else list(json.loads(raw))


# -- the precomputed index (ticket 01: entity→hyperedge + hyperedge→entity, built once) ---------


class HypergraphIndex:
    """The decoded read-index over one store: postings + incidence as int-indexed CSR arrays.

    Built once per store FILE per process (:func:`_shared_index`) and shared by every
    reader over that file, this holds exactly what :func:`hypergraph_signals`'s expansion
    and prune scan per query — the entity→hyperedge postings (stored γ-desc order
    preserved), the hyperedge→entity incidence (stored JSON order preserved), each
    hyperedge's γ, and γ's global rank under the ``(-γ, hyperedge_id)`` truncation order.
    Node and hyperedge ids are mapped to dense ints; because the store's keys are sorted
    byte-wise and UTF-8 byte order equals code-point order, **int order equals id string
    order**, which is what makes every tie-break below reproduce the string-keyed original
    exactly.

    A posted hyperedge id with no solver row is dropped at build — byte-identical to the
    per-call path, which skips exactly those ids during expansion. The index carries no
    buffer and no file handle: solver fields for the (small) retained set are read through
    the caller's open reader, so the store file's lifetime stays the caller's.

    ``accelerated=None`` (default) uses ``numpy`` kernels when importable and the
    pure-Python kernels otherwise; both are pinned byte-identical by tests. The index is
    immutable after build.
    """

    def __init__(self, store: HyperedgeKV, *, accelerated: bool | None = None) -> None:
        hids: list[str] = []
        gammas = array("d")
        ent_off = array("q", [0])
        ent_lists: list[list[str]] = []
        self._solver_base = store._lower_bound(b"e|")  # the e| rows are one contiguous range
        for suffix, raw in store._prefix_items(b"e|"):
            fields = json.loads(raw)
            hids.append(suffix.decode())
            gammas.append(float(fields["gamma"]))
            ent_lists.append([str(v) for v in fields["entity_node_ids"]])
        hid_pos = {hid: row for row, hid in enumerate(hids)}

        posted: list[tuple[str, list[str]]] = []
        node_set: set[str] = set()
        for suffix, raw in store._prefix_items(b"p|"):
            node = suffix.decode()
            ids, _gammas = json.loads(raw)
            posted.append((node, [str(h) for h in ids]))
            node_set.add(node)
        for nodes in ent_lists:
            node_set.update(nodes)
        nids = sorted(node_set)
        nid_pos = {nid: row for row, nid in enumerate(nids)}

        ent_flat = array("i")
        for nodes in ent_lists:
            ent_flat.extend(nid_pos[v] for v in nodes)
            ent_off.append(len(ent_flat))

        rows_of: dict[int, list[int]] = {
            nid_pos[node]: [hid_pos[h] for h in ids if h in hid_pos] for node, ids in posted
        }
        post_flat = array("i")
        post_off = array("q", [0])
        for row in range(len(nids)):
            post_flat.extend(rows_of.get(row, ()))
            post_off.append(len(post_flat))

        order = sorted(range(len(hids)), key=lambda row: (-gammas[row], row))
        gamma_rank = array("q", bytes(8 * len(hids)))
        for rank, row in enumerate(order):
            gamma_rank[row] = rank

        self._hids = hids
        self._nids = nids
        self._nid_pos = nid_pos
        self._gammas = gammas
        self._gamma_rank = gamma_rank
        self._ent_flat = ent_flat
        self._ent_off = ent_off
        self._post_flat = post_flat
        self._post_off = post_off

        if accelerated is None:
            accelerated = _np is not None
        if accelerated and _np is None:
            msg = "HypergraphIndex(accelerated=True) requires numpy, which is not importable."
            raise ValueError(msg)
        self.accelerated = bool(accelerated)
        if self.accelerated:
            self._f_post = _np.frombuffer(post_flat, dtype=_np.int32)
            self._f_post_off = _np.frombuffer(post_off, dtype=_np.int64)
            self._f_ent = _np.frombuffer(ent_flat, dtype=_np.int32)
            self._f_ent_off = _np.frombuffer(ent_off, dtype=_np.int64)
            self._f_rank = _np.frombuffer(gamma_rank, dtype=_np.int64)

    # -- id mapping ---------------------------------------------------------------

    def hyperedge_id(self, row: int) -> str:
        """The hyperedge id at dense ``row``."""
        return self._hids[row]

    def node_rows(self, node_ids: Iterator[str] | list[str]) -> list[int]:
        """Dense rows of the known ids among ``node_ids``, ascending (== ascending ids)."""
        positions = self._nid_pos
        return sorted(positions[v] for v in node_ids if v in positions)

    def entity_node_ids(self, row: int) -> list[str]:
        """Hyperedge ``row``'s entity node ids, in stored (JSON) order."""
        nids = self._nids
        return [nids[v] for v in self._ent_flat[self._ent_off[row] : self._ent_off[row + 1]]]

    def solver_fields_at(self, store: HyperedgeKV, row: int) -> dict[str, Any]:
        """Hyperedge ``row``'s solver record, read straight off ``store``'s value table.

        The ``e|`` rows are one contiguous, sorted KV range, so the index only remembers
        where that range starts — no per-id binary search at query time. ``store`` must
        read the same store content the index was built from, which the file-identity
        cache key guarantees for cache-attached readers.
        """
        return dict(json.loads(store._value_at(self._solver_base + row)))

    # -- the expansion + truncation kernel ---------------------------------------

    def expand(self, anchor_rows: list[int], config: HypergraphConfig) -> list[int]:
        """Hop-priority, hub-capped expansion + per-layer γ truncation, in int space.

        Returns the candidate hyperedge rows in exactly the order the per-call scan
        produces its candidate ids: layers in hop order, first-encounter order within an
        untruncated layer, ``(-γ, id)`` order within a truncated one.
        """
        by_hop = (
            self._layers_np(anchor_rows, config)
            if self.accelerated
            else self._layers_py(anchor_rows, config)
        )
        candidates: list[int] = []
        for layer in by_hop:
            room = config.max_candidates - len(candidates)
            if room <= 0:
                break
            if len(layer) > room:
                layer = self._top_by_gamma(layer, room)
            candidates.extend(int(row) for row in layer)
        return candidates

    def _layers_py(self, anchor_rows: list[int], config: HypergraphConfig) -> list[list[int]]:
        post_flat, post_off = self._post_flat, self._post_off
        ent_flat, ent_off = self._ent_flat, self._ent_off
        seen = bytearray(len(self._hids))
        by_hop: list[list[int]] = []
        frontier = anchor_rows
        for hop in range(config.hops):
            layer: list[int] = []
            append = layer.append
            for v in frontier:
                start = post_off[v]
                stop = min(post_off[v + 1], start + config.hub_cap)
                for h in post_flat[start:stop]:
                    if not seen[h]:
                        seen[h] = 1
                        append(h)
            if not layer:
                break
            by_hop.append(layer)
            if hop + 1 < config.hops:
                nxt: set[int] = set()
                for h in layer[: config.expand_cap]:
                    nxt.update(ent_flat[ent_off[h] : ent_off[h + 1]])
                frontier = sorted(nxt)
        return by_hop

    def _layers_np(self, anchor_rows: list[int], config: HypergraphConfig) -> list[Any]:
        post_flat, post_off = self._f_post, self._f_post_off
        ent_flat, ent_off = self._f_ent, self._f_ent_off
        seen = _np.zeros(len(self._hids), dtype=bool)
        by_hop: list[Any] = []
        frontier: Any = _np.asarray(anchor_rows, dtype=_np.int64)
        for hop in range(config.hops):
            parts = []
            for v in frontier:
                start = post_off[v]
                stop = min(post_off[v + 1], start + config.hub_cap)
                if stop > start:
                    parts.append(post_flat[start:stop])
            layer = _np.empty(0, dtype=_np.int32)
            if parts:
                stream = _np.concatenate(parts)
                fresh = stream[~seen[stream]]
                if fresh.size:
                    # First-occurrence dedupe WITHOUT a sort: scatter each position into a
                    # per-value slot with the reversed stream (so the earliest write wins),
                    # then keep exactly the positions that read back their own index. O(n),
                    # order-preserving — the argsort inside np.unique was 60%+ of the lane.
                    slots = _np.empty(len(self._hids), dtype=_np.int64)
                    positions = _np.arange(fresh.size, dtype=_np.int64)
                    slots[fresh[::-1]] = positions[::-1]
                    layer = fresh[slots[fresh] == positions]
                    seen[layer] = True
            if layer.size == 0:
                break
            by_hop.append(layer)
            if hop + 1 < config.hops:
                sel = layer[: config.expand_cap]
                ent_parts = [ent_flat[ent_off[h] : ent_off[h + 1]] for h in sel]
                frontier = (
                    _np.unique(_np.concatenate(ent_parts))
                    if ent_parts
                    else _np.empty(0, dtype=_np.int32)
                )
        return by_hop

    def _top_by_gamma(self, layer: Any, room: int) -> Any:
        """The ``room`` best rows of ``layer`` under ``(-γ, id)`` — γ ranks are unique ints."""
        if self.accelerated and not isinstance(layer, list):
            ranks = self._f_rank[layer]
            part = _np.argpartition(ranks, room - 1)[:room]
            return layer[part[_np.argsort(ranks[part], kind="stable")]]
        rank = self._gamma_rank
        return heapq.nsmallest(room, layer, key=rank.__getitem__)

    # -- the s_hyp prune kernel ---------------------------------------------------

    def prune(
        self,
        candidates: list[int],
        anchor_rows: list[int],
        beta: float,
        sims: dict[str, float],
        config: HypergraphConfig,
    ) -> list[int]:
        """Top-``k_h`` of ``candidates`` by ``(-s_hyp, id)`` — Eq 16–19 in int space.

        ``s_hyp`` reproduces the per-call arithmetic operation-for-operation (same IEEE
        ops in the same order), so the retained set and its order are byte-identical.
        """
        if len(candidates) <= config.k_h:
            return candidates
        if self.accelerated:
            cand = _np.asarray(candidates, dtype=_np.int64)
            mask = _np.zeros(len(self._nids), dtype=bool)
            if anchor_rows:
                mask[_np.asarray(anchor_rows, dtype=_np.int64)] = True
            prefix = _np.concatenate(
                (_np.zeros(1, dtype=_np.int64), _np.cumsum(mask[self._f_ent]))
            )
            starts = self._f_ent_off[cand]
            stops = self._f_ent_off[cand + 1]
            overlap = (prefix[stops] - prefix[starts]) / ((stops - starts) + 1e-9)
            if sims:
                hids = self._hids
                sim_values = _np.fromiter(
                    (float(sims.get(hids[row], 0.0)) for row in candidates),
                    dtype=_np.float64,
                    count=len(candidates),
                )
            else:
                sim_values = _np.zeros(len(candidates), dtype=_np.float64)
            scores = beta * overlap + (1.0 - beta) * sim_values
            order = _np.lexsort((cand, -scores))
            return [int(row) for row in cand[order[: config.k_h]]]
        anchor_set = set(anchor_rows)
        ent_flat, ent_off = self._ent_flat, self._ent_off
        hids = self._hids
        scores_py: list[float] = []
        for row in candidates:
            nodes = ent_flat[ent_off[row] : ent_off[row + 1]]
            overlap_row = sum(1 for v in nodes if v in anchor_set) / (len(nodes) + 1e-9)
            scores_py.append(
                beta * overlap_row + (1.0 - beta) * float(sims.get(hids[row], 0.0))
            )
        order_py = sorted(
            range(len(candidates)), key=lambda i: (-scores_py[i], candidates[i])
        )
        return [candidates[i] for i in order_py[: config.k_h]]


#: Process-shared indices, one per store FILE version (device/inode/size/mtime), with the
#: :mod:`~latence_retrieval.read_cache` lifetime policy: an index for a rewritten store evicts
#: the superseded version's index (a serving process that outlives export refreshes holds at
#: most one index per store), and the whole cache is LRU-bounded.
_INDEX_CACHE: FileIdentityCache[HypergraphIndex] = FileIdentityCache()


def _shared_index(path: str, reader: HyperedgeKV) -> HypergraphIndex:
    """The cached index for ``path``'s current content, built through ``reader`` on a miss."""
    return _INDEX_CACHE.get_or_build(path, lambda: HypergraphIndex(reader))


def clear_index_cache() -> None:
    """Drop every cached :class:`HypergraphIndex` (tests / long-lived processes)."""
    _INDEX_CACHE.clear()


# -- configuration and results --------------------------------------------------


@dataclass(frozen=True)
class HypergraphConfig:
    """Knobs for entry + propagation, defaulting to the ticket-20-verified operating point.

    Every default traces to a measurement or the verified paper order — none is a guess:
    ``hops``/``hub_cap``/``expand_cap`` from the ticket-20 expansion guards, ``k_h`` from
    the paper's small-query-hypergraph regime, ``restart``/``steps`` from Eq 27 (λ=0.2,
    finite-step approximation), ``theta_v`` the relative vertex gate (Eq 30), ``beta`` the
    s_hyp trade-off (Eq 16), ``alpha`` ADR-0059's tier-0 anchor weight.
    """

    hops: int = 2
    hub_cap: int = 2000
    expand_cap: int = 3000
    max_candidates: int = 20000
    k_h: int = 200
    beta: float = 0.5
    alpha: float = 0.2
    restart: float = 0.2
    steps: int = 6
    theta_v: float = 0.1


@dataclass(frozen=True)
class HypergraphSignals:
    """The propagation result the CoE selector (M4) consumes.

    ``rho`` covers exactly the RETAINED hyperedges (post vertex-gate + coverage); every id
    in ``rho`` has its solver fields in ``fields``. ``vertex_confidence`` is the converged
    c* per entity — the "reasoning beyond cosine" witness (a 2-hop entity with zero query
    similarity earns mass here) and the input M5's chain ordering needs. ``anchors`` is the
    restart vector that was used (audit/debug); ``pruned_candidates``/
    ``expanded_candidates`` witness how much each stage narrowed — honest observability,
    not silent truncation.
    """

    rho: dict[str, float]
    fields: dict[str, dict[str, Any]]
    anchors: dict[str, float]
    vertex_confidence: dict[str, float]
    expanded_candidates: int
    pruned_candidates: int


# -- the Propagator seam (spec 6.2: kernel is pluggable; RWR is the default) ----


@runtime_checkable
class Propagator(Protocol):
    """Diffuses anchored restart mass over a query-local bipartite incidence."""

    def propagate(
        self,
        incidence: dict[str, list[str]],
        restart: dict[str, float],
        config: HypergraphConfig,
    ) -> dict[str, float]:
        """``incidence``: hyperedge_id → entity node_ids. Returns vertex confidences c*."""
        ...


class RWRPropagator:
    """The default kernel: two-stage random walk with restart (paper Eq 26–28), pure Python.

    S = B·D_e⁻¹·Bᵀ·D_v⁻¹ applied as two normalised passes per step (vertex → hyperedge →
    vertex), with ``c(t+1) = λ·r + (1−λ)·S·c(t)``. Column-stochasticity holds by
    construction: each pass divides by the emitting side's degree, so every column of S
    sums to 1 (pinned by test, not by trust). Runs on the PRUNED graph only (≤ k_h
    hyperedges), which is what makes a dependency-free implementation fast enough.
    """

    def propagate(
        self,
        incidence: dict[str, list[str]],
        restart: dict[str, float],
        config: HypergraphConfig,
    ) -> dict[str, float]:
        """Iterate to c* (finite-step approximation of the unique fixed point)."""
        edges_of: dict[str, list[str]] = {}
        for hid, nodes in incidence.items():
            for v in nodes:
                edges_of.setdefault(v, []).append(hid)
        if not edges_of:
            return {}
        d_v = {v: len(hs) for v, hs in edges_of.items()}
        d_e = {hid: len(nodes) for hid, nodes in incidence.items() if nodes}

        total = sum(restart.get(v, 0.0) for v in edges_of)
        if total <= 0.0:
            r = {v: 1.0 / len(edges_of) for v in edges_of}
        else:
            r = {v: restart.get(v, 0.0) / total for v in edges_of}

        lam = config.restart
        c = dict(r)
        for _ in range(config.steps):
            # vertex → hyperedge (divide by the emitting vertex's degree)…
            edge_mass: dict[str, float] = {}
            for v, mass in c.items():
                if mass == 0.0:
                    continue
                share = mass / d_v[v]
                for hid in edges_of[v]:
                    edge_mass[hid] = edge_mass.get(hid, 0.0) + share
            # …hyperedge → vertex (divide by the hyperedge's arity)
            nxt = {v: 0.0 for v in c}
            for hid, mass in edge_mass.items():
                share = mass / d_e[hid]
                for v in incidence[hid]:
                    if v in nxt:
                        nxt[v] += share
            c = {v: lam * r[v] + (1.0 - lam) * nxt[v] for v in c}
        return c


# -- the pipeline: entry → expand → prune → propagate → gate --------------------


def hypergraph_signals(
    store: HyperedgeKV,
    *,
    alias_surfaces: list[str] | None = None,
    ann_node_ids: list[str] | None = None,
    eu_sims: dict[str, float] | None = None,
    config: HypergraphConfig | None = None,
    propagator: Propagator | None = None,
) -> HypergraphSignals:
    """Entry + candidate expansion + propagation over an OPEN store, for one query.

    The caller owns the store's lifetime (``with HyperedgeKV.open(...) as kv:``) so a
    query performs exactly one open/close — nothing outlives the call. Tier 0/1 entry per
    ADR-0059: ``alias_surfaces`` anchor at weight 1.0 (the query names them), ``ann_node_ids``
    at ``alpha``. ``eu_sims`` is the caller's eu_vecs ANN similarity (s_rel); absent, the
    ``s_hyp`` prune degrades to entry-overlap only (β=1) — a documented tier-0 degradation,
    never a silent one.

    A store opened with the (default) shared :class:`HypergraphIndex` runs expansion and
    prune over the precomputed int-indexed postings/incidence; an uncached or
    :meth:`~HyperedgeKV.from_bytes` store scans per call. The two paths are byte-identical
    (pinned by tests and the ticket-01 real-data parity run).
    """
    cfg = config or HypergraphConfig()
    kernel = propagator or RWRPropagator()

    # -- entry (ADR-0059): anchored restart weights ------------------------
    anchors: dict[str, float] = {}
    for node_id in ann_node_ids or []:
        anchors[node_id] = max(anchors.get(node_id, 0.0), cfg.alpha)
    for surface in alias_surfaces or []:
        for node_id in store.alias(surface):
            anchors[node_id] = 1.0  # V_llm: the query actually names it (spec 2.1)
    if not anchors:
        return HypergraphSignals(
            rho={},
            fields={},
            anchors={},
            vertex_confidence={},
            expanded_candidates=0,
            pruned_candidates=0,
        )

    sims = eu_sims or {}
    beta = cfg.beta if sims else 1.0  # tier-0 degradation: entry-overlap only

    index = store.index
    fields_of: Callable[[str], dict[str, Any]]
    if index is not None:
        anchor_rows = index.node_rows(list(anchors))
        candidate_rows = index.expand(anchor_rows, cfg)
        expanded = len(candidate_rows)
        candidate_rows = index.prune(candidate_rows, anchor_rows, beta, sims, cfg)
        pruned = len(candidate_rows)
        candidates = [index.hyperedge_id(row) for row in candidate_rows]
        incidence = {
            hid: index.entity_node_ids(row)
            for hid, row in zip(candidates, candidate_rows, strict=True)
        }

        row_of = dict(zip(candidates, candidate_rows, strict=True))
        resolved_index = index

        def _indexed_fields(hid: str) -> dict[str, Any]:
            return resolved_index.solver_fields_at(store, row_of[hid])

        fields_of = _indexed_fields
    else:
        candidates, incidence, expanded, pruned, fields_of = _scan_candidates(
            store, anchors, beta, sims, cfg
        )

    # -- propagation on the small query-aware hypergraph (Eq 26–28) ----------
    c_star = kernel.propagate(incidence, anchors, cfg)

    # -- ρ pooling + vertex gate + coverage (Eq 29–33) -----------------------
    c_max = max(c_star.values(), default=0.0)
    retained = {v for v, val in c_star.items() if c_max > 0.0 and val >= cfg.theta_v * c_max}
    rho: dict[str, float] = {}
    retained_fields: dict[str, dict[str, Any]] = {}
    for hid in candidates:
        nodes = incidence[hid]
        if not nodes or not any(v in retained for v in nodes):
            continue  # coverage gate: no retained entity → not a candidate evidence unit
        vals = [c_star.get(v, 0.0) for v in nodes]
        rho[hid] = sum(vals) / len(vals)  # mean pooling (Eq 29; sweep 6b: within noise)
        retained_fields[hid] = fields_of(hid)
    return HypergraphSignals(
        rho=rho,
        fields=retained_fields,
        anchors=anchors,
        vertex_confidence=c_star,
        expanded_candidates=expanded,
        pruned_candidates=pruned,
    )


def _scan_candidates(
    store: HyperedgeKV,
    anchors: dict[str, float],
    beta: float,
    sims: dict[str, float],
    cfg: HypergraphConfig,
) -> tuple[
    list[str], dict[str, list[str]], int, int, Callable[[str], dict[str, Any]]
]:
    """The per-call expansion + prune over the store bytes — the original, index-free path.

    Kept verbatim as the uncached/:meth:`~HyperedgeKV.from_bytes` path and as the parity
    oracle the indexed path is pinned against.
    """
    # -- expansion: hop-priority, hub-capped (ticket 20 — both guards measured) --
    seen: dict[str, dict[str, Any]] = {}
    by_hop: list[list[str]] = []
    frontier = sorted(anchors)
    for hop in range(cfg.hops):
        layer: list[str] = []
        for v in frontier:
            ids, _gammas = store.postings(v)
            for hid in ids[: cfg.hub_cap]:  # hub guard: γ-sorted, top slice only
                if hid not in seen:
                    solver = store.solver_fields(hid)
                    if solver is None:  # pragma: no cover - a postings/solver mismatch
                        continue
                    seen[hid] = solver
                    layer.append(hid)
        if not layer:
            break
        by_hop.append(layer)
        if hop + 1 < cfg.hops:
            nxt: set[str] = set()
            for hid in layer[: cfg.expand_cap]:
                nxt.update(seen[hid]["entity_node_ids"])
            frontier = sorted(nxt)

    # hop-priority truncation: NEVER sacrifice an inner hop to the cap (ticket 20 —
    # γ-global truncation silently reverts a multi-hop retriever to 1-hop).
    candidates: list[str] = []
    for layer in by_hop:
        room = cfg.max_candidates - len(candidates)
        if room <= 0:
            break
        if len(layer) > room:
            layer = sorted(layer, key=lambda h: (-seen[h]["gamma"], h))[:room]
        candidates.extend(layer)
    expanded = len(candidates)

    # -- s_hyp prune BEFORE propagation (Eq 16–19; the ticket-20 verified fix) ----
    def s_hyp(hid: str) -> float:
        nodes = seen[hid]["entity_node_ids"]
        overlap = sum(1 for v in nodes if v in anchors) / (len(nodes) + 1e-9)
        return beta * overlap + (1.0 - beta) * float(sims.get(hid, 0.0))

    if len(candidates) > cfg.k_h:
        candidates = sorted(candidates, key=lambda h: (-s_hyp(h), h))[: cfg.k_h]
    pruned = len(candidates)

    incidence = {hid: list(seen[hid]["entity_node_ids"]) for hid in candidates}
    return candidates, incidence, expanded, pruned, seen.__getitem__

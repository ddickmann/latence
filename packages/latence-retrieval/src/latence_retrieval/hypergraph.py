# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Hypergraph entry + confidence propagation (M3: ADR-0056/0059, ticket-20 verified order).

The query-time half of chain-of-evidence retrieval, over the sidecar KV the pipeline
emitted (``graph-hyperedges.kv``, ADR-0056). Everything here is stateless in the ADR-0048
sense — the store is opened per call, read, and closed; nothing is held between queries —
and **zero-dep**: the paper-faithful pipeline order makes pure-Python propagation viable,
because the expensive step (propagation) runs only AFTER the candidate universe is pruned
to a small query-aware hypergraph.

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

The KV reader here is a second, independent implementation of the ADR-0056 store FORMAT
(the format is the contract, not the library): ``latence-retrieval`` depends on pydantic
alone and cannot import ``latence-core``. Cross-implementation interop is pinned by a test
in latence-core (writer there, reader here, byte round-trip).
"""

from __future__ import annotations

import json
import mmap
import struct
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "HyperedgeKV",
    "HypergraphConfig",
    "HypergraphSignals",
    "Propagator",
    "RWRPropagator",
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
    def open(cls, path: str) -> HyperedgeKV:
        """Open the store file for one call's reads (mmap; closed with the reader)."""
        fh = open(path, "rb")  # noqa: SIM115 - lifetime is the reader's, closed in close()
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        return cls(buf, _close=(buf, fh))

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
    sims = eu_sims or {}
    beta = cfg.beta if sims else 1.0  # tier-0 degradation: entry-overlap only

    def s_hyp(hid: str) -> float:
        nodes = seen[hid]["entity_node_ids"]
        overlap = sum(1 for v in nodes if v in anchors) / (len(nodes) + 1e-9)
        return beta * overlap + (1.0 - beta) * float(sims.get(hid, 0.0))

    if len(candidates) > cfg.k_h:
        candidates = sorted(candidates, key=lambda h: (-s_hyp(h), h))[: cfg.k_h]
    pruned = len(candidates)

    # -- propagation on the small query-aware hypergraph (Eq 26–28) ----------
    incidence = {hid: list(seen[hid]["entity_node_ids"]) for hid in candidates}
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
        retained_fields[hid] = seen[hid]
    return HypergraphSignals(
        rho=rho,
        fields=retained_fields,
        anchors=anchors,
        vertex_confidence=c_star,
        expanded_candidates=expanded,
        pruned_candidates=pruned,
    )

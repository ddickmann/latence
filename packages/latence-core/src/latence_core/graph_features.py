"""Pure reference graph-feature engines — centrality + community, deterministic (ADR-0051).

The dependency-free algorithms behind the OPTIONAL ``graph_features`` enrichment (ADR-0051): given
the assembled graph as plain ``(node_id)`` + directed ``(source, target)`` tuples, compute a
per-node **centrality** score and a per-node **community** id. Two honest reference families ship
here — degree/connected-components and PageRank/label-propagation — exactly how the GraphAssembler
shipped a pure reference builder before any learned Provider. The blessed Louvain/Leiden community
detection is a pluggable, license-verified package (ADR-0051), not a core dependency.

Every engine is DETERMINISTIC: nodes are processed in sorted ``node_id`` order, every score is
rounded to :data:`_ROUND` decimals (so accumulated float error never makes two runs differ by a
byte), and every community id is the community's lexicographically-smallest ``node_id`` (a
content-addressed, stable id). Self-loops are impossible (a
:class:`~latence_core.contracts.GraphEdge` forbids equal endpoints); an edge to an unknown node is
dropped. The engines are pure over plain strings/tuples (no Pydantic) — the Providers in
:mod:`latence_core.stages.graph_features` map the result onto the
:class:`~latence_core.capability.GraphFeatures` contract.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Decimal places every centrality score is rounded to. Rounding is what makes an iterative
#: engine (PageRank) byte-deterministic: the last ulps of accumulated float error are discarded so
#: two independent runs (or processes) agree exactly, the Baseline bar.
_ROUND = 8

#: PageRank power-iteration parameters (the standard, widely-reproduced defaults). Fixed so a run is
#: deterministic and terminates: damping 0.85, at most _PR_MAX_ITER sweeps, stop early when the
#: L1 change between sweeps drops below _PR_TOL.
_PR_DAMPING = 0.85
_PR_MAX_ITER = 100
_PR_TOL = 1e-12

#: Label-propagation bound: at most _LPA_MAX_ROUNDS synchronous sweeps (it converges far sooner on
#: real KGs; the bound guarantees termination even on a pathological oscillating graph).
_LPA_MAX_ROUNDS = 100


def _sorted_nodes(nodes: Iterable[str]) -> list[str]:
    """The unique node ids in deterministic (sorted) order — the single processing order."""
    return sorted(set(nodes))


def _clean_edges(
    node_set: set[str], edges: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Directed edges with both endpoints known and distinct, deduped + sorted (determinism)."""
    kept = {
        (src, tgt)
        for src, tgt in edges
        if src != tgt and src in node_set and tgt in node_set
    }
    return sorted(kept)


def _undirected_adjacency(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, set[str]]:
    """Undirected neighbour sets (both directions) — the shape degree + LPA + components read."""
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for src, tgt in edges:
        adjacency[src].add(tgt)
        adjacency[tgt].add(src)
    return adjacency


# ---------------------------------------------------------------------------
# Community: connected components (Union-Find) and label propagation
# ---------------------------------------------------------------------------


def connected_component_communities(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, str]:
    """Community id per node = the lexicographically-smallest node id in its connected component.

    Union-Find over the undirected edges (the classic components). An isolated node is its own
    community (its own id). The id is content-addressed (the min member id), so it is stable across
    runs and independent of edge/iteration order — the deterministic community label ADR-0051 wants.
    """
    parent: dict[str, str] = {node: node for node in nodes}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        # Path-compress so repeated finds stay near-flat (determinism unaffected — same roots).
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for src, tgt in edges:
        r_src, r_tgt = find(src), find(tgt)
        if r_src != r_tgt:
            # Union by smaller id as the parent so the eventual root of a component is deterministic
            # regardless of union order (though the canonical id below does not rely on it).
            hi, lo = (r_src, r_tgt) if r_src > r_tgt else (r_tgt, r_src)
            parent[hi] = lo

    # Canonicalise: every node's community id is the MIN node id among all nodes sharing its root —
    # a content-addressed, order-independent label.
    members: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        members[find(node)].append(node)
    canonical: dict[str, str] = {}
    for group in members.values():
        community_id = min(group)
        for node in group:
            canonical[node] = community_id
    return canonical


def label_propagation_communities(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, str]:
    """Community id per node via DETERMINISTIC synchronous label propagation (ADR-0051 reference).

    Each node starts labelled by its own id. Each SYNCHRONOUS round, a node adopts the label carried
    by the most nodes among ITSELF plus its neighbours (undirected); ties break to the
    lexicographically-smallest label — a total, order-independent rule, so the fixed point is
    byte-identical across runs (unlike the classic *asynchronous, randomised* LPA). Counting the
    node's OWN current label (semi-synchronous LPA) is what damps the well-known synchronous
    oscillation on bipartite structures — without it a bare 2-node component would swap labels
    forever and never merge; with it, both nodes converge to the smaller id. Iterated to a fixed
    point (or :data:`_LPA_MAX_ROUNDS`). An isolated node keeps its own label. The returned id is
    then canonicalised to the community's smallest member id (as with components), so both reference
    families share the same id scheme.
    """
    adjacency = _undirected_adjacency(nodes, edges)
    labels: dict[str, str] = {node: node for node in nodes}

    for _ in range(_LPA_MAX_ROUNDS):
        next_labels: dict[str, str] = {}
        for node in nodes:
            neighbours = adjacency[node]
            if not neighbours:
                next_labels[node] = labels[node]
                continue
            counts: dict[str, int] = defaultdict(int)
            # Count the node's OWN current label first (semi-synchronous LPA — damps oscillation).
            counts[labels[node]] += 1
            for neighbour in neighbours:
                counts[labels[neighbour]] += 1
            best_count = max(counts.values())
            # Highest count, ties → smallest label (total deterministic order).
            next_labels[node] = min(
                label for label, count in counts.items() if count == best_count
            )
        if next_labels == labels:
            break
        labels = next_labels

    # Canonicalise a final label to its community's smallest member id (matches components' scheme).
    members: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        members[labels[node]].append(node)
    canonical: dict[str, str] = {}
    for group in members.values():
        community_id = min(group)
        for node in group:
            canonical[node] = community_id
    return canonical


# ---------------------------------------------------------------------------
# Centrality: degree and PageRank
# ---------------------------------------------------------------------------


def degree_centrality(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, float]:
    """Normalised undirected degree centrality: distinct-neighbour count / (n - 1), in ``[0, 1]``.

    The standard degree-centrality normalisation (Freeman): a node's score is the fraction of the
    other ``n - 1`` nodes it is adjacent to. A single-node graph (``n <= 1``) scores 0.0 (no other
    node to be central relative to). Rounded to :data:`_ROUND` for byte-determinism.
    """
    adjacency = _undirected_adjacency(nodes, edges)
    n = len(nodes)
    if n <= 1:
        return {node: 0.0 for node in nodes}
    denom = float(n - 1)
    return {node: round(len(adjacency[node]) / denom, _ROUND) for node in nodes}


def pagerank_centrality(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, float]:
    """DETERMINISTIC PageRank over the DIRECTED edges (damping 0.85, power iteration).

    Standard PageRank: rank flows along out-edges, a ``1 - damping`` teleport keeps it well-defined,
    and a **dangling** node (no out-edges) redistributes its rank uniformly (the canonical dangling
    handling, so the vector stays a probability distribution). Iteration order is fixed (sorted
    nodes), init is uniform ``1/n``, and each sweep's result is rounded, so two runs are
    byte-identical. Returns ``{}`` for an empty graph.
    """
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    out_neighbours: dict[str, list[str]] = {node: [] for node in nodes}
    for src, tgt in edges:
        out_neighbours[src].append(tgt)
    dangling = [node for node in nodes if not out_neighbours[node]]

    base = (1.0 - _PR_DAMPING) / n
    rank: dict[str, float] = {node: 1.0 / n for node in nodes}

    for _ in range(_PR_MAX_ITER):
        # Rank mass sitting on dangling nodes teleports uniformly (keeps the vector normalised).
        dangling_mass = _PR_DAMPING * sum(rank[node] for node in dangling) / n
        nxt: dict[str, float] = {node: base + dangling_mass for node in nodes}
        for src in nodes:
            targets = out_neighbours[src]
            if not targets:
                continue
            share = _PR_DAMPING * rank[src] / len(targets)
            for tgt in targets:
                nxt[tgt] += share
        delta = sum(abs(nxt[node] - rank[node]) for node in nodes)
        rank = nxt
        if delta < _PR_TOL:
            break

    return {node: round(rank[node], _ROUND) for node in nodes}

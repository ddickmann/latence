# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Chain assembly + structured context emission (M5: spec 3.4, the 6.3 directed preference).

Selection (M4, :func:`~latence_retrieval.coe.select`) gives a *set*; generation wants *chains*.
:func:`assemble_chains` turns a :class:`~latence_retrieval.coe.CoeSelection` plus the
:class:`~latence_retrieval.hypergraph.HypergraphSignals` it was computed from into ordered
evidence chains and the structured context string spec 3.4 prescribes. Like every query-time
component it is stateless (ADR-0048) and zero-dep: pure Python over the pydantic-only package.

**Chains are connected components of G_sel** — nodes = the selected Hyperedges, an (undirected)
link iff two share ≥ 1 entity. G_sel is never materialised as a pairwise matrix: components are
found by BFS over an inverted node→selected-candidate index (the same incremental/sparse
discipline ticket 20 imposed on the selector; a dense ``|A|²`` overlap structure is the measured
latency shape this milestone line refuses everywhere).

**Intra-chain order** (spec 3.4): steps are emitted greedily by minimal key
``(hop, direction_rank, −ρ, id)``:

* ``hop`` — BFS layer distance of the Hyperedge's closest entity to the anchor entities V0
  (:attr:`~latence_retrieval.hypergraph.HypergraphSignals.anchors`), computed over shared
  entities within the chain. Evidence distance from the query stays the PRIMARY order.
* ``direction_rank`` — the spec 6.3 **directed-relation preference**, the partial mitigation for
  sequentially-constrained multi-hop. Solver fields deliberately carry no relation triples, so
  direction rides an optional caller seam: ``relations_lookup(hyperedge_id) → [(head_node,
  label, tail_node), …]`` (e.g. backed by the emitted graph parquet). The exact rule: with the
  lookup supplied, rank 0 when ≥ 1 of the Hyperedge's triples has its **head** node already
  *resolved* (V0 ∩ chain for an anchored chain, plus every entity of the steps emitted so far —
  the step continues an established head→tail path), rank 2 when it carries triples but none of
  their heads is resolved (its relations point backward into the resolved evidence at best), and
  rank 1 when it carries no triples (no directional testimony either way). Without the lookup the
  rank is constant, so the order is EXACTLY the undirected ``(hop, −ρ, id)`` of spec 3.4 —
  the preference refines within a hop layer, it never overrides evidence distance.
* ``−ρ, id`` — propagated confidence descending, then ascending Hyperedge id: the same
  determinism discipline as the selector. Identical inputs yield identical output **bytes**.

**Peripheral evidence** (spec 3.4): a component whose entities never touch V0 (possible after
2-hop expansion) is emitted LAST under one ``## Peripheral evidence`` header — present but
honestly demoted, never silently dropped here (budget-tightness already had its say in
selection). A peripheral chain has no V0 to BFS from, so it is seeded at its (ρ desc, id asc)
best Hyperedge — step 1 by definition — with hops measured from that seed's entities and the
greedy loop resolving forward from there. Anchored chains order among themselves by max-ρ
descending (smallest id tie-break); peripheral chains likewise, after all anchored ones.

**τ text is redaction-safe by construction**: ``tau_lookup`` is typically an open store's
:meth:`~latence_retrieval.hypergraph.HyperedgeKV.tau_text`, and ADR-0057 enforces AT THE WRITER
that a stored ``tau_text`` is the corpus redaction variant — the rendered context can never
contain text the exported corpus would not contain, without this module doing (or being able to
do) anything about it. A selected id whose τ cannot be fetched fails loudly (the ADR-0058
posture: unverifiable evidence never ships on trust). ``src_lookup`` optionally maps a Hyperedge
id to a human doc/chunk ref for the ``src=`` slot; absent, the content-addressed Hyperedge id —
itself a stable reference — is emitted.

**The trace section** (spec 3.5): with a ``trace_sink`` supplied, one JSON-serialisable dict
(``stage="chains"``: chain count/sizes, peripheral count, per-chain ordered ids) goes through
the same sink shape as the selector's trace. Traces carry **ids and counts only, never raw
``tau_text``** — a deliberate privacy choice, because the trace is the weak-supervision seed for
the future learned utility and may travel further than the corpus it describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from latence_retrieval.coe import CoeSelection
    from latence_retrieval.hypergraph import HypergraphSignals

__all__ = [
    "Chain",
    "ChainAssembly",
    "ChainStep",
    "assemble_chains",
]


@dataclass(frozen=True)
class ChainStep:
    """One emitted evidence step: a selected Hyperedge in its chain position.

    ``hop`` is the BFS layer distance from the chain's seed set (V0 for an anchored chain, the
    seed Hyperedge's entities for a peripheral one). ``relations`` is whatever the caller's
    ``relations_lookup`` returned (sorted for determinism), empty when no lookup was supplied.
    """

    hyperedge_id: str
    hop: int
    rho: float
    gamma: float
    entity_node_ids: list[str]
    relations: list[tuple[str, str, str]]
    text: str


@dataclass(frozen=True)
class Chain:
    """One connected component of G_sel, steps already in emission order.

    ``anchor_node_ids`` is the sorted set of this chain's entities that are anchors (V0);
    empty means the chain is peripheral evidence (spec 3.4).
    """

    anchor_node_ids: list[str]
    steps: list[ChainStep]

    @property
    def peripheral(self) -> bool:
        """True when no entity of this chain touches the query's anchor set."""
        return not self.anchor_node_ids


@dataclass(frozen=True)
class ChainAssembly:
    """The chain-assembly result: ordered chains + the rendered generation context.

    ``chains`` holds anchored chains first (max-ρ descending), peripheral chains last;
    ``text`` is the spec-3.4 structured context — deterministic bytes for identical inputs.
    """

    chains: list[Chain]
    text: str


def _entity_hops(
    seeds: set[str],
    members: list[str],
    nodes_of: dict[str, list[str]],
    node_to: dict[str, list[str]],
) -> dict[str, int]:
    """BFS layer distance of each of ``members``'s entities from ``seeds``, over shared edges."""
    dist = {node: 0 for node in seeds}
    member_set = set(members)
    frontier = sorted(seeds)
    layer = 0
    while frontier:
        nxt: set[str] = set()
        for node in frontier:
            for hid in node_to.get(node, []):
                if hid not in member_set:
                    continue
                for other in nodes_of[hid]:
                    if other not in dist:
                        dist[other] = layer + 1
                        nxt.add(other)
        frontier = sorted(nxt)
        layer += 1
    return dist


def _order_members(
    members: list[str],
    hops: dict[str, int],
    resolved: set[str],
    nodes_of: dict[str, list[str]],
    rho: dict[str, float],
    relations: dict[str, list[tuple[str, str, str]]],
    directed: bool,
) -> list[str]:
    """The greedy ``(hop, direction_rank, −ρ, id)`` emission order (docstring rule)."""

    def direction_rank(hid: str) -> int:
        if not directed:
            return 0  # no lookup: constant rank, the plain undirected spec-3.4 order
        triples = relations[hid]
        if not triples:
            return 1  # no directional testimony either way
        return 0 if any(head in resolved for head, _, _ in triples) else 2

    remaining = set(members)
    ordered: list[str] = []
    while remaining:
        best = min(
            remaining,
            key=lambda hid: (hops[hid], direction_rank(hid), -rho[hid], hid),
        )
        ordered.append(best)
        remaining.discard(best)
        resolved.update(nodes_of[best])
    return ordered


def assemble_chains(
    selection: CoeSelection,
    signals: HypergraphSignals,
    *,
    tau_lookup: Callable[[str], str | None],
    relations_lookup: Callable[[str], list[tuple[str, str, str]]] | None = None,
    src_lookup: Callable[[str], str | None] | None = None,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> ChainAssembly:
    """Assemble ``selection`` into ordered chains and the spec-3.4 structured context.

    ``signals`` must be the SAME propagation result the selection was computed from (every
    selected id is looked up in ``signals.rho``/``signals.fields`` — a miss is a loud error,
    never a silent skip). ``tau_lookup`` supplies each step's span text — typically an open
    store's :meth:`~latence_retrieval.hypergraph.HyperedgeKV.tau_text`, redaction-safe by
    construction (ADR-0057, enforced at the writer). ``relations_lookup`` opts into the spec
    6.3 directed ordering preference (see the module docstring for the exact rule);
    ``src_lookup`` optionally names a doc/chunk ref for the ``src=`` slot. ``trace_sink``
    receives one ``stage="chains"`` dict — ids and counts only, never ``tau_text``.

    Deterministic by construction: identical inputs yield identical output bytes.
    """
    anchors = set(signals.anchors)
    ordered_ids = list(selection.selected)

    nodes_of: dict[str, list[str]] = {}
    rho: dict[str, float] = {}
    for hid in ordered_ids:
        record = signals.fields.get(hid)
        if record is None or hid not in signals.rho:
            msg = (
                f"chain assembly got selected hyperedge {hid!r} that is absent from the "
                "signals it was allegedly selected from — selection and signals must come "
                "from the same query."
            )
            raise ValueError(msg)
        nodes_of[hid] = [str(node) for node in record["entity_node_ids"]]
        rho[hid] = float(signals.rho[hid])

    relations: dict[str, list[tuple[str, str, str]]] = {}
    for hid in ordered_ids:
        if relations_lookup is None:
            relations[hid] = []
        else:
            relations[hid] = sorted(
                (str(head), str(label), str(tail))
                for head, label, tail in relations_lookup(hid)
            )

    # -- G_sel components via the inverted node→selected index (never a dense matrix) --
    node_to: dict[str, list[str]] = {}
    for hid in sorted(ordered_ids):
        for node in set(nodes_of[hid]):
            node_to.setdefault(node, []).append(hid)
    components: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(ordered_ids):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[str] = []
        while stack:
            hid = stack.pop()
            component.append(hid)
            for node in nodes_of[hid]:
                for other in node_to[node]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        components.append(sorted(component))

    # -- order within each component; anchored chains first, peripheral last ----------
    directed = relations_lookup is not None
    anchored: list[Chain] = []
    peripheral: list[Chain] = []
    for members in components:
        chain_nodes = {node for hid in members for node in nodes_of[hid]}
        chain_anchors = sorted(chain_nodes & anchors)
        if chain_anchors:
            seeds = set(chain_anchors)
            resolved = set(chain_anchors)
            hops = _entity_hops(seeds, members, nodes_of, node_to)
            edge_hop = {
                hid: min(hops[node] for node in nodes_of[hid] if node in hops)
                for hid in members
            }
            order = _order_members(
                members, edge_hop, resolved, nodes_of, rho, relations, directed
            )
        else:
            # peripheral: no V0 to BFS from — seed at the (ρ desc, id asc) best Hyperedge
            seed = min(members, key=lambda hid: (-rho[hid], hid))
            seeds = set(nodes_of[seed])
            hops = _entity_hops(seeds, members, nodes_of, node_to)
            edge_hop = {
                hid: min(hops[node] for node in nodes_of[hid] if node in hops)
                for hid in members
            }
            rest = [hid for hid in members if hid != seed]
            resolved = set(nodes_of[seed])
            order = [seed, *_order_members(
                rest, edge_hop, resolved, nodes_of, rho, relations, directed
            )]

        steps: list[ChainStep] = []
        for hid in order:
            text = tau_lookup(hid)
            if text is None:
                msg = (
                    f"chain assembly could not fetch tau_text for selected hyperedge {hid!r}: "
                    "evidence that cannot be rendered fails loudly, it is never emitted blank."
                )
                raise ValueError(msg)
            record = signals.fields[hid]
            steps.append(
                ChainStep(
                    hyperedge_id=hid,
                    hop=edge_hop[hid],
                    rho=rho[hid],
                    gamma=float(record["gamma"]),
                    entity_node_ids=sorted(set(nodes_of[hid])),
                    relations=relations[hid],
                    text=text,
                )
            )
        chain = Chain(anchor_node_ids=chain_anchors, steps=steps)
        (anchored if chain_anchors else peripheral).append(chain)

    def chain_key(chain: Chain) -> tuple[float, str]:
        return (
            -max(step.rho for step in chain.steps),
            min(step.hyperedge_id for step in chain.steps),
        )

    anchored.sort(key=chain_key)
    peripheral.sort(key=chain_key)
    chains = [*anchored, *peripheral]

    # -- spec 3.4 structured context emission -----------------------------------------
    lines: list[str] = []
    peripheral_header_emitted = False
    for number, chain in enumerate(chains, start=1):
        if lines:
            lines.append("")
        if chain.peripheral and not peripheral_header_emitted:
            lines.append("## Peripheral evidence")
            lines.append("")
            peripheral_header_emitted = True
        if chain.peripheral:
            lines.append(f"## Chain {number} (peripheral)")
        else:
            lines.append(f"## Chain {number} (anchor: {', '.join(chain.anchor_node_ids)})")
        for position, step in enumerate(chain.steps, start=1):
            src = None if src_lookup is None else src_lookup(step.hyperedge_id)
            src_ref = src if src is not None else step.hyperedge_id
            lines.append(
                f"[step {position}] (rho={step.rho:.2f}, gamma={step.gamma:.2f}, "
                f"src={src_ref}) {step.text}"
            )
            detail = f"    entities: {', '.join(step.entity_node_ids)}"
            if step.relations:
                rendered = ", ".join(
                    f"{head} --{label}--> {tail}" for head, label, tail in step.relations
                )
                detail += f" | relations: {rendered}"
            lines.append(detail)
    text = "\n".join(lines) + "\n" if lines else ""

    if trace_sink is not None:
        trace_sink(
            {
                "stage": "chains",
                "selected_count": len(ordered_ids),
                "chain_count": len(chains),
                "chain_sizes": [len(chain.steps) for chain in chains],
                "anchored_count": len(anchored),
                "peripheral_count": len(peripheral),
                "directed_preference": directed,
                "chains": [
                    {
                        "hyperedge_ids": [step.hyperedge_id for step in chain.steps],
                        "anchor_node_ids": chain.anchor_node_ids,
                        "peripheral": chain.peripheral,
                    }
                    for chain in chains
                ],
            }
        )
    return ChainAssembly(chains=chains, text=text)

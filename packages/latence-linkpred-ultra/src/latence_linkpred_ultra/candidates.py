"""Type-aware tiered 2-hop candidate generation (bounded, rare-relation full recall).

The candidate generator the learned :mod:`~latence_linkpred_ultra.provider` scores. Given the
observed KG — node types (from the node labels) and observed ``(head, label, tail)`` edges — it
proposes the ``(head, relation, tail)`` triples worth scoring, so the learned scorer never has to
rank the full ``N x R x N`` product. It is the ASSET-INVENTORY *candidate_generator* discipline
(the Link-prediction stack, 5/5, PORT-AS-IS): **type-aware** + **tiered 2-hop** + **bounded** with a
**rare-relation recall guarantee**, and every cap that drops a candidate is logged (no silent
truncation).

The generator is deliberately **pure and torch-free** (plain dataclasses over ids), so it is unit-
testable offline and the heavy scorer is a separate seam. It never proposes an observed pair (the
asserted-collision drop is upstream of scoring, matching the reference completer) and never a
self-loop.

The recipe (per relation ``r``):

* **Type filter.** From the observed edges of ``r`` collect the set of head types and tail types
  ``r`` connects (a relation's *type signature*). A candidate ``(h, r, t)`` is type-valid only when
  ``type(h)`` is an observed head type of ``r`` AND ``type(t)`` is an observed tail type of ``r`` —
  so ``partner_of`` (org→org) never proposes an org→person edge. A relation observed on untyped or
  freely-typed endpoints (no consistent signature) admits any type (the filter never *invents* a
  constraint that would hurt recall).
* **Tiered 2-hop candidate set.** For each ``r`` the candidate heads/tails are the entities within
  the **2-hop neighbourhood** of ``r``'s observed endpoints (reachable over any relation in ≤2
  hops), intersected with the type filter — the tiers that make the search local instead of
  quadratic. When the 2-hop neighbourhood is empty for a relation (an isolated component), it falls
  back to the type-valid entity pool so the relation is still completable.
* **Bounded fan-out with rare-relation recall.** A **rare** relation (observed at most
  ``rare_relation_max_edges`` times) keeps ALL its type-valid candidates — full recall, never
  capped — because rare relations are exactly where a learned scorer earns its keep and a cap would
  silently forfeit them. A **dense** relation is capped at ``max_candidates_per_relation`` (the
  bounded-fan-out guard, the ported relation quadratic-guard discipline); the drop is **logged**
  with the count, never silent. Selection under the cap is deterministic (sorted candidate key), so
  a seeded run is byte-identical.

The output is a deterministic, sorted list of :class:`Candidate` triples (by ``(relation, head,
tail)`` id) — the fixed candidate order the provider scores greedily (no sampling), so the whole
pipeline is byte-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservedEdge:
    """One asserted edge the generator reasons over (head node id, relation label, tail node id)."""

    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class Candidate:
    """One proposed (not-yet-asserted) triple to score: ``(head, relation, tail)`` node/label ids.

    ``rare`` records whether the candidate's relation was treated as rare (kept at full recall) —
    surfaced so a caller/report can see which relations bypassed the fan-out cap.
    """

    head: str
    relation: str
    tail: str
    rare: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        """The deterministic sort/identity key (relation, head, tail)."""
        return (self.relation, self.head, self.tail)


@dataclass(frozen=True)
class CandidateConfig:
    """Tuning knobs for :class:`CandidateGenerator` (all deterministic).

    ``max_candidates_per_relation`` bounds the fan-out per **dense** relation (0 = unlimited). A
    relation observed at most ``rare_relation_max_edges`` times is **rare** and keeps ALL its
    type-valid candidates regardless of the cap (the recall guarantee). ``two_hop`` restricts
    candidates to the 2-hop neighbourhood of a relation's observed endpoints (the tiered locality);
    set it False to consider the whole type-valid pool (recall-max, slower). ``type_aware``
    enables the head/tail type filter from a relation's observed signature.
    """

    max_candidates_per_relation: int = 512
    rare_relation_max_edges: int = 4
    two_hop: bool = True
    type_aware: bool = True


class CandidateGenerator:
    """Propose type-aware, tiered-2-hop, bounded candidates with rare-relation full recall.

    Pure and deterministic: given the same nodes/types/observed edges + config it returns the same
    sorted :class:`Candidate` list. It never proposes an observed pair or a self-loop. Any cap that
    drops candidates is logged with the count (no silent truncation).
    """

    def __init__(self, config: CandidateConfig | None = None) -> None:
        self._config = config or CandidateConfig()
        #: Populated per :meth:`generate` call: ``relation -> dropped_candidate_count`` for every
        #: dense relation whose type-valid candidate set exceeded the cap. A report can surface it.
        self.capped: dict[str, int] = {}

    def generate(
        self,
        node_types: dict[str, str],
        observed: list[ObservedEdge],
    ) -> list[Candidate]:
        """Return the deterministic, bounded, type-aware candidate triples for the observed graph.

        ``node_types`` maps every node id to its type (the node label); a node absent from the map
        is treated as untyped (admitted by any relation's type filter). ``observed`` is the asserted
        edge set. Candidates already observed, or self-loops, are never proposed.
        """
        self.capped = {}
        observed_set = {(e.head, e.relation, e.tail) for e in observed}
        nodes = sorted(set(node_types) | {e.head for e in observed} | {e.tail for e in observed})

        by_relation = self._group_by_relation(observed)
        adjacency = self._adjacency(observed)

        out: list[Candidate] = []
        for relation in sorted(by_relation):
            out.extend(
                self._candidates_for_relation(
                    relation,
                    by_relation[relation],
                    node_types,
                    nodes,
                    adjacency,
                    observed_set,
                )
            )
        out.sort(key=lambda c: c.key)
        return out

    # -- per-relation candidate construction --------------------------------

    def _candidates_for_relation(
        self,
        relation: str,
        edges: list[ObservedEdge],
        node_types: dict[str, str],
        nodes: list[str],
        adjacency: dict[str, set[str]],
        observed_set: set[tuple[str, str, str]],
    ) -> list[Candidate]:
        head_types, tail_types = self._type_signature(edges, node_types)
        endpoints = {e.head for e in edges} | {e.tail for e in edges}

        # The tiered 2-hop entity pool for this relation (or the whole node set if not tiering /
        # the neighbourhood is empty). Deterministic (sorted).
        pool = self._two_hop_pool(endpoints, adjacency, nodes)

        heads = [n for n in pool if self._type_ok(node_types.get(n), head_types)]
        tails = [n for n in pool if self._type_ok(node_types.get(n), tail_types)]

        raw: list[Candidate] = []
        rare = len(edges) <= self._config.rare_relation_max_edges
        for h in heads:
            for t in tails:
                if h == t:
                    continue  # no self-loop
                if (h, relation, t) in observed_set:
                    continue  # already asserted — never propose an observed pair
                raw.append(Candidate(head=h, relation=relation, tail=t, rare=rare))

        raw.sort(key=lambda c: c.key)
        return self._bound(relation, raw, rare)

    def _bound(self, relation: str, raw: list[Candidate], rare: bool) -> list[Candidate]:
        cap = self._config.max_candidates_per_relation
        if rare or cap <= 0 or len(raw) <= cap:
            # Rare relations keep FULL recall; a dense relation under the cap is untouched.
            return raw
        dropped = len(raw) - cap
        self.capped[relation] = dropped
        _LOG.info(
            "candidates: relation %r is dense (%d type-valid candidates); capping to %d "
            "(dropping %d lowest-key candidates — logged, not silent).",
            relation,
            len(raw),
            cap,
            dropped,
        )
        # Deterministic selection: keep the first ``cap`` by sorted candidate key.
        return raw[:cap]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _group_by_relation(observed: list[ObservedEdge]) -> dict[str, list[ObservedEdge]]:
        by_relation: dict[str, list[ObservedEdge]] = {}
        for e in observed:
            by_relation.setdefault(e.relation, []).append(e)
        return by_relation

    @staticmethod
    def _adjacency(observed: list[ObservedEdge]) -> dict[str, set[str]]:
        """Undirected adjacency over ALL relations (for the 2-hop neighbourhood tiers)."""
        adj: dict[str, set[str]] = {}
        for e in observed:
            adj.setdefault(e.head, set()).add(e.tail)
            adj.setdefault(e.tail, set()).add(e.head)
        return adj

    def _two_hop_pool(
        self,
        endpoints: set[str],
        adjacency: dict[str, set[str]],
        nodes: list[str],
    ) -> list[str]:
        if not self._config.two_hop:
            return nodes
        reach: set[str] = set(endpoints)
        one_hop: set[str] = set()
        for n in endpoints:
            one_hop |= adjacency.get(n, set())
        reach |= one_hop
        for n in one_hop:
            reach |= adjacency.get(n, set())
        if not reach:
            return nodes  # isolated component — fall back to the full pool so it is completable
        return sorted(reach)

    @staticmethod
    def _type_signature(
        edges: list[ObservedEdge],
        node_types: dict[str, str],
    ) -> tuple[frozenset[str], frozenset[str]]:
        """The (head types, tail types) a relation is observed to connect — its type signature."""
        head_types = frozenset(
            node_types[e.head] for e in edges if e.head in node_types
        )
        tail_types = frozenset(
            node_types[e.tail] for e in edges if e.tail in node_types
        )
        return head_types, tail_types

    def _type_ok(self, node_type: str | None, allowed: frozenset[str]) -> bool:
        """Whether a node's type passes a relation's type filter.

        When type-awareness is off, or the relation has no observed type constraint (an untyped /
        freely-typed relation), or the node itself is untyped, the filter admits it — it never
        *invents* a constraint that would forfeit recall.
        """
        if not self._config.type_aware:
            return True
        if not allowed:
            return True  # the relation admits any type (no observed constraint)
        if node_type is None:
            return True  # an untyped node is not excluded by a typed relation
        return node_type in allowed

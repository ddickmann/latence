"""GraphCompletionEngine — the pure reference link-prediction engine (W2-linkpred Slice 1/2).

The deterministic, dependency-free engine behind the OPTIONAL ``graph_completion`` Stage: it
takes the assembled graph's observed edges (head node, label, tail node) and proposes
ADDITIONAL edges the graph does not yet assert, by a **2-hop transitive/symmetric closure** over
the observed relations. This is the honest *reference* completer that proves the seam +
inferred-edge contract end to end without torch — exactly how the GraphAssembler shipped a
reference builder before any learned Provider. The learned ULTRA/PyKEEN scorer is a drop-in
Provider behind the same seam (Slice 2), and this engine gives it a bake-off baseline to beat.

The rule (configurable per relation label):

* **Transitivity** — if ``A --p--> B`` and ``B --p--> C`` are both observed and ``p`` is declared
  transitive, propose ``A --p--> C`` (the 2-hop closure). ``partner_of`` / ``located_in`` /
  ``part_of`` are natural transitive relations.
* **Symmetry** — if ``A --p--> B`` is observed and ``p`` is declared symmetric, propose
  ``B --p--> A`` (the reverse edge). ``partner_of`` / ``sibling_of`` are natural symmetric
  relations.

Scoring is an explainable **path-support fraction**: a proposed edge's score is the number of
distinct 2-hop paths (or the single symmetric witness) that support it, divided by a
``support_norm`` (default: the max path-support seen this run), clamped to ``[0, 1]``. Only
proposals at or above ``min_score`` are kept. **Fan-out is bounded** (the ported relation
quadratic guard discipline): per source node, at most ``max_candidates_per_node`` proposals
survive (the highest-scoring, ties broken by deterministic edge key), so a hub node cannot
explode the candidate set. Every collection is sorted, so the engine is byte-identical across
runs over the same graph (Baseline bar).

The engine is pure over plain tuples (no Pydantic, no ids beyond the node ids it is handed): the
Provider maps its :class:`PredictedEdge`s onto the inferred-edge contract (``scope=EDGE``,
``properties["inferred"]=True``, honest model-derived Evidence). It never invents nodes — a
predicted edge's endpoints are always two distinct observed nodes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservedEdge:
    """One asserted edge the engine reasons over (head node, label, tail node)."""

    source_node_id: str
    label: str
    target_node_id: str


@dataclass(frozen=True)
class PredictedEdge:
    """One proposed (not-yet-asserted) edge, with its scorer provenance.

    ``score`` is the calibrated path-support fraction in ``[0, 1]``; ``support`` the raw witness
    count (distinct 2-hop paths, or 1 for a symmetric witness); ``rule`` names why it was
    proposed (``"transitive"`` / ``"symmetric"``); ``rank`` is its 0-based position in the
    deterministic global ordering (highest score first, ties by edge key). Endpoints are always
    two distinct observed nodes.
    """

    source_node_id: str
    label: str
    target_node_id: str
    score: float
    support: int
    rule: str
    rank: int = 0


@dataclass(frozen=True)
class CompletionConfig:
    """The reference completer's tuning knobs (all deterministic).

    ``transitive_labels`` / ``symmetric_labels`` declare which relation labels get the 2-hop
    transitive closure / the symmetric reverse. When BOTH sets are empty the engine treats
    **every** observed label as both transitive and symmetric (a sensible zero-config default so
    the seam is exercised out of the box); when either set is non-empty only the named labels get
    that rule. ``min_score`` is the keep threshold (a proposal below it is dropped).
    ``max_candidates_per_node`` bounds the fan-out per source node (0 = unlimited — but the global
    2-hop closure is inherently bounded by the observed edge count). ``support_norm`` fixes the
    score denominator (0 = normalise by the run's max path-support, so the strongest proposal
    scores 1.0).
    """

    transitive_labels: frozenset[str] = frozenset()
    symmetric_labels: frozenset[str] = frozenset()
    min_score: float = 0.0
    max_candidates_per_node: int = 32
    support_norm: float = 0.0


class GraphCompletionEngine:
    """Proposes missing edges by a deterministic, bounded 2-hop transitive/symmetric closure.

    Pure and deterministic: given the same observed edges + config it returns byte-identical
    :class:`PredictedEdge`s in the same order (score desc, then edge key). It never proposes an
    edge that is already observed (the asserted-collision drop happens here, so the Provider never
    emits a predicted edge that duplicates an asserted one), never a self-loop, and never an edge
    touching a node it was not handed.
    """

    def __init__(self, config: CompletionConfig | None = None) -> None:
        self._config = config or CompletionConfig()

    def _is_transitive(self, label: str) -> bool:
        cfg = self._config
        if not cfg.transitive_labels and not cfg.symmetric_labels:
            return True  # zero-config default: every label is transitive + symmetric.
        return label in cfg.transitive_labels

    def _is_symmetric(self, label: str) -> bool:
        cfg = self._config
        if not cfg.transitive_labels and not cfg.symmetric_labels:
            return True  # zero-config default: every label is transitive + symmetric.
        return label in cfg.symmetric_labels

    def predict(self, observed: list[ObservedEdge]) -> list[PredictedEdge]:
        """Return the deterministic, bounded, thresholded predicted edges for ``observed``.

        Algorithm (per relation label ``p``, so paths never cross relation types):

        1. Collect the observed ``(head, tail)`` pairs for ``p`` and the ``head -> {tail}`` map.
        2. **Transitivity** (if ``p`` is transitive): for every observed ``A --p--> B`` and
           every observed ``B --p--> C`` with ``A != C``, the pair ``(A, C)`` gains one path of
           support. A proposal already observed for ``p`` is dropped (already asserted).
        3. **Symmetry** (if ``p`` is symmetric): for every observed ``A --p--> B`` with
           ``B --p--> A`` NOT observed, ``(B, A)`` gains one witness of support.
        4. Score = ``support / support_norm`` (or the run-max path-support), clamped to ``[0, 1]``;
           drop below ``min_score``.
        5. Bound fan-out: keep at most ``max_candidates_per_node`` proposals per source node (top
           score, ties by deterministic edge key).
        6. Rank globally (score desc, then edge key) and stamp ``rank``.
        """
        observed_set = {(e.source_node_id, e.label, e.target_node_id) for e in observed}
        # Adjacency per label: label -> {head -> sorted[tail]} (sorted for determinism).
        by_label: dict[str, dict[str, list[str]]] = {}
        for edge in observed:
            adj = by_label.setdefault(edge.label, {})
            adj.setdefault(edge.source_node_id, [])
            if edge.target_node_id not in adj[edge.source_node_id]:
                adj[edge.source_node_id].append(edge.target_node_id)
        for adj in by_label.values():
            for tails in adj.values():
                tails.sort()

        # support[(source, label, target)] = raw witness count.
        support: dict[tuple[str, str, str], int] = {}

        for label in sorted(by_label):
            adj = by_label[label]
            if self._is_transitive(label):
                self._accumulate_transitive(label, adj, observed_set, support)
            if self._is_symmetric(label):
                self._accumulate_symmetric(label, adj, observed_set, support)

        if not support:
            return []

        norm = self._config.support_norm
        if norm <= 0.0:
            norm = float(max(support.values()))
        # Rule attribution: a proposal reachable transitively is a transitive proposal; else it is
        # symmetric (the only two rules). Recompute per key deterministically.
        proposals = self._score(support, norm, by_label, observed_set)
        proposals = self._bound_fan_out(proposals)
        proposals.sort(key=lambda p: (-p.score, p.source_node_id, p.label, p.target_node_id))
        return [
            PredictedEdge(
                source_node_id=p.source_node_id,
                label=p.label,
                target_node_id=p.target_node_id,
                score=p.score,
                support=p.support,
                rule=p.rule,
                rank=rank,
            )
            for rank, p in enumerate(proposals)
        ]

    @staticmethod
    def _accumulate_transitive(
        label: str,
        adj: dict[str, list[str]],
        observed_set: set[tuple[str, str, str]],
        support: dict[tuple[str, str, str], int],
    ) -> None:
        for source in sorted(adj):
            for mid in adj[source]:
                for target in adj.get(mid, []):
                    if target == source:
                        continue  # no self-loop
                    key = (source, label, target)
                    if key in observed_set:
                        continue  # already asserted — never re-propose
                    support[key] = support.get(key, 0) + 1

    @staticmethod
    def _accumulate_symmetric(
        label: str,
        adj: dict[str, list[str]],
        observed_set: set[tuple[str, str, str]],
        support: dict[tuple[str, str, str], int],
    ) -> None:
        for source in sorted(adj):
            for target in adj[source]:
                if target == source:
                    continue  # no self-loop
                reverse = (target, label, source)
                if reverse in observed_set:
                    continue  # the reverse is already asserted
                # A symmetric witness only *adds* support; it never overwrites a transitive count.
                support[reverse] = support.get(reverse, 0) + 1

    def _score(
        self,
        support: dict[tuple[str, str, str], int],
        norm: float,
        by_label: dict[str, dict[str, list[str]]],
        observed_set: set[tuple[str, str, str]],
    ) -> list[PredictedEdge]:
        scored: list[PredictedEdge] = []
        for key, raw in support.items():
            score = min(raw / norm, 1.0) if norm > 0 else 0.0
            if score < self._config.min_score:
                continue
            source, label, target = key
            rule = self._rule_for(source, label, target, by_label, observed_set)
            scored.append(
                PredictedEdge(
                    source_node_id=source,
                    label=label,
                    target_node_id=target,
                    score=round(score, 6),
                    support=raw,
                    rule=rule,
                )
            )
        return scored

    @staticmethod
    def _rule_for(
        source: str,
        label: str,
        target: str,
        by_label: dict[str, dict[str, list[str]]],
        observed_set: set[tuple[str, str, str]],
    ) -> str:
        """Attribute a proposal to ``transitive`` if a 2-hop path witnesses it, else ``symmetric``.

        A transitive witness is a middle node ``M`` with ``source --p--> M`` and
        ``M --p--> target`` both observed. If none exists the proposal came from the symmetric
        rule (the reverse of an observed ``target --p--> source``).
        """
        adj = by_label.get(label, {})
        for mid in adj.get(source, []):
            if target in adj.get(mid, []) and mid != source and mid != target:
                return "transitive"
        return "symmetric"

    def _bound_fan_out(self, proposals: list[PredictedEdge]) -> list[PredictedEdge]:
        """Keep at most ``max_candidates_per_node`` per source node (top score, det. ties).

        The bounded-fan-out guard: a hub source node with many 2-hop reachable targets cannot
        explode the candidate set. Selection is deterministic — sort each source's proposals by
        (score desc, edge key) and take the head slice.
        """
        cap = self._config.max_candidates_per_node
        if cap <= 0:
            return proposals
        by_source: dict[str, list[PredictedEdge]] = {}
        for p in proposals:
            by_source.setdefault(p.source_node_id, []).append(p)
        kept: list[PredictedEdge] = []
        for source in sorted(by_source):
            group = sorted(
                by_source[source],
                key=lambda p: (-p.score, p.label, p.target_node_id),
            )
            kept.extend(group[:cap])
        return kept

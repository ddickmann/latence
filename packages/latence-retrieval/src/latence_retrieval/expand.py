"""The ``expand`` seam: :class:`Expander` (a :class:`CandidateProcessor`), 1-hop graph-augmented.

``expand`` is the query-time transform that **graph-augments** an already-fetched candidate list
using the knowledge-graph columns the pipeline denormalised onto each chunk (CONTEXT ``Expand``;
ADR-0051's *default 1-hop path*). It is a
:class:`~latence_retrieval.processor.CandidateProcessor` — the single-list ``(query, candidates) →
candidates`` shape — not a new seam: a caller fetched the candidates from its engine, and the
expander only *rescores and reorders* them by graph proximity to the list's own top hits. Like every
query-time component it is strictly stateless (ADR-0048): a pure transform that holds no graph, runs
**no graph database**, fetches nothing, and never mutates its input
(:class:`~latence_retrieval.contracts.Candidate` is frozen).

**Why this needs no graph DB (ADR-0051).** The pipeline already projects each chunk's 1-hop KG
neighbourhood *onto the chunk itself* — its own entity node ids (``kg_node_ids``), the node ids its
triples reach (``neighbor_node_ids``), and the optional per-entity ``kg_node_community`` /
``kg_node_centrality`` graph features. Those travel with the chunk into any vector store as
metadata, so 1-hop expansion and graph-aware rescoring are **pure metadata operations on the
candidates already in hand** — they work identically on Qdrant, Weaviate, Azure AI Search, Elastic,
or a proprietary
store. Only *multi-hop* traversal needs the graph queryable (the deferred DuckDB / networkx / Neo4j
path); the default single hop does not.

**What it computes.** A small *seed set* — the top ``seed_top_k`` candidates by first-stage score —
is the graph anchor. Every candidate then earns a graph bonus for its 1-hop connectivity to those
seeds and its own structural importance, added to its first-stage score, after which the list is
re-sorted:

* **1-hop link** (``hop_weight``) — a candidate is a graph neighbour of a seed when a KG **edge**
  joins them: the candidate's own nodes reach into the seed's ``neighbor_node_ids`` (or the seed's
  reach into the candidate's nodes), *or* they **co-mention** a shared entity node. The bonus is
  ``hop_weight ×`` the number of distinct seeds it links to, so a candidate corroborated by several
  top hits rises most.
* **shared community** (``community_weight``, opt-in) — a candidate sharing a ``kg_node_community``
  id with any seed earns a flat bonus: the graph clustered them together even without a direct edge.
* **own centrality** (``centrality_weight``, opt-in) — a candidate whose entities are graph hubs
  (high ``kg_node_centrality``) earns ``centrality_weight × its peak centrality``.

Because the seeds sit at the top of a wider pool the caller fetched, their graph neighbours are
*lifted into* the returned window — the "neighbor-expanded" result — while the anchors themselves
keep their first-stage standing. Run ``expand`` after the first-stage search (and optionally after
``rerank``), before ``pack``.

**Graceful degradation (the AC).** Every graph signal is read from candidate ``metadata`` and simply
contributes nothing when absent, so the expander runs on candidates that carry *no* KG columns at
all (then it is an order-preserving no-op) and gets sharper as more columns are present — exactly
the optional-signal discipline ``pack`` uses. Where a KG column *is* present it must be well-formed
(a list of node-id strings; finite centrality floats), and a malformed column fails **loudly** at
the boundary rather than silently scoring a candidate wrong — the anti-false-green contract.

There is deliberately **no new model/store seam** here: 1-hop expand has no external dependency
(unlike the deferred multi-hop path, whose graph source *is* a real two-adapter seam), so it is a
pure-Python reference transform — another adapter of the existing :class:`CandidateProcessor` seam,
resolved by config through :func:`~latence_retrieval.registry.resolve_processor`
(``processor.expand``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_retrieval.kg_columns import KgColumnKeys, KgColumns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_retrieval.contracts import Candidate

_DEFAULT_SEED_TOP_K = 3
_DEFAULT_HOP_WEIGHT = 0.5


class Expander:
    """Graph-augment a candidate list by 1-hop KG proximity to its own top hits (ADR-0051/0048).

    A :class:`~latence_retrieval.processor.CandidateProcessor`: :meth:`process` adds a graph bonus
    to each candidate's first-stage ``score`` — for its 1-hop links to the seed set, optionally its
    shared-community and own-centrality signals (see the module docstring) — replaces the ``score``
    with that graph-augmented total, and returns a new list sorted by descending score (ties broken
    by ascending ``id`` so identical inputs always yield an identical order, the determinism the
    fuse / rerank / pack seams hold too). The input list is never mutated
    (:class:`~latence_retrieval.contracts.Candidate` is frozen; a fresh list is returned) and the
    expander keeps no state across calls — the same inputs always produce the same output.

    Config:
        seed_top_k: int — how many top-scored candidates form the graph anchor (the seed set the
            1-hop/community bonuses are measured against). **Positive int**, default ``3``. A larger
            seed set rewards corroboration by more of the head; ``1`` anchors strictly on the best
            hit.
        hop_weight: float — the bonus **per distinct seed** a candidate is 1-hop-linked to (a KG
            edge into/out of a seed, or a shared entity node). Default ``0.5``; must be ``>= 0``
            (``0`` disables edge expansion). This is the default graph signal — it needs only
            ``kg_node_ids`` and ``neighbor_node_ids``, present whenever context enrichment ran.
        community_weight: float — a flat bonus for sharing a ``kg_node_community`` id with any seed.
            Default ``0.0`` (opt-in; needs the optional graph-features projection). ``>= 0``.
        centrality_weight: float — coefficient on a candidate's **peak** ``kg_node_centrality`` (its
            most central entity). Default ``0.0`` (opt-in; needs graph features). Must be ``>= 0``.
        node_ids_key: str — metadata key holding the candidate's own entity node ids. Default
            ``"kg_node_ids"``.
        neighbor_ids_key: str — metadata key holding the node ids the candidate's triples reach (its
            1-hop expansion keys). Default ``"neighbor_node_ids"``.
        community_key: str — metadata key holding the candidate's entity community ids. Default
            ``"kg_node_community"``.
        centrality_key: str — metadata key holding the candidate's per-entity centrality floats.
            Default ``"kg_node_centrality"``.
        top_k: int | None — if set (a positive int), the augmented list is truncated to its top
            ``top_k`` (surface only the graph-lifted head). Default ``None`` keeps every candidate,
            reordered.

    A candidate lacking a given metadata key simply earns nothing from that signal; when *no*
    candidate carries any KG column the transform is an order-preserving no-op (up to the
    deterministic ``id`` tiebreak on exact score ties). A *present* column must be well-formed or
    :meth:`process` raises :class:`ValueError` / :class:`TypeError` at the boundary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self._seed_top_k = self._positive_int(
            cfg.get("seed_top_k", _DEFAULT_SEED_TOP_K), "seed_top_k"
        )

        self._hop_weight = self._non_negative(
            cfg.get("hop_weight", _DEFAULT_HOP_WEIGHT), "hop_weight"
        )
        self._community_weight = self._non_negative(
            cfg.get("community_weight", 0.0), "community_weight"
        )
        self._centrality_weight = self._non_negative(
            cfg.get("centrality_weight", 0.0), "centrality_weight"
        )

        self._keys = KgColumnKeys(
            node_ids=self._non_empty_str(
                cfg.get("node_ids_key", "kg_node_ids"), "node_ids_key"
            ),
            neighbor_ids=self._non_empty_str(
                cfg.get("neighbor_ids_key", "neighbor_node_ids"), "neighbor_ids_key"
            ),
            community=self._non_empty_str(
                cfg.get("community_key", "kg_node_community"), "community_key"
            ),
            centrality=self._non_empty_str(
                cfg.get("centrality_key", "kg_node_centrality"), "centrality_key"
            ),
        )

        top_k = cfg.get("top_k")
        if top_k is not None:
            top_k = self._positive_int(top_k, "top_k")
        self._top_k: int | None = top_k

    # --- construction-time config parsing --------------------------------------------------------

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            msg = f"expand {name!r} must be a positive integer, got {value!r}."
            raise ValueError(msg)
        return value

    @staticmethod
    def _non_negative(value: object, name: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            msg = f"expand {name!r} must be a number >= 0, got {value!r}."
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def _non_empty_str(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            msg = f"expand {name!r} must be a non-empty string metadata key, got {value!r}."
            raise ValueError(msg)
        return value

    # --- the transform ---------------------------------------------------------------------------

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Graph-augment ``candidates``' scores by 1-hop proximity to the seed set, then reorder.

        Reads each candidate's denormalised KG columns, adds the configured graph bonuses (1-hop
        links to the top-``seed_top_k`` seeds, plus optional community/centrality), replaces each
        ``score`` with the augmented total, and returns a new list in descending augmented score
        (ascending-``id`` tiebreak), optionally truncated to ``top_k``. The input is never mutated;
        ``query`` is part of the seam signature but the expander augments the first-stage score with
        the *graph* signal — it does not re-score against the query text (that is ``rerank``'s job).
        An empty input returns ``[]``.
        """
        items = list(candidates)
        if not items:
            return []

        # Parse-once per candidate: the whole KG-column read contract lives in `kg_columns`
        # (graceful-absent / loud-malformed), this transform only *scores* off the parsed view.
        columns = [
            KgColumns.parse(candidate, keys=self._keys, context="expand") for candidate in items
        ]

        seed_indices = self._seed_indices(items)
        seed_communities: set[str] = set()
        for s in seed_indices:
            seed_communities |= columns[s].communities

        augmented: list[Candidate] = []
        for i, candidate in enumerate(items):
            bonus = 0.0
            if self._hop_weight:
                links = sum(
                    1 for s in seed_indices if s != i and self._linked(columns[i], columns[s])
                )
                bonus += self._hop_weight * links
            if self._community_weight and columns[i].communities & seed_communities:
                bonus += self._community_weight
            if self._centrality_weight:
                bonus += self._centrality_weight * columns[i].peak_centrality
            augmented.append(candidate.model_copy(update={"score": candidate.score + bonus}))

        augmented.sort(key=lambda c: (-c.score, c.id))
        if self._top_k is not None:
            return augmented[: self._top_k]
        return augmented

    def _seed_indices(self, items: Sequence[Candidate]) -> list[int]:
        """Indices of the top-``seed_top_k`` candidates by first-stage score (descending, id tie).

        Chosen deterministically off the *original* scores (before any bonus) so the anchor is
        stable: identical inputs always select the same seeds regardless of input order among score
        ties.
        """
        order = sorted(range(len(items)), key=lambda i: (-items[i].score, items[i].id))
        return order[: self._seed_top_k]

    @staticmethod
    def _linked(a: KgColumns, b: KgColumns) -> bool:
        """True when candidate ``a`` and seed ``b`` are 1-hop connected in the KG.

        A connection is a KG **edge** (``a``'s own nodes fall in ``b``'s reach set, or ``b``'s
        nodes in ``a``'s reach) or a **shared entity** node (co-mention). Both directions are
        checked because the denormalised ``neighbor_node_ids`` records where a chunk's triples
        point, which need not be symmetric across two independently-chunked hits.
        """
        return bool((a.nodes & b.reach) or (b.nodes & a.reach) or (a.nodes & b.nodes))


__all__ = ["Expander"]

# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Chain-of-evidence selection (M4: spec 3.1–3.3 over M3's signals, ADR-0058 verified budget).

The selection half of chain-of-evidence retrieval: given the
:class:`~latence_retrieval.hypergraph.HypergraphSignals` M3 emitted for one query (ρ over the
GATED candidate set — post ``s_hyp`` prune + vertex/coverage gate, the ticket-20 verified order;
selection over a raw expansion is the measured failure mode), :func:`select` returns the
token-budgeted subset of Hyperedges that maximises query-conditional *marginal* utility
``v(e | A)`` — because plain additive knapsack values cannot express redundancy or chain
formation. Like every query-time component it is stateless (ADR-0048) and **zero-dep**: pure
Python over the pydantic-only package, nothing held between calls.

**The utility seam** (spec 3.5). :class:`UtilityModel` is the one interface the future learned
token-level classifier replaces: ``base`` is the context-free value v0, ``marginal`` the
selected-set-conditional v(e|A). :class:`HeuristicUtility` is the day-one implementation of spec
3.2: a weighted sum of propagated confidence (ρ, max-normalised), EU-text/query similarity,
extraction confidence γ, anchor overlap, a log-damped frequency prior over ``support``, and a
caller-supplied per-Hyperedge relation-type bonus (the w_relq signal — GLiNER2 zero-shot relation
typing on the query, run by the CALLER; this module never assumes the signal exists, ADR-0059's
tiering discipline). An absent signal is never silently zero-scored against the others: its
weight **redistributes** — v0 is the weighted mean over the *active* signals, so a tier-0 caller
with no ANN sims and no query-relation model gets honestly comparable values, not deflated ones.
Weak Hyperedges (``has_relations=False``, ADR-0057) carry the spec's ×0.6 discount. One deliberate
deviation from spec 3.5's ``(query_feats, edge_feats)`` signature: every feature here is already
query-conditional (ρ, sim, anchor overlap are computed for THIS query), so the query folds into
:class:`EdgeFeatures` rather than travelling as a separate argument.

**The solver** (spec 3.3). Budgeted lazy-greedy over marginal density ``v(e|A)/token_cost`` with
a ``theta_min = theta_ratio · max(v0)`` filler stop — the budget is never spent just because it
exists. Redundancy and connectivity are **incremental sparse** state, never a dense ``|E_c|²``
matrix: ticket 20 measured the dense overlap matrix as THE latency cap (1.6 GB at 20k candidates)
and its replacement — a running max-shared-entities vector updated by one sparse pass over an
inverted node→candidate index per acceptance — as a ~4× win. Connectivity is the one signal that
can *grow* as A grows (it breaks strict submodularity), so the lazy heap seeds
optimistic-connectivity keys, re-evaluates the true marginal density on every pop, and accepts
only against the true value; ties break by ascending Hyperedge id (like
:class:`~latence_retrieval.pack.Packer`), so identical inputs always yield the identical
selection.

**Spec 3.2's relation-signature hard skip is a DECLARED DEVIATION, approximated by the entity
overlap discount** (corrected 2026-08-02 by the close-out audit — the earlier claim that the
case was "unrepresentable by construction" was wrong). What dedup does guarantee: identical
(entity-set, signature) pairs collapse into one record (same content ⇒ same id, ``support += 1``).
What it does NOT prevent: a record whose signature is a strict SUBSET of another's, or an
identical signature over a different entity set — both trivially coexist because the id is
addressed over entity set + signature jointly, and both are pinned by test as representable.
The selector cannot evaluate the signature-subset condition at all: the KV solver fields
(ADR-0056's deliberate minimum) carry no relation signatures. The shipped handling is the
entity-overlap redundancy discount — for a subset-signature candidate the shared entities drive
``red`` toward 1.0, collapsing its marginal to the connectivity bonus alone. No entity-level
red=1.0 hard skip is added on top: full entity containment does not imply fact redundancy (the
same entity pair can carry a distinct predicate), so skipping there would sacrifice recall
without data — the standing rule forbids it. If judged data ever shows subset-signature
redundancy materially spending budget, the measured fix is signatures in the solver fields (a
KV format rev), not a guess here.

**The budget guarantee is ADR-0058's exact verification pass, not the stored weights.** Stored
``token_cost`` is an approximate index-time weight carrying its counter's identity; selection
ranks and packs on it, and then — when the caller supplies ``tau_lookup`` + a
:class:`~latence_retrieval.pack.TokenCounter` + that counter's identity — the SELECTED set only
(15–40 items, ~2–3 ms) is re-counted exactly and the tail trimmed lowest-marginal-first if the
true total overflows. The pass is **mandatory** whenever any selected record's stored
``token_counter`` differs from the verification counter's identity, optional
(:attr:`CoeConfig.verify_on_match`) when they match, and NEVER replaced by a cross-family scale
factor — a scale factor converts a detectable error into a plausible-looking wrong number. The
result says what happened (``verified``/``trimmed``): honest observability, never a silent repair.

**``knapsack_refine`` is deliberately dropped** (spec 3.3's final phase; the ticket-01 question).
Recorded rationale: (1) the existing :class:`~latence_retrieval.pack.Packer` is the same greedy
marginal-density family, not DP/ILP, so composing it as a "boundary repair" phase re-runs the
same decision rule on frozen values — a structural no-op; (2) a real DP over frozen v(e|A_final)
would optimise a linear proxy that ignores exactly the interaction terms (redundancy,
connectivity) that motivated the marginal formulation, so a frozen-value "improvement" can lower
the true objective — an unverifiable win; (3) ``theta_min`` deliberately leaves budget unspent,
and a repair phase that back-fills the boundary with sub-threshold filler contradicts it; (4) the
ADR-0058 verification pass may trim the tail anyway, re-breaking any repaired boundary. The
measured levers on selection quality were pipeline order and candidate reach (ticket 20), never
boundary packing.

**The selection trace** (spec 3.5, M5). With an optional ``trace_sink`` supplied, :func:`select`
emits ONE JSON-serialisable dict per query — anchors, expansion/prune counts, a per-candidate
record (v0, marginal, accepted/rejected, trimmed), the final selection and the verification
outcome — the weak-supervision seed for the future learned utility classifier. A rejected
candidate's marginal is evaluated against the FINAL selected set (the counterfactual a learned
utility wants to score); an accepted one's is its value at acceptance, exactly what drove the
decision. Traces carry **ids, counts and scores only — never raw ``tau_text``**: a deliberate
privacy choice, because a trace may travel further than the corpus it describes. The default
``trace_sink=None`` path is byte-identical to the pre-hook selector.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from latence_retrieval.hypergraph import HypergraphSignals
    from latence_retrieval.pack import TokenCounter

__all__ = [
    "CoeConfig",
    "CoeSelection",
    "EdgeFeatures",
    "HeuristicUtility",
    "SelectionState",
    "UtilityModel",
    "select",
]


@dataclass(frozen=True)
class CoeConfig:
    """Knobs for the selector, defaulting to spec 3.2/3.3's published operating point.

    The six v0 weights are spec 3.2's starting values (they sum to 1.0, so the weighted mean
    over all-active signals equals the spec's plain weighted sum); ``w_conn`` and the
    ``theta_ratio``/``weak_discount`` constants are spec 3.2/3.3's. ``verify_on_match`` governs
    the ADR-0058 pass ONLY when the stored and verification counter identities agree — on a
    mismatch the pass is mandatory and no configuration can switch it off.
    """

    w_prop: float = 0.35
    w_rel: float = 0.20
    w_ext: float = 0.15
    w_ent: float = 0.15
    w_sup: float = 0.10
    w_relq: float = 0.05
    w_conn: float = 0.15
    theta_ratio: float = 0.15
    weak_discount: float = 0.6
    verify_on_match: bool = False

    def __post_init__(self) -> None:
        weights = {
            "w_prop": self.w_prop,
            "w_rel": self.w_rel,
            "w_ext": self.w_ext,
            "w_ent": self.w_ent,
            "w_sup": self.w_sup,
            "w_relq": self.w_relq,
            "w_conn": self.w_conn,
            "theta_ratio": self.theta_ratio,
        }
        for name, value in weights.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                msg = f"coe {name!r} must be a number >= 0, got {value!r}."
                raise ValueError(msg)
        if (
            not isinstance(self.weak_discount, (int, float))
            or isinstance(self.weak_discount, bool)
            or not 0.0 <= self.weak_discount <= 1.0
        ):
            msg = f"coe 'weak_discount' must be a number in [0, 1], got {self.weak_discount!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class EdgeFeatures:
    """One candidate Hyperedge's query-conditional features, as the utility model sees them.

    ``rho_norm``/``anchor_overlap``/``support_norm`` are already normalised over THIS query's
    candidate set (ρ by the set max, support log-damped by the set max, anchor overlap the mean
    restart weight over the entity set — the tiered anchors generalise spec 3.2's binary V_llm
    indicator: an alias anchor counts 1.0, an ANN anchor its α, ADR-0059). ``sim`` and
    ``relation_bonus`` are ``None`` when the caller supplied no such signal AT ALL for this call —
    the utility redistributes their weight — and a plain float (0.0 for an edge the caller's dict
    simply misses) when the signal exists.
    """

    rho_norm: float
    gamma: float
    anchor_overlap: float
    support_norm: float
    sim: float | None
    relation_bonus: float | None
    has_relations: bool


@dataclass(frozen=True)
class SelectionState:
    """The selected-set-conditional state for one candidate: what A currently does to it.

    ``redundancy`` is spec 3.2's entity-overlap ratio ``max_{a∈A} |entities(e) ∩ entities(a)| /
    |entities(e)|`` — maintained incrementally by :func:`select`, never via a dense matrix.
    ``connected`` is the chain-formation indicator: e shares ≥ 1 entity with A ∪ V0 (the anchors
    count from the start, so the chain starts anchored at the query).
    """

    redundancy: float
    connected: bool


@runtime_checkable
class UtilityModel(Protocol):
    """The one seam the future learned utility classifier replaces (spec 3.5).

    ``base`` is the context-free value v0(e); ``marginal`` the selected-set-conditional value
    v(e|A) given the incremental :class:`SelectionState`. The heuristic below is the day-one
    implementation; a learned model drops in behind this Protocol with no pipeline change.
    """

    def base(self, features: EdgeFeatures) -> float:
        """Context-free value v0 of a candidate with ``features`` (higher is better)."""
        ...

    def marginal(self, features: EdgeFeatures, state: SelectionState) -> float:
        """Marginal value v(e|A) of the candidate given the current selection ``state``."""
        ...


class HeuristicUtility:
    """Spec 3.2's heuristic v0 + marginal, with absent-signal weight redistribution.

    v0 is the weighted MEAN over the active signals — dividing by the active weight sum is what
    makes "absent signal ⇒ weight redistributes" true: with every signal active the divisor is
    1.0 (the six weights sum to 1) and the value equals the spec's plain weighted sum; with
    ``sim``/``relation_bonus`` absent (``None``) the remaining weights renormalise, so values
    stay on one comparable scale across signal tiers instead of silently deflating (ADR-0059:
    never assume a signal exists). Weak Hyperedges are discounted by ``weak_discount`` (spec
    1.2/3.2). The marginal is ``v0·(1 − redundancy) + w_conn·connected`` (spec 3.2).
    """

    def __init__(self, config: CoeConfig | None = None) -> None:
        self._config = config or CoeConfig()

    def base(self, features: EdgeFeatures) -> float:
        """v0: the redistribution-normalised weighted sum of the active signals."""
        cfg = self._config
        weighted: list[tuple[float, float]] = [
            (cfg.w_prop, features.rho_norm),
            (cfg.w_ext, features.gamma),
            (cfg.w_ent, features.anchor_overlap),
            (cfg.w_sup, features.support_norm),
        ]
        if features.sim is not None:
            weighted.append((cfg.w_rel, features.sim))
        if features.relation_bonus is not None:
            weighted.append((cfg.w_relq, features.relation_bonus))
        active = sum(weight for weight, _ in weighted)
        value = sum(weight * signal for weight, signal in weighted) / active if active else 0.0
        return value * self._config.weak_discount if not features.has_relations else value

    def marginal(self, features: EdgeFeatures, state: SelectionState) -> float:
        """v(e|A) = v0·(1 − red) + w_conn·conn — redundancy-discounted, chain-bonused."""
        connectivity = self._config.w_conn if state.connected else 0.0
        return self.base(features) * (1.0 - state.redundancy) + connectivity


@dataclass(frozen=True)
class CoeSelection:
    """What :func:`select` chose, and what the ADR-0058 pass did about it — honestly.

    ``selected`` is in acceptance order (the chain-assembly input M5 wants), with any trimmed
    ids already removed. ``marginals`` records v(e|A) at acceptance for EVERY accepted id —
    including later-trimmed ones — the per-step selection trace spec 3.5 asks to log.
    ``total_cost`` is the cost basis that actually guarantees the budget: the exact re-counted
    total when ``verified`` is True, the stored index-time total otherwise; it never exceeds
    ``budget``. ``verified`` means an exact re-count pass ran over the emitted set; ``trimmed``
    lists what that pass removed (lowest-marginal-first) — never silently.
    """

    selected: list[str]
    marginals: dict[str, float]
    total_cost: int
    budget: int
    verified: bool
    trimmed: list[str]


def select(
    signals: HypergraphSignals,
    *,
    budget: int,
    eu_sims: dict[str, float] | None = None,
    relation_bonus: dict[str, float] | None = None,
    utility: UtilityModel | None = None,
    config: CoeConfig | None = None,
    tau_lookup: Callable[[str], str | None] | None = None,
    token_counter: TokenCounter | None = None,
    counter_identity: str | None = None,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> CoeSelection:
    """Select the budgeted chain-of-evidence subset of ``signals``' gated candidates.

    ``signals`` is M3's :func:`~latence_retrieval.hypergraph.hypergraph_signals` result; every id
    in ``signals.rho`` is a candidate and its solver fields are read from ``signals.fields``.
    ``eu_sims`` (Hyperedge id → query similarity from the caller's eu_vecs ANN) and
    ``relation_bonus`` (Hyperedge id → query-relation-type match confidence, the w_relq signal)
    are OPTIONAL caller signals: pass ``None`` and the corresponding weight redistributes —
    a documented tier degradation, never a silent one (ADR-0059).

    The ADR-0058 verification pass runs over the selected set when the exact-counting triple is
    supplied: ``tau_lookup`` (typically an open store's
    :meth:`~latence_retrieval.hypergraph.HyperedgeKV.tau_text`), a
    :class:`~latence_retrieval.pack.TokenCounter` for the GENERATION tokenizer, and
    ``counter_identity`` naming it (the string compared against each record's stored
    ``token_counter``). Supply all three or none — a partial triple is a loud config error. On
    any identity mismatch the pass is mandatory; on a full match it runs only when
    :attr:`CoeConfig.verify_on_match` says so. The pass can only ever SHRINK the selection
    (under-packing is safe, overflow is not — ADR-0058); a selected id whose ``tau_text`` cannot
    be fetched fails loudly rather than shipping unverified.

    ``trace_sink`` (spec 3.5, M5) receives ONE JSON-serialisable dict for this query — anchors,
    expansion/prune counts, one accepted/rejected record per candidate with its v0 and marginal,
    the final selection ids and the verification outcome. Ids/counts/scores only, never raw
    ``tau_text`` (the module docstring records why); ``None`` (the default) changes nothing.

    Deterministic by construction: no randomness, ties broken by ascending Hyperedge id.
    """
    cfg = config or CoeConfig()
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        msg = f"coe 'budget' must be a positive integer (a token budget), got {budget!r}."
        raise ValueError(msg)
    verify_triple = (tau_lookup, token_counter, counter_identity)
    if any(part is not None for part in verify_triple) and not all(
        part is not None for part in verify_triple
    ):
        msg = (
            "coe verification needs all three of 'tau_lookup', 'token_counter' and "
            "'counter_identity' (ADR-0058: an exact pass is a text lookup + a counter + that "
            "counter's identity), or none of them."
        )
        raise ValueError(msg)

    ids = sorted(signals.rho)
    if not ids:
        if trace_sink is not None:
            trace_sink(
                {
                    "stage": "select",
                    "anchors": {node: signals.anchors[node] for node in sorted(signals.anchors)},
                    "expanded_candidates": signals.expanded_candidates,
                    "pruned_candidates": signals.pruned_candidates,
                    "candidate_count": 0,
                    "budget": budget,
                    "steps": [],
                    "selected": [],
                    "trimmed": [],
                    "verified": False,
                    "total_cost": 0,
                }
            )
        return CoeSelection(
            selected=[], marginals={}, total_cost=0, budget=budget, verified=False, trimmed=[]
        )
    model: UtilityModel = utility if utility is not None else HeuristicUtility(cfg)

    # -- per-candidate features (normalised over THIS query's candidate set) --------
    anchors = signals.anchors
    nodes_of: list[list[str]] = []
    costs: list[int] = []
    supports: list[int] = []
    for hid in ids:
        record = signals.fields[hid]
        nodes_of.append([str(node) for node in record["entity_node_ids"]])
        costs.append(max(int(record["token_cost"]), 1))
        supports.append(max(int(record.get("support", 1)), 0))
    rho_max = max(signals.rho.values())
    support_den = math.log1p(max(supports))
    features: list[EdgeFeatures] = []
    for position, hid in enumerate(ids):
        record = signals.fields[hid]
        nodes = nodes_of[position]
        overlap = sum(anchors.get(node, 0.0) for node in nodes) / len(nodes) if nodes else 0.0
        features.append(
            EdgeFeatures(
                rho_norm=signals.rho[hid] / rho_max if rho_max > 0.0 else 0.0,
                gamma=float(record["gamma"]),
                anchor_overlap=overlap,
                support_norm=(
                    math.log1p(supports[position]) / support_den if support_den > 0.0 else 0.0
                ),
                sim=float(eu_sims.get(hid, 0.0)) if eu_sims is not None else None,
                relation_bonus=(
                    float(relation_bonus.get(hid, 0.0)) if relation_bonus is not None else None
                ),
                has_relations=bool(record.get("has_relations", True)),
            )
        )
    theta_min = cfg.theta_ratio * max(model.base(feature) for feature in features)

    # -- incremental sparse overlap state (ticket 20: NEVER a dense |E_c|² matrix) --
    index_of = {hid: position for position, hid in enumerate(ids)}
    node_to: dict[str, list[int]] = {}
    for position, nodes in enumerate(nodes_of):
        for node in set(nodes):
            node_to.setdefault(node, []).append(position)
    max_shared = [0] * len(ids)
    connected = [any(node in anchors for node in nodes) for nodes in nodes_of]
    picked = [False] * len(ids)

    def redundancy(position: int) -> float:
        arity = len(nodes_of[position])
        return max_shared[position] / arity if arity else 0.0

    def true_marginal(position: int) -> float:
        return model.marginal(
            features[position],
            SelectionState(redundancy=redundancy(position), connected=connected[position]),
        )

    # -- the lazy-greedy marginal loop (spec 3.3), density-ordered ------------------
    # Initial keys are an optimistic-connectivity bound (connectivity is the one signal that
    # can GROW as A grows; redundancy only shrinks value). On pop the TRUE marginal density is
    # recomputed and the entry re-queued whenever a fresher candidate now ranks higher — the
    # lazy evaluation. A re-queued key is exact under the current A; a later acceptance can
    # raise it again (a conn flip), which costs at most greedy-ordering quality, never the
    # budget or termination (keys are stable while A is stable, so every pop makes progress).
    heap: list[tuple[float, str]] = [
        (
            -model.marginal(
                features[position],
                SelectionState(redundancy=0.0, connected=True),
            )
            / costs[position],
            hid,
        )
        for position, hid in enumerate(ids)
    ]
    heapq.heapify(heap)
    accepted: list[str] = []
    marginals: dict[str, float] = {}
    used = 0
    while used < budget and heap:
        _cached, hid = heapq.heappop(heap)
        position = index_of[hid]
        value = true_marginal(position)
        density = value / costs[position]
        if heap and -heap[0][0] > density:
            heapq.heappush(heap, (-density, hid))  # stale: re-queue exact, try the next best
            continue
        if used + costs[position] > budget or value <= theta_min:
            continue  # spec 3.3: does not fit, or filler below theta_min — dropped for good
        picked[position] = True
        accepted.append(hid)
        marginals[hid] = value
        used += costs[position]
        # one sparse pass per acceptance: bump the running max-shared vector + connectivity
        shared: dict[int, int] = {}
        for node in set(nodes_of[position]):
            for other in node_to[node]:
                if other != position and not picked[other]:
                    shared[other] = shared.get(other, 0) + 1
        for other, count in shared.items():
            if count > max_shared[other]:
                max_shared[other] = count
            connected[other] = True

    # -- ADR-0058: the exact query-time verification pass over the SELECTED set -----
    selected = list(accepted)
    trimmed: list[str] = []
    verified = False
    total = sum(costs[index_of[hid]] for hid in selected)
    if token_counter is not None and tau_lookup is not None:
        mismatch = any(
            signals.fields[hid].get("token_counter") != counter_identity for hid in selected
        )
        if mismatch or cfg.verify_on_match:
            exact: dict[str, int] = {}
            for hid in selected:
                text = tau_lookup(hid)
                if text is None:
                    msg = (
                        f"coe verification could not fetch tau_text for selected hyperedge "
                        f"{hid!r}: an unverifiable selection fails loudly, it never ships on "
                        "trust (ADR-0058)."
                    )
                    raise ValueError(msg)
                exact[hid] = max(token_counter.count(text), 1)
            total = sum(exact.values())
            acceptance_rank = {hid: rank for rank, hid in enumerate(accepted)}
            by_weakness = sorted(
                selected, key=lambda hid: (marginals[hid], -acceptance_rank[hid])
            )
            while total > budget and by_weakness:
                victim = by_weakness.pop(0)  # lowest marginal first; latest-accepted on ties
                selected.remove(victim)
                trimmed.append(victim)
                total -= exact[victim]
            verified = True

    if trace_sink is not None:
        # spec 3.5: one trace per query — the weak-supervision record for the learned utility.
        # Accepted steps first, in acceptance order, with the marginal that drove the accept;
        # then every never-accepted candidate exactly once as rejected, its marginal evaluated
        # against the FINAL selected state. Ids/counts/scores only — never raw tau_text.
        accepted_set = set(accepted)
        steps: list[dict[str, Any]] = [
            {
                "hyperedge_id": hid,
                "v0": model.base(features[index_of[hid]]),
                "marginal": marginals[hid],
                "decision": "accepted",
                "trimmed": hid in trimmed,
            }
            for hid in accepted
        ]
        steps.extend(
            {
                "hyperedge_id": hid,
                "v0": model.base(features[index_of[hid]]),
                "marginal": true_marginal(index_of[hid]),
                "decision": "rejected",
                "trimmed": False,
            }
            for hid in ids
            if hid not in accepted_set
        )
        trace_sink(
            {
                "stage": "select",
                "anchors": {node: signals.anchors[node] for node in sorted(signals.anchors)},
                "expanded_candidates": signals.expanded_candidates,
                "pruned_candidates": signals.pruned_candidates,
                "candidate_count": len(ids),
                "budget": budget,
                "steps": steps,
                "selected": list(selected),
                "trimmed": list(trimmed),
                "verified": verified,
                "total_cost": total,
            }
        )

    return CoeSelection(
        selected=selected,
        marginals=marginals,
        total_cost=total,
        budget=budget,
        verified=verified,
        trimmed=trimmed,
    )

# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
#
# This file carries TWO licence claims. They are independent and must not be collapsed: one is
# about where the code came FROM, the other about what you may do WITH it.
#
# OUTBOUND GRANT (what you may do with it) -- DELIBERATE, BOUNDED EXCEPTION, not a header nobody
# updated at the relicence. This repository's outbound licence is PolyForm-Noncommercial-1.0.0
# (root LICENSE); this module is one of seven the copyright holder keeps under Apache-2.0 ON
# PURPOSE, so the query-time retrieval primitives stay permissively reusable -- commercially
# included -- by anyone who wants only them. The exact set is authoritative in LICENSE, section
# "SCOPE: THE RETAINED APACHE-2.0 MODULES", and pinned by
# packages/latence-core/tests/test_publish_readiness.py, which fails the build if an eighth file
# acquires the grant or one of the seven loses it.
#
# INBOUND PROVENANCE (where it came from) -- the record that the port was clean to take, which is
# true regardless of what this file is licensed OUT under. The Knapsack QKP context-packer ported
# here derives from the maintainer's own `colsearch` project, relicensed Apache-2.0 for this use
# (provenance: docs/retrieval-tooling.md). It is a deterministic, CPU-only, pure-Python greedy
# solver for the Quadratic Knapsack Problem (ADR-0049); no third-party or non-permissive code is
# embedded.
"""The ``pack`` seam: :class:`Packer` (a :class:`CandidateProcessor`) over a :class:`TokenCounter`.

``pack`` is the query-time transform that selects a **token-budgeted subset** of an already-fetched
candidate list — the final assembly before an LLM's context window (CONTEXT ``Pack``). It maximises
a coverage/relevance objective **minus** a redundancy penalty under a token budget: the ported
Knapsack QKP context-packer. Like every query-time component it is stateless (ADR-0048) — a pure
``(query, candidates) → candidates`` transform that holds nothing, fetches nothing, never mutates
its input (:class:`~latence_retrieval.contracts.Candidate` is frozen). It is a
:class:`~latence_retrieval.processor.CandidateProcessor`, not a new shape: a caller fetched the
candidates from its engine, and the packer only *selects* a subset of them.

**What varies is how a candidate's token cost is measured**, so that is the seam: a
:class:`TokenCounter` is a genuine two-adapter seam (the codebase-design bar).
:class:`HeuristicTokenCounter` is the pure-Python, dependency-free reference that ships and runs on
**day one** (no config, no download); :class:`TiktokenCounter` is the real adapter (a
license-verified ``tiktoken`` encoding behind this package's ``[pack]`` extra, lazy, imported only
on first use). The :class:`Packer` **accepts** a :class:`TokenCounter` (dependency injection) or
builds the configured one lazily — a test hands it a spy/heuristic and production hands it the real
tokenizer, so the seam never has a single implementation.

**The objective.** For a chosen subset ``S`` the packer maximises

    Σ_{i∈S} value(i)  −  redundancy_weight · Σ_{i<j∈S} similarity(i, j)

subject to ``Σ_{i∈S} tokens(i) ≤ budget``. ``value(i)`` is the candidate's first-stage relevance
(``Candidate.score``, scaled to ``[0, 1]`` by the set maximum so the redundancy penalty is
commensurate) plus any optional numeric richness signals; ``similarity(i, j) ∈ [0, 1]`` is the
redundancy between two candidates. This is the (NP-hard) Quadratic Knapsack Problem; the ported
solver is a **deterministic greedy** marginal-gain approximation (pick the unselected candidate with
the highest positive marginal-value-per-token that still fits, ties broken by ascending ``id``),
which is CPU-only and reproducible by default (ADR-0049).

**Relevance is not recomputed against the query.** ``value`` starts from the caller's first-stage
``score`` (already query-conditioned); re-scoring a candidate against the query is ``rerank``'s job,
not ``pack``'s — so ``query`` is part of the seam signature but the packer selects, it does not
re-rank. Run ``rerank`` before ``pack`` if you want a stronger relevance signal.

**Optional signals degrade gracefully** (the AC): every richness signal is read from candidate
``metadata`` and simply contributes nothing when absent, so the packer runs on defaults with no
signals at all and gets better as more are supplied. Two general mechanisms cover them:

* **additive value signals** (``value_signals``: ``metadata-key → weight``) — each present numeric
  field (e.g. ``recency``, a numeric role priority) adds ``weight · value`` to that candidate's
  value; a **per-entity list** (the framework's own ``kg_node_centrality`` graph-feature shape) is
  first reduced to its **peak**, identically to how ``expand`` reads the same column, so the
  documented graph-augmented-pack wiring is alive on the framework's own export; an absent key
  contributes nothing.
* **redundancy signals** — how similar two candidates are. Default: lexical token-overlap (Jaccard)
  over ``text`` (runs day one). ``cluster_key``: candidates sharing that metadata value are treated
  as maximally redundant (``clusters``). ``similarity_key``: a caller-supplied per-candidate
  ``{other_id: sim}`` map is used directly (``similarity`` / precomputed embedding redundancy).

Licence discipline (ADR-0012): **no tokenizer encoding is pinned.** The default is the dependency-
free heuristic; the real :class:`TiktokenCounter` requires an explicit ``encoding`` name, so a
tokenizer is only ever run when the operator names one. ``tiktoken`` itself is MIT (recorded in the
package README).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_retrieval.contracts import Candidate
from latence_retrieval.errors import PackTokenizerError

if TYPE_CHECKING:
    from collections.abc import Sequence

# Unicode alphanumeric runs (underscore excluded, so it stays a boundary) — the same term unit the
# reference tokenizers emit (`latence_core.bm25`, `latence_core.embedding`, `latence_retrieval.bm25`
# — R5-wizard_profile-13). An ASCII-only pattern tokenised every non-Latin text to the empty set,
# which made the default lexical-Jaccard redundancy penalty below a permanent no-op for exactly the
# multilingual corpora the blessed `enterprise-sota` profile targets (R6-cross_lane_integration-13).
# ASCII-identical to the previous `[a-z0-9]+` (the text is lowercased first).
_TOKEN = re.compile(r"[^\W_]+")
# The heuristic token proxy: word pieces AND standalone punctuation, so a comma/period costs a
# token as it roughly does under a real BPE tokenizer (a deterministic over-word estimate).
_HEURISTIC_TOKEN = re.compile(r"\w+|[^\w\s]")
_DEFAULT_REDUNDANCY_WEIGHT = 0.5


@runtime_checkable
class TokenCounter(Protocol):
    """A token-cost estimator: ``count(text) → number of tokens``.

    The whole seam the packer depends on — deliberately one method. It answers "how many tokens of
    the budget does this text consume?", which is exactly where behaviour varies (a rough word/
    punctuation proxy vs. a real BPE tokenizer). It holds no state, so identical text always yields
    the same count. A real tokenizer raises on a non-string input; every adapter here does too
    (:class:`TypeError`) rather than silently coercing.
    """

    def count(self, text: str) -> int:
        """Return the token count of ``text`` (``>= 0``); raise :class:`TypeError` if not a str."""
        ...


class HeuristicTokenCounter:
    """The pure-Python reference :class:`TokenCounter` — the day-one default, no dependency.

    Counts word pieces and standalone punctuation with a fixed regex, a deterministic, CPU-only
    estimate of a real tokenizer's cost that needs no model, no download, and no config (ADR-0016/
    0049 pure-Python reference discipline, like RRF and the basic BM25 rescorer). It is an
    *estimate*, not a specific tokenizer's exact count — the packer only needs a consistent,
    deterministic cost to respect a budget. For an exact count against a named model, inject or
    configure :class:`TiktokenCounter`.
    """

    def count(self, text: str) -> int:
        """Count word/punctuation tokens in ``text``; raise :class:`TypeError` if not a str."""
        if not isinstance(text, str):
            msg = f"HeuristicTokenCounter.count needs a string, got {type(text).__name__}."
            raise TypeError(msg)
        return len(_HEURISTIC_TOKEN.findall(text))


class TiktokenCounter:
    """The real :class:`TokenCounter` — an exact count from a ``tiktoken`` encoding (``[pack]``).

    Wraps ``tiktoken.get_encoding(name).encode`` for an exact token count against a named encoding.
    The heavy dep ships behind this package's ``[pack]`` extra (never latence-core or the pure-
    Python retrieval core, ADR-0016/0049) and is imported **lazily** on first :meth:`count`, so
    constructing the counter — and registry discovery / :class:`Packer` construction — never imports
    ``tiktoken``.

    Licence discipline (ADR-0012): **no encoding is pinned** — ``encoding`` is required, so a caller
    only runs a tokenizer it named. ``tiktoken`` is MIT (code); its encoding data ships with the
    library — no model weights are attested here. A missing name is a :class:`ValueError`; an
    unknown encoding is a :class:`ValueError` (caller config); a native runtime fault is a
    :class:`~latence_retrieval.errors.PackTokenizerError` carrying no raw text.
    """

    def __init__(self, encoding: str) -> None:
        if not encoding:
            msg = (
                "pack's real tokenizer requires an 'encoding' name (e.g. a tiktoken encoding): no "
                "default is pinned (ADR-0012). Omit the tokenizer to use the pure-Python heuristic."
            )
            raise ValueError(msg)
        self._encoding_name = encoding
        self._encoding: Any = None

    def _load(self) -> None:
        if self._encoding is not None:
            return
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'tiktoken' package is required to run pack with the real tokenizer. Install "
                "it with: pip install 'latence-retrieval[pack]' (or omit the tokenizer for the "
                "dependency-free heuristic)."
            )
            raise ImportError(msg) from exc
        try:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        except (ValueError, KeyError) as exc:
            # An unknown encoding name is a caller config error, not a runtime fault.
            msg = f"pack could not load tiktoken encoding {self._encoding_name!r}."
            raise ValueError(msg) from exc

    def count(self, text: str) -> int:
        """Exact ``tiktoken`` token count of ``text``; :class:`TypeError` if not a str."""
        if not isinstance(text, str):
            msg = f"TiktokenCounter.count needs a string, got {type(text).__name__}."
            raise TypeError(msg)
        self._load()
        try:
            return len(self._encoding.encode(text))
        except Exception as exc:
            # A native tiktoken fault → typed error with NO raw text (no leak).
            msg = f"pack tokenizer failure during encode: {type(exc).__name__}."
            raise PackTokenizerError(msg) from exc


class Packer:
    """Select a token-budgeted subset maximising value − redundancy (the Knapsack QKP, ADR-0048).

    A :class:`~latence_retrieval.processor.CandidateProcessor`: :meth:`process` returns the subset
    of the input candidates that greedily maximises the packing objective (see the module docstring)
    under ``budget`` tokens. The returned candidates are the *originals*, unchanged (``pack``
    selects, it does not rescore), in a new list ordered by descending first-stage ``score`` with an
    ascending-``id`` tiebreak, so identical inputs always yield an identical subset and order. The
    input list is never mutated and the packer keeps no state across calls.

    Config:
        budget: int — the token budget. **Required**, must be a positive int. Every candidate costs
            at least one token, so a subset never exceeds this.
        redundancy_weight: float — the penalty coefficient ``λ`` on pairwise similarity. Default
            ``0.5``; must be ``>= 0`` (``0`` disables the redundancy penalty → a pure token-budgeted
            top-value knapsack).
        value_signals: dict[str, float] — optional additive richness signals, ``metadata-key →
            weight``. Each present numeric metadata field adds ``weight · value`` to that
            candidate's value (recency, a numeric role priority); a **per-entity list** value (the
            framework's own ``kg_node_centrality`` graph-feature column, emitted as a
            ``list<double>``) is reduced to its **peak** first — identically to ``expand`` — so the
            documented graph-augmented-pack wiring runs on the framework's own export. An absent key
            contributes nothing (graceful degradation). Default ``{}``.
        token_key: str | None — if set, per-candidate token cost is read from
            ``metadata[token_key]`` (e.g. an exact count made at index time) when present, falling
            back to the :class:`TokenCounter` when a candidate lacks it. Default ``None`` (always
            count via the :class:`TokenCounter`).
        cluster_key: str | None — if set, two candidates sharing ``metadata[cluster_key]`` are
            treated as maximally redundant (similarity ``1.0``). Default ``None``.
        similarity_key: str | None — if set, ``metadata[similarity_key]`` is read as a
            ``{other_id: similarity}`` map and used directly for redundancy (a caller-supplied /
            embedding-based signal). Default ``None``.
        tokenizer: dict — optional real-tokenizer spec ``{"encoding": <name>}``; selects
            :class:`TiktokenCounter` (``[pack]`` extra) instead of the heuristic. The encoding is
            required and license-verified separately (ADR-0012); the dep is imported lazily on first
            use, so construction never needs it. Omit for the dependency-free heuristic.

    Redundancy precedence, per candidate pair: an explicit ``similarity_key`` entry (either
    direction) wins; else equal ``cluster_key`` values → ``1.0``; else lexical token-overlap
    (Jaccard) over ``text``. The tokenizer is resolved lazily on first :meth:`process`.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, *, tokens: TokenCounter | None = None
    ) -> None:
        cfg = dict(config or {})
        budget = cfg.get("budget")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            msg = f"pack 'budget' must be a positive integer (a token budget), got {budget!r}."
            raise ValueError(msg)
        self._budget = budget

        redundancy_weight = cfg.get("redundancy_weight", _DEFAULT_REDUNDANCY_WEIGHT)
        if (
            not isinstance(redundancy_weight, (int, float))
            or isinstance(redundancy_weight, bool)
            or redundancy_weight < 0
        ):
            msg = (
                f"pack 'redundancy_weight' must be a number >= 0, got {redundancy_weight!r}."
            )
            raise ValueError(msg)
        self._redundancy_weight = float(redundancy_weight)

        self._value_signals = self._parse_value_signals(cfg.get("value_signals", {}))
        self._token_key = self._optional_str(cfg.get("token_key"), "token_key")
        self._cluster_key = self._optional_str(cfg.get("cluster_key"), "cluster_key")
        self._similarity_key = self._optional_str(cfg.get("similarity_key"), "similarity_key")
        self._tokenizer_spec = self._parse_tokenizer_spec(cfg.get("tokenizer"))

        # Dependency injection: an explicit counter (tests / reuse) wins; otherwise the configured
        # real tokenizer or the heuristic is built lazily on first use (no import at construction).
        self._counter: TokenCounter | None = tokens

    # --- construction-time config parsing -----------------------------------------------------

    @staticmethod
    def _parse_value_signals(raw: object) -> dict[str, float]:
        if not isinstance(raw, dict):
            msg = f"pack 'value_signals' must be a mapping of metadata-key → weight, got {raw!r}."
            raise TypeError(msg)
        parsed: dict[str, float] = {}
        for key, weight in raw.items():
            if not isinstance(key, str) or not key:
                msg = f"pack 'value_signals' keys must be non-empty strings, got {key!r}."
                raise ValueError(msg)
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                msg = f"pack 'value_signals[{key!r}]' must be a number, got {weight!r}."
                raise ValueError(msg)
            parsed[key] = float(weight)
        return parsed

    @staticmethod
    def _optional_str(value: object, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            msg = (
                f"pack {name!r} must be a non-empty string metadata key or omitted, got {value!r}."
            )
            raise ValueError(msg)
        return value

    @staticmethod
    def _parse_tokenizer_spec(raw: object) -> dict[str, str] | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            msg = f"pack 'tokenizer' must be a mapping like {{'encoding': <name>}}, got {raw!r}."
            raise TypeError(msg)
        encoding = raw.get("encoding")
        if not isinstance(encoding, str) or not encoding:
            msg = "pack 'tokenizer' requires a non-empty string 'encoding' name (ADR-0012)."
            raise ValueError(msg)
        return {"encoding": encoding}

    def _resolve_counter(self) -> TokenCounter:
        if self._counter is None:
            if self._tokenizer_spec is not None:
                self._counter = TiktokenCounter(self._tokenizer_spec["encoding"])
            else:
                self._counter = HeuristicTokenCounter()
        return self._counter

    # --- the transform ------------------------------------------------------------------------

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Select the token-budgeted subset maximising value − redundancy; see the class docstring.

        Returns a new list of the *original* candidates (unchanged) in descending-``score`` /
        ascending-``id`` order. An empty input returns ``[]`` without touching the tokenizer (so no
        weights/encoding are needed to pack nothing). ``query`` is part of the seam signature but
        the packer selects on the first-stage ``score``; re-scoring against the query is the job of
        ``rerank``.
        """
        items = list(candidates)
        if not items:
            return []
        costs = [self._token_cost(candidate) for candidate in items]
        values = self._values(items)
        chosen = self._greedy_pack(items, values, costs)
        selected = [items[index] for index in chosen]
        selected.sort(key=lambda candidate: (-candidate.score, candidate.id))
        return selected

    def _token_cost(self, candidate: Candidate) -> int:
        """Per-candidate token cost (``>= 1``): metadata override when present, else the counter."""
        if self._token_key is not None and self._token_key in candidate.metadata:
            raw = candidate.metadata[self._token_key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                msg = (
                    f"pack token_key {self._token_key!r} must be a non-negative number, "
                    f"got {raw!r} for candidate {candidate.id!r}."
                )
                raise ValueError(msg)
            if raw < 0:
                msg = (
                    f"pack token_key {self._token_key!r} must be non-negative, "
                    f"got {raw!r} for candidate {candidate.id!r}."
                )
                raise ValueError(msg)
            cost = int(raw)
        else:
            cost = self._resolve_counter().count(candidate.text)
        # Every candidate costs at least one token, so a zero-length/empty text is never "free"
        # (which would let it always fit and divide-by-zero the density) — a documented invariant.
        return max(cost, 1)

    def _values(self, items: Sequence[Candidate]) -> list[float]:
        """Per-candidate value: max-scaled relevance + weighted numeric signals."""
        relevance = _relevance_scaled([candidate.score for candidate in items])
        return [
            rel + self._signal_bonus(candidate)
            for rel, candidate in zip(relevance, items, strict=True)
        ]

    def _signal_bonus(self, candidate: Candidate) -> float:
        bonus = 0.0
        for key, weight in self._value_signals.items():
            if key not in candidate.metadata:
                continue  # graceful: an absent signal contributes nothing
            bonus += weight * _signal_scalar(key, candidate)
        return bonus

    def _greedy_pack(
        self, items: Sequence[Candidate], values: Sequence[float], costs: Sequence[int]
    ) -> list[int]:
        """Deterministic greedy QKP: add the best positive-marginal-per-token item that still fits.

        Repeatedly picks the unselected candidate with the highest marginal-value-per-token — where
        marginal = ``value − redundancy_weight · Σ similarity(candidate, already-selected)`` — among
        those that fit the remaining budget, stopping once no remaining item has a positive marginal
        (adding one would *lower* the objective) or none fits. Ties break by higher marginal, then
        by ascending ``id``, so the result is fully deterministic on CPU (ADR-0049).
        """
        remaining = self._budget
        chosen: list[int] = []
        chosen_set: set[int] = set()
        while True:
            best_index = -1
            best_key: tuple[float, float] | None = None
            for index in range(len(items)):
                if index in chosen_set or costs[index] > remaining:
                    continue
                marginal = values[index] - self._redundancy_weight * sum(
                    self._similarity(items[index], items[picked]) for picked in chosen
                )
                if marginal <= 0:
                    continue
                key = (marginal / costs[index], marginal)
                if (
                    best_key is None
                    or key > best_key
                    or (key == best_key and items[index].id < items[best_index].id)
                ):
                    best_index, best_key = index, key
            if best_index == -1:
                break
            chosen.append(best_index)
            chosen_set.add(best_index)
            remaining -= costs[best_index]
        return chosen

    def _similarity(self, a: Candidate, b: Candidate) -> float:
        """Redundancy in ``[0, 1]``: similarity_key, then cluster equality, then lexical Jaccard."""
        if self._similarity_key is not None:
            explicit = _lookup_similarity(a, b, self._similarity_key)
            if explicit is not None:
                return explicit
        if self._cluster_key is not None:
            cluster_a = a.metadata.get(self._cluster_key)
            cluster_b = b.metadata.get(self._cluster_key)
            if cluster_a is not None and cluster_a == cluster_b:
                return 1.0
        return _lexical_similarity(a.text, b.text)


def _signal_scalar(key: str, candidate: Candidate) -> float:
    """Reduce a present ``value_signals`` metadata value to the scalar the packer weights.

    A richness signal is either an already-scalar number (``recency``, a numeric role priority) or
    the framework's own **per-entity graph feature** shape — ``kg_node_centrality`` is emitted as a
    ``list<double>``, one centrality per entity of the chunk (``export._context_columns_for_row``).
    So a list/tuple is reduced to its **peak** (``max``), identically to how ``expand`` consumes the
    same column (:meth:`~latence_retrieval.expand.Expander._peak_centrality`): the two query-time
    consumers ADR-0051 names read the one emitted column the same way, so the documented
    graph-augmented-pack wiring is alive on the framework's own corpus instead of raising.

    Graceful + loud, matching ``expand``: a scalar is used directly; an empty list contributes
    ``0.0`` (no signal); a non-numeric / non-finite entry, a ``bool``, or a ``str`` fails loudly
    (opaque caller metadata is never silently coerced).
    """
    value = candidate.metadata[key]
    if isinstance(value, bool):
        msg = (
            f"pack value_signal {key!r} must be numeric where present, "
            f"got {value!r} for candidate {candidate.id!r}."
        )
        raise ValueError(msg)
    if isinstance(value, (int, float)):
        return _finite_signal(key, value, candidate)
    if isinstance(value, (list, tuple)):
        # The per-entity graph-feature shape (e.g. kg_node_centrality): peak, like ``expand``.
        # An empty list is no signal (0.0); otherwise the true max (never clamped), matching
        # ``expand._peak_centrality`` so both consumers read the one emitted column identically.
        elements = [_finite_signal(key, element, candidate) for element in value]
        return max(elements) if elements else 0.0
    msg = (
        f"pack value_signal {key!r} must be a number or a per-entity list of numbers where "
        f"present, got {value!r} for candidate {candidate.id!r}."
    )
    raise ValueError(msg)


def _finite_signal(key: str, element: object, candidate: Candidate) -> float:
    """A single signal number: reject ``bool`` / non-numeric / non-finite, else its ``float``."""
    if isinstance(element, bool) or not isinstance(element, (int, float)):
        msg = (
            f"pack value_signal {key!r} must be numeric where present, "
            f"got {element!r} for candidate {candidate.id!r}."
        )
        raise ValueError(msg)
    if not math.isfinite(element):
        msg = (
            f"pack value_signal {key!r} must be finite where present, "
            f"got {element!r} for candidate {candidate.id!r}."
        )
        raise ValueError(msg)
    return float(element)


def _relevance_scaled(scores: Sequence[float]) -> list[float]:
    """Scale first-stage ``scores`` to relevance in ``[0, 1]`` by the set maximum.

    ``relevance_i = max(score_i, 0) / max_score`` — a *ratio* scaling, not min-max: the least-
    relevant positive candidate keeps a positive value (it is not crushed to ``0``), so a modest
    ``redundancy_weight`` can genuinely trade relevance against diversity, and a candidate scored at
    or below ``0`` contributes no relevance (only redundancy cost). When no candidate has a positive
    score there is no relevance signal, so every candidate is treated as equally relevant (``1.0``)
    and packing is driven purely by redundancy — a sensible degenerate fallback.
    """
    if not scores:
        return []
    top = max(scores)
    if top <= 0:
        return [1.0] * len(scores)
    return [max(score, 0.0) / top for score in scores]


def _lookup_similarity(a: Candidate, b: Candidate, key: str) -> float | None:
    """Read a caller-supplied similarity between ``a`` and ``b`` from ``metadata[key]``.

    ``metadata[key]`` is a ``{other_id: similarity}`` map; ``a``→``b`` is tried first, then
    ``b``→``a`` (symmetry), and the value is clamped to ``[0, 1]``. Returns ``None`` when neither
    carries the pair (so the caller falls through to cluster/lexical redundancy) and ignores a
    malformed (non-mapping / non-numeric) entry rather than crashing on opaque caller metadata.
    """
    for source, other in ((a, b), (b, a)):
        table = source.metadata.get(key)
        if not isinstance(table, Mapping):
            continue
        raw = table.get(other.id)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        return min(1.0, max(0.0, float(raw)))
    return None


def _lexical_similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' token sets — the dependency-free default redundancy.

    ``|A ∩ B| / |A ∪ B|`` over lowercased word tokens; ``0.0`` when either side has no tokens (an
    empty text is redundant with nothing). Deterministic and CPU-only.
    """
    terms_a = set(_TOKEN.findall(a.lower()))
    terms_b = set(_TOKEN.findall(b.lower()))
    if not terms_a or not terms_b:
        return 0.0
    union = terms_a | terms_b
    return len(terms_a & terms_b) / len(union)


__all__ = ["HeuristicTokenCounter", "Packer", "TiktokenCounter", "TokenCounter"]

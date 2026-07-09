"""``graph_completion.ultra`` — the learned GraphCompleter Provider (ULTRA + PyKEEN, ADR-0037 §7).

The learned link-prediction Provider on the Slice-1 :class:`~latence_core.capability.GraphCompleter`
seam. It consumes the whole run's assembled :class:`~latence_core.contracts.GraphRecord`s (NODE +
EDGE), builds the integer-indexed KG, generates type-aware tiered 2-hop candidates, scores them with
the inductive **ULTRA** scorer (zero-shot from a pretrained checkpoint) — falling back to **PyKEEN**
when ULTRA's deps/checkpoint are absent, and to a typed :class:`~latence_core.errors.ProviderError`
when NEITHER is available (never a silent no-op) — calibrates the raw scores (temperature scaling +
a precision-targeted threshold), and yields ADDITIONAL inferred EDGE-scope GraphRecords via the
Slice-1 inferred-edge contract EXACTLY:

* ``scope=EDGE``, ``properties["inferred"]=True``, ``scorer`` (``"ultra"`` / ``"pykeen"``),
  ``score`` (the CALIBRATED probability), ``rank``, and ``calibrated=True`` (the temperature-scaled
  ULTRA/PyKEEN probability IS calibrated, unlike the reference completer's raw fraction);
* deterministic ``edge_id = sha256(corpus_id, "predicted", head, label, tail)`` and
  ``relation_id = f"predicted:{scorer}:{edge_id}"`` — identical id scheme to Slice 1, so an inferred
  edge from either scorer never collides with an asserted edge;
* **honest, model-derived Evidence** — ``mention_ids=[]`` (a predicted edge has no textual mention),
  ``document_ids`` = the union of the head + tail nodes' ``source_document_ids``, ``snippet`` names
  ULTRA/PyKEEN + the calibrated score, ``confidence`` = the calibrated probability;
* an inferred edge COLLIDING (same endpoints + label) with an asserted edge is DROPPED (logged),
  and predicted edges are yielded in ``edge_id`` order (deterministic).

The heavy deps live only in this package (ADR-0016); the scorer imports torch/pykeen lazily so
provider discovery stays torch-free. On a CPU-only host with the ``[ultra]``/``[pykeen]`` extras
absent, the Stage is **skipped-with-flag** via the device seam (``compute="either"``), never faked.
The KG-quality metrics (held-out MRR / Hits@k with genuine leak detection) are computed by the
:mod:`~latence_linkpred_ultra.evaluation` harness and surfaced on the run (the real numbers are
re-validated on the pod, never fabricated).

Built through :class:`~latence_core.providers.adapter.AdapterBase` (P3-F2 / ADR-0036 §1): the device
seam routes ``either`` → cpu/cuda / skip, and :meth:`~...AdapterBase.guard` maps a native scorer
fault to a typed :class:`~latence_core.errors.ProviderError` carrying no record content / PII.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    Evidence,
    GraphEdge,
    GraphRecord,
    GraphScope,
)
from latence_core.errors import ProviderError
from latence_core.graph import make_graph_id
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import ProviderProfile

from latence_linkpred_ultra.calibrator import Calibrator, CalibratorConfig
from latence_linkpred_ultra.candidates import (
    Candidate,
    CandidateConfig,
    CandidateGenerator,
    ObservedEdge,
)
from latence_linkpred_ultra.evaluation import (
    LeakDetectionError,
    RankingMetrics,
    assert_no_leak,
    metrics_from_ranks,
    rank_of,
    split_edges,
)
from latence_linkpred_ultra.scorer import (
    BuiltKG,
    PykeenScorer,
    Scorer,
    ScorerUnavailableError,
    UltraScorer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

_LOG = logging.getLogger(__name__)

# ULTRA + PyKEEN license facts (ADR-0012, verified 2026-07-09; weights AND code, SEPARATELY):
#   * ULTRA code:   MIT (github.com/DeepGraphLearning/ULTRA/blob/main/LICENSE, MilaGraph)
#   * ULTRA weights: MIT (HF mgalkin/ultra_3g / ultra_4g / ultra_50g model cards, license: mit)
#   * PyKEEN:       MIT (github.com/pykeen/pykeen/blob/master/LICENSE)
_LICENSE_SOURCE = (
    "https://github.com/DeepGraphLearning/ULTRA/blob/main/LICENSE (code, MIT); "
    "https://huggingface.co/mgalkin/ultra_3g (weights, MIT); "
    "https://github.com/pykeen/pykeen/blob/master/LICENSE (PyKEEN, MIT)"
)


class UltraGraphCompleter(AdapterBase):
    """Learned GraphCompleter: assembled GraphRecords -> calibrated inferred EDGE GraphRecords.

    Config:
        corpus_id: str  — stable corpus identity mixed into every deterministic predicted
            ``edge_id`` (default ``"corpus"``).
        scorer: str  — ``"ultra"`` (default), ``"pykeen"``, or ``"auto"`` (ULTRA, PyKEEN on
            unavailability). ``"auto"`` is the graceful path.
        checkpoint: str  — the ULTRA checkpoint path/id (default ``ultra_3g``; not vendored).
        pykeen_model: str  — the PyKEEN fallback model class (default ``DistMult``).
        device: str  — ``auto|cpu|cuda`` (the AdapterBase device seam).
        target_precision: float  — the calibrator's target precision for the emit threshold (0.9).
        temperature: float  — fallback temperature when no held-out labelled set fits it (1.0).
        min_probability: float  — fallback emit floor when no labelled set derives a cut (0.5).
        max_candidates_per_relation: int  — dense-relation fan-out cap (512; 0 = unlimited).
        rare_relation_max_edges: int  — a relation observed <= this many times keeps recall (4).
        two_hop: bool  — restrict candidates to the 2-hop neighbourhood (True).
        type_aware: bool  — apply the relation type-signature filter (True).
        holdout_fraction: float  — fraction of observed edges held out for MRR/Hits@k (0.2).
        hits_k: list[int]  — the k's for Hits@k (default [1, 3, 10]).
        eval_seed: int  — the held-out split seed (0).
        max_emit: int  — cap on emitted inferred edges (0 = unlimited); the highest-scored survive.
    """

    #: A missing ULTRA checkpoint / a bad scorer name is an operator config error, not a bug.
    config_exceptions = (ValueError, KeyError)

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): learned, ``either`` compute, MIT-licensed.

        ``compute="either"``: ULTRA runs on CPU (proven on the pod) and accelerates on CUDA via the
        device seam; a CPU-only host with the extras absent gets the skip-with-flag decision, never
        a faked device (P3-F1). License **verified** per ADR-0012 (2026-07-09, weights AND code
        separately): ULTRA code MIT, ULTRA weights MIT, PyKEEN MIT — all permissive, so
        ``license_verified=True``. ``deterministic=False``: learned float inference is not
        guaranteed byte-identical across hardware/library versions, so the profile declares it
        rather than overclaim reproducibility (the candidate order + greedy scoring ARE fixed; the
        floats are not). ``model_id`` is the default checkpoint id; the WEIGHTS are not vendored.
        """
        return ProviderProfile(
            compute="either",
            memory_mb=2048,
            model_id="ultra_3g",
            license="MIT",
            license_verified=True,
            license_code="MIT",
            license_weights="MIT",
            license_source=_LICENSE_SOURCE,
            license_verified_on="2026-07-09",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self._config
        self._corpus_id = str(cfg.get("corpus_id", "corpus"))
        self._scorer_choice = str(cfg.get("scorer", "ultra")).lower()
        if self._scorer_choice not in {"ultra", "pykeen", "auto"}:
            msg = f"scorer must be one of ultra|pykeen|auto, got {self._scorer_choice!r}"
            raise ValueError(msg)
        self._checkpoint = str(cfg.get("checkpoint", "ultra_3g"))
        self._pykeen_model = str(cfg.get("pykeen_model", "DistMult"))
        self._candidate_config = CandidateConfig(
            max_candidates_per_relation=int(cfg.get("max_candidates_per_relation", 512)),
            rare_relation_max_edges=int(cfg.get("rare_relation_max_edges", 4)),
            two_hop=bool(cfg.get("two_hop", True)),
            type_aware=bool(cfg.get("type_aware", True)),
        )
        self._calibrator_config = CalibratorConfig(
            temperature=float(cfg.get("temperature", 1.0)),
            target_precision=float(cfg.get("target_precision", 0.9)),
            min_probability=float(cfg.get("min_probability", 0.5)),
        )
        self._holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
        self._hits_k = tuple(int(k) for k in cfg.get("hits_k", [1, 3, 10]))
        self._eval_seed = int(cfg.get("eval_seed", 0))
        self._max_emit = int(cfg.get("max_emit", 0))
        # When True (the OPTIONAL trailing-stage posture in gpu-sota stacks), a host with NO learned
        # scorer available (extras/checkpoint absent, e.g. CPU-only CI) SKIPS-with-flag — yields no
        # inferred edges, logs the reason — instead of raising. Default False: the issue's rule is a
        # TYPED ProviderError when neither scorer is available (never a silent no-op) unless a stack
        # explicitly opts into skip-with-flag for the optional stage.
        self._skip_if_unavailable = bool(cfg.get("skip_if_unavailable", False))
        #: Populated by :meth:`complete`: the held-out MRR / Hits@k metrics (or None if not eval'd).
        self.last_metrics: RankingMetrics | None = None
        #: Populated by :meth:`complete`: which scorer actually ran (``"ultra"`` / ``"pykeen"``).
        self.last_scorer: str | None = None

    # -- the GraphCompleter Capability --------------------------------------

    def complete(self, records: Iterable[GraphRecord]) -> Iterator[GraphRecord]:
        """Consume the assembled graph; yield calibrated inferred EDGE records (Slice-1 contract).

        If the device seam skipped this Provider (GPU-only host, extras absent) it yields NOTHING —
        the honest skip-with-flag disposition (the reference completer covers the CPU spine). Else
        it
        builds the KG, generates candidates, scores + calibrates them, evaluates held-out MRR/Hits@k
        (with leak detection), and emits the surviving inferred edges in edge-id order.
        """
        if self.device.skipped:
            _LOG.info(
                "graph_completion.ultra: skipped-with-flag (%s) — no inferred edges emitted "
                "on this host.",
                self.device.reason,
            )
            return

        graph = _AssembledGraph.from_records(records)
        if not graph.observed:
            _LOG.info("graph_completion.ultra: no observed edges; nothing to complete.")
            return

        with self.guard("complete"):
            scorer = self._build_scorer(graph)
            if scorer is None:
                # skip_if_unavailable: no learned scorer on this host — skip-with-flag, emit
                # nothing.
                return
            self.last_scorer = scorer.name
            self._evaluate(graph, scorer)
            emitted = self._score_and_emit(graph, scorer)

        emitted.sort(key=lambda r: r.edge.edge_id if r.edge is not None else "")
        yield from emitted

    # -- scorer selection (graceful fallback → typed error) -----------------

    def _build_scorer(self, graph: _AssembledGraph) -> Scorer | None:
        device = self.device.device or "cpu"
        kg = graph.built_kg()
        errors: list[str] = []

        def try_ultra() -> Scorer | None:
            scorer = UltraScorer(checkpoint=self._checkpoint, device=device)
            try:
                scorer.prepare(kg)
            except ScorerUnavailableError as exc:
                errors.append(f"ultra: {exc}")
                return None
            return scorer

        def try_pykeen() -> Scorer | None:
            scorer = PykeenScorer(model=self._pykeen_model, device=device, seed=self._eval_seed)
            try:
                scorer.prepare(kg)
            except ScorerUnavailableError as exc:
                errors.append(f"pykeen: {exc}")
                return None
            return scorer

        order: list[Callable[[], Scorer | None]]
        if self._scorer_choice == "ultra":
            order = [try_ultra]
        elif self._scorer_choice == "pykeen":
            order = [try_pykeen]
        else:  # auto: ULTRA first, PyKEEN fallback
            order = [try_ultra, try_pykeen]

        for attempt in order:
            scorer = attempt()
            if scorer is not None:
                return scorer

        # Neither scorer is available.
        detail = "; ".join(errors) if errors else "no scorer available"
        if self._skip_if_unavailable:
            # The OPTIONAL trailing-stage posture (gpu-sota on CPU): skip-with-flag, not a crash.
            _LOG.info(
                "graph_completion.ultra: no learned scorer available (choice=%r): %s — "
                "skip_if_unavailable set, skipping-with-flag (no inferred edges).",
                self._scorer_choice,
                detail,
            )
            return None
        # Default: a TYPED error, never a silent no-op (the issue's rule).
        msg = (
            "graph_completion.ultra: no learned scorer available "
            f"(choice={self._scorer_choice!r}): {detail}. Install latence-linkpred-ultra[ultra] "
            "(+ an ULTRA checkpoint) or latence-linkpred-ultra[pykeen]."
        )
        raise ProviderError(msg)

    # -- candidate scoring + emission ---------------------------------------

    def _score_and_emit(self, graph: _AssembledGraph, scorer: Scorer) -> list[GraphRecord]:
        candidates = CandidateGenerator(self._candidate_config).generate(
            graph.node_types, graph.observed
        )
        if not candidates:
            return []
        calibrator = Calibrator(self._calibrator_config)
        calibration = calibrator.fit(self._labelled_scores(graph, scorer))

        # Score every candidate's tail: one query per (head, relation), reused across its tails.
        scored = self._score_candidates(graph, scorer, candidates)

        emitted: list[GraphRecord] = []
        for cand, raw in scored:
            prob = calibrator.probability(raw, calibration.temperature)
            if prob < calibration.threshold:
                continue  # precision-targeted: don't emit a below-target belief
            key = (cand.head, cand.relation, cand.tail)
            if key in graph.asserted_keys:
                _LOG.debug(
                    "graph_completion.ultra: dropping candidate colliding with asserted: %s", key
                )
                continue
            emitted.append(self._predicted_record(graph, cand, prob, scorer.name))

        # Rank globally by calibrated prob (desc), then edge key; stamp rank; optional emit cap.
        emitted.sort(
            key=lambda r: (
                -(r.edge.properties["score"] if r.edge else 0.0),
                r.edge.edge_id if r.edge else "",
            )
        )
        if self._max_emit > 0 and len(emitted) > self._max_emit:
            dropped = len(emitted) - self._max_emit
            _LOG.info(
                "graph_completion.ultra: capping emitted inferred edges to %d (dropping %d "
                "lowest-scored — logged, not silent).",
                self._max_emit,
                dropped,
            )
            emitted = emitted[: self._max_emit]
        for rank, record in enumerate(emitted):
            if record.edge is not None:
                record.edge.properties["rank"] = rank
        return emitted

    def _score_candidates(
        self,
        graph: _AssembledGraph,
        scorer: Scorer,
        candidates: Sequence[Candidate],
    ) -> list[tuple[Candidate, float]]:
        """Score each candidate's tail via one (head, relation) query per distinct pair."""
        query_index: dict[tuple[int, int], int] = {}
        queries: list[tuple[int, int]] = []
        for cand in candidates:
            h = graph.entity_index[cand.head]
            rel = graph.relation_index[cand.relation]
            if (h, rel) not in query_index:
                query_index[(h, rel)] = len(queries)
                queries.append((h, rel))
        score_vectors = scorer.score_tails(queries)
        out: list[tuple[Candidate, float]] = []
        for cand in candidates:
            h = graph.entity_index[cand.head]
            rel = graph.relation_index[cand.relation]
            t = graph.entity_index[cand.tail]
            vec = score_vectors[query_index[(h, rel)]]
            out.append((cand, vec[t]))
        return out

    def _labelled_scores(
        self,
        graph: _AssembledGraph,
        scorer: Scorer,
    ) -> list[tuple[float, bool]] | None:
        """A held-out labelled set (raw_score, is_true) to fit the calibrator, or None.

        Positives are the held-out true edges' raw scores; negatives are a deterministic sample of
        corrupt tails per positive. With too few observed edges to split, returns None (the
        calibrator falls back to the configured temperature + min_probability floor).
        """
        split = self._split(graph)
        if split is None or not split.held_out:
            return None
        queries = [
            (graph.entity_index[e.head], graph.relation_index[e.relation])
            for e in split.held_out
        ]
        vectors = scorer.score_tails(queries)
        labelled: list[tuple[float, bool]] = []
        for e, vec in zip(split.held_out, vectors, strict=True):
            t = graph.entity_index[e.tail]
            labelled.append((vec[t], True))
            # One deterministic hard-ish negative: the highest-scoring non-true tail.
            neg_idx = _argmax_excluding(vec, exclude={t})
            if neg_idx is not None:
                labelled.append((vec[neg_idx], False))
        return labelled

    # -- held-out MRR / Hits@k with leak detection --------------------------

    def _evaluate(self, graph: _AssembledGraph, scorer: Scorer) -> None:
        split = self._split(graph)
        if split is None or not split.held_out:
            self.last_metrics = None
            return
        # GENUINE leak detection: the held-out edges + inverses must be gone from the scored graph.
        # The scorer was prepared over the FULL observed graph for candidate emission; for an honest
        # held-out eval we re-prepare it over the observed-minus-heldout graph, then assert no leak.
        assert_no_leak(split.observed, split.held_out)
        # assert_no_leak returns only if NO held-out edge (or inverse) leaks into the observed
        # graph — it raises LeakDetectionError otherwise. Reaching here means the guard ran and
        # passed; record that fact and thread it into the metrics so the flag is not tautological.
        leak_checked = True
        eval_kg = graph.built_kg_over(split.observed)
        eval_scorer = self._prepare_like(scorer, eval_kg)

        queries = [
            (graph.entity_index[e.head], graph.relation_index[e.relation])
            for e in split.held_out
        ]
        vectors = eval_scorer.score_tails(queries)
        # Filtered ranking: other true tails for the same (h, r) in the FULL graph are filtered out.
        true_tails = graph.true_tails_by_hr()
        ranks: list[int] = []
        for e, vec in zip(split.held_out, vectors, strict=True):
            h = graph.entity_index[e.head]
            rel = graph.relation_index[e.relation]
            t = graph.entity_index[e.tail]
            filter_idx = set(true_tails.get((h, rel), set())) - {t}
            ranks.append(rank_of(t, vec, filter_idx))
        self.last_metrics = metrics_from_ranks(ranks, self._hits_k, leak_checked=leak_checked)
        _LOG.info(
            "graph_completion.ultra: held-out eval (%s) — MRR=%.4f, Hits@%s, queries=%d",
            scorer.name,
            self.last_metrics.mrr,
            self.last_metrics.hits,
            self.last_metrics.num_queries,
        )

    def _prepare_like(self, scorer: Scorer, kg: BuiltKG) -> Scorer:
        """A fresh scorer of the same kind prepared over ``kg`` (the leak-free held-out eval)."""
        device = self.device.device or "cpu"
        if scorer.name == "ultra":
            fresh: Scorer = UltraScorer(checkpoint=self._checkpoint, device=device)
        else:
            fresh = PykeenScorer(model=self._pykeen_model, device=device, seed=self._eval_seed)
        fresh.prepare(kg)
        return fresh

    def _split(self, graph: _AssembledGraph) -> Any:
        if len(graph.observed) < 2:
            return None
        try:
            return split_edges(
                graph.observed,
                holdout_fraction=self._holdout_fraction,
                seed=self._eval_seed,
            )
        except LeakDetectionError:
            raise
        except ValueError:
            return None

    # -- inferred-edge record (Slice-1 contract EXACTLY) --------------------

    def _predicted_record(
        self,
        graph: _AssembledGraph,
        cand: Candidate,
        prob: float,
        scorer_name: str,
    ) -> GraphRecord:
        edge_id = make_graph_id(
            self._corpus_id, "predicted", cand.head, cand.relation, cand.tail
        )
        relation_id = f"predicted:{scorer_name}:{edge_id}"
        document_ids = sorted(
            set(graph.node_docs.get(cand.head, []))
            | set(graph.node_docs.get(cand.tail, []))
        )
        evidence = Evidence(
            mention_ids=[],
            document_ids=document_ids,
            snippet=(
                f"Predicted edge (not asserted in text); {scorer_name} "
                f"calibrated probability {prob:.3f}"
            ),
            confidence=prob,
        )
        source = graph.node_source[cand.head]
        graph_edge = GraphEdge(
            record_id=f"edge:{edge_id}",
            edge_id=edge_id,
            relation_id=relation_id,
            source_node_id=cand.head,
            target_node_id=cand.tail,
            label=cand.relation,
            confidence=prob,
            properties={
                "inferred": True,
                "scorer": scorer_name,
                "score": prob,
                "rank": 0,
                "calibrated": True,
                "rare_relation": cand.rare,
            },
            evidence=evidence,
            provenance=source.provenance,
            classification=source.classification,
        )
        return GraphRecord(
            record_id=f"edge:{edge_id}#predicted",
            scope=GraphScope.EDGE,
            edge=graph_edge,
            provenance=source.provenance,
            classification=source.classification,
        )


def _argmax_excluding(vec: list[float], exclude: set[int]) -> int | None:
    best_i: int | None = None
    best_v = float("-inf")
    for i, v in enumerate(vec):
        if i in exclude:
            continue
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


class _AssembledGraph:
    """The materialised assembled graph: entity vocab, relation vocab, observed edges, node meta.

    Built once from the run's :class:`~latence_core.contracts.GraphRecord` stream. Entity/relation
    indices are assigned in sorted id order so the KG is deterministic. Carries the per-node source
    documents (for honest Evidence), a representative source record per node (for inherited
    Provenance/Classification), and the asserted-edge collision set.
    """

    def __init__(self) -> None:
        self.node_types: dict[str, str] = {}
        self.node_docs: dict[str, list[str]] = {}
        self.node_source: dict[str, GraphRecord] = {}
        self.observed: list[ObservedEdge] = []
        self.asserted_keys: set[tuple[str, str, str]] = set()
        self.entity_index: dict[str, int] = {}
        self.relation_index: dict[str, int] = {}

    @classmethod
    def from_records(cls, records: Iterable[GraphRecord]) -> _AssembledGraph:
        g = cls()
        for record in records:
            if record.scope is GraphScope.NODE and record.node is not None:
                node = record.node
                g.node_types[node.node_id] = node.label
                g.node_docs[node.node_id] = list(node.source_document_ids)
                g.node_source[node.node_id] = record
            elif record.scope is GraphScope.EDGE and record.edge is not None:
                edge = record.edge
                g.observed.append(
                    ObservedEdge(
                        head=edge.source_node_id,
                        relation=edge.label,
                        tail=edge.target_node_id,
                    )
                )
                g.asserted_keys.add((edge.source_node_id, edge.label, edge.target_node_id))
        g._index()
        return g

    def _index(self) -> None:
        entities = sorted(
            set(self.node_types)
            | {e.head for e in self.observed}
            | {e.tail for e in self.observed}
        )
        relations = sorted({e.relation for e in self.observed})
        self.entity_index = {eid: i for i, eid in enumerate(entities)}
        self.relation_index = {rel: i for i, rel in enumerate(relations)}

    def built_kg(self) -> BuiltKG:
        return self.built_kg_over(self.observed)

    def built_kg_over(self, edges: list[ObservedEdge]) -> BuiltKG:
        triples = [
            (self.entity_index[e.head], self.relation_index[e.relation], self.entity_index[e.tail])
            for e in edges
        ]
        return BuiltKG(
            num_entities=len(self.entity_index),
            num_relations=len(self.relation_index),
            edges=triples,
        )

    def true_tails_by_hr(self) -> dict[tuple[int, int], set[int]]:
        out: dict[tuple[int, int], set[int]] = {}
        for e in self.observed:
            h = self.entity_index[e.head]
            rel = self.relation_index[e.relation]
            t = self.entity_index[e.tail]
            out.setdefault((h, rel), set()).add(t)
        return out

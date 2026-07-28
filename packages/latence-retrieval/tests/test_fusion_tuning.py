"""T-FUSE-1: the weighted fuser made first-class — theoretical min-max + a runnable weight sweep.

The audit's finding was that :class:`~latence_retrieval.fuser.WeightedFuser` already implements the
per-list min-max-normalised weighted sum that Bruch et al. (TOIS 2023) recommend over RRF — but
**every** harness calls ``RrfFuser()`` with the default ``k=60`` and uniform weights
(``run_benchmark.py:396``, ``build_questions.py:690``, ``rfc_bench.py:82``), and nothing in the
library could produce the comparison table the ticket asks for. So the gap is wiring and tooling,
not the algorithm.

This file covers the two halves:

1. **theoretical min-max** — the *query-independent* normalisation Bruch recommends for serving
   time, previously hard-rejected at ``fuser.py:176-182``. Its whole point is that the mapping from
   raw score to ``[0, 1]`` does not depend on the other hits in *this* query's list, so a document's
   normalised score is stable across queries.
2. :mod:`latence_retrieval.tuning` — evaluate a fuser on a labelled slice, sweep the weights on it
   ("a handful of labeled queries" suffices, per Bruch), and emit the RRF-vs-tuned table. Whichever
   wins, wins: the module reports the comparison, it does not prefer an outcome.

RRF is **not** deprecated anywhere here — it stays the no-labels fallback, and
:func:`test_rrf_remains_the_no_labels_default` pins that.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from latence_retrieval import (
    Candidate,
    Capability,
    OrchestrationPipeline,
    QueryBundle,
    RrfFuser,
    WeightedFuser,
    retrieve,
)
from latence_retrieval.testing import FakeBackend
from latence_retrieval.tuning import (
    FusionRun,
    compare_fusers,
    evaluate_fusion,
    format_comparison_table,
    sweep_fusion_weights,
)


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


# --- theoretical min-max: query-independent normalisation --------------------------------------


def test_weighted_accepts_theoretical_min_max() -> None:
    fuser = WeightedFuser({"normalization": "theoretical_min_max", "bounds": [0.0, 1.0]})
    fused = fuser.fuse([[Candidate(id="a", score=0.75), Candidate(id="b", score=0.25)]])
    assert [round(c.score, 6) for c in fused] == [0.75, 0.25]


def test_theoretical_min_max_is_query_independent() -> None:
    """The property that makes it safe at serving time, and that per-query min-max lacks.

    ``a`` scores 0.75 in both queries. Under per-query min-max its normalised score depends on the
    *other* hits in the list; under theoretical bounds it does not.
    """
    theoretical = WeightedFuser({"normalization": "theoretical_min_max", "bounds": [0.0, 1.0]})
    per_query = WeightedFuser()
    wide = [Candidate(id="a", score=0.75), Candidate(id="b", score=0.0)]
    narrow = [Candidate(id="a", score=0.75), Candidate(id="b", score=0.70)]

    assert theoretical.fuse([wide])[0].score == theoretical.fuse([narrow])[0].score
    assert per_query.fuse([wide])[0].score == per_query.fuse([narrow])[0].score  # both saturate
    # ...but the *loser* moves: per-query min-max stretches 0.70 to 0.0, theoretical keeps it at .7
    assert per_query.fuse([narrow])[1].score == 0.0
    assert theoretical.fuse([narrow])[1].score == pytest.approx(0.70)


def test_theoretical_min_max_accepts_one_bound_pair_per_list() -> None:
    """A cosine list in ``[-1, 1]`` and a BM25 list in ``[0, 40]`` declare their own ranges."""
    fuser = WeightedFuser(
        {"normalization": "theoretical_min_max", "bounds": [[-1.0, 1.0], [0.0, 40.0]]}
    )
    dense = [Candidate(id="a", score=0.0)]  # midpoint of [-1, 1] → 0.5
    lexical = [Candidate(id="b", score=20.0)]  # midpoint of [0, 40] → 0.5
    fused = fuser.fuse([dense, lexical])
    assert [c.score for c in fused] == [0.5, 0.5]


def test_theoretical_min_max_clamps_a_score_outside_its_declared_range() -> None:
    """A declared theoretical bound is a claim about the *score function*, not about this list.

    A BM25 score above the declared ceiling means the ceiling was mis-declared; clamping keeps the
    fusion in ``[0, 1]`` (and makes the mis-declaration visible as saturation) rather than letting
    one list's overshoot silently dominate the convex combination.
    """
    fuser = WeightedFuser({"normalization": "theoretical_min_max", "bounds": [0.0, 1.0]})
    fused = fuser.fuse([[Candidate(id="a", score=3.0), Candidate(id="b", score=-2.0)]])
    assert [c.score for c in fused] == [1.0, 0.0]


def test_theoretical_min_max_requires_bounds() -> None:
    with pytest.raises(ValueError, match="bounds"):
        WeightedFuser({"normalization": "theoretical_min_max"})


def test_min_max_rejects_bounds_it_would_ignore() -> None:
    """Silently ignoring a config key is how a knob stops at a hop; reject it instead."""
    with pytest.raises(ValueError, match="bounds"):
        WeightedFuser({"normalization": "min_max", "bounds": [0.0, 1.0]})


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, 0.0],  # lo >= hi
        [0.0, 0.0],
        [0.0],  # not a pair
        ["a", "b"],
        0.5,
        [[0.0, 1.0], [1.0, 0.0]],
    ],
)
def test_theoretical_min_max_rejects_bad_bounds(bad: Any) -> None:
    with pytest.raises(ValueError, match="bounds"):
        WeightedFuser({"normalization": "theoretical_min_max", "bounds": bad})


def test_per_list_bounds_length_must_match_the_ranked_lists() -> None:
    fuser = WeightedFuser(
        {"normalization": "theoretical_min_max", "bounds": [[0.0, 1.0], [0.0, 1.0]]}
    )
    with pytest.raises(ValueError, match="bounds"):
        fuser.fuse([[Candidate(id="a", score=0.5)]])


def test_weighted_still_rejects_an_unknown_normalization() -> None:
    with pytest.raises(ValueError, match="normalization"):
        WeightedFuser({"normalization": "z_score"})


def test_weighted_rejects_unknown_config_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        WeightedFuser({"normalisation": "min_max"})


def test_theoretical_min_max_is_reachable_from_operator_config() -> None:
    """field → config → provider: the knob must not stop at the registry."""
    pipeline = OrchestrationPipeline.from_config(
        {
            "fuser": {
                "provider": "fuser.weighted",
                "config": {"normalization": "theoretical_min_max", "bounds": [0.0, 1.0]},
            },
            "fuse_weights": {"dense": 2.0, "sparse": 1.0},
        }
    )
    assert isinstance(pipeline.fuser, WeightedFuser)
    assert pipeline.fuse_weights == {"dense": 2.0, "sparse": 1.0}
    fused = pipeline.fuser.fuse([[Candidate(id="a", score=0.25)]], weights=[1.0])
    assert fused[0].score == pytest.approx(0.25)


# --- the labelled slice the sweep runs on ------------------------------------------------------
#
# Three questions. The dense leg is reliable (gold tops it every time); the sparse leg is noisy
# (it tops a distractor the dense leg also ranks second). Under UNIFORM weights the distractor's
# two mid-to-top placements outweigh the gold's single first place and recall@1 is 0 — exactly the
# failure mode D6 describes when one signal's reliability is query-dependent and fusion ignores it.
# A higher dense weight recovers the gold. That is a property of THIS fixture, not a claim about
# corpora in general: the sweep is here to measure, and this fixture is here to prove the sweep can
# actually find a better point than uniform.

_RUNS = (
    FusionRun(
        query_id="q1",
        gold_ids=("g1",),
        rankings={
            "dense": (
                Candidate(id="g1", score=0.90),
                Candidate(id="a1", score=0.50),
                Candidate(id="z1", score=0.10),
            ),
            "sparse": (Candidate(id="a1", score=9.0), Candidate(id="z1", score=1.0)),
        },
    ),
    FusionRun(
        query_id="q2",
        gold_ids=("g2",),
        rankings={
            "dense": (
                Candidate(id="g2", score=0.95),
                Candidate(id="a2", score=0.55),
                Candidate(id="z2", score=0.15),
            ),
            "sparse": (Candidate(id="a2", score=8.0), Candidate(id="z2", score=2.0)),
        },
    ),
    FusionRun(
        query_id="q3",
        gold_ids=("g3",),
        rankings={
            "dense": (
                Candidate(id="g3", score=0.80),
                Candidate(id="a3", score=0.40),
                Candidate(id="z3", score=0.05),
            ),
            "sparse": (Candidate(id="a3", score=7.0), Candidate(id="z3", score=1.5)),
        },
    ),
)


def test_the_fixture_makes_uniform_weights_genuinely_lose() -> None:
    """Guard the guard: if uniform already won, the sweep test below would prove nothing."""
    uniform = evaluate_fusion(_RUNS, WeightedFuser(), weights={"dense": 1.0, "sparse": 1.0}, k=1)
    assert uniform.recall_at_k == pytest.approx(0.0)


# --- metrics ------------------------------------------------------------------------------------


def test_evaluate_reports_recall_mrr_and_ndcg_at_k() -> None:
    score = evaluate_fusion(_RUNS, RrfFuser(), k=1)
    assert score.queries == 3
    assert 0.0 <= score.recall_at_k <= 1.0
    assert 0.0 <= score.ndcg_at_k <= 1.0
    assert 0.0 <= score.mrr <= 1.0
    assert score.k == 1


def test_recall_at_k_is_computed_against_the_fused_top_k() -> None:
    run = FusionRun(
        query_id="q",
        gold_ids=("a", "b"),
        rankings={"dense": (Candidate(id="a", score=1.0), Candidate(id="b", score=0.5))},
    )
    assert evaluate_fusion([run], RrfFuser(), k=1).recall_at_k == pytest.approx(0.5)
    assert evaluate_fusion([run], RrfFuser(), k=2).recall_at_k == pytest.approx(1.0)


def test_mrr_reads_the_rank_of_the_first_gold_hit() -> None:
    run = FusionRun(
        query_id="q",
        gold_ids=("b",),
        rankings={"dense": (Candidate(id="a", score=1.0), Candidate(id="b", score=0.5))},
    )
    assert evaluate_fusion([run], RrfFuser(), k=10).mrr == pytest.approx(0.5)


def test_ndcg_is_one_when_every_gold_leads() -> None:
    run = FusionRun(
        query_id="q",
        gold_ids=("a",),
        rankings={"dense": (Candidate(id="a", score=1.0), Candidate(id="b", score=0.5))},
    )
    assert evaluate_fusion([run], RrfFuser(), k=10).ndcg_at_k == pytest.approx(1.0)


def test_ndcg_discounts_a_gold_hit_at_rank_two() -> None:
    run = FusionRun(
        query_id="q",
        gold_ids=("b",),
        rankings={"dense": (Candidate(id="a", score=1.0), Candidate(id="b", score=0.5))},
    )
    assert evaluate_fusion([run], RrfFuser(), k=10).ndcg_at_k == pytest.approx(1 / math.log2(3))


def test_a_query_with_no_gold_hit_scores_zero_not_an_error() -> None:
    run = FusionRun(
        query_id="q",
        gold_ids=("missing",),
        rankings={"dense": (Candidate(id="a", score=1.0),)},
    )
    score = evaluate_fusion([run], RrfFuser(), k=10)
    assert (score.recall_at_k, score.mrr, score.ndcg_at_k) == (0.0, 0.0, 0.0)


# --- the sweep ----------------------------------------------------------------------------------


def test_sweep_finds_weights_that_beat_the_uniform_combination() -> None:
    swept = sweep_fusion_weights(
        _RUNS,
        grid={"dense": (1.0, 2.0, 4.0), "sparse": (1.0,)},
        k=1,
        fuser=WeightedFuser(),
    )
    uniform = evaluate_fusion(_RUNS, WeightedFuser(), weights={"dense": 1.0, "sparse": 1.0}, k=1)
    assert uniform.recall_at_k == pytest.approx(0.0)
    assert swept.best.score.recall_at_k == pytest.approx(1.0)
    assert swept.best.weights["dense"] > swept.best.weights["sparse"]


def test_sweep_enumerates_the_whole_grid_in_deterministic_order() -> None:
    swept = sweep_fusion_weights(
        _RUNS, grid={"dense": (1.0, 2.0), "sparse": (1.0, 3.0)}, k=1, fuser=WeightedFuser()
    )
    assert len(swept.rows) == 4
    again = sweep_fusion_weights(
        _RUNS, grid={"dense": (1.0, 2.0), "sparse": (1.0, 3.0)}, k=1, fuser=WeightedFuser()
    )
    assert [row.model_dump() for row in swept.rows] == [row.model_dump() for row in again.rows]


def test_sweep_rows_are_ordered_best_first() -> None:
    swept = sweep_fusion_weights(
        _RUNS, grid={"dense": (1.0, 4.0), "sparse": (1.0,)}, k=1, fuser=WeightedFuser()
    )
    recalls = [row.score.recall_at_k for row in swept.rows]
    assert recalls == sorted(recalls, reverse=True)
    assert swept.best is swept.rows[0]


def test_sweep_refuses_a_grid_over_a_mode_no_run_carries() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        sweep_fusion_weights(_RUNS, grid={"nonsense": (1.0,)}, k=1)


def test_sweep_caps_the_combination_count() -> None:
    with pytest.raises(ValueError, match="combination"):
        sweep_fusion_weights(
            _RUNS,
            grid={"dense": tuple(float(i) for i in range(40)),
                  "sparse": tuple(float(i) for i in range(40))},
            k=1,
            max_combinations=100,
        )


def test_sweep_needs_a_non_empty_grid() -> None:
    with pytest.raises(ValueError, match="grid"):
        sweep_fusion_weights(_RUNS, grid={}, k=1)


def test_sweep_needs_labelled_runs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        sweep_fusion_weights([], grid={"dense": (1.0,)}, k=1)


# --- the comparison table the acceptance criterion asks for -------------------------------------


def test_compare_fusers_reports_both_on_the_same_questions() -> None:
    tuned = sweep_fusion_weights(
        _RUNS, grid={"dense": (1.0, 2.0, 4.0), "sparse": (1.0,)}, k=1, fuser=WeightedFuser()
    )
    rows = compare_fusers(
        _RUNS,
        {
            "rrf-k60": (RrfFuser(), None),
            "weighted-uniform": (WeightedFuser(), {"dense": 1.0, "sparse": 1.0}),
            "weighted-tuned": (WeightedFuser(), tuned.best.weights),
        },
        k=1,
    )
    assert [row.name for row in rows] == ["rrf-k60", "weighted-uniform", "weighted-tuned"]
    assert all(row.score.queries == 3 for row in rows)


def test_the_comparison_table_renders_deterministically() -> None:
    rows = compare_fusers(_RUNS, {"rrf-k60": (RrfFuser(), None)}, k=1)
    table = format_comparison_table(rows)
    assert "rrf-k60" in table
    assert "recall@1" in table
    assert format_comparison_table(rows) == table


def test_config_weights_reach_the_live_fusion_not_just_the_recipe() -> None:
    """The last hop: operator config → built pipeline → the ORDER `retrieve` actually returns.

    Before this ticket ``from_config`` raised ``NameError`` the moment ``fuse_weights`` was
    supplied (``Mapping`` was imported only under ``TYPE_CHECKING``) — so the config-driven weight
    path was dead and no test had ever walked it. This walks it end to end.
    """
    corpus = [
        Candidate(id="hit", score=0.90, text="alpha"),
        Candidate(id="other", score=0.85, text="beta"),
    ]
    backend = FakeBackend(corpus, capabilities={Capability.DENSE, Capability.BM25}, dimension=2)
    bundle = QueryBundle(text="alpha", vector=(0.1, 0.2))

    weighted = OrchestrationPipeline.from_config(
        {
            "fuser": {
                "provider": "fuser.weighted",
                "config": {"normalization": "theoretical_min_max", "bounds": [0.0, 1.0]},
            },
            "fuse_weights": {"dense": 3.0, "bm25": 1.0},
        }
    )
    result = retrieve(bundle, backend, weighted)
    assert result.fetched_modes == (Capability.BM25, Capability.DENSE)
    # theoretical bounds [0, 1] + weights 1 (bm25) and 3 (dense): 0.90*4 = 3.6 vs 0.85*4 = 3.4.
    assert [c.id for c in result.candidates] == ["hit", "other"]
    assert result.candidates[0].score == pytest.approx(3.6)


def test_rrf_remains_the_no_labels_default() -> None:
    """RRF is not deprecated: with no labels there is nothing to tune, so it stays the default."""
    assert isinstance(OrchestrationPipeline().fuser, RrfFuser)


# --- FusionRun is a validated contract ----------------------------------------------------------


def test_fusion_run_requires_gold_ids() -> None:
    with pytest.raises(ValueError, match="gold_ids"):
        FusionRun(query_id="q", gold_ids=(), rankings={"dense": ()})


def test_fusion_run_requires_at_least_one_ranking() -> None:
    with pytest.raises(ValueError, match="rankings"):
        FusionRun(query_id="q", gold_ids=("a",), rankings={})


def test_fusion_run_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        FusionRun(
            query_id="q", gold_ids=("a",), rankings={"dense": ()}, extra=1
        )  # type: ignore[call-arg]


def test_evaluate_aligns_weights_to_each_run_in_sorted_mode_order() -> None:
    """A weight keyed by mode name must reach the right list even when runs differ in shape."""
    only_sparse = FusionRun(
        query_id="q",
        gold_ids=("s",),
        rankings={"sparse": (Candidate(id="s", score=1.0),)},
    )
    both = FusionRun(
        query_id="r",
        gold_ids=("d",),
        rankings={
            "dense": (Candidate(id="d", score=1.0),),
            "sparse": (Candidate(id="s", score=1.0),),
        },
    )
    score = evaluate_fusion(
        [only_sparse, both], WeightedFuser(), weights={"dense": 10.0, "sparse": 1.0}, k=1
    )
    assert score.recall_at_k == pytest.approx(1.0)


def test_evaluate_ignores_a_weight_for_a_mode_a_run_lacks() -> None:
    run = FusionRun(
        query_id="q", gold_ids=("a",), rankings={"dense": (Candidate(id="a", score=1.0),)}
    )
    score = evaluate_fusion([run], WeightedFuser(), weights={"dense": 1.0, "graph": 5.0}, k=1)
    assert score.recall_at_k == pytest.approx(1.0)

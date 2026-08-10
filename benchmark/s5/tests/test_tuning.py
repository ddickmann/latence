# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The weighted-fusion variant: weight threading, the split, and that the fit never sees test.

The load-bearing test in here is :func:`test_the_tuner_never_sees_a_test_query`. Every other
property (the split is deterministic, the weight reaches the fuser, the default is unchanged) is
checkable by inspection; "the reported number was not fitted on the queries it is reported on" is
not, and it is the only thing that makes a tuned number worth printing at all. So it is tested
adversarially: dev and test are built to want OPPOSITE weights, and the fit is required to pick
dev's — which means the tuned test number comes out WORSE than the equal-weight one. A leaking
tuner would look better and fail this test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from benchmark.s5.corpus import ChunkRow, Export
from benchmark.s5.legs import Runner, SignalRanking, fuse_signals, leg_signals
from benchmark.s5.tuning import (
    DEV_TUNE_CAP,
    DevRun,
    ScoredQuery,
    TunedLeg,
    dev_objective,
    is_dev,
    is_tunable_leg,
    run_tuned_leg,
    split_queries,
    tune_weights,
)
from latence_retrieval.contracts import Candidate
from latence_retrieval.fuser import RrfFuser

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The candidate ids every scripted query ranks over. ``dense`` prefers A, the second signal
#: prefers B, and at equal RRF weight the second signal wins — which is what lets one weight
#: knob flip the top hit between two gold choices.
_DENSE_ORDER = ("A", "B", "C")
_OTHER_ORDER = ("B", "C", "A")


def _export() -> Export:
    rows = [ChunkRow(cid, f"text {cid}", "d.md", 0, 10, ()) for cid in _DENSE_ORDER]
    return Export(dataset="t", root=None, rows=rows, embeddings=None, pages={})  # type: ignore[arg-type]


def _ranking(name: str, ids: Sequence[str]) -> SignalRanking:
    return SignalRanking(
        name,
        [Candidate(id=cid, score=1.0 / (rank + 1), text=cid) for rank, cid in enumerate(ids)],
    )


class _ScriptedRunner(Runner):
    """A ``Runner`` whose retrieval is a fixed script, so fusion is the only moving part."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            export=_export(), dense=None, bm25=None, graph=None, kv_path=None, **kwargs
        )

    def signals(self, question: str, vector: Any, leg: str) -> list[SignalRanking]:
        names = leg_signals(leg)
        out = [_ranking(names[0], _DENSE_ORDER)]
        out += [_ranking(name, _OTHER_ORDER) for name in names[1:]]
        return out


class _SpyFuser(RrfFuser):
    """An ``RrfFuser`` that records the weight vector it was handed, then fuses normally."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_calls: list[list[float] | None] = []

    def fuse(
        self,
        rankings: Sequence[Sequence[Candidate]],
        weights: Sequence[float] | None = None,
    ) -> list[Candidate]:
        self.weight_calls.append(None if weights is None else [float(w) for w in weights])
        return super().fuse(rankings, weights)


# -- weight threading ---------------------------------------------------------------------------


def test_a_legs_weight_reaches_the_fuser_aligned_to_the_right_list() -> None:
    """A per-signal weight becomes the fuser's positional vector, in the list's own order."""
    spy = _SpyFuser()
    runner = _ScriptedRunner(weights={"bm25": 0.5}, fuser=spy)
    runner.run("q", None, "bm25")
    assert spy.weight_calls == [[1.0, 0.5]]


def test_a_named_weight_for_a_list_this_query_did_not_produce_is_simply_unused() -> None:
    """A weight is a preference, never a filter: naming an absent signal cannot delete a list."""
    spy = _SpyFuser()
    runner = _ScriptedRunner(weights={"coe": 0.1}, fuser=spy)
    runner.run("q", None, "bm25")
    assert spy.weight_calls == [[1.0, 1.0]]


def test_the_default_is_equal_weight_and_passes_no_vector_at_all() -> None:
    """The untuned ladder must make exactly the call it made before weights were reachable."""
    spy = _SpyFuser()
    _ScriptedRunner(fuser=spy).run("q", None, "coe")
    assert spy.weight_calls == [None]


def test_explicit_all_ones_fuses_identically_to_the_default() -> None:
    """Equal weights are equal weights — the default cannot be a different ranking in disguise."""
    default = _ScriptedRunner().run("q", None, "coe")
    ones = _ScriptedRunner(
        weights=dict.fromkeys(leg_signals("coe"), 1.0)
    ).run("q", None, "coe")
    assert default.ranked == ones.ranked
    assert default.ranked


def test_a_zero_weight_drops_a_lists_pull_without_dropping_its_candidates() -> None:
    """Weight 0.0 means "this list gets no vote", not "these documents never existed"."""
    runner = _ScriptedRunner(weights={"bm25": 0.0})
    assert runner.run("q", None, "bm25").ranked[0] == "A"  # dense's pick, unopposed
    assert set(runner.run("q", None, "bm25").ranked) == set(_DENSE_ORDER)
    assert _ScriptedRunner().run("q", None, "bm25").ranked[0] == "B"  # equal weight: bm25 wins


# -- the dev/test split -------------------------------------------------------------------------

#: A fixed id list with its side of the split spelled out. Hard-coded rather than recomputed, so
#: a change to the salt or the dev share fails HERE instead of silently re-drawing every split a
#: previously-written cell was fitted under.
_EXPECTED_DEV = {"q003", "q007", "q011", "q012", "q015"}
_ALL_IDS = tuple(f"q{i:03d}" for i in range(20))


def test_the_split_is_a_pinned_function_of_the_query_id_alone() -> None:
    assert {qid for qid in _ALL_IDS if is_dev(qid)} == _EXPECTED_DEV
    # …and the bucket really is the declared share, not a lopsided hash. Measured over 10k ids so
    # the assertion is about the hash, not about twenty lucky strings.
    wide = sum(is_dev(f"q{i:05d}") for i in range(10_000))
    assert 0.27 < wide / 10_000 < 0.33


def test_the_split_is_deterministic_and_disjoint_and_loses_nothing() -> None:
    first = split_queries(_ALL_IDS, key=str)
    second = split_queries(_ALL_IDS, key=str)
    assert first == second
    assert not set(first.dev) & set(first.test)
    assert set(first.dev) | set(first.test) == set(_ALL_IDS)
    assert len(first.dev) + len(first.test) == len(_ALL_IDS)


def test_a_querys_side_does_not_depend_on_what_was_sampled_alongside_it() -> None:
    """``--limit`` must not move a query across the split; the hash reads the id and nothing else."""
    whole = split_queries(_ALL_IDS, key=str)
    subset = split_queries(_ALL_IDS[::3], key=str)
    assert set(subset.dev) <= set(whole.dev)
    assert set(subset.test) <= set(whole.test)


def test_the_split_record_states_the_sizes_and_how_to_re_derive_it() -> None:
    described = split_queries(_ALL_IDS, key=str).describe(tuned_on=5)
    assert described["dev_queries"] == len(_EXPECTED_DEV)
    assert described["test_queries"] == len(_ALL_IDS) - len(_EXPECTED_DEV)
    assert described["dev_queries_tuned_on"] == 5
    assert described["dev_tune_cap"] == DEV_TUNE_CAP
    assert "blake2b" in str(described["scheme"])
    assert described["salt"]


# -- the fit ------------------------------------------------------------------------------------


def _queries(gold_of: Mapping[str, str]) -> list[ScoredQuery]:
    return [
        ScoredQuery(query_id=qid, question=qid, vector=None, gold=frozenset({gold}))
        for qid, gold in gold_of.items()
    ]


def _split_by_side(gold_dev: str, gold_test: str) -> list[ScoredQuery]:
    """40 queries whose gold depends only on which side of the split the id landed on."""
    ids = [f"q{i:03d}" for i in range(40)]
    return _queries({qid: (gold_dev if is_dev(qid) else gold_test) for qid in ids})


def test_the_tuner_never_sees_a_test_query() -> None:
    """Dev and test are built to want OPPOSITE weights; the fit must pick dev's and lose on test.

    Dev queries want ``A`` (which only the dense list ranks first, so the second signal must be
    silenced); test queries want ``B`` (which equal-weight fusion already puts first). A fit that
    peeked at test would keep weight 1.0 and score *better* on test. Requiring it to score WORSE
    is the assertion no leaking implementation can satisfy.
    """
    runner = _ScriptedRunner()
    result = run_tuned_leg(runner, "bm25", _split_by_side(gold_dev="A", gold_test="B"))
    assert isinstance(result, TunedLeg)

    assert result.tuning.weights == {"dense": 1.0, "bm25": 0.0}
    assert not set(result.dev_query_ids) & set(result.test_query_ids)
    assert set(result.tuning.dev_query_ids) <= set(result.dev_query_ids)
    # The honest cost of not peeking, stated as a number rather than trusted.
    tuned, equal = result.tuned["ndcg@10"], result.equal["ndcg@10"]
    assert tuned is not None and equal is not None
    assert tuned < equal


def test_the_fit_improves_the_held_out_number_when_dev_and_test_agree() -> None:
    """The other half of the same experiment: when the fit generalises, test moves with it."""
    result = run_tuned_leg(_ScriptedRunner(), "bm25", _split_by_side("A", "A"))
    assert isinstance(result, TunedLeg)
    assert result.tuning.weights["bm25"] == 0.0
    tuned, equal = result.tuned["ndcg@10"], result.equal["ndcg@10"]
    assert tuned is not None and equal is not None
    assert tuned > equal
    # …and both variants were scored on exactly the same held-out queries.
    assert result.tuned["queries"] == result.equal["queries"] == len(result.test_query_ids)


def test_both_variants_are_scored_on_the_test_half_only() -> None:
    result = run_tuned_leg(_ScriptedRunner(), "bm25", _split_by_side("A", "B"))
    assert isinstance(result, TunedLeg)
    assert result.split["test_queries"] == len(result.test_query_ids)
    assert result.tuned["queries"] == len(result.test_query_ids)
    assert result.tuned["queries"] != result.split["dev_queries"]


def test_a_tie_on_dev_keeps_the_equal_weight_incumbent() -> None:
    """The fit may only displace the baseline on a STRICT improvement, never on noise."""
    runner = _ScriptedRunner()
    # Gold ``B`` is already rank 1 under equal weights, so no weight can beat 1.0 — only tie it.
    result = tune_weights(runner, "bm25", _dev_runs(runner, "bm25", _queries({"q1": "B"})))
    assert result.weights == {"dense": 1.0, "bm25": 1.0}
    assert result.dev_score_best == pytest.approx(result.dev_score_equal)


def _dev_runs(runner: Runner, leg: str, queries: Sequence[ScoredQuery]) -> list[DevRun]:
    return [
        DevRun(
            query_id=q.query_id,
            rankings=tuple(runner.signals(q.question, q.vector, leg)),
            gold=q.gold,
        )
        for q in queries
    ]


def test_a_single_signal_leg_has_nothing_to_fit_and_says_so() -> None:
    """One list means the RRF weight is a pure monotone scale — a fit over nothing, labelled."""
    runner = _ScriptedRunner()
    result = tune_weights(runner, "dense", _dev_runs(runner, "dense", _queries({"q1": "A"})))
    assert result.weights == {"dense": 1.0}
    assert result.tunable == ()
    assert result.pinned is None
    assert result.passes == 0
    assert result.evaluations == 1
    assert result.converged is True
    assert is_tunable_leg("dense") is False
    assert is_tunable_leg("dense_content_only") is False
    assert is_tunable_leg("coe") is True


def test_an_empty_half_is_a_refusal_not_a_same_split_number() -> None:
    only_dev = [q for q in _split_by_side("A", "A") if is_dev(q.query_id)]
    refusal = run_tuned_leg(_ScriptedRunner(), "bm25", only_dev)
    assert not isinstance(refusal, TunedLeg)
    assert refusal.to_dict()["status"] == "insufficient-split"
    assert refusal.to_dict()["test_queries"] == 0


def test_the_objective_refuses_an_empty_fitting_set() -> None:
    with pytest.raises(ValueError, match="at least one dev run"):
        dev_objective(_ScriptedRunner(), [], None)


def test_the_objective_is_the_reported_ndcg_definition() -> None:
    """The number climbed and the number printed must be the same quantity, not two cousins."""
    runner = _ScriptedRunner()
    runs = _dev_runs(runner, "bm25", _queries({"q1": "A"}))
    silenced = dev_objective(runner, runs, {"dense": 1.0, "bm25": 0.0})
    equal = dev_objective(runner, runs, None)
    assert silenced == pytest.approx(1.0)  # gold at rank 1
    assert equal == pytest.approx(1.0 / 1.5849625007211563)  # gold at rank 2
    assert runs[0].rankings[0].name == "dense"


def test_fuse_signals_with_no_lists_fuses_to_nothing() -> None:
    assert fuse_signals(RrfFuser(), [], {"dense": 2.0}) == []


# -- the CLI switch -----------------------------------------------------------------------------


def test_the_cli_defaults_to_equal_fusion_and_the_baseline_results_directory() -> None:
    """The in-flight matrix run must not change behaviour because a second variant now exists."""
    from benchmark.s5 import run_matrix

    plan = run_matrix.parse_args(["--datasets", "vidoseek", "--legs", "dense"])
    assert plan.fusion == "equal"
    assert plan.results_dir.name == "results"


def test_the_tuned_variant_writes_its_own_namespace() -> None:
    from benchmark.s5 import run_matrix

    plan = run_matrix.parse_args(
        ["--datasets", "vidoseek", "--legs", "dense", "--fusion", "tuned"]
    )
    assert plan.fusion == "tuned"
    assert plan.results_dir.name == "results-weighted"
    assert plan.results_dir != run_matrix.parse_args(["--datasets", "vidoseek"]).results_dir


def test_an_unknown_fusion_variant_is_refused() -> None:
    from benchmark.s5 import run_matrix

    with pytest.raises(SystemExit):
        run_matrix.parse_args(["--fusion", "guesswork"])

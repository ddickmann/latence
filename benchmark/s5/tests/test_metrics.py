# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The metric definitions — set-gold recall, capped-ideal nDCG, chain coverage."""

from __future__ import annotations

import math

from benchmark.s5.metrics import (
    CellAccumulator,
    chain_metrics,
    ndcg_at_k,
    percentile,
    recall_at_k,
)


def test_recall_is_set_recall_not_hit_rate() -> None:
    """Retrieving one of a page's three gold chunks is 1/3, not 1.0."""
    gold = frozenset({"a", "b", "c"})
    assert recall_at_k(["a", "x", "y"], gold, 5) == 1 / 3
    assert recall_at_k(["a", "b", "c"], gold, 5) == 1.0
    assert recall_at_k(["x", "a"], gold, 1) == 0.0


def test_recall_of_empty_gold_is_zero_not_one() -> None:
    assert recall_at_k(["a"], frozenset(), 10) == 0.0


def test_ndcg_ideal_is_capped_at_k_so_large_gold_is_not_penalised() -> None:
    gold = frozenset(str(i) for i in range(30))
    perfect = [str(i) for i in range(10)]
    assert ndcg_at_k(perfect, gold) == 1.0


def test_ndcg_rewards_earlier_placement() -> None:
    gold = frozenset({"g"})
    assert ndcg_at_k(["g", "x"], gold) == 1.0
    assert math.isclose(ndcg_at_k(["x", "g"], gold), 1 / math.log2(3))
    assert ndcg_at_k(["x"] * 20 + ["g"], gold) == 0.0


def test_chain_metrics_need_at_least_two_slots() -> None:
    assert chain_metrics(["a"], [frozenset({"a"})]) is None
    chain = [frozenset({"a"}), frozenset({"b"})]
    assert chain_metrics(["a", "z"], chain) == (0.5, 0.0)
    assert chain_metrics(["a", "b"], chain) == (1.0, 1.0)


def test_full_chain_hit_needs_every_slot_inside_k() -> None:
    chain = [frozenset({"a"}), frozenset({"b"})]
    ranked = ["a", *[f"x{i}" for i in range(20)], "b"]
    coverage, full = chain_metrics(ranked, chain, k=10) or (None, None)
    assert (coverage, full) == (0.5, 0.0)


def test_percentile_is_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == 2.0
    assert percentile(values, 0.95) == 4.0
    assert percentile([], 0.5) == 0.0


def test_accumulator_summary_reports_chain_columns_only_when_chains_exist() -> None:
    cell = CellAccumulator()
    cell.add(["a"], frozenset({"a"}), [], 1.5)
    summary = cell.summary()
    assert summary["recall@1"] == 1.0
    assert summary["chain_coverage@10"] is None
    assert summary["chain_queries"] == 0

    chained = CellAccumulator()
    chained.add(["a", "b"], frozenset({"a", "b"}), [frozenset({"a"}), frozenset({"b"})], 2.0)
    assert chained.summary()["full_chain_hit@10"] == 1.0


def test_accumulator_records_packed_tokens_against_the_budget() -> None:
    cell = CellAccumulator()
    cell.add(["a"], frozenset({"a"}), [], 1.0, tokens_packed=3200, budget=4000)
    summary = cell.summary()
    assert summary["tokens_packed_max"] == 3200
    assert summary["token_budget"] == 4000

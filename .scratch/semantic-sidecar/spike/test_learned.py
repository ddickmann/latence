"""The scorer fitted on reviewer labels: its evidence, its label samples, its certified lines,
and the case it was built for: two registers whose shared addresses mislead the first stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from learned import certify, draw_training, fit_score
from pairfeat import org_features, pattern_strata, rank_features, squeezed, three_way
from run_a import normalise_org, run


def test_a_missing_value_is_never_a_difference() -> None:
    out = three_way(["a", "a", None, "b"], np.array([0, 0, 0]), np.array([1, 2, 3]))
    assert out.tolist() == [1.0, 0.0, -1.0]
    assert squeezed("Shedden Farms Ltd.") == squeezed("SHEDDEN FARMS LTD") and squeezed(" ") is None


def test_rank_features_say_which_pair_is_a_records_best_and_by_how_much() -> None:
    left, right, score = np.array([0, 0, 1, 2]), np.array([1, 2, 2, 3]), np.array([0.9, 0.2, 0.5, 0.5])
    R, names = rank_features(4, left, right, score)
    assert names == ["best_for_both", "margin_min", "best_for_one", "margin_max"]
    assert R[0].round(2).tolist() == [1.0, 0.4, 1.0, 0.7]  # best for both records, ahead by 0.4 and 0.7
    assert R[1].round(2).tolist() == [0.0, -0.7, 0.0, -0.3]  # second best on both sides
    assert R[3].round(2).tolist() == [1.0, 0.0, 1.0, 1.0]  # a tie on one side, the only pair on the other


def test_the_evidence_of_two_spellings_of_one_company() -> None:
    raw = pd.DataFrame({
        "unique_id": ["a1", "b1", "b2"], "source_system": ["a", "b", "b"], "entity_type": ["organization"] * 3,
        "name": ["SHEDDEN FARMS LIMITED", "Shedden Farms Ltd", "Shedden Estates Ltd"], "city": ["York", "YORK", "York"],
        "postcode": ["YO30 1AP", "yo301ap", None], "country": ["GB"] * 3, "industry": [None] * 3, "reg_id": [None] * 3, "cluster": ["1", "1", "2"],
    })
    rec = normalise_org(raw)
    X, names = org_features(rec, np.array([0, 0, 1]), np.array([1, 2, 2]))
    f = {n: X[:, k].tolist() for k, n in enumerate(names)}
    assert f["core_same"] == [1.0, 0.0, 0.0] and f["name_same"] == [0.0, 0.0, 0.0]  # "LIMITED" and "Ltd" are one legal form, not one string
    assert f["legal"] == [1.0, 1.0, 1.0] and f["same_source"] == [0.0, 0.0, 1.0]
    assert f["postcode"] == [1.0, 0.0, 0.0]  # spelled differently, the same postcode; a missing one is no difference
    assert f["token_share"][0] == 1.0 and 0 < f["token_share"][1] < 1


def test_training_labels_are_spread_over_the_patterns_of_evidence() -> None:
    strata = np.array([0] * 1000 + [1] * 5 + [2] * 40)
    idx, w = draw_training(strata, 60, np.random.default_rng(0))
    assert len(idx) == len(set(idx.tolist())) == 60
    assert np.bincount(strata[idx]).tolist() == [28, 5, 27]  # the small stratum whole, the rest shared
    assert np.isclose(w[strata[idx] == 0][0], 1000 / 28) and np.isclose(w[strata[idx] == 1][0], 1.0)
    X = np.column_stack([np.r_[np.zeros(1000), np.ones(45)], np.r_[np.zeros(1005), np.ones(40)]])
    assert pattern_strata(X, ["a", "b"], by=("a", "b")).max() == 2


def test_lines_are_certified_on_fresh_labels_on_their_own_side_of_the_floor() -> None:
    rng = np.random.default_rng(1)
    score = np.r_[rng.uniform(0.9, 1.0, 3000), rng.uniform(0.5, 0.6, 2000), rng.uniform(0.0, 0.1, 20000)]
    y = np.r_[np.ones(3000), rng.random(2000) < 0.5, np.zeros(20000)].astype(int)  # the middle is a coin toss
    used = np.arange(0, 25000, 50)
    c = certify(score, y, used, 600, 2000, np.random.default_rng(2))
    assert c.accept_at is not None and 0.5 < c.accept_at <= 0.9 and c.accept_at >= c.floor  # takes a few coin-toss pairs, never all of them
    assert y[score >= c.accept_at].mean() >= 0.95
    rejected = score < c.reject_below if c.reject_below is not None else np.zeros(len(score), bool)
    assert 19000 <= rejected.sum() <= 20600 and y[rejected].mean() <= 0.01  # the clear non-matches, and hardly a coin-toss pair
    few = certify(score, y, used, 600, 250, np.random.default_rng(2))
    assert few.reject_below is None  # 250 labels cannot carry a 0.99 claim even without one error: no line, not a wrong one
    assert not set(c.accept_test.tolist()) & set(used.tolist()) and not set(c.reject_test.tolist()) & set(used.tolist())
    assert (score[c.accept_test] >= c.floor).all() and (score[c.reject_test] < c.floor).all()
    hopeless = certify(rng.uniform(0.5, 1.0, 5000), (rng.random(5000) < 0.5).astype(int), np.array([], dtype=int), 600, 0, np.random.default_rng(3))
    assert hopeless.accept_at is None  # a scorer that knows nothing certifies nothing


def test_a_reject_line_is_found_among_scores_that_are_all_close_to_zero() -> None:
    # Measured on the synthetic estate (2026-11-11): the fitted scorer puts nearly every
    # non-match below 0.005, the first step of a grid in steps of 0.005. The few true matches
    # among them failed that one step, no reject line was certified and 287,000 pairs went to
    # review. The lines are now tried at quantiles of the scores, which reads no label.
    rng = np.random.default_rng(5)
    low, mid = rng.uniform(0.0, 0.001, 20000), rng.uniform(0.001, 0.005, 20000)
    score = np.r_[low, mid, rng.uniform(0.9, 1.0, 3000)]
    y = np.r_[np.zeros(20000), rng.random(20000) < 0.03, np.ones(3000)].astype(int)
    c = certify(score, y, np.array([], dtype=int), 600, 1000, np.random.default_rng(6))
    assert c.reject_below is not None
    rejected = score < c.reject_below
    assert rejected.sum() >= 15000 and y[rejected].mean() <= 0.01


def test_a_few_wrong_pairs_at_the_very_top_do_not_end_the_accept_test() -> None:
    # Measured at 10^6 records (2026-11-11): the best three quarters of the pairs above even
    # odds were 99.8% right and no accept line was certified. The sequence began on 100
    # labelled pairs, where three wrong ones fail the step, and a failed first step ends it.
    score = np.r_[np.linspace(0.99, 0.6, 2000), np.linspace(0.1, 0.0, 3000)]
    y = np.r_[np.ones(2000), np.zeros(3000)].astype(int)
    y[np.arange(0, 200, 30)] = 0  # seven wrong pairs among the 200 best, none after
    c = certify(score, y, np.array([], dtype=int), 2000, 1000, np.random.default_rng(4))  # every pair above the floor is labelled
    assert c.accept_at is not None and c.accept_at <= 0.61
    assert y[score >= c.accept_at].mean() >= 0.95


def test_no_pair_is_accepted_that_the_scorer_itself_rates_below_even_odds() -> None:
    # Measured (2026-11-11): with a floor that followed the scorer down, the accept line sank to
    # 0.001 and took in pairs that were right one time in three, because the set as a whole
    # still met 0.95. Certified, and not what anyone means by accepting.
    rng = np.random.default_rng(7)
    score = np.r_[np.full(6000, 0.99), rng.uniform(0.01, 0.4, 300), rng.uniform(0.0, 0.01, 10000)]
    y = np.r_[np.ones(6000), rng.random(300) < 0.35, np.zeros(10000)].astype(int)
    c = certify(score, y, np.array([], dtype=int), 1500, 1000, np.random.default_rng(8))
    assert c.floor == 0.5 and c.accept_at is not None and c.accept_at >= 0.5
    assert c.reject_below is not None and c.reject_below <= 0.5


def test_the_fitted_scores_are_probabilities_not_verdicts() -> None:
    # The cluster check reads the scores as probabilities (it blocks a merge across a pair rated
    # under 0.05), so a coin toss must not be rated as a certainty. This held before the scores
    # were calibrated across folds too; it is kept as a property, not as the test of that change.
    rng = np.random.default_rng(0)
    n = 6000
    signal = rng.integers(0, 3, n)  # match rates 0.1, 0.5 and 0.9
    y = (rng.random(n) < np.array([0.1, 0.5, 0.9])[signal]).astype(int)
    X = np.column_stack([signal, rng.normal(size=(n, 10))])  # ten columns of noise to overfit on
    idx = rng.permutation(n)[:800]
    fresh = np.ones(n, bool)
    fresh[idx] = False
    s = fit_score(X, y, idx, None)
    middle = s[fresh & (signal == 1)]
    assert 0.35 <= middle.mean() <= 0.65
    assert ((middle < 0.05) | (middle > 0.95)).mean() < 0.02  # a coin toss is not rated as a certainty
    assert s[fresh & (signal == 2)].mean() > 0.8 and s[fresh & (signal == 0)].mean() < 0.2


def test_a_scorer_fitted_on_labels_learns_which_evidence_counts() -> None:
    rng = np.random.default_rng(0)
    name, addr = rng.random(4000) < 0.3, rng.random(4000) < 0.5
    X, y = np.column_stack([name, addr]).astype(float), name.astype(int)  # the address says nothing
    idx = rng.permutation(4000)[:300]
    for model in ("gbt", "logit"):
        s = fit_score(X, y, idx, None, model)
        assert s[name].min() > 0.9 and s[~name].max() < 0.1


def two_registers(n: int, seed: int) -> pd.DataFrame:
    """Register A and register B keep the same companies under the same names, spelled their
    own way, and often at another address. B also holds many other companies, most of them at
    a handful of addresses that thousands share (formation agents), with names that begin like
    a company in A. What the two registers agree on is the name; what most pairs agree on is
    the address."""
    rng = np.random.default_rng(seed)
    first = [f"{a}{b}" for a in ("Ark", "Bel", "Cor", "Dun", "Esk", "Fen", "Gar", "Hol", "Ivo", "Jur", "Kel", "Lom") for b in ("wood", "field", "stone", "mere", "ford", "wick", "ley", "ton", "by", "ham")]
    second = ["Holdings", "Capital", "Estates", "Trading", "Partners", "Property", "Systems", "Foods", "Energy", "Logistics", "Finance", "Marine", "Homes", "Media", "Health"]
    agents = [f"EC{k} {k}AA" for k in range(1, 6)]
    rows = []
    for i in range(n):
        name = f"{first[i % len(first)]} {second[(i // len(first)) % len(second)]} {i // (len(first) * len(second)) + 1}".replace(" 1", "")
        home = f"LS{rng.integers(1, 30)} {rng.integers(1, 9)}{'ABDEFGHJ'[rng.integers(0, 8)]}{'LNPQRSTU'[rng.integers(0, 8)]}"
        pc_a = agents[rng.integers(0, 5)] if rng.random() < 0.4 else home
        rows.append((f"a:{i}", "reg_a", f"{name} LIMITED".upper(), "London" if pc_a in agents else "Leeds", pc_a, str(i)))
        if rng.random() < 0.8:
            pc_b = pc_a if rng.random() < 0.6 else (agents[rng.integers(0, 5)] if rng.random() < 0.5 else home)
            spelled = [f"{name} Ltd", f"{name} Limited", f"{name} LIMITED".upper(), f"{name} Ltd."][rng.integers(0, 4)]
            rows.append((f"b:{i}", "reg_b", spelled, "LONDON" if pc_b in agents else "LEEDS", pc_b, str(i)))
    for j in range(3 * n):
        name = f"{first[rng.integers(0, len(first))]} {second[rng.integers(0, len(second))]} {'Group' if rng.random() < 0.5 else 'UK'} {j}"
        rows.append((f"b:x{j}", "reg_b", f"{name} Ltd", "LONDON", agents[rng.integers(0, 5)], f"x{j}"))
    out = pd.DataFrame(rows, columns=["unique_id", "source_system", "name", "city", "postcode", "cluster"])
    out["entity_type"], out["country"], out["industry"], out["reg_id"] = "organization", "GB", None, None
    return out


def test_two_registers_with_shared_addresses_are_linked_once_the_labels_say_what_counts(tmp_path: Path) -> None:
    # Measured on two real company registers (2026-11-11): the first stage alone certified
    # nothing and, with a neural second stage, accepted 36% of the true links. The scorer
    # fitted on the same number of labels accepted 97%.
    path = tmp_path / "registers.parquet"
    two_registers(1200, seed=3).to_parquet(path, index=False)
    res = run("estate", path, 500, 900, 7, baseline_max=0, learned=True)
    info, mo = res["bands"]["learned"], res["clustering"]["machine_only"]
    assert info["labels_used"]["training"] == 500 and info["labels_used"]["accept_test"] + info["labels_used"]["reject_test"] <= 900
    assert res["labels_used"] == {"calibration": 500, "ltt": info["labels_used"]["accept_test"] + info["labels_used"]["reject_test"]}
    assert res["bands"]["accept_at"] is not None and res["bands"]["accept_at"] >= info["floor"]
    assert res["bands"]["auto_accept_precision_realised"] >= 0.95
    assert mo["pairwise_precision"] >= 0.95 and mo["pairwise_recall"] >= 0.9
    assert sum(res["bands"]["counts"].values()) == res["blocking"]["_union"]["pairs"]
    assert "name_same" in info["features"] and "first_stage_logit" in info["features"]
    assert info["accept_trace"] and {"threshold", "n", "errors", "p_value"} <= set(info["accept_trace"][0])  # how close each line came
    before = run("estate", path, 500, 900, 7, baseline_max=0)  # the same labels, spent on calibrating the first stage
    assert before["clustering"]["machine_only"]["pairwise_recall"] < 0.5


def test_without_the_learned_stage_nothing_changes(tmp_path: Path) -> None:
    path = tmp_path / "registers.parquet"
    two_registers(600, seed=4).to_parquet(path, index=False)
    res = run("estate", path, 400, 600, 7, baseline_max=0)
    assert res["bands"]["learned"] is None

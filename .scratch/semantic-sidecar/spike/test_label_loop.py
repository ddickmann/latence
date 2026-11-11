"""The matcher asks for the labels it needs, in two rounds, and takes them from a reviewer
(ticket 30). The truth column of the test estate plays the reviewer where a test says so."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from link_records import link
from test_learned import namesakes


@pytest.fixture(scope="module")
def estate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    path = tmp_path_factory.mktemp("loop") / "namesakes.parquet"
    df = namesakes(300, seed=2)
    df.to_parquet(path, index=False)
    return path, dict(zip(df.unique_id.tolist(), df.cluster.astype(str).tolist()))


def _answer(pairs: list[dict[str, Any]], truth: dict[str, str]) -> dict[tuple[str, str], int]:
    """What a careful reviewer would say about the pairs asked for."""
    return {(p["a"], p["b"]): int(truth[p["a"]] == truth[p["b"]]) for p in pairs}


def _run(path: Path, labels: dict[tuple[str, str], int] | None, truth_stands_in: bool) -> dict[str, Any]:
    return link(path, 300, 600, duplicate_free=("master",), labels=labels, truth_stands_in=truth_stands_in)


def test_with_no_label_and_no_truth_to_stand_in_nothing_is_accepted_and_the_pairs_to_learn_from_are_asked_for(estate: tuple[Path, dict[str, str]]) -> None:
    path, _ = estate
    out = _run(path, {}, truth_stands_in=False)
    assert out["state"] == "awaiting_training" and out["certified"] is False and out["certified_on"] is None
    assert out["accepted"] == [] and out["counts"]["accept"] == 0
    asked = out["requests"]
    assert len(asked["training"]) == 300 and asked["accept_test"] == [] and asked["reject_test"] == []
    assert out["labels"]["training"] == {"reviewer": 0, "simulated": 0, "open": 300}
    one = asked["training"][0]
    assert set(one) == {"id", "a", "b", "evidence", "label", "by"} and one["label"] is None  # no score: the reviewer is not led
    assert one["id"] == f"same:{one['a']}~{one['b']}" and one["a"].startswith("m:")  # the master record first, as review items name a pair


def test_the_tests_are_drawn_once_the_training_pairs_are_answered_and_a_line_is_certified_once_they_are(estate: tuple[Path, dict[str, str]]) -> None:
    path, truth = estate
    first = _run(path, {}, truth_stands_in=False)
    labels = _answer(first["requests"]["training"], truth)
    second = _run(path, labels, truth_stands_in=False)
    assert second["state"] == "awaiting_tests" and second["accepted"] == []
    tests = second["requests"]["accept_test"] + second["requests"]["reject_test"]
    assert len(second["requests"]["accept_test"]) > 100 and len(second["requests"]["reject_test"]) > 100
    assert all(t["label"] is None and "p" not in t for t in tests)
    assert not {t["id"] for t in tests} & {t["id"] for t in second["requests"]["training"]}  # a pair the scorer learned from never tests it
    again = _run(path, labels, truth_stands_in=False)
    assert again["requests"] == second["requests"]  # the same answers give the same samples
    labels.update(_answer(tests, truth))
    third = _run(path, labels, truth_stands_in=False)
    assert third["state"] == "certified" and third["certified"] is True and third["certified_on"] == "reviewer"
    assert third["labels"]["training"] == {"reviewer": 300, "simulated": 0, "open": 0}
    assert third["labels"]["accept_test"]["open"] == 0 and third["labels"]["accept_test"]["simulated"] == 0
    assert len(third["accepted"]) > 300 and third["realised"]["accept_precision"] >= 0.95
    assert [(t["label"], t["by"]) for t in third["requests"]["training"][:2]] == [(("same", "different")[1 - labels[(t["a"], t["b"])]], "reviewer") for t in third["requests"]["training"][:2]]


def test_where_the_truth_may_stand_in_the_summary_says_how_many_labels_came_from_a_person(estate: tuple[Path, dict[str, str]]) -> None:
    path, truth = estate
    simulated = _run(path, None, truth_stands_in=True)
    assert simulated["state"] == "certified" and simulated["certified_on"] == "simulated"
    some = _answer(simulated["requests"]["training"][:40], truth)
    mixed = _run(path, some, truth_stands_in=True)
    assert mixed["labels"]["training"] == {"reviewer": 40, "simulated": 260, "open": 0} and mixed["certified_on"] == "mixed"
    assert mixed["accepted"] == simulated["accepted"]  # the reviewer agreed with the truth: nothing changes


def test_a_reviewer_who_says_otherwise_changes_what_the_scorer_learns(estate: tuple[Path, dict[str, str]]) -> None:
    path, truth = estate
    simulated = _run(path, None, truth_stands_in=True)
    contrary = {pair: 1 - said for pair, said in _answer(simulated["requests"]["training"], truth).items()}
    out = _run(path, contrary, truth_stands_in=True)
    assert out["labels"]["training"]["reviewer"] == 300
    assert {(d["a"], d["b"]) for d in out["accepted"]} != {(d["a"], d["b"]) for d in simulated["accepted"]}
    # It learned the opposite: what it now accepts, if anything, is wrong by the truth.
    assert out["accepted"] == [] or out["realised"]["accept_precision"] < 0.5

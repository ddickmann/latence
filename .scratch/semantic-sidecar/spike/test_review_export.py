"""The review export carries what a reviewer needs, and its numbers agree with the run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate import generate
from run_a import run


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    tmp = tmp_path_factory.mktemp("export")
    path = tmp / "estate.parquet"
    generate(1500, seed=5).to_parquet(path, index=False)
    out = tmp / "review.json"
    result = run("estate", path, 800, 1200, 7, baseline_max=0, export_review=out, export_groups=40)
    return json.loads(out.read_text()), result


def test_export_matches_the_run_it_came_from(export: tuple[dict, dict]) -> None:
    data, result = export
    assert data["schema"] == "review-export/1" and data["meta"]["synthetic"] is True
    assert data["meta"]["counts"] == result["bands"]["counts"]
    assert data["meta"]["review_groups_total"] == result["review_effort"]["review_groups"]
    assert len(data["groups"]) == min(40, data["meta"]["review_groups_total"])


def test_every_item_is_a_review_pair_with_field_evidence(export: tuple[dict, dict]) -> None:
    data, _ = export
    comparisons = {c["comparison"] for c in data["meta"]["comparisons"]}
    by_id = {i["id"]: i for i in data["items"]}
    assert len(by_id) == len(data["items"]) > 0
    for it in data["items"]:
        assert it["band"] == "review" and it["group"] is not None
        assert {e["comparison"] for e in it["evidence"]} == comparisons
        # The parts add up to the whole: prior plus the per-field weights is the match weight.
        assert it["match_weight"] == pytest.approx(it["prior_weight"] + sum(e["weight"] for e in it["evidence"]), abs=0.02)
        assert it["left"]["unique_id"] != it["right"]["unique_id"]
        assert it["why_unsure"].startswith("Calibrated probability")
        said = it["why_unsure"].split(". ", 1)[1]
        assert said[0].isupper() and said.endswith(".")
    for g in data["groups"]:
        assert set(g["items"]) <= set(by_id)
        members = set(g["records"])
        assert all(e["left"] in members and e["right"] in members for e in g["edges"])
        assert set(g["labels"]) == members


def test_groups_are_ranked_and_the_audit_sample_is_accepted_pairs(export: tuple[dict, dict]) -> None:
    data, _ = export
    priorities = [g["priority"] for g in data["groups"]]
    assert priorities == sorted(priorities, reverse=True)
    assert [g["rank"] for g in data["groups"]] == list(range(len(data["groups"])))
    assert data["audit"] and all(a["band"] == "accept" and a["group"] is None for a in data["audit"])


def test_missing_values_are_null_not_the_string_nan(export: tuple[dict, dict]) -> None:
    data, _ = export
    text = json.dumps(data)
    assert "NaN" not in text and '"nan"' not in text

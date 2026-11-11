"""End to end on a small generated estate: the second-stage arm with a fake pair scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from generate import generate
from run_a import run


class JaccardScorer:
    """Stands in for a neural pair scorer: token overlap of the two record texts. It can
    disagree with Splink (it sees the raw text, typos and all), which is what a fake must do."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.calls += len(pairs)
        out = []
        for a, b in pairs:
            ta, tb = set(a.lower().split()), set(b.lower().split())
            out.append(len(ta & tb) / max(1, len(ta | tb)))
        return np.array(out)


def test_second_stage_only_touches_uncertain_pairs(tmp_path: Path) -> None:
    path = tmp_path / "estate.parquet"
    generate(1500, seed=5).to_parquet(path, index=False)
    base = run("estate", path, 800, 1200, 7, baseline_max=0)
    fake = JaccardScorer()
    res = run("estate", path, 800, 1200, 7, baseline_max=0, rescore=["rerank"], n_cal2=150, n_ltt2=250, scorers={"rerank": fake})

    s2 = res["bands"]["second_stage"]
    assert base["bands"]["second_stage"] is None
    assert res["bands"]["delta"] == 0.025 and s2["delta_per_stage"] == 0.025  # the 0.05 is shared
    assert fake.calls == s2["uncertain_pairs"] > 0  # scored exactly the uncertain pairs, once
    assert sum(s2["moved"].values()) == s2["uncertain_pairs"]
    counts = res["bands"]["counts"]
    assert sum(counts.values()) == res["blocking"]["_union"]["pairs"]
    assert s2["labels_used"]["calibration"] + s2["labels_used"]["ltt"] <= s2["uncertain_pairs"]
    assert set(s2["features"]) >= {"splink_logit", "rerank"}


@pytest.mark.parametrize("kind,word", [("org", "organisation"), ("person", "person")])
def test_the_scorers_are_asked_about_the_kind_of_record_at_hand(kind: str, word: str) -> None:
    # Measured on the pod (2026-11-11): the public person dataset was scored with the
    # organisation question, and neither scorer certified anything.
    from run_a import prompts_for

    question, instruction = prompts_for(kind)
    assert word in question and word in instruction
    other = "person" if word == "organisation" else "organisation"
    assert other not in question and other not in instruction

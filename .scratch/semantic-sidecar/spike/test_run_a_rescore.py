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


# ---- the pass below the reject line (2026-11-11) ---------------------------------------------


class TruthScorer:
    """A scorer that knows which records are the same organisation (from the generated estate):
    the best a model could do, so what the stage does with a perfect signal can be pinned."""

    def __init__(self, path: Path) -> None:
        import pandas as pd

        from rescore import ORG_TEXT_COLS, record_text

        raw = pd.read_parquet(path)
        self.cluster = {record_text(r, ORG_TEXT_COLS): r["cluster"] for r in raw.to_dict("records")}

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array([1.0 if self.cluster[a] == self.cluster[b] else 0.0 for a, b in pairs])


class BlindScorer:
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.full(len(pairs), 0.5)


@pytest.fixture(scope="module")
def estate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rescue") / "estate.parquet"
    generate(2500, seed=5).to_parquet(path, index=False)
    return path


def test_the_rescue_band_is_the_k_best_rejected_pairs_without_a_veto() -> None:
    from rescore import rescue_region

    p = np.array([0.90, 0.30, 0.20, 0.25, 0.10, 0.28, 0.05])
    rej = np.array([False, True, True, True, True, True, True])
    veto = np.array([False, False, False, False, False, True, False])
    assert rescue_region(p, rej, veto, 3).tolist() == [1, 3, 2]  # 0.30, 0.25, 0.20; the veto at 0.28 is skipped
    assert rescue_region(p, rej, veto, 0).tolist() == []
    assert len(rescue_region(p, rej, veto, 99)) == 5


def test_a_perfect_scorer_below_the_reject_line_raises_recall_and_keeps_precision(estate: Path) -> None:
    kw = dict(baseline_max=0, rescore=["rerank"], n_cal2=300, n_ltt2=500)
    plain = run("estate", estate, 1500, 2500, 7, scorers={"rerank": TruthScorer(estate)}, **kw)  # type: ignore[arg-type]
    rescued = run("estate", estate, 1500, 2500, 7, scorers={"rerank": TruthScorer(estate)}, rescue=3000, **kw)  # type: ignore[arg-type]
    band = rescued["bands"]["second_stage"]["rescue"]
    assert band["pairs"] == 3000 and band["true_matches"] > 0
    assert band["accepted"] > 0 and band["accepted_precision"] >= 0.95
    assert band["labels_used"]["ltt"] > 0  # its own labelled sample, not the undecided pairs'
    assert rescued["clustering"]["machine_only"]["pairwise_recall"] > plain["clustering"]["machine_only"]["pairwise_recall"]
    assert rescued["bands"]["auto_accept_precision_realised"] >= 0.95
    assert plain["bands"]["second_stage"]["rescue"] is None


def test_a_scorer_that_knows_nothing_rescues_nothing(estate: Path) -> None:
    res = run("estate", estate, 1500, 2500, 7, baseline_max=0, rescore=["rerank"], n_cal2=300, n_ltt2=500,
              scorers={"rerank": BlindScorer()}, rescue=3000)
    assert res["bands"]["second_stage"]["rescue"]["accepted"] == 0
    assert res["bands"]["auto_accept_precision_realised"] >= 0.95


def test_the_error_budget_is_split_over_the_stages_that_run(estate: Path) -> None:
    one = run("estate", estate, 1500, 2500, 7, baseline_max=0)
    two = run("estate", estate, 1500, 2500, 7, baseline_max=0, relabel=True, n_cal2=300, n_ltt2=500)
    three = run("estate", estate, 1500, 2500, 7, baseline_max=0, relabel=True, rescore=["rerank"], n_cal2=300, n_ltt2=500,
                scorers={"rerank": BlindScorer()})
    assert one["bands"]["delta"] == pytest.approx(0.05)
    assert two["bands"]["delta"] == pytest.approx(0.05 / 2)
    assert three["bands"]["delta"] == pytest.approx(0.05 / 3)
    four = run("estate", estate, 1500, 2500, 7, baseline_max=0, relabel=True, rescore=["rerank"], n_cal2=300, n_ltt2=500,
               scorers={"rerank": BlindScorer()}, rescue=3000)
    assert four["bands"]["delta"] == pytest.approx(0.05 / 4)  # the rescue band is tested on its own
    assert one["bands"]["relabel_stage"] is None


def test_labels_alone_decide_undecided_pairs_without_any_scorer(estate: Path) -> None:
    res = run("estate", estate, 1500, 2500, 7, baseline_max=0, relabel=True, n_cal2=300, n_ltt2=500)
    stage = res["bands"]["relabel_stage"]
    assert stage["scorers"] == [] and res["bands"]["second_stage"] is None
    assert sum(stage["moved"].values()) == stage["uncertain_pairs"] > 0
    assert res["bands"]["auto_accept_precision_realised"] >= 0.95


def test_scores_come_back_in_the_order_asked_whatever_the_batching() -> None:
    from models import in_length_order

    pairs = [("a" * n, "b") for n in (5, 1, 9, 3, 7, 2)]
    seen: list[int] = []

    def score_batch(batch: list[tuple[str, str]]) -> list[float]:
        seen.append(len(batch))
        return [float(len(a)) for a, _ in batch]

    assert in_length_order(pairs, score_batch, 4).tolist() == [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
    assert seen == [4, 2]

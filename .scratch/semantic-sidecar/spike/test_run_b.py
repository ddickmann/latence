"""Part B wiring, with fakes that can disagree with the lexical baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from generate_text import generate
from run_b import _core, match_extracted, run


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("text")
    for k, rows in generate(400, 31, 700).items():
        with open(out / f"{k}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:  # type: ignore[attr-defined]
                fh.write(json.dumps(row) + "\n")
    return out


def test_match_extracted_counts_precision_and_recall_exactly() -> None:
    docs = {"d": "Acme Ltd met Bolt Inc and Cray PLC in Leeds."}
    gold = [
        {"mention_id": "m0", "doc_id": "d", "start": 0, "end": 8, "surface": "Acme Ltd"},
        {"mention_id": "m1", "doc_id": "d", "start": 13, "end": 21, "surface": "Bolt Inc"},
        {"mention_id": "m2", "doc_id": "d", "start": 26, "end": 34, "surface": "Cray PLC"},
    ]
    extracted = {"d": [
        {"start": 0, "end": 4, "label": "organization"},    # "Acme": half of the gold span -> found
        {"start": 9, "end": 43, "label": "organization"},   # swallows the rest of the sentence -> not a match
        {"start": 38, "end": 43, "label": "organization"},  # "Leeds": spurious
    ]}
    found, stats = match_extracted(docs, gold, extracted)
    assert [m["mention_id"] for m in found] == ["m0"]
    assert found[0]["surface"] == "Acme"  # linking gets the extracted surface, not the gold one
    assert stats["precision"] == pytest.approx(1 / 3) and stats["recall"] == pytest.approx(1 / 3)


class OracleishReranker:
    """Scores by exact match of the normalised name core, so it fixes what char n-grams confuse
    and still fails on acronyms and stems: a fake that is neither always right nor always wrong."""

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        for ctx, anchor in pairs:
            surface = ctx[ctx.index("[") + 1 : ctx.index("]")]
            out.append(5.0 if _core(surface) == _core(anchor.split(",")[0]) else -5.0)
        return np.array(out)


class ConstantDecider:
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array([np.nan if i % 50 == 0 else 0.5 for i in range(len(pairs))])


class GoldMinusExtractor:
    """Returns the gold spans, drops every tenth and adds one spurious span per fifth document."""

    def __init__(self, mentions: list[dict[str, Any]]) -> None:
        self.by_doc: dict[str, list[dict[str, Any]]] = {}
        for i, m in enumerate(mentions):
            if i % 10:
                self.by_doc.setdefault(m["doc_id"], []).append({"start": m["start"], "end": m["end"], "label": "organization"})
        self.order: list[str] = []

    def extract(self, texts: list[str], entity_labels: Any, relation_labels: Any) -> tuple[list[list[dict[str, Any]]], list[list[Any]]]:
        ents = []
        for i, d in enumerate(self.order):
            spans = list(self.by_doc.get(d, []))
            if i % 5 == 0:
                spans.append({"start": 0, "end": 4, "label": "organization"})
            ents.append(spans)
        return ents, [[] for _ in texts]


def test_reranker_and_decider_feed_the_decision_model(data: Path) -> None:
    lexical = run(data, "lexical", 5, 300, 500, 7)
    full = run(data, "rerank", 5, 300, 500, 7, reranker=OracleishReranker(), decide="decide", decider=ConstantDecider())
    assert full["mentions"] == lexical["mentions"]
    assert full["retrieval_recall_in_kb"] == lexical["retrieval_recall_in_kb"]  # same candidates
    assert set(full["features"]) == {"top1_score", "margin", "top1_lexical", "decide_p_yes", "decide_missing"}
    assert abs(full["features"]["decide_p_yes"]) < 1e-6  # a constant answer carries no weight
    assert full["throughput"]["scored_pairs"] == lexical["throughput"]["scored_pairs"]
    share = full["bands"]["share"]
    assert share["accept"] + share["review"] + share["reject"] == pytest.approx(1.0)


def test_extracted_mentions_replace_gold_and_extraction_is_scored(data: Path) -> None:
    mentions = [json.loads(x) for x in open(data / "mentions.jsonl", encoding="utf-8")]
    docs = [json.loads(x)["doc_id"] for x in open(data / "docs.jsonl", encoding="utf-8")]
    ex = GoldMinusExtractor(mentions)
    ex.order = docs
    res = run(data, "lexical", 5, 300, 500, 7, mentions_from="relex", extractor=ex)
    kept = sum(1 for i in range(len(mentions)) if i % 10)
    assert res["mentions"] == res["extraction"]["matched"] <= kept
    assert res["extraction"]["recall"] == pytest.approx(res["mentions"] / len(mentions))
    assert res["extraction"]["precision"] < 1.0  # the spurious spans count against precision

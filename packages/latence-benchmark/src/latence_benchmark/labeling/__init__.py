# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The ticket-21 labeling harness — manual gold set + Hyperedge γ calibration labels.

One labeling workflow, two artifacts (tickets 08/09/21):

* **Gold-set mode** — browse the exported corpus (an S1 export's ``records.jsonl`` /
  ``records.parquet``), write a multi-hop question, mark supporting-fact spans by text
  selection, tag the hop count. Output: append-only JSONL, one object per question, offsets
  verified against the corpus text AT WRITE TIME (:mod:`latence_benchmark.labeling.goldset`).
* **Calibration mode** — judge ~200 Hyperedges (``graph-hyperedges.parquet``) as
  correct / incorrect / partial for the γ-calibration sample
  (:mod:`latence_benchmark.labeling.calibration`).

The interface is a local single-page HTML UI served by stdlib ``http.server`` on
``127.0.0.1`` (:mod:`latence_benchmark.labeling.server`) — no external assets, no new
dependencies — started via the ``latence-label`` console entry
(:mod:`latence_benchmark.labeling.cli`). Ticket 17 calls ``latence-label validate`` before
the eval reads the gold set.
"""

from latence_benchmark.labeling.calibration import (
    JUDGMENTS,
    HyperedgeJudgment,
    HyperedgeRow,
    append_judgment,
    load_hyperedges,
    load_judgments,
    validate_calibration,
)
from latence_benchmark.labeling.goldset import (
    ONE_HOP_WARNING_SHARE,
    ChunkRecord,
    Corpus,
    Coverage,
    GoldQuestion,
    SpanIntegrityError,
    SupportingFact,
    append_question,
    coverage,
    load_corpus,
    load_gold,
    question_id,
    validate_gold,
)

__all__ = [
    "JUDGMENTS",
    "ONE_HOP_WARNING_SHARE",
    "ChunkRecord",
    "Corpus",
    "Coverage",
    "GoldQuestion",
    "HyperedgeJudgment",
    "HyperedgeRow",
    "SpanIntegrityError",
    "SupportingFact",
    "append_judgment",
    "append_question",
    "coverage",
    "load_corpus",
    "load_gold",
    "load_hyperedges",
    "load_judgments",
    "question_id",
    "validate_calibration",
    "validate_gold",
]

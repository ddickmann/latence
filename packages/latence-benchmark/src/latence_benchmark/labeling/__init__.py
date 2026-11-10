# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The ticket-21 labeling harness — the manual gold set.

Browse the exported corpus (an S1 export's ``records.jsonl`` / ``records.parquet``), write a
multi-hop question, mark supporting-fact spans by text selection, tag the hop count. Output:
append-only JSONL, one object per question, offsets verified against the corpus text AT
WRITE TIME (:mod:`latence_benchmark.labeling.goldset`).

The harness had a second mode that served a judgment UI over the knowledge-graph export's
hyperedge parquet. No pipeline writes that file since ADR-0068, so the mode was removed.

The interface is a local single-page HTML UI served by stdlib ``http.server`` on
``127.0.0.1`` (:mod:`latence_benchmark.labeling.server`) — no external assets, no new
dependencies — started via the ``latence-label`` console entry
(:mod:`latence_benchmark.labeling.cli`). Ticket 17 calls ``latence-label validate`` before
the eval reads the gold set.
"""

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
    "ONE_HOP_WARNING_SHARE",
    "ChunkRecord",
    "Corpus",
    "Coverage",
    "GoldQuestion",
    "SpanIntegrityError",
    "SupportingFact",
    "append_question",
    "coverage",
    "load_corpus",
    "load_gold",
    "question_id",
    "validate_gold",
]

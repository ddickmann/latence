# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Resolve the planted chains against the EXPORTED corpus and write gold.jsonl + CHAINS.md.

    PYTHONHASHSEED=0 uv run python benchmark/enterprise-gold/build_gold.py

Every fact sentence in chains.py is located in the exported chunk text — ``masked_content``
when the Redaction Stage produced one, else ``content`` (exactly the rule
``latence_benchmark.labeling.goldset`` applies), matched whitespace-insensitively because
the source wraps lines and the markdown chunker strips markup. The MATCHED substring is
what is stored, so ``chunk_text[char_start:char_end] == text`` holds by construction and
``latence-label validate`` re-checks it.

Refuses to write on any of: a fact that does not resolve, a fact that resolves in more than
one document (ambiguous), a fact that lands in a different document than chains.py claims,
a question whose fact count differs from its hop count, or a multi-hop question whose facts
are not spread over distinct documents. The last one is the whole point of the set: a
multi-hop question must be unanswerable from any single document.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "store" / "_latence" / "runs" / "gold-v0" / "export"
GOLD = HERE / "gold.jsonl"
CHAINS_MD = HERE / "CHAINS.md"

sys.path.insert(0, str(HERE))
from chains import FACTS, QUESTIONS  # noqa: E402


def _pattern(sentence: str) -> re.Pattern[str]:
    """The fact sentence as a whitespace-insensitive regex (line wraps + stripped markup)."""
    return re.compile(r"\s+".join(re.escape(tok) for tok in sentence.split()))


def main() -> int:
    from latence_benchmark.labeling.goldset import (
        SupportingFact,
        append_question,
        coverage,
        load_corpus,
        load_gold,
    )

    corpus = load_corpus(EXPORT)
    print(f"corpus: {len(corpus.documents)} documents, {len(corpus.chunks)} chunks")

    # fact id -> SupportingFact, resolved against the exported chunk text
    resolved: dict[str, SupportingFact] = {}
    problems: list[str] = []
    for fact_id, (doc_number, sentence) in FACTS.items():
        pattern = _pattern(sentence)
        hits = []
        for chunk in corpus.chunks.values():
            match = pattern.search(chunk.text)
            if match:
                hits.append((chunk, match))
        if not hits:
            problems.append(f"{fact_id}: not found in any exported chunk ({sentence[:60]!r})")
            continue
        docs = {chunk.file_name for chunk, _ in hits}
        if len(docs) > 1:
            problems.append(f"{fact_id}: ambiguous — matches {sorted(docs)}")
            continue
        chunk, match = min(hits, key=lambda h: (h[0].order, h[0].chunk_record_id))
        if not chunk.file_name.startswith(doc_number):
            problems.append(
                f"{fact_id}: claims document {doc_number} but resolved in {chunk.file_name}"
            )
            continue
        resolved[fact_id] = SupportingFact(
            document_id=chunk.document_id,
            chunk_record_id=chunk.chunk_record_id,
            char_start=match.start(),
            char_end=match.end(),
            text=match.group(0),
        )

    # question-level invariants
    seen_questions: set[str] = set()
    for question, hops, _answer, fact_ids in QUESTIONS:
        if question in seen_questions:
            problems.append(f"duplicate question text: {question[:60]!r}")
        seen_questions.add(question)
        if len(fact_ids) != hops:
            problems.append(f"{question[:50]!r}: {len(fact_ids)} facts for {hops} hops")
        missing = [f for f in fact_ids if f not in resolved]
        if missing:
            problems.append(f"{question[:50]!r}: unresolved facts {missing}")
            continue
        docs = [FACTS[f][0] for f in fact_ids]
        if hops >= 2 and len(set(docs)) != len(docs):
            problems.append(
                f"{question[:50]!r}: facts share a document ({docs}) — not a cross-document chain"
            )

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"{len(problems)} problem(s); nothing written", file=sys.stderr)
        return 1

    if GOLD.exists():
        GOLD.unlink()  # deterministic rebuild, not an append-resume session
    for question, hops, _answer, fact_ids in QUESTIONS:
        append_question(GOLD, corpus, question, hops, [resolved[f] for f in fact_ids])

    cov = coverage(load_gold(GOLD).values())
    print(f"wrote {GOLD} — {cov.total} questions")
    for hop, count in sorted(cov.hop_distribution.items()):
        print(f"  {hop}-hop: {count}")
    print(f"1-hop share: {cov.one_hop_share:.0%}  warning={cov.warning}")

    _write_chains_md(resolved)
    print(f"wrote {CHAINS_MD}")
    return 0


def _write_chains_md(resolved: dict) -> None:
    """The honesty artifact: every planted fact and every chain built from it."""
    lines = [
        "# Planted chains — the enterprise-gold corpus",
        "",
        "GENERATED by `build_gold.py` from `chains.py`; do not edit by hand.",
        "",
        "This is the review sheet for the gold set: every fact that was deliberately planted",
        "in the corpus, which document asserts it, and every question chain built out of",
        "those facts. A reviewer checks the corpus against this file — if a chain link is not",
        "actually asserted in the named document, the question built on it is invalid.",
        "",
        "All entities are invented. Any resemblance to a real company or person is",
        "unintended (see PROVENANCE.md).",
        "",
        "## 1. Planted facts",
        "",
        "| fact id | document | asserting sentence (as exported) |",
        "|---------|----------|----------------------------------|",
    ]
    for fact_id, (doc_number, _sentence) in FACTS.items():
        text = resolved[fact_id].text.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{fact_id}` | {doc_number} | {text} |")
    counts: dict[int, int] = {}
    for _q, hops, _a, _f in QUESTIONS:
        counts[hops] = counts.get(hops, 0) + 1
    lines += [
        "",
        "## 2. Chains (one per gold question)",
        "",
        f"{len(QUESTIONS)} questions: "
        + ", ".join(f"{n}× {h}-hop" for h, n in sorted(counts.items())),
        "",
        "Every chain of depth >= 2 spans distinct documents (enforced by build_gold.py), so no",
        "multi-hop question is answerable from a single document.",
        "",
        "| hops | question | answer | chain (fact id @ document) |",
        "|------|----------|--------|----------------------------|",
    ]
    for question, hops, answer, fact_ids in QUESTIONS:
        chain = " → ".join(f"`{f}`@{FACTS[f][0]}" for f in fact_ids)
        lines.append(f"| {hops} | {question} | {answer} | {chain} |")
    lines.append("")
    CHAINS_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

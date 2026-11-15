"""Measure each older resolution module of the text path against the same gold without it
(ticket 50; the owner's rule of 14 Nov: an older module is kept only where a measurement shows
it adds value).

The rule, fixed here before any run (``RULE``): a module is kept when, on its gold set, it
raises pairwise F1 by at least ``MIN_GAIN`` over the same run without it and its precision is
not lower; otherwise it is retired.

- **Mentions** (``gold-mentions.json``: synthetic English documents that name organisations,
  with noisy surfaces and each mention's true entity). Without a module: the cascade restricted
  to exact normalised surfaces (``EntityResolver(exact_only=True)``, which is also the
  ``ExactSurfaceDisambiguator``). With: the string cascade (``EntityResolver()``, its five
  rungs), and the embedding Disambiguator (``latence-disambig-embedding``) on the in-core hashing
  embedder (the learned embedder needs a GPU: not run here).
- **Type labels** (``gold-types.json``). Without: labels equal after lower-casing and trimming.
  With: Type Consolidation (``TypeCanonicalizer()``, its lexical rungs; the embedding rung needs
  an embedder and is off by default).
- **The GLinker linker** (``latence-disambig-glinker``) needs its model on a GPU and is not run
  here.

    PYTHONHASHSEED=0 uv run python benchmark/resolution-modules/measure.py > results.json
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    PageMap,
    PageSlice,
    Provenance,
)
from latence_core.disambiguation import EntityResolver, MentionInput
from latence_core.type_consolidation import TypeCanonicalizer
from latence_disambig_embedding.provider import EmbeddingDisambiguator

HERE = Path(__file__).resolve().parent
MIN_GAIN = 0.02
RULE = (
    f"kept when pairwise F1 rises by at least {MIN_GAIN} over the run without the module and "
    "precision does not fall; retired otherwise"
)
_CLS = Classification(language="en", category="synthetic", sensitivity="internal")


def pairwise(predicted: Iterable[Iterable[str]], gold: dict[str, str]) -> dict[str, float]:
    """Pairwise precision, recall and F1 of clusters against a gold label per item."""
    said = {frozenset(p) for c in predicted for p in itertools.combinations(sorted(c), 2)}
    by_gold: dict[str, list[str]] = {}
    for item, label in gold.items():
        by_gold.setdefault(label, []).append(item)
    true = {frozenset(p) for c in by_gold.values() for p in itertools.combinations(sorted(c), 2)}
    hit = len(said & true)
    precision = hit / len(said) if said else 1.0
    recall = hit / len(true) if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "pairs_said": len(said), "pairs_true": len(true)}


def _cascade(mentions: list[dict[str, Any]], exact_only: bool) -> list[list[str]]:
    resolver = EntityResolver(exact_only=exact_only)
    resolver.add_all([MentionInput(m["mention_id"], m["doc_id"], m["surface"], "ORG", 0.9)
                      for m in mentions])
    clusters, _audits = resolver.resolve()
    return [c.member_mention_ids for c in clusters]


def _prov(doc: str, start: int, end: int, size: int) -> Provenance:
    return Provenance(source_uri=f"file://{doc}.txt", file_name=f"{doc}.txt", file_type="txt",
                      file_size=size, document_id=doc, char_start=start, char_end=end,
                      page_start=1, page_end=1)


def _embedding(mentions: list[dict[str, Any]], docs: dict[str, str]) -> list[list[str]]:
    chunks = {
        d: ChunkRecord(
            record_id=f"{d}:c0", document_record_id=f"{d}:doc", index=0, content=text,
            token_count=len(text.split()), provenance=_prov(d, 0, len(text), len(text)),
            classification=_CLS,
            page_slice=PageSlice.for_span(PageMap.from_page_texts([text]).page_map, 0, len(text)),
        )
        for d, text in docs.items()
    }
    entity_mentions = [
        EntityMention(
            record_id=m["mention_id"], document_record_id=f"{m['doc_id']}:doc",
            chunk_record_id=f"{m['doc_id']}:c0", index=n, label="ORG", text=m["surface"],
            confidence=0.9,
            provenance=_prov(m["doc_id"], m["start"], m["end"], len(docs[m["doc_id"]])),
            classification=_CLS,
        )
        for n, m in enumerate(mentions)
    ]
    disambiguator = EmbeddingDisambiguator({"max_pairs": 10_000_000})
    clusters = disambiguator._resolve_clusters(entity_mentions, list(chunks.values()))
    return [c.member_mention_ids for c in clusters]


def _types_exact(labels: list[dict[str, Any]]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for item in labels:
        groups.setdefault(item["label"].strip().lower(), []).append(item["label"])
    return list(groups.values())


def _types_consolidated(labels: list[dict[str, Any]]) -> list[list[str]]:
    vocabulary = TypeCanonicalizer().build({item["label"]: item["count"] for item in labels})
    return [sorted({t.canonical, *t.aliases}) for t in vocabulary.types]


def _timed(run: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    out = run()
    return out, round(time.perf_counter() - start, 3)


def _verdict(without: dict[str, float], with_: dict[str, float]) -> str:
    gain = with_["f1"] - without["f1"]
    kept = gain >= MIN_GAIN and with_["precision"] >= without["precision"]
    return "keep" if kept else "retire"


def main() -> None:
    gold_mentions = json.loads((HERE / "gold-mentions.json").read_text(encoding="utf-8"))
    mentions = gold_mentions["mentions"]
    docs = {d["doc_id"]: d["text"] for d in gold_mentions["docs"]}
    truth = {m["mention_id"]: m["entity_id"] for m in mentions}
    labels = json.loads((HERE / "gold-types.json").read_text(encoding="utf-8"))["labels"]
    type_truth = {item["label"]: item["type"] for item in labels}

    without, t0 = _timed(lambda: _cascade(mentions, exact_only=True))
    cascade, t1 = _timed(lambda: _cascade(mentions, exact_only=False))
    embedding, t2 = _timed(lambda: _embedding(mentions, docs))
    types_without, t3 = _timed(lambda: _types_exact(labels))
    types_with, t4 = _timed(lambda: _types_consolidated(labels))

    base = pairwise(without, truth)
    type_base = pairwise(types_without, type_truth)
    modules = {
        "disambiguation.cascade": ({**pairwise(cascade, truth), "seconds": t1}, base),
        "disambiguation.embedding (hashing embedder)": (
            {**pairwise(embedding, truth), "seconds": t2}, base),
        "type_consolidation": ({**pairwise(types_with, type_truth), "seconds": t4}, type_base),
    }
    result = {
        "rule": RULE,
        "gold": {
            "mentions": {"file": "gold-mentions.json", "mentions": len(mentions),
                         "entities": len(set(truth.values())), "documents": len(docs),
                         "generator": gold_mentions["generator"]},
            "types": {"file": "gold-types.json", "labels": len(labels),
                      "types": len(set(type_truth.values()))},
        },
        "without": {"mentions (exact surfaces)": {**base, "seconds": t0},
                    "types (lower-cased labels)": {**type_base, "seconds": t3}},
        "modules": {
            name: {**scores, "f1_gain": round(scores["f1"] - against["f1"], 4),
                   "precision_change": round(scores["precision"] - against["precision"], 4),
                   "verdict": _verdict(against, scores)}
            for name, (scores, against) in modules.items()
        },
        "not_run": {"disambiguation.glinker": "needs its model on a GPU (a live request)",
                    "disambiguation.embedding (learned embedder)": "needs a GPU (a live request)"},
    }
    json.dump(result, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

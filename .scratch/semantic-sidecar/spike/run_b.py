"""Part B of the ticket-20 spike: anchor-first linking of text mentions, then NIL clustering.

  mentions: the gold spans, or GLiNER-Relex extraction from the raw documents (scored against
            the span gold)
  -> candidate anchors: char-ngram TF-IDF on names UNION EmbeddingGemma 2 on mention-in-context
  -> pairwise score of each (mention-in-context, anchor): Qwen3-Reranker (or the lexical cosine)
  -> optional decision feature: P(the top-1 anchor is the referent) from Decide or GLiDE
  -> logistic probability that top-1 is right -> Learn-then-Test bands:
     accept = link top-1, reject = NIL, review = simulated expert
  -> NIL mentions clustered among themselves.

Usage:
  python run_b.py --data data/text_2k --scorer lexical --out out/text_2k_lexical.json
  python run_b.py --data data/text_2k --scorer rerank --embed --decide decide --out out/text_2k_full.json
  python run_b.py --data data/text_2k --scorer rerank --embed --mentions relex --out out/text_2k_e2e.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from er import learn_then_test_bands, pairwise_and_bcubed, wilson_lower

LEGAL = r"\b(ltd|limited|inc|incorporated|llc|corp|corporation|plc|co|company)\b\.?"


def _core(s: str) -> str:
    s = re.sub(LEGAL, " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def load(path: Path) -> tuple[list[dict], dict[str, str], list[dict]]:
    def rd(n: str) -> list[dict]:
        with open(path / f"{n}.jsonl", encoding="utf-8") as fh:
            return [json.loads(x) for x in fh]

    docs = {d["doc_id"]: d["text"] for d in rd("docs")}
    return rd("anchors"), docs, rd("mentions")


def match_extracted(
    docs: dict[str, str], gold: list[dict], extracted: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict], dict[str, float]]:
    """Align extracted organisation spans with the gold mentions of the same document.

    A gold mention counts as found when an extracted span overlaps it by at least half of the
    longer of the two (so a span that swallows half a sentence does not count). Each extracted
    span matches at most one gold mention. Returns the found mentions, carrying the EXTRACTED
    span and surface (that is what linking has to work with), and extraction precision/recall.
    """
    by_doc: dict[str, list[dict]] = {}
    for m in gold:
        by_doc.setdefault(m["doc_id"], []).append(m)
    found: list[dict] = []
    n_ext = sum(len(v) for v in extracted.values())
    for doc_id, golds in by_doc.items():
        free = list(extracted.get(doc_id, []))
        for g in golds:
            best, best_ov = None, 0.0
            for e in free:
                inter = min(g["end"], e["end"]) - max(g["start"], e["start"])
                ov = inter / max(g["end"] - g["start"], e["end"] - e["start"]) if inter > 0 else 0.0
                if ov > best_ov:
                    best, best_ov = e, ov
            if best is not None and best_ov >= 0.5:
                free.remove(best)
                found.append({**g, "start": best["start"], "end": best["end"], "surface": docs[doc_id][best["start"] : best["end"]]})
    return found, {
        "extracted_spans": n_ext,
        "gold_mentions": len(gold),
        "matched": len(found),
        "precision": len(found) / n_ext if n_ext else float("nan"),
        "recall": len(found) / len(gold) if gold else float("nan"),
    }


def run(
    path: Path,
    scorer: str,
    k: int,
    n_cal: int,
    n_ltt: int,
    seed: int,
    *,
    use_embedder: bool = False,
    decide: str = "none",
    mentions_from: str = "gold",
    embedder: Any = None,
    reranker: Any = None,
    decider: Any = None,
    extractor: Any = None,
) -> dict[str, Any]:
    """The keyword objects inject ready models (tests use fakes); left as None, the agreed
    models are loaded from models.py when an arm asks for them."""
    t0 = time.time()
    anchors, docs, gold_mentions = load(path)
    timings: dict[str, float] = {}
    extraction: dict[str, float] | None = None
    mentions = gold_mentions
    if mentions_from == "relex":
        t = time.time()
        if extractor is None:
            import models as M

            extractor = M.RelexExtractor()
        doc_ids = list(docs)
        ents, _rels = extractor.extract([docs[d] for d in doc_ids], ["organization"], [])
        extracted = {d: [e for e in es if e["label"] == "organization"] for d, es in zip(doc_ids, ents)}
        mentions, extraction = match_extracted(docs, gold_mentions, extracted)
        timings["extract"] = time.time() - t
        extraction["docs_per_s"] = len(doc_ids) / timings["extract"] if timings["extract"] > 0 else float("inf")

    a_ids = [a["anchor_id"] for a in anchors]
    a_text = [a["text"] for a in anchors]
    ctx = [
        docs[m["doc_id"]][: m["start"]] + "[" + m["surface"] + "]" + docs[m["doc_id"]][m["end"] :]
        for m in mentions
    ]

    # ---- candidate retrieval -------------------------------------------------------------
    t = time.time()
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
    A = tf.fit_transform([_core(a["name"]) for a in anchors])
    Mx = tf.transform([_core(m["surface"]) for m in mentions])
    lex = (Mx @ A.T).toarray()
    cand = [set(row) for row in np.argsort(-lex, axis=1)[:, :k]]
    if use_embedder:
        if embedder is None:
            import models as M

            embedder = M.Embedder()
        ea = embedder.encode(a_text, "Document")
        em = embedder.encode(ctx, "SearchQuery")
        for i, row in enumerate(np.argsort(-(em @ ea.T), axis=1)[:, :k]):
            cand[i] |= set(row)
    timings["retrieve"] = time.time() - t
    gold_idx = {aid: j for j, aid in enumerate(a_ids)}
    in_kb = np.array([m["in_kb"] for m in mentions], dtype=bool)
    recall_at_cand = float(np.mean([gold_idx[m["entity_id"]] in cand[i] for i, m in enumerate(mentions) if m["in_kb"]]))

    # ---- pairwise scoring ------------------------------------------------------------------
    t = time.time()
    flat = [(i, int(j)) for i, cs in enumerate(cand) for j in sorted(cs)]
    if scorer == "lexical":
        s = np.array([lex[i, j] for i, j in flat])
    else:
        if reranker is None:
            import models as M

            reranker = M.Reranker(M.LINK_INSTRUCTION)
        s = np.asarray(reranker.score([(ctx[i], a_text[j]) for i, j in flat]), dtype=float)
    timings["score"] = time.time() - t
    pairs_per_s = len(flat) / timings["score"] if timings["score"] > 0 else float("inf")

    top1 = np.full(len(mentions), -1)
    s1 = np.full(len(mentions), -1e9)
    s2 = np.full(len(mentions), -1e9)
    for (i, j), v in zip(flat, s):
        if v > s1[i]:
            s2[i], s1[i], top1[i] = s1[i], v, j
        elif v > s2[i]:
            s2[i] = v
    s2[s2 < -1e8] = s1[s2 < -1e8] - 1.0
    lex1 = np.array([lex[i, top1[i]] for i in range(len(mentions))])
    feats, feat_names = [s1, s1 - s2, lex1], ["top1_score", "margin", "top1_lexical"]

    # ---- optional decision feature on the top-1 pair -----------------------------------------
    if decide != "none":
        t = time.time()
        if decider is None:
            import models as M

            decider = M.GlinerDecide(M.LINK_QUESTION) if decide == "decide" else M.GlideDecide(M.LINK_QUESTION)
        d = np.asarray(decider.score([(ctx[i], a_text[top1[i]]) for i in range(len(mentions))]), dtype=float)
        miss = ~np.isfinite(d)
        feats += [np.where(miss, np.nanmedian(np.where(miss, np.nan, d)) if (~miss).any() else 0.5, d), miss.astype(float)]
        feat_names += [f"{decide}_p_yes", f"{decide}_missing"]
        timings["decide"] = time.time() - t
    X = np.column_stack(feats)
    correct = np.array([m["in_kb"] and a_ids[top1[i]] == m["entity_id"] for i, m in enumerate(mentions)], dtype=int)

    # ---- decision: calibrated probability that top-1 is the right link ------------------------
    order = np.random.default_rng(seed).permutation(len(mentions))
    cal, ltt = order[:n_cal], order[n_cal : n_cal + n_ltt]
    mu, sd = X[cal].mean(axis=0), X[cal].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    lr = LogisticRegression(max_iter=2000).fit(Z[cal], correct[cal])
    p = lr.predict_proba(Z)[:, 1]
    bands = learn_then_test_bands(p[ltt], correct[ltt], target=0.95, delta=0.05, reject_target=0.99)
    band = np.array([bands.band(x) for x in p])
    acc, rev, rej = band == "accept", band == "review", band == "reject"

    # ---- assignments -----------------------------------------------------------------------
    def assign(with_expert: bool) -> dict[str, str]:
        out: dict[str, str] = {}
        nil: list[int] = []
        for i, m in enumerate(mentions):
            mid = m["mention_id"]
            if acc[i]:
                out[mid] = a_ids[top1[i]]
            elif rej[i]:
                nil.append(i)
            elif not with_expert:
                out[mid] = f"unresolved:{mid}"  # the machine alone leaves the review band open
            elif correct[i]:
                out[mid] = a_ids[top1[i]]
            elif m["in_kb"]:
                out[mid] = m["entity_id"]  # the expert picks the right anchor
            else:
                nil.append(i)
        for i in nil:  # NIL clustering: same normalised surface core -> one new entity
            out[mentions[i]["mention_id"]] = "nil:" + _core(mentions[i]["surface"])
        return out

    gold = {m["mention_id"]: m["entity_id"] for m in mentions}
    res_machine, res_expert = assign(False), assign(True)
    true_nil = {m["mention_id"] for m in mentions if not m["in_kb"]}

    def nil_quality(pred: dict[str, str]) -> dict[str, float]:
        decided = {mid for mid, v in pred.items() if v.startswith("nil:")}
        tp = len(decided & true_nil)
        return {
            "nil_precision": tp / len(decided) if decided else float("nan"),
            "nil_recall": tp / len(true_nil) if true_nil else float("nan"),
        }

    by_variant: dict[str, dict[str, float]] = {}
    for v in sorted({m["variant"] for m in mentions}):
        idx = [i for i, m in enumerate(mentions) if m["variant"] == v and m["in_kb"]]
        if idx:
            by_variant[v] = {
                "n": len(idx),
                "top1_accuracy": float(np.mean(correct[idx])),
                "auto_linked_share": float(np.mean(acc[idx])),
            }

    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {
        "data": str(path),
        "arm": {"mentions": mentions_from, "embedder": use_embedder, "scorer": scorer, "decide": decide},
        "anchors": len(anchors),
        "mentions": len(mentions),
        "nil_mentions": int((~in_kb).sum()),
        "extraction": extraction,
        "candidates_per_mention": float(np.mean([len(c) for c in cand])),
        "retrieval_recall_in_kb": recall_at_cand,
        "top1_accuracy_in_kb": float(correct[in_kb].mean()),
        "labels_used": {"calibration": int(len(cal)), "ltt": int(len(ltt))},
        "features": dict(zip(feat_names, (float(c) for c in lr.coef_[0]))),
        "bands": {
            "accept_at": bands.accept_at,
            "reject_below": bands.reject_below,
            "share": {"accept": float(acc.mean()), "review": float(rev.mean()), "reject": float(rej.mean())},
            "auto_link_precision_realised": float(correct[acc].mean()) if acc.any() else float("nan"),
            "auto_link_precision_wilson_lower": wilson_lower(int(correct[acc].sum()), int(acc.sum())),
            "rejected_but_linkable_rate": float(correct[rej].mean()) if rej.any() else float("nan"),
        },
        "by_variant": by_variant,
        "assignment": {
            "machine_only": {**pairwise_and_bcubed(res_machine, gold), **nil_quality(res_machine)},
            "machine_plus_expert": {**pairwise_and_bcubed(res_expert, gold), **nil_quality(res_expert)},
        },
        "review_items": int(rev.sum()),
        "throughput": {"scored_pairs": len(flat), "pairs_per_s": pairs_per_s},
        "timings_s": {**{k_: round(v, 2) for k_, v in timings.items()}, "total": round(time.time() - t0, 1)},
        "hardware": {"cpus": os.cpu_count(), "gpu": gpu},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scorer", choices=["lexical", "rerank"], required=True)
    ap.add_argument("--embed", action="store_true", help="add EmbeddingGemma 2 candidates to the TF-IDF ones")
    ap.add_argument("--decide", choices=["none", "decide", "glide"], default="none")
    ap.add_argument("--mentions", choices=["gold", "relex"], default="gold")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-cal", type=int, default=800)
    ap.add_argument("--n-ltt", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = run(a.data, a.scorer, a.k, a.n_cal, a.n_ltt, a.seed, use_embedder=a.embed, decide=a.decide, mentions_from=a.mentions)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, default=float))
    keys = ("arm", "extraction", "retrieval_recall_in_kb", "top1_accuracy_in_kb", "bands", "assignment", "throughput", "timings_s")
    print(json.dumps({x: r[x] for x in keys}, indent=1, default=float))


if __name__ == "__main__":
    main()

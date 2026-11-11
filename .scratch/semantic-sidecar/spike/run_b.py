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

from aliases import AliasTable, is_short_form, mine_definitions, strongest
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


def _scopes(path: Path) -> dict[str, str]:
    """The thread each document belongs to (its own id when the data has no threads)."""
    with open(path / "docs.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh]
    return {d["doc_id"]: d.get("thread_id", d["doc_id"]) for d in rows}


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
    use_aliases: bool = False,
    alias_cap: int = 30,
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
    # With aliases on, a short form ("IBM") is not looked up by its characters: its candidates
    # come from the alias routes, once the full names around it have been linked.
    short = np.array([use_aliases and is_short_form(m["surface"]) for m in mentions], dtype=bool)
    if use_embedder:
        if embedder is None:
            import models as M

            embedder = M.Embedder()
        ea = embedder.encode(a_text, "Document")
        # Search with the mention alone. A whole document as the query retrieves every
        # organisation it names, and the reranker then picks the wrong one of them.
        em = embedder.encode([m["surface"] for m in mentions], "SearchQuery")
        for i, row in enumerate(np.argsort(-(em @ ea.T), axis=1)[:, :k]):
            cand[i] |= set(row)
    timings["retrieve"] = time.time() - t
    gold_idx = {aid: j for j, aid in enumerate(a_ids)}
    in_kb = np.array([m["in_kb"] for m in mentions], dtype=bool)

    # ---- pairwise scoring ------------------------------------------------------------------
    score_secs, n_scored = 0.0, 0

    def score_pairs(flat: list[tuple[int, int]]) -> np.ndarray:
        nonlocal reranker, score_secs, n_scored
        t = time.time()
        if scorer == "lexical":
            out = np.array([lex[i, j] for i, j in flat], dtype=float)
        else:
            if reranker is None:
                import models as M

                reranker = M.Reranker(M.LINK_INSTRUCTION)
            out = np.asarray(reranker.score([(ctx[i], a_text[j]) for i, j in flat]), dtype=float)
        score_secs += time.time() - t
        n_scored += len(flat)
        return out

    top1 = np.full(len(mentions), -1)
    s1 = np.full(len(mentions), -1e9)
    s2 = np.full(len(mentions), -1e9)

    def rank(rows: np.ndarray) -> None:
        flat = [(int(i), int(j)) for i in rows for j in sorted(cand[i])]
        for (i, j), v in zip(flat, score_pairs(flat)):
            if v > s1[i]:
                s2[i], s1[i], top1[i] = s1[i], v, j
            elif v > s2[i]:
                s2[i] = v

    rank(np.where(~short)[0])

    # ---- short forms: every route proposes, one small ranker chooses --------------------------
    order = np.random.default_rng(seed).permutation(len(mentions))
    cal, ltt = order[:n_cal], order[n_cal : n_cal + n_ltt]
    route = np.full(len(mentions), "", dtype=object)
    ranker_weights: dict[str, float] | None = None
    if use_aliases:
        t = time.time()
        before = score_secs
        scope_of = _scopes(path)
        scope = [scope_of.get(m["doc_id"], m["doc_id"]) for m in mentions]
        table = AliasTable(a["name"] for a in anchors)
        for j, a in enumerate(anchors):
            for alias in a.get("aliases", []):
                table.add_alias(alias, j)
        by_doc: dict[str, list[int]] = {}
        for i, m in enumerate(mentions):
            by_doc.setdefault(m["doc_id"], []).append(i)
            if not short[i]:  # a full name linked here speaks for its short forms in this thread
                table.add_in_scope(scope[i], int(top1[i]))
        for doc_id, text in docs.items():
            for d in mine_definitions(text):
                over = [i for i in by_doc.get(doc_id, []) if not short[i]
                        and mentions[i]["start"] < d["end"] and mentions[i]["end"] > d["start"]]  # type: ignore[operator]
                if over:
                    table.add_definition(str(d["short"]), int(top1[over[0]]))
        popularity = np.bincount(top1[~short], minlength=len(anchors))  # full-name links per anchor
        thread_text: dict[str, str] = {}
        for doc_id, text in docs.items():
            sc = scope_of.get(doc_id, doc_id)
            thread_text[sc] = thread_text.get(sc, "") + " " + text.lower()
        attrs = [  # what an anchor's description says beyond its name: city, industry, ...
            set(re.findall(r"[a-z]{4,}", a["text"].lower())) - set(re.findall(r"[a-z]{4,}", a["name"].lower())) - {"company"}
            for a in anchors
        ]
        evidence: dict[tuple[int, int], dict[str, int]] = {}
        for i in np.where(short)[0]:
            cs = table.candidates(mentions[i]["surface"], scope[i])
            if not cs:
                continue  # no route knows it: the character candidates stay, as for any mention
            keep = sorted(cs, key=lambda j: (strongest(cs[j]) == "initials", -popularity[j], j))[:alias_cap]
            cand[i] = set(keep)
            evidence.update({(int(i), j): cs[j] for j in keep})
        flat = [(int(i), int(j)) for i in np.where(short)[0] for j in sorted(cand[i])]
        base = score_pairs(flat)
        names_f = ["score", "in_thread", "defined", "alias_column", "popularity", "context_doc", "context_thread"]

        def share(words: set[str], text: str) -> float:
            return sum(w in text for w in words) / len(words) if words else 0.0

        F = np.array([
            [
                base[r],
                float("scope" in evidence.get((i, j), {})),
                np.log1p(evidence.get((i, j), {}).get("defined", 0)),
                float("alias" in evidence.get((i, j), {})),
                np.log1p(popularity[j]),
                share(attrs[j], docs[mentions[i]["doc_id"]].lower()),
                share(attrs[j], thread_text[scope[i]]),
            ]
            for r, (i, j) in enumerate(flat)
        ], dtype=float).reshape(len(flat), len(names_f))
        y = np.array([mentions[i]["in_kb"] and a_ids[j] == mentions[i]["entity_id"] for i, j in flat], dtype=int)
        in_cal = np.zeros(len(mentions), dtype=bool)
        in_cal[cal] = True
        rows = np.array([in_cal[i] for i, _ in flat], dtype=bool)
        if len(flat) and min(y[rows].sum(), (1 - y[rows]).sum()) >= 10:
            mu_f, sd_f = F[rows].mean(axis=0), F[rows].std(axis=0)
            sd_f[sd_f == 0] = 1.0
            pair_lr = LogisticRegression(max_iter=2000).fit((F[rows] - mu_f) / sd_f, y[rows])
            logit = pair_lr.decision_function((F - mu_f) / sd_f)
            ranker_weights = dict(zip(names_f, (float(c) for c in pair_lr.coef_[0])))
        else:  # too few labelled short forms to fit anything: a fixed order of the evidence
            logit = 6.0 * F[:, 1] + 3.0 * (F[:, 2] > 0) + 3.0 * F[:, 3] + F[:, 5] + 0.5 * F[:, 6] + 0.1 * F[:, 4]
        for (i, j), v in zip(flat, logit):
            if v > s1[i]:
                s2[i], s1[i], top1[i] = s1[i], v, j
            elif v > s2[i]:
                s2[i] = v
        for i in np.where(short)[0]:
            route[i] = strongest(evidence.get((int(i), int(top1[i])), {}))
        timings["aliases"] = time.time() - t - (score_secs - before)
    timings["score"] = score_secs
    pairs_per_s = n_scored / score_secs if score_secs > 0 else float("inf")
    recall_at_cand = float(np.mean([gold_idx[m["entity_id"]] in cand[i] for i, m in enumerate(mentions) if m["in_kb"]]))
    s2[s2 < -1e8] = s1[s2 < -1e8] - 1.0
    lex1 = np.array([lex[i, top1[i]] for i in range(len(mentions))])
    if use_aliases:  # a short form's score is the ranker's, on another scale than a full name's
        feats = [np.where(short, 0.0, s1), np.where(short, 0.0, s1 - s2), lex1, short.astype(float), np.where(short, s1, 0.0), np.where(short, s1 - s2, 0.0)]
        feat_names = ["top1_score", "margin", "top1_lexical", "short_form", "short_top1_score", "short_margin"]
    else:
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

    by_route: dict[str, dict[str, float]] = {}
    by_signal: dict[str, dict[str, float]] = {}
    if use_aliases:
        groups: dict[str, list[int]] = {}
        sig: dict[str, list[int]] = {}
        for i, m in enumerate(mentions):
            if short[i] and m["in_kb"]:
                groups.setdefault(str(route[i]), []).append(i)
                for name in m.get("signals") or (["none"] if "signals" in m else []):
                    sig.setdefault(name, []).append(i)
        for target, source in ((by_route, groups), (by_signal, sig)):
            for name, idx in sorted(source.items()):
                target[name] = {
                    "n": len(idx),
                    "top1_accuracy": float(np.mean(correct[idx])),
                    "auto_linked_share": float(np.mean(acc[idx])),
                    "auto_link_precision": float(np.mean(correct[idx][acc[idx]])) if acc[idx].any() else float("nan"),
                }

    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {
        "data": str(path),
        "arm": {"mentions": mentions_from, "embedder": use_embedder, "scorer": scorer, "decide": decide, "aliases": use_aliases},
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
        "short_forms": {"mentions": int(short.sum()), "ranker_weights": ranker_weights, "by_route": by_route, "by_signal": by_signal} if use_aliases else None,
        "assignment": {
            "machine_only": {**pairwise_and_bcubed(res_machine, gold), **nil_quality(res_machine)},
            "machine_plus_expert": {**pairwise_and_bcubed(res_expert, gold), **nil_quality(res_expert)},
        },
        "review_items": int(rev.sum()),
        "throughput": {"scored_pairs": n_scored, "pairs_per_s": pairs_per_s},
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
    ap.add_argument("--aliases", action="store_true", help="resolve short forms (acronyms) through the alias routes of aliases.py")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-cal", type=int, default=800)
    ap.add_argument("--n-ltt", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = run(a.data, a.scorer, a.k, a.n_cal, a.n_ltt, a.seed, use_embedder=a.embed, decide=a.decide, mentions_from=a.mentions, use_aliases=a.aliases)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, default=float))
    keys = ("arm", "extraction", "retrieval_recall_in_kb", "top1_accuracy_in_kb", "bands", "assignment", "throughput", "timings_s")
    print(json.dumps({x: r[x] for x in keys}, indent=1, default=float))


if __name__ == "__main__":
    main()

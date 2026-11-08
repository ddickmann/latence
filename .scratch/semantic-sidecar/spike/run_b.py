"""Part B of the ticket-20 spike: anchor-first linking of text mentions, then NIL clustering.

  retrieve candidates (char-ngram TF-IDF on names U bge-small embedding on mention-in-context)
  -> cross-encoder scores each (mention, anchor) pair (ONNX Runtime, CPU)
  -> per-mention decision features (top score, margin, lexical cosine) -> logistic probability
  -> Learn-then-Test bands: accept = link top-1, reject = NIL, review = simulated expert
  -> NIL mentions clustered among themselves.

Usage: python run_b.py --data data/text_2k --scorer bge-reranker-base --out out/text_2k_bge.json
       python run_b.py --data data/text_2k --scorer lexical --out out/text_2k_lexical.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tokenizers import Tokenizer

from er import learn_then_test_bands, pairwise_and_bcubed, wilson_lower

MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L6-v2",
}
LEGAL = r"\b(ltd|limited|inc|incorporated|llc|corp|corporation|plc|co|company)\b\.?"


def _model_dir(key: str) -> str:
    repo = MODELS[key].replace("/", "--")
    hits = glob.glob(f"models/models--{repo}/snapshots/*/")
    if not hits:
        raise FileNotFoundError(f"model {MODELS[key]} not downloaded")
    return hits[0]


class OnnxModel:
    def __init__(self, key: str, max_len: int = 128) -> None:
        d = _model_dir(key)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(d + "onnx/model.onnx", so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(d + "tokenizer.json")
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def _run(self, enc: list[Any]) -> np.ndarray:
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        return self.sess.run(None, feed)[0]

    def embed(self, texts: list[str], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            h = self._run(self.tok.encode_batch(texts[i : i + bs]))[:, 0, :]  # CLS pooling (bge)
            out.append(h / np.linalg.norm(h, axis=1, keepdims=True))
        return np.vstack(out)

    def score_pairs(self, pairs: list[tuple[str, str]], bs: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), bs):
            logits = self._run(self.tok.encode_batch(pairs[i : i + bs]))
            out.append(logits[:, 0] if logits.ndim == 2 else logits)
        return np.concatenate(out)


def _core(s: str) -> str:
    s = re.sub(LEGAL, " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def load(path: Path) -> tuple[list[dict], dict[str, str], list[dict]]:
    rd = lambda n: [json.loads(x) for x in open(path / f"{n}.jsonl", encoding="utf-8")]  # noqa: E731
    docs = {d["doc_id"]: d["text"] for d in rd("docs")}
    return rd("anchors"), docs, rd("mentions")


def run(path: Path, scorer: str, k: int, n_cal: int, n_ltt: int, seed: int) -> dict[str, Any]:
    t0 = time.time()
    anchors, docs, mentions = load(path)
    a_ids = [a["anchor_id"] for a in anchors]
    a_text = [a["text"] for a in anchors]
    ctx = [
        docs[m["doc_id"]][: m["start"]] + "[" + m["surface"] + "]" + docs[m["doc_id"]][m["end"] :]
        for m in mentions
    ]
    timings: dict[str, float] = {}

    # ---- candidate retrieval -------------------------------------------------------------
    t = time.time()
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
    A = tf.fit_transform([_core(a["name"]) for a in anchors])
    M = tf.transform([_core(m["surface"]) for m in mentions])
    lex = (M @ A.T).toarray()
    lex_top = np.argsort(-lex, axis=1)[:, :k]
    cand = [set(row) for row in lex_top]
    if scorer != "lexical":
        emb = OnnxModel("bge-small")
        ea = emb.embed(a_text)
        em = emb.embed([f"{m['surface']} | {c}" for m, c in zip(mentions, ctx)])
        emb_top = np.argsort(-(em @ ea.T), axis=1)[:, :k]
        for i, row in enumerate(emb_top):
            cand[i] |= set(row)
    timings["retrieve"] = time.time() - t
    gold_idx = {aid: j for j, aid in enumerate(a_ids)}
    in_kb = np.array([m["in_kb"] for m in mentions])
    recall_at_cand = float(np.mean([gold_idx[m["entity_id"]] in cand[i] for i, m in enumerate(mentions) if m["in_kb"]]))

    # ---- scoring ---------------------------------------------------------------------------
    t = time.time()
    flat = [(i, j) for i, cs in enumerate(cand) for j in sorted(cs)]
    if scorer == "lexical":
        s = np.array([lex[i, j] for i, j in flat])
    else:
        ce = OnnxModel(scorer)
        s = ce.score_pairs([(ctx[i], a_text[j]) for i, j in flat])
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
    X = np.column_stack([s1, s1 - s2, lex1])
    correct = np.array([m["in_kb"] and a_ids[top1[i]] == m["entity_id"] for i, m in enumerate(mentions)], dtype=int)

    # ---- decision: calibrated probability that top-1 is the right link ------------------------
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mentions))
    cal, ltt = order[:n_cal], order[n_cal : n_cal + n_ltt]
    lr = LogisticRegression(max_iter=1000).fit(X[cal], correct[cal])
    p = lr.predict_proba(X)[:, 1]
    bands = learn_then_test_bands(p[ltt], correct[ltt], target=0.95, delta=0.05, reject_target=0.99)
    band = np.array([bands.band(x) for x in p])
    acc, rev, rej = band == "accept", band == "review", band == "reject"

    # ---- assignments -----------------------------------------------------------------------
    def assign(with_expert: bool) -> dict[str, str]:
        out: dict[str, str] = {}
        nil: list[int] = []
        for i, m in enumerate(mentions):
            if acc[i]:
                out[m["mention_id"]] = a_ids[top1[i]]
            elif rev[i] and with_expert:
                if correct[i]:
                    out[m["mention_id"]] = a_ids[top1[i]]
                elif m["in_kb"]:
                    out[m["mention_id"]] = m["entity_id"]  # the expert picks the right anchor
                else:
                    nil.append(i)
            else:
                nil.append(i) if (rej[i] or with_expert) else out.__setitem__(m["mention_id"], f"unresolved:{m['mention_id']}")
        for i in nil:  # NIL clustering: same normalised surface core -> one new entity
            out[mentions[i]["mention_id"]] = "nil:" + _core(mentions[i]["surface"])
        return out

    gold = {m["mention_id"]: m["entity_id"] for m in mentions}
    res_machine = assign(False)
    res_expert = assign(True)
    nil_ids = [m["mention_id"] for m in mentions if not m["in_kb"]]

    def nil_quality(pred: dict[str, str]) -> dict[str, float]:
        decided_nil = {mid for mid, v in pred.items() if v.startswith("nil:")}
        true_nil = set(nil_ids)
        tp = len(decided_nil & true_nil)
        return {
            "nil_precision": tp / len(decided_nil) if decided_nil else float("nan"),
            "nil_recall": tp / len(true_nil) if true_nil else float("nan"),
        }

    by_variant: dict[str, dict[str, float]] = {}
    for v in sorted({m["variant"] for m in mentions}):
        idx = [i for i, m in enumerate(mentions) if m["variant"] == v and m["in_kb"]]
        if idx:
            by_variant[v] = {
                "n": len(idx),
                "top1_accuracy": float(np.mean([correct[i] for i in idx])),
                "auto_linked_share": float(np.mean([acc[i] for i in idx])),
            }

    return {
        "data": str(path),
        "scorer": scorer,
        "anchors": len(anchors),
        "mentions": len(mentions),
        "nil_mentions": int((~in_kb).sum()),
        "candidates_per_mention": float(np.mean([len(c) for c in cand])),
        "retrieval_recall_in_kb": recall_at_cand,
        "top1_accuracy_in_kb": float(correct[in_kb].mean()),
        "labels_used": {"calibration": int(len(cal)), "ltt": int(len(ltt))},
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
        "hardware": {"cpus": os.cpu_count(), "gpu": None, "runtime": f"onnxruntime {ort.__version__} CPU"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scorer", choices=["lexical", "bge-reranker-base", "minilm-l6"], required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-cal", type=int, default=800)
    ap.add_argument("--n-ltt", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = run(a.data, a.scorer, a.k, a.n_cal, a.n_ltt, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, default=float))
    print(json.dumps({x: r[x] for x in ("scorer", "retrieval_recall_in_kb", "top1_accuracy_in_kb", "bands", "assignment", "throughput", "timings_s")}, indent=1, default=float))


if __name__ == "__main__":
    main()

"""Load each agreed model once and run a handful of examples through models.py.

This proves the code path (ids resolve, the library calls match, outputs have the expected
shape). It is NOT a measurement: a few hand-written examples say nothing about quality, and no
number printed here may be reported as a result.

Usage: python smoke_models.py [embed] [rerank] [decide] [relex] [glide]   (default: all but glide)
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import models as M

PAIRS = [
    ("Northwind Logistics Ltd | London | EC12 4AB | reg 12345678", "NORTHWIND LOGISTICS LIMITED | London | reg 12-345-678"),
    ("Northwind Logistics Ltd | London | EC12 4AB", "Northwind Logistics 2 Ltd | London | EC12 4AB"),
    ("Silverbrook Software Inc | Boston", "Oakfield Foods PLC | Leeds"),
]
TEXTS = [
    "Northwind Logistics Ltd, based in London, supplies Silverbrook Software with routing data.",
    "Oakfield Foods acquired a division of Ironbridge Manufacturing in 2019.",
]


def _timed(name: str, fn):  # type: ignore[no-untyped-def]
    t = time.time()
    try:
        out = fn()
        print(json.dumps({"stage": name, "ok": True, "secs": round(time.time() - t, 1), "out": out}, default=str))
    except Exception as e:  # noqa: BLE001 - the smoke test reports every failure and carries on
        print(json.dumps({"stage": name, "ok": False, "secs": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:600]}))


def embed() -> dict[str, object]:
    e = M.Embedder()
    v = e.encode([a for a, _ in PAIRS] + [b for _, b in PAIRS], "SentenceSimilarity")
    sims = [float(v[i] @ v[i + len(PAIRS)]) for i in range(len(PAIRS))]
    return {"shape": v.shape, "norms": np.linalg.norm(v, axis=1).round(3).tolist(), "pair_cosine": np.round(sims, 3).tolist(), "revision": M.revision(M.EMBEDDER_ID)}


def rerank() -> dict[str, object]:
    r = M.Reranker(M.SAME_RECORD_INSTRUCTION)
    return {"logits": r.score(PAIRS).round(3).tolist(), "revision": M.revision(M.RERANKER_ID)}


def decide() -> dict[str, object]:
    d = M.GlinerDecide(M.SAME_QUESTION)
    return {"class": type(d.model).__name__, "p_yes": d.score(PAIRS).round(3).tolist(), "revision": M.revision(M.DECIDE_ID)}


def relex() -> dict[str, object]:
    x = M.RelexExtractor()
    ents, rels = x.extract(TEXTS, ["organization", "location", "date"], ["supplies", "acquired", "based in"])
    return {
        "class": type(x.model).__name__,
        "entities": [[(e["text"], e["label"], e.get("start"), e.get("end")) for e in doc] for doc in ents],
        "relations": [[(r["head"]["text"], r["relation"], r["tail"]["text"]) for r in doc] for doc in rels],
        "revision": M.revision(M.RELEX_ID),
    }


def glide() -> dict[str, object]:
    g = M.GlideDecide(M.SAME_QUESTION)
    return {"p_yes": g.score(PAIRS).round(3).tolist()}


if __name__ == "__main__":
    stages = {"embed": embed, "rerank": rerank, "decide": decide, "relex": relex, "glide": glide}
    for name in sys.argv[1:] or ["embed", "rerank", "decide", "relex"]:
        _timed(name, stages[name])

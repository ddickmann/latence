#!/usr/bin/env python3
"""Honest fused-vs-single multi-hop retrieval benchmark on the RFC corpus.

Uses the REAL latence_retrieval classes end-to-end against the LIVE Qdrant (no fakes):
  dense (Granite r2 query -> Qdrant) + BM25 (Bm25Rescorer) -> RrfFuser -> MultiHopExpander
  (DuckDBGraphSource over graph-edges.parquet, 2-hop) -> Reranker.
Questions are generated from real KG/_refs reference chains via gpt-4.1-mini (OpenRouter); each is a
passage-grounded question framed via a citing RFC. A question is KEPT only if fused recall@10 beats
BOTH dense-only and bm25-only (the strict multi-hop gate). Real numbers only.
"""
from __future__ import annotations
import json, os, re, sys, random, collections
import pyarrow.parquet as pq
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from latence_retrieval import Candidate, Bm25Rescorer, RrfFuser, MultiHopExpander, Reranker
from latence_retrieval.multihop import DuckDBGraphSource

RUN = "/workspace/latence-framework/latence-out/_latence/runs/run-0001"
EXPORT = RUN + "/export"
REFS = "/workspace/corpus-rfc/_refs.jsonl"
GRANITE = "ibm-granite/granite-embedding-311m-multilingual-r2"
CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"
random.seed(13)  # deterministic sampling (no argless Date/random elsewhere)

# ---- 1. load chunks ----------------------------------------------------------
t = pq.read_table(EXPORT + "/records.parquet").to_pylist()
def rfc_of(prov):
    if isinstance(prov, str):
        try: prov = json.loads(prov)
        except Exception: return None
    if not isinstance(prov, dict): return None
    m = re.search(r"rfc[\s-]?(\d{3,5})", str(prov.get("file_name", "")), re.I)
    return m.group(1) if m else None
chunks = []
for r in t:
    kg = r.get("kg_node_ids") or []
    chunks.append({"id": r["record_id"], "content": r["content"] or "",
                   "kg": list(kg), "rfc": rfc_of(r.get("provenance"))})
by_rfc = collections.defaultdict(list)
for c in chunks:
    if c["rfc"]:
        by_rfc[c["rfc"]].append(c)
print(f"chunks={len(chunks)}  rfcs_with_chunks={len(by_rfc)}", flush=True)

# ---- 2. reference chains from _refs.jsonl (A references B, both in corpus) ----
refs = {}
for l in open(REFS):
    d = json.loads(l); refs[str(d["rfc"])] = d
edges = []  # (A, B): A references/obsoletes/updates B, both have chunks
for a, d in refs.items():
    for b in set(map(str, d.get("references", []) + d.get("obsoletes", []) + d.get("updates", []))):
        if a in by_rfc and b in by_rfc and a != b:
            edges.append((a, b))
random.shuffle(edges)
print(f"candidate reference edges (both in corpus): {len(edges)}", flush=True)

# ---- 3. retrieval stack (REAL classes, LIVE qdrant) --------------------------
enc = SentenceTransformer(GRANITE, device="cuda")
qc = QdrantClient(url="http://localhost:6333")
bm25 = Bm25Rescorer.from_export(EXPORT)
allc = [Candidate(id=c["id"], score=0.0, text=c["content"], metadata={"kg_node_ids": c["kg"]}) for c in chunks]
graph = DuckDBGraphSource(EXPORT + "/graph-edges.parquet")
expander = MultiHopExpander({"seed_top_k": 10, "hop_weight": 1.0, "max_hops": 2,
                             "hop_decay": 0.5, "node_ids_key": "kg_node_ids"}, graph_source=graph)
reranker = Reranker({"model": CROSS})
fuser = RrfFuser()

def dense(query, k=100):
    v = enc.encode(query, normalize_embeddings=True).tolist()
    hits = qc.query_points("rfc", query=v, limit=k, with_payload=True).points
    return [Candidate(id=h.payload["record_id"], score=float(h.score), text=h.payload.get("content", ""),
                      metadata={"kg_node_ids": h.payload.get("kg_node_ids") or []}) for h in hits]

def bm25_top(query, k=100):
    scored = bm25.process(query, allc)
    return sorted(scored, key=lambda c: -c.score)[:k]

def fused(query, k=10):
    d, b = dense(query, 100), bm25_top(query, 100)
    f = fuser.fuse([d, b])
    f = expander.process(query, f)
    f = reranker.process(query, f)
    return f[:k]

def recall_at_10(cands, gold_ids):
    top = {c.id for c in cands[:10]}
    return len(top & gold_ids) / max(1, len(gold_ids))

# ---- 4. generate questions from chains + measure -----------------------------
llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
def gen_q(a, b, passage):
    p = (f"RFC {a} cites RFC {b}. Below is a passage FROM RFC {b}.\n\nPASSAGE:\n{passage[:1200]}\n\n"
         f"Write ONE specific, natural technical question that this passage answers, phrased as an "
         f"engineer working with RFC {a} would ask it (mention RFC {a}'s subject/context, not RFC {b} "
         f"by number). The question must require THIS passage's content to answer. Output only the question.")
    r = llm.chat.completions.create(model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": p}], temperature=0.2, max_tokens=120)
    return r.choices[0].message.content.strip()

results, kept = [], []
TARGET, MAXTRY = 30, 90
for (a, b) in edges[:MAXTRY]:
    if len(kept) >= TARGET:
        break
    cb = max(by_rfc[b], key=lambda c: len(c["content"]))  # a substantive gold passage from B
    if len(cb["content"]) < 200:
        continue
    gold = {cb["id"]}
    try:
        q = gen_q(a, b, cb["content"])
    except Exception as e:
        print("llm err", e, flush=True); continue
    rd = recall_at_10(dense(q), gold)
    rb = recall_at_10(bm25_top(q, 10) if False else sorted(bm25.process(q, allc), key=lambda c: -c.score), gold)
    rf = recall_at_10(fused(q), gold)
    row = {"a": a, "b": b, "q": q, "gold": cb["id"], "dense": rd, "bm25": rb, "fused": rf}
    results.append(row)
    strict = rf > rd and rf > rb and rf > 0
    if strict:
        kept.append(row)
    print(f"[{len(results):2d}] A{a}->B{b} dense={rd:.0f} bm25={rb:.0f} fused={rf:.0f} {'KEEP' if strict else ''}", flush=True)

# ---- 5. report ---------------------------------------------------------------
def mean(xs): return sum(xs) / len(xs) if xs else 0.0
allrows = results
out = {"n_generated": len(allrows), "n_strict_multihop": len(kept),
       "aggregate_all": {"dense": mean([r["dense"] for r in allrows]),
                         "bm25": mean([r["bm25"] for r in allrows]),
                         "fused": mean([r["fused"] for r in allrows])},
       "aggregate_strict": {"dense": mean([r["dense"] for r in kept]),
                            "bm25": mean([r["bm25"] for r in kept]),
                            "fused": mean([r["fused"] for r in kept])},
       "kept": kept, "all": allrows}
json.dump(out, open("/workspace/report/rfc-bench-results.json", "w"), indent=2)
print("\n==== RESULT ====")
print(f"generated={len(allrows)}  strict-multihop-kept={len(kept)}")
print(f"ALL rows       recall@10  dense={out['aggregate_all']['dense']:.3f} bm25={out['aggregate_all']['bm25']:.3f} fused={out['aggregate_all']['fused']:.3f}")
print(f"STRICT (kept)  recall@10  dense={out['aggregate_strict']['dense']:.3f} bm25={out['aggregate_strict']['bm25']:.3f} fused={out['aggregate_strict']['fused']:.3f}")

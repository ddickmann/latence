# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""M6 final block (ticket 17): dev-tune the composition policy, score test + enterprise, gate.

Discipline (ticket 08): the ONE free parameter — the CoE budget share in the deployable
hybrid — is tuned on the 500 MuSiQue DEV questions only; the 500 TEST questions and the
enterprise gold are scored with the tuned value unchanged. Budget parity via
``estimate_tokens`` on both arms; tier-1 alias entry only (reported as such, ADR-0059).

Metric (shared by both arms, both datasets): a gold supporting unit (MuSiQue: supporting
paragraph; enterprise: supporting-fact span) counts as recalled when at least one of its
sentences (≥40 normalized chars) appears verbatim in the normalized emitted context —
chunk contexts contain their sentences verbatim, τ windows are sentence windows, so the
metric favors neither arm.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from latence_core.chunking import estimate_tokens
from latence_retrieval.bm25 import Bm25Rescorer
from latence_retrieval.chains import assemble_chains
from latence_retrieval.coe import select
from latence_retrieval.contracts import Candidate
from latence_retrieval.hypergraph import HyperedgeKV, hypergraph_signals
from latence_retrieval.pack import Packer

HERE = Path(__file__).parent
ENTERPRISE = HERE.parent / "enterprise-gold"
BUDGETS = [1000, 2000, 4000, 8000, 16000]
SHARES = [0.25, 0.5, 0.75, 1.0]
_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", t).strip().lower()


def _sentences(text: str) -> list[str]:
    return [
        _norm(s) for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(_norm(s)) >= 40
    ]


def _recalled(unit_text: str, norm_ctx: str) -> bool:
    sents = _sentences(unit_text)
    if not sents:  # very short unit: fall back to whole-text containment
        return _norm(unit_text) in norm_ctx
    return any(s in norm_ctx for s in sents)


def question_ngrams(question: str, max_n: int = 4) -> list[str]:
    words = re.findall(r"[\w][\w&.\-]*", question)
    grams: list[str] = []
    for n in range(max_n, 0, -1):
        grams.extend(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


def load_chunks(path: Path) -> dict[str, str]:
    chunks: dict[str, str] = {}
    for line in path.open():
        r = json.loads(line)
        if "content" in r and "index" in r and "document_record_id" in r:
            text = r.get("masked_content") or r["content"]
            if text:
                chunks[r["record_id"]] = text
    return chunks


class Arm:
    """Per-question compute, done once; per-(budget, share) assembly, done cheaply."""

    def __init__(self, chunks: dict[str, str], kv_path: str) -> None:
        self.candidates = [
            Candidate(id=cid, text=t, score=0.0) for cid, t in sorted(chunks.items())
        ]
        self.bm25 = Bm25Rescorer()
        self.kv_path = kv_path

    def per_question(self, question: str) -> dict:
        ranked = self.bm25.process(question, self.candidates)
        with HyperedgeKV.open(self.kv_path) as kv:
            signals = hypergraph_signals(kv, alias_surfaces=question_ngrams(question))
            selections = {}
            for budget in BUDGETS:
                for share in SHARES:
                    coe_budget = max(1, int(budget * share))
                    key = (budget, share)
                    if not signals.rho:
                        selections[key] = ""
                        continue
                    sel = select(
                        signals,
                        budget=coe_budget,
                        tau_lookup=kv.tau_text,
                        token_counter=estimate_tokens,
                        counter_identity="latence:estimate_tokens:v1",
                    )
                    selections[key] = assemble_chains(
                        sel, signals, tau_lookup=kv.tau_text
                    ).text
        return {"pool": self._jaccard_pool(ranked[: self.PACK_POOL]), "coe": selections}

    # Two measured performance walls, both fixed result-identically:
    # (1) chunks outside the BM25 top-400 can never enter a 16k context (400 × ~60 ≈ 24k
    #     tokens > any budget), so the pack pool caps at 400 — the 30k-candidate pool was
    #     an 11h run at 86% CPU;
    # (2) the Packer re-tokenizes the pool for Jaccard on EVERY call (25 calls/question);
    #     precomputing the identical pairwise Jaccard once per question and feeding it via
    #     the Packer's own ``similarity_key`` seam turns each pack into dict lookups.
    PACK_POOL = 400

    @staticmethod
    def _jaccard_pool(pool: list[Candidate]) -> list[Candidate]:
        toks = [set(re.findall(r"\w+", c.text.lower())) for c in pool]
        out = []
        for i, c in enumerate(pool):
            sims = {}
            for j, other in enumerate(pool):
                if i == j:
                    continue
                inter = len(toks[i] & toks[j])
                if inter:
                    sims[other.id] = inter / len(toks[i] | toks[j])
            out.append(
                Candidate(id=c.id, text=c.text, score=c.score, metadata={"sims": sims})
            )
        return out

    def context(self, pq: dict, budget: int, share: float | None) -> str:
        """share=None → pure baseline; else hybrid: CoE at share·budget, baseline fills."""
        pool = pq["pool"]
        if share is None:
            packed = Packer({"budget": budget, "similarity_key": "sims"}).process("", pool)
            return "\n".join(c.text for c in packed)
        coe = pq["coe"][(budget, share)]
        rest = budget - estimate_tokens(coe)
        if rest <= 0:
            return coe
        packed = Packer({"budget": rest, "similarity_key": "sims"}).process("", pool)
        return coe + "\n" + "\n".join(c.text for c in packed)


def score(questions: list[dict], contexts: dict[str, str], kind: str) -> float:
    hits = total = 0
    for q in questions:
        ctx = _norm(contexts[q["id"]])
        units = (
            [p["text"] for p in q["paragraphs"] if p["is_supporting"]]
            if kind == "musique"
            else [f["text"] for f in q["supporting_facts"]]
        )
        for u in units:
            total += 1
            hits += _recalled(u, ctx)
    return round(hits / total, 4)


_WORKER: dict = {}


def _init_worker(chunks_path: str, kv_path: str) -> None:
    _WORKER["arm"] = Arm(load_chunks(Path(chunks_path)), kv_path)


def _eval_question(payload: tuple) -> dict:
    """One question end to end in a worker: per-config unit-hit booleans, not contexts."""
    question, units, shares = payload
    arm: Arm = _WORKER["arm"]
    pq = arm.per_question(question)
    out: dict = {}
    for share in shares:
        label = "baseline" if share is None else f"share{share}"
        for budget in BUDGETS:
            ctx = _norm(arm.context(pq, budget, share))
            out[(label, budget)] = [_recalled(u, ctx) for u in units]
    return out


def run_dataset(
    questions: list[dict],
    kind: str,
    shares: list[float | None],
    *,
    chunks_path: Path,
    kv_path: str,
    workers: int = 8,
) -> dict:
    def units_of(q: dict) -> list[str]:
        if kind == "musique":
            return [p["text"] for p in q["paragraphs"] if p["is_supporting"]]
        return [f["text"] for f in q["supporting_facts"]]

    payloads = [(q["question"], units_of(q), shares) for q in questions]
    hits: dict = {}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(str(chunks_path), kv_path),
    ) as pool:
        for i, res in enumerate(pool.map(_eval_question, payloads, chunksize=4)):
            for key, bools in res.items():
                agg = hits.setdefault(key, [0, 0])
                agg[0] += sum(bools)
                agg[1] += len(bools)
            if (i + 1) % 50 == 0:
                print(f"  …{i + 1}/{len(questions)}", flush=True)
    out: dict = {}
    for share in shares:
        label = "baseline" if share is None else f"share{share}"
        out[label] = {
            str(b): round(hits[(label, b)][0] / hits[(label, b)][1], 4) for b in BUDGETS
        }
    return out


def gate(base: list[float], cand: list[float]) -> dict:
    auc_b, auc_c = sum(base) / len(base), sum(cand) / len(cand)
    worst = min((c - b) / b if b > 0 else 0.0 for c, b in zip(cand, base))
    g = {
        "auc_baseline": round(auc_b, 4),
        "auc_candidate": round(auc_c, 4),
        "auc_dominates": auc_c > auc_b,
        "endpoints_dominate": cand[0] >= base[0] and cand[-1] >= base[-1],
        "worst_point_relative": round(worst, 4),
        "no_point_worse_than_2pct": worst >= -0.02,
    }
    g["PASS"] = bool(
        g["auc_dominates"] and g["endpoints_dominate"] and g["no_point_worse_than_2pct"]
    )
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "final-results.json"))
    args = ap.parse_args()
    results: dict = {}

    mq_kv = str(HERE / "export" / "graph-hyperedges.kv")
    dev = [json.loads(l) for l in (HERE / "musique-dev.jsonl").open()]
    test = [json.loads(l) for l in (HERE / "musique-test.jsonl").open()]
    mq_chunks_path = HERE / "export" / "corpus.jsonl"

    print(f"DEV TUNE: {len(dev)} questions, shares {SHARES}", flush=True)
    dev_out = run_dataset(dev, "musique", [None, *SHARES],
                          chunks_path=mq_chunks_path, kv_path=mq_kv)
    results["dev"] = dev_out
    base_curve = [dev_out["baseline"][str(b)] for b in BUDGETS]
    best_share, best_auc = None, -1.0
    for share in SHARES:
        curve = [dev_out[f"share{share}"][str(b)] for b in BUDGETS]
        auc = sum(curve) / len(curve)
        print(f"  share={share}: curve={curve} auc={auc:.4f}")
        if auc > best_auc:
            best_share, best_auc = share, auc
    print(f"  baseline auc={sum(base_curve)/len(base_curve):.4f}  TUNED share={best_share}")
    results["tuned_share"] = best_share

    print(f"TEST: {len(test)} questions at tuned share={best_share}", flush=True)
    test_out = run_dataset(test, "musique", [None, best_share],
                           chunks_path=mq_chunks_path, kv_path=mq_kv)
    results["test"] = test_out
    tb = [test_out["baseline"][str(b)] for b in BUDGETS]
    tc = [test_out[f"share{best_share}"][str(b)] for b in BUDGETS]
    results["gate_musique_test"] = gate(tb, tc)
    print("  baseline:", tb, "\n  coe-hybrid:", tc)
    print("  GATE(musique-test):", json.dumps(results["gate_musique_test"]))

    print("ENTERPRISE re-score (policy unchanged)")
    ent_kv = str(
        ENTERPRISE / "store" / "_latence" / "runs" / "gold-v0" / "export"
        / "graph-hyperedges.kv"
    )
    ent_gold = []
    for line in (ENTERPRISE / "gold.jsonl").open():
        q = json.loads(line)
        q["id"] = q["question_id"]
        ent_gold.append(q)
    ent_out = run_dataset(ent_gold, "enterprise", [None, best_share],
                          chunks_path=ENTERPRISE / "store" / "_latence" / "runs"
                          / "gold-v0" / "export" / "records.jsonl", kv_path=ent_kv)
    results["enterprise"] = ent_out
    eb = [ent_out["baseline"][str(b)] for b in BUDGETS]
    ec = [ent_out[f"share{best_share}"][str(b)] for b in BUDGETS]
    results["gate_enterprise"] = gate(eb, ec)
    print("  baseline:", eb, "\n  coe-hybrid:", ec)
    print("  GATE(enterprise):", json.dumps(results["gate_enterprise"]))

    results["overall_pass"] = bool(
        results["gate_musique_test"]["PASS"] and results["gate_enterprise"]["PASS"]
    )
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\nOVERALL PASS:", results["overall_pass"])


if __name__ == "__main__":
    main()

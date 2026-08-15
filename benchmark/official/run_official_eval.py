# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""M6 OFFICIAL scoring driver: two full corpora, per-tier dev curves, four A/Bs, gates.

Discipline (ticket 08): every axis — entry tier, CoE budget share, weak on/off, exact vs
heuristic token costs, calibrated γ — is explored on the DEV split only; the pre-registered
TEST scoring is baseline vs the single dev-best CoE config per dataset, scored once, gated
AUC+endpoints (piecewise policy allowed exactly as in the subset gate: any budget point may
fall back to pure baseline when dev says so).

Baseline BM25 uses an inverted index with the SAME scoring math and constants as the
shipped `Bm25Rescorer` (Lucene IDF, k1=1.5, b=0.75) — indexed because rescoring 160k chunks
per question is the measured 11h wall; scores are identical, only the lookup differs.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "musique-subset"))

from latence_core.chunking import estimate_tokens  # noqa: E402
from latence_retrieval.chains import assemble_chains  # noqa: E402
from latence_retrieval.coe import select  # noqa: E402
from latence_retrieval.contracts import Candidate  # noqa: E402
from latence_retrieval.hypergraph import HyperedgeKV  # noqa: E402
from latence_retrieval.pack import Packer  # noqa: E402
from official_eval import (  # noqa: E402
    Tier0Index,
    apply_calibration,
    build_exact_kv,
    entry_signals,
    filter_weak,
)
from run_eval_final import Arm, _norm, _recalled, gate  # noqa: E402

BUDGETS = [1000, 2000, 4000, 8000, 16000]
K1, B = 1.5, 0.75
_TOKEN_RE = re.compile(r"\w+")


class BM25Index:
    """Inverted-index BM25 with Bm25Rescorer's exact scoring math (indexed lookup)."""

    def __init__(self, chunks: dict[str, str]) -> None:
        self.ids = sorted(chunks)
        self.texts = [chunks[i] for i in self.ids]
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        df: Counter = Counter()
        for t in self.texts:
            toks = _TOKEN_RE.findall(t.lower())
            c = Counter(toks)
            self.tf.append(c)
            self.dl.append(len(toks))
            df.update(c.keys())
        self.n = len(self.texts)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0
        self.idf = {
            t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings.setdefault(t, []).append(i)

    def rank(self, query: str, top: int = 400) -> list[Candidate]:
        scores: dict[int, float] = {}
        for t in set(_TOKEN_RE.findall(query.lower())):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                tf = self.tf[i][t]
                denom = tf + K1 * (1 - B + B * self.dl[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [
            Candidate(id=self.ids[i], text=self.texts[i], score=s) for i, s in ranked
        ]


# -- worker ----------------------------------------------------------------------

_W: dict = {}


def _init(base: str, kv_path: str, exact_kv_path: str, calib_path: str) -> None:
    base_p = Path(base)
    chunks = {}
    for line in (base_p / "export" / "corpus.jsonl").open():
        r = json.loads(line)
        if "content" in r and "index" in r and "document_record_id" in r:
            text = r.get("masked_content") or r["content"]
            if text:
                chunks[r["record_id"]] = text
    _W["bm25"] = BM25Index(chunks)
    _W["kv_path"] = kv_path
    _W["exact_kv_path"] = exact_kv_path
    _W["tier0"] = Tier0Index(str(base_p / "export" / "graph-nodes.parquet"))
    _W["surfaces"] = {
        json.loads(l)["id"]: json.loads(l)["surfaces"]
        for l in (base_p / "query-surfaces.jsonl").open()
    }
    _W["calibrate"] = None
    if calib_path and Path(calib_path).exists():
        maps = json.loads(Path(calib_path).read_text())

        def _interp(m: dict, g: float) -> float:
            import bisect

            xs, ys = m["x"], m["y"]
            j = bisect.bisect_right(xs, g)
            if j == 0:
                return ys[0]
            if j >= len(xs):
                return ys[-1]
            x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
            return y0 + (y1 - y0) * ((g - x0) / (x1 - x0) if x1 > x0 else 0.0)

        def calibrate(g: float, has_relations: bool) -> float:
            regime = "strong" if has_relations else "weak"
            return _interp(maps[regime], g)

        _W["calibrate"] = calibrate


def _pack_sequence(pool: list[Candidate]) -> list[tuple[str, int]]:
    """ONE 16k QKP-greedy pack per question; every budget is its prefix.

    Profiling showed the shipped Packer's greedy loop at 0.43s/call × 25 calls/question
    (2.36M similarity lookups) — the grid's dominant cost. Packing once at the largest
    budget and truncating the pick sequence per budget is a uniform, disclosed
    simplification applied to EVERY arm (baseline and fallback alike); it slightly favors
    the baseline if anything (no budget-boundary waste)."""
    packed = Packer({"budget": 16000, "similarity_key": "sims"}).process("", pool)
    return [(c.text, max(1, estimate_tokens(c.text))) for c in packed]


def _ctx_from_seq(seq: list[tuple[str, int]], budget: int) -> str:
    out, used = [], 0
    for text, cost in seq:
        if used + cost > budget:
            break
        out.append(text)
        used += cost
    return "\n".join(out)


def _eval_q(payload: tuple) -> dict:
    q, configs = payload
    units = [p["text"] for p in q["paragraphs"] if p["is_supporting"]]
    bm25: BM25Index = _W["bm25"]
    seq = _pack_sequence(Arm._jaccard_pool(bm25.rank(q["question"])))
    out: dict = {}
    kv_paths = {"heuristic": _W["kv_path"], "exact": _W["exact_kv_path"]}
    sig_cache: dict = {}
    for cfg in configs:
        name = cfg["name"]
        if cfg.get("baseline"):
            for budget in BUDGETS:
                ctx = _norm(_ctx_from_seq(seq, budget))
                out[(name, budget)] = [_recalled(u, ctx) for u in units]
            continue
        skey = (cfg["tier"], cfg["costs"], cfg["weak"], cfg["calib"])
        if skey not in sig_cache:
            with HyperedgeKV.open(kv_paths[cfg["costs"]]) as kv:
                sig = entry_signals(kv, q, _W["surfaces"], cfg["tier"], _W["tier0"])
                if cfg["weak"] == "off":
                    sig = filter_weak(sig)
                if cfg["calib"] == "on" and _W["calibrate"]:
                    sig = apply_calibration(sig, _W["calibrate"])
                sels = {}
                for budget in BUDGETS:
                    share = cfg["shares"].get(str(budget), 0.0)
                    if share <= 0 or not sig.rho:
                        sels[budget] = ""
                        continue
                    # ADR-0058 discipline: the verification counter must BE the identity it
                    # claims. The exact-KV arm stores tiktoken counts, so its verification
                    # triple is tiktoken too (close-out audit fix — the earlier code passed
                    # estimate_tokens labeled as tiktoken; inert because matching identities
                    # skip the optional pass, but wrong if verify_on_match were enabled).
                    if cfg["costs"] == "heuristic":
                        v_counter, v_identity = (
                            estimate_tokens, "latence:estimate_tokens:v1",
                        )
                    else:
                        import tiktoken

                        _enc = tiktoken.get_encoding("cl100k_base")
                        v_counter, v_identity = (
                            lambda t, _e=_enc: len(_e.encode(t)),
                            "tiktoken:cl100k_base",
                        )
                    sel = select(
                        sig,
                        budget=max(1, int(budget * share)),
                        tau_lookup=kv.tau_text,
                        token_counter=v_counter,
                        counter_identity=v_identity,
                    )
                    sels[budget] = assemble_chains(sel, sig, tau_lookup=kv.tau_text).text
                sig_cache[skey] = sels
        sels = sig_cache[skey]
        for budget in BUDGETS:
            coe = sels[budget]
            rest = budget - estimate_tokens(coe) if coe else budget
            ctx = coe + ("\n" + _ctx_from_seq(seq, rest) if rest > 0 else "")
            out[(name, budget)] = [_recalled(u, _norm(ctx)) for u in units]
    return out


def run(questions: list[dict], configs: list[dict], base: Path, exact_kv: str,
        calib: str, workers: int = 8, tag: str = "") -> dict:
    payloads = [(q, configs) for q in questions]
    agg: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init,
        initargs=(str(base), str(base / "export" / "graph-hyperedges.kv"), exact_kv, calib),
    ) as pool:
        for i, res in enumerate(pool.map(_eval_q, payloads, chunksize=4)):
            for key, bools in res.items():
                a = agg.setdefault(key, [0, 0])
                a[0] += sum(bools)
                a[1] += len(bools)
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i + 1}/{len(questions)} ({time.time()-t0:.0f}s)", flush=True)
    out: dict = {}
    for (name, budget), (h, t) in agg.items():
        out.setdefault(name, {})[str(budget)] = round(h / t, 4)
    return out


def main() -> None:
    results: dict = {}
    for ds in ("musique-full", "hotpot-full"):
        base = HERE.parent / "official-data" / ds
        dev = [json.loads(l) for l in (base / f"{ds}-dev.jsonl").open()]
        test = [json.loads(l) for l in (base / f"{ds}-test.jsonl").open()]
        exact_kv = str(base / "export" / "graph-hyperedges-exact.kv")
        if not Path(exact_kv).exists():
            print(f"[{ds}] building exact-cost KV…", flush=True)
            build_exact_kv(
                str(base / "export" / "graph-hyperedges.parquet"),
                str(base / "export" / "graph-nodes.parquet"),
                exact_kv,
            )
        calib = str(base / "calibration-map.json")
        flat = {str(b): 0.25 for b in BUDGETS}

        # DEV: tier sweep + A/Bs, all at flat share 0.25 (the subset-gate share), plus
        # share 0.5 on the best tier — a small, pre-registered grid.
        dev_cfgs = [{"name": "baseline", "baseline": True}]
        for tier in ("tier0", "tier1", "tier2", "tier12"):
            dev_cfgs.append({"name": f"{tier}", "tier": tier, "costs": "heuristic",
                             "weak": "on", "calib": "off", "shares": flat})
        dev_cfgs += [
            {"name": "tier12-weakoff", "tier": "tier12", "costs": "heuristic",
             "weak": "off", "calib": "off", "shares": flat},
            {"name": "tier12-exact", "tier": "tier12", "costs": "exact",
             "weak": "on", "calib": "off", "shares": flat},
            *([{"name": "tier12-calib", "tier": "tier12", "costs": "heuristic",
                "weak": "on", "calib": "on", "shares": flat}]
              if Path(calib).exists()
              else (print(f"[{ds}] calibration map absent — calib A/B deferred, NOT silently "
                          "scored uncalibrated", flush=True) or [])),
            {"name": "tier12-s50", "tier": "tier12", "costs": "heuristic",
             "weak": "on", "calib": "off", "shares": {str(b): 0.5 for b in BUDGETS}},
        ]
        print(f"[{ds}] DEV: {len(dev)} questions × {len(dev_cfgs)} configs", flush=True)
        dev_out = run(dev, dev_cfgs, base, exact_kv, calib, tag=f"{ds}-dev")
        results[f"{ds}-dev"] = dev_out

        base_curve = [dev_out["baseline"][str(b)] for b in BUDGETS]
        best_name, best_auc = None, -1.0
        for name, curves in dev_out.items():
            if name == "baseline":
                continue
            auc = sum(curves[str(b)] for b in BUDGETS) / len(BUDGETS)
            print(f"  [{ds}] {name}: auc={auc:.4f} curve={[curves[str(b)] for b in BUDGETS]}",
                  flush=True)
            if auc > best_auc:
                best_name, best_auc = name, auc
        # piecewise: per budget, use CoE only where dev beats baseline (subset-gate policy)
        best_cfg = next(c for c in dev_cfgs if c["name"] == best_name)
        shares = dict(best_cfg["shares"])
        for b in BUDGETS:
            if dev_out[best_name][str(b)] < dev_out["baseline"][str(b)]:
                shares[str(b)] = 0.0
        tuned = {**best_cfg, "name": "tuned", "shares": shares}
        print(f"  [{ds}] TUNED: {best_name} with piecewise shares {shares}", flush=True)
        results[f"{ds}-tuned-config"] = {k: v for k, v in tuned.items() if k != "baseline"}

        print(f"[{ds}] TEST: {len(test)} questions (single scoring)", flush=True)
        test_out = run(test, [{"name": "baseline", "baseline": True}, tuned],
                       base, exact_kv, calib, tag=f"{ds}-test")
        results[f"{ds}-test"] = test_out
        tb = [test_out["baseline"][str(b)] for b in BUDGETS]
        tc = [test_out["tuned"][str(b)] for b in BUDGETS]
        results[f"{ds}-gate"] = gate(tb, tc)
        print(f"  [{ds}] baseline: {tb}\n  [{ds}] tuned:    {tc}", flush=True)
        print(f"  [{ds}] GATE: {json.dumps(results[f'{ds}-gate'])}", flush=True)

    results["overall_pass"] = all(
        results[f"{ds}-gate"]["PASS"] for ds in ("musique-full", "hotpot-full")
    )
    (HERE / "official-results.json").write_text(json.dumps(results, indent=2))
    print("\nOFFICIAL OVERALL PASS:", results["overall_pass"], flush=True)


if __name__ == "__main__":
    main()

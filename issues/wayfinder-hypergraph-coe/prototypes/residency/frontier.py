"""Throwaway prototype — evidence for wayfinder ticket 20 (frontier headroom).

NOT product code. Extends sim.py's synthetic harness with a QUALITY axis so knobs can be
traded against recall instead of guessed: multi-hop chains with known ground-truth Hyperedges
are planted into the corpus, plus lexical distractors, and we measure supporting-fact
recall@budget — the same metric shape as the M6 gate.

Sweeps: |E_c|, RWR steps K, 2-hop expansion, weighted vs binary incidence, heat-kernel vs RWR,
and the index-time cost of truncated PPR sketches.

Run:  uv run --with numpy --with scipy --with psutil python frontier.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from sim import (
    BUDGET_TOKENS,
    W_CONN,
    W_ENT,
    W_EXT,
    W_PROP,
    W_REL,
    W_SUP,
    THETA_FRAC,
    Corpus,
    generate,
)

ENTRY_NOISE = 9  # noise anchors alongside the true chain seed (simulates ANN recall noise)


# --- planted ground truth --------------------------------------------------
@dataclass
class Planted:
    qid: int
    seed_entity: int
    chain_edges: list[int]     # ground-truth Hyperedges (the "supporting facts")
    distractors: list[int]     # lexically similar, off-chain


def plant(corpus: Corpus, n_chains: int, hops: int, seed: int = 4242) -> tuple[Corpus, list[Planted]]:
    """Append chains of Hyperedges linked by shared entities, plus lexical distractors.

    Chain i: e0 -[H0]- e1 -[H1]- e2 ... so recovering it needs `hops` propagation steps.
    Only e0 is given to the query, so H1..H(n-1) are reachable ONLY through propagation --
    exactly the capability the whole design exists to provide.
    """
    rng = np.random.default_rng(seed)
    ent_flat = list(corpus.ent_flat)
    offsets = list(corpus.ent_offsets)
    tok = list(corpus.token_cost)
    gam = list(corpus.gamma)
    sup = list(corpus.support)
    n = corpus.n_eus
    planted: list[Planted] = []

    # use fresh entity ids beyond the vocabulary so chains are not polluted by hubs
    next_ent = corpus.n_entities
    for qid in range(n_chains):
        chain_ents = list(range(next_ent, next_ent + hops + 1))
        next_ent += hops + 1
        chain_edges = []
        for h in range(hops):
            ents = [chain_ents[h], chain_ents[h + 1], int(rng.integers(0, corpus.n_entities))]
            ent_flat.extend(ents)
            offsets.append(offsets[-1] + len(ents))
            tok.append(int(rng.integers(60, 100)))
            gam.append(0.80)
            sup.append(1)
            chain_edges.append(n)
            n += 1
        # distractors: share the SEED entity but lead nowhere (the trap for pure similarity)
        distractors = []
        for _ in range(3):
            ents = [chain_ents[0], int(rng.integers(0, corpus.n_entities))]
            ent_flat.extend(ents)
            offsets.append(offsets[-1] + len(ents))
            tok.append(int(rng.integers(60, 100)))
            gam.append(0.95)  # HIGHER gamma than the chain: must be beaten on structure, not score
            sup.append(3)
            distractors.append(n)
            n += 1
        planted.append(Planted(qid, chain_ents[0], chain_edges, distractors))

    c = Corpus(
        n_eus=n,
        n_entities=next_ent,
        ent_flat=np.array(ent_flat, dtype=np.int32),
        ent_offsets=np.array(offsets, dtype=np.int64),
        token_cost=np.array(tok, dtype=np.int32),
        gamma=np.array(gam, dtype=np.float32),
        support=np.array(sup, dtype=np.int32),
    )
    # rebuild postings
    order = np.argsort(c.ent_flat, kind="stable")
    sorted_ent = c.ent_flat[order]
    edge_of_ref = np.repeat(np.arange(c.n_eus, dtype=np.int64), np.diff(c.ent_offsets))
    sorted_edges = edge_of_ref[order]
    bounds = np.searchsorted(sorted_ent, np.arange(c.n_entities + 1))
    for v in range(c.n_entities):
        edges = sorted_edges[bounds[v] : bounds[v + 1]]
        if edges.size:
            c.postings[v] = edges[np.argsort(-c.gamma[edges], kind="stable")]
    return c, planted


# --- parameterised online path --------------------------------------------
def candidates(
    c: Corpus, v0: list[int], cap: int, hops: int,
    hub_cap: int = 2000, expand_cap: int = 3000,
) -> np.ndarray:
    """k-hop candidate universe over the postings (global A_E is gone, ADR-0056).

    Two guards make deep expansion affordable, and both are findings in their own right:

    * **hub guard** — an entity in >`hub_cap` Hyperedges contributes only its top-γ slice.
      Without it a single hub (366k edges at 1M EUs) drags the whole corpus into `E_c`.
    * **hop-priority truncation** — when the cap bites, DROP THE OUTERMOST HOP FIRST and
      truncate within it by γ. Truncating globally by γ discards exactly the bridging
      evidence propagation exists to surface (chain γ=0.80 loses to distractor γ=0.95),
      which silently converts a multi-hop retriever into a 1-hop one.
    """
    seen: set[int] = set()
    by_hop: list[list[int]] = []
    frontier = {int(v) for v in v0}
    for h in range(hops):
        new: list[int] = []
        for v in frontier:
            p = c.postings.get(v)
            if p is None:
                continue
            if p.size > hub_cap:
                p = p[:hub_cap]  # postings are γ-sorted
            for e in p:
                ei = int(e)
                if ei not in seen:
                    seen.add(ei)
                    new.append(ei)
        if not new:
            break
        by_hop.append(new)
        if h + 1 < hops:
            # Expand from the MOST PROMISING edges, not an arbitrary slice. The spec expands
            # "top edges by adjacency to E_c" -- a ranked notion. Ranking by (contains a V0
            # entity, then gamma) keeps a noisy ANN anchor set from crowding out the edges
            # that actually touch the query's own entities.
            arr = np.array(new, dtype=np.int64)
            v0set = {int(v) for v in v0}
            hit = np.array(
                [1 if v0set & {int(x) for x in c.entities_of(int(e))} else 0 for e in arr],
                dtype=np.int8,
            )
            order = np.lexsort((-c.gamma[arr], -hit))
            nxt: set[int] = set()
            for e in arr[order][:expand_cap]:
                nxt.update(int(x) for x in c.entities_of(int(e)))
            frontier = nxt

    out: list[int] = []
    for layer in by_hop:  # inner hops first — they are never sacrificed to the cap
        room = cap - len(out)
        if room <= 0:
            break
        if len(layer) > room:
            arr = np.array(layer, dtype=np.int64)
            layer = list(arr[np.argsort(-c.gamma[arr], kind="stable")[:room]])
        out.extend(int(x) for x in layer)
    cand = np.array(sorted(out), dtype=np.int64)
    return cand


def build_B(
    cand: np.ndarray, ent_lists: list[np.ndarray], weighted: bool, rng: np.random.Generator
) -> tuple[sp.csr_matrix, dict[int, int]]:
    """Query-local incidence. Weighted = link_conf x role_weight (spec 6.1)."""
    verts: dict[int, int] = {}
    rows, cols, vals = [], [], []
    for j, ents in enumerate(ent_lists):
        for k, v in enumerate(ents):
            iv = verts.setdefault(int(v), len(verts))
            rows.append(iv)
            cols.append(j)
            if weighted:
                # first two entities of an EU stand in for relation head/tail (role 1.0),
                # the rest are peripheral mentions (0.6); link_conf ~ U(0.7,1.0)
                role = 1.0 if k < 2 else 0.6
                vals.append(role * float(rng.uniform(0.7, 1.0)))
            else:
                vals.append(1.0)
    if not rows:
        return sp.csr_matrix((0, 0), dtype=np.float32), verts
    B = sp.csr_matrix(
        (np.array(vals, np.float32), (rows, cols)), shape=(len(verts), len(cand))
    )
    return B, verts


def propagate(
    B: sp.csr_matrix, verts: dict[int, int], ent_lists: list[np.ndarray],
    r_weights: dict[int, float], steps: int, kernel: str, lam: float = 0.2,
    agg: str = 'mean',
) -> np.ndarray:
    ne = B.shape[1]
    if ne == 0:
        return np.zeros(0, np.float32)
    d_e = np.asarray(B.sum(axis=0)).ravel()
    d_v = np.asarray(B.sum(axis=1)).ravel()
    d_e[d_e == 0] = 1.0
    d_v[d_v == 0] = 1.0
    S = B @ sp.diags((1.0 / d_e).astype(np.float32)) @ B.T @ sp.diags((1.0 / d_v).astype(np.float32))

    # Restart is ANCHORED, not uniform (spec 2.1): entities the query actually names carry
    # the mass; ANN-recalled anchors carry a fraction. Equal weighting dilutes the signal
    # across noise hubs and destroys multi-hop recall -- measured, see ticket 20.
    r = np.zeros(B.shape[0], dtype=np.float32)
    for v, w in r_weights.items():
        if v in verts:
            r[verts[v]] = w
    r = r / r.sum() if r.sum() else np.full_like(r, 1.0 / max(1, r.size))

    if kernel == "heat":
        L = sp.identity(S.shape[0], format="csr", dtype=np.float32) - S
        c = np.asarray(expm_multiply(-0.5 * steps * L, r)).ravel()
    else:
        c = r.copy()
        for _ in range(steps):
            c = lam * r + (1.0 - lam) * (S @ c)

    # The spec pools rho by MEAN over incident vertices. That structurally penalises a
    # bridge: it has one hot vertex (shared with the anchored side) and several cold ones,
    # so the mean washes the signal out. max/sum are the obvious alternatives -- measured.
    fn = {"mean": np.mean, "max": np.max, "sum": np.sum}[agg]
    rho = np.zeros(ne, dtype=np.float32)
    for j, ents in enumerate(ent_lists):
        idx = [verts[int(v)] for v in ents if int(v) in verts]
        if idx:
            rho[j] = float(fn(c[idx]))
    return rho


def select(
    cand: np.ndarray, ent_lists: list[np.ndarray], rho: np.ndarray, c: Corpus,
    v0: set[int], sims: np.ndarray, Bbin: sp.csr_matrix, w_prop: float | None = None,
) -> list[int]:
    """Selection with INCREMENTAL SPARSE overlap.

    A dense |E_c|x|E_c| overlap matrix is 1.6 GB at |E_c|=20000 and dominates latency --
    it is what caps the candidate universe near 5k, not propagation. Since redundancy only
    ever needs `overlap(j, a)` for the ~15-40 ALREADY-CHOSEN items, keep a running
    `max_shared` vector and update it with one sparse product per acceptance: O(nnz) each,
    ~40 times. Removes the cap entirely.
    """
    import heapq

    tok = c.token_cost[cand]
    gam = c.gamma[cand]
    sup = c.support[cand]
    arity = np.array([max(1, len(e)) for e in ent_lists], np.float32)
    ent_sets = [set(int(x) for x in e) for e in ent_lists]
    v0_arr = np.array([len(ent_sets[j] & v0) / arity[j] for j in range(len(cand))], np.float32)
    s_max = float(np.log1p(sup.max())) or 1.0
    wp = W_PROP if w_prop is None else w_prop
    rest = (1.0 - wp) / (1.0 - W_PROP)  # rescale the other terms so weights still sum to 1
    base = (
        wp * (rho / (rho.max() or 1.0)) + rest * (W_REL * sims + W_EXT * gam
        + W_ENT * v0_arr + W_SUP * (np.log1p(sup) / s_max))
    ).astype(np.float32)
    anchored = v0_arr > 0
    theta = THETA_FRAC * float(base.max() or 0.0)

    max_shared = np.zeros(len(cand), dtype=np.float32)
    BT = Bbin.T.tocsr()

    heap = [(-base[j] / tok[j], j) for j in range(len(cand))]
    heapq.heapify(heap)
    chosen: list[int] = []
    used = 0
    while heap and used < BUDGET_TOKENS:
        _, j = heapq.heappop(heap)
        if chosen:
            red = min(1.0, float(max_shared[j]) / arity[j])
            conn = 1.0 if max_shared[j] > 0 or anchored[j] else 0.0
        else:
            red, conn = 0.0, (1.0 if anchored[j] else 0.0)
        val = base[j] * (1.0 - red) + W_CONN * conn
        dens = val / tok[j]
        if heap and dens < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-dens, j))
            continue
        if used + tok[j] <= BUDGET_TOKENS and val > theta:
            chosen.append(j)
            used += int(tok[j])
            shared = np.asarray((BT @ Bbin[:, j]).todense()).ravel()
            np.maximum(max_shared, shared, out=max_shared)
    return [int(cand[j]) for j in chosen]


def run_config(
    c: Corpus, planted: list[Planted], hubs: np.ndarray, *, cap: int, steps: int,
    hops: int, weighted: bool, kernel: str, anchor_noise_w: float = 0.2,
    prune_to: int | None = None, w_prop: float | None = None, rho_agg: str = 'mean',
    rng_seed: int = 11,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    lat, recalls, cand_sizes, cand_hits = [], [], [], []
    for p in planted:
        noise = [int(x) for x in rng.choice(hubs, size=ENTRY_NOISE, replace=False)]
        v0 = [p.seed_entity, *noise]
        # spec 2.1: r_hat = alpha*sim + (1-alpha)*1[v in V_llm]; the named entity is V_llm,
        # ANN-recalled anchors are V_vec only.
        r_weights = {p.seed_entity: 1.0, **{n: anchor_noise_w for n in noise}}

        t0 = time.perf_counter()
        cand = candidates(c, v0, cap, hops)
        if cand.size == 0:
            continue
        ent_lists = [c.entities_of(int(e)) for e in cand]
        B, verts = build_B(cand, ent_lists, weighted, rng)
        rho = propagate(B, verts, ent_lists, r_weights, steps, kernel, agg=rho_agg)
        # distractors get maximal textual similarity: they must be beaten structurally
        sims = np.full(len(cand), 0.1, np.float32)
        pos = {int(e): i for i, e in enumerate(cand)}
        for d in p.distractors:
            if d in pos:
                sims[pos[d]] = 1.0
        # PRUNE BY PROPAGATED CONFIDENCE before packing. Expansion lifts candidate recall but
        # drowns the bridge in noise; rho is the structural signal that knows which candidates
        # the query actually reaches, so use it as the filter and let selection pack what
        # survives. Measured in ticket 20.
        if prune_to is not None and len(cand) > prune_to:
            keep = np.argsort(-rho)[:prune_to]
            keep.sort()
            cand = cand[keep]
            rho = rho[keep]
            sims = sims[keep]
            ent_lists = [ent_lists[i] for i in keep]
        Bbin, _ = build_B(cand, ent_lists, False, rng)
        sel = select(cand, ent_lists, rho, c, set(v0), sims, Bbin, w_prop)
        lat.append(time.perf_counter() - t0)

        hit = len(set(sel) & set(p.chain_edges))
        recalls.append(hit / len(p.chain_edges))
        cand_hits.append(len(set(int(x) for x in cand) & set(p.chain_edges)) / len(p.chain_edges))
        cand_sizes.append(int(cand.size))

    return {
        "recall": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "cand_recall": round(statistics.mean(cand_hits), 4) if cand_hits else 0.0,
        "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 2) if lat else 0.0,
        "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 2) if lat else 0.0,
        "mean_cand": int(statistics.mean(cand_sizes)) if cand_sizes else 0,
    }


def ppr_sketch_cost(c: Corpus, top_k: int = 200, sample: int = 2000) -> dict:
    """Index-time cost/size of truncated PPR sketches (spec 1.7 item 5).

    Unlike A_E these are BOUNDED BY CONSTRUCTION: top_k per entity, not sum d_v^2.
    """
    rng = np.random.default_rng(5)
    ents = rng.choice([v for v in c.postings], size=min(sample, len(c.postings)), replace=False)
    t0 = time.perf_counter()
    total = 0
    for v in ents:
        edges = c.postings[int(v)][:top_k]
        reach = set()
        for e in edges[:50]:  # 1-step neighbourhood, truncated
            reach.update(int(x) for x in c.entities_of(int(e)))
        total += min(len(reach), top_k)
    secs = time.perf_counter() - t0
    per_entity = secs / len(ents)
    return {
        "sampled_entities": len(ents),
        "projected_full_build_s": round(per_entity * len(c.postings), 1),
        "projected_entries": int(top_k * len(c.postings)),
        "projected_mib": round(top_k * len(c.postings) * 12 / 1024**2, 1),
        "bounded_by_construction": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eus", type=int, default=1_000_000)
    ap.add_argument("--chains", type=int, default=60)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--out", default="frontier-results.json")
    args = ap.parse_args()

    print(f"generating {args.eus:,} EUs + {args.chains} planted {args.hops}-hop chains...")
    base = generate(args.eus)
    c, planted = plant(base, args.chains, args.hops)
    hubs = np.array([v for v in c.postings if len(c.postings[v]) > 5][: 50_000])
    print(f"corpus: {c.n_eus:,} EUs, {c.n_entities:,} entities\n")

    out: dict = {"baseline_spec_defaults": None, "sweeps": {}}
    base_cfg = dict(cap=2000, steps=5, hops=1, weighted=False, kernel="rwr")

    print("=== BASELINE (the spec defaults I inherited without testing) ===")
    b = run_config(c, planted, hubs, **base_cfg)
    out["baseline_spec_defaults"] = {**base_cfg, **b}
    print(f"  cap=2000 K=5 hops=1 binary RWR -> recall={b['recall']:.3f}  p50={b['p50_ms']}ms\n")

    print("=== SWEEP 0: candidate-expansion HOPS (the one the spec calls optional) ===")
    out["sweeps"]["hops"] = []
    for h in (1, 2, 3, 4):
        r = run_config(c, planted, hubs, **{**base_cfg, "hops": h, "cap": 20000})
        out["sweeps"]["hops"].append({"hops": h, **r})
        print(f"  hops={h}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:8.2f}ms  cand={r['mean_cand']}")
    print()

    print("=== SWEEP 1: |E_c| ===")
    out["sweeps"]["cap"] = []
    for cap in (2000, 5000, 10000, 20000):
        r = run_config(c, planted, hubs, **{**base_cfg, "cap": cap, "hops": 3})
        out["sweeps"]["cap"].append({"cap": cap, **r})
        print(f"  |E_c|={cap:6d}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:7.2f}ms")

    print("\n=== SWEEP 2: RWR steps K ===")
    out["sweeps"]["steps"] = []
    for k in (5, 10, 15, 20):
        r = run_config(c, planted, hubs, **{**base_cfg, "steps": k, "cap": 20000, "hops": 3})
        out["sweeps"]["steps"].append({"steps": k, **r})
        print(f"  K={k:3d}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:7.2f}ms")


    print("\n=== SWEEP 4: weighted vs binary incidence (spec 6.1) ===")
    out["sweeps"]["weighted"] = []
    for w in (False, True):
        r = run_config(c, planted, hubs, **{**base_cfg, "weighted": w, "cap": 20000, "hops": 3})
        out["sweeps"]["weighted"].append({"weighted": w, **r})
        print(f"  weighted={str(w):5s}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:7.2f}ms")

    print("\n=== SWEEP 5: propagation kernel (spec 6.2) ===")
    out["sweeps"]["kernel"] = []
    for k in ("rwr", "heat"):
        r = run_config(c, planted, hubs, **{**base_cfg, "kernel": k, "cap": 20000, "hops": 3})
        out["sweeps"]["kernel"].append({"kernel": k, **r})
        print(f"  kernel={k:5s}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:7.2f}ms")

    print("\n=== SWEEP 6: rho-pruning before packing (hops=3, cap=20000) ===")
    out["sweeps"]["prune"] = []
    for pr in (None, 5000, 2000, 500):
        r = run_config(c, planted, hubs, **{**base_cfg, "hops": 3, "cap": 20000, "prune_to": pr})
        out["sweeps"]["prune"].append({"prune_to": pr, **r})
        print(f"  prune_to={str(pr):5s}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:8.2f}ms")

    print("\n=== SWEEP 6b: rho pooling (spec says mean) — hops=3, cap=20000 ===")
    out["sweeps"]["rho_agg"] = []
    for agg in ("mean", "max", "sum"):
        r = run_config(c, planted, hubs, **{**base_cfg, "hops": 3, "cap": 20000, "rho_agg": agg})
        out["sweeps"]["rho_agg"].append({"rho_agg": agg, **r})
        print(f"  rho_agg={agg:5s}  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:8.2f}ms")
    for agg in ("mean", "max"):
        r = run_config(c, planted, hubs, **{**base_cfg, "hops": 3, "cap": 20000, "rho_agg": agg, "prune_to": 2000})
        out["sweeps"]["rho_agg"].append({"rho_agg": agg, "prune_to": 2000, **r})
        print(f"  rho_agg={agg:5s} +prune2000  recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:8.2f}ms")

    print("\n=== SWEEP 7: propagation weight w_prop (hops=3, cap=20000, prune=2000) ===")
    out["sweeps"]["w_prop"] = []
    for wp in (0.35, 0.55, 0.75):
        r = run_config(c, planted, hubs, **{**base_cfg, "hops": 3, "cap": 20000, "prune_to": 2000, "w_prop": wp})
        out["sweeps"]["w_prop"].append({"w_prop": wp, **r})
        print(f"  w_prop={wp:.2f}  recall={r['recall']:.3f}  p50={r['p50_ms']:8.2f}ms")

    print("\n=== PPR sketches: index-time cost (spec 1.7 item 5) ===")
    ppr = ppr_sketch_cost(c)
    out["ppr_sketches"] = ppr
    print(f"  {ppr}")

    print("\n=== BEST-IN-BUDGET (everything the data justifies, under 50ms) ===")
    best = run_config(c, planted, hubs, cap=20000, steps=15, hops=3, weighted=True, kernel="rwr", prune_to=2000, w_prop=0.55, rho_agg="max")
    out["best_in_budget"] = best
    print(f"  cap=20000 K=15 hops=3 weighted RWR prune=2000 w_prop=.55 rho=max -> recall={best['recall']:.3f}  cand_recall={best['cand_recall']:.3f}  "
          f"p50={best['p50_ms']}ms  p95={best['p95_ms']}ms")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Ticket 20 addendum — paper-faithful pipeline vs my deviation (HyCE-RAG 2607.22597).

Diagnosis being tested. My frontier.py pipeline deviated from the paper in two structural
gates, and the harness denied a third assumption:

  D1 (Eq 16/19): the paper scores every candidate s_hyp = beta*s_entry + (1-beta)*s_rel and
      PRUNES to TopK_h BEFORE propagation. I propagated over the raw 20k expansion, so
      propagation ran on a noise-dominated graph and rho was corrupted from the start.
      (My sweep-6 "rho-pruning" pruned AFTER propagating on the polluted graph — backwards.)

  D2 (Eq 30-33): after propagation the paper thresholds VERTICES (c* >= theta_v), gates
      hyperedges on covering a retained entity, and scores coverage over retained entities.
      I had no post-propagation structural gate at all.

  D3 (harness): the paper's s_rel term assumes bridging evidence carries SOME textual
      relevance to the question — true in real MHQA (the question names the chain's
      endpoints). My harness gave bridges sim=0.1 and distractors sim=1.0, making the
      bridge distinguishable ONLY by structure in a structureless random graph —
      information-theoretically unfair to any bounded method.

Run:  uv run --with numpy --with scipy --with psutil python frontier2.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import scipy.sparse as sp

from frontier import ENTRY_NOISE, build_B, candidates, plant, propagate, select
from sim import generate

BETA = 0.5          # Eq 16 trade-off: structural entry overlap vs textual relevance
K_H = 200           # Eq 19: retained hyperedges — the query-aware hypergraph is SMALL
THETA_V = 0.10      # Eq 30: vertex confidence threshold (relative to max)


def sims_for(cand: np.ndarray, p, bridge_sim: tuple[float, float], rng) -> np.ndarray:
    """Textual relevance. Distractors always 1.0; bridge sim per scenario (D3)."""
    sims = rng.uniform(0.05, 0.3, size=len(cand)).astype(np.float32)
    pos = {int(e): i for i, e in enumerate(cand)}
    for e in p.chain_edges:
        if e in pos:
            sims[pos[e]] = rng.uniform(*bridge_sim)
    for d in p.distractors:
        if d in pos:
            sims[pos[d]] = 1.0
    return sims


def paper_pipeline(c, p, v0, r_weights, cand, sims, rng) -> list[int]:
    """Eq 16→19→26-29→30-33→selection, on top of the same expansion + budget machinery."""
    # --- D1: s_hyp scoring and TopK_h prune BEFORE propagation (Eq 16, 17, 19)
    v0set = set(v0)
    ent_lists = [c.entities_of(int(e)) for e in cand]
    s_entry = np.array(
        [len({int(x) for x in ents} & v0set) / (len(ents) + 1e-9) for ents in ent_lists],
        dtype=np.float32,
    )
    s_hyp = BETA * s_entry + (1 - BETA) * sims
    if len(cand) > K_H:
        keep = np.argsort(-s_hyp)[:K_H]
        keep.sort()
        cand, sims = cand[keep], sims[keep]
        ent_lists = [ent_lists[i] for i in keep]

    # --- propagate on the SMALL query-aware hypergraph (Eq 26-29)
    B, verts = build_B(cand, ent_lists, False, rng)
    rho = propagate(B, verts, ent_lists, r_weights, 6, "rwr")

    # --- D2: vertex threshold + coverage gate (Eq 30-33)
    # reconstruct vertex confidences from propagate()'s internals is awkward; recompute:
    d_e = np.asarray(B.sum(axis=0)).ravel(); d_e[d_e == 0] = 1.0
    d_v = np.asarray(B.sum(axis=1)).ravel(); d_v[d_v == 0] = 1.0
    S = B @ sp.diags((1.0/d_e).astype(np.float32)) @ B.T @ sp.diags((1.0/d_v).astype(np.float32))
    r = np.zeros(B.shape[0], np.float32)
    for v, w in r_weights.items():
        if v in verts:
            r[verts[v]] = w
    r = r / r.sum() if r.sum() else r
    cstar = r.copy()
    for _ in range(6):
        cstar = 0.2 * r + 0.8 * (S @ cstar)
    retained = {v for v, i in verts.items() if cstar[i] >= THETA_V * float(cstar.max() or 1)}
    cov = np.array(
        [len({int(x) for x in ents} & retained) for ents in ent_lists], dtype=np.float32
    )
    gate = cov > 0
    if gate.any():
        idx = np.where(gate)[0]
        cand, rho, sims = cand[idx], rho[idx], sims[idx]
        ent_lists = [ent_lists[i] for i in idx]

    # --- budgeted selection (shared machinery)
    Bbin, _ = build_B(cand, ent_lists, False, rng)
    return select(cand, ent_lists, rho, c, v0set, sims, Bbin)


def my_pipeline(c, p, v0, r_weights, cand, sims, rng) -> list[int]:
    """frontier.py's path: propagate over the FULL expansion, no gates."""
    ent_lists = [c.entities_of(int(e)) for e in cand]
    B, verts = build_B(cand, ent_lists, False, rng)
    rho = propagate(B, verts, ent_lists, r_weights, 6, "rwr")
    Bbin, _ = build_B(cand, ent_lists, False, rng)
    return select(cand, ent_lists, rho, c, set(v0), sims, Bbin)


def run(c, planted, hubs, pipeline, bridge_sim, hops=3, cap=20000, seed=11) -> dict:
    rng = np.random.default_rng(seed)
    recalls, cand_recalls, lat = [], [], []
    for p in planted:
        noise = [int(x) for x in rng.choice(hubs, size=ENTRY_NOISE, replace=False)]
        v0 = [p.seed_entity, *noise]
        r_weights = {p.seed_entity: 1.0, **{n: 0.2 for n in noise}}
        t0 = time.perf_counter()
        cand = candidates(c, v0, cap, hops)
        if not cand.size:
            continue
        sims = sims_for(cand, p, bridge_sim, rng)
        sel = pipeline(c, p, v0, r_weights, cand, sims, rng)
        lat.append(time.perf_counter() - t0)
        recalls.append(len(set(sel) & set(p.chain_edges)) / len(p.chain_edges))
        cand_recalls.append(len(set(int(x) for x in cand) & set(p.chain_edges)) / len(p.chain_edges))
    return {
        "recall": round(statistics.mean(recalls), 4),
        "cand_recall": round(statistics.mean(cand_recalls), 4),
        "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eus", type=int, default=200_000)
    ap.add_argument("--chains", type=int, default=40)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--out", default="frontier2-results.json")
    args = ap.parse_args()

    print(f"generating {args.eus:,} EUs + {args.chains} planted {args.hops}-hop chains...")
    base = generate(args.eus)
    c, planted = plant(base, args.chains, args.hops)
    hubs = np.array([v for v in c.postings if len(c.postings[v]) > 5][:50_000])

    out = {}
    scenarios = [
        ("adversarial (old harness: bridge sim 0.1)", (0.05, 0.15)),
        ("realistic (bridge shares question terms)", (0.40, 0.70)),
    ]
    for name, bs in scenarios:
        print(f"\n=== scenario: {name} ===")
        for pname, pipe in (("mine (no gates)", my_pipeline), ("paper (D1+D2 gates)", paper_pipeline)):
            r = run(c, planted, hubs, pipe, bs, hops=args.hops + 1)
            out[f"{name} | {pname}"] = r
            print(f"  {pname:22s} recall={r['recall']:.3f}  cand_recall={r['cand_recall']:.3f}  p50={r['p50_ms']:8.2f}ms")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

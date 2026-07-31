"""Throwaway prototype — evidence for wayfinder ticket 03 (sidecar statelessness ruling).

NOT product code. Synthetic hypergraph, no pipeline integration, no real models.

Measures three things the A/B/C residency decision turns on:

  A  STRICT    — hold nothing; per query, read postings + EU rows from parquet
                 (DuckDB predicate pushdown) and compute edge-overlap locally.
  C  RESIDENT  — hold everything; global CSR incidence + precomputed A_E in memory.
  B  HANDLE    — caller owns the loaded objects. Latency-identical to C by
                 construction, so it is not measured separately; it differs from C
                 only in who owns the object and who pays the load.

Reported: per-phase latency (p50/p95/p99), selection-set equality between A and C,
resident memory, and the offline cost/feasibility of materialising global A_E.

Run:  uv run --with numpy --with scipy --with duckdb --with pyarrow --with psutil \
          python sim.py --scales 10000,100000
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

# --- online-path constants (spec Part 2/3) ---------------------------------
ENTRY_K = 10          # |V0| — linked + ANN entry entities
CAND_CAP = 2000       # |E_c| cap
RWR_LAM = 0.2
RWR_STEPS = 5
BUDGET_TOKENS = 8000  # T — evidence budget
W_PROP, W_REL, W_EXT, W_ENT, W_SUP = 0.35, 0.20, 0.15, 0.15, 0.10
W_CONN = 0.15
THETA_FRAC = 0.15

# Materialising A_E is abandoned above this many non-zeros (the finding, not a crash).
AE_NNZ_CAP = 200_000_000


# --- synthetic corpus ------------------------------------------------------
@dataclass
class Corpus:
    n_eus: int
    n_entities: int
    ent_flat: np.ndarray        # concatenated entity ids per EU
    ent_offsets: np.ndarray     # CSR-style offsets into ent_flat (len n_eus+1)
    token_cost: np.ndarray
    gamma: np.ndarray
    support: np.ndarray
    postings: dict[int, np.ndarray] = field(default_factory=dict)

    def entities_of(self, e: int) -> np.ndarray:
        return self.ent_flat[self.ent_offsets[e] : self.ent_offsets[e + 1]]


def generate(n_eus: int, seed: int = 7) -> Corpus:
    """Zipf-distributed entity references — hub entities are what stress A_E."""
    rng = np.random.default_rng(seed)
    n_entities = max(100, n_eus // 5)

    # arity 2..12, skewed low (spec caps m_max=12)
    arity = np.clip(rng.geometric(0.32, size=n_eus) + 1, 2, 12)
    total = int(arity.sum())

    # Zipf popularity over the entity vocabulary -> genuine hubs
    ranks = np.arange(1, n_entities + 1)
    p = 1.0 / ranks**1.07
    p /= p.sum()
    ent_flat = rng.choice(n_entities, size=total, p=p).astype(np.int32)

    offsets = np.zeros(n_eus + 1, dtype=np.int64)
    np.cumsum(arity, out=offsets[1:])

    # dedupe entities within each EU (a hyperedge is a *set*)
    keep = np.ones(total, dtype=bool)
    for e in range(n_eus):
        s, t = offsets[e], offsets[e + 1]
        seen: set[int] = set()
        for i in range(s, t):
            v = int(ent_flat[i])
            if v in seen:
                keep[i] = False
            else:
                seen.add(v)
    ent_flat = ent_flat[keep]
    new_arity = np.add.reduceat(keep.astype(np.int64), offsets[:-1])
    offsets = np.zeros(n_eus + 1, dtype=np.int64)
    np.cumsum(new_arity, out=offsets[1:])

    token_cost = np.clip(rng.normal(80, 22, n_eus), 40, 120).astype(np.int32)
    gamma = rng.beta(6, 2, n_eus).astype(np.float32)
    support = (rng.geometric(0.75, n_eus)).astype(np.int32)

    c = Corpus(n_eus, n_entities, ent_flat, offsets, token_cost, gamma, support)

    # inverted lists: entity -> edges, sorted by gamma desc
    order = np.argsort(ent_flat, kind="stable")
    sorted_ent = ent_flat[order]
    edge_of_ref = np.repeat(np.arange(n_eus, dtype=np.int64), np.diff(offsets))
    sorted_edges = edge_of_ref[order]
    bounds = np.searchsorted(sorted_ent, np.arange(n_entities + 1))
    for v in range(n_entities):
        edges = sorted_edges[bounds[v] : bounds[v + 1]]
        if edges.size:
            c.postings[v] = edges[np.argsort(-gamma[edges], kind="stable")]
    return c


# --- deterministic per-(query, edge) textual similarity ---------------------
def sim_scores(qid: int, edges: np.ndarray) -> np.ndarray:
    """Stand-in for cosine(query, tau_text). Deterministic so A and C must agree."""
    h = (edges.astype(np.uint64) * np.uint64(40503)) ^ np.uint64(qid * 2654435761)
    return ((h % np.uint64(1000)).astype(np.float32)) / 1000.0


# --- the shared online algorithm -------------------------------------------
def propagate(cand: np.ndarray, ent_lists: list[np.ndarray], v0: set[int]) -> np.ndarray:
    """RWR over the query-local bipartite incidence -> rho(e)."""
    verts: dict[int, int] = {}
    rows, cols = [], []
    for j, ents in enumerate(ent_lists):
        for v in ents:
            iv = verts.setdefault(int(v), len(verts))
            rows.append(iv)
            cols.append(j)
    if not rows:
        return np.zeros(len(cand), dtype=np.float32)

    nv, ne = len(verts), len(cand)
    B = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nv, ne))

    d_e = np.asarray(B.sum(axis=0)).ravel()
    d_v = np.asarray(B.sum(axis=1)).ravel()
    d_e[d_e == 0] = 1.0
    d_v[d_v == 0] = 1.0
    Dei = sp.diags((1.0 / d_e).astype(np.float32))
    Dvi = sp.diags((1.0 / d_v).astype(np.float32))
    S = B @ Dei @ B.T @ Dvi

    r = np.zeros(nv, dtype=np.float32)
    for v in v0:
        if v in verts:
            r[verts[v]] = 1.0
    if r.sum() == 0:
        r[:] = 1.0 / nv
    else:
        r /= r.sum()

    c = r.copy()
    for _ in range(RWR_STEPS):
        c = RWR_LAM * r + (1.0 - RWR_LAM) * (S @ c)

    rho = np.zeros(ne, dtype=np.float32)
    for j, ents in enumerate(ent_lists):
        if len(ents):
            rho[j] = float(np.mean([c[verts[int(v)]] for v in ents]))
    return rho


def coe_select(
    qid: int,
    cand: np.ndarray,
    ent_lists: list[np.ndarray],
    rho: np.ndarray,
    tok: np.ndarray,
    gamma: np.ndarray,
    support: np.ndarray,
    v0: set[int],
    ov: np.ndarray,
) -> list[int]:
    """Lazy-greedy marginal-utility selection under the token budget (spec 3.3).

    ``ov`` is the candidate-local edge-overlap matrix (|cand| x |cand|, shared-entity
    counts). Both strategies supply it; they differ only in how they obtained it.
    """
    sims = sim_scores(qid, cand)
    s_max = float(np.log1p(support.max())) or 1.0
    arity = np.array([max(1, len(e)) for e in ent_lists], dtype=np.float32)
    ent_sets = [set(int(x) for x in e) for e in ent_lists]

    v0_arr = np.array(
        [len(ent_sets[j] & v0) / arity[j] for j in range(len(cand))], dtype=np.float32
    )
    base = (
        W_PROP * (rho / (rho.max() or 1.0))
        + W_REL * sims
        + W_EXT * gamma
        + W_ENT * v0_arr
        + W_SUP * (np.log1p(support) / s_max)
    ).astype(np.float32)
    anchored = v0_arr > 0

    theta = THETA_FRAC * float(base.max() or 0.0)
    heap = [(-base[j] / tok[j], j) for j in range(len(cand))]
    heapq.heapify(heap)

    chosen: list[int] = []
    used = 0
    while heap and used < BUDGET_TOKENS:
        _, j = heapq.heappop(heap)
        if chosen:
            shared = ov[j, chosen]
            red = min(1.0, float(shared.max()) / arity[j])
            conn = 1.0 if (shared > 0).any() or anchored[j] else 0.0
        else:
            red = 0.0
            conn = 1.0 if anchored[j] else 0.0
        val = base[j] * (1.0 - red) + W_CONN * conn
        dens = val / tok[j]
        if heap and dens < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-dens, j))
            continue
        if used + tok[j] <= BUDGET_TOKENS and val > theta:
            chosen.append(j)
            used += int(tok[j])
    return sorted(int(cand[j]) for j in chosen)


def local_overlap(ent_lists: list[np.ndarray]) -> np.ndarray:
    """Recompute edge-overlap on the candidate set only — strategy A's substitute for A_E."""
    verts: dict[int, int] = {}
    rows, cols = [], []
    for j, ents in enumerate(ent_lists):
        for v in ents:
            iv = verts.setdefault(int(v), len(verts))
            rows.append(j)
            cols.append(iv)
    n = len(ent_lists)
    if not rows:
        return np.zeros((n, n), dtype=np.float32)
    B = sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(verts))
    )
    A = (B @ B.T).toarray().astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


# --- strategy A: strict per-call, nothing held -----------------------------
class Strict:
    name = "A-strict"

    def __init__(self, store: str) -> None:
        import duckdb

        self._duckdb = duckdb
        self.postings_pq = os.path.join(store, "postings.parquet")
        self.edges_pq = os.path.join(store, "hyperedges.parquet")

    def query(self, qid: int, v0: list[int]) -> tuple[list[int], dict[str, float]]:
        t = {}
        con = self._duckdb.connect(":memory:")  # per call, closed below
        try:
            t0 = time.perf_counter()
            ids = ",".join(str(v) for v in v0)
            cand = con.execute(
                f"SELECT DISTINCT edge_id FROM read_parquet('{self.postings_pq}') "
                f"WHERE entity_id IN ({ids}) ORDER BY gamma DESC LIMIT {CAND_CAP}"
            ).fetchnumpy()["edge_id"]
            t["entry+cand"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            elist = ",".join(str(int(e)) for e in cand)
            rows = con.execute(
                f"SELECT edge_id, entity_ids, token_cost, gamma, support "
                f"FROM read_parquet('{self.edges_pq}') WHERE edge_id IN ({elist})"
            ).fetchall()
            t["fetch_rows"] = time.perf_counter() - t0
        finally:
            con.close()  # holds nothing between calls

        rows.sort(key=lambda r: r[0])
        cand = np.array([r[0] for r in rows], dtype=np.int64)
        ent_lists = [np.asarray(r[1], dtype=np.int32) for r in rows]
        tok = np.array([r[2] for r in rows], dtype=np.int32)
        gam = np.array([r[3] for r in rows], dtype=np.float32)
        sup = np.array([r[4] for r in rows], dtype=np.int32)

        t0 = time.perf_counter()
        rho = propagate(cand, ent_lists, set(v0))
        t["propagate"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ov = local_overlap(ent_lists)  # the A_E substitute: recomputed, never stored
        t["overlap"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sel = coe_select(qid, cand, ent_lists, rho, tok, gam, sup, set(v0), ov)
        t["select"] = time.perf_counter() - t0
        return sel, t


# --- strategy A': strict per-call, but point-lookup KV instead of columnar scan ---
class StrictKV:
    """A' — holds nothing, but reads via LMDB point lookups (the spec's own KV layout).

    Strategy A's cost turned out to be columnar *scan* I/O: fetching 2000 scattered
    edge_ids from a 1M-row parquet touches most row groups. A KV store answers the same
    2000 reads as point lookups. The env is opened and closed per call — still nothing held.
    """

    name = "A'-strict-kv"

    def __init__(self, store: str) -> None:
        import lmdb

        self._lmdb = lmdb
        self.path = os.path.join(store, "kv")

    def query(self, qid: int, v0: list[int]) -> tuple[list[int], dict[str, float]]:
        import struct

        t = {}
        env = self._lmdb.open(self.path, readonly=True, lock=False, max_dbs=0)
        try:
            t0 = time.perf_counter()
            with env.begin(buffers=True) as txn:
                ids, gms = [], []
                for v in v0:
                    raw = txn.get(b"p" + struct.pack("<i", v))
                    if raw is not None:
                        buf = bytes(raw)
                        (n,) = struct.unpack_from("<i", buf, 0)
                        ids.append(np.frombuffer(buf, np.int64, count=n, offset=4))
                        gms.append(
                            np.frombuffer(buf, np.float32, count=n, offset=4 + 8 * n)
                        )
                if ids:
                    all_ids = np.concatenate(ids)
                    all_gms = np.concatenate(gms)
                    cand, first = np.unique(all_ids, return_index=True)
                    # top-k by gamma, then id order — identical rule to the other strategies
                    if cand.size > CAND_CAP:
                        keep = np.argsort(-all_gms[first], kind="stable")[:CAND_CAP]
                        cand = cand[keep]
                        cand.sort()
                else:
                    cand = np.array([], np.int64)
                t["entry+cand"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                ent_lists, tok, gam, sup = [], [], [], []
                for e in cand:
                    raw = bytes(txn.get(b"e" + struct.pack("<q", int(e))))
                    tc, g, s, n = struct.unpack_from("<ifii", raw, 0)
                    ent_lists.append(np.frombuffer(raw, np.int32, count=n, offset=16))
                    tok.append(tc)
                    gam.append(g)
                    sup.append(s)
                t["fetch_rows"] = time.perf_counter() - t0
        finally:
            env.close()  # holds nothing between calls

        tok = np.array(tok, np.int32)
        gam = np.array(gam, np.float32)
        sup = np.array(sup, np.int32)

        t0 = time.perf_counter()
        rho = propagate(cand, ent_lists, set(v0))
        t["propagate"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ov = local_overlap(ent_lists)
        t["overlap"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sel = coe_select(qid, cand, ent_lists, rho, tok, gam, sup, set(v0), ov)
        t["select"] = time.perf_counter() - t0
        return sel, t


def write_kv(corpus: Corpus, store: str) -> float:
    import struct

    import lmdb

    path = os.path.join(store, "kv")
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, map_size=8 << 30)
    with env.begin(write=True) as txn:
        for v, edges in corpus.postings.items():
            e64 = edges.astype(np.int64)
            txn.put(
                b"p" + struct.pack("<i", v),
                struct.pack("<i", e64.size)
                + e64.tobytes()
                + corpus.gamma[edges].astype(np.float32).tobytes(),
            )
        for e in range(corpus.n_eus):
            ents = corpus.entities_of(e)
            head = struct.pack(
                "<ifii", int(corpus.token_cost[e]), float(corpus.gamma[e]),
                int(corpus.support[e]), len(ents),
            )
            txn.put(b"e" + struct.pack("<q", e), head + ents.astype(np.int32).tobytes())
    env.close()
    return sum(
        os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
    ) / 1024**2


# --- strategy C: everything resident ---------------------------------------
class Resident:
    name = "C-resident"

    def __init__(self, corpus: Corpus, A_E: sp.csr_matrix | None) -> None:
        self.c = corpus
        self.A_E = A_E

    def query(self, qid: int, v0: list[int]) -> tuple[list[int], dict[str, float]]:
        t = {}
        t0 = time.perf_counter()
        acc: list[np.ndarray] = [self.c.postings[v] for v in v0 if v in self.c.postings]
        cand = np.unique(np.concatenate(acc)) if acc else np.array([], dtype=np.int64)
        if cand.size > CAND_CAP:
            cand = cand[np.argsort(-self.c.gamma[cand], kind="stable")[:CAND_CAP]]
            cand.sort()
        t["entry+cand"] = time.perf_counter() - t0
        t["fetch_rows"] = 0.0

        ent_lists = [self.c.entities_of(int(e)) for e in cand]
        tok = self.c.token_cost[cand]
        gam = self.c.gamma[cand]
        sup = self.c.support[cand]

        t0 = time.perf_counter()
        rho = propagate(cand, ent_lists, set(v0))
        t["propagate"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.A_E is not None:
            # the whole point of holding A_E: slice the candidate submatrix, no recompute
            ov = self.A_E[cand][:, cand].toarray().astype(np.float32)
            np.fill_diagonal(ov, 0.0)
        else:
            ov = local_overlap(ent_lists)
        t["overlap"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sel = coe_select(qid, cand, ent_lists, rho, tok, gam, sup, set(v0), ov)
        t["select"] = time.perf_counter() - t0
        return sel, t


# --- offline: is global A_E even feasible? ---------------------------------
def ae_projection(corpus: Corpus) -> dict:
    """nnz(A_E) ~ sum_v d_v^2 — computed analytically before attempting to build."""
    d_v = np.array([len(corpus.postings.get(v, ())) for v in range(corpus.n_entities)],
                   dtype=np.float64)
    projected = float((d_v**2).sum())
    return {
        "projected_nnz": projected,
        "projected_gib": projected * 12 / 1024**3,  # int32 idx + float32 val + overhead
        "max_entity_degree": int(d_v.max()),
        "feasible": projected <= AE_NNZ_CAP,
    }


def build_ae(corpus: Corpus) -> tuple[sp.csr_matrix | None, float, float]:
    proj = ae_projection(corpus)
    if not proj["feasible"]:
        return None, 0.0, 0.0
    rows = np.repeat(np.arange(corpus.n_eus), np.diff(corpus.ent_offsets))
    B = sp.csr_matrix(
        (np.ones(corpus.ent_flat.size, np.float32), (corpus.ent_flat, rows)),
        shape=(corpus.n_entities, corpus.n_eus),
    )
    t0 = time.perf_counter()
    A = (B.T @ B).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    secs = time.perf_counter() - t0
    mib = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1024**2
    return A, secs, mib


def write_store(corpus: Corpus, store: str) -> float:
    os.makedirs(store, exist_ok=True)
    ent_lists = [corpus.entities_of(e).tolist() for e in range(corpus.n_eus)]
    pq.write_table(
        pa.table({
            "edge_id": pa.array(np.arange(corpus.n_eus), pa.int64()),
            "entity_ids": pa.array(ent_lists, pa.list_(pa.int32())),
            "token_cost": pa.array(corpus.token_cost, pa.int32()),
            "gamma": pa.array(corpus.gamma, pa.float32()),
            "support": pa.array(corpus.support, pa.int32()),
        }),
        os.path.join(store, "hyperedges.parquet"),
    )
    ent, edg, gam = [], [], []
    for v, edges in corpus.postings.items():
        ent.append(np.full(edges.size, v, dtype=np.int32))
        edg.append(edges.astype(np.int64))
        gam.append(corpus.gamma[edges])
    e_arr, d_arr, g_arr = np.concatenate(ent), np.concatenate(edg), np.concatenate(gam)
    order = np.argsort(e_arr, kind="stable")  # sorted -> row-group pushdown works
    pq.write_table(
        pa.table({
            "entity_id": pa.array(e_arr[order], pa.int32()),
            "edge_id": pa.array(d_arr[order], pa.int64()),
            "gamma": pa.array(g_arr[order], pa.float32()),
        }),
        os.path.join(store, "postings.parquet"),
    )
    return sum(
        os.path.getsize(os.path.join(store, f)) for f in os.listdir(store)
    ) / 1024**2


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p) * 1000)  # ms


def run_scale(n_eus: int, n_queries: int, out: dict) -> None:
    print(f"\n{'=' * 72}\nSCALE: {n_eus:,} EUs\n{'=' * 72}")
    proc = psutil.Process()

    t0 = time.perf_counter()
    corpus = generate(n_eus)
    gen_s = time.perf_counter() - t0
    print(f"generated in {gen_s:.1f}s  |  entities={corpus.n_entities:,}  "
          f"refs={corpus.ent_flat.size:,}")

    proj = ae_projection(corpus)
    print(f"A_E projection: nnz~{proj['projected_nnz']:,.0f} "
          f"({proj['projected_gib']:.2f} GiB)  max_entity_degree={proj['max_entity_degree']:,}  "
          f"feasible={proj['feasible']}")

    A_E, ae_secs, ae_mib = build_ae(corpus)
    if A_E is not None:
        print(f"A_E built: {ae_secs:.1f}s, {ae_mib:.0f} MiB, nnz={A_E.nnz:,}")
    else:
        print("A_E NOT MATERIALISED — projected nnz exceeds cap (this is the finding)")

    store = tempfile.mkdtemp(prefix=f"resid{n_eus}_")
    disk_mib = write_store(corpus, store)
    kv_mib = write_kv(corpus, store)
    print(f"parquet store: {disk_mib:.0f} MiB  |  kv store: {kv_mib:.0f} MiB")

    rss_before = proc.memory_info().rss
    strict = Strict(store)
    strict_kv = StrictKV(store)
    resident = Resident(corpus, A_E)
    rss_resident = (proc.memory_info().rss - rss_before) / 1024**2

    rng = np.random.default_rng(99)
    hot = [v for v in corpus.postings if len(corpus.postings[v]) > 1]
    queries = [
        (qid, list(rng.choice(hot, size=min(ENTRY_K, len(hot)), replace=False)))
        for qid in range(n_queries)
    ]

    results = {}
    for strat in (strict, strict_kv, resident):
        lat, phases, sels = [], [], []
        for qid, v0 in queries:
            t0 = time.perf_counter()
            sel, ph = strat.query(qid, [int(v) for v in v0])
            lat.append(time.perf_counter() - t0)
            phases.append(ph)
            sels.append(sel)
        results[strat.name] = {
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "p99_ms": pct(lat, 99),
            "phases_ms": {
                k: round(statistics.mean(p[k] for p in phases) * 1000, 2)
                for k in phases[0]
            },
            "selections": sels,
            "mean_selected": round(statistics.mean(len(s) for s in sels), 1),
        }
        print(f"  {strat.name:12s} p50={results[strat.name]['p50_ms']:8.2f}ms  "
              f"p95={results[strat.name]['p95_ms']:8.2f}ms  "
              f"phases={results[strat.name]['phases_ms']}")

    c_sel = results["C-resident"]["selections"]
    quality = {}
    for name in ("A-strict", "A'-strict-kv"):
        a_sel = results[name]["selections"]
        identical = sum(1 for a, c in zip(a_sel, c_sel) if a == c)
        jac = [
            len(set(a) & set(c)) / max(1, len(set(a) | set(c)))
            for a, c in zip(a_sel, c_sel)
        ]
        quality[name] = {
            "identical_vs_resident": f"{identical}/{len(a_sel)}",
            "mean_jaccard": round(statistics.mean(jac), 6),
        }
        print(f"  QUALITY {name:14s} vs C: identical {identical}/{len(a_sel)}  "
              f"mean Jaccard={statistics.mean(jac):.4f}")

    out[str(n_eus)] = {
        "entities": corpus.n_entities,
        "ae_projection": proj,
        "ae_build_s": round(ae_secs, 2),
        "ae_mib": round(ae_mib, 1),
        "parquet_mib": round(disk_mib, 1),
        "kv_mib": round(kv_mib, 1),
        "resident_mib": round(rss_resident, 1),
        "quality_vs_resident": quality,
        "strategies": {
            k: {kk: vv for kk, vv in v.items() if kk != "selections"}
            for k, v in results.items()
        },
    }
    shutil.rmtree(store, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="10000,100000")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    out: dict = {}
    for s in [int(x) for x in args.scales.split(",")]:
        run_scale(s, args.queries, out)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

# Throwaway benchmark for ticket 19: per-call HyperedgeStoreReader.open -> reads -> close
# under CONCURRENT query load (multiprocessing, matching production's process-per-query).
#
# Fills ADR-0056's stated gap: its numbers were single-process, warm page cache. This measures
# the REAL store implementation (latence_core.hyperedge_store), not a synthetic stand-in:
# a real store file is built with build_store_bytes(store_entries(...)) over real
# GraphHyperedge/GraphNode contract records (construction patterns from
# packages/latence-core/tests/test_hyperedge.py fixtures), at ~50k and ~500k hyperedges.
#
# Per query, the simulated read pattern is: open the store, ~10 postings lookups,
# ~200 solver_fields, ~40 tau_text reads, close. p50/p95/p99 per query at concurrency
# 1 / 4 / 16, warm cache. open() cost is reported separately from the read cost.
#
# Cold cache: /usr/sbin/purge requires root on this host ("Operation not permitted"
# without sudo), so the cold-cache run is NOT included — warm-cache only, stated honestly.
#
# Run from the repo root:  uv run python issues/wayfinder-hypergraph-coe/prototypes/residency/kv_concurrency.py

from __future__ import annotations

import bisect
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import tempfile
import time

from latence_core.contracts import Classification, Evidence, GraphHyperedge, GraphNode, Provenance
from latence_core.hyperedge_store import HyperedgeStoreReader, build_store_bytes, store_entries

# -- workload shape (the ticket's stated read pattern) ---------------------------

POSTINGS_PER_QUERY = 10
SOLVER_PER_QUERY = 200
TAU_PER_QUERY = 40
QUERIES_PER_WORKER = 100
CONCURRENCY_LEVELS = (1, 4, 16)
SCALES = ((50_000, 10_000), (500_000, 100_000))  # (hyperedges, nodes)
ZIPF_ALPHA = 1.07  # entity popularity, as in the ADR-0056 residency sim

_CLS = Classification(language="en", category="memo", sensitivity="internal")


def _prov(doc: str, cs: int, ce: int) -> Provenance:
    # Same construction pattern as tests/test_hyperedge.py::_prov
    return Provenance(
        source_uri=f"file:///{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=500,
        document_id=doc,
        char_start=cs,
        char_end=ce,
        page_start=1,
        page_end=1,
    )


def _node_id(i: int) -> str:
    return f"n{i:08d}"


def _hyperedge_id(i: int) -> str:
    return f"h{i:08d}"


def build_store_file(path: str, n_hyperedges: int, n_nodes: int, seed: int = 7) -> int:
    """Build a REAL store file from real contract records; returns file size in bytes."""
    rng = random.Random(seed)
    # Zipf(alpha) popularity over node indices, via inverse-CDF on cumulative weights.
    weights = [1.0 / (j + 1) ** ZIPF_ALPHA for j in range(n_nodes)]
    cum: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w
        cum.append(acc)
    total = cum[-1]

    def zipf_node() -> int:
        return bisect.bisect_left(cum, rng.random() * total)

    ev = Evidence(mention_ids=["m1"], document_ids=["doc-1"], confidence=0.9)
    prov = _prov("doc-1", 0, 120)
    tau_base = (
        "Acme Corp acquired Beta Ltd in 2019 for 2 billion euros; "
        "the deal was approved by the commission after review. "
    )

    hyperedges = []
    for i in range(n_hyperedges):
        arity = rng.randint(2, 6)
        members = {zipf_node() for _ in range(arity)}
        while len(members) < 2:
            members.add(zipf_node())
        entity_node_ids = sorted(_node_id(j) for j in members)
        hyperedges.append(
            GraphHyperedge(
                record_id=f"he:{i}",
                hyperedge_id=_hyperedge_id(i),
                entity_node_ids=entity_node_ids,
                tau_text=f"{tau_base}[span {i}]",
                token_cost=rng.randint(12, 60),
                token_counter="latence:estimate_tokens:v1",
                gamma=round(rng.random(), 6),
                support=1,
                has_relations=False,  # weak units keep the validator simple; layout is identical
                evidence=ev,
                provenance=prov,
                classification=_CLS,
            )
        )

    nodes = [
        GraphNode(
            record_id=f"node:{_node_id(j)}",
            node_id=_node_id(j),
            entity_id=f"e:{j}",
            label="ORG",
            canonical_name=f"Entity {j} GmbH",
            confidence=0.9,
            evidence=ev,
            provenance=prov,
            classification=_CLS,
        )
        for j in range(n_nodes)
    ]

    raw = build_store_bytes(store_entries(hyperedges, nodes))
    with open(path, "wb") as fh:
        fh.write(raw)
    return len(raw)


# -- the measured query -----------------------------------------------------------


def run_one_query(store_path: str, rng: random.Random, n_hyperedges: int, n_nodes: int) -> tuple[int, int, int]:
    """One production-shaped query. Returns (open_ns, reads_ns, close_ns)."""
    t0 = time.perf_counter_ns()
    store = HyperedgeStoreReader.open(store_path)
    t1 = time.perf_counter_ns()
    candidate_ids: list[str] = []
    for _ in range(POSTINGS_PER_QUERY):
        ids, _gammas = store.postings(_node_id(rng.randrange(n_nodes)))
        candidate_ids.extend(ids[:64])
    if not candidate_ids:
        candidate_ids = [_hyperedge_id(rng.randrange(n_hyperedges))]
    for _ in range(SOLVER_PER_QUERY):
        store.solver_fields(rng.choice(candidate_ids))
    for _ in range(TAU_PER_QUERY):
        store.tau_text(rng.choice(candidate_ids))
    t2 = time.perf_counter_ns()
    store.close()
    t3 = time.perf_counter_ns()
    return (t1 - t0, t2 - t1, t3 - t2)


def worker(store_path: str, n_hyperedges: int, n_nodes: int, seed: int,
           start_evt, out_q) -> None:
    rng = random.Random(seed)
    start_evt.wait()
    samples = [run_one_query(store_path, rng, n_hyperedges, n_nodes)
               for _ in range(QUERIES_PER_WORKER)]
    out_q.put(samples)


def pctl(vals: list[float], p: float) -> float:
    return statistics.quantiles(vals, n=100, method="inclusive")[int(p) - 1]


def summarize(samples: list[tuple[int, int, int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, idx in (("open", 0), ("reads", 1), ("close", 2)):
        ms = [s[idx] / 1e6 for s in samples]
        out[name] = {"p50": pctl(ms, 50), "p95": pctl(ms, 95), "p99": pctl(ms, 99)}
    totals = [(s[0] + s[1] + s[2]) / 1e6 for s in samples]
    out["total"] = {"p50": pctl(totals, 50), "p95": pctl(totals, 95), "p99": pctl(totals, 99)}
    return out


def run_level(store_path: str, n_hyperedges: int, n_nodes: int, concurrency: int) -> dict:
    ctx = mp.get_context("spawn")
    start_evt = ctx.Event()
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=worker,
                    args=(store_path, n_hyperedges, n_nodes, 1000 + w, start_evt, out_q))
        for w in range(concurrency)
    ]
    for p in procs:
        p.start()
    time.sleep(0.5)  # let workers import + reach the barrier
    start_evt.set()
    samples: list[tuple[int, int, int]] = []
    for _ in procs:
        samples.extend(out_q.get())
    for p in procs:
        p.join()
    return summarize(samples)


def warm_cache(store_path: str) -> None:
    with open(store_path, "rb") as fh:
        while fh.read(1 << 24):
            pass


def main() -> None:
    results: dict[str, dict] = {"host": os.uname().sysname, "note": "warm page cache only; "
                                "/usr/sbin/purge needs root on this host, cold run not possible"}
    tmpdir = tempfile.mkdtemp(prefix="kv_concurrency_")
    for n_hyperedges, n_nodes in SCALES:
        label = f"{n_hyperedges // 1000}k"
        store_path = os.path.join(tmpdir, f"store_{label}.lhkv")
        t = time.perf_counter()
        size = build_store_file(store_path, n_hyperedges, n_nodes)
        print(f"[{label}] built {size / 1e6:.1f} MB store in {time.perf_counter() - t:.1f}s",
              flush=True)
        warm_cache(store_path)
        results[label] = {"file_mb": round(size / 1e6, 1), "levels": {}}
        for c in CONCURRENCY_LEVELS:
            r = run_level(store_path, n_hyperedges, n_nodes, c)
            results[label]["levels"][f"c{c}"] = r
            print(f"[{label}] c={c:2d}  total p50={r['total']['p50']:.3f}ms "
                  f"p95={r['total']['p95']:.3f}ms p99={r['total']['p99']:.3f}ms | "
                  f"open p50={r['open']['p50']:.3f}ms p99={r['open']['p99']:.3f}ms | "
                  f"reads p50={r['reads']['p50']:.3f}ms p99={r['reads']['p99']:.3f}ms",
              flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

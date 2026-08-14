"""The :class:`GraphSource` seam (#169), tested at the ``neighbors`` boundary.

Four adapters of one deep seam, each at its own honest boundary:

* the shipped faithful :class:`FakeGraphSource` — the canonical traversal semantics over an in-memory
  edge list, asserted through the public :meth:`neighbors` (undirected, minimum hop distance, seeds
  excluded, bounded depth, graceful missing seed);
* the **real DuckDB** and **real networkx** engines — gated on their extra (skipped offline): a tiny
  ``graph-edges.parquet`` fixture is traversed by the real engine and asserted **byte-for-byte equal**
  to the faithful fake on the same graph (anti-false-green: the fake is not an independent
  re-implementation that could drift — the engines must *match* it), plus each engine's runtime-fault
  → typed :class:`GraphSourceError` and its missing-dependency install hint;
* the optional **Neo4j** adapter — driven through a faithful fake ``neo4j`` module (the same
  fake-module boundary the rerank/pack/qdrant tests use): the adapter composes one variable-length
  Cypher query with the hop bound inlined and the seeds bound, maps the records back to
  ``node → hop``, closes the driver, and maps a driver fault to a typed error leaking no URI/creds; a
  live-Neo4j smoke is gated on the dep + a URL env var (skipped offline).

Plus the :func:`build_graph_source` factory that selects an engine by config name.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types
from typing import Any

import pytest
from latence_retrieval import (
    DuckDBGraphSource,
    GraphSource,
    Neo4jGraphSource,
    NetworkxGraphSource,
    build_graph_source,
)
from latence_retrieval.errors import GraphSourceError
from latence_retrieval.multihop import (
    _DUCKDB_ADJACENCY_CACHE,
    _NETWORKX_GRAPH_CACHE,
    _traverse,
    clear_graph_source_caches,
)
from latence_retrieval.testing import FakeGraphSource

_HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None
_HAS_NETWORKX = importlib.util.find_spec("networkx") is not None

# A graph with a chain, a branch, and an unreachable component:
#   n0 - n1 - n2 - n3 ,  n0 - k1 ,  p - q (isolated)
_EDGES = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n0", "k1"), ("p", "q")]


def _write_edges_parquet(path: str, edges: list[tuple[str, str]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "source_node_id": [e[0] for e in edges],
            "target_node_id": [e[1] for e in edges],
        }
    )
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


# --- the faithful fake (the canonical semantics) ----------------------------------------------


def test_fake_source_satisfies_the_seam() -> None:
    assert isinstance(FakeGraphSource([]), GraphSource)


def test_fake_source_returns_min_hop_distances_seeds_excluded() -> None:
    got = FakeGraphSource(_EDGES).neighbors(["n0"], 3)
    assert got == {"n1": 1, "k1": 1, "n2": 2, "n3": 3}  # n0 (seed) excluded, min distance each


def test_fake_source_is_undirected() -> None:
    # Only a n1->n0 direction is stored; traversal from n0 must still reach n1 (undirected).
    got = FakeGraphSource([("n1", "n0")]).neighbors(["n0"], 1)
    assert got == {"n1": 1}


def test_fake_source_bounds_depth() -> None:
    assert FakeGraphSource(_EDGES).neighbors(["n0"], 1) == {"n1": 1, "k1": 1}


def test_fake_source_merges_minimum_over_multiple_seeds() -> None:
    # n2 is 1 hop from seed n1 and 2 hops from seed n0 → the minimum (1) wins.
    got = FakeGraphSource(_EDGES).neighbors(["n0", "n1"], 3)
    assert got["n2"] == 1


def test_fake_source_empty_seeds_returns_empty() -> None:
    assert FakeGraphSource(_EDGES).neighbors([], 2) == {}


def test_fake_source_missing_seed_contributes_nothing() -> None:
    assert FakeGraphSource(_EDGES).neighbors(["absent"], 2) == {}


@pytest.mark.parametrize("bad", [0, -1, 65, 1.5, True])
def test_fake_source_rejects_an_out_of_range_hop_bound(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_hops"):
        FakeGraphSource(_EDGES).neighbors(["n0"], bad)


# --- real DuckDB (gated on the extra) — equivalence to the faithful fake -----------------------


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
@pytest.mark.parametrize("max_hops", [1, 2, 3])
def test_real_duckdb_matches_the_faithful_fake(tmp_path: Any, max_hops: int) -> None:
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    seeds = ["n0", "n1"]
    duck = DuckDBGraphSource(uri).neighbors(seeds, max_hops)
    assert duck == FakeGraphSource(_EDGES).neighbors(seeds, max_hops)
    assert duck == _traverse(_EDGES, seeds, max_hops)


# A dense, cyclic graph (the complete graph K12 — every pair connected, so every node sits on many
# cycles) is what a real knowledge graph looks like. It is the audit-R1 regression fixture: on the
# *old* recursive-CTE (``UNION ALL``, no visited set) a bounded ``max_hops`` bounds depth but not
# work — every path is enumerated, so the walk relation grows ~``degree**max_hops`` and the query
# never returns; on the fixed engine (``UNION`` = global ``(node, hop)`` dedup) the same result is
# computed in bounded, polynomial work.
_K12 = [(f"n{i}", f"n{j}") for i in range(12) for j in range(12) if i < j]


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_real_duckdb_bounds_work_on_a_cyclic_graph_audit_r1(tmp_path: Any) -> None:
    """Regression (audit-R1): traversal of a cyclic graph must be bounded in *work*, not just depth.

    Fails on the pre-fix engine (``UNION ALL`` recursive CTE): with ``max_hops=9`` over K12 the
    walk enumerates ~``11**9`` (~2.4e9) paths and does not return within the budget, so the worker
    thread is still alive when the timeout elapses and the completion assertion fails (availability
    /DoS). Passes on the fixed engine (``UNION`` global dedup): the same neighbourhood — every other
    node is a direct 1-hop neighbour in K12 — is computed in well under the budget, and it equals the
    faithful fake / networkx reference exactly.
    """
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _K12)
    seeds = ["n0"]
    max_hops = 9  # would blow up under UNION ALL; within the ceiling (64)

    holder: dict[str, dict[str, int]] = {}

    def _run() -> None:
        holder["result"] = DuckDBGraphSource(uri).neighbors(seeds, max_hops)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=20.0)

    # Bounded work: the traversal returned inside the budget (the pre-fix UNION ALL query does not).
    assert not worker.is_alive(), (
        "duckdb multi-hop traversal did not complete within 20s on a cyclic graph — the recursive "
        "CTE is enumerating paths (missing visited-set / UNION dedup), an availability/DoS blowup."
    )
    # ...and the bounded query is still exactly the reference neighbourhood.
    expected = FakeGraphSource(_K12).neighbors(seeds, max_hops)
    assert holder["result"] == expected == {f"n{i}": 1 for i in range(1, 12)}


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_real_duckdb_empty_seeds_never_touches_a_missing_file() -> None:
    # A non-existent parquet is fine when there are no seeds (short-circuit, no engine touched).
    assert DuckDBGraphSource("/no/such/file.parquet").neighbors([], 2) == {}


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_real_duckdb_missing_parquet_is_a_typed_error_with_no_leak() -> None:
    secret_path = "/no/such/super-secret-path.parquet"
    with pytest.raises(GraphSourceError) as excinfo:
        DuckDBGraphSource(secret_path).neighbors(["n0"], 2)
    assert secret_path not in str(excinfo.value)  # only the exception type name, no raw path


def test_duckdb_missing_dependency_raises_a_clear_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"latence-retrieval\[graph-duckdb\]"):
        DuckDBGraphSource("x.parquet").neighbors(["n0"], 2)


# --- real networkx (gated on the extra) — equivalence to the faithful fake ---------------------


@pytest.mark.skipif(not _HAS_NETWORKX, reason="networkx extra not installed (offline)")
@pytest.mark.parametrize("max_hops", [1, 2, 3])
def test_real_networkx_matches_the_faithful_fake(tmp_path: Any, max_hops: int) -> None:
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    seeds = ["n0", "n1"]
    nx = NetworkxGraphSource(uri).neighbors(seeds, max_hops)
    assert nx == FakeGraphSource(_EDGES).neighbors(seeds, max_hops)


@pytest.mark.skipif(not _HAS_NETWORKX, reason="networkx extra not installed (offline)")
def test_real_networkx_missing_parquet_is_a_typed_error() -> None:
    with pytest.raises(GraphSourceError):
        NetworkxGraphSource("/no/such/file.parquet").neighbors(["n0"], 2)


def test_networkx_missing_dependency_raises_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(ImportError, match=r"latence-retrieval\[graph-networkx\]"):
        NetworkxGraphSource("x.parquet").neighbors(["n0"], 2)


# --- optional Neo4j — driven through a faithful fake `neo4j` module ----------------------------


class _FakeSession:
    def __init__(self, records: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        self._records = records
        self._calls = calls
        self.database: str | None = None

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    # Explicit signature (no **kwargs beyond the documented params): a param the real driver lacks
    # would surface here too.
    def run(self, cypher: str, seeds: list[str] | None = None) -> list[dict[str, Any]]:
        self._calls.append({"cypher": cypher, "seeds": seeds, "database": self.database})
        return self._records


class _FakeDriver:
    def __init__(self, records: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        self._records = records
        self._calls = calls
        self.closed = False

    def session(self, database: str | None = None) -> _FakeSession:
        session = _FakeSession(self._records, self._calls)
        session.database = database
        return session

    def close(self) -> None:
        self.closed = True


def _install_fake_neo4j(
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, Any]],
    *,
    fail: bool = False,
) -> dict[str, Any]:
    state: dict[str, Any] = {"calls": [], "conn": None, "drivers": []}

    class _GraphDatabase:
        @staticmethod
        def driver(uri: str, auth: Any = None) -> _FakeDriver:
            state["conn"] = {"uri": uri, "auth": auth}
            if fail:
                raise RuntimeError("bolt connection refused to super-secret-host")
            driver = _FakeDriver(records, state["calls"])
            state["drivers"].append(driver)
            return driver

    module = types.ModuleType("neo4j")
    module.GraphDatabase = _GraphDatabase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neo4j", module)
    return state


def test_neo4j_satisfies_the_seam() -> None:
    assert isinstance(Neo4jGraphSource("bolt://localhost:7687"), GraphSource)


def test_neo4j_maps_records_and_composes_a_bounded_cypher_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [{"node": "n1", "hop": 1}, {"node": "n2", "hop": 2}]
    state = _install_fake_neo4j(monkeypatch, records)
    source = Neo4jGraphSource(
        "bolt://localhost:7687", username="neo", password="secret", database="graph"  # pragma: allowlist secret
    )
    got = source.neighbors(["n0"], 3)

    assert got == {"n1": 1, "n2": 2}
    call = state["calls"][0]
    assert call["seeds"] == ["n0"]  # seeds bound as a parameter
    assert "*1..3" in call["cypher"]  # the hop bound is inlined into the variable-length pattern
    assert call["database"] == "graph"
    assert state["conn"]["auth"] == ("neo", "secret")
    assert state["drivers"][0].closed is True  # the driver is closed after the call


def test_neo4j_traversal_is_bounded_by_shortest_path_audit_r1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (audit-R1): the variable-length query must be a bounded ``shortestPath`` BFS.

    Fails on the pre-fix adapter, which composed a bare ``MATCH p = (s)-[*1..k]-(n)`` — on a cyclic
    graph Neo4j enumerates *every* path (~``degree**k`` of them) only to collapse them with
    ``min(length(p))``, an exponential-work availability/DoS blowup that bounds depth but not work.
    The fixed adapter wraps the pattern in ``shortestPath(...)`` so the engine runs a single bounded
    BFS per ``(s, n)`` pair. The composed query is the adapter's observable output across the seam
    (a live cyclic Neo4j cannot be stood up offline), so it is asserted directly — and the record
    mapping still holds.
    """
    records = [{"node": "n1", "hop": 1}, {"node": "n2", "hop": 2}]
    state = _install_fake_neo4j(monkeypatch, records)
    got = Neo4jGraphSource("bolt://localhost:7687").neighbors(["n0"], 4)

    cypher = state["calls"][0]["cypher"]
    assert "shortestPath(" in cypher  # bounded BFS, not raw path enumeration
    assert "*1..4" in cypher  # the hop bound is still inlined into the pattern
    assert got == {"n1": 1, "n2": 2}  # record mapping unchanged


def test_neo4j_empty_seeds_short_circuits_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_neo4j(monkeypatch, [])
    assert Neo4jGraphSource("bolt://localhost:7687").neighbors([], 2) == {}
    assert state["conn"] is None  # never opened a driver


def test_neo4j_driver_fault_is_a_typed_error_with_no_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_neo4j(monkeypatch, [], fail=True)
    source = Neo4jGraphSource("bolt://super-secret-host:7687", username="neo", password="hunter2")
    with pytest.raises(GraphSourceError) as excinfo:
        source.neighbors(["n0"], 2)
    message = str(excinfo.value)
    assert "hunter2" not in message
    assert "super-secret-host" not in message
    assert "refused" not in message  # no raw driver text, only the exception type name


def test_neo4j_requires_username_and_password_together() -> None:
    with pytest.raises(ValueError, match="username.*password"):
        Neo4jGraphSource("bolt://localhost:7687", username="neo")


def test_neo4j_missing_dependency_raises_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "neo4j", None)
    with pytest.raises(ImportError, match=r"latence-retrieval\[graph-neo4j\]"):
        Neo4jGraphSource("bolt://localhost:7687").neighbors(["n0"], 2)


@pytest.mark.skipif(
    importlib.util.find_spec("neo4j") is None or not os.environ.get("LATENCE_NEO4J_URI"),
    reason="live Neo4j validation is out of band (set LATENCE_NEO4J_URI + install the extra)",
)
def test_live_neo4j_smoke() -> None:  # pragma: no cover - runs only against a live store
    uri = os.environ["LATENCE_NEO4J_URI"]
    source = Neo4jGraphSource(
        uri,
        username=os.environ.get("LATENCE_NEO4J_USER"),
        password=os.environ.get("LATENCE_NEO4J_PASSWORD"),
    )
    seed = os.environ.get("LATENCE_NEO4J_SEED", "")
    result = source.neighbors([seed] if seed else [], 2)
    assert isinstance(result, dict)


# --- the build_graph_source factory -----------------------------------------------------------


def test_factory_builds_each_engine_by_name() -> None:
    assert isinstance(build_graph_source({"source": "duckdb", "edges_uri": "e.parquet"}), DuckDBGraphSource)
    assert isinstance(
        build_graph_source({"source": "networkx", "edges_uri": "e.parquet"}), NetworkxGraphSource
    )
    assert isinstance(
        build_graph_source({"source": "neo4j", "uri": "bolt://localhost:7687"}), Neo4jGraphSource
    )


def test_factory_rejects_an_unknown_source() -> None:
    with pytest.raises(ValueError, match="unknown graph source"):
        build_graph_source({"source": "kuzu", "edges_uri": "e.parquet"})


def test_factory_requires_the_parquet_uri_for_the_embedded_engines() -> None:
    with pytest.raises(ValueError, match="edges_uri"):
        build_graph_source({"source": "duckdb"})


def test_factory_requires_a_uri_for_neo4j() -> None:
    with pytest.raises(ValueError, match="uri"):
        build_graph_source({"source": "neo4j"})


def test_factory_rejects_a_non_mapping_spec() -> None:
    with pytest.raises(TypeError, match="mapping"):
        build_graph_source("nope")  # type: ignore[arg-type]


# --- the read cache behind the parquet engines (arch pass Wave 1-A, FINDINGS F1) ---------------
#
# `cache=True` (the default) serves traversals from a process-shared decoded structure keyed by
# the parquet's FILE IDENTITY; `cache=False` is the original per-call engine path, preserved
# verbatim as the parity oracle. The bar is byte-identity: same dict, same hops, on every graph —
# and a rewritten parquet is re-decoded, never served stale.


def _rewrite_edges_parquet(path: str, edges: list[tuple[str, str]]) -> None:
    """Rewrite the fixture parquet, forcing a distinct mtime_ns on coarse filesystem clocks."""
    _write_edges_parquet(path, edges)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
@pytest.mark.parametrize("max_hops", [1, 2, 3])
def test_cached_duckdb_is_byte_identical_to_the_uncached_oracle(
    tmp_path: Any, max_hops: int
) -> None:
    clear_graph_source_caches()
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    seeds = ["n0", "n1"]
    cached = DuckDBGraphSource(uri).neighbors(seeds, max_hops)
    uncached = DuckDBGraphSource(uri, cache=False).neighbors(seeds, max_hops)
    assert cached == uncached == FakeGraphSource(_EDGES).neighbors(seeds, max_hops)


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_duckdb_shares_one_decode_and_uncached_touches_no_cache(tmp_path: Any) -> None:
    clear_graph_source_caches()
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    DuckDBGraphSource(uri).neighbors(["n0"], 2)
    DuckDBGraphSource(uri).neighbors(["n1"], 2)  # a second instance over the same file
    assert len(_DUCKDB_ADJACENCY_CACHE) == 1  # one decode per file version, shared
    DuckDBGraphSource(uri, cache=False).neighbors(["n0"], 2)
    assert len(_DUCKDB_ADJACENCY_CACHE) == 1  # the oracle path never populates the cache


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_duckdb_reindexes_a_rewritten_parquet(tmp_path: Any) -> None:
    clear_graph_source_caches()
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    source = DuckDBGraphSource(uri)
    assert source.neighbors(["n0"], 1) == {"n1": 1, "k1": 1}
    # Rewrite with a different graph: the cached decode must NOT serve the old edges.
    _rewrite_edges_parquet(uri, [("n0", "z9")])
    assert source.neighbors(["n0"], 1) == {"z9": 1}
    assert source.neighbors(["n0"], 1) == DuckDBGraphSource(uri, cache=False).neighbors(["n0"], 1)


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_duckdb_missing_parquet_is_a_typed_error_with_no_leak() -> None:
    clear_graph_source_caches()
    secret_path = "/no/such/super-secret-path.parquet"
    with pytest.raises(GraphSourceError) as excinfo:
        DuckDBGraphSource(secret_path).neighbors(["n0"], 2)
    assert secret_path not in str(excinfo.value)  # the stat fault leaks no raw path either


@pytest.mark.skipif(not _HAS_NETWORKX, reason="networkx extra not installed (offline)")
@pytest.mark.parametrize("max_hops", [1, 2, 3])
def test_cached_networkx_is_byte_identical_to_the_uncached_oracle(
    tmp_path: Any, max_hops: int
) -> None:
    clear_graph_source_caches()
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    seeds = ["n0", "n1"]
    cached = NetworkxGraphSource(uri).neighbors(seeds, max_hops)
    uncached = NetworkxGraphSource(uri, cache=False).neighbors(seeds, max_hops)
    assert cached == uncached == FakeGraphSource(_EDGES).neighbors(seeds, max_hops)


@pytest.mark.skipif(not _HAS_NETWORKX, reason="networkx extra not installed (offline)")
def test_cached_networkx_shares_one_graph_and_reindexes_a_rewrite(tmp_path: Any) -> None:
    clear_graph_source_caches()
    uri = str(tmp_path / "graph-edges.parquet")
    _write_edges_parquet(uri, _EDGES)
    source = NetworkxGraphSource(uri)
    source.neighbors(["n0"], 2)
    NetworkxGraphSource(uri).neighbors(["n1"], 2)
    assert len(_NETWORKX_GRAPH_CACHE) == 1  # one loaded graph per file version, shared
    _rewrite_edges_parquet(uri, [("n0", "z9")])
    assert source.neighbors(["n0"], 2) == {"z9": 1}  # re-loaded, never stale
    assert len(_NETWORKX_GRAPH_CACHE) == 1  # the superseded version's graph was evicted


@pytest.mark.skipif(not _HAS_NETWORKX, reason="networkx extra not installed (offline)")
def test_cached_networkx_missing_parquet_is_a_typed_error() -> None:
    clear_graph_source_caches()
    with pytest.raises(GraphSourceError):
        NetworkxGraphSource("/no/such/file.parquet").neighbors(["n0"], 2)


def test_factory_passes_cache_through_and_rejects_a_non_boolean() -> None:
    for name in ("duckdb", "networkx"):
        cached = build_graph_source({"source": name, "edges_uri": "e.parquet"})
        uncached = build_graph_source({"source": name, "edges_uri": "e.parquet", "cache": False})
        assert cached._cache is True  # type: ignore[attr-defined]
        assert uncached._cache is False  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="cache"):
        build_graph_source({"source": "duckdb", "edges_uri": "e.parquet", "cache": "yes"})

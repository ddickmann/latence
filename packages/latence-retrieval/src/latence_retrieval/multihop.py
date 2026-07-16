# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The multi-hop ``expand`` path: :class:`MultiHopExpander` over a :class:`GraphSource` seam.

Where 1-hop ``expand`` (:class:`~latence_retrieval.expand.Expander`, ADR-0051's *default* path)
rescores candidates from the KG columns the pipeline *denormalised onto each chunk* — and so needs
**no** graph at all — multi-hop ``expand`` rewards candidates for graph proximity that only a real
**traversal of the emitted knowledge graph** can reveal: a seed's 2-, 3-, …, *k*-hop neighbourhood
is nowhere on the candidates (each chunk carries only its *own* 1-hop reach), so deeper expansion
must query the graph itself (CONTEXT ``Expand``; ADR-0051's *multi-hop* path). That is the one place
the query-time tooling needs the graph *queryable* — and the load-bearing decision (ADR-0051) is
that it stays **zero-ops**: the graph is queried **read-only, per call, over the very parquet the
pipeline already emitted** (``graph-edges.parquet``), with **no server to run** and **no new
artifact** — still strictly stateless (ADR-0048).

**The seam that varies — :class:`GraphSource`.** Multi-hop is exactly where behaviour varies across
engines, so the graph is reached through a genuine multi-adapter seam (the codebase-design bar),
deliberately *deep*: one method, :meth:`GraphSource.neighbors`, hides every traversal engine behind
``(seed node ids, hop bound) → {node id: minimum hop distance}``. Its adapters:

* :class:`DuckDBGraphSource` — the **zero-ops default** (ADR-0051): an embedded, serverless DuckDB
  runs a recursive-CTE traversal directly over ``graph-edges.parquet``, read-only, opening and
  closing an in-process connection per call (holds nothing). Behind the ``[graph-duckdb]`` extra.
* :class:`NetworkxGraphSource` — the **pure-Python reference for small graphs** (ADR-0051): loads
  the same edge parquet into an undirected ``networkx`` graph and reads shortest-path hop distances.
  Behind the ``[graph-networkx]`` extra.
* :class:`Neo4jGraphSource` — the **optional enterprise adapter** for shops that already run Neo4j
  (ADR-0051): a variable-length Cypher path query against a live deployment. Behind the
  ``[graph-neo4j]`` extra.
* :class:`~latence_retrieval.testing.FakeGraphSource` — the shipped **faithful fake** every offline
  test drives (the seam's second, always-available adapter): it runs the *canonical* traversal
  semantics (:func:`_traverse`) over an in-memory edge list, so a test can plant a graph and assert
  exactly which candidates a hop bound lifts — and the real adapters are proven **identical** to it
  on the same fixture graph (anti-false-green: the fake is not an independent re-implementation that
  could drift; it *is* the reference the engines must match).

**The processor — :class:`MultiHopExpander`.** A :class:`~latence_retrieval.processor.\
CandidateProcessor` (the ``(query, candidates) → candidates`` shape), it **accepts** a
:class:`GraphSource` (dependency injection, exactly as :class:`~latence_retrieval.rerank.Reranker`
accepts a :class:`~latence_retrieval.rerank.CrossEncoderModel`): a test hands it the fake,
production
hands it a config that builds the DuckDB/networkx/Neo4j adapter lazily — never a seam with a single
implementation. Its transform: take the top-``seed_top_k`` candidates as the graph anchor, read
their
entity node ids, traverse the graph out to a **bounded** ``max_hops`` (AC3), and give every
candidate
a hop-decayed bonus for the *closest* hop at which one of its own entity nodes falls in the seeds'
neighbourhood — then re-sort. It is stateless (ADR-0048): it holds no graph across calls, never
mutates its frozen input, and the same inputs always yield the same output.

**Where multi-hop differs from 1-hop (kept complementary, not redundant).** 1-hop ``expand`` rewards
the denormalised neighbourhood *and co-mention* of a shared entity with a flat, seed-count-scaled
bonus; multi-hop rewards **edge-path closeness** — a direct (1-hop) graph neighbour of a seed earns
``hop_weight``, a 2-hop neighbour ``hop_weight × hop_decay``, a *k*-hop neighbour
``hop_weight × hop_decay**(k-1)`` — a signal only a real traversal exposes. A caller who wants both
composes ``expand`` then ``multihop_expand``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_retrieval.contracts import Candidate
from latence_retrieval.errors import GraphSourceError

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

_DEFAULT_SEED_TOP_K = 3
_DEFAULT_HOP_WEIGHT = 0.5
_DEFAULT_MAX_HOPS = 2
_DEFAULT_HOP_DECAY = 0.5
# Hop depth is bounded (ADR-0051 AC): a hop bound above this is almost certainly a config mistake
# (an unbounded traversal over a large graph), so it is rejected at the boundary rather than run.
_MAX_HOPS_CEILING = 64


# --- the canonical traversal semantics --------------------------------------------------------


def _traverse(
    edges: Iterable[tuple[str, str]], seeds: Collection[str], max_hops: int
) -> dict[str, int]:
    """The reference multi-hop traversal: ``node id → minimum hop distance`` (1..``max_hops``).

    Undirected breadth-first search over ``edges`` (each ``(source, target)`` pair is walked in both
    directions, matching the 1-hop expander's direction-agnostic link check) from the ``seeds`` node
    set, returning every node reachable within ``max_hops`` hops keyed to its **minimum** hop
    distance. The seed nodes themselves are excluded (distance 0 is the anchor, not the expansion),
    so a seed never earns a self-bonus — the neighbours are what get lifted. Deterministic and
    stateless: it holds nothing and identical inputs always yield the same mapping.

    This is the *canonical* semantics the DuckDB / networkx / Neo4j adapters must reproduce and the
    faithful :class:`~latence_retrieval.testing.FakeGraphSource` runs directly, so equivalence to it
    on a fixture graph is the anti-false-green proof that an engine's traversal is faithful.
    """
    adjacency: dict[str, set[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    seen = set(seeds)
    frontier = set(seeds)
    result: dict[str, int] = {}
    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        nxt: set[str] = set()
        for node in frontier:
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    result[neighbour] = hop
                    nxt.add(neighbour)
        frontier = nxt
    return result


def _validate_max_hops(max_hops: int) -> int:
    """A hop bound must be a positive int within the ceiling — bounded traversal (ADR-0051 AC)."""
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 1:
        msg = f"max_hops must be a positive integer, got {max_hops!r}."
        raise ValueError(msg)
    if max_hops > _MAX_HOPS_CEILING:
        msg = f"max_hops must be <= {_MAX_HOPS_CEILING} (hop depth is bounded), got {max_hops}."
        raise ValueError(msg)
    return max_hops


def _dedupe_seeds(seeds: Collection[str]) -> list[str]:
    """The distinct seed node ids, order-preserved (a stable parameter for the engines)."""
    return list(dict.fromkeys(seeds))


# --- the GraphSource seam ----------------------------------------------------------------------


@runtime_checkable
class GraphSource(Protocol):
    """A read-only, stateless view of the emitted knowledge graph for multi-hop expansion.

    The whole seam multi-hop ``expand`` depends on — deliberately one deep method. Given a set of
    ``seeds`` (entity node ids to anchor on) and a **bounded** ``max_hops``, it returns every node
    reachable within that many undirected hops keyed to its **minimum** hop distance
    (1..``max_hops``),
    with the seeds themselves excluded. Every adapter (embedded DuckDB over parquet, a networkx
    reference, a live Neo4j) hides its traversal engine entirely behind this one call.

    The contract every adapter honours (so the faithful fake and the real engines are
    interchangeable): an empty ``seeds`` returns ``{}`` (nothing to expand, no engine touched); a
    ``max_hops`` below 1 or above the ceiling is a :class:`ValueError` at the boundary (bounded
    depth); a seed absent from the graph simply contributes nothing (graceful); and a runtime engine
    fault is a :class:`~latence_retrieval.errors.GraphSourceError` carrying no raw engine text. It
    holds nothing across calls — the graph is re-read read-only per call (ADR-0048).
    """

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """Return ``node id → minimum hop distance`` for the seeds' ≤``max_hops`` neighbourhood."""
        ...


class DuckDBGraphSource:
    """The zero-ops default :class:`GraphSource`: embedded DuckDB over ``graph-edges.parquet``.

    Runs a recursive-CTE traversal directly against the edge parquet the pipeline already emitted
    (ADR-0051) — an in-process, serverless engine, so there is **no graph database to operate**.
    Each
    :meth:`neighbors` call opens a fresh in-memory DuckDB connection, reads the parquet
    **read-only**,
    and closes it, so the source holds nothing between calls (statelessness is structural,
    ADR-0048).
    The parquet is never mutated — a graph query is inherently read-only. The traversal reproduces
    the
    canonical :func:`_traverse` semantics (undirected, minimum hop distance, seeds excluded), proven
    by an equivalence test against the faithful fake on the same fixture graph.

    Args:
        edges_uri: path to ``graph-edges.parquet`` (the emitted ``source_node_id`` /
        ``target_node_id``
            edge table). Required, non-empty.
        source_column / target_column: the edge endpoint column names. Default the emitted
            ``"source_node_id"`` / ``"target_node_id"``.

    Packaging: ``duckdb`` (MIT, an embedded engine — no server, no model) rides this package's
    ``[graph-duckdb]`` extra and is imported lazily on first :meth:`neighbors`, so constructing the
    source — which a registry/config does — never imports it or needs the extra.
    """

    def __init__(
        self,
        edges_uri: str,
        *,
        source_column: str = "source_node_id",
        target_column: str = "target_node_id",
    ) -> None:
        self._edges_uri = _require_non_empty_str(edges_uri, "edges_uri")
        self._source_column = _require_non_empty_str(source_column, "source_column")
        self._target_column = _require_non_empty_str(target_column, "target_column")

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """Traverse the edge parquet in DuckDB; ``node id → min hop distance``, seeds excluded."""
        max_hops = _validate_max_hops(max_hops)
        seed_list = _dedupe_seeds(seeds)
        if not seed_list:
            return {}
        duckdb = self._load()
        # The endpoint columns are validated identifiers (non-empty strings the operator configures,
        # not query input); the edge URI and the seed list travel as bound parameters, so the only
        # interpolated text is the operator's own column names.
        #
        # The recursive ``walk`` step is ``UNION`` (set, i.e. distinct), *not* ``UNION ALL`` — this
        # is load-bearing, not a style choice. Real knowledge graphs are cyclic and high-degree, and
        # a recursive CTE has no per-node visited set of its own: with ``UNION ALL`` every path is
        # enumerated, so on a cycle the working set grows ~``degree**max_hops`` (exponential *work*,
        # not just depth) even though ``min(hop)`` is still correct — the "bounded traversal" of
        # ADR-0051 would bound depth but not work, an availability/DoS blowup precisely on the
        # cyclic graphs this feature exists to traverse. ``UNION`` deduplicates ``(node, hop)`` over
        # iterations, so each distinct ``(node, hop)`` pair is expanded at most once and the total
        # work is bounded by ``max_hops × |edges|`` — polynomial, matching the reference
        # :func:`_traverse` (visited-set BFS) and networkx (bounded shortest-path). Same result set,
        # bounded work.
        sql = (
            "WITH RECURSIVE edges AS ("
            f"  SELECT {self._source_column} AS a, {self._target_column} AS b "
            "   FROM read_parquet(?) "
            "  UNION ALL "
            f"  SELECT {self._target_column} AS a, {self._source_column} AS b "
            "   FROM read_parquet(?)"
            "), "
            "seeds AS (SELECT unnest(?::VARCHAR[]) AS node), "
            "walk(node, hop) AS ("
            "  SELECT e.b, 1 FROM edges e JOIN seeds s ON e.a = s.node "
            "  UNION "
            "  SELECT e.b, w.hop + 1 FROM edges e JOIN walk w ON e.a = w.node WHERE w.hop < ?"
            ") "
            "SELECT node, min(hop) AS hop FROM walk "
            "WHERE node NOT IN (SELECT node FROM seeds) GROUP BY node"
        )
        connection = duckdb.connect(database=":memory:")
        try:
            rows = connection.execute(
                sql, [self._edges_uri, self._edges_uri, seed_list, max_hops]
            ).fetchall()
        except Exception as exc:
            # A native DuckDB fault (missing/unreadable parquet, bad column) → typed error, no leak.
            msg = f"duckdb multi-hop traversal failed: {type(exc).__name__}."
            raise GraphSourceError(msg) from exc
        finally:
            connection.close()
        return {str(node): int(hop) for node, hop in rows}

    @staticmethod
    def _load() -> Any:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            msg = (
                "The 'duckdb' package is required for zero-ops multi-hop graph expansion over the "
                "emitted graph parquet. Install it with: "
                "pip install 'latence-retrieval[graph-duckdb]'"
            )
            raise ImportError(msg) from exc
        return duckdb


class NetworkxGraphSource:
    """The pure-Python reference :class:`GraphSource` for small graphs (ADR-0051): ``networkx``.

    Loads the same emitted ``graph-edges.parquet`` into an **undirected** ``networkx`` graph and
    reads
    hop distances with ``single_source_shortest_path_length`` bounded by ``max_hops`` — the
    reference
    an adopter reaches for on a tiny graph where standing up DuckDB is overkill. Reproduces the
    canonical :func:`_traverse` semantics (proven by an equivalence test against the faithful fake).
    Read-only and stateless: it re-reads the parquet per call and holds nothing between calls.

    Args:
        edges_uri: path to ``graph-edges.parquet``. Required, non-empty.
        source_column / target_column: edge endpoint column names (defaults match the emitter).

    Packaging: ``networkx`` (BSD, pure-Python) rides this package's ``[graph-networkx]`` extra and
    is
    imported lazily on first :meth:`neighbors`; ``pyarrow`` (already the pipeline's parquet spine)
    is
    used to read the edge table. Constructing the source never imports either.
    """

    def __init__(
        self,
        edges_uri: str,
        *,
        source_column: str = "source_node_id",
        target_column: str = "target_node_id",
    ) -> None:
        self._edges_uri = _require_non_empty_str(edges_uri, "edges_uri")
        self._source_column = _require_non_empty_str(source_column, "source_column")
        self._target_column = _require_non_empty_str(target_column, "target_column")

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """Build an undirected networkx graph from the edge parquet; read bounded hop distances."""
        max_hops = _validate_max_hops(max_hops)
        seed_list = _dedupe_seeds(seeds)
        if not seed_list:
            return {}
        nx = self._load()
        try:
            edges = self._read_edges()
            graph = nx.Graph()
            graph.add_edges_from(edges)
            seed_set = set(seed_list)
            best: dict[str, int] = {}
            for seed in seed_list:
                if seed not in graph:
                    continue
                lengths = nx.single_source_shortest_path_length(graph, seed, cutoff=max_hops)
                for node, distance in lengths.items():
                    if distance == 0 or node in seed_set:
                        continue
                    if node not in best or distance < best[node]:
                        best[node] = int(distance)
        except GraphSourceError:
            raise
        except Exception as exc:
            msg = f"networkx multi-hop traversal failed: {type(exc).__name__}."
            raise GraphSourceError(msg) from exc
        return best

    def _read_edges(self) -> list[tuple[str, str]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - pyarrow is the pipeline's parquet spine
            msg = (
                "The 'pyarrow' package is required to read the graph edge parquet for the networkx "
                "multi-hop reference. Install it with: "
                "pip install 'latence-retrieval[graph-networkx]'"
            )
            raise ImportError(msg) from exc
        try:
            table = pq.read_table(  # type: ignore[no-untyped-call]
                self._edges_uri, columns=[self._source_column, self._target_column]
            )
        except Exception as exc:
            msg = f"networkx multi-hop could not read the edge parquet: {type(exc).__name__}."
            raise GraphSourceError(msg) from exc
        sources = table.column(self._source_column).to_pylist()
        targets = table.column(self._target_column).to_pylist()
        return [
            (str(s), str(t))
            for s, t in zip(sources, targets, strict=True)
            if s is not None and t is not None
        ]

    @staticmethod
    def _load() -> Any:
        try:
            import networkx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            msg = (
                "The 'networkx' package is required for the pure-Python multi-hop reference. "
                "Install it with: "
                "pip install 'latence-retrieval[graph-networkx]'"
            )
            raise ImportError(msg) from exc
        return networkx


class Neo4jGraphSource:
    """The optional enterprise :class:`GraphSource` (ADR-0051): a live Neo4j deployment.

    For shops that already run Neo4j — not a default the tooling stands up (that would be graph-DB
    ops, the very thing ADR-0051 refuses). It runs a **variable-length** Cypher path query against
    an
    existing graph, matching nodes by a configurable id property, and returns the seeds' bounded
    neighbourhood keyed to minimum hop distance — the same contract the embedded engines honour. The
    driver is opened and closed per call, so the source holds nothing between calls (the read-only,
    stateless boundary of ADR-0048 holds even against a server the *customer* operates).

    Args:
        uri: the Neo4j connection URI (e.g. ``"bolt://localhost:7687"``). Required, non-empty.
        username / password: optional basic-auth credentials (both or neither).
        database: optional target database name.
        node_id_property: the node property holding the graph node id used to match seeds and read
            results. Default ``"node_id"`` (the emitted node key).

    Packaging: the ``neo4j`` driver (Apache-2.0) rides this package's ``[graph-neo4j]`` extra and is
    imported lazily on first :meth:`neighbors`; constructing the source never imports it.
    """

    def __init__(
        self,
        uri: str,
        *,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        node_id_property: str = "node_id",
    ) -> None:
        self._uri = _require_non_empty_str(uri, "uri")
        if (username is None) != (password is None):
            msg = "Neo4jGraphSource requires both 'username' and 'password' together, or neither."
            raise ValueError(msg)
        self._auth: tuple[str, str] | None = (
            (username, password) if username is not None and password is not None else None
        )
        self._database = database
        self._node_id_property = _require_non_empty_str(node_id_property, "node_id_property")

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """Run a variable-length Cypher query; ``node id → min hop distance``, seeds excluded."""
        max_hops = _validate_max_hops(max_hops)
        seed_list = _dedupe_seeds(seeds)
        if not seed_list:
            return {}
        neo4j = self._load()
        prop = self._node_id_property
        # max_hops is a validated int (a variable-length bound cannot be a Cypher parameter); the
        # seed list travels as the bound $seeds parameter.
        #
        # The variable-length pattern is wrapped in ``shortestPath`` — load-bearing, not cosmetic.
        # Real knowledge graphs are cyclic and high-degree; a bare ``MATCH p = (s)-[*1..k]-(n)``
        # enumerates *every* path (there are ~``degree**k`` of them on a cycle) only to collapse
        # them with ``min(length(p))``, an exponential-work availability/DoS blowup that bounds
        # depth but not work. ``shortestPath`` makes Neo4j run a single bounded BFS per ``(s, n)``
        # pair instead, returning one shortest path — polynomial work, same ``min(hop)`` result,
        # matching the reference :func:`_traverse` and the DuckDB/networkx engines.
        cypher = (
            f"MATCH (s) WHERE s.`{prop}` IN $seeds "
            f"MATCH p = shortestPath((s)-[*1..{max_hops}]-(n)) "
            f"WHERE NOT n.`{prop}` IN $seeds "
            f"RETURN n.`{prop}` AS node, min(length(p)) AS hop"
        )
        try:
            driver = neo4j.GraphDatabase.driver(self._uri, auth=self._auth)
            try:
                with driver.session(database=self._database) as session:
                    result = session.run(cypher, seeds=seed_list)
                    rows = [(record["node"], record["hop"]) for record in result]
            finally:
                driver.close()
        except Exception as exc:
            msg = f"neo4j multi-hop traversal failed: {type(exc).__name__}."
            raise GraphSourceError(msg) from exc
        return {str(node): int(hop) for node, hop in rows if node is not None}

    @staticmethod
    def _load() -> Any:
        try:
            import neo4j
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            msg = (
                "The 'neo4j' package is required for the optional Neo4j multi-hop backend. "
                "Install it with: pip install 'latence-retrieval[graph-neo4j]'"
            )
            raise ImportError(msg) from exc
        return neo4j


_GRAPH_SOURCE_BUILDERS = ("duckdb", "networkx", "neo4j")


def build_graph_source(spec: Mapping[str, Any]) -> GraphSource:
    """Build a :class:`GraphSource` from a config block (the ``config["graph"]`` of the expander).

    ``spec["source"]`` selects the adapter by name — ``"duckdb"`` (the zero-ops default),
    ``"networkx"`` (the small-graph reference), or ``"neo4j"`` (the optional enterprise backend) —
    so
    swapping the traversal engine is a one-line config change (user story 36). The remaining keys
    are
    the selected adapter's constructor arguments (``edges_uri`` for the parquet engines; ``uri`` +
    optional auth/database for Neo4j). An unknown ``source`` fails loudly at wiring time.
    """
    if not isinstance(spec, Mapping):
        msg = f"graph source spec must be a mapping, got {type(spec).__name__}."
        raise TypeError(msg)
    source = spec.get("source")
    if source == "duckdb":
        return DuckDBGraphSource(
            _require_non_empty_str(spec.get("edges_uri"), "edges_uri"),
            source_column=str(spec.get("source_column", "source_node_id")),
            target_column=str(spec.get("target_column", "target_node_id")),
        )
    if source == "networkx":
        return NetworkxGraphSource(
            _require_non_empty_str(spec.get("edges_uri"), "edges_uri"),
            source_column=str(spec.get("source_column", "source_node_id")),
            target_column=str(spec.get("target_column", "target_node_id")),
        )
    if source == "neo4j":
        return Neo4jGraphSource(
            _require_non_empty_str(spec.get("uri"), "uri"),
            username=_optional_str(spec.get("username"), "username"),
            password=_optional_str(spec.get("password"), "password"),
            database=_optional_str(spec.get("database"), "database"),
            node_id_property=str(spec.get("node_id_property", "node_id")),
        )
    msg = (
        f"unknown graph source {source!r}; expected one of {_GRAPH_SOURCE_BUILDERS} "
        "(or inject a GraphSource directly)."
    )
    raise ValueError(msg)


class MultiHopExpander:
    """Graph-augment a candidate list by *multi-hop* KG proximity to its top hits (ADR-0051/0048).

    A :class:`~latence_retrieval.processor.CandidateProcessor`: :meth:`process` anchors on the
    top-``seed_top_k`` candidates, reads their entity node ids, traverses the emitted graph through
    the injected :class:`GraphSource` out to a **bounded** ``max_hops``, and adds each candidate a
    hop-decayed bonus for the **closest** hop at which one of its own entity nodes lands in the
    seeds'
    neighbourhood — a direct (1-hop) neighbour earns ``hop_weight``, a *k*-hop neighbour
    ``hop_weight × hop_decay**(k-1)`` — then returns a new list sorted by descending augmented score
    (ties broken by ascending ``id`` for determinism, as fuse/rerank/expand hold). The input is
    never
    mutated (:class:`~latence_retrieval.contracts.Candidate` is frozen) and the expander keeps no
    state across calls (ADR-0048): the graph is the source's, re-read read-only per call.

    It **accepts** a :class:`GraphSource` (dependency injection): a test hands it the faithful
    :class:`~latence_retrieval.testing.FakeGraphSource`, production hands it a ``config["graph"]``
    block that builds the DuckDB/networkx/Neo4j adapter lazily on first use — never a seam with a
    single implementation.

    Config:
        seed_top_k: int — how many top-scored candidates anchor the traversal (the seed set).
        Positive
            int, default ``3``.
        hop_weight: float — the bonus a **direct** (1-hop) graph neighbour of a seed earns; deeper
            hops decay from it. ``>= 0``, default ``0.5`` (``0`` disables the transform).
        max_hops: int — the **bounded** traversal depth (AC3). Positive int within the ceiling
            (``64``), default ``2``.
        hop_decay: float — the per-extra-hop geometric decay, in ``(0, 1]``. Default ``0.5`` (a
        2-hop
            neighbour earns half a 1-hop neighbour's bonus, a 3-hop a quarter, …).
        node_ids_key: str — metadata key holding each candidate's own entity node ids. Default
            ``"kg_node_ids"`` (the emitted denormalised column, shared with 1-hop ``expand``).
        top_k: int | None — if set (positive int), truncate the augmented list to its top ``top_k``.
            Default ``None`` keeps every candidate, reordered.
        graph: mapping — the :func:`build_graph_source` spec (``{"source": "duckdb"|"networkx"|\
            "neo4j", ...}``) used to build the source lazily when none is injected. Required unless
            a
            ``graph_source`` is injected.

    A candidate lacking the node-ids key simply earns nothing (graceful, as 1-hop ``expand`` is); a
    *present* key must be a list of non-empty id strings or :meth:`process` raises at the boundary
    (anti-false-green). When no candidate carries any node id — or ``hop_weight`` is ``0`` — the
    transform is an order-preserving no-op (up to the deterministic ``id`` tiebreak) and never
    touches
    the graph source.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, *, graph_source: GraphSource | None = None
    ) -> None:
        cfg = dict(config or {})
        self._seed_top_k = _positive_int(cfg.get("seed_top_k", _DEFAULT_SEED_TOP_K), "seed_top_k")
        self._hop_weight = _non_negative(cfg.get("hop_weight", _DEFAULT_HOP_WEIGHT), "hop_weight")
        self._max_hops = _validate_max_hops(cfg.get("max_hops", _DEFAULT_MAX_HOPS))
        self._hop_decay = _unit_interval(cfg.get("hop_decay", _DEFAULT_HOP_DECAY), "hop_decay")
        self._node_ids_key = _require_non_empty_str(
            cfg.get("node_ids_key", "kg_node_ids"), "node_ids_key"
        )
        top_k = cfg.get("top_k")
        if top_k is not None:
            top_k = _positive_int(top_k, "top_k")
        self._top_k: int | None = top_k

        graph_spec = cfg.get("graph")
        if graph_spec is not None and not isinstance(graph_spec, dict):
            msg = (
                f"multihop_expand 'graph' config must be a mapping if present, got {graph_spec!r}."
            )
            raise TypeError(msg)
        self._graph_spec: dict[str, Any] | None = dict(graph_spec) if graph_spec else None
        self._graph_source: GraphSource | None = graph_source

    def _resolve_source(self) -> GraphSource:
        if self._graph_source is None:
            if not self._graph_spec:
                msg = (
                    "multihop_expand needs a graph to traverse: pass a 'graph' source config "
                    "(e.g. {'source': 'duckdb', 'edges_uri': '.../graph-edges.parquet'}) or inject "
                    "a GraphSource."
                )
                raise ValueError(msg)
            self._graph_source = build_graph_source(self._graph_spec)
        return self._graph_source

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Traverse the graph from the seed set and rescore ``candidates`` by hop-decayed proximity.

        Anchors on the top-``seed_top_k`` candidates, reads their entity node ids, asks the
        :class:`GraphSource` for the ≤``max_hops`` neighbourhood, adds each candidate the
        hop-decayed
        bonus for the closest hop one of its own nodes appears at, and returns a new list in
        descending augmented score (ascending-``id`` tiebreak), optionally truncated to ``top_k``.
        The input is never mutated; ``query`` is part of the seam signature but the expander
        augments
        the first-stage score with the *graph* signal (it does not re-score against the query text —
        that is ``rerank``'s job). An empty input returns ``[]`` and never touches the graph source;
        so does a call where no candidate carries a node id (a pure no-op).
        """
        items = list(candidates)
        if not items:
            return []

        candidate_nodes = [self._node_ids(candidate) for candidate in items]
        if not self._hop_weight or not any(candidate_nodes):
            # No graph signal is possible (weight off, or no node ids at all): an order-preserving
            # no-op that never touches the graph source (statelessness + zero-ops both honoured).
            return sorted(items, key=lambda c: (-c.score, c.id))

        seed_indices = self._seed_indices(items)
        seed_nodes: set[str] = set()
        for index in seed_indices:
            seed_nodes |= candidate_nodes[index]

        distances: dict[str, int] = (
            self._resolve_source().neighbors(seed_nodes, self._max_hops) if seed_nodes else {}
        )

        augmented: list[Candidate] = []
        for candidate, nodes in zip(items, candidate_nodes, strict=True):
            bonus = self._bonus(nodes, distances)
            augmented.append(candidate.model_copy(update={"score": candidate.score + bonus}))
        augmented.sort(key=lambda c: (-c.score, c.id))
        if self._top_k is not None:
            return augmented[: self._top_k]
        return augmented

    def _bonus(self, nodes: set[str], distances: dict[str, int]) -> float:
        """The hop-decayed bonus for a candidate: the closest hop any of its nodes reaches."""
        hits = [distances[node] for node in nodes if node in distances]
        if not hits:
            return 0.0
        closest = min(hits)
        return self._hop_weight * (self._hop_decay ** (closest - 1))

    def _seed_indices(self, items: Sequence[Candidate]) -> list[int]:
        """Indices of the top-``seed_top_k`` candidates by first-stage score (descending, id tie).

        Chosen off the *original* scores so the anchor is stable: identical inputs always select the
        same seeds regardless of input order among score ties (the determinism 1-hop ``expand``
        holds).
        """
        order = sorted(range(len(items)), key=lambda i: (-items[i].score, items[i].id))
        return order[: self._seed_top_k]

    def _node_ids(self, candidate: Candidate) -> set[str]:
        """Read ``metadata[node_ids_key]`` as a set of id strings; ``set()`` if the key is absent.

        Absent → no signal (graceful, as 1-hop ``expand``). Present-but-malformed → loud: the value
        must be a list/tuple of non-empty strings (what the denormalised KG column is), so a dict, a
        scalar, or a non-string element is a boundary error, not a silently-dropped signal
        (anti-false-green).
        """
        key = self._node_ids_key
        if key not in candidate.metadata:
            return set()
        raw = candidate.metadata[key]
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            msg = (
                f"multihop_expand metadata {key!r} must be a list of node-id strings, got {raw!r} "
                f"for candidate {candidate.id!r}."
            )
            raise TypeError(msg)
        result: set[str] = set()
        for element in raw:
            if not isinstance(element, str) or not element:
                msg = (
                    f"multihop_expand metadata {key!r} must contain non-empty strings, got "
                    f"{element!r} for candidate {candidate.id!r}."
                )
                raise ValueError(msg)
            result.add(element)
        return result


# --- construction-time config parsing (shared) ------------------------------------------------


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"multihop_expand {name!r} must be a positive integer, got {value!r}."
        raise ValueError(msg)
    return value


def _non_negative(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        msg = f"multihop_expand {name!r} must be a number >= 0, got {value!r}."
        raise ValueError(msg)
    return float(value)


def _unit_interval(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 < value <= 1):
        msg = f"multihop_expand {name!r} must be a number in (0, 1], got {value!r}."
        raise ValueError(msg)
    return float(value)


def _require_non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{name!r} must be a non-empty string, got {value!r}."
        raise ValueError(msg)
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        msg = f"{name!r} must be a non-empty string if present, got {value!r}."
        raise ValueError(msg)
    return value


__all__ = [
    "DuckDBGraphSource",
    "GraphSource",
    "MultiHopExpander",
    "Neo4jGraphSource",
    "NetworkxGraphSource",
    "build_graph_source",
]

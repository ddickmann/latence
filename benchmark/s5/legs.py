# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The cumulative leg ladder: dense → +bm25 → +KG → +hyperedges → +CoE → +packer.

Each leg is the previous one **plus** one signal, so every row of the S5 table attributes one
increment. All scoring is done by the shipped ``latence-retrieval`` components — this module
only supplies the two things a stateless library deliberately does not own (ADR-0048): the
caller's first-stage *engine*, and the caller's query-signal generation.

**What the harness legitimately implements, and why it is not "reimplemented scoring".**

* :class:`DenseBackend` — the S3 export ships vectors but no ANN index (the ticket's F-VEC
  residual). A ``RetrievalBackend`` is by contract *the caller's engine*; brute-force cosine
  over an L2-normalised matrix is one ``numpy`` matmul and is exact, so an approximate index
  would only add error. Memory is bounded by scoring in query blocks — never an all-queries ×
  all-chunks matrix.
* :class:`Bm25Backend` — the export ships an inverted index (``bm25-postings.parquet``) and
  corpus statistics (``bm25-stats.json``), but the library's BM25 is a *rescorer* over a
  candidate pool (ADR-0048: it never builds a first stage — that is the caller's job). So the
  backend builds the first stage itself, as one :class:`_Bm25TermMatrix`: the export's postings
  turned into a term-major sparse matrix of the **query-independent** per-document BM25 weights,
  so a query is a single sparse mat-vec against the global IDF vector. The formula, the constants
  and the corpus statistics are not re-derived — they are read off the shipped
  :class:`~latence_retrieval.bm25.Bm25Rescorer` built with
  :meth:`~latence_retrieval.bm25.Bm25Rescorer.from_export`, which stays the definition of the
  numbers and is the oracle ``tests/test_legs.py`` checks every score and every rank against on
  the real export. Documents holding no query term score exactly ``0`` under Okapi BM25 and are
  dropped, exactly as the pool-prefilter used to drop them — vectorising changes cost, not
  results.
* :class:`EdgeGraphSource` — the shipped ``NetworkxGraphSource`` re-reads and rebuilds the edge
  graph on *every* call, which would dominate a per-query latency measurement. This holds the
  same networkx graph across a run and answers with the same call; ``tests/test_legs.py``
  asserts it returns exactly what ``NetworkxGraphSource`` returns on the real edge parquet.
* :class:`ChunkGraphDocumentSource` — the ``duckdb`` extra is not in this repo's dev environment,
  so the node→document index is served from the already-loaded ``kg_node_ids`` column, honouring
  the same ``GraphDocumentSource`` contract (each document at most once, ascending id).

Everything else is the real thing: :class:`~latence_retrieval.graph_retrieval.GraphRetriever`
and :class:`~latence_retrieval.graph_retrieval.AliasQuerySeeder` for leg 3,
:func:`~latence_retrieval.hypergraph.hypergraph_signals` for leg 4,
:func:`~latence_retrieval.coe.select` for leg 5,
:class:`~latence_retrieval.pack.Packer` for leg 6, and
:class:`~latence_retrieval.fuser.RrfFuser` to merge the legs — rank fusion, because the legs'
scores (cosine, Okapi, hop decay, ρ) are on flatly incomparable scales.

**Fusion weights.** ``RrfFuser`` sums ``weight_i / (k + rank_i)`` over the lists a candidate
appears in, so the *relative* weights of the fused lists are a real knob and not a formality:
appending a noisier list at the same weight as a strong one necessarily drags the fused ranking
towards the noise. The ladder's default is — and stays — **equal weights** (``weights=None``,
passed straight through to the fuser), because that is the assumption-free baseline and the one
the S5 matrix was run under. :mod:`benchmark.s5.tuning` supplies the second variant: per-signal
weights fitted on a dev split and reported on a disjoint test split. Nothing here chooses between
them; :func:`fuse_signals` just makes the weight vector reachable, aligned by signal *name* rather
than by list position so a weight can never land on the wrong list.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_RETRIEVAL_SRC = Path(__file__).resolve().parents[2] / "packages" / "latence-retrieval" / "src"
if str(_RETRIEVAL_SRC) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(_RETRIEVAL_SRC))

from benchmark.s5.corpus import (  # noqa: E402
    CONTENT_ONLY_VECTORS,
    Export,
    hyperedge_chunk_map,
    node_alias_map,
)
from latence_retrieval.bm25 import Bm25Rescorer  # noqa: E402
from latence_retrieval.coe import select as coe_select  # noqa: E402
from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery  # noqa: E402
from latence_retrieval.fuser import RrfFuser  # noqa: E402
from latence_retrieval.graph_retrieval import (  # noqa: E402
    AliasQuerySeeder,
    GraphDocument,
    GraphRetriever,
)
from latence_retrieval.hypergraph import (  # noqa: E402
    HyperedgeKV,
    hypergraph_signals,
)
from latence_retrieval.pack import Packer  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

#: The legs, in ladder order. Each name is the *increment*; leg N runs signals 1..N.
LEG_NAMES = ("dense", "bm25", "kg", "hyperedges", "coe", "packer")

#: Ablation legs — deliberately NOT rungs of the ladder above. Each re-runs ONE rung against a
#: different *input* so the report can attribute that input, and each is reported in its own
#: section so the ladder's cumulative Δ chain stays a chain of single-signal increments.
#:
#: ``dense_content_only`` is the ``dense`` rung ranked with vectors built from each chunk's
#: CONTENT ONLY, against the ``dense`` rung's vectors which embed content **plus** the W13
#: ContextHeader (the chunk's KG entities and neighbour triples) — because that is what the S3
#: Export wrote into ``corpus.parquet``. Without this leg the header would be an invisible
#: advantage folded into every dense number, and worse, an *uneven* one: the S3 corpora carry
#: populated headers while the reused M6 exports carry ``context_header: null`` on every row, so
#: the same "dense" label would mean different things per dataset. Only the CORPUS side changes —
#: queries have no context header, so both legs rank the identical query vectors.
ABLATION_LEG_NAMES = ("dense_content_only",)

#: Every leg name the matrix accepts, ladder first.
ALL_LEG_NAMES = (*LEG_NAMES, *ABLATION_LEG_NAMES)

#: Which ladder leg each ablation leg is the counterfactual of — the pair the report subtracts.
ABLATION_BASELINE = {"dense_content_only": "dense"}

#: The ranked lists that can enter the fusion, in the order the ladder appends them. ``packer`` is
#: deliberately absent: it is a post-fusion *selector* over the fused list, not a list to fuse.
FUSION_SIGNALS = ("dense", "bm25", "kg", "hyperedges", "coe")

#: The fusible signals each leg contributes. Cumulative along the ladder; ``packer`` fuses exactly
#: what ``coe`` fuses and then packs the result, and each ablation leg is its own single signal
#: (it fuses nothing, which is what makes it subtractable against the rung it mirrors).
_LEG_SIGNALS: dict[str, tuple[str, ...]] = {
    "dense": ("dense",),
    "bm25": ("dense", "bm25"),
    "kg": ("dense", "bm25", "kg"),
    "hyperedges": ("dense", "bm25", "kg", "hyperedges"),
    "coe": FUSION_SIGNALS,
    "packer": FUSION_SIGNALS,
    "dense_content_only": ("dense_content_only",),
}


def leg_signals(leg: str) -> tuple[str, ...]:
    """The signal names ``leg`` may put into the fusion, in ladder order.

    "May", not "does": a signal whose backend is absent, or that returned nothing for a query,
    contributes no list at all — :meth:`Runner.signals` reports what a query *actually* produced.
    """
    return _LEG_SIGNALS[leg]


def tunable_signals(leg: str) -> tuple[str, ...]:
    """The signals whose weight is worth fitting for ``leg`` — every one except the first.

    An RRF weight vector has one redundant degree of freedom: scaling *every* weight by the same
    positive constant leaves ``weight_i / (k + rank_i)`` ordered identically. So the first signal
    (always ``dense``, the rung the ladder starts from) is pinned at ``1.0`` and the rest are
    fitted relative to it — the same search, one dimension smaller. A single-signal leg therefore
    has nothing to tune at all, which is correct rather than a shortcut: with one list the weight
    is a pure monotone scale and every value gives the identical ranking.
    """
    return _LEG_SIGNALS[leg][1:]


def vector_source(leg: str) -> str:
    """Where ``leg``'s dense vectors come from: ``"corpus"`` or ``"sidecar"``.

    ``corpus`` is ``corpus.parquet``'s ``embedding`` column (content + ContextHeader, as
    exported); ``sidecar`` is :data:`~benchmark.s5.corpus.CONTENT_ONLY_VECTORS` next to it. One
    function, so no call site has to re-derive the mapping from a leg name.
    """
    return "sidecar" if leg in ABLATION_LEG_NAMES else "corpus"

#: Candidates each first-stage leg returns before fusion. Wide enough that recall@10 is a
#: property of the legs rather than of the truncation.
POOL_K = 100

#: The packer's token budget (leg 6) — the "what actually fits in a prompt" question the ladder
#: ends on. Reported against tokens actually packed.
PACK_BUDGET = 4000

#: The CoE selector's token budget (leg 5), matching M6's mid-range operating point.
COE_BUDGET = 4000

_WORD = re.compile(r"[\w][\w&.\-]*")


def query_ngrams(question: str, max_n: int = 4) -> list[str]:
    """Descending-length word n-grams of a question — the tier-1 zero-model entry surfaces.

    Byte-identical to the M6 harness's ``question_ngrams`` (``benchmark/musique-subset``), so
    the hypergraph leg's entry behaves the same way it did in the M6 numbers this run is
    compared against.
    """
    words = _WORD.findall(question)
    grams: list[str] = []
    for n in range(max_n, 0, -1):
        grams.extend(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


# -- the caller's engines (ADR-0048: the backend is ours to supply, the scoring is not) --------


class DenseBackend:
    """Exact brute-force cosine over the export's L2-normalised vectors (a ``RetrievalBackend``).

    ``matrix`` overrides the export's own vectors with an alternative source (the content-only
    sidecar) while every other view of the corpus — rows, ids, text — stays the one loaded export.
    Whichever matrix it is given, the row count must equal the corpus's: a matrix off by one row
    still ranks, and every id it returns would be the wrong chunk's, so the mismatch is refused at
    construction rather than discovered in a recall number.
    """

    def __init__(
        self,
        export: Export,
        *,
        matrix: Any | None = None,
        source: str = "corpus.parquet:embedding",
    ) -> None:
        vectors = export.embeddings if matrix is None else matrix
        if vectors is None:
            msg = f"{export.dataset}: corpus.parquet carries no 'embedding' column."
            raise ValueError(msg)
        if int(vectors.shape[0]) != len(export.rows):
            msg = (
                f"{export.dataset}: {source} has {int(vectors.shape[0])} vectors for "
                f"{len(export.rows)} corpus rows; refusing to rank a misaligned matrix."
            )
            raise ValueError(msg)
        self._export = export
        self._matrix = vectors
        self._source = source

    @property
    def source(self) -> str:
        """A human-readable name for where this backend's vectors came from."""
        return self._source

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{DENSE}`` — a vector store, nothing more."""
        return frozenset({Capability.DENSE})

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Top-``top_k`` by cosine; ties broken by ascending id, like every leg here."""
        vector = np.asarray(query.vector, dtype=np.float32)
        scores = self._matrix @ vector
        top = min(query.top_k, scores.shape[0])
        part = np.argpartition(-scores, top - 1)[:top]
        order = part[np.argsort(-scores[part], kind="stable")]
        rows = self._export.rows
        return [
            Candidate(id=rows[i].record_id, score=float(scores[i]), text=rows[i].content)
            for i in order
        ]


def _single_array(column: Any) -> Any:
    """One contiguous arrow ``Array`` for a (possibly multi-chunk) table column."""
    combined = column.combine_chunks()
    return combined.chunk(0) if isinstance(combined, pa.ChunkedArray) else combined


class _Bm25TermMatrix:
    """The export's BM25 document-term weights as a term-major CSR — built once, one matmul/query.

    **Why this exists.** The shipped rescorer is a candidate processor: it re-tokenises every
    candidate it is handed and accumulates the Okapi sum in Python. Handing it "every document
    holding at least one query term" is, on a real corpus, most of the corpus — on the 57,812-chunk
    ``wiki2multihop`` export the median query selects 53,398 chunks (92%), which cost 1.78 s per
    query and made BM25 the single dominant cost of the whole S5 sweep. That is a property of the
    *engine*, not of the formula.

    **The decomposition.** Split the Okapi score into its query-independent and query-dependent
    halves::

        score(d) = Σ_{t ∈ Q ∩ d} idf(t) · w(t, d)
        w(t, d)  = tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 − b + b · |d| / avgdl) )

    ``w`` depends only on the corpus, so it is computed **once** for every ``(term, document)`` pair
    ``bm25-postings.parquet`` records — that artifact carries ``doc_len`` and ``term_frequencies``
    beside ``terms``, i.e. exactly ``|d|`` and ``tf`` — and stored term-major (CSR over
    ``vocabulary × documents``). A query is then ``scores = W[Q].T @ idf(Q)``: one sparse mat-vec
    over the few thousand-to-million stored entries of the query's own terms, with no Python loop
    over documents and no re-tokenisation of anything but the query.

    **Why the numbers are the same numbers.** Nothing here re-derives BM25. ``k1``, ``b``, ``avgdl``
    and the per-term IDF are read off the *shipped* :class:`~latence_retrieval.bm25.Bm25Rescorer`
    instance built by :meth:`~latence_retrieval.bm25.Bm25Rescorer.from_export` — including its
    ``_CorpusStats.idf``, so the Lucene/Robertson IDF and the **global** ``bm25-stats.json``
    statistics are literally the same code and the same numbers, and cannot drift. ``w`` is
    evaluated in ``float64`` in the same operation order as
    :meth:`~latence_retrieval.bm25.Bm25Rescorer._term_saturation`, so each ``(t, d)`` product is
    bit-identical to the rescorer's. The one and only difference is the *order* the per-term
    products are summed in (ascending term id here; Python ``set`` iteration order there), which is
    a last-ulp effect: ``tests/test_legs.py`` pins agreement with the rescorer on a real export at
    ``1e-9`` relative on every score and exact equality on the returned order.

    **Not scipy.** The design is scipy's CSR mat-vec, but this repo's environment does not carry
    scipy (it is in no ``pyproject`` dependency group, and adding one to a harness while a sweep is
    running is the wrong trade). The three CSR arrays and the mat-vec are therefore plain ``numpy``
    — the same linear algebra, one fewer dependency, and a summation order this module controls
    (which is what makes the equivalence above assertable rather than hopeful).

    **Footprint.** ``indices`` is ``int32`` and ``data`` ``float64``, so 12 bytes per stored pair
    plus ``8·(|vocabulary| + 1)``. Measured on the largest export (``uda``: 220,129 chunks, 497,559
    terms, 35,156,494 pairs) the arrays are 430 MB and the term lookup ~60 MB, built in ~20 s with
    a ~3.1 GB transient peak (the doc-major→term-major permutation) — comfortably inside the
    sweep's 29 GB / 4-process budget, and below what the ``{term: [row, ...]}`` Python postings
    dict it replaces held resident for the same export.
    """

    __slots__ = ("_data", "_idf", "_indices", "_indptr", "_lookup", "n_docs")

    def __init__(self, export: Export, rescorer: Bm25Rescorer) -> None:
        stats = rescorer._supplied_stats  # noqa: SLF001 - the ONE source of truth
        if stats is None:
            # `from_export` always supplies them; a rescorer without them would silently switch to
            # POOL-LOCAL IDF, which is a different ranking wearing the same name.
            msg = (
                f"{export.dataset}: the BM25 rescorer carries no corpus statistics; refusing to "
                "build a first stage that would silently use pool-local IDF."
            )
            raise ValueError(msg)
        k1, b, avgdl = rescorer._k1, rescorer._b, stats.avgdl  # noqa: SLF001

        path = export.root / "bm25-postings.parquet"
        record_ids = pq.read_table(path, columns=["record_id"]).column("record_id").to_pylist()  # type: ignore[no-untyped-call]
        if record_ids != [row.record_id for row in export.rows]:
            # The postings row order IS the corpus row order in every S3 export, and the scoring
            # below addresses corpus rows by postings row. A mismatch would rank real scores under
            # the wrong ids — confident nonsense — so it is refused here, not discovered in a recall
            # number.
            msg = (
                f"{export.dataset}: bm25-postings.parquet holds {len(record_ids)} rows that are "
                f"not corpus.parquet's {len(export.rows)} record ids in order; refusing to score a "
                "misaligned index."
            )
            raise ValueError(msg)
        n_docs = len(record_ids)
        del record_ids

        doc_len = (
            pq.read_table(path, columns=["doc_len"])  # type: ignore[no-untyped-call]
            .column("doc_len")
            .to_numpy()
            .astype(np.float64)
        )
        frequencies = _single_array(
            pq.read_table(path, columns=["term_frequencies"]).column("term_frequencies")  # type: ignore[no-untyped-call]
        )
        # `data` starts life as tf and is turned into w(t, d) in place — one float64 array of the
        # stored-pair length instead of three, which is what keeps the uda build inside its budget.
        data = frequencies.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        per_doc = np.diff(frequencies.offsets.to_numpy())
        del frequencies

        terms = _single_array(pq.read_table(path, columns=["terms"]).column("terms"))  # type: ignore[no-untyped-call]
        flat = terms.flatten()
        if flat.null_count:
            msg = f"{export.dataset}: bm25-postings.parquet 'terms' holds nulls."
            raise ValueError(msg)
        encoded = flat.dictionary_encode()
        codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32)
        vocabulary: list[str] = encoded.dictionary.to_pylist()
        del terms, flat, encoded
        if codes.shape[0] != data.shape[0]:
            msg = (
                f"{export.dataset}: bm25-postings.parquet has {codes.shape[0]} terms against "
                f"{data.shape[0]} term_frequencies."
            )
            raise ValueError(msg)

        # w(t, d), in the rescorer's own operation order so every product is bit-identical to
        # `Bm25Rescorer._term_saturation(tf, |d|, avgdl)`.
        document_of = np.repeat(np.arange(n_docs, dtype=np.int32), per_doc)
        denominator = (1.0 - b + b * (doc_len / avgdl))[document_of]
        denominator *= k1
        denominator += data
        data *= k1 + 1.0
        data /= denominator
        del denominator, doc_len

        # Doc-major (as the parquet stores it) → term-major, the orientation a query slices.
        order = np.argsort(codes, kind="stable")
        indptr = np.zeros(len(vocabulary) + 1, dtype=np.int64)
        np.cumsum(np.bincount(codes, minlength=len(vocabulary)), out=indptr[1:])
        del codes
        self._indptr = indptr
        self._indices = document_of[order]
        del document_of
        self._data = data[order]
        del data, order

        self.n_docs = n_docs
        self._idf = np.fromiter(
            (stats.idf(term) for term in vocabulary), dtype=np.float64, count=len(vocabulary)
        )
        self._lookup = {term: index for index, term in enumerate(vocabulary)}

    @property
    def nnz(self) -> int:
        """Stored ``(term, document)`` pairs — the export's whole inverted index."""
        return int(self._data.shape[0])

    @property
    def nbytes(self) -> int:
        """Bytes the three CSR arrays plus the IDF vector occupy (the reportable footprint)."""
        arrays = (self._indptr, self._indices, self._data, self._idf)
        return int(sum(array.nbytes for array in arrays))

    def scores(self, text: str) -> Any:
        """Every document's exact global Okapi BM25 score for ``text`` — one sparse mat-vec.

        Terms the corpus never emitted are skipped (they have no stored pairs, so they contribute
        nothing to any document's score whatever their IDF). The query's term ids are visited in
        ascending order, which is what makes the float accumulation deterministic across runs and
        independent of ``PYTHONHASHSEED``.
        """
        term_ids = sorted(
            {
                term_id
                for term in set(_tokenize(text))
                if (term_id := self._lookup.get(term)) is not None
            }
        )
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term_id in term_ids:
            start, stop = int(self._indptr[term_id]), int(self._indptr[term_id + 1])
            out[self._indices[start:stop]] += self._idf[term_id] * self._data[start:stop]
        return out


class Bm25Backend:
    """Exact global Okapi BM25 over the whole corpus, as one sparse mat-vec per query.

    The scores and the order are the shipped :class:`~latence_retrieval.bm25.Bm25Rescorer`'s —
    see :class:`_Bm25TermMatrix` for how that is guaranteed rather than hoped. The rescorer
    instance is kept because it *is* the definition: this backend's statistics, constants and IDF
    are read off it.
    """

    def __init__(self, export: Export) -> None:
        self._export = export
        self._rescorer = Bm25Rescorer.from_export(str(export.root))
        self._matrix = _Bm25TermMatrix(export, self._rescorer)
        # Rank of each row's record id in ascending id order, so the rescorer's `(-score, id)`
        # tiebreak becomes an integer lexsort key instead of a per-query string sort.
        ids = [row.record_id for row in export.rows]
        by_id = np.fromiter(
            sorted(range(len(ids)), key=ids.__getitem__), dtype=np.int64, count=len(ids)
        )
        self._id_rank = np.empty(len(ids), dtype=np.int64)
        self._id_rank[by_id] = np.arange(len(ids))

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{BM25}``."""
        return frozenset({Capability.BM25})

    @property
    def matrix(self) -> _Bm25TermMatrix:
        """The first-stage weight matrix — exposed so a run can report its footprint."""
        return self._matrix

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Top-``top_k`` by exact global BM25, ties broken by ascending id (the rescorer's order).

        Documents holding no query term score exactly ``0`` and are dropped rather than returned
        as padding — the same set the postings prefilter used to hand the rescorer, and the same
        length the rescorer's truncated output had.
        """
        scores = self._matrix.scores(query.text or "")
        rows = self._export.rows
        return [
            Candidate(id=rows[i].record_id, score=float(scores[i]), text=rows[i].content)
            for i in self._top(scores, query.top_k)
        ]

    def _top(self, scores: Any, top_k: int) -> Any:
        """Row positions of the ``top_k`` highest scores, ordered by ``(-score, record_id)``.

        Partition first, then sort only the ``top_k``-and-ties tail: an exact top-k over 220,129
        scores without paying an O(N log N) sort per query. Taking every row whose score reaches
        the partition's own threshold is what keeps the tail exact when scores tie across the
        cut-off.
        """
        scoring = np.flatnonzero(scores > 0.0)
        if top_k < 1 or scoring.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        keep = min(top_k, int(scoring.shape[0]))
        if scoring.shape[0] > keep:
            values = scores[scoring]
            threshold = values[np.argpartition(-values, keep - 1)[:keep]].min()
            scoring = scoring[values >= threshold]
        order = np.lexsort((self._id_rank[scoring], -scores[scoring]))
        return scoring[order][:keep]


_BM25_TOKEN = re.compile(r"[^\W_]+")


def _tokenize(text: str) -> list[str]:
    """The index-side term unit (``latence_core.bm25.RegexWordTokenizer``), mirrored for lookup."""
    return _BM25_TOKEN.findall(text.lower())


class EdgeGraphSource:
    """A run-lifetime ``GraphSource`` over ``graph-edges.parquet`` (networkx, same semantics)."""

    def __init__(self, edges_parquet: Path) -> None:
        import networkx as nx

        table = pq.read_table(  # type: ignore[no-untyped-call]
            edges_parquet, columns=["source_node_id", "target_node_id"]
        )
        graph = nx.Graph()
        graph.add_edges_from(
            zip(
                (str(s) for s in table.column("source_node_id").to_pylist()),
                (str(t) for t in table.column("target_node_id").to_pylist()),
                strict=True,
            )
        )
        self._graph = graph

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """``node id → minimum hop distance`` within ``max_hops``, seeds excluded."""
        import networkx as nx

        seed_list = list(dict.fromkeys(seeds))
        seed_set = set(seed_list)
        best: dict[str, int] = {}
        for seed in seed_list:
            if seed not in self._graph:
                continue
            for node, distance in nx.single_source_shortest_path_length(
                self._graph, seed, cutoff=max_hops
            ).items():
                if distance == 0 or node in seed_set:
                    continue
                if node not in best or distance < best[node]:
                    best[node] = int(distance)
        return best


class ChunkGraphDocumentSource:
    """The node→document index over the export's own ``kg_node_ids`` column."""

    def __init__(self, export: Export) -> None:
        self._export = export

    def documents(self, node_ids: Collection[str]) -> list[GraphDocument]:
        """Documents mentioning any of ``node_ids``, each once, ascending id."""
        rows = self._export.rows
        hits = [rows[i] for i in self._export.rows_for_nodes(list(node_ids))]
        hits.sort(key=lambda row: row.record_id)
        return [
            GraphDocument(id=row.record_id, node_ids=row.node_ids, text=row.content)
            for row in hits
        ]


# -- the ladder ---------------------------------------------------------------------------------


@dataclass
class LegOutcome:
    """One query's result for one leg: the ranked ids plus the leg-specific observability."""

    ranked: list[str]
    latency_ms: float
    tokens_packed: int | None = None
    budget: int | None = None


@dataclass(frozen=True)
class SignalRanking:
    """One named ranked list on its way into the fusion.

    Carrying the *name* alongside the candidates is the whole point: fusion weights are declared
    per signal (``{"bm25": 0.5}``) but consumed by the fuser as a positional vector, and which
    positions exist varies per query (a graph leg with no alias seed produces no list). Pairing
    them here means the alignment is done once, next to the list, instead of at every call site.
    """

    name: str
    candidates: list[Candidate]


def fuse_signals(
    fuser: RrfFuser,
    rankings: Sequence[SignalRanking],
    weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """Fuse ``rankings`` into one list, aligning ``weights`` to them by signal name.

    ``weights=None`` is passed through to the fuser untouched — the equal-weight default, which is
    byte-for-byte the call the ladder made before weights were reachable. A named weight that
    matches no list this query produced is simply unused, and a list with no named weight keeps
    ``1.0``: a weight vector is a preference, never a filter, so a missing entry cannot silently
    delete a signal.
    """
    lists = [signal.candidates for signal in rankings]
    if not lists:
        return []
    vector = (
        None if weights is None else [float(weights.get(s.name, 1.0)) for s in rankings]
    )
    return fuser.fuse(lists, vector)


@dataclass
class Runner:
    """Everything one dataset's ladder needs, built once and reused across legs and queries."""

    export: Export
    dense: DenseBackend | None
    bm25: Bm25Backend | None
    graph: GraphRetriever | None
    kv_path: Path | None
    #: The ablation's second dense engine — the content-only sidecar over the same corpus rows.
    dense_content_only: DenseBackend | None = None
    hyperedge_chunks: dict[str, str] = field(default_factory=dict)
    fuser: RrfFuser = field(default_factory=RrfFuser)
    packer: Packer = field(default_factory=lambda: Packer({"budget": PACK_BUDGET}))
    seeder: AliasQuerySeeder | None = None
    #: Per-signal RRF weights, or ``None`` for the equal-weight default. ``None`` is not "all
    #: ones supplied explicitly": it is the absence of a weight argument, so the default ladder
    #: run makes exactly the call it always made.
    weights: dict[str, float] | None = None

    def _candidates_by_id(self, ids: Sequence[str]) -> list[Candidate]:
        rows = self.export.rows
        out: list[Candidate] = []
        for rank, chunk_id in enumerate(ids):
            position = self.export.index_of(chunk_id)
            if position is None:
                continue
            out.append(
                Candidate(
                    id=chunk_id,
                    score=1.0 / (1 + rank),
                    text=rows[position].content,
                )
            )
        return out

    def run(self, question: str, vector: Any, leg: str) -> LegOutcome:
        """Run the ladder up to and including ``leg`` for one query (or one ablation leg)."""
        started = time.perf_counter()
        rankings = self.signals(question, vector, leg)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        return self.finish(
            question, rankings, leg, weights=self.weights, retrieval_ms=retrieval_ms
        )

    def signals(self, question: str, vector: Any, leg: str) -> list[SignalRanking]:
        """The named, NON-EMPTY ranked lists ``leg`` produced for this query, in ladder order.

        Separated from the fusion so a caller can retrieve once and settle several fusion
        variants against the identical candidate lists — the only way an equal-weight and a
        weighted number can differ by the fusion and nothing else. Empty lists are dropped here
        rather than passed on, matching what the ladder always did: a signal that found nothing
        must not occupy a weight slot.
        """
        if leg in ABLATION_LEG_NAMES:
            return self._ablation_signals(vector, leg)
        out: list[SignalRanking] = []

        if self.dense is not None:
            out.append(
                SignalRanking(
                    "dense",
                    self.dense.search(
                        RetrievalQuery(
                            mode=Capability.DENSE,
                            vector=tuple(float(x) for x in vector),
                            top_k=POOL_K,
                        )
                    ),
                )
            )
        if leg != "dense" and self.bm25 is not None:
            out.append(
                SignalRanking(
                    "bm25",
                    self.bm25.search(
                        RetrievalQuery(mode=Capability.BM25, text=question, top_k=POOL_K)
                    ),
                )
            )
        if leg in ("kg", "hyperedges", "coe", "packer") and self.graph is not None:
            out.append(SignalRanking("kg", self._graph_leg(question)))
        if leg in ("hyperedges", "coe", "packer"):
            hyper, coe = self._hypergraph_legs(question, want_coe=leg in ("coe", "packer"))
            out.append(SignalRanking("hyperedges", hyper))
            out.append(SignalRanking("coe", coe))
        return [signal for signal in out if signal.candidates]

    def finish(
        self,
        question: str,
        rankings: Sequence[SignalRanking],
        leg: str,
        *,
        weights: Mapping[str, float] | None = None,
        retrieval_ms: float = 0.0,
    ) -> LegOutcome:
        """Fuse (and, for ``packer``, pack) already-retrieved ``rankings`` into a leg outcome.

        ``retrieval_ms`` is what producing ``rankings`` cost; the returned latency adds this
        variant's own fusion time to it, so two variants sharing one retrieval each report the
        full per-query cost rather than one of them looking free.
        """
        started = time.perf_counter()
        fused = fuse_signals(self.fuser, rankings, weights)
        tokens = budget = None
        if leg == "packer" and fused:
            fused, tokens = self.pack(question, fused)
            budget = PACK_BUDGET
        return LegOutcome(
            ranked=[c.id for c in fused],
            latency_ms=retrieval_ms + (time.perf_counter() - started) * 1000.0,
            tokens_packed=tokens,
            budget=budget,
        )

    def pack(self, question: str, fused: Sequence[Candidate]) -> tuple[list[Candidate], int]:
        """The ``packer`` rung: the token-budgeted subset of ``fused``, and its token count."""
        packed = self.packer.process(question, list(fused[:POOL_K]))
        return packed, sum(_heuristic_tokens(c.text) for c in packed)

    def _ablation_signals(self, vector: Any, leg: str) -> list[SignalRanking]:
        """One ablation leg's SINGLE signal, ranked against the alternative vector source.

        One list, so the fusion downstream is a no-op on the ordering (RRF over a single list is
        ``w / (k + rank)`` — strictly decreasing in rank, whatever ``w`` is), exactly like the
        ``dense`` rung it is the counterfactual of. That is what makes the two numbers
        subtractable. An absent backend yields NO list rather than falling back to the enriched
        vectors: a leg with no content-only sidecar must report nothing, never the very number it
        exists to contrast with.
        """
        backend = {"dense_content_only": self.dense_content_only}[leg]
        if backend is None:
            return []
        hits = backend.search(
            RetrievalQuery(
                mode=Capability.DENSE,
                vector=tuple(float(x) for x in vector),
                top_k=POOL_K,
            )
        )
        return [SignalRanking(leg, hits)] if hits else []

    def _graph_leg(self, question: str) -> list[Candidate]:
        if self.graph is None or self.seeder is None:
            return []
        seeds = self.seeder.seeds(question)
        if not seeds:
            return []
        return self.graph.search(
            RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=seeds, top_k=POOL_K)
        )

    def _hypergraph_legs(
        self, question: str, *, want_coe: bool
    ) -> tuple[list[Candidate], list[Candidate]]:
        if self.kv_path is None:
            return ([], [])
        with HyperedgeKV.open(str(self.kv_path)) as kv:
            signals = hypergraph_signals(kv, alias_surfaces=query_ngrams(question))
            hyper = self._chunks_from_hyperedges(
                sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
            )
            coe: list[Candidate] = []
            if want_coe and signals.rho:
                selection = coe_select(signals, budget=COE_BUDGET)
                coe = self._chunks_from_hyperedges(selection.selected)
        return (hyper, coe)

    def _chunks_from_hyperedges(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        rows = self.export.rows
        seen: set[str] = set()
        out: list[Candidate] = []
        for hyperedge_id in hyperedge_ids:
            chunk_id = self.hyperedge_chunks.get(hyperedge_id)
            if chunk_id is None or chunk_id in seen:
                continue
            position = self.export.index_of(chunk_id)
            if position is None:
                continue
            seen.add(chunk_id)
            out.append(
                Candidate(
                    id=chunk_id, score=1.0 / (1 + len(out)), text=rows[position].content
                )
            )
            if len(out) >= POOL_K:
                break
        return out


def _heuristic_tokens(text: str) -> int:
    """The packer's own dependency-free token estimate, reused for the packed-tokens report."""
    from latence_retrieval.pack import HeuristicTokenCounter

    return HeuristicTokenCounter().count(text)


def build_runner(
    export: Export,
    legs: Sequence[str],
    *,
    content_only_vectors: Any | None = None,
    weights: dict[str, float] | None = None,
) -> Runner:
    """Assemble only the components the requested ``legs`` actually need.

    ``content_only_vectors`` is the matrix :func:`~benchmark.s5.corpus.read_vector_sidecar`
    returned for this export — supplied by the caller rather than loaded here, because the caller
    is the one that already knows the corpus row ids the sidecar has to align with.

    ``weights`` are per-signal RRF weights for every leg this runner runs; the default ``None``
    is the equal-weight ladder. A runner is built per (dataset, run), but weights are fitted per
    (dataset, leg), so the tuned path leaves this ``None`` and passes its per-leg vector to
    :meth:`Runner.finish` instead of rebuilding a runner per leg.
    """
    root = export.root
    wanted = set(legs)
    dense = DenseBackend(export) if export.embeddings is not None else None
    dense_content_only = None
    if content_only_vectors is not None and wanted & set(ABLATION_LEG_NAMES):
        dense_content_only = DenseBackend(
            export, matrix=content_only_vectors, source=CONTENT_ONLY_VECTORS
        )
    bm25 = None
    if wanted & {"bm25", "kg", "hyperedges", "coe", "packer"} and (
        root / "bm25-stats.json"
    ).is_file():
        bm25 = Bm25Backend(export)
    graph = None
    seeder = None
    if wanted & {"kg", "hyperedges", "coe", "packer"} and (root / "graph-edges.parquet").is_file():
        graph = GraphRetriever(
            graph_source=EdgeGraphSource(root / "graph-edges.parquet"),
            document_source=ChunkGraphDocumentSource(export),
        )
        seeder = AliasQuerySeeder(node_alias_map(root / "graph-nodes.parquet"))
    kv_path = None
    hyperedges: dict[str, str] = {}
    if wanted & {"hyperedges", "coe", "packer"} and (root / "graph-hyperedges.kv").is_file():
        kv_path = root / "graph-hyperedges.kv"
        hyperedges = hyperedge_chunk_map(root / "graph-hyperedges.parquet")
    return Runner(
        export=export,
        dense=dense,
        bm25=bm25,
        graph=graph,
        kv_path=kv_path,
        dense_content_only=dense_content_only,
        hyperedge_chunks=hyperedges,
        seeder=seeder,
        weights=weights,
    )


def missing_inputs(export_dir: Path, leg: str) -> list[str]:
    """Artifacts ``leg`` needs that ``export_dir`` does not have — a refusal, never a fake run."""
    if leg in ABLATION_LEG_NAMES:
        # Not cumulative: an ablation leg is one rung re-run against one other input, so it needs
        # the corpus table (rows and ids) and its own vector file, nothing from the rungs above.
        return [
            f
            for f in ("corpus.parquet", CONTENT_ONLY_VECTORS)
            if not (export_dir / f).is_file()
        ]
    needs: dict[str, tuple[str, ...]] = {
        "dense": ("corpus.parquet",),
        "bm25": ("bm25-stats.json", "bm25-postings.parquet"),
        "kg": ("graph-edges.parquet", "graph-nodes.parquet"),
        "hyperedges": ("graph-hyperedges.kv", "graph-hyperedges.parquet"),
        "coe": ("graph-hyperedges.kv",),
        "packer": (),
    }
    cumulative: list[str] = []
    for name in LEG_NAMES:
        cumulative.extend(needs[name])
        if name == leg:
            break
    return [f for f in dict.fromkeys(cumulative) if not (export_dir / f).is_file()]


def has_embeddings(export_dir: Path) -> bool:
    """Whether the export's corpus table carries the dense ``embedding`` column."""
    return "embedding" in pq.read_schema(export_dir / "corpus.parquet").names  # type: ignore[no-untyped-call]


__all__ = [
    "ABLATION_BASELINE",
    "ABLATION_LEG_NAMES",
    "ALL_LEG_NAMES",
    "COE_BUDGET",
    "FUSION_SIGNALS",
    "LEG_NAMES",
    "PACK_BUDGET",
    "POOL_K",
    "Bm25Backend",
    "ChunkGraphDocumentSource",
    "DenseBackend",
    "EdgeGraphSource",
    "LegOutcome",
    "Runner",
    "SignalRanking",
    "build_runner",
    "fuse_signals",
    "has_embeddings",
    "leg_signals",
    "missing_inputs",
    "query_ngrams",
    "tunable_signals",
    "vector_source",
]

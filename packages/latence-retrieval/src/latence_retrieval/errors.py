"""Typed errors the query-time seam raises — mirroring what a real store rejects.

A faithful backend fake (:mod:`latence_retrieval.testing`) must reject *exactly* what a real vector
store rejects, so those rejections have named types a test can assert on (anti-false-green): asking
for a capability the store did not declare, or handing a dense query a vector of the wrong width. A
real Qdrant/Weaviate/Elastic call raises on both; the fake raises these so a test proves the
behaviour, not a flag.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for every error the query-time tooling raises."""


class OrchestrationError(RetrievalError):
    """The Level-2 orchestrator could not compose a runnable pipeline for the query (ADR-0052).

    Raised by :func:`~latence_retrieval.orchestrator.retrieve` **before any backend fetch**, when
    the query bundle and the backend's declared capabilities have no runnable intersection:

    * **No fetch leg** — the backend declares first-stage modes, but the bundle carries no matching
      query signal for *any* of them (a dense-only store with no query ``vector``, a sparse-only
      store with no ``sparse_terms``), so there is nothing to search. The orchestrator refuses to
      return an empty result silently — a caller who ran query-signal-gen for the wrong modality
      learns it loudly.
    * **Unsupported filter** — the bundle carries a metadata ``filter`` but the backend does not
      declare :class:`~latence_retrieval.contracts.Capability.FILTER`. The orchestrator will **not**
      silently drop the predicate and return unfiltered hits the caller never asked for
      (anti-false-green); dropping a *fetch modality* is graceful, dropping a *filter* changes the
      result set, so it fails loudly instead.

    This is the orchestrator's own *planning* guard, distinct from a fault the backend raises
    *during* a leg it accepted (:class:`BackendCapabilityError` / :class:`BackendDimensionError` /
    :class:`BackendSearchError`), which propagate unchanged.
    """


class BackendCapabilityError(RetrievalError):
    """A search asked for a capability (query mode or filter) the backend did not declare.

    The real-store analogue: a store rejects a hybrid/sparse/filtered query it was not configured
    for. The orchestrator's job is to *never* send such a query (it composes only declared legs,
    ADR-0052); this error is the backend's own guard so a mis-wired call fails loudly rather than
    silently returning wrong results.
    """


class BackendDimensionError(RetrievalError):
    """A dense/multivector query's vector width did not match the backend's configured dimension.

    The real-store analogue: every vector index is created at a fixed dimension and rejects a query
    vector of any other width. The fake enforces the same invariant so a dimension bug surfaces in
    an offline test instead of only against a live index.
    """


class BackendSearchError(RetrievalError):
    """A backend's first-stage search failed at runtime inside the customer's store.

    The real-store analogue: a ``qdrant_client`` call raises mid-request — the collection does not
    exist, the connection drops, gRPC/HTTP faults, or the server rejects the query. A real
    :class:`~latence_retrieval.backend.RetrievalBackend` maps that native fault to this typed error
    carrying **no** raw store text (no leak of a URL, api-key, payload, or driver stack), so a
    caller catches one named type instead of an opaque third-party exception — and a test asserts
    the behaviour, not the underlying client's exception class.

    It is distinct from :class:`BackendCapabilityError` / :class:`BackendDimensionError`, which the
    adapter raises **before** ever touching the store (a query the store was never asked to run);
    this type is reserved for a fault *during* the store call.
    """


class RerankModelError(RetrievalError):
    """A cross-encoder model failed at runtime (a native transformers/torch fault during scoring).

    The real-model analogue: a ``sentence-transformers`` ``CrossEncoder.predict`` raises inside its
    tokenizer/forward. The real adapter maps that to this typed error carrying **no** raw model text
    (no leak), so a caller catches one named type instead of an opaque third-party exception — and a
    test asserts the behaviour, not the underlying library's exception class.
    """


class QuerySignalError(RetrievalError):
    """The injected query-signal generator faulted, or produced an unusable signal bundle.

    The query-side analogue of :class:`RerankModelError`. The MCP server accepts an
    operator-injected ``query_signal_gen`` (``text -> {vector?, sparse_terms?}``) — a plain callable
    wrapping the
    caller's own dense/sparse query encoder — and runs it on the *raw agent query text*. A real
    tokenizer/encoder raises arbitrary native faults on some inputs (a torch/OOM error, an OSError,
    a tokenizer failure), and a mis-wired generator can emit a signal key
    :class:`~latence_retrieval.orchestrator.QueryBundle` rejects (``extra="forbid"``). Neither is a
    :class:`RetrievalError` on its own, so the server maps both to this typed error carrying **no**
    raw generator text (no leak of a model path, payload, or driver stack) — one named type an agent
    (and a test) sees as a leak-free ``BACKEND_ERROR`` envelope instead of an uncaught exception
    crossing the tool-call boundary (ADR-0052, the no-leak-into-responses invariant).
    """


class MaxSimError(RetrievalError):
    """A MaxSim rerank could not score a candidate against its sidecar multi-vectors (ADR-0050).

    The experimental late-interaction rerank path scores each candidate's *raw multi-vectors* —
    the ``multivectors.parquet`` sidecar the caller loaded onto ``candidate.metadata`` — against the
    query's multi-vector. This typed error is raised when that per-candidate input is absent or
    malformed rather than silently fabricating a score (ADR-0050 anti-false-green): a candidate
    carrying no multi-vector under the configured metadata key (the caller did not load the
    sidecar), or a document token vector whose width does not match the query's token width (the two
    were encoded in different spaces — a real MaxSim kernel cannot dot them either). It is a
    caller/data error surfaced loudly at query time, distinct from the model-runtime
    :class:`RerankModelError`.
    """


class GraphSourceError(RetrievalError):
    """A multi-hop traversal could not query the graph source at runtime (ADR-0051).

    The multi-hop ``expand`` path (:class:`~latence_retrieval.multihop.MultiHopExpander`) traverses
    the graph the pipeline emitted (``graph-edges.parquet``) through a
    :class:`~latence_retrieval.multihop.GraphSource` — an embedded DuckDB engine over that parquet
    (the zero-ops default), a networkx reference, or an optional Neo4j deployment. This typed error
    is raised when the underlying engine faults *during* a traversal — the parquet path does not
    exist or is unreadable, a DuckDB/Neo4j driver call raises, the graph store is unreachable — and
    the adapter maps that native fault to one named type carrying **no** raw engine text (no leak of
    a path, URI, credential, or driver stack), so a caller catches one type instead of an opaque
    third-party exception, and a test asserts the behaviour rather than the underlying library's
    exception class. A *bad config* (a non-existent source name, a negative hop bound) is a caller
    error and stays a :class:`ValueError`; this type is reserved for a runtime traversal fault.
    """


class PackTokenizerError(RetrievalError):
    """A real token counter failed at runtime (a native tokenizer fault during ``encode``).

    The real-tokenizer analogue: ``tiktoken``'s ``Encoding.encode`` raises inside its BPE. The real
    :class:`~latence_retrieval.pack.TiktokenCounter` adapter maps that to this typed error carrying
    **no** raw text (no leak), so ``pack`` fails with one named type instead of an opaque
    third-party exception — the same discipline the ``rerank`` seam uses. (A *bad encoding name* is
    a caller config error and stays a :class:`ValueError`; this type is reserved for a runtime
    fault.)
    """


class CandidateProcessorError(RetrievalError):
    """A caller-configured candidate processor rejected a candidate's malformed metadata at query
    time (ADR-0052, the MCP no-leak-into-responses invariant).

    The processor-chain analogue of :class:`QuerySignalError`. The MCP server runs the operator's
    configured :class:`~latence_retrieval.processor.CandidateProcessor` chain
    (``expand`` / ``multihop_expand`` / ``rerank`` / ``pack``) over the *caller-supplied* candidate
    list inside the guarded dispatch region. The graph-augmenting processors read denormalised KG
    columns off each candidate's metadata and — by their deliberate anti-false-green contract — fail
    **loudly** on a malformed column (a scalar/JSON-string where a list of node-id strings is
    required, a non-string element): :class:`~latence_retrieval.expand.Expander` and
    :class:`~latence_retrieval.multihop.MultiHopExpander` raise a plain :class:`TypeError` /
    :class:`ValueError` whose message embeds the raw offending value and candidate id.

    Neither a plain ``TypeError`` nor ``ValueError`` is a :class:`RetrievalError` on its own, so at
    the MCP tool-call boundary the server maps it to this typed error carrying **no** raw candidate
    text (no leak of store metadata, a payload, or a document id) — one named type an agent (and a
    test) sees as a leak-free ``BACKEND_ERROR`` envelope instead of an uncaught exception crossing
    the tool-call boundary. Direct Level-1 callers of the processors still get the loud, descriptive
    ``TypeError`` / ``ValueError`` (this containment lives only at the MCP seam, ADR-0052); the loud
    processor contract is intentional there and unchanged.
    """

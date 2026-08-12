"""The reference production :class:`RetrievalBackend`: :class:`QdrantBackend` (ADR-0048/0052, #173).

This is the *real* second adapter of the bring-your-own-engine seam — the one an adopter copies for
their own store. The offline :class:`~latence_retrieval.testing.FakeBackend` is the seam's faithful
fake; :class:`QdrantBackend` is the concrete example over a store an enterprise actually runs
(Qdrant), so the seam has the two implementations the codebase-design bar requires: a faithful fake
for tests and a real store for prod.

The whole adapter is one small interface — :attr:`~QdrantBackend.capabilities` and
:meth:`~QdrantBackend.search` — over a lot of hidden work: it holds no index and owns no store
(ADR-0048), it validates a query against the store's declared capabilities and its fixed dimension
*before* touching the wire (exactly as :class:`~latence_retrieval.testing.FakeBackend` does), then
translates one :class:`~latence_retrieval.contracts.RetrievalQuery` into a single ``query_points``
call, and maps the store's ``ScoredPoint``\\ s back into
:class:`~latence_retrieval.contracts.Candidate`\\ s. The ``qdrant-client`` import is deferred to
first :meth:`search` (behind the ``[backend-qdrant]`` extra, ADR-0016/0049), so constructing it —
which an operator or the Level-2 orchestrator does up front — never needs the heavy client and never
opens a connection.

Scope (honest capability declaration — a backend must not advertise a mode it cannot faithfully
serve, ADR-0052): this reference adapter translates ``{dense, sparse, filter}`` — the AC's focus and
the modes that map cleanly onto a single ``query_points`` call. ``bm25`` and ``multivector`` are
declared *unsupported by this adapter* (a clear construction error) and left as seam-compatible
follow-ons: the ``RetrievalBackend`` Protocol is the contract, so a later Databricks/Azure/Elastic
adapter — or a richer Qdrant one — adds them without changing the seam.

Licensing (ADR-0012, verified 2026-07-16 — code only; a vector store ships **no** model weights):
    ``qdrant-client`` is **Apache-2.0** (code:
    <https://github.com/qdrant/qdrant-client/blob/master/LICENSE>, PyPI metadata
    ``License: Apache-2.0``). The Qdrant engine it talks to is **Apache-2.0**
    (<https://github.com/qdrant/qdrant/blob/master/LICENSE>). There is no model weight to attest —
    a store is data infrastructure, not a checkpoint. Nothing is downloaded and no connection opens
    unless an operator constructs this adapter and calls :meth:`search`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_retrieval.contracts import Candidate, Capability
from latence_retrieval.errors import (
    BackendCapabilityError,
    BackendDimensionError,
    BackendSearchError,
)

if TYPE_CHECKING:
    from latence_retrieval.contracts import RetrievalQuery

# The modes this reference adapter faithfully translates onto a single ``query_points`` call. FILTER
# is a predicate (carried by any mode), DENSE/SPARSE are query modes. BM25/MULTIVECTOR are out of
# scope for the reference adapter (declared explicitly, never silently) — see the module docstring.
_SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.DENSE, Capability.SPARSE, Capability.FILTER}
)
_VECTOR_MODES: frozenset[Capability] = frozenset({Capability.DENSE})
# Connection knobs handed straight to ``QdrantClient(**...)``; everything else in config is adapter
# state (collection, capabilities, dimension, vector names, text_key). Kept explicit so a stray/typo
# config key is a loud error, never a silently-ignored setting or a kwarg the client would reject.
_CONNECTION_KEYS: frozenset[str] = frozenset(
    {"url", "host", "port", "grpc_port", "prefer_grpc", "https", "api_key", "timeout", "location"}
)
_ADAPTER_KEYS: frozenset[str] = frozenset(
    {
        "collection",
        "capabilities",
        "dimension",
        "vector_name",
        "sparse_vector_name",
        "text_key",
        "id_key",
        "search_params",
    }
)
#: The keys ``search_params`` may carry — Qdrant's per-query ANN accuracy knobs, and exactly the
#: two a serving deployment tunes against a brute-force reference: ``hnsw_ef`` (the HNSW beam
#: width; higher = more exact, slower) and ``exact`` (bypass the ANN index entirely — the
#: rescoring option that turns the store into the brute-force reference itself).
_SEARCH_PARAM_KEYS: frozenset[str] = frozenset({"hnsw_ef", "exact"})
#: The payload field the chunk text lives under, by default. This is the framework's OWN Export
#: column name (``content`` — ``latence_core.stages.export``), which ``latence-sink-qdrant``
#: (#183) writes verbatim into the payload of every point it loads. The default therefore makes
#: the two shipped halves — the sink that writes a collection and this backend that reads it —
#: agree with no configuration at all. Point this adapter at a collection somebody else built and
#: set ``text_key`` to whatever key *they* used; a key the payload does not carry is a loud
#: :class:`~latence_retrieval.errors.BackendSearchError`, never a silent empty ``Candidate.text``.
DEFAULT_TEXT_KEY = "content"
#: The payload field the corpus record identity lives under, by default. Qdrant point ids are
#: unsigned integers or UUIDs, so ``latence-sink-qdrant`` derives a UUIDv5 point id from the
#: Export's content-addressed ``record_id`` — and writes the ``record_id`` itself into the payload
#: precisely so the identity survives the trip. Every downstream consumer of this seam
#: (deduplicating fusers, gold scoring, :class:`~latence_retrieval.rescue.GraphRescueRetrieval`'s
#: append-and-dedupe, hyperedge→chunk resolution) keys on :attr:`Candidate.id` being the *corpus*
#: id, so the default restores it: a payload carrying this key yields ``Candidate.id`` from it
#: (and drops it from ``metadata`` — no duplication, mirroring ``text_key``), while a payload
#: without it — a collection somebody else built, keyed by its own point ids — gracefully keeps
#: the stringified point id, which is the correct identity for that collection. Present-but-
#: malformed is a loud :class:`~latence_retrieval.errors.BackendSearchError`; set ``id_key`` to
#: another payload field, or to ``None`` to always use the store's point id.
DEFAULT_ID_KEY = "record_id"


class QdrantBackend:
    """A capability-declaring :class:`~latence_retrieval.backend.RetrievalBackend` over Qdrant.

    Constructed from a config block (the store's honest self-description) and, optionally, an
    already-built client (dependency injection — a pooled/shared ``QdrantClient``, or a faithful
    fake in tests); with no client supplied it builds a real ``QdrantClient`` lazily from the
    config on first :meth:`search`. It declares only what it can serve and enforces the seam's
    invariants itself, before the wire — so a mis-composed query fails loudly and offline, never
    as a silent empty result from the store.

    Config (``{"provider"...}``-free — this is a bring-your-own backend the operator constructs, not
    a registry-resolved processor):
        collection: str — the Qdrant collection to search. **Required.**
        capabilities: list[str] — the modes/predicate this store declares, a subset of
            ``{"dense", "sparse", "filter"}``. **Required, non-empty.** Declaring ``"bm25"`` or
            ``"multivector"`` is a construction error (this reference adapter does not serve them).
        dimension: int — the fixed width of the dense vector index. **Required iff** ``"dense"`` is
            declared (a dense index cannot exist without one); rejected as a wrong-width vector at
            query time, exactly as the store's index would.
        vector_name: str | None — the named dense vector to query (``using=``). ``None`` ⇒ the
            collection's single unnamed vector.
        sparse_vector_name: str | None — the named sparse vector to query for a sparse mode.
        text_key: str — the payload field holding the chunk text (default
            :data:`DEFAULT_TEXT_KEY` = ``"content"``, the Export column name that
            ``latence-sink-qdrant`` writes, so a collection this framework loaded needs no
            override); it becomes :attr:`Candidate.text` and is dropped from
            :attr:`Candidate.metadata` (no duplication). A non-empty payload missing this key is a
            :class:`~latence_retrieval.errors.BackendSearchError`, never a silent empty text.
        id_key: str | None — the payload field holding the corpus record identity (default
            :data:`DEFAULT_ID_KEY` = ``"record_id"``, the column ``latence-sink-qdrant`` writes
            into every point precisely so the content-addressed id survives the UUID point-id
            derivation). A payload carrying it yields :attr:`Candidate.id` from it (dropped from
            ``metadata``); a payload without it keeps the stringified point id — the correct
            identity for a collection somebody else keyed — while present-but-malformed is a loud
            :class:`~latence_retrieval.errors.BackendSearchError`. ``None`` always uses the
            store's point id.
        search_params: dict | None — per-query ANN accuracy knobs threaded to ``query_points`` as
            ``models.SearchParams``: ``hnsw_ef`` (int ≥ 1, the HNSW search beam width) and/or
            ``exact`` (bool — bypass the ANN index and score exactly, the rescoring option).
            Default ``None`` (the collection's own defaults). These are the two knobs a serving
            deployment tunes when proving recall parity against a brute-force reference.
        url / host / port / grpc_port / prefer_grpc / https / api_key / timeout / location —
            connection kwargs threaded verbatim to ``QdrantClient`` when it is built lazily. Ignored
            when a ``client`` is injected.

    A dense/multivector query vector of the wrong width raises
    :class:`~latence_retrieval.errors.BackendDimensionError`; a mode or filter outside
    :attr:`capabilities` raises :class:`~latence_retrieval.errors.BackendCapabilityError`; a native
    store fault during the call raises :class:`~latence_retrieval.errors.BackendSearchError`
    carrying no raw store text.
    """

    def __init__(self, config: dict[str, Any] | None = None, *, client: Any | None = None) -> None:
        cfg = dict(config or {})
        unknown = set(cfg) - _ADAPTER_KEYS - _CONNECTION_KEYS
        if unknown:
            msg = f"QdrantBackend: unknown config key(s) {sorted(unknown)}."
            raise ValueError(msg)

        collection = cfg.get("collection")
        if not isinstance(collection, str) or not collection:
            msg = "QdrantBackend: 'collection' (a non-empty string) is required."
            raise ValueError(msg)
        self._collection: str = collection

        self._capabilities = self._parse_capabilities(cfg.get("capabilities"))

        dimension = cfg.get("dimension")
        declares_dense = bool(self._capabilities & _VECTOR_MODES)
        if declares_dense and dimension is None:
            msg = "QdrantBackend declaring the 'dense' capability requires a 'dimension'."
            raise ValueError(msg)
        if dimension is not None and (
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
        ):
            msg = f"QdrantBackend 'dimension' must be an int >= 1, got {dimension!r}."
            raise ValueError(msg)
        self._dimension: int | None = dimension

        self._vector_name = self._optional_str(cfg.get("vector_name"), "vector_name")
        self._sparse_vector_name = self._optional_str(
            cfg.get("sparse_vector_name"), "sparse_vector_name"
        )
        text_key = cfg.get("text_key", DEFAULT_TEXT_KEY)
        if not isinstance(text_key, str) or not text_key:
            msg = f"QdrantBackend 'text_key' must be a non-empty string, got {text_key!r}."
            raise ValueError(msg)
        self._text_key = text_key

        id_key = cfg.get("id_key", DEFAULT_ID_KEY)
        if id_key is not None and (not isinstance(id_key, str) or not id_key):
            msg = (
                f"QdrantBackend 'id_key' must be a non-empty string or None, got {id_key!r}."
            )
            raise ValueError(msg)
        self._id_key: str | None = id_key

        self._search_params = self._parse_search_params(cfg.get("search_params"))

        self._client = client
        self._client_kwargs = {k: cfg[k] for k in _CONNECTION_KEYS if k in cfg}
        self._models: Any = None

    @staticmethod
    def _parse_capabilities(raw: object) -> frozenset[Capability]:
        if not isinstance(raw, (list, tuple, set, frozenset)) or not raw:
            msg = "QdrantBackend: 'capabilities' (a non-empty list of mode names) is required."
            raise ValueError(msg)
        parsed: set[Capability] = set()
        for name in raw:
            try:
                cap = Capability(name)
            except ValueError:
                allowed = sorted(c.value for c in Capability)
                msg = f"QdrantBackend: unknown capability {name!r}; valid names: {allowed}."
                raise ValueError(msg) from None
            if cap not in _SUPPORTED:
                supported = sorted(c.value for c in _SUPPORTED)
                msg = (
                    f"QdrantBackend does not serve capability {cap.value!r}; this reference "
                    f"adapter supports {supported} (bm25/multivector are follow-ons; 'graph' is "
                    "served by GraphRetriever, composed alongside this store with "
                    "CompositeBackend)."
                )
                raise ValueError(msg)
            parsed.add(cap)
        return frozenset(parsed)

    @staticmethod
    def _parse_search_params(raw: object) -> dict[str, Any] | None:
        """Validate the ``search_params`` block at construction — offline, before any wire.

        A typo'd knob or a wrong type here would otherwise surface as an opaque server rejection
        (or, worse, a silently-ignored setting) three components deep into a serving run.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict) or not raw:
            msg = (
                "QdrantBackend 'search_params' must be a non-empty dict with keys from "
                f"{sorted(_SEARCH_PARAM_KEYS)} or omitted, got {raw!r}."
            )
            raise ValueError(msg)
        unknown = set(raw) - _SEARCH_PARAM_KEYS
        if unknown:
            msg = (
                f"QdrantBackend 'search_params' has unknown key(s) {sorted(unknown)}; "
                f"valid keys: {sorted(_SEARCH_PARAM_KEYS)}."
            )
            raise ValueError(msg)
        hnsw_ef = raw.get("hnsw_ef")
        if hnsw_ef is not None and (
            not isinstance(hnsw_ef, int) or isinstance(hnsw_ef, bool) or hnsw_ef < 1
        ):
            msg = f"QdrantBackend 'search_params.hnsw_ef' must be an int >= 1, got {hnsw_ef!r}."
            raise ValueError(msg)
        exact = raw.get("exact")
        if exact is not None and not isinstance(exact, bool):
            msg = f"QdrantBackend 'search_params.exact' must be a bool, got {exact!r}."
            raise ValueError(msg)
        return dict(raw)

    @staticmethod
    def _optional_str(value: object, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            msg = f"QdrantBackend '{name}' must be a non-empty string or omitted, got {value!r}."
            raise ValueError(msg)
        return value

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The first-stage capabilities this Qdrant collection declares it can serve."""
        return self._capabilities

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Run one first-stage Qdrant ``query_points`` and return its hits as ``Candidate``\\ s.

        Guards first (offline, before the wire), mirroring the seam contract and the faithful fake:
        a ``mode`` outside :attr:`capabilities` or a ``filter`` when ``FILTER`` was not declared →
        :class:`~latence_retrieval.errors.BackendCapabilityError`; a dense query vector whose width
        is not the index ``dimension`` → :class:`~latence_retrieval.errors.BackendDimensionError`.
        Then it translates the query into a single ``query_points`` call, sorts nothing itself (the
        store returns descending-score hits, truncated to ``query.top_k``), and maps each
        ``ScoredPoint`` to a ``Candidate``. A native store fault becomes a
        :class:`~latence_retrieval.errors.BackendSearchError` with no raw store text.
        """
        if query.mode not in self._capabilities:
            msg = (
                f"backend does not declare capability {query.mode.value!r}; "
                f"declared: {sorted(c.value for c in self._capabilities)}."
            )
            raise BackendCapabilityError(msg)
        if query.filter is not None and Capability.FILTER not in self._capabilities:
            msg = "backend does not declare the 'filter' capability but a filter was supplied."
            raise BackendCapabilityError(msg)
        if query.mode in _VECTOR_MODES:
            width = len(query.vector or ())
            if width != self._dimension:
                msg = (
                    f"query vector has width {width} but the index dimension is {self._dimension}."
                )
                raise BackendDimensionError(msg)

        models = self._load_models()
        query_obj, using = self._build_query(query, models)
        query_filter = self._build_filter(query.filter, models)
        client = self._resolve_client()
        try:
            response = client.query_points(
                collection_name=self._collection,
                query=query_obj,
                using=using,
                query_filter=query_filter,
                limit=query.top_k,
                with_payload=True,
                search_params=self._build_search_params(models),
            )
            points = list(response.points)
        except (BackendCapabilityError, BackendDimensionError):  # pragma: no cover - never reached
            raise
        except Exception as exc:
            # A native qdrant fault (missing collection, transport error, server rejection) → typed
            # error with NO raw store text (no url/api-key/payload/driver-stack leak).
            msg = f"qdrant search failed on collection {self._collection!r}: {type(exc).__name__}."
            raise BackendSearchError(msg) from exc
        return [self._to_candidate(point) for point in points]

    def _build_query(self, query: RetrievalQuery, models: Any) -> tuple[Any, str | None]:
        """Translate the mode payload into a ``query_points`` query + its ``using`` vector name."""
        if query.mode is Capability.DENSE:
            # A raw dense vector; RetrievalQuery guarantees it is present for this mode.
            return list(query.vector or ()), self._vector_name
        # SPARSE: RetrievalQuery guarantees non-empty sparse_terms for this mode. The keys are the
        # integer token ids the SPLADE composition emits (ADR-0053 sparse_indices); a non-integer
        # key is a caller error (a real sparse index cannot be a term string).
        indices: list[int] = []
        values: list[float] = []
        for token, weight in (query.sparse_terms or {}).items():
            try:
                indices.append(int(token))
            except (TypeError, ValueError):
                msg = (
                    f"QdrantBackend sparse query term {token!r} is not an integer index; sparse "
                    "terms are {token_id: weight} (ADR-0053 sparse_indices)."
                )
                raise ValueError(msg) from None
            values.append(float(weight))
        return models.SparseVector(indices=indices, values=values), self._sparse_vector_name

    def _build_search_params(self, models: Any) -> Any | None:
        """The configured ANN knobs as a ``models.SearchParams``, or ``None`` for store defaults."""
        if self._search_params is None:
            return None
        return models.SearchParams(**self._search_params)

    @staticmethod
    def _build_filter(predicate: dict[str, Any] | None, models: Any) -> Any | None:
        """Translate the exact per-field equality predicate into a Qdrant ``Filter`` (``must`` AND).

        Mirrors the faithful fake's filter semantics exactly: every field must equal its value
        (logical AND), so a candidate lacking a filtered field is excluded — the ``must`` list of
        ``MatchValue`` conditions is Qdrant's expression of the same predicate.
        """
        if not predicate:
            return None
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in predicate.items()
        ]
        return models.Filter(must=conditions)

    def _to_candidate(self, point: Any) -> Candidate:
        """Map one Qdrant ``ScoredPoint`` (id/score/payload) to a frozen :class:`Candidate`.

        The payload's ``text_key`` becomes the candidate ``text`` and the payload's ``id_key`` —
        when the payload carries one — becomes the candidate ``id``; both are dropped from
        ``metadata`` (no duplication), and every other payload field is carried through as opaque
        provenance. Without an ``id_key`` value the point id (Qdrant allows ``int`` or UUID
        ``str``) is stringified to the seam's string ``id`` — see :data:`DEFAULT_ID_KEY` for why
        the graceful-absent / loud-malformed split is the honest one here.

        A **non-empty** payload that does not carry ``text_key`` is a configuration fault, not a
        result: silently returning ``text=""`` would hand the pipeline a full page of Candidates
        with no text at all — the reranker would score nothing, the Packer would pack nothing, and
        the run would report a green retrieval over a collection that carries the text perfectly
        well. So it raises a typed :class:`~latence_retrieval.errors.BackendSearchError` naming the
        key it looked for and the keys the point actually has (ADR-0034: no silent errors; payload
        *keys* are schema, never content). A point with no payload at all is a different thing —
        the store returned nothing to map — and stays an empty candidate.
        """
        payload = dict(point.payload or {})
        if payload and self._text_key not in payload:
            msg = (
                f"qdrant collection {self._collection!r} has no {self._text_key!r} field in its "
                f"point payload (keys present: {sorted(payload)}); set 'text_key' to the payload "
                f"field holding the chunk text. Collections loaded by latence-sink-qdrant use the "
                f"Export's own column name, {DEFAULT_TEXT_KEY!r} (the default)."
            )
            raise BackendSearchError(msg)
        text = payload.pop(self._text_key, "")
        return Candidate(
            id=self._candidate_id(point, payload),
            score=float(point.score),
            text=str(text) if text is not None else "",
            metadata=payload,
        )

    def _candidate_id(self, point: Any, payload: dict[str, Any]) -> str:
        """The seam identity for one point: the payload's ``id_key`` value, or the point id.

        Pops the consumed key from ``payload`` (the caller's metadata dict) so the identity is
        never duplicated into ``metadata`` — the same discipline ``text_key`` follows. Absent is
        graceful (a foreign collection's points ARE identified by their point ids); present-but-
        malformed is loud, because an id that stringifies a dict or an empty value would corrupt
        every downstream dedupe and join silently.
        """
        if self._id_key is None or self._id_key not in payload:
            return str(point.id)
        raw = payload.pop(self._id_key)
        if not isinstance(raw, str) or not raw:
            msg = (
                f"qdrant collection {self._collection!r} carries a non-string or empty "
                f"{self._id_key!r} payload field on point {point.id!s}; 'id_key' must name a "
                "payload field holding non-empty string record ids (or be None to use the "
                "store's point ids)."
            )
            raise BackendSearchError(msg)
        return raw

    def _load_models(self) -> Any:
        """Import ``qdrant_client.models`` lazily (cached) for the filter/sparse translation."""
        if self._models is None:
            self._models = self._import_qdrant()[1]
        return self._models

    def _resolve_client(self) -> Any:
        """Return the injected client, or build a real ``QdrantClient`` lazily from the config."""
        if self._client is None:
            self._client = self._import_qdrant()[0](**self._client_kwargs)
        return self._client

    @staticmethod
    def _import_qdrant() -> tuple[Any, Any]:
        """Defer the heavy import to first use; return ``(QdrantClient, models)``.

        ``models`` is a submodule, not a top-level attribute, so it is imported explicitly. A
        missing dependency surfaces as one clear, actionable install hint at first :meth:`search` —
        never at construction or module import.
        """
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'qdrant-client' package is required to run the QdrantBackend. Install it "
                "with: pip install 'latence-retrieval[backend-qdrant]'"
            )
            raise ImportError(msg) from exc
        return QdrantClient, models


__all__ = ["DEFAULT_ID_KEY", "DEFAULT_TEXT_KEY", "QdrantBackend"]

"""Endpoint-backed Embedder Provider — the Embedder performance path (S12, ADR-0007, ADR-0017).

The heavy-model Embedder Provider: it maps RAG-corpus text to vectors via any **OpenAI-compatible /
vLLM / Triton** ``/embeddings`` endpoint serving a real embedding model (e.g. multilingual-e5-large,
BGE-M3). It is the *same opt-in Embedder Capability* as the in-core deterministic
``embedding.hashing`` reference — a Pipeline swaps the toy hashing embedder for a real served model
with **one config change** (point ``base_url`` at the endpoint), no code edit (ADR-0007).

Fault tolerance is the ported endpoint machinery (:mod:`latence_core.endpoint`): a shared
reference-counted lease (co-located Stages pay one cold-start) and per-call
:func:`~latence_core.endpoint.call_with_retry` (backoff+jitter, 429/5xx retryable, terminal 4xx
fails fast). Retries absorbed land on ``endpoint_retries`` for the Quality Report. Texts are
embedded in input order (the Embedder contract) with a configurable batch size so a long corpus is
sent in bounded requests.

Licensing: framework code (Apache-2.0); bundles no weights (calls a user-supplied
endpoint). The ``openai`` client is Apache-2.0 (code). The client import is deferred to first use.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

from latence_core.endpoint import EndpointError, EndpointPool, RetryPolicy, call_with_retry
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_DEFAULT_MODEL = "multilingual-e5-large"
_DEFAULT_DIMENSION = 1024
_POOL = EndpointPool()


class EndpointEmbedder:
    """Embedder Provider backed by an OpenAI-compatible ``/embeddings`` endpoint.

    Config:
        dimension: int  — REQUIRED. The fixed embedding width the served model produces (e.g. 1024
            for multilingual-e5-large). Every returned vector is validated to this width so a
            downstream vector DB gets a uniform column (ADR-0017).
        model: str  — the served embedding model id (default ``multilingual-e5-large``).
        base_url: str | None  — the OpenAI-compatible endpoint (vLLM / Triton / hosted / local).
        api_key: str | None  — the key; falls back to ``$OPENAI_API_KEY``.
        batch_size: int  — texts per request (default 32), to bound each call over a long corpus.
        retry_attempts / retry_base_delay / retry_max_delay  — the ported retry policy knobs.

    The client is constructed lazily on first ``embed`` so building the Provider never needs the
    endpoint reachable.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): an OpenAI-compatible embeddings endpoint adapter.

        ``compute="either"``: a thin HTTP client that runs on the local CPU (no local CUDA
        required); the served embedding model lives behind the endpoint, so the device seam does
        not skip it on a CPU-only host. Real quality/throughput needs a live rig (#64);
        conformance verifies it against a stub. ``model_id=None``: this package ships no weights
        (the served model id, default ``multilingual-e5-large`` — MIT — is config-supplied, not a
        bundled checkpoint). License is framework code (Apache-2.0, verified; the
        ``openai`` ``openai`` client is Apache-2.0). ``deterministic=False`` (a served model is not
        guaranteed byte-identical); ``cost_per_1k=None`` (deployment/vendor-specific, not
        fabricated).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=64,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        dimension = int(config.get("dimension", _DEFAULT_DIMENSION))
        if dimension < 1:
            msg = f"EndpointEmbedder dimension must be >= 1, got {dimension}."
            raise ValueError(msg)
        self._dimension = dimension
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._batch_size = max(1, int(config.get("batch_size", 32)))
        self._policy = RetryPolicy(
            attempts=int(config.get("retry_attempts", 4)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
        )
        self._rng = random.Random(f"{self._base_url}|{self._model}")
        self._lease = _POOL.lease(f"{self._base_url}|{self._model}")
        self._client: Any = None
        self.endpoint_retries = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'openai' package is required to run EndpointEmbedder. "
                "Install it with: pip install latence-embedder-endpoint"
            )
            raise ImportError(msg) from exc
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    # -- embedding ----------------------------------------------------------

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        batch: list[str] = []
        client: Any = None
        acquired = False
        try:
            for text in texts:
                batch.append(text)
                if len(batch) >= self._batch_size:
                    if client is None:
                        client = self._get_client()
                        self._lease.acquire()
                        acquired = True
                    yield from self._embed_batch(client, batch)
                    batch = []
            if batch:
                if client is None:
                    client = self._get_client()
                    self._lease.acquire()
                    acquired = True
                yield from self._embed_batch(client, batch)
        finally:
            if acquired:
                self._lease.release()

    def _embed_batch(self, client: Any, batch: list[str]) -> list[list[float]]:
        def do_call() -> list[list[float]]:
            return self._one_call(client, batch)

        result = call_with_retry(
            do_call,
            self._policy,
            rng=self._rng.random,
            sleep=lambda _d: None,
            on_retry=self._count_retry,
        )
        vectors = result if isinstance(result, list) else []
        if len(vectors) != len(batch):
            msg = (
                f"Embedding endpoint returned {len(vectors)} vectors for a batch of "
                f"{len(batch)} texts."
            )
            raise EndpointError(msg)
        for vector in vectors:
            if len(vector) != self._dimension:
                msg = (
                    f"Embedding endpoint returned a {len(vector)}-dim vector; "
                    f"configured dimension is {self._dimension}."
                )
                raise EndpointError(msg)
        return vectors

    def _count_retry(self) -> None:
        self.endpoint_retries += 1

    def _one_call(self, client: Any, batch: list[str]) -> list[list[float]]:
        try:
            response = client.embeddings.create(model=self._model, input=batch)
        except Exception as exc:  # noqa: BLE001 - normalized into a classed EndpointError below
            raise _classify_exception(exc) from exc
        # OpenAI-compatible shape: response.data[i].embedding, ordered by input index.
        ordered = sorted(response.data, key=lambda d: d.index)
        return [[float(x) for x in d.embedding] for d in ordered]


def _classify_exception(exc: Exception) -> EndpointError:
    """Normalize any client exception into a classed :class:`EndpointError` for the ported retry."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return EndpointError(str(exc), status=status)
    return EndpointError(str(exc))

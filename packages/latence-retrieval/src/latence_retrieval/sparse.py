# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
#
# DELIBERATE, BOUNDED EXCEPTION -- not a header nobody updated at the relicence. This
# repository's outbound licence is PolyForm-Noncommercial-1.0.0 (root LICENSE); this module is
# one of seven the copyright holder keeps under Apache-2.0 ON PURPOSE, so the query-time
# retrieval primitives stay permissively reusable -- commercially included -- by anyone who
# wants only them. The grant is his to make and safe to make: sole copyright holder, no
# third-party and no copyleft code in this file. The exact set is authoritative in LICENSE,
# section "SCOPE: THE RETAINED APACHE-2.0 MODULES", and pinned by
# packages/latence-core/tests/test_publish_readiness.py, which fails the build if an eighth
# file acquires the grant or one of the seven loses it.
"""Query-side sparse generation: :class:`SpladeQueryEncoder` (ADR-0053, #170).

The query-time twin of the index-time SPLADE signal generator. ADR-0053 is explicit that its sparse
vector is *one composition applied to two clocks*: at index time the pipeline's
:class:`~latence_splade.provider.SpladeCompositionEmbedder` composes each chunk's vector, and at
query time the **same** composition must run over the query text so the query and corpus land in the
same sparse space. This encoder is that query-time entry point — and it does **not** re-implement
the composition (which would drift from the index-time signal): it *reuses*
:class:`~latence_splade.provider.SpladeCompositionEmbedder` directly, so the query vector is
byte-for-byte the composition the corpus was built with (the AC: "mirrors the index-time SPLADE
composition for the query text").

It is stateless (ADR-0048): a pure ``query text → sparse vector`` transform that holds no index and
fetches nothing. The caller hands the returned :class:`~latence_core.capability.SparseVector` to its
own store as the sparse leg of a hybrid query — this encoder never runs the search itself.

**The model is the seam, and it is reused too.** The composition depends on a
:class:`~latence_splade.model.SpladeModel` (the two-adapter seam from #162: the real
``transformers`` checkpoint vs. an offline faithful fake). This encoder **accepts** one by injection
(exactly as the index-time embedder does) so a test drives a fake and production drives the real,
license-verified checkpoint — no seam with a single implementation, and no model pinned until its
weights+code license is cleared (ADR-0012/0053).

Packaging: ``latence-splade`` is an **optional** dependency behind this package's ``[sparse]`` extra
(it pulls ``latence-core`` and, only when a real model is wired, the ``transformers`` stack). The
import is deferred to first use so the pure-Python retrieval core — and registry discovery — never
needs it; an adopter who wants query-side sparse installs ``latence-retrieval[sparse]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from latence_core.capability import SparseVector
    from latence_splade.model import SpladeModel
    from latence_splade.provider import SpladeCompositionEmbedder


@runtime_checkable
class SparseQueryEncoder(Protocol):
    """A query-time sparse-signal generator: ``query text → sparse vector`` (ADR-0053).

    The whole seam a caller drives for query-side sparse — deliberately one method. Given the query
    text (and, optionally, query entities and metadata to compose in exactly as the index-time
    signal does), it returns the query's :class:`~latence_core.capability.SparseVector` in the same
    term-index space as the emitted corpus, ready to hand to the store's sparse index. It holds no
    state, so identical inputs yield the same vector.
    """

    def encode(
        self,
        query: str,
        *,
        entities: Sequence[tuple[str, float]] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> SparseVector:
        """Return the query's composed sparse vector (see :class:`SpladeQueryEncoder`)."""
        ...


class SpladeQueryEncoder:
    """Compose a query's sparse vector by **reusing** the index-time SPLADE composition (ADR-0053).

    Wraps :class:`~latence_splade.provider.SpladeCompositionEmbedder` — the *same* class the
    pipeline ran at index time — so the query vector is the identical composition (model leg ⊕
    confidence-gated entity terms ⊕ selected metadata terms). :meth:`encode` composes exactly one
    query and returns its :class:`~latence_core.capability.SparseVector`.

    Config (passed straight through to the reused embedder, so query- and index-time are configured
    identically — user story 36's one-line swap):
        model: str — the multilingual SPLADE checkpoint id. **Required, no default** (ADR-0012/0053)
            *only when a real model is built*; an injected ``model`` (tests / query-time reuse of an
            already-loaded index-time embedder) needs no id.
        mode / confidence_threshold / entity_boost / metadata_boost / model_weight /
            metadata_fields / max_length — the ADR-0053 composition knobs, identical to the
            index-time embedder's.
        redact: bool — default ``True``. At query time this means an injected entity is only
            composed in if its surface actually occurs in the query text (query entities do), so it
            is a no-op for ordinary querying; set ``False`` to inject *expansion* terms that are not
            present in the query string.

    The model (and the ``latence_splade`` import) is resolved lazily on first :meth:`encode`, so
    constructing the encoder — which the registry does — never imports ``latence_splade`` or
    downloads weights.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, *, model: SpladeModel | None = None
    ) -> None:
        self._config: dict[str, Any] = dict(config or {})
        # Dependency injection: an explicit model (tests / reuse of a loaded index-time embedder)
        # wins; otherwise the reused embedder builds the real transformers adapter lazily from
        # config on first encode. No latence_splade import happens until then.
        self._model: SpladeModel | None = model
        self._embedder: SpladeCompositionEmbedder | None = None

    def _resolve_embedder(self) -> SpladeCompositionEmbedder:
        if self._embedder is None:
            try:
                from latence_splade import SpladeCompositionEmbedder
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                msg = (
                    "The 'latence-splade' package is required for query-side sparse generation. "
                    "Install it with: pip install 'latence-retrieval[sparse]'"
                )
                raise ImportError(msg) from exc
            self._embedder = SpladeCompositionEmbedder(self._config, model=self._model)
        return self._embedder

    def encode(
        self,
        query: str,
        *,
        entities: Sequence[tuple[str, float]] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> SparseVector:
        """Compose the query's sparse vector via the reused index-time SPLADE composition.

        Builds one :class:`~latence_splade.composition.SpladeInput` from the query text, the
        optional ``(surface, confidence)`` query entities, and the optional metadata, then runs the
        reused embedder's composition over it and returns the single resulting
        :class:`~latence_core.capability.SparseVector`. Empty/degenerate query text yields the empty
        vector (never a crash), exactly as the index-time seam does.
        """
        from latence_splade import EntityTerm, SpladeInput

        item = SpladeInput(
            text=query,
            entities=[EntityTerm(surface=surface, confidence=conf) for surface, conf in entities],
            metadata=dict(metadata or {}),
        )
        (vector,) = list(self._resolve_embedder().encode_composed([item]))
        return vector


__all__ = ["SpladeQueryEncoder", "SparseQueryEncoder"]

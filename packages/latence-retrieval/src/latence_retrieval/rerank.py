"""The ``rerank`` seam: :class:`Reranker` (a :class:`CandidateProcessor`) over a cross-encoder.

``rerank`` is the query-time transform that re-scores **one** already-fetched candidate list with a
stronger model and reorders it (CONTEXT ``Rerank``). It is a
:class:`~latence_retrieval.processor.CandidateProcessor` — the single-list ``(query, candidates) →
candidates`` shape — not a new seam: a caller fetched the candidates from its engine, and the
reranker only rescores/reorders them. It holds nothing and fetches nothing (ADR-0048); it is a pure
transform whose only power is a better relevance score.

The scoring model is itself a small, deep seam — :class:`CrossEncoderModel` — because *that* is
where behaviour varies: a real learned cross-encoder vs. an offline faithful fake. It is a genuine
two-adapter seam (the codebase-design bar): :class:`SentenceTransformersCrossEncoder` is the real
adapter (a config-selected, **license-verified-before-wiring** cross-encoder checkpoint via
``sentence-transformers``, heavy, lazy, ADR-0012), and
:class:`~latence_retrieval.testing.FakeCrossEncoder` is the shipped faithful fake every offline test
drives. The :class:`Reranker` **accepts** a :class:`CrossEncoderModel` (dependency injection) rather
than constructing a hidden one, so a test hands it a fake and production hands it the real adapter —
no seam with a single implementation.

Licence discipline (ADR-0012): **no checkpoint is pinned.** ``config["model"]`` is required, so a
caller can only run a cross-encoder whose weights AND code license it has cleared separately. The
``sentence-transformers`` library itself is Apache-2.0 (recorded in the package README); the
*weights* of any given cross-encoder are the operator's to verify, so nothing is hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_retrieval.contracts import Candidate
from latence_retrieval.errors import RerankModelError

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_MAX_LENGTH = 512


@runtime_checkable
class CrossEncoderModel(Protocol):
    """A cross-encoder relevance scorer: ``(query, texts) → one score per text``.

    The whole seam a reranker depends on — deliberately one method. Given a ``query`` and a batch of
    candidate ``texts``, it returns one relevance score per text, **in input order** (higher is more
    relevant). Batching is the contract, not an optimisation: a real cross-encoder scores the whole
    list in one forward pass, so :meth:`score` takes the list and returns the list — the reranker
    never loops one-text-at-a-time. The scorer holds no state, so identical inputs yield the same
    output.
    """

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance score per text (higher = more relevant), in input order."""
        ...


class Reranker:
    """Re-score a candidate list with a :class:`CrossEncoderModel` and reorder it (ADR-0048).

    A :class:`~latence_retrieval.processor.CandidateProcessor`: :meth:`process` scores every
    candidate's ``text`` against the ``query`` with the model **in one batch call**, replaces each
    candidate's ``score`` with that cross-encoder score, and returns a new list sorted by descending
    score (ties broken by ascending ``id`` so identical inputs always yield an identical order — the
    determinism the fuse seam holds too). The input list is never mutated
    (:class:`~latence_retrieval.contracts.Candidate` is frozen; a fresh list is returned), and the
    reranker keeps no state across calls.

    Config:
        model: str — the cross-encoder checkpoint id. **Required, no default** (ADR-0012): a caller
            names a model whose weights AND code license it has cleared separately; a restricted
            model is a deliberate opt-in, never a hidden default. Only consulted when the real
            adapter is built — an injected ``model`` (tests / a pre-built adapter) needs no id.
        top_k: int | None — if set (a positive int), the reranked list is truncated to its top
            ``top_k`` (the "improve top-k precision" use). Default ``None`` keeps every candidate,
            reordered.
        max_length: int — the model's max sequence length (passed to the real adapter). Default 512.

    The model is loaded lazily on first :meth:`process` (constructing the ``Reranker`` — which the
    registry does — never downloads weights).
    """

    def __init__(
        self, config: dict[str, Any] | None = None, *, model: CrossEncoderModel | None = None
    ) -> None:
        cfg = dict(config or {})
        top_k = cfg.get("top_k")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
        ):
            msg = f"rerank 'top_k' must be a positive integer or omitted, got {top_k!r}."
            raise ValueError(msg)
        self._top_k: int | None = top_k
        max_length = cfg.get("max_length", _DEFAULT_MAX_LENGTH)
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
            msg = f"rerank 'max_length' must be a positive integer, got {max_length!r}."
            raise ValueError(msg)
        self._max_length = max_length
        # Dependency injection: an explicit model (tests / query-time reuse) wins; otherwise the
        # real sentence-transformers adapter is built lazily from config on first use (no weights at
        # construction). A missing 'model' id is rejected by the adapter, not swallowed here.
        self._model: CrossEncoderModel | None = model
        self._config_model_id = str(cfg.get("model", "") or "")

    def _resolve_model(self) -> CrossEncoderModel:
        if self._model is None:
            self._model = SentenceTransformersCrossEncoder(
                self._config_model_id, max_length=self._max_length
            )
        return self._model

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Re-score ``candidates`` against ``query``, reorder by descending score, and (optionally)
        truncate to ``top_k``.

        Scores the whole list in one model call (batch-efficient), replaces each candidate's
        ``score`` with its cross-encoder score, and sorts descending with an ascending-``id``
        tiebreak. Returns a new list; the input is never mutated. An empty input returns ``[]``
        without touching the model (so an empty query needs no weights).
        """
        candidates = list(candidates)
        if not candidates:
            return []
        model = self._resolve_model()
        scores = model.score(query, [c.text for c in candidates])
        if len(scores) != len(candidates):
            # A model that returns the wrong number of scores has broken the seam's one-per-text
            # contract; fail loudly rather than silently mis-aligning scores to candidates.
            msg = (
                f"cross-encoder returned {len(scores)} scores for {len(candidates)} candidates; "
                "the contract is one score per candidate, in order."
            )
            raise RerankModelError(msg)
        rescored = [
            candidate.model_copy(update={"score": float(score)})
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-c.score, c.id))
        if self._top_k is not None:
            return rescored[: self._top_k]
        return rescored


class SentenceTransformersCrossEncoder:
    """The real :class:`CrossEncoderModel` — a config-selected ``sentence-transformers`` checkpoint.

    Wraps ``sentence_transformers.CrossEncoder`` and scores ``[query, text]`` pairs in one
    ``predict`` batch. Heavy deps (``sentence-transformers``/``torch``) ship behind this package's
    ``[rerank]`` extra, never latence-core or the pure-Python retrieval core (ADR-0016/0049), and
    are imported lazily on first use so ``Reranker`` *construction* and registry discovery never
    need them.

    Licence discipline (ADR-0012): **no checkpoint is pinned** — ``model_id`` is required, so a
    caller can only run a cross-encoder it has cleared (weights AND code verified separately). A
    missing id is a :class:`ValueError`; a runtime model fault is a
    :class:`~latence_retrieval.errors.RerankModelError` carrying no raw text.
    """

    def __init__(self, model_id: str, *, max_length: int = _DEFAULT_MAX_LENGTH) -> None:
        if not model_id:
            msg = (
                "rerank requires a 'model' (a cross-encoder checkpoint id): no default is pinned "
                "until a model's weights+code license is verified separately (ADR-0012)."
            )
            raise ValueError(msg)
        if max_length < 1:
            msg = f"rerank max_length must be >= 1, got {max_length}."
            raise ValueError(msg)
        self._model_id = model_id
        self._max_length = max_length
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'sentence-transformers' package is required to run rerank with the real "
                "cross-encoder. Install it with: pip install 'latence-retrieval[rerank]'"
            )
            raise ImportError(msg) from exc
        try:
            self._model = CrossEncoder(self._model_id, max_length=self._max_length)
        except (OSError, ValueError, KeyError) as exc:
            # A bad/unavailable checkpoint id is a caller config error, not a runtime fault.
            msg = f"rerank could not load cross-encoder {self._model_id!r}: {type(exc).__name__}."
            raise ValueError(msg) from exc

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score each ``[query, text]`` pair with the cross-encoder; one float per text, ordered."""
        texts = list(texts)
        if not texts:
            return []
        self._load()
        try:
            pairs = [[query, text] for text in texts]
            raw = self._model.predict(pairs)
            scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        except Exception as exc:
            # A native sentence-transformers/torch failure → typed error with NO raw text (no leak).
            msg = f"rerank cross-encoder failure during scoring: {type(exc).__name__}."
            raise RerankModelError(msg) from exc
        if len(scores) != len(texts):  # pragma: no cover - a real predict is one-per-pair
            msg = (
                f"cross-encoder returned {len(scores)} scores for {len(texts)} texts; "
                "the contract is one score per text, in order."
            )
            raise RerankModelError(msg)
        return [float(s) for s in scores]


__all__ = ["CrossEncoderModel", "Reranker", "SentenceTransformersCrossEncoder"]

"""The SPLADE model seam — the boundary where the *real* multilingual sparse model plugs in.

ADR-0053's composition needs one thing from a model: a term-weight map over a fixed vocabulary for
a piece of text, plus a way to map an injected surface/metadata string into that **same**
vocabulary so the structured terms union with the model's. That is the whole :class:`SpladeModel`
seam — a small, deep interface behind which the heavy, varying behaviour (a transformers MLM
forward + the SPLADE pooling) hides.

It is a genuine two-adapter seam (the codebase-design bar): :class:`TransformersSpladeModel` is the
real adapter (a config-selected multilingual SPLADE checkpoint via ``transformers``, heavy, lazy,
license-verified-before-wiring per ADR-0012), and a **faithful fake** is the second adapter used to
exercise the composition + Provider offline (in this package's tests and the conformance stub). The
Provider *accepts* a ``SpladeModel`` (dependency injection) rather than constructing a hidden one,
so a test hands it a fake and production hands it the transformers adapter — no seam with a single
implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_core.errors import ConfigError, ProviderError

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_MAX_LENGTH = 512


@runtime_checkable
class SpladeModel(Protocol):
    """A multilingual sparse (SPLADE-family) model: text → term-weights, and term → vocab ids.

    The seam ADR-0053's composition depends on. Two methods and one property, all in the model's
    own fixed vocabulary so the composition's union is well-defined:

    * :attr:`vocab_size` — the term-index space; every id a :class:`SpladeModel` returns is in
      ``[0, vocab_size)``.
    * :meth:`term_weights` — the SPLADE reduction: one ``{vocab_id: weight}`` map per input text
      (max-pooled ``log(1 + relu(logits))`` over the sequence), with only non-zero weights present.
    * :meth:`term_ids` — map an injected surface/metadata string to the vocab ids the composition
      unions in (the model's own subword tokenisation of the term).
    """

    @property
    def vocab_size(self) -> int:
        """The size of the term-index space: every returned id is in ``[0, vocab_size)``."""
        ...

    def term_weights(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Map each text to its SPLADE term-weight map (non-zero weights only), in order."""
        ...

    def term_ids(self, term: str) -> list[int]:
        """The unique vocab ids the term's subword tokens occupy (for structured-term injection)."""
        ...


def splade_reduce(
    logits: Sequence[Sequence[Sequence[float]]],
    attention_mask: Sequence[Sequence[float]],
) -> list[dict[int, float]]:
    """The SPLADE pooling: ``max_t attention · log(1 + relu(logits[t]))`` → per-text term map.

    Pure-Python and backend-free: it consumes the MLM logits ``[batch, seq, vocab]`` and the
    attention mask ``[batch, seq]`` as plain nested numbers (a real torch tensor is materialised
    with ``.tolist()`` by the adapter; a fake supplies lists directly), and returns one sparse
    ``{vocab_id: weight}`` map per text with only strictly-positive weights (a ``relu`` floors the
    rest to zero, and ``log1p`` of a masked/zero logit is zero — never stored). Deterministic given
    deterministic logits.

    (Reference-correct rather than throughput-tuned: on a 30k-vocab checkpoint the accelerated
    torch reduction is a documented follow-on; correctness is validated on real weights out of
    band.)
    """
    out: list[dict[int, float]] = []
    for seq_logits, seq_mask in zip(logits, attention_mask, strict=True):
        acc: dict[int, float] = {}
        for token_logits, mask_value in zip(seq_logits, seq_mask, strict=True):
            if not mask_value:
                continue
            for vocab_id, logit in enumerate(token_logits):
                if logit <= 0.0:
                    continue
                value = math.log1p(logit)
                previous = acc.get(vocab_id)
                if previous is None or value > previous:
                    acc[vocab_id] = value
        out.append(acc)
    return out


class TransformersSpladeModel:
    """The real :class:`SpladeModel` — a config-selected multilingual SPLADE checkpoint via HF.

    Wraps a ``transformers`` ``AutoTokenizer`` + ``AutoModelForMaskedLM`` and applies the SPLADE
    reduction (:func:`splade_reduce`). Heavy deps (``transformers``/``torch``) ship with this
    package, never latence-core (ADR-0016), and are imported lazily on first use so Provider
    *discovery* never needs them.

    Licence discipline (ADR-0012/0053): **no checkpoint is pinned** — ``model_id`` is required, so
    a Pipeline can only run a model it has cleared (weights AND code verified separately). A bad
    model id surfaces as a typed :class:`~latence_core.errors.ConfigError`; a runtime failure as a
    :class:`~latence_core.errors.ProviderError` carrying no raw text.
    """

    def __init__(self, model_id: str, *, max_length: int = _DEFAULT_MAX_LENGTH) -> None:
        if not model_id:
            msg = (
                "sparse.splade requires a 'model' (a multilingual SPLADE checkpoint id): ADR-0053 "
                "pins no default until its weights+code license are verified (ADR-0012)."
            )
            raise ConfigError(msg)
        if max_length < 1:
            msg = f"sparse.splade max_length must be >= 1, got {max_length}."
            raise ConfigError(msg)
        self._model_id = model_id
        self._max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._vocab_size: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'transformers' package is required to run sparse.splade. Install it with: "
                "pip install latence-splade"
            )
            raise ImportError(msg) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForMaskedLM.from_pretrained(self._model_id)
        except (OSError, ValueError, KeyError) as exc:
            # A bad/unavailable checkpoint id is an operator config error, not a runtime fault.
            msg = f"sparse.splade could not load model {self._model_id!r}: {type(exc).__name__}."
            raise ConfigError(msg) from exc
        if hasattr(model, "eval"):
            model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._vocab_size = int(model.config.vocab_size)

    @property
    def vocab_size(self) -> int:
        self._load()
        assert self._vocab_size is not None  # noqa: S101 - set by _load
        return self._vocab_size

    def term_weights(self, texts: Sequence[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        self._load()
        try:
            encoding = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            output = self._forward(encoding)
            logits = output.logits.tolist()
            attention_mask = encoding["attention_mask"].tolist()
        except (ConfigError, ProviderError):
            raise
        except Exception as exc:
            # A native transformers/torch failure → typed ProviderError with NO raw text (no leak).
            msg = f"sparse.splade model failure during encode: {type(exc).__name__}."
            raise ProviderError(msg) from exc
        return splade_reduce(logits, attention_mask)

    def _forward(self, encoding: Any) -> Any:
        model = self._model
        try:
            import torch
        except ImportError:  # pragma: no cover - torch present in the real runtime env
            return model(**encoding)
        with torch.no_grad():
            return model(**encoding)

    def term_ids(self, term: str) -> list[int]:
        self._load()
        try:
            ids = self._tokenizer(term, add_special_tokens=False)["input_ids"]
            id_list = list(ids.tolist()) if hasattr(ids, "tolist") else list(ids)
        except Exception as exc:
            msg = f"sparse.splade could not tokenise an injected term: {type(exc).__name__}."
            raise ProviderError(msg) from exc
        # Unique ids, order-stable, in-vocab — a repeated subword must not multiply the injection
        # boost, and an id must land in the same space the model's own weights use.
        seen: set[int] = set()
        unique: list[int] = []
        vocab = self._vocab_size
        for raw in id_list:
            idx = int(raw)
            if vocab is not None and not (0 <= idx < vocab):
                continue
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        return unique


__all__ = ["SpladeModel", "TransformersSpladeModel", "splade_reduce"]

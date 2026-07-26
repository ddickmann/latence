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

import contextlib
import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_core.errors import ConfigError, ProviderError
from latence_core.providers.device import DeviceDecision
from latence_core.providers.perf import inference_context, load_transformers_model

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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


def _import_transformers() -> tuple[Any, Any]:
    """Return the (``AutoModelForMaskedLM``, ``AutoTokenizer``) pair — the import factory.

    Deferred so Provider discovery never needs transformers/torch (ADR-0016); the install hint stays
    in this package because the native dependency does (ADR-0036 2). The model half is handed to
    :func:`~latence_core.providers.perf.load_transformers_model` as a factory rather than imported
    inline, so the seam evaluates it at the one instant that is correct (after the TorchScript JIT
    is dead) instead of that being a rule this module has to remember.
    """
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        msg = (
            "The 'transformers' package is required to run sparse.splade. Install it with: "
            "pip install latence-splade"
        )
        raise ImportError(msg) from exc
    return AutoModelForMaskedLM, AutoTokenizer


def _import_masked_lm() -> Any:
    """The model class alone — the shape :func:`load_transformers_model` expects."""
    return _import_transformers()[0]


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

    def __init__(
        self,
        model_id: str,
        *,
        max_length: int = _DEFAULT_MAX_LENGTH,
        device: DeviceDecision | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
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
        # (c): this adapter got NOTHING from the perf seam — a bare
        # ``AutoModelForMaskedLM.from_pretrained(model_id)`` (fp32 on a GPU, the ~2x miss) and no
        # placement at all, because the fp32 regression guard's CURATED provider list never included
        # ``sparse.splade``. It is a registered learned Provider running real transformer forwards.
        # The DeviceDecision + config are now threaded in so the load goes through the same
        # choreography as every other learned Provider; ``None`` (a caller that resolves no device,
        # e.g. an offline test) keeps the pre-existing no-placement behaviour.
        self._device = device or DeviceDecision(
            device=None, skipped=False, reason="no device resolved by the caller"
        )
        self._config: dict[str, Any] = dict(config or {})
        self._tokenizer: Any = None
        self._model: Any = None
        self._vocab_size: int | None = None

    @staticmethod
    @contextlib.contextmanager
    def _config_error_guard(op: str) -> Iterator[None]:
        """Map a bad/unavailable checkpoint id to a typed :class:`ConfigError`, not a crash.

        The typed-error policy this adapter has always had, now handed to the load choreography as
        a ``guard`` so it applies around the native load and nothing else. ``OSError`` /
        ``ValueError`` / ``KeyError`` from transformers mean the operator named a checkpoint that
        does not exist or
        cannot be read — their config, not a runtime fault. Anything else stays a ProviderError from
        the seam. The native message is never interpolated (no content leak), only its type name.
        """
        try:
            yield
        except (OSError, ValueError, KeyError) as exc:
            msg = f"sparse.splade could not load model in {op!r}: {type(exc).__name__}."
            raise ConfigError(msg) from exc

    def _load_tokenizer(self) -> Any:
        """Load the tokenizer — no perf kwargs, no placement, same ConfigError mapping."""
        tokenizer_cls = _import_transformers()[1]
        with self._config_error_guard("load_tokenizer"):
            return tokenizer_cls.from_pretrained(self._model_id)

    def _load(self) -> None:
        """Load tokenizer + model on first use, the model through the shared load choreography.

        The model half is :func:`~latence_core.providers.perf.load_transformers_model`: JIT kill ->
        intra-op thread cap -> backbone import -> bf16-on-GPU + sdpa/FA2 perf kwargs (graceful
        FA2->sdpa) -> guarded load -> **verified** placement on the resolved device -> guarded
        compile. Before this it was a bare ``AutoModelForMaskedLM.from_pretrained(model_id)``: fp32
        on a GPU and resident wherever ``from_pretrained`` happened to leave it.

        The bad-checkpoint-id -> :class:`~latence_core.errors.ConfigError` mapping is preserved by
        handing the seam a ``guard``: the same operator-facing distinction, applied around the
        native load only, so the missing-``transformers`` ``ImportError`` still surfaces raw.
        """
        if self._model is not None:
            return
        model = load_transformers_model(
            _import_masked_lm,
            self._model_id,
            self._config,
            self._device,
            provider_name="sparse.splade",
            guard=self._config_error_guard,
        )
        tokenizer = self._load_tokenizer()
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
            # Placement has TWO halves. ``_load`` moved the model; the tokenizer does not know
            # about that and always returns CPU tensors, so the encoding has to follow it or the
            # forward mixes a cuda module with cpu inputs and torch raises. Same two lines as the
            # sibling transformers adapters (``parser.lighton``/``parser.glm``), keyed off the same
            # resolved device the load placed on; ``None`` (no device resolved) moves nothing.
            device = self._device.device
            if device is not None:
                encoding = encoding.to(device)
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
        """One MLM forward under :func:`~latence_core.providers.perf.inference_context`.

        (c): this used ``torch.no_grad()``, which is STRICTLY WEAKER. ``no_grad`` stops gradients
        being *computed*, but the forward still builds an autograd graph and pins every intermediate
        activation alive for a backward that never comes; across a long run those never-freed
        activations accumulate (pod-measured 27.28 GiB resident on a ~2-3 GB model) until the next
        allocation CUDA-OOMs, and ``empty_cache()`` cannot reclaim any of it because it is held
        LIVE. ``inference_mode`` is what killed that OOM: no graph is built at all, so each
        forward's activations free the moment they fall out of scope. A SPLADE forward never
        backpropagates, so
        this is free — and byte-identical: inference mode only suppresses grad bookkeeping.
        Torch-free-safe (the offline env gets a null context).
        """
        with inference_context():
            return self._model(**encoding)

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

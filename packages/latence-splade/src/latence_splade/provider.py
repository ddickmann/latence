"""``SpladeCompositionEmbedder`` — the real multilingual SPLADE ``SparseEmbedder`` (ADR-0053, #162).

The blessed *real* sparse signal generator the in-core ``sparse.hashing`` reference stands in for.
It satisfies the framework's :class:`~latence_core.capability.SparseEmbedder` Protocol
(``vocab_size`` + ``encode_sparse(texts)``), so #160's Export wires it exactly like the reference —
emitting the parallel ``sparse_indices`` / ``sparse_values`` columns a store's sparse index ingests
— but its vector is the ADR-0053 **structured composition**, not plain ``SPLADE(text)``.

Two entry points onto one composition (:func:`~latence_splade.composition.compose`):

* :meth:`encode_sparse` — the ``SparseEmbedder`` Protocol method #160's text-only Export seam calls.
  With only text available it composes the **model leg** over the (already-redacted) corpus text —
  the honest, trivially PII-safe index-time column.
* :meth:`encode_composed` — the deep method: given :class:`~latence_splade.composition.SpladeInput`
  items (text + confidence-bearing entities + selected metadata), it emits the **full** ADR-0053
  composition. This is the vector ADR-0053 specifies, and the *same* call composes a query's sparse
  vector at query time (one composition, two clocks).

Design (codebase-design): the model is an **injected** :class:`~latence_splade.model.SpladeModel`
dependency, not a hidden construction — production passes the transformers adapter, tests pass a
faithful fake — so the deep composition/gating behaviour is exercised offline without weights.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.capability import SparseVector
from latence_core.errors import ConfigError
from latence_core.providers.device import select_device
from latence_core.providers.profile import UNVERIFIED_LICENSE, ProviderProfile

from latence_splade.composition import EntityTerm, SpladeInput, compose
from latence_splade.model import SpladeModel, TransformersSpladeModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from latence_core.capability import SparseSignals
    from latence_core.providers.device import DeviceDecision

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_DEFAULT_ENTITY_BOOST = 1.0
_DEFAULT_METADATA_BOOST = 1.0
_DEFAULT_MODEL_WEIGHT = 1.0
_MODES = ("inject", "weighted")

# Word tokens for the redaction survival check: lower-cased ``\w+`` runs, mirroring the model's
# lower-cased subword vocabulary. Used to test whether a masked surface still occurs as a whole
# token/phrase in the redacted text — NOT as a bare substring (which would re-leak a masked name
# whose letters happen to sit inside a surviving word, e.g. "ann" inside "annual").
_WORD = re.compile(r"\w+", re.UNICODE)


def _word_tokens(text: str) -> list[str]:
    """Lower-cased word tokens of ``text`` (the redaction-gate token space)."""
    return _WORD.findall(text.casefold())


def _occurs_as_phrase(needle: list[str], haystack: list[str]) -> bool:
    """True iff the token list ``needle`` occurs as a contiguous sub-sequence of ``haystack``.

    Word-boundary survival: a single-token surface must match a whole surviving token (``ann`` does
    NOT survive inside ``annual``), and a multi-word surface (``["john", "smith"]``) must survive as
    an adjacent run. An empty ``needle`` (a punctuation-only surface with no word tokens) never
    survives — there is nothing that could have been redacted, and it injects no ids regardless.
    """
    n = len(needle)
    if n == 0:
        return False
    return any(
        haystack[start : start + n] == needle for start in range(len(haystack) - n + 1)
    )


class SpladeCompositionEmbedder:
    """Structured-composition SPLADE Provider satisfying the ``SparseEmbedder`` Protocol (ADR-0053).

    Config:
        model: str  — the multilingual SPLADE checkpoint id. **Required, no default**: ADR-0053 pins
            no model until its weights AND code license are verified separately (ADR-0012), so a
            Pipeline names one it has cleared. A restricted-license checkpoint is a deliberate
            opt-in.
        mode: str  — ``"inject"`` (default, term-injection: the model's weights are kept as-is and
        the boosted structured terms are added on top) or ``"weighted"`` (the model leg is scaled
        by ``model_weight`` so its contribution is tunable against the structured legs).
        confidence_threshold: float  — entities with extraction ``confidence`` below this are
        dropped so weak NER never pollutes the sparse signal (ADR-0053). Default 0.5.
        entity_boost / metadata_boost: float  — the weight added per injected entity / metadata
        vocab id. Default 1.0.
        model_weight: float  — the multiplier on the model leg in ``"weighted"`` mode (ignored in
            ``"inject"`` mode, where it is pinned to 1.0). Default 1.0.
        metadata_fields: list[str]  — which metadata fields' values are injected as terms (e.g.
            ``["classification"]``). Default none (a bare model+entity composition).
        redact: bool  — when True (default), an entity whose surface no longer occurs in the
        redacted corpus text is **not** injected, so a masked PII surface is never re-leaked as a
        term (ADR-0053). ``False`` is an explicit opt-in for fully-internal deployments that inject
        every confidence-gated entity regardless of the redaction floor.
        max_length: int  — the model's max sequence length. Default 512.

    The model is loaded lazily on first encode (constructing the Provider — which the Runner does
    per Stage — never downloads weights).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) — an honest UNVERIFIED-license, no-default posture.

        ADR-0053 pins **no** model until a multilingual SPLADE checkpoint's weights AND code
        licence are verified separately (ADR-0012 / the research-diligence directive). So this
        profile declares ``model_id=None`` and the headline ``license`` as the honest
        :data:`UNVERIFIED` sentinel with ``license_verified=False`` — it never fabricates a
        permissive claim for a model the operator has not yet chosen and cleared. The **package
        code** and the ``transformers`` library are Apache-2.0 (``license_code``), recorded
        truthfully; the *weights* licence is left ``None`` and unverified because there is no
        default checkpoint to attest. When an operator wires a cleared model, its licence is
        verified out of band and recorded there.

        ``deterministic=False``: a learned MLM forward (torch matmuls, possibly bf16/GPU) is not
        guaranteed byte-identical across hardware/library versions — declared honestly rather than
        overclaiming reproducibility (the deterministic ``sparse.hashing`` reference stays the
        golden-path default). ``batch=True`` (the tokenizer/model batch natively).
        ``cost_per_1k=None`` (in-process, no per-call vendor cost).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=1200,
            model_id=None,
            license=UNVERIFIED_LICENSE,
            license_verified=False,
            license_code="Apache-2.0",
            license_weights=None,
            license_source=(
                "package + https://github.com/huggingface/transformers/blob/main/LICENSE "
                "(code, Apache-2.0); weights UNVERIFIED — no default model is pinned (ADR-0053): a "
                "cleared multilingual SPLADE checkpoint is config-selected and its weights+code "
                "license verified separately (ADR-0012) before it is wired."
            ),
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note="reference-correct SPLADE pooling; torch-accelerated reduce is a follow-on",
        )

    def __init__(
        self, config: dict[str, Any] | None = None, *, model: SpladeModel | None = None
    ) -> None:
        cfg = config or {}
        mode = str(cfg.get("mode", "inject"))
        if mode not in _MODES:
            msg = f"sparse.splade 'mode' must be one of {_MODES}, got {mode!r}."
            raise ConfigError(msg)
        self._mode = mode
        self._confidence_threshold = float(
            cfg.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD)
        )
        self._entity_boost = float(cfg.get("entity_boost", _DEFAULT_ENTITY_BOOST))
        self._metadata_boost = float(cfg.get("metadata_boost", _DEFAULT_METADATA_BOOST))
        # Term-injection keeps the model's weights untouched (model_weight == 1.0); only weighted
        # mode honours a configured multiplier — so the two modes are observably distinct.
        self._model_weight = (
            float(cfg.get("model_weight", _DEFAULT_MODEL_WEIGHT))
            if mode == "weighted"
            else _DEFAULT_MODEL_WEIGHT
        )
        metadata_fields = cfg.get("metadata_fields") or []
        if not isinstance(metadata_fields, (list, tuple)):
            msg = "sparse.splade 'metadata_fields' must be a list of field names."
            raise ConfigError(msg)
        self._metadata_fields = [str(f) for f in metadata_fields]
        self._redact = bool(cfg.get("redact", True))
        # Dependency injection: an explicit model (tests / query-time reuse) wins; otherwise the
        # real
        # transformers adapter is built lazily from config (no weights are touched at construction).
        self._model: SpladeModel | None = model
        self._config_model_id = str(cfg.get("model", "") or "")
        self._max_length = int(cfg.get("max_length", 512))
        # (c): ``sparse.splade`` is a registered LEARNED Provider running real transformer forwards,
        # and it had no device routing and no perf seam at all — the fp32 regression guard's curated
        # list simply never named it. Resolve the device ONCE here, through the ADR-0036 seam, like
        # every other learned Provider; the adapter then loads through the shared choreography.
        # ``compute="either"`` (see :meth:`profile`), so a CPU-only host resolves to cpu rather than
        # skipping. An invalid ``device`` string is an operator ConfigError at construction.
        self._device = self._resolve_device(cfg.get("device", "auto"))
        self._perf_config: dict[str, Any] = dict(cfg)

    @classmethod
    def _resolve_device(cls, requested: object) -> DeviceDecision:
        """Route the ``device`` config value (``auto|cpu|cuda``) through the device seam."""
        try:
            return select_device(str(requested), cls.profile())
        except ValueError as exc:
            msg = f"sparse.splade: invalid device config: {exc}"
            raise ConfigError(msg) from exc

    def _resolve_model(self) -> SpladeModel:
        if self._model is None:
            # Constructs (does not load) the transformers adapter; a missing 'model' is rejected
            # there. The resolved device + perf config travel with it so the eventual load runs the
            # shared choreography rather than a bare fp32 ``from_pretrained``.
            self._model = TransformersSpladeModel(
                self._config_model_id,
                max_length=self._max_length,
                device=self._device,
                config=self._perf_config,
            )
        return self._model

    @property
    def vocab_size(self) -> int:
        """The model's term-index space: every emitted index lies in ``[0, vocab_size)``."""
        return self._resolve_model().vocab_size

    def encode_sparse(self, texts: Iterable[str]) -> Iterator[SparseVector]:
        """Map each text to its composed :class:`SparseVector`, in order (the Protocol method).

        The ``SparseEmbedder`` seam #160's Export calls: with only text available (no structured
        signals cross the text-only seam) it composes the **model leg** over the given — already
        redacted, per :func:`latence_core.stages.export._corpus_text` — corpus text. One vector per
        text, in order; empty text yields the empty ``SparseVector()``.
        """
        yield from self.encode_composed(SpladeInput(text=text) for text in texts)

    def encode_composed(self, items: Iterable[SpladeInput]) -> Iterator[SparseVector]:
        """Map each :class:`SpladeInput` to its full ADR-0053 composed :class:`SparseVector`.

        The deep method: unions the model's term-weights with the confidence-gated entity surface
        terms and the selected metadata terms (:func:`~latence_splade.composition.compose`). The
        model is invoked once over the whole batch of texts (order preserved); each item's
        structured legs are resolved into the model's vocabulary via :meth:`SpladeModel.term_ids`
        and composed. The same call composes a query's sparse vector at query time (ADR-0053, one
        composition/two clocks).
        """
        batch = list(items)
        if not batch:
            return
        model = self._resolve_model()
        model_maps = model.term_weights([item.text for item in batch])
        if len(model_maps) != len(batch):
            msg = (
                f"sparse.splade model returned {len(model_maps)} term maps for {len(batch)} texts; "
                "the contract yields one map per input text, in order."
            )
            raise ConfigError(msg)
        for item, model_terms in zip(batch, model_maps, strict=True):
            entity_terms = self._entity_terms(model, item)
            metadata_terms = self._metadata_terms(model, item.metadata)
            composed = compose(
                model_terms=model_terms,
                entity_terms=entity_terms,
                metadata_terms=metadata_terms,
                confidence_threshold=self._confidence_threshold,
                entity_boost=self._entity_boost,
                metadata_boost=self._metadata_boost,
                model_weight=self._model_weight,
            )
            yield SparseVector.from_terms(composed)

    def encode_sparse_composed(
        self, texts: Iterable[str], signals: Sequence[SparseSignals]
    ) -> Iterator[SparseVector]:
        """Compose each text with its per-record :class:`SparseSignals` (composing seam, ADR-0053).

        The index-time entry point the Export calls when it holds the chunk's structured signals
        (:class:`~latence_core.capability.ComposingSparseEmbedder`): it builds one
        :class:`~latence_splade.composition.SpladeInput` per ``(text, signals)`` — the (already
        redacted) corpus text plus the chunk's confidence-bearing entity surfaces and selected
        metadata — and runs the **same** :meth:`encode_composed` the query-side twin runs, so the
        corpus and a query land in one sparse space (one composition, two clocks). ``signals`` is
        parallel to ``texts`` (same length + order); one vector per input, in order; a record with
        empty signals composes to exactly the model leg (identical to :meth:`encode_sparse`).
        """
        text_list = list(texts)
        signal_list = list(signals)
        if len(text_list) != len(signal_list):
            msg = (
                f"sparse.splade encode_sparse_composed got {len(text_list)} texts for "
                f"{len(signal_list)} signals; the signals must be parallel to the texts "
                "(one SparseSignals per text, in order)."
            )
            raise ConfigError(msg)
        items = [
            SpladeInput(
                text=text,
                entities=[
                    EntityTerm(surface=surface, confidence=conf) for surface, conf in sig.entities
                ],
                metadata=dict(sig.metadata),
            )
            for text, sig in zip(text_list, signal_list, strict=True)
        ]
        yield from self.encode_composed(items)

    def _entity_terms(
        self, model: SpladeModel, item: SpladeInput
    ) -> list[tuple[list[int], float]]:
        """Resolve the item's entities into ``(vocab_ids, confidence)`` legs, applying redaction.

        The confidence gate is applied downstream in :func:`compose`; the **redaction** gate is
        applied here (it needs the text): when ``redact`` is on (default), an entity whose surface
        no longer occurs — *as a whole word/phrase* — in the redacted corpus text is dropped, so a
        masked PII surface is never re-injected as a term (ADR-0053). Matching is on lower-cased
        **word tokens** (mirroring the model's lower-cased subword vocabulary), not raw substrings:
        a masked ``"ann"`` does NOT survive inside the surviving word ``"annual"`` (a bare ``in``
        substring test would re-leak it), and a multi-word surface must survive as an adjacent run.
        """
        haystack = _word_tokens(item.text)
        legs: list[tuple[list[int], float]] = []
        for entity in item.entities:
            surface = entity.surface
            if self._redact and not _occurs_as_phrase(_word_tokens(surface), haystack):
                continue
            ids = model.term_ids(surface)
            if ids:
                legs.append((ids, entity.confidence))
        return legs

    def _metadata_terms(
        self, model: SpladeModel, metadata: Mapping[str, str]
    ) -> list[list[int]]:
        """Resolve the operator-selected metadata field values into vocab-id legs (order-stable)."""
        legs: list[list[int]] = []
        for field in self._metadata_fields:
            value = metadata.get(field)
            if not value:
                continue
            ids = model.term_ids(str(value))
            if ids:
                legs.append(ids)
        return legs


def entity_terms_from(pairs: Sequence[tuple[str, float]]) -> list[EntityTerm]:
    """Convenience: build :class:`EntityTerm`s from ``(surface, confidence)`` pairs (query-time
    helper)."""
    return [EntityTerm(surface=s, confidence=c) for s, c in pairs]


__all__ = ["SpladeCompositionEmbedder", "entity_terms_from"]

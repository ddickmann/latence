"""Learned sentence-transformers Embedder Provider over multilingual-e5-small (W1-embed, #93).

The first real *learned* Embedder adapter (ADR-0004; ADR-0007 CPU-first; ADR-0016 heavy-dep
isolation — torch/sentence-transformers ship HERE, not in the thin core), built through the
:class:`~latence_core.providers.adapter.AdapterBase` template (P3-F2/ADR-0036 §1). It wraps a
permissively-licensed model via the ``sentence-transformers`` library so it runs in-process on a
laptop CPU — first contact is ``pip install latence-embedder-st`` → real semantic embeddings, no
CUDA — and accelerates on a GPU when one is present (the device seam, ``device=`` in ST).

It is the *same opt-in Embedder Capability* as the in-core deterministic ``embedding.hashing``
reference and the ``embedding.endpoint`` performance path — a Pipeline swaps the toy hashing
embedder for a real semantic model with **one config line** (ADR-0007), so the exported RAG corpus
carries embeddings a vector DB ingests directly (ADR-0017), and the F5 bake-off gets a real
embedding matrix (``embedding.hashing`` vs a learned model).

Model card convention (multilingual-e5-small, verified 2026-07-08):
    e5 models are trained with an instruction **prefix** — ``query: `` for a search query and
    ``passage: `` for a document/corpus text — and their vectors are **L2-normalised** (the model
    card scores cosine similarity over unit vectors). This Provider embeds RAG-corpus text, which
    is the *document* side, so it prepends ``passage: `` by default (configurable ``prefix``) and
    passes ``normalize_embeddings=True`` to ST so every exported vector is unit-length. The
    ``dimension`` (e5-small = 384) is read from the loaded model, so the exported vectors are that
    width. See the model card: https://huggingface.co/intfloat/multilingual-e5-small .

Licensing (ADR-0012, verified 2026-07-08):
    * ``sentence-transformers`` library (PyPI + github.com/UKPLab/sentence-transformers):
      **Apache-2.0** (code).
    * Default checkpoint ``intfloat/multilingual-e5-small``: **MIT** (weights; the model the
      Embedder docstring already names as the default). ``MiniLM`` (Apache-2.0) is a documented
      config-selectable alternative.
    Both weights AND code are permissive; nothing is downloaded unless a Pipeline actually
    configures this Provider.

The ``sentence-transformers`` import is deferred to first use so importing this module (for
Provider discovery) never requires torch to be installed, and a missing optional dependency
surfaces as a clear, actionable error only when the Provider is actually run — not at registry
enumeration time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.errors import ProviderError
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.perf import load_with_fa2_fallback as _load_with_fa2_fallback
from latence_core.providers.profile import ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# MIT weights (verified). The model the Embedder docstring names as the default; 384-dim.
_DEFAULT_MODEL = "intfloat/multilingual-e5-small"
# The width the default checkpoint produces — declared in the profile (P3-F1 is descriptor-only)
# and re-asserted against the loaded model at runtime so a mis-declared dimension fails loudly.
_DEFAULT_DIMENSION = 384
# e5's document-side instruction prefix (the RAG-corpus text is the passage/document side).
_DEFAULT_PREFIX = "passage: "


class SentenceTransformersEmbedder(AdapterBase):
    """Learned Embedder Provider backed by a sentence-transformers checkpoint (CPU/GPU).

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass satisfying the
    :class:`~latence_core.capability.Embedder` Protocol (``dimension`` property +
    ``embed(texts) -> Iterator[list[float]]``). It composes the AdapterBase seams: :meth:`device`
    for cpu|cuda placement, :meth:`batched` for input-order-preserving streaming batches, and
    :meth:`guard` for native-exception → typed :class:`~latence_core.errors.ProviderError`
    mapping (no raw text / PII in the message).

    Config:
        model: str  — the sentence-transformers checkpoint id (default
            ``intfloat/multilingual-e5-small``, MIT; ``sentence-transformers/all-MiniLM-L6-v2``,
            Apache-2.0, is a documented alternative). A restricted-license checkpoint is a
            deliberate opt-in, never the default.
        dimension: int | None  — the expected embedding width. Default 384 (e5-small). Validated
            against the loaded model's real width on first ``embed`` so a config that disagrees
            with the model fails loudly (a downstream vector DB needs a uniform column, ADR-0017).
        prefix: str  — the e5 instruction prefix prepended to each text (default ``"passage: "``,
            the document side). Set to ``""`` for a model that needs no prefix (e.g. MiniLM).
        batch_size: int  — texts per ``encode`` call (default from AdapterBase), to bound memory
            over a long corpus. The output is input-order identical for any positive batch size.
        device: str  — ``auto|cpu|cuda`` through the shared device seam.

    The model is loaded lazily and cached on first ``embed`` call, so constructing the Provider
    (which the Runner does for every Stage) never downloads weights.
    """

    #: A native ``ValueError``/``KeyError`` from ST/transformers means an operator handed a bad
    #: config (a bad ``model`` id, a bad parameter) → :class:`~latence_core.errors.ConfigError`
    #: via :meth:`guard`; everything else is a runtime :class:`~latence_core.errors.ProviderError`.
    config_exceptions = (ValueError, KeyError, OSError)

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default multilingual-e5-small checkpoint.

        ``compute="either"``: sentence-transformers runs in-process on a laptop CPU (the ADR-0007
        CPU path — first contact is ``pip install`` → real embeddings, no CUDA) and accelerates on
        a GPU when present, so the device seam routes it to ``auto``/``cpu``/``cuda`` rather than
        skipping it on a CPU-only host.

        License is **verified** per ADR-0012, recorded evidence-bearingly in the structured fields
        (H-B1 — the claim is inseparable from its proof): weights and code are recorded
        **separately** (``license_code="Apache-2.0"`` for the ``sentence-transformers`` library;
        ``license_weights="MIT"`` for the default checkpoint ``intfloat/multilingual-e5-small``),
        with a ``license_source`` citing the HF model-card + the GitHub LICENSE where a human
        checked, and ``license_verified_on="2026-07-08"``. The headline ``license`` is the
        **weights** SPDX (``"MIT"``) — the effective license of what the vectors embody — and
        ``license_verified=True`` because BOTH sides are permissive.

        ``deterministic=False``: learned float inference (torch matmuls) is not guaranteed
        byte-identical across hardware/library/BLAS versions (unlike the pure-code hashing
        embedder), so the profile declares it non-deterministic rather than overclaim
        reproducibility (verified: ST CPU inference is *stable within a fixed environment* but the
        cross-environment byte-identity C4 checks is not promised — declared False, not assumed).
        ``batch=True`` (ST batches its ``encode`` natively). ``memory_mb`` is the approximate
        resident footprint of the e5-small checkpoint + its transformers stack (declared, not
        measured here — P3-F1 is descriptor-only). ``cost_per_1k=None`` (in-process, no per-call
        vendor cost).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=512,
            model_id=_DEFAULT_MODEL,
            license="MIT",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="MIT",
            license_source=(
                "https://huggingface.co/intfloat/multilingual-e5-small (weights, MIT); "
                "https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE "
                "(library, Apache-2.0)"
            ),
            license_verified_on="2026-07-08",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            # W8: honest declared load-time perf posture (descriptor-only, P3-F1; not measured).
            perf_note="bf16-GPU/fp32-CPU, sdpa (FA2 when flash_attn present), via ST model_kwargs",
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        dimension = cfg.get("dimension", _DEFAULT_DIMENSION)
        if dimension is not None:
            dimension = int(dimension)
            if dimension < 1:
                msg = f"{self.provider_name}: dimension must be >= 1, got {dimension}."
                raise ProviderError(msg)
        self._declared_dimension: int | None = dimension
        self._model_id = str(cfg.get("model", _DEFAULT_MODEL))
        self._prefix = str(cfg.get("prefix", _DEFAULT_PREFIX))
        self._model: Any = None
        # Resolved from the loaded model on first embed; falls back to the declared dimension so
        # ``dimension`` is answerable before the first ``embed`` (a downstream Export reads it to
        # size its column). Re-checked against the model's real width when the model loads.
        self._dimension: int = dimension if dimension is not None else _DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        """The fixed embedding width every returned vector has (the model's, else declared)."""
        return self._dimension

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        """Import ``sentence-transformers`` and load the checkpoint on first use (cached).

        Deferred so Provider discovery (importing this module) never needs torch, and a missing
        optional dependency surfaces as a clear install hint only when the Provider is actually
        run. The chosen device (cpu|cuda) flows from the AdapterBase device seam into ST's
        ``device=`` so placement stays the adapter's job (ADR-0003). After load, the model's real
        embedding width is read and cross-checked against a declared ``dimension`` — a mismatch is
        a loud :class:`~latence_core.errors.ProviderError`, never a silently wrong column.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'sentence-transformers' package is required to run "
                "SentenceTransformersEmbedder. Install it with: pip install latence-embedder-st"
            )
            raise ImportError(msg) from exc
        # Map the resolved device seam to ST's device= (None ⇒ ST auto-selects; the seam never
        # hands a fabricated device — a GPU-only host would have been skipped upstream).
        st_device = self.device.device
        # W8 perf seam: thread the resolved dtype + attn implementation into the underlying
        # transformer via ST's ``model_kwargs`` (its documented pass-through to from_pretrained),
        # so the checkpoint loads bf16-on-GPU + sdpa/FA2 instead of the fp32/eager default. A FA2
        # load that a model does not support degrades gracefully to sdpa (never a crash), then the
        # model is optionally compiled (guarded, off by default).
        perf = self.load_perf_kwargs()
        with self.guard("load_model"):
            model = self._load_st(SentenceTransformer, self._model_id, st_device, perf)
        # Read the real embedding width from the loaded (uncompiled) model before any compile wrap —
        # ``get_sentence_embedding_dimension`` is a container method, read it off the eager module.
        real_dim = int(model.get_sentence_embedding_dimension())
        if self._declared_dimension is not None and real_dim != self._declared_dimension:
            msg = (
                f"{self.provider_name}: model {self._model_id!r} produces {real_dim}-dim vectors, "
                f"but config declared dimension {self._declared_dimension}."
            )
            raise ProviderError(msg)
        self._dimension = real_dim
        # Optional guarded compile (off by default) — after the width is read, so a compile wrap can
        # never hide the container's introspection methods.
        self._model = self.maybe_compile(model)
        return self._model

    @staticmethod
    def _load_st(st_cls: Any, model_id: str, device: str | None, perf: dict[str, Any]) -> Any:
        """Construct the ``SentenceTransformer``, threading perf kwargs, FA2→sdpa on FA2 failure.

        The W8 perf kwargs (``torch_dtype`` + ``attn_implementation``) go through ST's documented
        ``model_kwargs`` pass-through to the backbone's ``from_pretrained``. A ``flash_attention_2``
        load can raise on a backbone that does not support it (e.g. no FA2 kernel for the arch); the
        shared :func:`~latence_core.providers.perf.load_with_fa2_fallback` catches that and retries
        once on ``sdpa`` so the embedder still loads — graceful degradation, never a crash. Any
        non-FA2 load error propagates (mapped by the guard).
        """
        return _load_with_fa2_fallback(
            lambda mid, **mk: st_cls(mid, device=device, model_kwargs=mk),
            model_id,
            perf,
        )

    # -- embedding ----------------------------------------------------------

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        """Map each input text to a ``dimension``-wide vector, in input order (ADR-0033 streaming).

        Batching is delegated to the AdapterBase :meth:`batched` helper (profile ``batch=True``),
        so the corpus is consumed one bounded batch at a time (memory O(batch_size), never the
        whole corpus) and the outputs are yielded in input order for any positive batch size. Each
        batch goes through :meth:`guard` so a native ST/torch failure (bad model, OOM, encode
        error) becomes a typed :class:`~latence_core.errors.ProviderError` carrying no raw text.
        """
        model = None

        def run_batch(batch: list[str]) -> list[list[float]]:
            nonlocal model
            if model is None:
                model = self._load_model()
            return self._encode_batch(model, batch)

        yield from self.batched(texts, run_batch)

    def _encode_batch(self, model: Any, batch: list[str]) -> list[list[float]]:
        """Encode one batch through ST, e5-prefixed + L2-normalised, in input order.

        The e5 instruction ``prefix`` is prepended to each text and ``normalize_embeddings=True``
        makes every vector unit-length (the model-card cosine convention, ADR-0017 uniform column).
        The width is re-asserted per vector so a model that ever returns an off-width row is caught
        as a :class:`~latence_core.errors.ProviderError`, not silently exported.
        """
        prefixed = [self._prefix + text for text in batch]
        with self.guard("embed"):
            raw = model.encode(
                prefixed,
                batch_size=len(prefixed),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        vectors = [[float(x) for x in row] for row in raw]
        if len(vectors) != len(batch):
            msg = (
                f"{self.provider_name}: model returned {len(vectors)} vectors for a batch of "
                f"{len(batch)} texts."
            )
            raise ProviderError(msg)
        for vector in vectors:
            if len(vector) != self._dimension:
                msg = (
                    f"{self.provider_name}: model returned a {len(vector)}-dim vector; "
                    f"embedding dimension is {self._dimension}."
                )
                raise ProviderError(msg)
        return vectors

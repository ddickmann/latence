"""Learned sentence-transformers Embedder Provider over IBM Granite Embedding r2 (W1-embed, W19).

The framework's default *learned* Embedder adapter (ADR-0004; ADR-0007 CPU-first; ADR-0016
heavy-dep isolation — torch/sentence-transformers ship HERE, not in the thin core), built through
the :class:`~latence_core.providers.adapter.AdapterBase` template (P3-F2/ADR-0036 §1). It wraps a
permissively-licensed model via the ``sentence-transformers`` library so it runs in-process on a
laptop CPU — first contact is ``pip install latence-embedder-st`` → real semantic embeddings, no
CUDA — and accelerates on a GPU when one is present (the device seam, ``device=`` in ST).

It is the *same opt-in Embedder Capability* as the in-core deterministic ``embedding.hashing``
reference and the ``embedding.endpoint`` performance path — a Pipeline swaps the toy hashing
embedder for a real semantic model with **one config line** (ADR-0007), so the exported RAG corpus
carries embeddings a vector DB ingests directly (ADR-0017), and the F5 bake-off gets a real
embedding matrix (``embedding.hashing`` vs a learned model).

Two tiers (ADR-0045), both **IBM Granite Embedding r2**, both Apache-2.0 weights, both ModernBERT-
based (alternating attention + GeGLU/SiLU + RoPE), single-vector dense, 32,768-token context,
200+ languages (enhanced 52 + code), NO instruction/query-document prefix:
    * **GPU tier — the default** — ``ibm-granite/granite-embedding-311m-multilingual-r2``
      (768-dim). On a GPU it loads **bf16 + flash_attention_2** through the W8 perf seam
      (``attn_implementation`` + ``torch_dtype`` via ST ``model_kwargs``), with a graceful ``sdpa``
      fallback when FA2 is unavailable
      (:func:`~latence_core.providers.perf.load_with_fa2_fallback`); on a CPU box the same perf seam
      resolves fp32 + sdpa, so it still runs (heavier, but no CUDA).
    * **CPU tier** — ``ibm-granite/granite-embedding-97m-multilingual-r2`` (384-dim), the lighter
      laptop-CPU option (fp32, no FA2). Selected with one config line
      (``model: ibm-granite/granite-embedding-97m-multilingual-r2``, ``dimension: 384``).

Model card convention (Granite r2, verified 2026-07-14):
    Granite r2 needs **NO prefix** — its usage example encodes text directly
    (``model.encode(texts)``), unlike the e5 family's ``query: ``/``passage: `` instruction
    prefixes. So the default ``prefix`` is the empty string ``""`` (set it only for a model that
    *needs* one). The Provider still passes ``normalize_embeddings=True`` so every exported vector
    is unit-length (cosine convention, ADR-0017 uniform column). The ``dimension`` is read from the
    loaded model (311m = 768, 97m = 384), so exported vectors are that width. Cards:
    https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2 and
    https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2 .

Licensing (ADR-0012/0045, verified 2026-07-14 — weights and code checked SEPARATELY):
    * ``sentence-transformers`` / ``transformers`` runtime library
      (PyPI + github.com/UKPLab/sentence-transformers): **Apache-2.0** (code).
    * Both Granite r2 checkpoints (``granite-embedding-311m-multilingual-r2`` and
      ``granite-embedding-97m-multilingual-r2``): **Apache-2.0** (weights; quoted card tag
      ``license: apache-2.0``). The ModernBERT backbone is Apache-2.0.
    * ``intfloat/multilingual-e5-small`` (MIT) and ``sentence-transformers/all-MiniLM-L6-v2``
      (Apache-2.0) remain documented tiny, config-selectable fallbacks — NOT the default.
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

# Apache-2.0 weights (verified 2026-07-14). The default = the GPU-tier IBM Granite Embedding r2
# flagship; 768-dim, ModernBERT-based, loads bf16+FA2 on a GPU through the W8 perf seam.
_DEFAULT_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
# The lighter CPU-tier Granite r2 checkpoint (384-dim, fp32, no FA2) — the documented one-line
# option for laptop-CPU deployments (``model: <this>``, ``dimension: 384``). Apache-2.0.
_CPU_TIER_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
# Documented tiny, config-selectable FALLBACKS (never the default): e5-small (MIT) and
# MiniLM (Apache-2.0). Kept so an adopter on a very small box can pin them explicitly.
_E5_FALLBACK_MODEL = "intfloat/multilingual-e5-small"
# The width the default checkpoint produces — declared in the profile (P3-F1 is descriptor-only)
# and re-asserted against the loaded model at runtime so a mis-declared dimension fails loudly.
_DEFAULT_DIMENSION = 768
# Granite r2 needs NO instruction/query-document prefix (its card encodes text directly), unlike
# e5's ``passage: ``/``query: ``. Empty by default; set only for a model that requires a prefix.
_DEFAULT_PREFIX = ""


class SentenceTransformersEmbedder(AdapterBase):
    """Learned Embedder Provider backed by a sentence-transformers checkpoint (CPU/GPU).

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass satisfying the
    :class:`~latence_core.capability.Embedder` Protocol (``dimension`` property +
    ``embed(texts) -> Iterator[list[float]]``). It composes the AdapterBase seams: :meth:`device`
    for cpu|cuda placement, :meth:`batched` for input-order-preserving streaming batches, and
    :meth:`guard` for native-exception → typed :class:`~latence_core.errors.ProviderError`
    mapping (no raw text / PII in the message).

    Config:
        model: str  — the sentence-transformers checkpoint id (default the GPU-tier
            ``ibm-granite/granite-embedding-311m-multilingual-r2``, Apache-2.0, 768-dim; the
            CPU-tier ``ibm-granite/granite-embedding-97m-multilingual-r2`` (Apache-2.0, 384-dim) is
            the one-line lighter option, and ``intfloat/multilingual-e5-small`` (MIT) /
            ``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0) are documented tiny fallbacks).
            A restricted-license checkpoint is a deliberate opt-in, never the default.
        dimension: int | None  — the expected embedding width. Default 768 (Granite r2 311m; the
            97m tier is 384). Validated against the loaded model's real width on first ``embed`` so
            a config that disagrees with the model fails loudly (a downstream vector DB needs a
            uniform column, ADR-0017).
        prefix: str  — an instruction prefix prepended to each text (default ``""`` — Granite r2
            needs none). Set it only for a model that requires one (e.g. an e5 fallback's
            ``"passage: "`` document-side prefix).
        attn_implementation / flash_attention / dtype: — the W8 perf knobs threaded to the backbone
            (``flash_attention_2`` + ``bfloat16`` on a GPU, graceful ``sdpa`` fallback; ``sdpa`` +
            ``float32`` on CPU) via ST ``model_kwargs``. See
            :func:`~latence_core.providers.perf.load_perf_kwargs`.
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
        """The declared descriptor (ADR-0036 §2) for the default Granite r2 311m checkpoint.

        ``compute="either"``: sentence-transformers runs in-process on a laptop CPU (the ADR-0007
        CPU path — first contact is ``pip install`` → real embeddings, no CUDA) and accelerates on
        a GPU when present, so the device seam routes it to ``auto``/``cpu``/``cuda`` rather than
        skipping it on a CPU-only host.

        License is **verified** per ADR-0012/0045, recorded evidence-bearingly in the structured
        fields (H-B1 — the claim is inseparable from its proof): weights and code are recorded
        **separately** (``license_code="Apache-2.0"`` for the ``sentence-transformers``/
        ``transformers`` library; ``license_weights="Apache-2.0"`` for the default checkpoint
        ``ibm-granite/granite-embedding-311m-multilingual-r2`` — quoted card tag
        ``license: apache-2.0`` — whose ModernBERT backbone is also Apache-2.0), with a
        ``license_source`` citing the HF model-card + the GitHub LICENSE where a human checked, and
        ``license_verified_on="2026-07-14"``. The headline ``license`` is the **weights** SPDX
        (``"Apache-2.0"``) and ``license_verified=True`` because BOTH sides are permissive.

        ``deterministic=False``: learned float inference (torch matmuls) is not guaranteed
        byte-identical across hardware/library/BLAS versions (unlike the pure-code hashing
        embedder), and the default GPU path adds **bf16 + flash_attention_2**, which is explicitly
        NOT bitwise-reproducible — so the profile declares it non-deterministic rather than
        overclaim reproducibility. The GPU FA2 stack is validated **structurally** (contract/shape/
        no-PII) by ``stack validate``, never by byte-identical determinism; the deterministic
        ``embedding.hashing`` reference stays the default on the unit-test/golden path.
        ``batch=True`` (ST batches its ``encode`` natively). ``memory_mb`` is the approximate
        resident footprint of the 311m checkpoint + its transformers stack (declared, not measured
        here — P3-F1 is descriptor-only; the 97m CPU tier is lighter). ``cost_per_1k=None``
        (in-process, no per-call vendor cost).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=1400,
            model_id=_DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=(
                "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2 "
                "(weights, apache-2.0); "
                "https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2 "
                "(CPU-tier weights, apache-2.0); "
                "https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE "
                "(library, Apache-2.0)"
            ),
            license_verified_on="2026-07-14",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            # W8: honest declared load-time perf posture (descriptor-only, P3-F1; not measured).
            perf_note="bf16+FA2-GPU (sdpa fallback) / fp32+sdpa-CPU, via ST model_kwargs",
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
        """Encode one batch through ST, (optionally) prefixed + L2-normalised, in input order.

        The configured ``prefix`` is prepended to each text — **empty by default** (Granite r2
        needs no instruction/query-document prefix), non-empty only for a fallback model that
        requires one (e.g. e5's ``"passage: "``). ``normalize_embeddings=True`` makes every vector
        unit-length (the cosine convention, ADR-0017 uniform column). The width is re-asserted per
        vector so a model that ever returns an off-width row is caught as a
        :class:`~latence_core.errors.ProviderError`, not silently exported.
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

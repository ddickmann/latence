"""Learned PII Redaction Provider over the fastino **GLiNER2** framework (ADR-0024/0036).

A **twin of** :class:`latence_pii_gliner.provider.GlinerPIIRedactor`: it reuses ALL the shared
redaction machinery (the :class:`~latence_core.redaction_policy.RedactionPolicy`, masking /
placeholder / action logic, :class:`~latence_core.contracts.PIISpan` /
:class:`~latence_core.contracts.RedactionRecord` contracts, offset handling, the
``PARSE_ERROR`` pass-through, and the ``Classification.sensitivity``-keyed policy) and swaps
ONLY the detection call — from the ``gliner`` library to the **``gliner2``** framework, a
*different* Python package with a *different* API.

The ``gliner2`` framework wraps an mdeberta-v3 encoder and exposes a single
``extract_entities`` call that returns a **per-label dict of dict-rows** (NOT the flat list of
dicts the ``gliner`` library returns). The REAL API (introspected + run on the pod, W3):

.. code-block:: python

    from gliner2 import GLiNER2
    # W9: gliner2 accepts NATIVE quantize (fp16) + compile (torch.compile + fused kernels) kwargs.
    model = GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi",
                                    quantize=True, compile=True)  # 2.83× at byte-identical PII
    out = model.extract_entities(text, entity_types, threshold=0.5,
                                 include_spans=True, include_confidence=True)
    # -> {"entities": {"<label>": [{"text": str, "confidence": float,
    #                               "start": int, "end": int}, ...], ...}}

Offsets are document-char ``start``/``end`` (half-open), exact — they map directly to
``PIISpan(char_start=start, char_end=end, pii_type=label, confidence=confidence)``.

It fulfils the same :class:`~latence_core.capability.PIIDetector` seam as the in-core
``redaction.hybrid_rule`` and the twin ``redaction.gliner_pii``, so it is a drop-in behind
Redaction — the Pipeline swaps ``provider: redaction.hybrid_rule`` (or ``redaction.gliner_pii``)
for ``provider: redaction.gliner2`` with no code change, same ``RedactionRecord`` contract. The
**raw PII value is never persisted** on a :class:`~latence_core.contracts.PIISpan` (S6 AC): the
surface string is used only transiently to derive a deterministic surrogate.

Licensing (verified 2026-07-09):
    * ``gliner2`` library (PyPI + github.com/fastino-ai/GLiNER2): **Apache-2.0** (code) —
      LICENSE file + README badge + PyPI ``OSI Approved :: Apache Software License`` classifier.
    * Default checkpoint ``fastino/GLiNER2-Guardrails-PII-Multi``: **Apache-2.0** (weights, per
      the HF model card; NOT gated / freely downloadable).
    Both weights AND code are permissive; nothing downloads unless a Pipeline actually
    configures this Provider.

The ``gliner2`` import is deferred to first use (ADR-0016) so importing this module (for Provider
discovery) never requires torch, and a missing optional dep raises a clear error only when the
Provider is run. It builds on :class:`~latence_core.providers.adapter.AdapterBase` for the
device seam (``compute="either"``: mdeberta runs on CPU or GPU, skip-with-flag routed by the
device seam) and the typed-error guard — a native ``gliner2`` fault becomes a typed
:class:`~latence_core.errors.ProviderError` carrying NO record content / PII (C2/C6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.contracts import RiskMarker
from latence_core.errors import ConfigError
from latence_core.induced import (
    LABEL_FLOOR_CONFIG_KEY,
    LabelFloor,
    LabelSlot,
    effective_labels,
)
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.device import cuda_available
from latence_core.providers.perf import (
    empty_cuda_cache,
    ensure_gliner2_on_cuda,
    inference_context,
    is_compile_backend_failure,
    is_cuda_oom,
    iter_token_budget_microbatches,
    load_gliner2_optimized,
    neutralize_torch_jit,
    resolve_gliner2_knob,
    resolve_max_len,
    resolve_microbatch_bounds,
)
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    GUARD_SAFETY_FLOOR,
    ChunkRedactionPlan,
    RedactionPolicy,
    finalize_redacted_chunk,
    guard_category,
    parse_action,
    plan_chunk_redaction,
)

from latence_pii_gliner2._compat import (
    require_batched_entity_api,
    require_supported_gliner2,
)
from latence_pii_gliner2._tokenizer_compat import coerce_list_extra_special_tokens

_log = logging.getLogger("latence.provider.gliner2_pii")

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord


@dataclass(frozen=True)
class _PendingChunk:
    """A chunk's model-independent redaction context, resolved before the batched call (W7/W16).

    The :class:`~latence_core.redaction_policy.ChunkRedactionPlan` (policy action/allow-list, the
    skip/near-empty decision, the chunk offset/page maps) and the effective label sets are per-chunk
    and model-independent, so they are computed once up front; a flush then groups the
    detect-needing chunks by their label sets and issues one ``batch_extract_entities`` call per
    group, mapping each per-text result back to its chunk by identity, keeping emission in
    input-chunk order and the output record-for-record identical to the per-chunk path.

    ``labels`` is the single list handed to the model: the effective PII types FOLLOWED BY the
    effective guardrail categories (T2), so one forward covers both. ``guard`` is the casefolded
    subset of those labels that must be read back as GUARDRAIL hits (a
    :class:`~latence_core.contracts.RiskMarker` on the chunk) rather than PII spans to mask — a
    label in both sets counts as a guard label, because dropping a security screen is the worse
    error. ``guard`` is empty when guardrailing is off, and then ``labels`` is exactly the PII set.
    """

    chunk: ChunkRecord
    text: str
    plan: ChunkRedactionPlan
    labels: tuple[str, ...]
    guard: frozenset[str] = frozenset()

# Apache-2.0 weights (verified 2026-07-09, HF model card, not gated). Multilingual PII NER over
# an mdeberta-v3 encoder via the fastino GLiNER2 framework.
_DEFAULT_MODEL = "fastino/GLiNER2-Guardrails-PII-Multi"

# W9-scaling: torch.compile is OPT-IN for the gliner2 engine (see __init__) — gliner2's
# ``dynamic=True`` compile recompile-thrashes on varied-length production chunks. Same default, same
# reason, as the twin ``fused_entity_relation.gliner2`` extractor.
_COMPILE_DEFAULT = False

# T2: this Provider's config ``pii_labels`` are a UNIVERSAL SAFETY FLOOR, not a relevance floor —
# the one place in the redesign where the per-chunk rule does NOT apply to the config list. Induced
# ``pii_types`` are scoped per chunk (they are relevance labels: what this chunk is likely to
# contain); the configured floor is stamped onto EVERY chunk regardless of what that chunk induced,
# because a leaked card/IBAN/SSN in a chunk whose induced types happen to omit finance is exactly
# the leak a per-chunk-only screen would ship. The floor is ~9 fixed labels (the wizard's
# ADR-0044 financial set), so PII label counts stay small either way and the zero-shot/label-count
# argument that justifies PER_CHUNK for entity/relation extraction does not buy enough to trade a
# security invariant for. Deliberately NOT configurable: see ``_reject_floor_subsetting``.
_PII_LABEL_FLOOR = LabelFloor.UNIVERSAL


class Gliner2PIIRedactor(AdapterBase):
    """Learned PIIDetector Provider backed by the fastino GLiNER2 framework (mdeberta-v3).

    A near-clone of :class:`latence_pii_gliner.provider.GlinerPIIRedactor` that reuses the shared
    redaction machinery and changes ONLY detection (the ``gliner2`` framework instead of the
    ``gliner`` library).

    Config:
        pii_labels: list[str]  — REQUIRED. The PII entity types to detect (e.g.
            ``["person", "email", "phone number"]``). GLiNER2 works best with lower-case,
            natural-language type names.
        guardrails: bool  — screen each chunk for the LEARNED guardrail categories the
            guardrails+PII checkpoint also covers (T2, default ``False`` ⇒ off, byte-identical to a
            stack without it). ON: every chunk is screened for the UNIVERSAL guard floor
            (:data:`~latence_core.redaction_policy.GUARD_SAFETY_FLOOR`) UNIONed with
            ``guard_labels`` and with the categories induced for THAT chunk
            (``induced_labels.guard_types``). The
            floor is never subsetted per chunk — same rule, same reason, as the PII floor. A hit is
            recorded as a :class:`~latence_core.contracts.RiskMarker` on the chunk (the
            ``screening.content_keyword`` vocabulary, so consumers filter one category set), NOT
            masked: guard categories mark risky content, they are not identifiers to redact.
            Orthogonal to the deterministic keyword ``content_screen`` Stage, which is a separate
            pre-filter and is untouched.
        guard_labels: list[str]  — extra learned guardrail categories on top of the universal floor
            (requires ``guardrails: true``; supplying them without it is a ConfigError, never a
            silent no-op). A label that collides with a configured ``pii_labels`` entry is a
            ConfigError.
        label_floor: str  — accepted ONLY as ``universal`` (T2). Unlike the sibling extractor, this
            Provider's config labels are SAFETY floors, not relevance labels, so they are applied to
            every chunk and the knob cannot subset them; any other value is a ConfigError rather
            than a silently ignored key.
        model: str  — the checkpoint id (default ``fastino/GLiNER2-Guardrails-PII-Multi``,
            Apache-2.0).
        threshold: float  — score threshold; spans below it are dropped (default 0.5).
        default_action: str  — action when a policy names none: ``mask`` (default),
            ``replace``, ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map (identical shape to the
            twin Provider): per ``Classification.sensitivity`` value an ``action`` / ``types``
            allow-list / ``skip`` flag. Keys off the document's sensitivity (S6 AC).
        salt: str  — salt mixed into ``replace``/``hash`` surrogates (default 'latence').
        min_chars: int  — a document whose assembled text is shorter is emitted clean
            (default 1).
        max_len: int  — the mdeberta-v3 model token window (default 768), threaded as the native
            ``max_len`` of ``batch_extract_entities`` (W12). A document that tokenizes past it is
            SILENTLY truncated by the model (tail PII spans lost, offsets past the window
            unreliable); this bounds the window explicitly. CAVEAT: the chunker counts in a
            DIFFERENT tokenizer, ~1.2× fewer tokens than mdeberta — keep the chunk cap ≤ ~640 for
            this 768 window; this guard is the hard backstop.
        batch_size: int  — the flush cadence: how many documents are BUFFERED per flush (default
            from AdapterBase; the P8 autotuner overrides per-device). It is NOT the model-forward
            tensor size — a flush's texts are split into token-budget micro-batches
            (``microbatch_max_texts`` / ``microbatch_token_budget``, below) before the native call,
            so a large ``batch_size`` shapes the flush/GC granularity, never a single padded
            activation (dogfood-2b-process). The emitted RedactionRecords (PIISpans/masking/order)
            are record-for-record identical for any positive batch size (W7). Must be ``>= 1``.
        microbatch_max_texts: int  — the hard cap on texts packed into ONE
            ``batch_extract_entities`` forward (default 4, the pod-safe count). gliner2 pads a
            micro-batch to its longest sequence, so this bounds the padded forward to
            ``max_texts × max_len`` positions regardless of ``batch_size`` — the PRIMARY OOM guard.
        microbatch_token_budget: int  — the token budget (summed per-text token lengths, each
            capped at ``max_len``) for ONE forward (default 3072 = 4 × 768). The SECONDARY bound,
            tighter than ``max_texts`` when ``max_len`` is raised. Both default safe on a modest
            GPU; raise them only on a card with more VRAM.
        quantize: bool | 'auto'  — load fp16 via the gliner2 native ``quantize`` kwarg (W9,
            default ``auto`` → True on cuda / False on cpu). Pod-measured 2.83× at byte-identical
            PII. Guarded: a quantize fault degrades to a plain load, never a crash.
        compile: bool | 'auto'  — ``torch.compile`` + fused GPU kernels via the gliner2 native
            ``compile`` kwarg (W9). **OPT-IN: defaulted OFF even on cuda** (W9-scaling), because
            gliner2 compiles with ``dynamic=True`` and recompile-thrashes on varied-length
            production chunks (pod-measured: worker at ~290% CPU, GPU 0%, no progress) — the same
            default the twin ``fused_entity_relation.gliner2`` extractor forces. Set it explicitly
            to enable it (worth it only for a homogeneous / uniform-length workload). Guarded like
            ``quantize``: a compile/warmup fault degrades to a plain load.
        device: str  — ``auto`` (default) / ``cpu`` / ``cuda``; routed by the device seam.

    The model is loaded lazily and cached on first use, so constructing the Provider never
    downloads weights.

    Induced labels (W12, ADR-0038; T2 floor policy): the detection label set is
    ``effective_labels(config ∪ induced)`` per CHUNK — the SAME seam the twin
    ``redaction.gliner_pii`` and the extractors use — so a ``pii_type`` an upstream
    ``schema_induction`` Stage induced (e.g. an ``IBAN`` / ``tax id``) is actually redacted. The
    per-chunk redesign scopes the INDUCED half to the chunk that induced it, and deliberately does
    NOT scope the config half: the ``pii_labels`` floor and the guardrail floor are
    :data:`~latence_core.induced.LabelFloor.UNIVERSAL` safety floors applied to every chunk, because
    a leaked card in a chunk whose induced types omit finance is precisely the leak a per-chunk-only
    screen would ship. That asymmetry — relevance labels scoped per chunk, safety floors universal —
    is the T2 rule, and it is enforced here (``label_floor`` cannot subset it) and again,
    model-independently, by the deterministic ADR-0044 regex floor at ``finalize_redacted_chunk``.
    Absent induction the effective set is exactly the config labels, byte-identical to before.

    Batching (W7, ADR-0036 §1 / ADR-0033): GLiNER2 (mdeberta-v3) is an ENCODER, so a per-document
    ``extract_entities`` loop under-feeds the GPU. This Provider buffers up to ``batch_size``
    documents and, on flush, GROUPS the detect-needing docs by their effective label set (one label
    set per ``batch_extract_entities`` call), then splits each group's texts into **token-budget
    micro-batches** and runs ONE ``model.batch_extract_entities(texts, labels, threshold=…,
    include_spans=True, include_confidence=True, max_len=…)`` forward pass PER MICRO-BATCH (the
    batched entry point — ``hasattr``-guarded here and pod-Gate-0 asserted), concatenating each
    micro-batch's per-text results IN ORDER and mapping each back to its document **by identity**.

    dogfood-2b-process: the micro-batch step is the OOM cure. gliner2 pads a group of N texts into
    ONE tensor and scores every span position, so the peak activation scales with (N × seq_len) —
    a whole ``batch_size`` group of long chunks demands a single multi-GiB allocation (measured: 32
    chunks × ~768 tokens → ~+17 GiB spike → OOM on a 32 GB Blackwell; 4 chunks peaked at 21.6 GiB
    and passed). Splitting each forward by a conservative token budget + text-count cap
    (:func:`~latence_core.providers.perf.iter_token_budget_microbatches`) bounds the padded
    activation for ANY ``batch_size``/``max_len``, PROACTIVELY — the reactive
    :meth:`_batch_extract` OOM split-on-fault stays as a last-resort net that now rarely fires.
    Because the output is batch-size-INVARIANT (W7 — same spans/masking/order for any positive
    batch), micro-batching is byte-identical.

    Documents that a policy SKIPS or that are near-empty need no model call and are emitted clean,
    in input order. The per-document policy resolution, span building, masking, and Provenance carry
    are UNCHANGED; the in-flight buffer is bounded to ``batch_size`` (never the whole corpus,
    ADR-0033).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default GLiNER2-PII checkpoint.

        ``compute="either"``: the GLiNER2 mdeberta-v3 model runs in-process on a laptop CPU
        (ADR-0007) and accelerates on a GPU when present, so the device seam routes it rather
        than skipping it on a CPU-only host.

        License **verified** (recorded 2026-07-09): the ``gliner2`` library is Apache-2.0 (code:
        github.com/fastino-ai/GLiNER2 LICENSE + PyPI classifier) and the default checkpoint
        ``fastino/GLiNER2-Guardrails-PII-Multi`` is Apache-2.0 (weights, per the HF model card,
        not gated) — both permissive, so ``license_verified=True``.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical across
        hardware/library versions, so the profile does not overclaim reproducibility (unlike the
        pure-code hybrid-rule Redactor). ``memory_mb`` is the approximate resident footprint of
        the checkpoint + transformers stack (declared, not measured).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=2048,
            model_id=_DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=(
                "https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi (weights, "
                "Apache-2.0, not gated); "
                "https://github.com/fastino-ai/GLiNER2 (library, Apache-2.0 — LICENSE + PyPI "
                "'OSI Approved :: Apache Software License' classifier)"
            ),
            license_verified_on="2026-07-09",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note=(
                "gliner2 native quantize (fp16) on cuda, off on cpu (W9; pod-measured 2.83× PII at "
                "byte-identical output; guarded → plain load on any fault). compile "
                "(torch.compile + fused GPU kernels) is OPT-IN, defaulted OFF even on cuda "
                "(W9-scaling: gliner2's dynamic=True compile recompile-thrashes on varied-length "
                "production chunks)"
            ),
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        pii_labels = config.get("pii_labels")
        if not pii_labels or not isinstance(pii_labels, list):
            msg = (
                "Gliner2PIIRedactor requires a non-empty 'pii_labels' list in its config "
                "(the PII entity type set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        # AdapterBase resolves the device seam (compute="either") against the declared profile.
        super().__init__(config)
        # W9-scaling (audit R4): default torch.compile OFF for this Provider (opt-in only), exactly
        # as the twin ``fused_entity_relation.gliner2`` extractor does. gliner2 compiles with
        # ``dynamic=True``, which recompile-thrashes on the varied-length production chunks this
        # redactor streams through ``batch_extract_entities`` — pod-measured to pin the worker at
        # ~290% CPU with the GPU at 0% and no progress. Without this the generic
        # ``resolve_gliner2_knob`` default (``auto`` → True on cuda) silently turned compile ON for
        # every GPU stack the wizard emits (it emits ``device: cuda`` and no ``compile`` key), and
        # the ``_reload_uncompiled`` latch cannot catch it: the thrash is silent slowness, not a
        # ``BackendCompilerFailed``. ``quantize`` is untouched (the fp16 win stays on for cuda). An
        # operator who sets ``compile`` explicitly still gets it. Injected into ``self._config`` —
        # the single source of truth read by BOTH the eager validation below and the lazy load.
        if "compile" not in self._config:
            self._config["compile"] = _COMPILE_DEFAULT
        self._labels = [str(label) for label in pii_labels]
        # T2: the PII floor is universal and non-negotiable — reject any attempt to subset it.
        self._reject_floor_subsetting(config)
        # T2 guardrails: the learned guard categories this Provider's guardrails+PII checkpoint is
        # also asked about. OFF by default, so an existing stack is byte-identical (no extra labels,
        # no risk markers). ON ⇒ every chunk carries the UNIVERSAL guard floor, plus any operator
        # categories, plus the categories induced FOR THAT CHUNK.
        self._guardrails, self._guard_labels = self._resolve_guard_config(config)
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        # W12 model-boundary guard: bound the mdeberta token window (default 768). gliner2's
        # batch_extract_entities takes a native ``max_len``; past it the model silently truncates
        # (tail PII spans lost, offsets past the window unreliable). Explicit + configurable.
        self._max_len = resolve_max_len(config)
        # Validate the batch_size knob eagerly (fail-fast at construction like the twin providers)
        # even though the batched call reads it lazily via AdapterBase.batch_size() (W7).
        self.batch_size()
        # dogfood-2b-process: resolve the token-budget micro-batch bounds that DECOUPLE the model
        # forward tensor from the progress-cadence ``batch_size``. ``batch_size`` buffers docs for
        # the flush; each flush then splits its texts into micro-batches bounded by BOTH a text
        # count and a token budget (perf-seam constants, pod-safe defaults) so a single
        # ``batch_extract_entities`` call can never demand an OOM-sized padded activation, for ANY
        # configured ``batch_size``/``max_len``. Validated eagerly (fail-fast) like the other knobs.
        try:
            self._microbatch_max_texts, self._microbatch_token_budget = resolve_microbatch_bounds(
                self._config
            )
        except ValueError as exc:
            msg = f"{self.provider_name}: {exc}"
            raise ConfigError(msg) from exc
        # Validate the W9 quantize/compile knobs eagerly too (fail-fast): an invalid value is an
        # operator ConfigError at construction, not a native fault buried in the lazy load.
        self._validate_perf_knobs()
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        # The framework owns policy + masking (shared with the in-core rule Provider and the
        # twin gliner Provider); this Provider owns only detection — the PIIDetector seam.
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        self._model: Any = None
        # Set once the inference-time compile guard has re-loaded the uncompiled model (W9), so the
        # fallback runs at most once per provider instance.
        self._compile_disabled: bool = False

    @property
    def provider_name(self) -> str:
        """The registered entry-point name, used only to tag typed errors (never carries PII)."""
        return "redaction.gliner2"

    def _reject_floor_subsetting(self, config: dict[str, Any]) -> None:
        """Refuse a ``label_floor`` that would scope this Redactor's SAFETY floors per chunk (T2).

        The sibling extractor exposes ``label_floor`` because entity/relation labels are relevance
        labels. Here they are not: the configured ``pii_labels`` (the ADR-0044 financial set the
        wizard emits) and the universal guard floor are security invariants that apply to every
        chunk. Rather than accept the key and quietly ignore it — which would leave an operator
        believing they had narrowed the PII screen when they had not, or vice versa — an attempt to
        set it to anything but ``universal`` is a loud :class:`ConfigError` (ADR-0034). The
        no-op ``universal`` value is accepted so a shared/templated config that states the policy
        explicitly still loads.
        """
        raw = config.get(LABEL_FLOOR_CONFIG_KEY)
        if raw is None or raw == LabelFloor.UNIVERSAL.value:
            return
        msg = (
            f"{self.provider_name}: {LABEL_FLOOR_CONFIG_KEY}={raw!r} is not accepted — a "
            "Redactor's config PII labels and its guardrail floor are UNIVERSAL safety floors "
            "applied to every chunk, not per-chunk relevance labels (a leaked card in an "
            "'irrelevant' chunk must still be masked). Only the INDUCED pii/guard types are "
            f"scoped per chunk. Omit the key, or set {LABEL_FLOOR_CONFIG_KEY}: "
            f"{LabelFloor.UNIVERSAL.value!r}."
        )
        raise ConfigError(msg)

    def _resolve_guard_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Resolve ``guardrails`` / ``guard_labels`` → (enabled, universal guard floor + extras).

        Two knobs rather than one, so neither can silently do nothing: ``guardrails`` is the master
        switch (default OFF ⇒ no guard labels reach the model and no risk marker is emitted, so an
        existing stack is byte-identical), and ``guard_labels`` adds operator categories on top.
        Supplying ``guard_labels`` WITHOUT ``guardrails: true`` is a :class:`ConfigError`, never a
        silently ignored list (ADR-0034).

        When enabled the returned list ALWAYS starts with the universal
        :data:`~latence_core.redaction_policy.GUARD_SAFETY_FLOOR` categories — the guard twin of the
        PII floor, applied to every chunk and not subsettable. A guard label that collides with a
        configured PII label is a ConfigError too: the same string cannot both be masked as PII and
        recorded as a guardrail hit, and resolving that silently would make one of the two controls
        vanish.
        """
        enabled = config.get("guardrails", False)
        if not isinstance(enabled, bool):
            msg = f"{self.provider_name}: guardrails must be a bool (got {enabled!r})"
            raise ConfigError(msg)
        extra_raw = config.get("guard_labels", [])
        if not isinstance(extra_raw, list):
            msg = f"{self.provider_name}: guard_labels must be a list (got {extra_raw!r})"
            raise ConfigError(msg)
        extra = [str(label) for label in extra_raw if str(label).strip()]
        if extra and not enabled:
            msg = (
                f"{self.provider_name}: guard_labels was supplied but guardrails is off, so the "
                "categories would never be screened for. Set guardrails: true (or drop "
                "guard_labels)."
            )
            raise ConfigError(msg)
        if not enabled:
            return False, []
        pii_keys = {label.casefold() for label in self._labels}
        collisions = sorted({label for label in extra if label.casefold() in pii_keys})
        if collisions:
            msg = (
                f"{self.provider_name}: guard_labels {collisions} are also configured as "
                "pii_labels; a category cannot be both masked as PII and recorded as a guardrail "
                "hit. Rename one of them."
            )
            raise ConfigError(msg)
        return True, [*GUARD_SAFETY_FLOOR, *extra]

    def _validate_perf_knobs(self) -> None:
        """Resolve the W9 ``quantize``/``compile`` knobs eagerly → ConfigError on a bad value.

        Resolving them at construction (against the resolved device) turns an invalid
        ``quantize``/``compile`` config into an operator :class:`ConfigError` up front, rather than
        a native fault surfaced only at the lazy load — the same fail-fast discipline as
        ``batch_size``. The resolved values are recomputed at load time (device-stable), so this is
        purely validation.
        """
        try:
            resolve_gliner2_knob("quantize", self.device.device, self._config)
            resolve_gliner2_knob("compile", self.device.device, self._config)
        except ValueError as exc:
            msg = f"{self.provider_name}: {exc}"
            raise ConfigError(msg) from exc

    # -- lazy model load ----------------------------------------------------

    def _gliner2_cls(self) -> Any:
        """The ``gliner2.GLiNER2`` class, after the installed version is checked (#192).

        The version gate runs HERE — before ``from_pretrained``, i.e. before any weight download —
        so an unsupported ``gliner2`` is a typed, actionable
        :class:`~latence_core.errors.ProviderError` naming the supported range, never the native
        ``AttributeError``/``TypeError`` the mismatched API would raise later (ADR-0034).

        W9-jit (#190, audit R1): the TorchScript JIT is neutralized FIRST, before the ``gliner2``
        import — that import transitively pulls in transformers' deberta_v2 modeling module, whose
        relative-position helpers are compiled by ``@torch.jit.script`` **AT IMPORT**. Neutralizing
        later (inside ``load_gliner2_optimized``, entered only once this class is resolved) cannot
        un-script them: the env switch is read by torch only at ITS import, and
        ``jit._state.disable()`` affects only FUTURE ``script()`` calls — so the scripted helpers
        would still die in nvrtc (``libnvrtc-builtins.so.13.0``) on the first forward on a
        Blackwell/CUDA-13 box. Throughput-only, byte-identical output.
        """
        neutralize_torch_jit(provider_name=self.provider_name)
        try:
            import gliner2
            from gliner2 import GLiNER2
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner2' package is required to run Gliner2PIIRedactor. "
                "Install it with: pip install latence-pii-gliner2"
            )
            raise ImportError(msg) from exc
        require_supported_gliner2(gliner2, provider_name=self.provider_name)
        return GLiNER2

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        # W9 perf seam: the ``gliner2`` framework (unlike plain ``gliner`` 0.2.27, and superseding
        # the W7 batching that REGRESSED redact via padding waste) exposes NATIVE ``quantize``
        # (fp16) + ``compile`` (torch.compile + fused GPU kernels, "no extra dependencies required")
        # kwargs on ``from_pretrained`` — pod-measured **2.83× at byte-identical PII**. Both default
        # True on cuda / False on cpu (or the ``quantize``/``compile`` config override), resolved
        # once here.
        # The optimized load is GUARDED: any quantize/compile fault degrades to a plain load
        # (throughput only, never a crash) — so a host whose torch/CUDA rejects fp16 or a compile
        # warmup fails still loads the identical model. An invalid ``quantize``/``compile`` value is
        # a ConfigError surfaced before the load (never a silent wrong knob).
        # The import is resolved OUTSIDE the guard so a missing-dep ImportError surfaces as the
        # clear "pip install latence-pii-gliner2" error, not a wrapped ProviderError.
        gliner2_cls = self._gliner2_cls()
        # dogfood-2b-process: the default PII checkpoint's tokenizer_config.json stores
        # ``extra_special_tokens`` as a LIST, which transformers >= 4.57 (pulled transitively via
        # gliner2[local], within this Provider's allowed range) rejects at load with a native
        # ``AttributeError: 'list' object has no attribute 'keys'`` — killing the whole pipeline
        # before any run dir is written. Coerce it to the dict shape 4.57 requires BEFORE the load
        # (guarded, best-effort, byte-identical tokenization). Mirrors the glinker backend's
        # _patch_tokenizer_config precedent; a dict-shaped checkpoint is an idempotent no-op.
        coerce_list_extra_special_tokens(self._model_id, provider_name=self.provider_name)
        with self.guard("load_model"):
            self._model = load_gliner2_optimized(
                gliner2_cls.from_pretrained,
                self._model_id,
                self.device.device,
                self._config,
                provider_name=self.provider_name,
            )
        # W9-scaling: ``load_gliner2_optimized`` uses ``device`` ONLY to resolve the quantize/
        # compile knobs — it performs NO placement. So when a GPU is present, ensure the fp16 model
        # actually on cuda and LOG its device: the defence against the #191 fp16-eager GPU-idle hang
        # (a fp16-converted model that silently stayed on CPU → a CPU-bound spin at 0% GPU, correct
        # output, no warning). Best-effort + guarded (never crashes); a model already placed
        # correctly makes the move a harmless no-op. Mirrors the twin fused_entity_relation.gliner2.
        ensure_gliner2_on_cuda(
            self._model,
            cuda=cuda_available(),
            provider_name=self.provider_name,
        )
        # #192: the loaded object must expose the batched-entity API this Provider drives. Checked
        # at LOAD (not at the first flush) so an API mismatch is a typed, actionable ProviderError
        # up front — the field failure was this surface raising a raw native AttributeError.
        require_batched_entity_api(self._model, provider_name=self.provider_name)
        return self._model

    def _reload_uncompiled(self) -> Any:
        """Re-load the model with ``compile=False`` (quantize kept) after an inference-time compile
        fault — the second half of the W9 compile guard (:func:`is_compile_backend_failure`).

        ``torch.compile`` compiles lazily on the first forward, so a backend that cannot compile the
        model fails INSIDE ``batch_extract_entities`` (a ``BackendCompilerFailed``), which the
        load-time guard cannot catch. On the first such fault the provider re-loads ONCE with
        ``compile`` forced off (``quantize`` unchanged — fp16 alone is fine) and uses the uncompiled
        model for the rest of the run, so a bad compile costs only throughput, never a crash.
        Idempotent: sets ``_compile_disabled`` so the fallback runs at most once.
        """
        self._compile_disabled = True
        cfg = {**self._config, "compile": False}
        _log.warning(
            "redaction.gliner2: torch.compile failed at inference; re-loading the uncompiled "
            "model (quantize kept) — throughput only, never a crash"
        )
        gliner2_cls = self._gliner2_cls()
        # Idempotent no-op after _load_model already patched the cached config (dogfood-2b); kept
        # for symmetry so this reload path is never coupled to a prior patch having run.
        coerce_list_extra_special_tokens(self._model_id, provider_name=self.provider_name)
        with self.guard("load_model"):
            self._model = load_gliner2_optimized(
                gliner2_cls.from_pretrained,
                self._model_id,
                self.device.device,
                cfg,
                provider_name=self.provider_name,
            )
        # Keep the uncompiled reload on cuda too (W9-scaling) — this model serves the REST of the
        # run, so skipping placement here would leave it fp16-on-CPU. Placement is idempotent.
        ensure_gliner2_on_cuda(
            self._model,
            cuda=cuda_available(),
            provider_name=self.provider_name,
        )
        return self._model

    def _extract_once(self, model: Any, texts: list[str], labels: list[str]) -> list[Any]:
        """A single ``batch_extract_entities`` forward under :meth:`AdapterBase.guard` (typed).

        ``results[i]`` is the per-label dict for ``texts[i]``. The guard maps any native fault to a
        typed :class:`~latence_core.errors.ProviderError` carrying no record content / PII.

        The forward runs INSIDE :func:`~latence_core.providers.perf.inference_context`
        (``torch.inference_mode()``): a redaction-only encoder never backpropagates, so building an
        autograd graph and pinning its activations alive would only ACCUMULATE VRAM run-over-run to
        a CUDA OOM (dogfood-2b-process). Under inference mode no graph is built and each forward's
        activations free immediately — same spans, bounded memory (torch-free-safe offline).
        """
        with self.guard("batch_extract_entities"), inference_context():
            return list(
                model.batch_extract_entities(
                    texts,
                    labels,
                    threshold=self._threshold,
                    include_spans=True,
                    include_confidence=True,
                    max_len=self._max_len,
                )
            )

    def _batch_extract(self, model: Any, texts: list[str], labels: list[str]) -> list[Any]:
        """One batched forward with the W9 compile guard AND a CUDA-OOM micro-batch backoff.

        ``labels`` is this group's effective label set (``config ∪ induced``, W12).

        Two inference-time faults are recovered here so ``latence process`` does not die mid-run and
        forfeit the whole run dir (the redaction floor, ADR-0044, must hold — no chunk is ever
        dropped or emitted un-scanned):

        * **compile fault** — ``torch.compile`` compiles lazily on the first forward, so a backend
          that cannot compile raises a ``BackendCompilerFailed`` (dynamo/inductor) HERE, not at load
          (:func:`~latence_core.providers.perf.is_compile_backend_failure`). The provider re-loads
          the uncompiled model ONCE (:meth:`_reload_uncompiled`; quantize kept) and retries.

        * **CUDA OOM** (dogfood-2b-process) — GLiNER2's span-scoring activation for a whole group of
          long chunks can demand a single multi-GiB allocation the GPU cannot satisfy (pod: 6.28 GiB
          on a 32 GB Blackwell already 27 GB resident), surfacing a native ``OutOfMemoryError``
          (:func:`~latence_core.providers.perf.is_cuda_oom`). Because the emitted RedactionRecords
          are **batch-size-invariant** (W7 — same spans/masking/order for any positive batch size),
          the group is split in half and each half retried recursively after a best-effort cache
          release; the activation scales with the batch dimension, so halving shrinks the peak
          allocation until it fits. A batch of ONE that still OOMs is genuinely unrecoverable on
          this hardware — it surfaces as a typed ``ProviderError`` (no fake green, no dropped
          chunk).

        Any OTHER native fault is surfaced through the guard as before (typed, no PII).
        """
        try:
            return self._extract_once(model, texts, labels)
        except Exception as exc:  # noqa: BLE001 - intercept recoverable faults before surfacing
            # ``guard`` wraps the native fault as ProviderError; both predicates walk the cause
            # chain to find the original BackendCompilerFailed / OutOfMemoryError underneath.
            if not self._compile_disabled and is_compile_backend_failure(exc):
                model = self._reload_uncompiled()
                return self._extract_once(model, texts, labels)
            if is_cuda_oom(exc) and len(texts) > 1:
                # Split the group and retry smaller — byte-identical output, lower peak VRAM. Free
                # the OOM'd batch's fragmented blocks first so the smaller retry allocates cleanly.
                empty_cuda_cache(provider_name=self.provider_name)
                mid = len(texts) // 2
                # Re-fetch the current model each half so a compile-reload inside one half is picked
                # up by the next (self._model is the source of truth after _reload_uncompiled).
                left = self._batch_extract(self._load_model(), texts[:mid], labels)
                right = self._batch_extract(self._load_model(), texts[mid:], labels)
                return left + right
            raise

    # -- redaction ----------------------------------------------------------

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Detect PII per chunk with the model; yield the SAME chunks with PII masked (W16).

        Each chunk's ``content`` (≤ the model window) is scanned in full — no truncation — and the
        chunk is re-emitted with ``masked_content`` + ``pii_spans`` set (``content`` UNCHANGED).
        Chunks are threaded 1:1 in first-seen order; a chunk-local span resolves to its ORIGINAL
        source page through the chunk's offset_map + page_map (ADR-0031/0042).
        """
        buffer: list[_PendingChunk] = []

        def flush() -> Iterator[ChunkRecord]:
            yield from self._flush_buffer(buffer)
            buffer.clear()

        for chunk in chunks:
            buffer.append(self._prepare(chunk))
            if len(buffer) >= self.batch_size():
                yield from flush()
        yield from flush()

    def _prepare(self, chunk: ChunkRecord) -> _PendingChunk:
        """Resolve everything a chunk needs BEFORE the (possibly batched) model call (W7/W16).

        The :class:`~latence_core.redaction_policy.ChunkRedactionPlan` (policy resolution, the
        near-empty/skip decision, the chunk offset/page maps) and the effective label sets are
        per-chunk and model-independent, so they are computed here up front; the flush then issues
        one batched model call per label group over the detect-needing chunks.

        The T2 rule, both halves, in one place:

        * **PII** — ``config floor ∪ induced[pii_types]`` under
          :data:`_PII_LABEL_FLOOR` (``UNIVERSAL``). The induced half is per chunk; the configured
          floor is NOT subsetted, so a chunk whose induction never mentioned finance is still
          screened for the whole ADR-0044 set. (The deterministic regex floor at
          ``finalize_redacted_chunk`` is the second, model-independent layer of the same
          guarantee.)
        * **GUARDRAILS** — ``universal guard floor ∪ config guard_labels ∪ induced[guard_types]``,
          same policy, same reason: the induced categories are this chunk's relevance labels, the
          floor is the security invariant. Empty when guardrailing is off.
        """
        plan = plan_chunk_redaction(
            chunk,
            self._policy,
            detector_types=set(self._labels),
            sensitive_set=set(),
            min_chars=self._min_chars,
        )
        # W12 label-source seam (ADR-0038): UNION the config PII floor with any pii_types the
        # OPTIONAL schema_induction Stage induced for this chunk (``induced_labels``), deduped +
        # order-stable — the SAME seam the twin ``redaction.gliner_pii`` and the extractors use, so
        # an induced IBAN/tax-id label is actually redacted. Absent (the default, every existing
        # stack) these are exactly the config labels — byte-identical. The config floor stays
        # mandatory (``pii_labels`` is validated non-empty in __init__); induction only EXTENDS it.
        pii_labels = effective_labels(
            self._labels, chunk.induced_labels, LabelSlot.PII, floor=_PII_LABEL_FLOOR
        )
        guard_labels = (
            effective_labels(
                self._guard_labels, chunk.induced_labels, LabelSlot.GUARD, floor=_PII_LABEL_FLOOR
            )
            if self._guardrails
            else []
        )
        guard_keys = {label.casefold() for label in guard_labels}
        # ONE forward covers both: the guard categories ride the same batched call as the PII types
        # (the checkpoint is a guardrails+PII model). A label claimed by BOTH sets is read back as a
        # guard hit — dropping a security screen is worse than losing one mask, and the config-level
        # collision is already a ConfigError, so this only arbitrates INDUCED overlap.
        pii_keys = {label.casefold() for label in pii_labels}
        labels = [
            *pii_labels,
            *(label for label in guard_labels if label.casefold() not in pii_keys),
        ]
        return _PendingChunk(
            chunk=chunk,
            text=chunk.content,
            plan=plan,
            labels=tuple(labels),
            guard=frozenset(guard_keys),
        )

    def _flush_buffer(self, buffer: list[_PendingChunk]) -> Iterator[ChunkRecord]:
        """Batch the detect-needing chunks (grouped by label set), then emit in input order (W16).

        ``batch_extract_entities`` takes ONE label set per call, so the detect-needing chunks are
        grouped by their effective ``labels`` (``config ∪ induced``, W12) and each group goes
        through one forward pass — ``ceil`` per group, not one-per-chunk; each per-text result maps
        back to its chunk by identity. Absent induction every chunk shares the config label set → a
        single group → byte-identical to the pre-W12 single-call path. Chunks that skip / are
        near-empty need no model call. Emission is in the buffer's (input) order regardless of
        grouping, so the output is record-for-record identical to the per-chunk path (the
        equivalence invariant).
        """
        if not buffer:
            return
        raw_by_chunk: dict[int, Any] = {}
        detect = [p for p in buffer if p.plan.needs_detection]
        if detect:
            # The batched entry point is version-gated + capability-checked at LOAD (#192,
            # :mod:`latence_pii_gliner2._compat`): an unsupported gliner2 or a checkpoint object
            # without 'batch_extract_entities' has already failed as a typed ProviderError there,
            # so by here the method is guaranteed callable. The pod script's Gate 0 asserts the
            # same on the real checkpoint before any throughput number.
            # Group by effective label set (one label set per batched call). Order-stable: a dict
            # preserves first-seen group order, and emission below re-walks the buffer in input
            # order.
            groups: dict[tuple[tuple[str, ...], frozenset[str]], list[_PendingChunk]] = {}
            for pending in detect:
                groups.setdefault((pending.labels, pending.guard), []).append(pending)
            for (labels, _guard), pendings in groups.items():
                texts = [p.text for p in pendings]
                # dogfood-2b-process: split the group's texts into token-budget MICRO-BATCHES and
                # issue one ``batch_extract_entities`` call per micro-batch, concatenating results
                # in order. This decouples the model-forward tensor from the progress-cadence
                # ``batch_size`` — a whole group of long chunks no longer pads into ONE multi-GiB
                # activation (the measured OOM), for ANY configured ``batch_size``/``max_len``. The
                # output is batch-invariant (same spans/order), so this is byte-identical.
                results_list: list[Any] = []
                for start, end in iter_token_budget_microbatches(
                    texts,
                    max_len=self._max_len,
                    max_texts=self._microbatch_max_texts,
                    token_budget=self._microbatch_token_budget,
                ):
                    # Re-fetch the model per micro-batch so a prior micro-batch's W9 uncompiled
                    # reload (which updates self._model) is picked up here, not the stale handle.
                    micro = self._batch_extract(self._load_model(), texts[start:end], list(labels))
                    if len(micro) != end - start:
                        msg = (
                            "redaction.gliner2: batch_extract_entities returned "
                            f"{len(micro)} results for a micro-batch of {end - start} texts."
                        )
                        raise ValueError(msg)
                    results_list.extend(micro)
                for pending, raw in zip(pendings, results_list, strict=True):
                    raw_by_chunk[id(pending)] = raw
        for pending in buffer:
            accepted: list[tuple[int, int, str, str, str, float]] = []
            markers: list[RiskMarker] = []
            if pending.plan.needs_detection:
                raw = raw_by_chunk[id(pending)]
                accepted = self._accepted_from_raw(
                    pending.text,
                    raw,
                    pending.plan.allowed,
                    guard=pending.guard,
                )
                markers = _guard_markers(raw, pending.guard)
            redacted = finalize_redacted_chunk(pending.chunk, pending.plan, accepted, self._salt)
            # A guard hit is FLAGGED, not masked: it marks the chunk's *content* as risky (the
            # screening.content_keyword precedent — a RiskMarker that survives into the corpus so a
            # RAG consumer can exclude the chunk), whereas masking would silently corrupt the text.
            # No hits ⇒ the record is returned untouched, so a guardrail-off (or clean) run is
            # byte-identical.
            yield (
                redacted.model_copy(update={"risk_markers": [*redacted.risk_markers, *markers]})
                if markers
                else redacted
            )

    @staticmethod
    def _accepted_from_raw(
        text: str,
        raw: Any,
        allowed: set[str] | None,
        *,
        guard: frozenset[str] = frozenset(),
    ) -> list[tuple[int, int, str, str, str, float]]:
        """Build an ordered, non-overlapping candidate list from THIS chunk's raw GLiNER2 output.

        ``gliner2``'s ``extract_entities`` / ``batch_extract_entities`` return a **per-label dict**
        of dict-rows: ``{"entities": {"<label>": [{"text", "confidence", "start", "end"}, ...],
        ...}}`` — a DIFFERENT shape from the ``gliner`` library's flat list. ``raw`` is this chunk's
        result (mapped back by index from the batched call, W7). The ``start``/``end`` are
        chunk-offset (relative to the chunk ``content`` passed in). This flattens the per-label dict
        → one candidate per row, then a type not in a policy's allow-list is dropped and overlaps
        resolved (earlier start, longer span, type) so a stable, deterministic, non-overlapping set
        results. Returns ``(local_start, local_end, pii_type, value, detector, confidence)`` tuples
        (chunk-content offsets); :func:`~latence_core.redaction_policy.finalize_redacted_chunk`
        derives spans + pages + placeholders + masking.

        ``guard`` (T2) is the casefolded set of GUARDRAIL categories this chunk was screened for.
        Those labels rode the same forward but are NOT PII: they are read out separately by
        :func:`_guard_markers` as risk markers, so they are dropped here rather than masked. Empty
        (guardrailing off) ⇒ this method behaves exactly as before.
        """
        # Faithful parse of the REAL gliner2 shape: a top-level {"entities": {label: [rows]}}
        # dict. A flat-list mis-parse (the gliner-library shape) would find no per-label rows
        # here and silently produce zero spans — the anchor test asserts the correct shape bites.
        entities_by_label: Any = raw["entities"] if isinstance(raw, dict) else raw
        if not isinstance(entities_by_label, dict):
            got = type(entities_by_label).__name__
            msg = (
                "redaction.gliner2: unexpected extract_entities output shape "
                f"(expected {{'entities': {{label: [rows]}}}}, got {got})"
            )
            raise ValueError(msg)

        # (start, end, type, value, score) tuples, filtered + sorted deterministically.
        candidates: list[tuple[int, int, str, str, float]] = []
        for label, rows in entities_by_label.items():
            pii_type = str(label)
            if not pii_type:
                continue
            if pii_type.casefold() in guard:
                continue  # a guardrail category — flagged, never masked (see _guard_markers)
            if allowed is not None and pii_type not in allowed:
                continue
            for row in rows:
                start = int(row["start"])
                end = int(row["end"])
                if end <= start or start < 0 or end > len(text):
                    continue  # bounds-safe: drop a malformed/out-of-range row
                score = min(max(float(row.get("confidence", 1.0)), 0.0), 1.0)
                candidates.append((start, end, pii_type, text[start:end], score))
        # Deterministic order: by start, then longest span, then label (matches the twin).
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        accepted: list[tuple[int, int, str, str, str, float]] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            accepted.append((start, end, pii_type, value, f"gliner2:{pii_type}", score))
            last_end = end
        return accepted


def _guard_markers(raw: Any, guard: frozenset[str]) -> list[RiskMarker]:
    """The GUARDRAIL half of one chunk's raw GLiNER2 output → :class:`RiskMarker`s (T2).

    The guardrails+PII checkpoint returns guard categories in the SAME per-label dict as PII types,
    so this reads the guard labels back out of ``raw`` and records ONE marker per category that
    fired: ``score`` is the strongest hit for that category (a chunk with three weak injection
    fragments is not three separate risks), ``category`` is the canonical screening vocabulary
    (:func:`~latence_core.redaction_policy.guard_category`, the same values
    ``screening.content_keyword`` emits, so a consumer filters ONE set), and markers are emitted in
    sorted category order so the record is deterministic.

    The ``reason`` deliberately carries the category, the hit count and the chunk-local offsets and
    **never the matched text** (C2/C6): a marker survives into the exported corpus, and echoing the
    harmful/injected string into it would re-publish exactly what was flagged.

    Returns ``[]`` when guardrailing is off or nothing fired — the byte-identical path.
    """
    if not guard:
        return []
    by_label: Any = raw["entities"] if isinstance(raw, dict) else raw
    if not isinstance(by_label, dict):
        # The same faithful-shape contract _accepted_from_raw enforces; it runs first on every
        # chunk, so an unexpected shape has already surfaced as a typed error before this point.
        return []
    hits: dict[str, tuple[float, int]] = {}
    for label, rows in by_label.items():
        label_str = str(label)
        if label_str.casefold() not in guard or not isinstance(rows, list):
            continue
        scores = [min(max(float(row.get("confidence", 1.0)), 0.0), 1.0) for row in rows]
        if not scores:
            continue
        category = guard_category(label_str)
        best, count = hits.get(category, (0.0, 0))
        hits[category] = (max(best, *scores), count + len(scores))
    return [
        RiskMarker(
            category=category,
            detector="redaction.gliner2:guardrail",
            score=score,
            reason=(
                f"learned guardrail category {category!r} matched {count} span(s) in this chunk "
                "(flagged, not masked; the matched text is deliberately not echoed)"
            ),
        )
        for category, (score, count) in sorted(hits.items())
    ]

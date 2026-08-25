"""PII Redaction over a **GLiNER 2.5** checkpoint — zero-shot, and honest about it (ADR-0064).

``redaction.gliner25``. It fulfils the same :class:`~latence_core.capability.PIIDetector` seam as
``redaction.hybrid_rule``, ``redaction.gliner_pii`` and ``redaction.gliner2``, reuses ALL the
shared redaction machinery (:class:`~latence_core.redaction_policy.RedactionPolicy`, the masking /
placeholder / action logic, the :class:`~latence_core.contracts.PIISpan` /
:class:`~latence_core.contracts.RedactionRecord` contracts, the offset/page carry, the
``PARSE_ERROR`` pass-through and the ``Classification.sensitivity``-keyed policy) and swaps ONLY
the detection call.

**THE CLAIM THIS PROVIDER MAKES, AND THE ONE IT DOES NOT.**

There is **no GLiNER 2.5 PII or guardrails checkpoint.** As of 2026-08-25 the fastino PII
checkpoints — ``GLiNER2-Guardrails-PII-Multi`` and ``gliner2-privacy-filter-PII-multi`` — are still
the 2.x-era *span* models that ``redaction.gliner2`` drives; the 2.5 family
(``gliner2.5-multi-v1`` / ``-base-v1`` / ``-small-v1``) is general-purpose. So "PII on 2.5" means
driving the GENERAL 2.5 checkpoint with PII labels **zero-shot**.

That is a real capability — the whole GLiNER family is zero-shot span extraction, and PII types are
ordinary label strings to it — but it is **not** the same thing as a model trained on the task, and
this package does not pretend otherwise:

* nothing here claims parity with ``redaction.gliner2``. No recall or precision number for this
  Provider exists yet; the measurement is a separate step (``docs/BENCHMARKS.md``), and until it
  lands the honest description is *unmeasured*.
* ``guardrails: true`` is **refused** on the default checkpoint (a typed
  :class:`~latence_core.errors.ConfigError`), because a guardrail screen is a security control and
  running one against a model with no guardrail training would produce a control that reports
  nothing while looking enabled. It is accepted, with a warning, on an operator-supplied checkpoint
  — someone who fine-tuned their own 2.5 guardrails head knows something this package does not.
* the ADR-0044 universal financial-PII floor is unaffected either way. It is a deterministic
  recogniser at the single shared ``finalize_redacted_chunk`` seam every Redaction Provider funnels
  through, so ``ssn``/``credit_card``/``iban`` are masked whatever the model does or fails to do.
  That floor is the reason a zero-shot PII Provider is shippable at all rather than merely
  plausible.

**The API**, verified against the unpacked ``gliner2`` 2.0.0 wheel (2026-08-25):

.. code-block:: python

    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1", quantize=…, compile=…)
    out = model.batch_extract_entities(texts, entity_types, threshold=0.5,
                                       include_spans=True, include_confidence=True, max_len=1280)
    # -> [{"entities": {"<label>": [{"text", "confidence", "start", "end"}, ...], ...}}, ...]

The per-label-dict shape is unchanged from 1.3.2, so the parse is unchanged. ``include_spans=True``
is REQUIRED — without it there are no char offsets and therefore nothing to mask — and is always
passed. Offsets are chunk-local ``start``/``end`` (half-open) and map directly onto the shared
redaction pipeline; a chunk-local span still resolves to its ORIGINAL source page through the
chunk's ``offset_map`` + the :class:`~latence_core.page_index.PageIndexResolver` (ADR-0031/0042).

The **raw PII value is never persisted** on a :class:`~latence_core.contracts.PIISpan`: the surface
string is used only transiently to derive a deterministic surrogate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError
from latence_core.induced import (
    LABEL_FLOOR_CONFIG_KEY,
    LabelFloor,
    LabelSlot,
    effective_labels,
)
from latence_core.page_index import PageIndexResolver
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.perf import (
    empty_cuda_cache,
    inference_context,
    is_compile_backend_failure,
    is_cuda_oom,
    iter_label_buckets,
    iter_token_budget_microbatches,
    resolve_gliner2_knob,
    resolve_intra_op_threads,
)
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    ChunkRedactionPlan,
    RedactionPolicy,
    finalize_redacted_chunk,
    parse_action,
    plan_chunk_redaction,
)

from latence_gliner25._backbone import (
    DEFAULT_MODEL,
    load_gliner25_model,
    require_chunk_fits_window,
    resolve_gliner25_max_len,
    resolve_window_safe_microbatch_bounds,
)

_log = logging.getLogger("latence.provider.gliner25_pii")

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

PROVIDER_NAME = "redaction.gliner25"

#: This Provider drives ONLY the entity half — a Redactor asks no relation question.
_METHODS: tuple[str, ...] = ("batch_extract_entities",)

#: The config ``pii_labels`` are a UNIVERSAL SAFETY FLOOR, not a relevance floor — the one place
#: where the per-chunk rule does NOT apply to the config list. Induced ``pii_types`` are scoped per
#: chunk (they are relevance labels: what THIS chunk is likely to contain); the configured floor is
#: stamped onto EVERY chunk regardless, because a leaked card/IBAN/SSN in a chunk whose induced
#: types happen to omit finance is exactly the leak a per-chunk-only screen would ship.
#: Deliberately NOT configurable — see :meth:`Gliner25PIIRedactor._reject_floor_subsetting`.
_PII_LABEL_FLOOR = LabelFloor.UNIVERSAL


@dataclass(frozen=True)
class _PendingChunk:
    """A chunk's model-independent redaction context, resolved before the batched call.

    The :class:`~latence_core.redaction_policy.ChunkRedactionPlan` (policy action/allow-list, the
    skip/near-empty decision, the chunk offset/page maps) and the effective label set are per-chunk
    and model-independent, so they are computed once up front; a flush then groups the
    detect-needing chunks by label set and issues one ``batch_extract_entities`` call per group,
    mapping each per-text result back to its chunk by identity, keeping emission in input-chunk
    order and the output record-for-record identical to a per-chunk path.
    """

    chunk: ChunkRecord
    text: str
    plan: ChunkRedactionPlan
    labels: tuple[str, ...]


class Gliner25PIIRedactor(AdapterBase):
    """Zero-shot PIIDetector Provider backed by a general GLiNER 2.5 checkpoint (ADR-0064).

    Config:
        pii_labels: list[str]  — REQUIRED. The PII entity types to detect (e.g.
            ``["person", "email", "phone number"]``). Lower-case, natural-language type names work
            best for a zero-shot span model.
        label_floor: str  — accepted ONLY as ``universal``. Unlike the sibling extractor, this
            Provider's config labels are SAFETY floors, not relevance labels, so the knob cannot
            subset them; any other value is a ConfigError rather than a silently ignored key.
        guardrails: bool  — **refused on the default checkpoint** (default ``False``). There is no
            GLiNER 2.5 guardrails checkpoint; see the module docstring. Accepted with a warning
            when ``model`` names an operator-supplied checkpoint.
        model: str  — the checkpoint id (default ``fastino/gliner2.5-multi-v1``, Apache-2.0). It
            must be a **boundary**-architecture checkpoint; a 1.x ``span`` checkpoint (including
            the dedicated PII ones) is refused at load with a typed error naming the Provider that
            does load it — ``redaction.gliner2``.
        threshold: float  — score threshold; spans below it are dropped (default 0.5).
        default_action: str  — action when a policy names none: ``mask`` (default), ``replace``,
            ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map (identical shape to the twin
            Providers): per ``Classification.sensitivity`` value an ``action`` / ``types``
            allow-list / ``skip`` flag.
        salt: str  — salt mixed into ``replace``/``hash`` surrogates (default 'latence').
        min_chars: int  — a chunk whose content is shorter is emitted clean (default 1).
        max_len: int  — the model token window (default 1280, capped at the checkpoint's 4096).
            Past it the model truncates SILENTLY — which for a Redactor means PII left UNMASKED in
            the exported corpus, a correctness failure rather than a throughput one — so the window
            is bounded explicitly and must cover the chunk budget.
        batch_size / microbatch_max_texts / microbatch_token_budget / intra_op_threads /
        quantize / compile / device  — as on ``fused_entity_relation.gliner25``; the micro-batch
            defaults are derived so that raising ``max_len`` does not widen the padded forward.

    The model is loaded lazily and cached on first use, so constructing the Provider never
    downloads weights.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default gliner2.5-multi-v1 checkpoint.

        Licence **verified** per ADR-0012, weights AND code separately, each with its date — see
        :data:`~latence_gliner25._backbone.DEFAULT_MODEL`.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical across
        hardware/library versions. ``memory_mb`` is the approximate resident footprint (declared,
        not measured).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=2048,
            model_id=DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=(
                "https://huggingface.co/fastino/gliner2.5-multi-v1 (weights, Apache-2.0, not "
                "gated — model card, checked 2026-08-25); "
                "https://github.com/fastino-ai/GLiNER2 (library, Apache-2.0 — the gliner2 2.0.0 "
                "wheel METADATA carries 'License-Expression: Apache-2.0' and bundles LICENSE, "
                "read 2026-08-25)"
            ),
            license_verified_on="2026-08-25",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note=(
                "ZERO-SHOT PII on the GENERAL GLiNER 2.5 checkpoint: there is no 2.5 PII or "
                "guardrails checkpoint, so this Provider asks a general model for PII labels. NO "
                "parity with the task-trained redaction.gliner2 checkpoint is claimed and no "
                "recall number has been measured (docs/BENCHMARKS.md). The ADR-0044 deterministic "
                "financial-PII floor at finalize_redacted_chunk applies regardless of backend. "
                "Native quantize/compile ride the shared cuda defaults, guarded at load and by "
                "the inference-time reload-uncompiled latch; TorchScript JIT neutralized and "
                "intra-op threads capped by the shared load seam (#190)"
            ),
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        pii_labels = config.get("pii_labels")
        if not pii_labels or not isinstance(pii_labels, list):
            msg = (
                "Gliner25PIIRedactor requires a non-empty 'pii_labels' list in its config "
                "(the PII entity type set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        # AdapterBase resolves the device seam (compute="either") against the declared profile.
        super().__init__(config)
        self._labels = [str(label) for label in pii_labels]
        self._reject_floor_subsetting(config)
        self._model_id = str(config.get("model", DEFAULT_MODEL))
        self._reject_unbacked_guardrails(config)
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        # fail_open=True: for a REDACTOR an over-wide window is not a recall trade, it is
        # "redacted" output that still contains PII with nothing reporting it. See
        # `resolve_gliner25_max_len` for the measurement and for the operator-checkpoint escape.
        self._max_len = resolve_gliner25_max_len(
            self._config,
            fail_open=True,
            provider_name=self.provider_name,
            model_id=self._model_id,
        )
        # Validate the batch_size knob eagerly (fail-fast at construction) even though the batched
        # call reads it lazily via AdapterBase.batch_size().
        self.batch_size()
        try:
            self._microbatch_max_texts, self._microbatch_token_budget = (
                resolve_window_safe_microbatch_bounds(self._config, max_len=self._max_len)
            )
            resolve_gliner2_knob("quantize", self.device.device, self._config)
            resolve_gliner2_knob("compile", self.device.device, self._config)
            resolve_intra_op_threads(self._config)
        except ValueError as exc:
            msg = f"{self.provider_name}: {exc}"
            raise ConfigError(msg) from exc
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        # The framework owns policy + masking (shared with every other Redaction Provider); this
        # Provider owns only detection — the PIIDetector seam.
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        self._model: Any = None
        self._compile_disabled: bool = False

    @property
    def provider_name(self) -> str:
        """The registered entry-point name, used only to tag typed errors (never carries PII)."""
        return PROVIDER_NAME

    def _reject_floor_subsetting(self, config: dict[str, Any]) -> None:
        """Refuse a ``label_floor`` that would scope this Redactor's SAFETY floor per chunk.

        The sibling extractor exposes ``label_floor`` because entity/relation labels are relevance
        labels. Here they are not. Rather than accept the key and quietly ignore it — leaving an
        operator believing they had narrowed the PII screen when they had not — anything but
        ``universal`` is a loud :class:`ConfigError` (ADR-0034). The no-op ``universal`` value is
        accepted so a shared/templated config that states the policy explicitly still loads.
        """
        raw = config.get(LABEL_FLOOR_CONFIG_KEY)
        if raw is None or raw == LabelFloor.UNIVERSAL.value:
            return
        msg = (
            f"{self.provider_name}: {LABEL_FLOOR_CONFIG_KEY}={raw!r} is not accepted — a "
            "Redactor's config PII labels are a UNIVERSAL safety floor applied to every chunk, "
            "not per-chunk relevance labels (a leaked card in an 'irrelevant' chunk must still be "
            "masked). Only the INDUCED pii types are scoped per chunk. Omit the key, or set "
            f"{LABEL_FLOOR_CONFIG_KEY}: {LabelFloor.UNIVERSAL.value!r}."
        )
        raise ConfigError(msg)

    def _reject_unbacked_guardrails(self, config: dict[str, Any]) -> None:
        """Refuse ``guardrails: true`` on a checkpoint that has no guardrail training.

        The twin ``redaction.gliner2`` screens learned guardrail categories because its default
        checkpoint IS a guardrails+PII model. No GLiNER 2.5 guardrails checkpoint exists, so on the
        default checkpoint the same switch would ask a general model about ``prompt injection`` and
        ``harmful content`` zero-shot and record whatever came back as a security finding. A
        security control that is enabled and unbacked is worse than one that is off, so this is a
        :class:`ConfigError` naming the two things that actually work: the deterministic
        ``screening.content_keyword`` Stage, and ``redaction.gliner2`` on the guardrails checkpoint.

        An operator-supplied ``model`` is a different situation — someone who trained a 2.5
        guardrails head knows something this package does not — so there the switch is accepted
        with a warning rather than refused. Either way the value must be a bool: a non-bool is a
        config typo, never a truthy accident.
        """
        enabled = config.get("guardrails", False)
        if not isinstance(enabled, bool):
            msg = f"{self.provider_name}: guardrails must be a bool (got {enabled!r})"
            raise ConfigError(msg)
        if not enabled:
            return
        if self._model_id != DEFAULT_MODEL:
            _log.warning(
                "%s: guardrails is ON against the operator-supplied checkpoint %r. No fastino "
                "GLiNER 2.5 guardrails checkpoint exists, so this Provider cannot verify that the "
                "model has a guardrail head; the categories are asked zero-shot",
                self.provider_name,
                self._model_id,
            )
            return
        msg = (
            f"{self.provider_name}: guardrails: true is not accepted on the default checkpoint "
            f"{DEFAULT_MODEL!r}. There is no GLiNER 2.5 guardrails/PII checkpoint — the fastino "
            "guardrails model (GLiNER2-Guardrails-PII-Multi) is a gliner2 1.x 'span' checkpoint — "
            "so enabling this would run a security screen against a model with no guardrail "
            "training and record its zero-shot guesses as findings. Use the deterministic "
            "'screening.content_keyword' Stage, and/or 'redaction.gliner2' (package "
            "latence-pii-gliner2) whose checkpoint IS a guardrails model. If you have your own "
            "GLiNER 2.5 guardrails checkpoint, set 'model' to it and this switch is accepted."
        )
        raise ConfigError(msg)

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        """Load the checkpoint on first use (cached) via the shared load choreography."""
        if self._model is not None:
            return self._model
        self._model = load_gliner25_model(
            self._model_id,
            self._config,
            self.device,
            provider_name=self.provider_name,
            methods=_METHODS,
            guard=self.guard,
        )
        return self._model

    def _reload_uncompiled(self) -> Any:
        """Re-load with ``compile=False`` (quantize kept) after an inference-time compile fault."""
        self._compile_disabled = True
        _log.warning(
            "%s: torch.compile failed at inference; re-loading the uncompiled model "
            "(quantize kept) — throughput only, never a crash",
            self.provider_name,
        )
        self._model = load_gliner25_model(
            self._model_id,
            {**self._config, "compile": False},
            self.device,
            provider_name=self.provider_name,
            methods=_METHODS,
            guard=self.guard,
        )
        return self._model

    def _extract_once(self, model: Any, texts: list[str], labels: list[str]) -> list[Any]:
        """A single ``batch_extract_entities`` forward under :meth:`AdapterBase.guard` (typed).

        Every argument after ``texts``/``entity_types`` is passed BY KEYWORD, because gliner2
        2.0.0's third positional parameter is ``batch_size`` — a positional ``threshold`` would
        silently become the model's internal sub-batch size. The forward runs inside
        ``torch.inference_mode()``: a redaction-only encoder never backpropagates, so an autograd
        graph would only accumulate VRAM run over run.
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
        """One batched forward with the compile guard AND a CUDA-OOM micro-batch backoff.

        Both faults are recovered so ``latence process`` does not die mid-run and forfeit the whole
        run dir — the ADR-0044 redaction floor must hold, so no chunk is ever dropped or emitted
        un-scanned. A batch of ONE that still OOMs is genuinely unrecoverable on this hardware and
        surfaces as a typed ``ProviderError`` (no fake green, no dropped chunk).
        """
        try:
            return self._extract_once(model, texts, labels)
        except Exception as exc:  # noqa: BLE001 - intercept recoverable faults before surfacing
            if not self._compile_disabled and is_compile_backend_failure(exc):
                return self._extract_once(self._reload_uncompiled(), texts, labels)
            if is_cuda_oom(exc) and len(texts) > 1:
                empty_cuda_cache(provider_name=self.provider_name)
                mid = len(texts) // 2
                left = self._batch_extract(self._load_model(), texts[:mid], labels)
                right = self._batch_extract(self._load_model(), texts[mid:], labels)
                return left + right
            raise

    # -- redaction ----------------------------------------------------------

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Detect PII per chunk with the model; yield the SAME chunks with PII masked (ADR-0042).

        Each chunk's ``content`` (≤ the model window) is scanned in full — no truncation — and the
        chunk is re-emitted with ``masked_content`` + ``pii_spans`` set (``content`` UNCHANGED).
        Chunks are threaded 1:1 in first-seen order; a chunk-local span resolves to its ORIGINAL
        source page through the chunk's ``offset_map`` + the
        :class:`~latence_core.page_index.PageIndexResolver`, which reads the chunk's own
        ``page_slice`` (v19) — so resolution is exact whatever order chunks arrive in.
        """
        resolver = PageIndexResolver()
        buffer: list[_PendingChunk] = []

        def flush() -> Iterator[ChunkRecord]:
            yield from self._flush_buffer(buffer)
            buffer.clear()

        for chunk in chunks:
            buffer.append(self._prepare(chunk, resolver))
            if len(buffer) >= self.batch_size():
                yield from flush()
        yield from flush()

    def _prepare(self, chunk: ChunkRecord, resolver: PageIndexResolver) -> _PendingChunk:
        """Resolve everything a chunk needs BEFORE the (possibly batched) model call.

        ``resolver`` is the call-scoped page resolver — planning runs for EVERY chunk (before the
        skip decision), so handing it in here is what absorbs the document's map from its first
        chunk. The label set is ``config floor ∪ induced[pii_types]`` under
        :data:`_PII_LABEL_FLOOR` (``UNIVERSAL``): the induced half is per chunk, the configured
        floor is NOT subsetted.
        """
        # THE fail-open guard. gliner2 truncates a chunk longer than the window SILENTLY, so the
        # tail would be emitted as "redacted" while still carrying its PII and nothing in the run
        # would say so. Checked before the plan, so a mis-sized pipeline stops at its first chunk
        # rather than exporting a corpus that looks masked.
        require_chunk_fits_window(
            token_count=chunk.token_count,
            record_id=chunk.record_id,
            max_len=self._max_len,
            provider_name=self.provider_name,
        )
        plan = plan_chunk_redaction(
            chunk,
            self._policy,
            detector_types=set(self._labels),
            sensitive_set=set(),
            min_chars=self._min_chars,
            resolver=resolver,
        )
        labels = effective_labels(
            self._labels, chunk.induced_labels, LabelSlot.PII, floor=_PII_LABEL_FLOOR
        )
        return _PendingChunk(chunk=chunk, text=chunk.content, plan=plan, labels=tuple(labels))

    def _flush_buffer(self, buffer: list[_PendingChunk]) -> Iterator[ChunkRecord]:
        """Batch the detect-needing chunks (grouped by label set), then emit in input order.

        ``batch_extract_entities`` takes ONE label set per call, so the detect-needing chunks are
        bucketed by their effective labels through the SHARED bucketing seam (the same primitive
        the fused extractor uses, so the two cannot drift on bucket order), and each bucket's texts
        are split into token-budget micro-batches. Each per-text result maps back to its chunk by
        identity. Chunks that skip / are near-empty need no model call. Emission is in the buffer's
        (input) order regardless of grouping.
        """
        if not buffer:
            return
        raw_by_chunk: dict[int, Any] = {}
        detect = [p for p in buffer if p.plan.needs_detection]
        if detect:
            keys = [pending.labels for pending in detect]
            for labels, indices in iter_label_buckets(keys):
                pendings = [detect[i] for i in indices]
                texts = [p.text for p in pendings]
                results_list: list[Any] = []
                for start, end in iter_token_budget_microbatches(
                    texts,
                    max_len=self._max_len,
                    max_texts=self._microbatch_max_texts,
                    token_budget=self._microbatch_token_budget,
                ):
                    # Re-fetch the model per micro-batch so a prior micro-batch's uncompiled reload
                    # (which updates self._model) is picked up here, not the stale handle.
                    micro = self._batch_extract(self._load_model(), texts[start:end], list(labels))
                    if len(micro) != end - start:
                        msg = (
                            f"{PROVIDER_NAME}: batch_extract_entities returned {len(micro)} "
                            f"results for a micro-batch of {end - start} texts."
                        )
                        raise ValueError(msg)
                    results_list.extend(micro)
                for pending, raw in zip(pendings, results_list, strict=True):
                    raw_by_chunk[id(pending)] = raw
        for pending in buffer:
            accepted: list[tuple[int, int, str, str, str, float]] = []
            if pending.plan.needs_detection:
                accepted = self._accepted_from_raw(
                    pending.text, raw_by_chunk[id(pending)], pending.plan.allowed
                )
            yield finalize_redacted_chunk(pending.chunk, pending.plan, accepted, self._salt)

    @staticmethod
    def _accepted_from_raw(
        text: str, raw: Any, allowed: set[str] | None
    ) -> list[tuple[int, int, str, str, str, float]]:
        """Build an ordered, non-overlapping candidate list from THIS chunk's raw output.

        ``gliner2``'s batched entity call returns a **per-label dict** of dict-rows:
        ``{"entities": {"<label>": [{"text", "confidence", "start", "end"}, ...], ...}}`` — the
        shape a flat-list mis-parse would silently read as ZERO spans, which is the false green the
        anchor test closes. ``start``/``end`` are chunk-local. This flattens the dict → one
        candidate per row, drops a type outside a policy's allow-list, and resolves overlaps
        (earlier start, longer span, type) into a stable, deterministic, non-overlapping set.
        Returns ``(local_start, local_end, pii_type, value, detector, confidence)`` tuples;
        :func:`~latence_core.redaction_policy.finalize_redacted_chunk` derives spans + pages +
        placeholders + masking from there.
        """
        entities_by_label: Any = raw["entities"] if isinstance(raw, dict) else raw
        if not isinstance(entities_by_label, dict):
            got = type(entities_by_label).__name__
            msg = (
                f"{PROVIDER_NAME}: unexpected batch_extract_entities output shape "
                f"(expected {{'entities': {{label: [rows]}}}}, got {got})"
            )
            raise ValueError(msg)

        candidates: list[tuple[int, int, str, str, float]] = []
        for label, rows in entities_by_label.items():
            pii_type = str(label)
            if not pii_type or not isinstance(rows, list):
                continue
            if allowed is not None and pii_type not in allowed:
                continue
            for row in rows:
                if not isinstance(row, dict) or "start" not in row or "end" not in row:
                    continue  # no span (include_spans not honoured) — nothing to mask
                start = int(row["start"])
                end = int(row["end"])
                if end <= start or start < 0 or end > len(text):
                    continue  # bounds-safe: drop a malformed/out-of-range row
                score = min(max(float(row.get("confidence", 1.0)), 0.0), 1.0)
                candidates.append((start, end, pii_type, text[start:end], score))
        # Deterministic order: by start, then longest span, then label (matches the twins).
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        accepted: list[tuple[int, int, str, str, str, float]] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            accepted.append((start, end, pii_type, value, f"gliner25:{pii_type}", score))
            last_end = end
        return accepted

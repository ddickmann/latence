"""Unsupervised schema-induction LabelInducer over any OpenAI-compatible endpoint (W4, ADR-0038).

Restores the "point it at a folder and it discovers its own schema" property: an OPTIONAL
``schema_induction`` Stage that uses a small OpenAI-compatible LLM with **structured JSON output**
to induce the entity / relation / PII **types** actually evidenced in the data, then annotates every
:class:`~latence_core.contracts.ChunkRecord` with an
:class:`~latence_core.contracts.InducedLabels` carrier. The extraction Stages then UNION those
induced types onto their config labels (the ``latence_core.induced`` seam) — no extractor rewrite.

**Induction is PER CHUNK (T1).** Each chunk is induced against its own text and keeps its own
small, relevant label set, because that is what the extractor downstream actually needs: a chunk
scored against ~5 relevant types beats the same chunk scored against a ~40-type corpus union on
both recall (no salience cap dropped the rare-but-critical type this chunk needed) and zero-shot
precision (no label competition from ~35 types that are not in it). It is also what keeps the
gliner2 span-scoring tensor small — the corpus-union label dimension was the root cause of the
CUDA-OOM, not the batch size.

The historical argument for the ``corpus`` granularity was **type consistency** — induce per chunk
and you get ``org`` here and ``organization`` there, and the KG's types drift. That guarantee does
NOT come from this Stage any more: consistency is **earned downstream**, by corpus-wide type-
vocabulary canonicalization in the resolution phase (T3), alongside the entity canonicalization
that already lives there. Imposing it here — before extraction — bought consistency by paying
recall, and this module no longer makes that trade for you. ``corpus`` mode remains selectable for
an operator who wants one frozen vocabulary, but it is a corpus-level operation inside what is
otherwise a chunk-level front half (see the phase-boundary check in
:mod:`latence_core.stacks.phase_boundary`), and it is not what ``latence setup`` emits.

Model-agnostic seam (HARD REQUIREMENT): the call goes through the official ``openai`` SDK with
config ``model`` / ``base_url`` / ``api_key`` — the SAME pattern ``relation.llm`` uses — so a client
switches OpenAI → OpenRouter → on-prem/air-gapped vLLM by config alone (all speak the OpenAI
chat-completions API). The default ``model`` is ``gpt-4.1-mini`` (a served-model id, not a bundled
checkpoint); ``base_url=None`` ⇒ OpenAI, set it to ``https://openrouter.ai/api/v1`` (OpenRouter) or
``http://<vllm-host>:8000/v1`` (on-prem). Fault tolerance is the ported endpoint machinery
(:func:`~latence_core.endpoint.call_with_retry` + :class:`~latence_core.endpoint.RetryPolicy`),
identical to the other endpoint Providers; the transient-retry count surfaces on
``endpoint_retries`` for the Quality Report.

The LLM is UNTRUSTED output — every safety net is mandatory (the run NEVER crashes on bad output):
  1. **Structured output**: request ``response_format={"type":"json_object"}`` (portable across
     OpenAI/OpenRouter/vLLM); the prompt pins the exact ``{entity_types,relation_types,pii_types}``
     shape.
  2. **Generous token budget + truncation detection**: ``max_tokens`` default 2048; a reply whose
     ``finish_reason == "length"`` is truncated → ONE retry at doubled budget → if still truncated,
     fall back to config labels (never a truncated-JSON crash).
  3. **Parse-and-repair**: ``json.loads`` in a guard; on ``JSONDecodeError`` do ONE corrective retry
     ("return ONLY valid JSON of this exact shape"); a second failure falls back. The parser is also
     tolerant of a JSON object the model wrapped in a Markdown fence or surrounding prose — it peels
     the wrapper and re-tries the STRICT parser on the extracted object (it never loosens parsing,
     so hallucinated noise still fails). NEVER ``eval``, NEVER execute model text.
  4. **Schema validation**: the parsed object MUST be ``{entity_types,relation_types,pii_types}``,
     each a list of strings; anything else → reject → fall back. Extra keys are ignored.
  5. **Label sanitization** (:mod:`~latence_schema_inducer.sanitize`): strip control chars, cap
     length ≤ 64, drop empties, dedupe case-insensitively, cap count at the configured
     ``label_ceiling`` — a SAFETY ceiling (64 on the per-chunk path), not a salience cap (dropped
     items logged).
     Unicode/German label text (e.g. ``Rechnungsnummer``) is preserved — only control/format chars
     are stripped, so a valid non-ASCII schema is never dropped to empty.
  6. **Empty-induction recovery (the robustness net)**: a reply that parses + validates but
     sanitizes to an EMPTY / below-``min_labels`` label set is NOT silently accepted — that quiet
     degrade to the bare config floor is the quality cliff induce exists to prevent. The Provider
     RE-PROMPTS once (the empty-recovery instruction at a doubled budget) and only a still-empty
     result falls back — LOUDLY (``induction_empty_after_sanitization`` + the WARNING), never a
     silent revert.
  7. **Fail-open, never crash** (G1 posture): ANY inducer failure (network, auth, malformed,
     truncated-twice, schema-invalid, empty-after-recovery) leaves the document's chunks'
     ``induced_labels`` as ``None`` so the extractors use their config labels and the run continues;
     the failure is a typed, no-content log + a Quality-Report note (``induction_fallbacks``, plus
     ``induction_empty_after_sanitization`` for the empty case). A missing/unreachable endpoint is a
     skip-with-flag (fail-open), not a crash.

Concurrency (W6, ADR-0036): the LLM calls are pure per-request latency, so ``induce`` **fans the
per-item calls out concurrently** — the ``openai`` async client (:class:`~openai.AsyncOpenAI`) +
an :class:`asyncio.Semaphore` (``max_concurrency``, default 8) + a single
:func:`asyncio.run` over :func:`asyncio.gather` (``return_exceptions=True``) inside the SYNC
``induce`` (the Runner is sync). The wall-clock collapses from ``N × latency`` to
``ceil(N / max_concurrency) × latency``. Every safety net above is applied PER concurrent result,
and a single item's failure fails open for THAT item only (the gather never aborts the batch);
results are re-assembled in **input order** (index-keyed) so the concurrency does not perturb the
byte-identical output. ``max_concurrency=1`` is exactly today's sequential behaviour.

Deterministic: ``temperature`` 0 and stable label order (sanitization preserves first-seen order),
plus index-keyed reassembly, so a seeded run is byte-reproducible **regardless of completion
order** (Baseline bar; a shuffled-completion mock proves it). The real ``gpt-4.1-mini`` numbers come
from the maintainer's keyed pod run (``scripts/validate-schema-induction-on-gpu.md``); the offline
mock proves the machinery + every safety net.

Licensing: this Provider is framework code (Apache-2.0) and calls a user-supplied endpoint; it
bundles NO model weights. The ``openai`` client is Apache-2.0 (code). No key or endpoint is embedded
— both come from config/environment. The ``openai`` import is deferred to first use so importing
this module (for Provider discovery) never requires the client installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, cast

from latence_core.contracts import InducedLabels
from latence_core.endpoint import EndpointError, RetryPolicy, call_with_retry_async
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

from latence_schema_inducer.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    DOMAIN_CONTEXT_TEMPLATE,
    EMPTY_RECOVERY_INSTRUCTION,
    REPAIR_INSTRUCTION,
)
from latence_schema_inducer.sanitize import (
    CHUNK_LABEL_CEILING,
    MAX_LABEL_CEILING,
    MAX_LABELS,
    sanitize_domain,
    sanitize_labels,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

_LOG = logging.getLogger("latence.schema_induction")

# A Markdown fenced code block (```` ```json … ``` ```` or a bare ```` ``` … ``` ````) some models
# wrap their JSON in despite ``response_format=json_object``. The parse-hardening path peels the
# fence and re-tries the STRICT parser on the body (never loosens parsing — see ``_try_parse``).
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*(?P<body>.*?)```", re.DOTALL)

# Default served-model id (a hosted service, no weights license). gpt-4.1-mini per the W4 spec —
# NOT gpt-4o-mini.
_DEFAULT_MODEL = "gpt-4.1-mini"
# Generous token budget so a normal induction never cuts off; a genuinely truncated reply
# (finish_reason == "length") is retried at double this before falling back (safety net #2).
_DEFAULT_MAX_TOKENS = 2048
# The three JSON slots the model must return, mapped to the InducedLabels fields.
_SLOTS = ("entity_types", "relation_types", "pii_types")
# A chunk shorter than this (stripped) is treated as too small to induce a schema from — skipped,
# its induced_labels left None (the extractors use config labels). Bounds a pointless LLM call.
_DEFAULT_MIN_CHARS = 1
# The minimum TOTAL induced labels a successful induction must yield (robustness fix). Below this
# the result is treated as empty/near-empty → one empty-recovery re-prompt, then a LOUD fall-back.
# 1 = "any label is enough"; an all-empty response (0 labels) always triggers recovery.
_DEFAULT_MIN_LABELS = 1
# The default bound on in-flight concurrent LLM calls (W6). 8 collapses the lbbw induce stage from
# ~44s to ~6-8s; 1 = today's sequential behaviour. Clamped to >= 1 in __init__.
_DEFAULT_MAX_CONCURRENCY = 8


# Internal reason tag for the "parsed + validated but sanitized empty / below-min" miss — the one
# safety-net outcome that triggers the empty-recovery re-prompt (not a direct fall-back).
_EMPTY_AFTER_SANITIZATION = "empty-after-sanitization"


class _Truncated(Exception):
    """Internal signal: the reply's ``finish_reason`` was ``length`` (truncated, safety net #2)."""


async def _async_noop_sleep(_delay: float) -> None:
    """The injected async backoff sleep — a no-op, mirroring the sync path's ``lambda _d: None``.

    The endpoint Providers pass an instant sleep to the ported retry (the retry's value is the
    classing + the ``endpoint_retries`` tally, not wall-time backoff in this batch context); keeping
    it a no-op makes a seeded test instant and byte-stable and preserves the pre-W6 behaviour.
    """
    return None


class LLMLabelInducer:
    """LabelInducer Provider backed by an OpenAI-compatible chat endpoint with structured output.

    Config:
        model: str  — the chat model id (default ``gpt-4.1-mini``). For OpenRouter/vLLM set
            ``base_url`` accordingly and use that endpoint's model id.
        base_url: str | None  — the OpenAI-compatible endpoint. ``None`` ⇒ OpenAI; set to
            ``https://openrouter.ai/api/v1`` (OpenRouter) or ``http://<vllm-host>:8000/v1`` (on-prem
            / air-gapped) to swap substrate with no code change.
        api_key: str | None  — the key; falls back to ``$OPENAI_API_KEY``. A local vLLM accepts any
            placeholder.
        granularity: "doc" | "chunk" | "corpus"  — ``chunk`` (what ``latence setup`` emits, T1):
            one call per DISTINCT chunk text, and that chunk keeps its OWN small, relevant label
            set — the extraction path (a chunk is then extracted against few, relevant types
            instead of a broadcast ~40-type union). ``doc``: one LLM call per document, its induced
            labels broadcast to all its chunks. ``corpus`` (W12): induce per-document (the same
            concurrent fan-out), then MERGE every document's induced entity/relation/pii label sets
            into ONE canonical, deduped, document-frequency-ranked (salience-capped) schema and
            stamp that SINGLE shared schema onto EVERY chunk. Requires a full-corpus pass (the whole
            chunk stream is held to build the merged schema). NOTE (T1/T3): corpus mode is NOT how
            the KG earns a consistent type vocabulary — see the module docstring; it imposes
            consistency upstream at the cost of recall, and it is a corpus-level operation inside
            the chunk-level front half.
        max_tokens: int  — the completion budget (default 2048); a truncated reply retries at
            double.
        max_labels: int  — the per-type SALIENCE cap: the number the prompt asks for and the bound
            the corpus merge's document-frequency ranking truncates to (default 40, hard max 40).
        label_ceiling: int  — the per-type SAFETY ceiling the sanitizer enforces on ONE induction's
            reply. Defaults to :data:`~latence_schema_inducer.sanitize.CHUNK_LABEL_CEILING` (64) at
            ``chunk`` granularity and to ``max_labels`` at doc/corpus granularity, and is clamped to
            :data:`~latence_schema_inducer.sanitize.MAX_LABEL_CEILING` (256). Split from
            ``max_labels`` because the two are different decisions (T1): capping a chunk's own
            induced set at the corpus-salience number silently drops labels that chunk needs.
        dedup: bool  — induce once per DISTINCT job text and fan the result to its duplicates
            (default True, T5). Boilerplate/templated chunks are common in enterprise corpora, so
            this removes a large fraction of the per-chunk calls at zero quality cost: at
            temperature 0 identical prompts induce identical schemas, so a duplicate receives the
            byte-identical :class:`InducedLabels` it would have been sent anyway. Set False only
            for a non-deterministic endpoint where that equality does not hold.
        max_induction_calls: int | None  — a per-run BUDGET on how many distinct induction jobs may
            be dispatched (default None ⇒ unbounded). Jobs beyond the budget make no call and fail
            open to the config-label floor, with ``induction_budget_exhausted`` set and ONE loud
            WARNING — a bounded, visible degrade instead of a silent runaway bill. Enforced on the
            deterministic input-order job prefix, so a budgeted run is still byte-reproducible.
            (A *token* budget is deliberately NOT enforced mid-flight: which jobs it would cut
            depends on completion order, which would make the run's OUTPUT non-deterministic.
            Tokens are reported, not enforced.)
        temperature: float  — sampling temperature (default 0.0 for determinism).
        system_prompt / user_prompt_template: str  — override the default prompts (the template
            carries ``{text}`` / ``{max_labels}`` placeholders).
        domain: str  — an OPTIONAL free-text description of the corpus's subject matter (what
            ``latence setup --domain "German customs law: tariff classification, ..."`` emits). The
            Provider folds it into its OWN prompt as a clearly-fenced BACKGROUND block
            (``DOMAIN_CONTEXT_TEMPLATE``) on every call — including the repair / empty-recovery
            re-prompts — so the model names and disambiguates types in the operator's domain instead
            of guessing from the text alone. Untrusted text: it is sanitized
            (:func:`~latence_schema_inducer.sanitize.sanitize_domain` — control chars stripped,
            fence-safe, one line, ≤ 512 chars) and framed as reference-material-not-instructions,
            and every safety net still applies to the reply. Absent / empty ⇒ NO block is added and
            the prompts are byte-identical to a run without a hint.
        min_chars: int  — a chunk/document shorter than this (stripped) is skipped (default 1).
        min_labels: int  — the minimum TOTAL induced labels (entity + relation + pii) a successful
            induction must yield (default 1). An induction that parses + validates but sanitizes to
            fewer than this triggers ONE empty-recovery re-prompt (doubled budget), then a LOUD
            fall-back (``induction_empty_after_sanitization`` + WARNING) — never a silent degrade to
            the config-label floor. Raise it to demand a richer schema before accepting a result.
        retry_attempts / retry_base_delay / retry_max_delay  — the ported endpoint-retry policy
            knobs (identical to the other endpoint Providers); retries surface on
            ``endpoint_retries``.
        max_concurrency: int  — the bound on in-flight concurrent LLM calls (W6, default 8; 1 =
            sequential). The per-item calls fan out under an ``asyncio.Semaphore(max_concurrency)``.
        request_timeout: float | None  — a per-call timeout (seconds) passed to the async client
            (``None`` ⇒ the client default). A timeout is a transport fault → retried like a 5xx.
    The client is constructed lazily on first ``induce`` so building the Provider (which the Runner
    does for every Stage) never needs the endpoint reachable.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): an OpenAI-compatible LLM endpoint adapter.

        ``compute="either"``: a thin HTTP client on the local CPU (no local CUDA); the served LLM
        lives behind the endpoint. ``model_id=None``: ships no weights — the served model id is
        config-supplied (default ``gpt-4.1-mini`` is a served-model id, not a bundled checkpoint).
        License is framework code (Apache-2.0, verified; the ``openai`` client is Apache-2.0).
        ``deterministic=False`` (a served LLM is not byte-identical), though the Provider pins
        temperature 0 + stable order so a mocked run IS reproducible. ``cost_per_1k=None``: an LLM's
        per-call cost is vendor/model-specific, left undeclared rather than fabricated.
        """
        return ProviderProfile(
            compute="either",
            memory_mb=64,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-09",
            deterministic=False,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        granularity = str(config.get("granularity", "doc")).lower()
        if granularity not in ("doc", "chunk", "corpus"):
            msg = (
                "LLMLabelInducer 'granularity' must be 'doc', 'chunk' or 'corpus', "
                f"got {granularity!r}."
            )
            raise ValueError(msg)
        self._granularity = granularity
        # W17 (ADR-0043): the FROZEN induced schema. When a delta reuses a prior Corpus Version's
        # induced schema (the default, so extraction labels stay stable across the delta and an
        # unchanged document's extraction remains valid), the DeltaRunner sets this to the persisted
        # :class:`~latence_core.contracts.InducedLabels` dump; ``induce`` then STAMPS that single
        # frozen schema onto every chunk and makes NO LLM call (induction is short-circuited).
        # ``--reinduce`` omits it, so induction runs normally over the full current corpus. A
        # ``None``/absent value is the ordinary (non-frozen) path — byte-identical to pre-W17.
        frozen = config.get("frozen_induced_labels")
        self._frozen_induced_labels = (
            InducedLabels.model_validate(frozen) if isinstance(frozen, dict) else None
        )
        self._max_tokens = max(1, int(config.get("max_tokens", _DEFAULT_MAX_TOKENS)))
        self._max_labels = max(1, min(int(config.get("max_labels", MAX_LABELS)), MAX_LABELS))
        # T1: the SAFETY ceiling the sanitizer enforces on ONE reply, split from the salience cap.
        # On the per-chunk path the corpus-salience number (40) is the WRONG bound — a chunk's own
        # induced set is naturally small, so a cap there only ever drops labels that chunk actually
        # evidenced (silent recall loss). Default: the generous CHUNK_LABEL_CEILING on the chunk
        # path, ``max_labels`` on doc/corpus (byte-identical to pre-T1 there). Clamped to the hard
        # MAX_LABEL_CEILING so an operator can raise the ceiling but never remove it.
        default_ceiling = CHUNK_LABEL_CEILING if granularity == "chunk" else self._max_labels
        self._label_ceiling = max(
            1, min(int(config.get("label_ceiling", default_ceiling)), MAX_LABEL_CEILING)
        )
        # T5: induce once per DISTINCT job text, fanning the result to duplicates. Deterministic
        # and lossless at temperature 0 (identical prompt ⇒ identical schema), so it is ON by
        # default; the escape hatch is for a non-deterministic endpoint.
        self._dedup = bool(config.get("dedup", True))
        # T5: the per-run call budget (None ⇒ unbounded). See the class docstring for why this is a
        # CALL budget and not a token budget (determinism).
        budget = config.get("max_induction_calls")
        self._max_induction_calls = None if budget is None else max(0, int(budget))
        # The minimum TOTAL induced labels (entity + relation + pii) a successful induction must
        # yield; below this the result is "empty / near-empty after sanitization" → the Provider
        # RE-PROMPTS once before falling back (the robustness fix — an empty induction is the
        # noisy-output failure induce exists to fix, never a silent accept). Default 1: a genuinely
        # all-empty response triggers recovery; an operator can raise it to demand a richer schema.
        self._min_labels = max(1, int(config.get("min_labels", _DEFAULT_MIN_LABELS)))
        self._temperature = float(config.get("temperature", 0.0))
        self._system_prompt = str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        self._user_prompt_template = str(
            config.get("user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)
        )
        # The operator's optional domain hint, sanitized at construction (once, not per call) into
        # the bounded, fence-safe line the prompt block embeds. "" ⇒ no block is emitted at all, so
        # a config without `domain` produces byte-identical prompts to pre-domain behaviour.
        self._domain = sanitize_domain(config.get("domain", ""))
        self._min_chars = max(0, int(config.get("min_chars", _DEFAULT_MIN_CHARS)))
        # Ported retry (same as relation.llm): a 429/5xx/transport blip is retried with
        # backoff+jitter and tallied on ``endpoint_retries``; a terminal 4xx fails fast (and, being
        # a failure, fails OPEN to config labels — never crashes the run). Seeded RNG off the
        # endpoint identity so a test run is byte-stable while a real run de-correlates a burst.
        self._policy = RetryPolicy(
            attempts=int(config.get("retry_attempts", 4)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
        )
        self._rng = random.Random(f"{self._base_url}|{self._model}")
        # W6 concurrency: bound the in-flight per-item LLM calls. Clamped to >= 1 so a configured
        # 0/negative means "one at a time" (sequential), never a zero-permit deadlock.
        self._max_concurrency = max(1, int(config.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY)))
        # Per-call timeout (seconds) forwarded to AsyncOpenAI; None ⇒ the client default. A timeout
        # is a transport fault → classed retryable by the ported retry (same as a 5xx).
        request_timeout = config.get("request_timeout")
        self._request_timeout = None if request_timeout is None else float(request_timeout)
        self._client: Any = None
        # Observability seams the Runner folds into the Quality Report (read defensively — ADR-0004,
        # no base class). ``endpoint_retries``: transient endpoint failures retried through (same
        # attribute the other endpoint Providers expose). ``induction_fallbacks`` /
        # ``induction_documents``: how many documents fell back to config labels (a safety net bit)
        # out of how many were attempted — the W4 fail-open note.
        self.endpoint_retries = 0
        self.induction_fallbacks = 0
        self.induction_documents = 0
        # LOUD signal for the specific regression this fix targets: how many documents fell back
        # AFTER the empty-recovery re-prompt still yielded an empty / below-min label set (the
        # quality cliff the silent fallback used to hide). A subset of ``induction_fallbacks``;
        # surfaced on its own Quality-Report field so an operator SEES "the corpus induced nothing"
        # rather than only reading a generic fallback count. 0 when every induction produced labels.
        self.induction_empty_after_sanitization = 0
        # T5 — the INDUCTION COST columns the Runner folds into the Quality Report, so per-chunk
        # induction is affordable *and visibly* affordable (an operator must be able to read what a
        # run cost, not infer it). All counts only, never content:
        #   ``induction_calls``       — logical chat-completion requests issued (one per attempt;
        #                               a truncation / repair / empty-recovery re-prompt is its own
        #                               request. Transport-level retries are counted separately on
        #                               ``endpoint_retries``, so the two do not double-count).
        #   ``induction_cache_hits``  — jobs served from the chunk-text dedup cache (no call made).
        #                               ``hits / (hits + distinct jobs)`` is the cache-hit rate.
        #   ``induction_prompt_tokens`` / ``induction_completion_tokens`` — summed from each
        #                               response's ``usage`` block when the endpoint reports one.
        #                               An endpoint that reports no usage leaves them 0 — an honest
        #                               "unknown", never an estimated/fabricated token count.
        #   ``induction_budget_exhausted`` — True iff ``max_induction_calls`` bound the run (jobs
        #                               were dropped to the config-label floor). The LOUD bit.
        self.induction_calls = 0
        self.induction_cache_hits = 0
        self.induction_prompt_tokens = 0
        self.induction_completion_tokens = 0
        self.induction_budget_exhausted = False

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'openai' package is required to run LLMLabelInducer. "
                "Install it with: pip install latence-schema-inducer"
            )
            raise ImportError(msg) from exc
        # Key sourcing (config > endpoint-appropriate env > OpenAI env > keyless local vLLM). An
        # OpenRouter base_url prefers $OPENROUTER_API_KEY so a client's OpenRouter key is not
        # confusingly parked in $OPENAI_API_KEY; a keyless on-prem vLLM accepts "not-needed".
        env_names = (
            ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
            if (self._base_url and "openrouter" in self._base_url)
            else ("OPENAI_API_KEY",)
        )
        env_key = next((os.environ[n] for n in env_names if os.environ.get(n)), None)
        api_key = self._api_key or env_key or "not-needed"
        # The ASYNC client (W6): the per-item calls fan out concurrently under one event loop per
        # induce() run. A per-call ``timeout`` is passed only when configured (None ⇒ client
        # default). base_url/key sourcing is unchanged, so the OpenAI/OpenRouter/vLLM swap is
        # byte-identical to the sync path.
        kwargs: dict[str, Any] = {"base_url": self._base_url, "api_key": api_key}
        if self._request_timeout is not None:
            kwargs["timeout"] = self._request_timeout
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    # -- induction ----------------------------------------------------------

    def induce(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Annotate each chunk with its document's induced labels (chunk→chunk, 1:1, order-stable).

        Doc granularity (default): group chunks by ``document_record_id`` in first-seen order, make
        ONE LLM call per document, and broadcast the induced labels to every chunk of that document.
        Chunk granularity: one call per chunk. Corpus granularity (W12): induce per document, MERGE
        every document's induced labels into ONE canonical shared schema, and stamp it on EVERY
        chunk. Every chunk is yielded exactly once in input order; a chunk whose document induction
        failed (fail-open) or which was too small flows through unchanged (``induced_labels`` stays
        ``None`` in doc/chunk mode; in corpus mode it still receives the shared corpus schema), so a
        downstream extractor uses its config labels and the run never crashes. The client is built
        lazily on the first chunk that needs it — a fully-empty / too-small run makes no network
        call.
        """
        chunk_list = list(chunks)
        if self._frozen_induced_labels is not None:
            # W17: reuse the frozen corpus schema — stamp it on every chunk, make NO LLM call, so
            # the delta's freshly-extracted documents use the SAME labels as the reused documents
            # (the coupling ADR-0043 breaks). Order-preserving, 1:1, exactly like the induce paths.
            frozen = self._frozen_induced_labels
            for chunk in chunk_list:
                yield chunk.model_copy(update={"induced_labels": frozen})
            return
        if self._granularity == "chunk":
            yield from self._induce_per_chunk(chunk_list)
            return
        if self._granularity == "corpus":
            yield from self._induce_corpus(chunk_list)
            return
        yield from self._induce_per_document(chunk_list)

    def _induce_per_document(self, chunk_list: list[ChunkRecord]) -> Iterator[ChunkRecord]:
        # Assemble each document's text (in chunk order) once, induce a schema per document
        # concurrently, then broadcast to that document's chunks. Documents keep first-seen order.
        by_doc: dict[str, list[ChunkRecord]] = {}
        order: list[str] = []
        for chunk in chunk_list:
            if chunk.document_record_id not in by_doc:
                by_doc[chunk.document_record_id] = []
                order.append(chunk.document_record_id)
            by_doc[chunk.document_record_id].append(chunk)

        # Build the induction jobs (index-keyed by document) for the ones big enough to induce.
        # A too-small document maps to None with no LLM call (its key is simply absent from jobs).
        jobs: list[tuple[str, str]] = []
        for doc_id in order:
            text = "\n".join(c.content for c in by_doc[doc_id]).strip()
            if len(text) >= self._min_chars:
                jobs.append((doc_id, text))

        induced_by_doc = self._induce_all(jobs)
        for chunk in chunk_list:
            induced = induced_by_doc.get(chunk.document_record_id)
            yield chunk if induced is None else chunk.model_copy(update={"induced_labels": induced})

    def _induce_corpus(self, chunk_list: list[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Induce per document, MERGE all docs into ONE canonical schema, stamp it on EVERY chunk.

        NOT the default path (T1): this is the opt-in "one frozen vocabulary for the whole corpus"
        mode. Its salience cap drops rare-but-locally-critical types and its broadcast schema scores
        every chunk against types that are not in it; the type consistency it buys is now earned
        downstream by type canonicalization in the resolution phase (T3) instead.

        The W12 corpus mode, a clean collect→broadcast two-phase over the same async fan-out:

          1. **Collect**: assemble each document's text (chunk order) and induce a schema per
             document CONCURRENTLY (the identical ``_induce_all`` fan-out doc mode uses — same
             per-doc fail-open, temperature-0 determinism, index-keyed reassembly).
          2. **Merge**: union every successfully-induced document's entity/relation/pii label sets
             into ONE schema — canonical normal form (already applied per label by the sanitizer),
             deduped, then ranked by DOCUMENT FREQUENCY (the corpus's own salience signal) and
             capped at ``max_labels`` (ties broken by label ascending). Ranking by frequency — not
             alphabetical order — keeps the corpus's most-agreed-on types inside the cap; the
             frequency→label tie-break makes the corpus schema a byte-stable function of the
             label→frequency map, independent of document/completion order.
          3. **Broadcast**: stamp that single shared schema onto EVERY chunk — including the chunks
             of a document whose own induction failed open, so the whole corpus is extracted
             against one vocabulary.

        What this mode does NOT do any more (T1/T3): it is no longer how the KG earns a consistent
        type vocabulary. That guarantee now comes from corpus-wide type canonicalization in the
        resolution phase, downstream of extraction, where it costs no recall. Corpus mode here only
        trades recall (the salience cap drops what a single chunk needed) and zero-shot precision
        (every chunk is scored against types that are not in it) for an upstream uniformity the
        pipeline no longer depends on.

        Requires the whole chunk stream in one pass (to build the merged schema before
        broadcasting); acceptable for the small-LLM-call induction Stage. When NO document induced
        anything usable
        the corpus schema is empty → every chunk flows through with ``induced_labels`` unchanged
        (``None``), the byte-identical fall-back-to-config path.
        """
        by_doc: dict[str, list[ChunkRecord]] = {}
        order: list[str] = []
        for chunk in chunk_list:
            if chunk.document_record_id not in by_doc:
                by_doc[chunk.document_record_id] = []
                order.append(chunk.document_record_id)
            by_doc[chunk.document_record_id].append(chunk)

        jobs: list[tuple[str, str]] = []
        for doc_id in order:
            text = "\n".join(c.content for c in by_doc[doc_id]).strip()
            if len(text) >= self._min_chars:
                jobs.append((doc_id, text))

        induced_by_doc = self._induce_all(jobs)
        corpus_schema = self._merge_corpus_schema(induced_by_doc.values())
        for chunk in chunk_list:
            yield chunk if corpus_schema is None else chunk.model_copy(
                update={"induced_labels": corpus_schema}
            )

    def _merge_corpus_schema(
        self, induced_items: Iterable[InducedLabels | None]
    ) -> InducedLabels | None:
        """Merge every document's induced labels into ONE canonical, salience-ranked, capped schema.

        Each input :class:`InducedLabels` already carries canonical-normal-form, per-doc-deduped
        labels (the sanitizer ran per document), so the merge counts, per slot, the DOCUMENT
        FREQUENCY of each label — how many documents independently induced it — and keeps the
        ``max_labels`` MOST salient. Ranking by frequency (not the label's alphabetical position)
        is the fix for the corpus-wide truncation defect: when the union exceeds ``max_labels``, a
        plain ``sorted()[:cap]`` drops every label sorting after the cut point ALPHABETICALLY,
        silently discarding a corpus's most explicit high-frequency relations (e.g. ``obsoletes`` /
        ``updates`` / ``references``, all sorting after ``~"c"``) while keeping rare early noise.
        Document frequency is the corpus's own salience signal, so the cap now keeps what the corpus
        most agrees on. Ties (equal frequency) break by canonical label ascending, so the merged
        schema stays a byte-stable function of the label→frequency map, independent of document /
        completion order. A wholly-empty merge (no document induced anything) returns ``None`` so
        corpus mode falls back to config labels rather than stamping an empty schema — the
        byte-identical backward-compat path.
        """
        entity: Counter[str] = Counter()
        relation: Counter[str] = Counter()
        pii: Counter[str] = Counter()
        for induced in induced_items:
            if induced is None:
                continue
            # Each slot's labels are already per-doc deduped by the sanitizer, so one document
            # contributes at most +1 to a label's document frequency; ``set()`` is defence-in-depth
            # against a hand-built InducedLabels that repeats a label within one slot.
            entity.update(set(induced.entity_types))
            relation.update(set(induced.relation_types))
            pii.update(set(induced.pii_types))
        merged_entity = self._rank_and_cap(entity)
        merged_relation = self._rank_and_cap(relation)
        merged_pii = self._rank_and_cap(pii)
        if not merged_entity and not merged_relation and not merged_pii:
            return None
        return InducedLabels(
            entity_types=merged_entity,
            relation_types=merged_relation,
            pii_types=merged_pii,
            model_id=self._model,
        )

    def _rank_and_cap(self, frequencies: Counter[str]) -> list[str]:
        """The ``max_labels`` most document-frequent labels, ties broken by label ascending.

        Ranks each label by DOCUMENT FREQUENCY descending (the corpus's own salience signal — how
        many documents independently induced it), breaking ties by the canonical label ascending so
        the result is a deterministic, byte-stable function of the frequency map (never of document
        or completion order). Capping the salience-ranked list — rather than an alphabetically
        ``sorted()`` one — is what keeps a corpus's most explicit, highest-agreement relations
        (which may sort alphabetically-late, e.g. ``obsoletes`` / ``updates``) inside the cap
        instead of silently discarding them for alphabetically-early rare noise.
        """
        ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        return [label for label, _count in ranked[: self._max_labels]]

    def _induce_per_chunk(self, chunk_list: list[ChunkRecord]) -> Iterator[ChunkRecord]:
        """The T1 extraction path: every chunk keeps ONLY the types induced for that chunk.

        No corpus merge and no salience cap — a chunk that legitimately evidences 15 types keeps
        all 15 (bounded only by the ``label_ceiling`` safety net), and a type that appears in
        exactly one chunk of the corpus survives on that chunk instead of being ranked out of a
        corpus-wide top-40. Cost is bounded by the dedup + budget layers in :meth:`_induce_all`,
        not by throwing labels away.
        """
        # One job per big-enough chunk, index-keyed by the chunk record_id (unique). Results are
        # reassembled in input order below, so completion order never perturbs the output.
        jobs: list[tuple[str, str]] = []
        for chunk in chunk_list:
            text = chunk.content.strip()
            if len(text) >= self._min_chars:
                jobs.append((chunk.record_id, text))

        induced_by_key = self._induce_all(jobs)
        for chunk in chunk_list:
            induced = induced_by_key.get(chunk.record_id)
            yield chunk if induced is None else chunk.model_copy(
                update={"induced_labels": induced}
            )

    def _induce_all(self, jobs: list[tuple[str, str]]) -> dict[str, InducedLabels | None]:
        """Induce every job's schema CONCURRENTLY (bounded), keyed by its key for order-stable use.

        Three layers, in this order — the cost path (T5) wraps the W6 fan-out:

        1. **Text dedup** (``dedup``, on by default): the jobs are collapsed to their DISTINCT
           texts in first-seen order, one induction runs per distinct text, and every duplicate
           receives that same :class:`InducedLabels` object. Lossless at temperature 0 (identical
           prompt ⇒ identical schema) and the single largest cost lever on a real corpus, where
           boilerplate headers/footers and templated clauses repeat across thousands of chunks.
           ``induction_cache_hits`` records how many jobs were served without a call.
        2. **Budget** (``max_induction_calls``): only the first N distinct jobs — the deterministic
           input-order prefix — are dispatched; the rest fail open to the config-label floor with
           ``induction_budget_exhausted`` and ONE loud WARNING. Taking a *prefix* (rather than
           whichever jobs happened to finish first) is what keeps a budget-bound run reproducible.
        3. **Fan-out** (W6): each surviving job's induction runs under an
           ``asyncio.Semaphore(max_concurrency)``, gathered with ``return_exceptions=True`` so one
           item's failure fails open for THAT item only (never aborts the batch). One event loop
           per ``induce`` run via a single :func:`asyncio.run`.

        ``induction_documents`` counts every attempted job (deduped or not) — the denominator an
        operator reads the fallback count against. Reassembly is key-indexed (the returned dict), so
        the caller re-emits in input order — byte-identical whatever the completion order.
        """
        if not jobs:
            return {}
        self.induction_documents += len(jobs)

        # -- 1. dedup: distinct texts in first-seen order ------------------------
        # ``slot_of_job[i]`` is the index in ``distinct`` whose induction answers job i. With dedup
        # off every job gets its own slot, so the rest of the method is one code path either way.
        distinct: list[str] = []
        slot_of_text: dict[str, int] = {}
        slot_of_job: list[int] = []
        for _key, text in jobs:
            slot = slot_of_text.get(text) if self._dedup else None
            if slot is None:
                slot = len(distinct)
                distinct.append(text)
                slot_of_text[text] = slot
            slot_of_job.append(slot)
        self.induction_cache_hits += len(jobs) - len(distinct)

        # -- 2. budget: the deterministic input-order prefix ---------------------
        dispatched = distinct
        if self._max_induction_calls is not None and len(distinct) > self._max_induction_calls:
            dispatched = distinct[: self._max_induction_calls]
            self._record_budget_exhausted(len(distinct) - len(dispatched))

        client = self._get_client() if dispatched else None

        async def _run() -> list[InducedLabels | None]:
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def _one(text: str) -> InducedLabels | None:
                async with semaphore:
                    return await self._induce_one(client, text)

            gathered = await asyncio.gather(
                *(_one(text) for text in dispatched), return_exceptions=True
            )
            # return_exceptions=True: a genuinely-unexpected exception that escaped _induce_one's
            # own fail-open guard (belt-and-suspenders) fails open for THAT item — never the batch.
            resolved: list[InducedLabels | None] = []
            for outcome in gathered:
                if isinstance(outcome, BaseException):
                    self._record_fallback(f"unexpected concurrent error ({type(outcome).__name__})")
                    resolved.append(None)
                else:
                    resolved.append(outcome)
            return resolved

        induced = asyncio.run(_run()) if dispatched else []
        # Budget-dropped jobs fail open (None), exactly like any other safety-net miss.
        induced.extend([None] * (len(distinct) - len(induced)))

        # -- 3. fan the per-text result back to every job that carried that text --
        return {key: induced[slot] for (key, _text), slot in zip(jobs, slot_of_job, strict=True)}

    async def _induce_one(self, client: Any, text: str) -> InducedLabels | None:
        """Induce one document's label schema with EVERY safety net; None ⇒ fall back to config.

        Runs the structured LLM call, detects truncation (retry at double budget), parses (repair
        retry on malformed JSON), validates the schema, and sanitizes the labels. ANY failure at any
        step returns ``None`` (fail-open): the caller leaves the chunks' ``induced_labels`` unset so
        the extractors use their config labels. A typed, no-content log records which safety net
        fired and the ``induction_fallbacks`` tally is bumped — the Quality-Report note — but the
        failing document's content is NEVER logged (the no-PII discipline).

        Robustness fix: an induction that parses + validates but sanitizes to an EMPTY / below-min
        label set is NOT silently accepted (that quiet degrade to the bare config floor is the
        quality cliff induce exists to prevent). It re-prompts the model ONCE (the empty-recovery
        instruction at a doubled budget — the same precedent as the truncation retry) before falling
        back, and a still-empty result falls back LOUDLY (``induction_empty_after_sanitization`` +
        the WARNING), never a quiet revert.
        """
        induced, reason = await self._induce_attempt(client, text, recover=False)
        if induced is not None:
            return induced
        if reason == _EMPTY_AFTER_SANITIZATION:
            # Don't accept an empty induction — re-prompt once, then fall back LOUDLY if still bare.
            induced, reason = await self._induce_attempt(client, text, recover=True)
            if induced is not None:
                return induced
            if reason == _EMPTY_AFTER_SANITIZATION:
                self._record_empty_after_sanitization()
                return None
        # Any other safety-net miss (endpoint / no-JSON / schema-invalid), or an empty→retry that
        # then failed for a different reason: the ordinary fail-open note.
        self._record_fallback(reason or "unknown inducer failure")
        return None

    async def _induce_attempt(
        self, client: Any, text: str, *, recover: bool
    ) -> tuple[InducedLabels | None, str | None]:
        """One full induce attempt: call → parse → validate → sanitize; ``(labels, None)`` on ok.

        On failure returns ``(None, reason)`` where ``reason`` is the safety-net tag; the sentinel
        :data:`_EMPTY_AFTER_SANITIZATION` distinguishes the "valid but empty/below-min" case (which
        the caller RETRIES) from the other misses (which fall back directly). ``recover=True`` runs
        the empty-recovery variant of the call (the :data:`EMPTY_RECOVERY_INSTRUCTION` prompt at a
        doubled budget). No tally is bumped here — the caller decides retry-vs-fallback and records.
        """
        try:
            raw = await self._call_with_truncation_and_repair(client, text, recover=recover)
        except EndpointError as exc:
            # Network / auth / terminal-4xx / unreachable endpoint: fail open, never crash.
            return None, f"endpoint error ({exc.__class__.__name__})"
        except Exception as exc:  # noqa: BLE001 - fail-open is the whole point; no bad output crashes
            return None, f"unexpected inducer error ({exc.__class__.__name__})"
        if raw is None:
            return None, "no valid JSON after repair/truncation retry"
        validated = _validate_schema(raw)
        if validated is None:
            return None, "schema-invalid (not {entity,relation,pii} lists of strings)"
        induced = self._sanitize_to_induced(validated)
        if induced is None or _total_labels(induced) < self._min_labels:
            return None, _EMPTY_AFTER_SANITIZATION
        return induced, None

    def _record_fallback(self, reason: str) -> None:
        """Record a fail-open: bump the tally + emit a typed, no-content log (no PII)."""
        self.induction_fallbacks += 1
        # No document content in the log — only the safety-net reason + the model id (no-PII).
        _LOG.warning(
            "schema_induction fell back to config labels: %s (model=%s)", reason, self._model
        )

    def _record_empty_after_sanitization(self) -> None:
        """Record the LOUD empty-induction signal: the dedicated counter PLUS the ordinary fallback.

        Fired only when the empty-recovery re-prompt ALSO sanitized empty/below-min — the corpus
        genuinely induced nothing usable. Bumps ``induction_empty_after_sanitization`` (its own
        Quality-Report field so an operator sees the quality regression directly) and then the
        normal ``_record_fallback`` (it IS a fallback, and emits the WARNING) — never a silent
        revert to the bare config floor.
        """
        self.induction_empty_after_sanitization += 1
        self._record_fallback("empty after sanitization (persisted through the recovery re-prompt)")

    def _record_budget_exhausted(self, skipped: int) -> None:
        """Record the T5 call-budget bound: N jobs dropped to the config-label floor, ONE warning.

        A budget that silently ate a third of a corpus's induction would be exactly the invisible
        degrade this Provider exists to prevent, so the bound sets its own Quality-Report flag
        (``induction_budget_exhausted``) AND tallies the dropped jobs as fallbacks (they DID fall
        back). One aggregate WARNING rather than one per job — a bounded run must not also produce
        an unbounded log.
        """
        self.induction_budget_exhausted = True
        self.induction_fallbacks += skipped
        _LOG.warning(
            "schema_induction budget exhausted: %d job(s) beyond max_induction_calls=%s fell back "
            "to config labels (model=%s)",
            skipped,
            self._max_induction_calls,
            self._model,
        )

    # -- the structured call + truncation + repair (safety nets #1, #2, #3) --

    async def _call_with_truncation_and_repair(
        self, client: Any, text: str, *, recover: bool = False
    ) -> dict[str, Any] | None:
        """Return the parsed JSON object, or ``None`` when repair/truncation retries are exhausted.

        Applies safety nets #2 (truncation) and #3 (parse-and-repair) in one bounded sequence, each
        allowing exactly ONE corrective retry (no unbounded loops):

          * Attempt 1: normal call. If ``finish_reason == length`` → retry ONCE at double budget; if
            still truncated → return ``None`` (fall back). If the body does not parse → repair.
          * The repair retry prepends a terse "return ONLY valid JSON of this exact shape"
            instruction (safety net #3). A second parse failure → return ``None`` (fall back).

        ``recover=True`` is the empty-recovery variant (the robustness fix): it prepends the
        :data:`EMPTY_RECOVERY_INSTRUCTION` (re-ask the model to actually extract the salient types)
        and starts at a DOUBLED budget, so the re-prompt has both a sharper instruction and room
        — the same "double the budget on retry" precedent as the truncation net.

        The network call itself rides the ported retry (:func:`call_with_retry`) for transient
        429/5xx/transport faults; a terminal fault raises :class:`EndpointError` up to the fail-open
        guard. NEVER ``eval`` — only ``json.loads`` in a guard.
        """
        base_tokens = self._max_tokens * 2 if recover else self._max_tokens
        instruction = EMPTY_RECOVERY_INSTRUCTION if recover else None
        # --- attempt 1 (with a one-shot doubled-budget retry on truncation) ---
        try:
            content = await self._chat(
                client, text, max_tokens=base_tokens, instruction=instruction
            )
        except _Truncated:
            try:
                content = await self._chat(
                    client, text, max_tokens=base_tokens * 2, instruction=instruction
                )
            except _Truncated:
                _LOG.warning(
                    "schema_induction reply truncated twice (max_tokens=%d then %d)",
                    base_tokens,
                    base_tokens * 2,
                )
                return None
        parsed = _try_parse(content)
        if parsed is not None:
            return parsed
        # --- one corrective repair retry (safety net #3) ---
        # On an empty-recovery attempt keep re-asserting the recovery ask alongside the JSON-shape
        # repair, so a re-prompt that ALSO returns malformed JSON still steers toward real types.
        repair_instruction = (
            REPAIR_INSTRUCTION
            if instruction is None
            else f"{EMPTY_RECOVERY_INSTRUCTION}\n\n{REPAIR_INSTRUCTION}"
        )
        try:
            repaired = await self._chat(
                client, text, max_tokens=base_tokens * 2, instruction=repair_instruction
            )
        except _Truncated:
            return None
        return _try_parse(repaired)

    async def _chat(
        self, client: Any, text: str, *, max_tokens: int, instruction: str | None
    ) -> str:
        """One structured chat-completions call; returns the reply string or raises on truncation.

        Requests ``response_format={"type":"json_object"}`` (safety net #1, portable) at
        temperature 0 (determinism). Raises :class:`_Truncated` when ``finish_reason == "length"``
        (safety net #2) so the caller retries at a bigger budget. The network call rides
        :func:`call_with_retry`; any client exception is normalized to a classed
        :class:`EndpointError` so the retry decides retry-vs-fail and a terminal failure re-raises
        up to the fail-open guard. When ``instruction`` is not ``None`` it is prepended to the user
        message (the repair and/or empty-recovery corrective).

        A configured ``domain`` hint is folded in as a fenced BACKGROUND block between any
        corrective instruction and the task itself: the corrective keeps top billing (it is the
        sharpened ask), the domain frames the corpus, and the task's own restatement of the JSON
        shape lands LAST — closest to the model's answer. The block rides every call (including the
        retries), so a re-prompt never loses the domain steer. No hint ⇒ the user message is
        byte-identical to the pre-domain build.
        """
        user = self._user_prompt_template.format(text=text, max_labels=self._max_labels)
        if self._domain:
            user = f"{DOMAIN_CONTEXT_TEMPLATE.format(domain=self._domain)}\n\n{user}"
        if instruction is not None:
            user = f"{instruction}\n\n{user}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

        self.induction_calls += 1

        async def do_call() -> tuple[str, str | None]:
            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - normalized into a classed EndpointError
                raise _classify_exception(exc) from exc
            self._count_usage(response)
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            return str(content), (str(finish_reason) if finish_reason is not None else None)

        result = await call_with_retry_async(
            do_call,
            self._policy,
            rng=self._rng.random,
            sleep=_async_noop_sleep,
            on_retry=self._count_retry,
        )
        content, finish_reason = cast("tuple[str, str | None]", result)
        if finish_reason == "length":
            raise _Truncated
        return content

    def _count_retry(self) -> None:
        self.endpoint_retries += 1

    def _count_usage(self, response: Any) -> None:
        """Add one response's reported token usage to the run's induction-cost tally (T5).

        Read DEFENSIVELY off the OpenAI ``usage`` block (``prompt_tokens`` / ``completion_tokens``):
        OpenAI and OpenRouter always report it, but an on-prem vLLM can be configured not to. A
        missing / non-int field contributes 0 — the honest "the endpoint did not tell us", never an
        estimated token count dressed up as a measured one (a fabricated cost figure is worse than
        an absent one, and an operator reading 0 tokens next to N calls can tell the difference).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", 0)
        if isinstance(prompt, int):
            self.induction_prompt_tokens += prompt
        if isinstance(completion, int):
            self.induction_completion_tokens += completion

    # -- schema validation (#4) + sanitization (#5) -------------------------

    def _sanitize_to_induced(self, validated: dict[str, list[str]]) -> InducedLabels | None:
        """Sanitize each slot's labels + build an :class:`InducedLabels`; ``None`` if wholly empty.

        Every slot is bounded by :mod:`~latence_schema_inducer.sanitize` (control-char strip, length
        ≤ 64, dedupe, count ≤ ``label_ceiling``). A wholly-empty result (the model returned only
        junk / nothing usable) returns ``None``; the caller then RE-PROMPTS once (empty-recovery)
        before falling back to config labels — an empty schema is never silently accepted.
        Dropped-for-count overflow is logged (no silent drop).

        The count bound here is the SAFETY ceiling (``label_ceiling``), not the salience cap
        (``max_labels``) — see the class docstring. On the per-chunk path they differ deliberately:
        the salience cap belongs to the corpus merge, and applying it to one chunk's own reply is
        the silent recall loss T1 removes.
        """
        results = {
            slot: sanitize_labels(validated.get(slot, []), max_labels=self._label_ceiling)
            for slot in _SLOTS
        }
        total_dropped = sum(r.dropped_for_count for r in results.values())
        if total_dropped:
            _LOG.info(
                "schema_induction capped %d label(s) at the per-type safety ceiling (%d)",
                total_dropped,
                self._label_ceiling,
            )
        entity = results["entity_types"].labels
        relation = results["relation_types"].labels
        pii = results["pii_types"].labels
        if not entity and not relation and not pii:
            return None
        return InducedLabels(
            entity_types=entity,
            relation_types=relation,
            pii_types=pii,
            model_id=self._model,
        )


def _loads_object(content: str) -> dict[str, Any] | None:
    """``json.loads`` in a guard: return the parsed value iff it is a dict, else ``None``.

    NEVER ``eval`` and NEVER execute model text — only ``json.loads``. A non-JSON body, a JSON
    array/scalar at the top level, or a decode error all yield ``None``.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_object_candidates(content: str) -> Iterator[str]:
    """Yield substrings that MIGHT be the JSON object, most-likely first (the parse-hardening path).

    A conformant model returns a bare JSON object (``response_format=json_object``), but a real
    OpenRouter/vLLM model sometimes wraps it — in a Markdown fence (```` ```json … ``` ````) or in a
    line of prose ("Here are the types: { … }"). Rather than reject that wholesale (a repair round
    trip that often re-wraps, then a silent fall-back that DISCARDS perfectly good labels), we peel
    the common wrappers and re-try the strict parser on each candidate. This never LOOSENS parsing —
    every candidate still goes through strict :func:`_loads_object`, which rejects anything that is
    not a JSON object, so genuine garbage (single-quoted pseudo-JSON, prose with no braces) still
    fails and falls back. It only changes WHERE we look for the object.
    """
    # 1) The body inside a fenced code block (```json … ``` or bare ``` … ```), if present.
    for match in _FENCE_RE.finditer(content):
        inner = match.group("body").strip()
        if inner:
            yield inner
    # 2) The first ``{`` through the last ``}`` — recovers an object embedded in surrounding prose.
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        yield content[start : end + 1]


def _try_parse(content: str) -> dict[str, Any] | None:
    """Parse the reply into a JSON object, tolerant of fenced / prose-wrapped bodies (net #3).

    Tries the strict parser on the raw body first (the conformant case), then on each wrapper-peeled
    candidate (:func:`_json_object_candidates`). Returns the first dict that parses, or ``None`` if
    none do (the caller repairs / falls back). Still NEVER ``eval`` — only strict ``json.loads`` on
    each candidate — so hallucinated noise is still rejected; the tolerance is only about extracting
    a genuine JSON object the model buried in prose or a code fence.
    """
    direct = _loads_object(content)
    if direct is not None:
        return direct
    for candidate in _json_object_candidates(content):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _total_labels(induced: InducedLabels) -> int:
    """Total induced labels across all three slots — the ``min_labels`` emptiness gauge."""
    return len(induced.entity_types) + len(induced.relation_types) + len(induced.pii_types)


def _validate_schema(obj: dict[str, Any]) -> dict[str, list[str]] | None:
    """Safety net #4: the object MUST have each slot as a list of strings, else reject (``None``).

    Every one of ``entity_types`` / ``relation_types`` / ``pii_types`` must be present AND a list
    whose every element is a ``str`` (a list of ints, a string-not-list, or a missing key all
    reject). Extra keys are ignored. Returning ``None`` triggers the Provider's fall-back to config
    labels — a schema-invalid reply never becomes a fabricated annotation.
    """
    out: dict[str, list[str]] = {}
    for slot in _SLOTS:
        if slot not in obj:
            return None
        value = obj[slot]
        if not isinstance(value, list):
            return None
        if not all(isinstance(item, str) for item in value):
            return None
        out[slot] = list(value)
    return out


def _classify_exception(exc: Exception) -> EndpointError:
    """Normalize any client exception into a classed :class:`EndpointError` for the ported retry.

    An exception exposing an HTTP ``status_code`` (the OpenAI client's ``APIStatusError`` shape) is
    classed by that status (429/5xx retryable, other 4xx terminal); anything else is a transport
    error → retryable. Mirrors the other endpoint Providers exactly. A terminal / exhausted failure
    re-raises up to the Provider's fail-open guard (which returns config labels, never crashes).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return EndpointError(str(exc), status=status)
    return EndpointError(str(exc))

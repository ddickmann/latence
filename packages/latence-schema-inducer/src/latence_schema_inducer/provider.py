"""Unsupervised schema-induction LabelInducer over any OpenAI-compatible endpoint (W4, ADR-0038).

Restores the "point it at a folder and it discovers its own schema" property: an OPTIONAL
``schema_induction`` Stage that uses a small OpenAI-compatible LLM with **structured JSON output**
to induce, per document, the entity / relation / PII **types** actually evidenced in the data, then
annotates every :class:`~latence_core.contracts.ChunkRecord` of that document with an
:class:`~latence_core.contracts.InducedLabels` carrier. The extraction Stages then UNION those
induced types onto their config labels (the ``latence_core.induced`` seam) — no extractor rewrite.

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
     ("return ONLY valid JSON of this exact shape"); a second failure falls back. NEVER ``eval``,
     NEVER execute model text.
  4. **Schema validation**: the parsed object MUST be ``{entity_types,relation_types,pii_types}``,
     each a list of strings; anything else → reject → fall back. Extra keys are ignored.
  5. **Label sanitization** (:mod:`~latence_schema_inducer.sanitize`): strip control chars, cap
     length ≤ 64, drop empties, dedupe case-insensitively, cap count ≤ 40 (dropped items logged).
  6. **Fail-open, never crash** (G1 posture): ANY inducer failure (network, auth, malformed,
     truncated-twice, schema-invalid, empty) leaves the document's chunks' ``induced_labels`` as
     ``None`` so the extractors use their config labels and the run continues; the failure is a
     typed, no-content log + a Quality-Report note (``induction_fallbacks``). A missing/unreachable
     endpoint is a skip-with-flag (fail-open), not a crash.

Deterministic: ``temperature`` 0 and stable label order (sanitization preserves first-seen order),
so a seeded run is byte-reproducible (Baseline bar). The real ``gpt-4.1-mini`` numbers come from the
maintainer's keyed pod run (``scripts/validate-schema-induction-on-gpu.md``); the offline mock
proves the machinery + every safety net.

Licensing: this Provider is framework code (Apache-2.0) and calls a user-supplied endpoint; it
bundles NO model weights. The ``openai`` client is Apache-2.0 (code). No key or endpoint is embedded
— both come from config/environment. The ``openai`` import is deferred to first use so importing
this module (for Provider discovery) never requires the client installed.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import TYPE_CHECKING, Any, cast

from latence_core.contracts import InducedLabels
from latence_core.endpoint import EndpointError, RetryPolicy, call_with_retry
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

from latence_schema_inducer.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    REPAIR_INSTRUCTION,
)
from latence_schema_inducer.sanitize import MAX_LABELS, sanitize_labels

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

_LOG = logging.getLogger("latence.schema_induction")

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


class _Truncated(Exception):
    """Internal signal: the reply's ``finish_reason`` was ``length`` (truncated, safety net #2)."""


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
        granularity: "doc" | "chunk"  — one LLM call per document (default, cheaper, plenty
            data-aware; the doc's induced labels broadcast to all its chunks) or one call per chunk.
        max_tokens: int  — the completion budget (default 2048); a truncated reply retries at
        double.        max_labels: int  — the per-type label-count cap forwarded to the prompt +
        sanitizer (default            40).
        temperature: float  — sampling temperature (default 0.0 for determinism).
        system_prompt / user_prompt_template: str  — override the default prompts (the template
            carries ``{text}`` / ``{max_labels}`` placeholders).
        min_chars: int  — a chunk/document shorter than this (stripped) is skipped (default 1).
        retry_attempts / retry_base_delay / retry_max_delay  — the ported endpoint-retry policy
            knobs (identical to the other endpoint Providers); retries surface on
            ``endpoint_retries``.
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
        if granularity not in ("doc", "chunk"):
            msg = (
                f"LLMLabelInducer 'granularity' must be 'doc' or 'chunk', got {granularity!r}."
            )
            raise ValueError(msg)
        self._granularity = granularity
        self._max_tokens = max(1, int(config.get("max_tokens", _DEFAULT_MAX_TOKENS)))
        self._max_labels = max(1, min(int(config.get("max_labels", MAX_LABELS)), MAX_LABELS))
        self._temperature = float(config.get("temperature", 0.0))
        self._system_prompt = str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        self._user_prompt_template = str(
            config.get("user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)
        )
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
        self._client: Any = None
        # Observability seams the Runner folds into the Quality Report (read defensively — ADR-0004,
        # no base class). ``endpoint_retries``: transient endpoint failures retried through (same
        # attribute the other endpoint Providers expose). ``induction_fallbacks`` /
        # ``induction_documents``: how many documents fell back to config labels (a safety net bit)
        # out of how many were attempted — the W4 fail-open note.
        self.endpoint_retries = 0
        self.induction_fallbacks = 0
        self.induction_documents = 0

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
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
        self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    # -- induction ----------------------------------------------------------

    def induce(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Annotate each chunk with its document's induced labels (chunk→chunk, 1:1, order-stable).

        Doc granularity (default): group chunks by ``document_record_id`` in first-seen order, make
        ONE LLM call per document, and broadcast the induced labels to every chunk of that document.
        Chunk granularity: one call per chunk. Every chunk is yielded exactly once in input order;
        a chunk whose document induction failed (fail-open) or which was too small flows through
        unchanged (``induced_labels`` stays ``None``), so a downstream extractor uses its config
        labels and the run never crashes. The client is built lazily on the first chunk that needs
        it — a fully-empty / too-small run makes no network call at all.
        """
        chunk_list = list(chunks)
        if self._granularity == "chunk":
            yield from self._induce_per_chunk(chunk_list)
            return
        yield from self._induce_per_document(chunk_list)

    def _induce_per_document(self, chunk_list: list[ChunkRecord]) -> Iterator[ChunkRecord]:
        # Assemble each document's text (in chunk order) once, induce a schema per document, then
        # broadcast to that document's chunks. Documents are processed in first-seen order.
        by_doc: dict[str, list[ChunkRecord]] = {}
        order: list[str] = []
        for chunk in chunk_list:
            if chunk.document_record_id not in by_doc:
                by_doc[chunk.document_record_id] = []
                order.append(chunk.document_record_id)
            by_doc[chunk.document_record_id].append(chunk)

        induced_by_doc: dict[str, InducedLabels | None] = {}
        client = None
        for doc_id in order:
            doc_chunks = by_doc[doc_id]
            text = "\n".join(c.content for c in doc_chunks).strip()
            if len(text) < self._min_chars:
                induced_by_doc[doc_id] = None
                continue
            self.induction_documents += 1
            if client is None:
                client = self._get_client()
            induced_by_doc[doc_id] = self._induce_one(client, text)

        for chunk in chunk_list:
            induced = induced_by_doc.get(chunk.document_record_id)
            yield chunk if induced is None else chunk.model_copy(update={"induced_labels": induced})

    def _induce_per_chunk(self, chunk_list: list[ChunkRecord]) -> Iterator[ChunkRecord]:
        client = None
        for chunk in chunk_list:
            text = chunk.content.strip()
            if len(text) < self._min_chars:
                yield chunk
                continue
            self.induction_documents += 1
            if client is None:
                client = self._get_client()
            induced = self._induce_one(client, text)
            yield chunk if induced is None else chunk.model_copy(
                update={"induced_labels": induced}
            )

    def _induce_one(self, client: Any, text: str) -> InducedLabels | None:
        """Induce one document's label schema with EVERY safety net; None ⇒ fall back to config.

        Runs the structured LLM call, detects truncation (retry at double budget), parses (repair
        retry on malformed JSON), validates the schema, and sanitizes the labels. ANY failure at any
        step returns ``None`` (fail-open): the caller leaves the chunks' ``induced_labels`` unset so
        the extractors use their config labels. A typed, no-content log records which safety net
        fired and the ``induction_fallbacks`` tally is bumped — the Quality-Report note — but the
        failing document's content is NEVER logged (the no-PII discipline).
        """
        try:
            raw = self._call_with_truncation_and_repair(client, text)
        except EndpointError as exc:
            # Network / auth / terminal-4xx / unreachable endpoint: fail open, never crash.
            self._record_fallback(f"endpoint error ({exc.__class__.__name__})")
            return None
        except Exception as exc:  # noqa: BLE001 - fail-open is the whole point; no bad output crashes
            self._record_fallback(f"unexpected inducer error ({exc.__class__.__name__})")
            return None
        if raw is None:
            # Parse/repair or truncation exhausted → already-fallen-back (logged + counted below).
            self._record_fallback("no valid JSON after repair/truncation retry")
            return None
        validated = _validate_schema(raw)
        if validated is None:
            self._record_fallback("schema-invalid (not {entity,relation,pii} lists of strings)")
            return None
        induced = self._sanitize_to_induced(validated)
        if induced is None:
            self._record_fallback("empty after sanitization")
            return None
        return induced

    def _record_fallback(self, reason: str) -> None:
        """Record a fail-open: bump the tally + emit a typed, no-content log (no PII)."""
        self.induction_fallbacks += 1
        # No document content in the log — only the safety-net reason + the model id (no-PII).
        _LOG.warning(
            "schema_induction fell back to config labels: %s (model=%s)", reason, self._model
        )

    # -- the structured call + truncation + repair (safety nets #1, #2, #3) --

    def _call_with_truncation_and_repair(self, client: Any, text: str) -> dict[str, Any] | None:
        """Return the parsed JSON object, or ``None`` when repair/truncation retries are exhausted.

        Applies safety nets #2 (truncation) and #3 (parse-and-repair) in one bounded sequence, each
        allowing exactly ONE corrective retry (no unbounded loops):

          * Attempt 1: normal call. If ``finish_reason == length`` → retry ONCE at double budget; if
            still truncated → return ``None`` (fall back). If the body does not parse → repair.
          * The repair retry prepends a terse "return ONLY valid JSON of this exact shape"
            instruction (safety net #3). A second parse failure → return ``None`` (fall back).

        The network call itself rides the ported retry (:func:`call_with_retry`) for transient
        429/5xx/transport faults; a terminal fault raises :class:`EndpointError` up to the fail-open
        guard. NEVER ``eval`` — only ``json.loads`` in a guard.
        """
        # --- attempt 1 (with a one-shot doubled-budget retry on truncation) ---
        try:
            content = self._chat(client, text, max_tokens=self._max_tokens, repair=False)
        except _Truncated:
            try:
                content = self._chat(
                    client, text, max_tokens=self._max_tokens * 2, repair=False
                )
            except _Truncated:
                _LOG.warning(
                    "schema_induction reply truncated twice (max_tokens=%d then %d)",
                    self._max_tokens,
                    self._max_tokens * 2,
                )
                return None
        parsed = _try_parse(content)
        if parsed is not None:
            return parsed
        # --- one corrective repair retry (safety net #3) ---
        try:
            repaired = self._chat(client, text, max_tokens=self._max_tokens * 2, repair=True)
        except _Truncated:
            return None
        return _try_parse(repaired)

    def _chat(self, client: Any, text: str, *, max_tokens: int, repair: bool) -> str:
        """One structured chat-completions call; returns the reply string or raises on truncation.

        Requests ``response_format={"type":"json_object"}`` (safety net #1, portable) at
        temperature 0 (determinism). Raises :class:`_Truncated` when ``finish_reason == "length"``
        (safety net #2) so the caller retries at a bigger budget. The network call rides
        :func:`call_with_retry`; any client exception is normalized to a classed
        :class:`EndpointError` so the retry decides retry-vs-fail and a terminal failure re-raises
        up to the fail-open guard. When ``repair`` is True the corrective instruction is prepended
        to        the user message.
        """
        user = self._user_prompt_template.format(text=text, max_labels=self._max_labels)
        if repair:
            user = f"{REPAIR_INSTRUCTION}\n\n{user}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

        def do_call() -> tuple[str, str | None]:
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - normalized into a classed EndpointError
                raise _classify_exception(exc) from exc
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            return str(content), (str(finish_reason) if finish_reason is not None else None)

        result = call_with_retry(
            do_call,
            self._policy,
            rng=self._rng.random,
            sleep=lambda _d: None,
            on_retry=self._count_retry,
        )
        content, finish_reason = cast("tuple[str, str | None]", result)
        if finish_reason == "length":
            raise _Truncated
        return content

    def _count_retry(self) -> None:
        self.endpoint_retries += 1

    # -- schema validation (#4) + sanitization (#5) -------------------------

    def _sanitize_to_induced(self, validated: dict[str, list[str]]) -> InducedLabels | None:
        """Sanitize each slot's labels + build an :class:`InducedLabels`; ``None`` if wholly empty.

        Every slot is bounded by :mod:`~latence_schema_inducer.sanitize` (control-char strip, length
        ≤ 64, dedupe, count ≤ ``max_labels``). A wholly-empty result (the model returned only junk /
        nothing usable) returns ``None`` so the Provider falls back to config labels rather than
        annotating with an empty schema. Dropped-for-count overflow is logged (no silent drop).
        """
        results = {slot: sanitize_labels(validated.get(slot, [])) for slot in _SLOTS}
        total_dropped = sum(r.dropped_for_count for r in results.values())
        if total_dropped:
            _LOG.info(
                "schema_induction capped %d label(s) at the per-type count limit (%d)",
                total_dropped,
                self._max_labels,
            )
        entity = results["entity_types"].labels[: self._max_labels]
        relation = results["relation_types"].labels[: self._max_labels]
        pii = results["pii_types"].labels[: self._max_labels]
        if not entity and not relation and not pii:
            return None
        return InducedLabels(
            entity_types=entity,
            relation_types=relation,
            pii_types=pii,
            model_id=self._model,
        )


def _try_parse(content: str) -> dict[str, Any] | None:
    """``json.loads`` in a guard (safety net #3): a dict is returned, anything else / bad JSON None.

    NEVER ``eval`` and NEVER execute model text — only ``json.loads``. A non-JSON body, a JSON
    array/scalar at the top level, or a decode error all yield ``None`` (the caller repairs / falls
    back).
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


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

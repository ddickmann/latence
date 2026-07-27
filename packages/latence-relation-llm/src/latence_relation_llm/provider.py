"""LLM-based RelationExtractor Provider over any OpenAI-compatible endpoint.

The S5 Relation Extraction *LLM* reference Provider (ADR-0013: the choice of RE
Provider is empirical; an LLM extractor over any OpenAI-compatible endpoint —
including a **local Ollama** — is one of the two shipped candidates). It runs
behind the OpenAI-compatible chat-completions API, so the same Provider drives
OpenAI, vLLM, Together, or a laptop-local Ollama with only a ``base_url`` change
(ADR-0007: the endpoint path is the performance path; nothing GPU is forced into
first-run).

Given a document's :class:`~latence_core.contracts.EntityMention`s plus the chunk
text for context, it prompts the model to emit typed directed relations over the
mention set, constrained to a per-Pipeline relation label set (zero-shot — the
Capability names no relation types; the label set is Stage ``config``). The model
returns JSON referencing mentions by an index into the numbered mention list; the
Provider resolves those indices back to real mention ``record_id``s and stamps a
:class:`~latence_core.contracts.RelationMention` whose Provenance span covers both
endpoints (resolving to the source page via the parent's page map — the S5
round-trip). A relation naming an out-of-range index, a self-relation, or an
unknown label is dropped, so a hallucinated reference never yields an
un-resolvable record.

Fault tolerance is the ported endpoint machinery (:mod:`latence_core.endpoint`),
the same as the other endpoint Providers (Parse/NER/Embedder): the chat call goes
through :func:`~latence_core.endpoint.call_with_retry_async` with
exponential-backoff-with-jitter, so a 429/5xx/transport blip is retried and a
terminal 4xx fails fast. The count of transient failures a run absorbed is exposed
on ``endpoint_retries`` and folded into the Quality Report by the Runner (issue
#44.3: RE retries surface for observability consistency with the other endpoint
Stages, not silently).

Concurrency (W6, ADR-0036): a document's relations are one LLM call, and the calls
are pure per-request latency, so ``relate`` fans the per-document calls out
concurrently — the ``openai`` async client (:class:`~openai.AsyncOpenAI`) + an
:class:`asyncio.Semaphore` (``max_concurrency``, default 8) + a single
:func:`asyncio.run` over :func:`asyncio.gather` (``return_exceptions=True``) inside
the SYNC ``relate`` (the Runner is sync). The public method stays a plain
``Iterator[RelationMention]``. The wall-clock collapses from ``N × latency`` to
``ceil(N / max_concurrency) × latency``. Only the network call is concurrent; the
deterministic index-resolution / provenance stamping runs afterward in **input
document order**, so a shuffled completion order never perturbs the byte-identical
output (temperature 0). A single document's endpoint failure surfaces as an
:class:`EndpointError` for that document (the gather is ``return_exceptions=True``,
so it never aborts the other documents' work); a well-formed non-JSON body still
yields no relations for that document, exactly as before. ``max_concurrency=1`` is
byte-identical to the pre-W6 sequential behaviour.

Licensing: this Provider is framework code (Apache-2.0) and calls a user-supplied
endpoint; it bundles no model weights. The ``openai`` client is Apache-2.0 (code).
No key or endpoint is embedded — both come from config/environment.

The ``openai`` import is deferred to first use so importing this module (for
Provider discovery) never requires the client to be installed, and a missing
optional dep raises a clear, actionable error only when the Provider is run.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.contracts import RelationMention
from latence_core.endpoint import EndpointError, RetryPolicy, call_with_retry_async
from latence_core.induced import LabelSlot, effective_labels
from latence_core.page_index import pages_for_covering_span
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import (
        ChunkRecord,
        EntityMention,
        InducedLabels,
        Provenance,
    )

_DEFAULT_MODEL = "gpt-4o-mini"
# The default bound on in-flight concurrent per-document LLM calls (W6). 8 collapses a
# multi-document relate stage from N×latency to ceil(N/8)×latency; 1 = sequential. Clamped >= 1.
_DEFAULT_MAX_CONCURRENCY = 8
_SYSTEM_PROMPT = (
    "You are a precise relation-extraction engine. Given a document's text and a "
    "numbered list of entity mentions found in it, identify the typed, directed "
    "relations between those mentions. Only use the provided relation labels. "
    "Refer to mentions by their number. Respond ONLY with a JSON object of the form "
    '{"relations": [{"head": <int>, "tail": <int>, "label": "<relation>", '
    '"confidence": <0..1>}]}. Emit no relation you cannot support from the text.'
)


async def _async_noop_sleep(_delay: float) -> None:
    """The injected async backoff sleep — a no-op, mirroring the sync path's ``lambda _d: None``.

    The endpoint Providers pass an instant sleep to the ported retry (the retry's value is the
    classing + the ``endpoint_retries`` tally, not wall-time backoff in this batch context); keeping
    it a no-op makes a seeded test instant and byte-stable and preserves the pre-W6 behaviour.
    """
    return None


@dataclass(frozen=True)
class _DocJob:
    """One document's fully-resolved relate job: everything the network + resolution steps need.

    Built ONCE per document (in first-seen order) before the concurrent fetch, so the effective
    label resolution, reading-ordered + capped mention list and text are all fixed and the
    fan-out is pure network — the resolution afterward is deterministic and network-free.
    """

    doc_id: str
    mentions: list[EntityMention]
    text: str
    labels: list[str]


class LLMRelationExtractor:
    """RelationExtractor Provider backed by an OpenAI-compatible chat endpoint.

    Config:
        labels: list[str]  — REQUIRED. The zero-shot relation types to extract (e.g.
            ``["works_for", "located_in"]``). Nothing is hardcoded in core (S5 seam).
        model: str  — the chat model id (default ``gpt-4o-mini``). For Ollama use e.g.
            ``"llama3.1"`` with ``base_url="http://localhost:11434/v1"``.
        base_url: str | None  — the OpenAI-compatible endpoint. ``None`` uses the
            client default (OpenAI). Set it to a vLLM/Ollama URL to swap substrate
            with no code change.
        api_key: str | None  — the key; falls back to ``$OPENAI_API_KEY``. A local
            Ollama accepts any placeholder (e.g. ``"ollama"``).
        temperature: float  — sampling temperature (default 0.0 for determinism).
        max_mentions: int  — cap on mentions sent per document to bound the prompt
            (default 100).
        default_confidence: float  — used when the model omits a confidence (default 0.8).
        retry_attempts / retry_base_delay / retry_max_delay  — the ported retry policy
            knobs (issue #44.3), identical to the other endpoint Providers; retries
            absorbed surface on ``endpoint_retries`` for the Quality Report.
        max_concurrency: int  — the bound on in-flight concurrent per-document LLM calls
            (W6, default 8; 1 = sequential). The calls fan out under an
            ``asyncio.Semaphore(max_concurrency)``.
        request_timeout: float | None  — a per-call timeout (seconds) passed to the async
            client (``None`` ⇒ the client default). A timeout is a transport fault → retried
            like a 5xx.

    The client is constructed lazily on first ``relate`` so building the Provider
    (which the Runner does for every Stage) never needs the endpoint reachable.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): an OpenAI-compatible LLM endpoint adapter.

        ``compute="either"``: a thin HTTP client that runs on the local CPU (no local CUDA
        required); the served LLM lives behind the endpoint (a hosted API or a self-hosted
        vLLM), so the device seam does not skip it on a CPU-only host. Real quality/throughput
        needs a live rig (#64); conformance verifies it against a stub. ``model_id=None``:
        ships no weights — the served model id is config-supplied (default ``gpt-4o-mini`` is a
        served-model id, not a bundled checkpoint). License is framework code (Apache-2.0,
        verified; the ``openai`` client is Apache-2.0). ``deterministic=False`` (a served LLM is
        not byte-identical). ``cost_per_1k=None``: an LLM's per-call cost is vendor- and
        model-specific and changes over time, so a number is left undeclared rather than
        fabricated — the bake-off reads the honest gap, not a stale price.
        """
        return ProviderProfile(
            compute="either",
            memory_mb=64,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=False,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "LLMRelationExtractor requires a non-empty 'labels' list in its config "
                "(zero-shot RE: the relation label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        # NOTE: the label SET is now computed per-document from the effective labels (config ∪
        # induced, W4/ADR-0038) in ``_valid_relations``, not a fixed instance set.
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._temperature = float(config.get("temperature", 0.0))
        self._max_mentions = int(config.get("max_mentions", 100))
        self._default_confidence = float(config.get("default_confidence", 0.8))
        # Ported retry (issue #44.3): the RE endpoint call now rides the same
        # backoff-with-jitter as Parse/NER/Embedder. Seeded RNG off the endpoint identity so a
        # test run is byte-stable while a real run still de-correlates a burst of retries.
        self._policy = RetryPolicy(
            attempts=int(config.get("retry_attempts", 4)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
        )
        self._rng = random.Random(f"{self._base_url}|{self._model}")
        # W6 concurrency: bound in-flight per-document LLM calls. Clamped >= 1 so a configured
        # 0/negative means "one at a time" (sequential), never a zero-permit deadlock.
        self._max_concurrency = max(1, int(config.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY)))
        # Per-call timeout (seconds) forwarded to AsyncOpenAI; None ⇒ the client default. A timeout
        # is a transport fault → classed retryable by the ported retry (same as a 5xx).
        request_timeout = config.get("request_timeout")
        self._request_timeout = None if request_timeout is None else float(request_timeout)
        self._client: Any = None
        # Observability: transient endpoint failures this Provider retried through. The Runner
        # reads this after the Stage runs (via the generic ``endpoint_retries`` attribute seam)
        # and folds it into the Quality Report — same as the other endpoint Stages.
        self.endpoint_retries = 0

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'openai' package is required to run LLMRelationExtractor. "
                "Install it with: pip install latence-relation-llm"
            )
            raise ImportError(msg) from exc
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        # The ASYNC client (W6): per-document calls fan out concurrently under one event loop per
        # relate() run. A per-call ``timeout`` is passed only when configured (None ⇒ client
        # default); base_url/key sourcing is unchanged, so the OpenAI/vLLM/Ollama swap is identical.
        kwargs: dict[str, Any] = {"base_url": self._base_url, "api_key": api_key}
        if self._request_timeout is not None:
            kwargs["timeout"] = self._request_timeout
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    # -- extraction ---------------------------------------------------------

    def relate(
        self, chunks: Iterable[ChunkRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[RelationMention]:
        chunk_list = list(chunks)
        doc_text = _document_texts(chunk_list)
        # W4 label-source seam (ADR-0038): the relation types the OPTIONAL schema_induction Stage
        # induced per document, read off the document's chunks. UNIONed onto the config labels
        # per document below; absent (the default) ⇒ the effective labels are exactly the config
        # labels (byte-identical to every existing stack).
        doc_induced = _document_induced_labels(chunk_list)

        by_doc: dict[str, list[EntityMention]] = {}
        order: list[str] = []
        for m in mentions:
            if m.document_record_id not in by_doc:
                by_doc[m.document_record_id] = []
                order.append(m.document_record_id)
            by_doc[m.document_record_id].append(m)

        # Build one job per document with >= 2 mentions, in first-seen order, resolving its
        # effective labels + reading-ordered mentions ONCE. A single-mention document is skipped.
        jobs: list[_DocJob] = []
        for doc_id in order:
            doc_mentions = by_doc[doc_id]
            if len(doc_mentions) < 2:
                continue
            labels = effective_labels(self._labels, doc_induced.get(doc_id), LabelSlot.RELATION)
            ordered = sorted(
                doc_mentions, key=lambda m: (m.provenance.char_start or 0, m.record_id)
            )[: self._max_mentions]
            jobs.append(
                _DocJob(
                    doc_id=doc_id,
                    mentions=ordered,
                    text=doc_text.get(doc_id, ""),
                    labels=labels,
                )
            )
        if not jobs:
            return

        # W6 fan-out: fetch every document's raw model output concurrently (bounded), index-keyed by
        # job position; then run the deterministic sync resolution below in input document order.
        raw_by_index = self._call_models_concurrent(jobs)
        for i, job in enumerate(jobs):
            yield from self._resolve_document(job, raw_by_index[i])

    def _call_models_concurrent(self, jobs: list[_DocJob]) -> list[list[dict[str, Any]]]:
        """Fetch each document's raw relation list CONCURRENTLY, keyed by input-order index (W6).

        The latency-bound network call per document runs under an
        ``asyncio.Semaphore(max_concurrency)`` and the jobs are gathered with
        ``return_exceptions=True`` so one document's endpoint fault never aborts the others' work.
        The wall-clock is ``ceil(len(jobs) / max_concurrency) x latency`` (the timing test proves).
        The results are returned in input-order index, so the sync resolution stays byte-identical
        whatever the completion order. If any document's call raised an :class:`EndpointError`, the
        FIRST one in document order is re-raised (preserving the pre-W6 "endpoint failure surfaces"
        contract — the RE Stage fails loud, deterministically, not on whichever finished first).
        """
        client = self._get_client()

        async def _run() -> list[list[dict[str, Any]] | BaseException]:
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def _one(job: _DocJob) -> list[dict[str, Any]]:
                async with semaphore:
                    return await self._call_model(client, job.text, job.mentions, job.labels)

            return await asyncio.gather(
                *(_one(job) for job in jobs), return_exceptions=True
            )

        gathered = asyncio.run(_run())
        # Deterministic error surfacing: re-raise the FIRST document-order EndpointError, not the
        # first to complete. A non-EndpointError (unexpected) is also surfaced in document order.
        for outcome in gathered:
            if isinstance(outcome, BaseException):
                raise outcome
        return [outcome for outcome in gathered if not isinstance(outcome, BaseException)]

    def _resolve_document(
        self, job: _DocJob, raw: list[dict[str, Any]]
    ) -> Iterator[RelationMention]:
        """Deterministic, network-free resolution of a document's raw relations → RelationMentions.

        Runs AFTER the concurrent fetch, in input document order, so the offset math + index
        resolution + provenance stamping are byte-identical to the pre-W6 sequential path.
        """
        for emitted, rel in enumerate(self._valid_relations(raw, job.mentions, job.labels)):
            head, tail, label, confidence = rel
            span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
            span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
            provenance = _relation_provenance(head, tail, span_start, span_end)
            yield RelationMention(
                record_id=f"{job.doc_id}#relation-{emitted}",
                document_record_id=job.doc_id,
                index=emitted,
                label=label,
                head_mention_id=head.record_id,
                tail_mention_id=tail.record_id,
                head_text=head.text,
                tail_text=tail.text,
                confidence=confidence,
                provenance=provenance,
                classification=head.classification,
            )

    async def _call_model(
        self, client: Any, text: str, mentions: list[EntityMention], labels: list[str]
    ) -> list[dict[str, Any]]:
        numbered = "\n".join(
            f"{i}: {m.text} ({m.label})" for i, m in enumerate(mentions)
        )
        user = (
            f"Relation labels: {', '.join(labels)}\n\n"
            f"Document text:\n{text}\n\n"
            f"Entity mentions:\n{numbered}\n\n"
            "Return the JSON object of relations now."
        )
        # Route the endpoint call through the ported retry (issue #44.3): a 429/5xx/transport blip
        # is retried with backoff+jitter and tallied on ``endpoint_retries``; a terminal 4xx fails
        # fast. Only the network call is wrapped — a well-formed HTTP 200 whose *body* is non-JSON
        # is a model-quality issue, not a transient fault, and still drops to no relations (below),
        # so retrying does not burn budget on a reply that will never parse.
        content = await self._call_endpoint(client, user)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # A non-JSON reply yields no relations rather than crashing the run.
            return []
        relations = parsed.get("relations", []) if isinstance(parsed, dict) else []
        return [r for r in relations if isinstance(r, dict)]

    async def _call_endpoint(self, client: Any, user: str) -> str:
        """One chat-completions call through the ported retry, returning the raw reply string.

        Classes any client exception into a :class:`EndpointError` (429/5xx retryable, other 4xx
        terminal, transport retryable) so :func:`call_with_retry_async` decides retry-vs-fail; the
        retry count is tallied on ``endpoint_retries`` via ``on_retry``. An exhausted / terminal
        failure re-raises the :class:`EndpointError` — the Runner surfaces the Stage failure,
        identical to the NER/Parse endpoint Providers.
        """

        async def do_call() -> str:
            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - normalized into a classed EndpointError
                raise _classify_exception(exc) from exc
            return response.choices[0].message.content or "{}"

        result = await call_with_retry_async(
            do_call,
            self._policy,
            rng=self._rng.random,
            sleep=_async_noop_sleep,
            on_retry=self._count_retry,
        )
        return str(result)

    def _count_retry(self) -> None:
        self.endpoint_retries += 1

    def _valid_relations(
        self, raw: list[dict[str, Any]], mentions: list[EntityMention], labels: list[str]
    ) -> Iterator[tuple[EntityMention, EntityMention, str, float]]:
        """Resolve model output to valid (head, tail, label, confidence) tuples.

        Drops any relation with an out-of-range mention index, an unknown label (a label not in
        the per-document effective set — config ∪ induced), or a head == tail self-reference — so a
        hallucinated reference never becomes an un-resolvable record. Deterministic order (as
        returned, then de-duplicated on head/tail/label).
        """
        label_set = set(labels)
        n = len(mentions)
        seen: set[tuple[str, str, str]] = set()
        for rel in raw:
            label = str(rel.get("label", ""))
            if label not in label_set:
                continue
            try:
                head_idx = int(rel["head"])
                tail_idx = int(rel["tail"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= head_idx < n) or not (0 <= tail_idx < n) or head_idx == tail_idx:
                continue
            head = mentions[head_idx]
            tail = mentions[tail_idx]
            key = (head.record_id, tail.record_id, label)
            if key in seen:
                continue
            seen.add(key)
            try:
                confidence = float(rel.get("confidence", self._default_confidence))
            except (TypeError, ValueError):
                confidence = self._default_confidence
            confidence = min(max(confidence, 0.0), 1.0)
            yield head, tail, label, confidence


def _classify_exception(exc: Exception) -> EndpointError:
    """Normalize any client exception into a classed :class:`EndpointError` for the ported retry.

    An exception exposing an HTTP ``status_code`` (the OpenAI client's ``APIStatusError`` shape) is
    classed by that status (429/5xx retryable, other 4xx terminal); anything else is a transport
    error → retryable. Mirrors the NER/Parse endpoint Providers exactly.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return EndpointError(str(exc), status=status)
    return EndpointError(str(exc))


# -- shared text/page-map/provenance helpers (mirror the core provider) ------


def _document_texts(chunks: list[ChunkRecord]) -> dict[str, str]:
    """Reconstruct each document's assembled text from its chunks' document spans."""
    by_doc: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_record_id, []).append(chunk)
    texts: dict[str, str] = {}
    for doc_id, doc_chunks in by_doc.items():
        end = 0
        for chunk in doc_chunks:
            start = chunk.provenance.char_start or 0
            end = max(end, start + len(chunk.content))
        buf = [" "] * end
        for chunk in doc_chunks:
            start = chunk.provenance.char_start or 0
            buf[start : start + len(chunk.content)] = chunk.content
        texts[doc_id] = "".join(buf)
    return texts


def _document_induced_labels(chunks: list[ChunkRecord]) -> dict[str, InducedLabels | None]:
    """The induced-label schema per document (first chunk carrying one wins; else None).

    The schema_induction Stage broadcasts one document's induced labels onto ALL its chunks, so the
    first chunk with a populated ``induced_labels`` carries the whole document's schema. A document
    whose chunks all have ``induced_labels is None`` (no schema_induction Stage, or an induction
    that    failed open) maps to ``None`` — the effective labels then equal the config labels.
    """
    labels: dict[str, InducedLabels | None] = {}
    for chunk in chunks:
        if chunk.document_record_id not in labels and chunk.induced_labels is not None:
            labels[chunk.document_record_id] = chunk.induced_labels
    return labels


def _relation_provenance(
    head: EntityMention,
    tail: EntityMention,
    span_start: int,
    span_end: int,
) -> Provenance:
    """Provenance for a relation: shared lineage + covering span AND its page(s).

    A covering span runs from one endpoint mention to the other, which may be in different
    chunks, so it is not a sub-chunk offset any one chunk can resolve. Both endpoints already
    carry their own exactly-resolved pages, so the covering span's pages are their union
    (:func:`~latence_core.page_index.pages_for_covering_span`). This Provider used to rebuild a
    whole-document ``PageOffsetIndex`` from the map it scraped off the chunk stream and look the
    span up in it — which needed the map-carrying chunk to be present, and fell back to exactly
    this union when it was not.
    """
    base = head.provenance
    update: dict[str, int] = {"char_start": span_start, "char_end": span_end}
    pages = pages_for_covering_span(head.provenance, tail.provenance)
    if pages is not None:
        update["page_start"], update["page_end"] = pages
    return base.model_copy(update=update)

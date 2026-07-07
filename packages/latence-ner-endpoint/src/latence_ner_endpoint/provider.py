"""Endpoint-backed EntityExtractor Provider — the NER performance path (S12, ADR-0007).

The heavy-model Entity Extraction Provider: it finds typed entity mentions by calling any
**OpenAI-compatible / vLLM / Triton** chat endpoint serving a large zero-shot NER model
(e.g. GLiNER-large). It is the *same Capability* as the in-core CPU references (``entity.gazetteer``
/ the ``entity.gliner`` in-process learned Provider) — a Pipeline swaps CPU↔endpoint with **one
config change** (point ``base_url`` at the endpoint), no code edit (ADR-0007).

Zero-shot: the label set is supplied per Pipeline (Stage ``config``); core hardcodes no labels — the
same endpoint tags ``PERSON``/``ORG`` for one Pipeline and ``GENE``/``DRUG`` for another. Each
emitted :class:`~latence_core.contracts.EntityMention` carries its label, the model's confidence,
and the mention's char span **in the parent document's assembled markdown** — the endpoint returns
chunk-local offsets, resolved to their TRUE original offsets through the chunk's
stripped-``content``→original ``offset_map`` (carried since v11) via
:class:`~latence_core.offsetmap.OffsetIndex` (so a mention resolves to its exact source offset even
when markdown markup was stripped before it — ADR-0022 corrected, ADR-0031; a pre-v11 chunk degrades
to the ``char_start + local`` shift), then re-resolved to the source page(s) through the parent's
page map via :class:`~latence_core.pagemap.PageOffsetIndex` (the S4 round-trip) — byte-for-byte the
same contract the CPU learned Provider produces, so the output is identical modulo the model.

Fault tolerance is the ported endpoint machinery (:mod:`latence_core.endpoint`): a shared
reference-counted :class:`~latence_core.endpoint.EndpointLease` means N Stages against one served
model pay the cold-start once, and every call goes through
:func:`~latence_core.endpoint.call_with_retry` (backoff+jitter, 429/5xx retryable, terminal 4xx
fails fast). Retries absorbed land on ``endpoint_retries`` for the Quality Report. A hallucinated
span (out of range, or text that does not match the chunk) is dropped, so a bad model reply never
yields an un-resolvable record.

Licensing: framework code (Apache-2.0); bundles no weights (calls a user-supplied endpoint). The
``openai`` client is Apache-2.0 (code). The client import is deferred to first use.
"""

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any

from latence_core.contracts import EntityMention
from latence_core.endpoint import EndpointError, EndpointPool, RetryPolicy, call_with_retry
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, Provenance

_DEFAULT_MODEL = "gliner-large"
_SYSTEM_PROMPT = (
    "You are a precise zero-shot named-entity-recognition engine. Given a text and a list of "
    "entity types, find every span of the text that is an instance of one of those types. Respond "
    'ONLY with a JSON object of the form {"entities": [{"start": <int>, "end": <int>, '
    '"label": "<type>", "confidence": <0..1>}]} where start/end are character offsets into the '
    "provided text. Emit no entity you cannot support from the text."
)

# One shared pool per process so two endpoint Stages against the SAME served model share one lease
# (co-located activation — the ported lifecycle), rather than each spinning the endpoint up alone.
_POOL = EndpointPool()


class EndpointEntityExtractor:
    """EntityExtractor Provider backed by an OpenAI-compatible endpoint (the NER performance path).

    Config:
        labels: list[str]  — REQUIRED. The zero-shot entity types to extract. Nothing is hardcoded
            in core (S4 AC).
        model: str  — the served model id (default ``gliner-large``).
        base_url: str | None  — the OpenAI-compatible endpoint (vLLM / Triton / hosted / local).
        api_key: str | None  — the key; falls back to ``$OPENAI_API_KEY``.
        temperature: float  — sampling temperature (default 0.0 for determinism).
        threshold: float  — drop mentions below this confidence (default 0.5).
        min_chars: int  — a chunk shorter than this is treated as near-empty and skipped without a
            model call (default 1), matching the CPU Provider.
        retry_attempts / retry_base_delay / retry_max_delay  — the ported retry policy knobs.

    The client is constructed lazily on first ``extract`` so building the Provider never needs the
    endpoint reachable.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): an OpenAI-compatible NER endpoint adapter.

        ``compute="either"``: a thin HTTP client that runs on the local CPU (no local CUDA
        required); the served NER model lives behind the endpoint, so the device seam does not
        skip it on a CPU-only host. Real quality/throughput needs a live rig (#64); conformance
        verifies it against a stub. ``model_id=None``: ships no weights (the served model id is
        config-supplied). License is framework code (Apache-2.0, verified; the ``openai`` client
        is Apache-2.0). ``deterministic=False`` (a served model is not byte-identical);
        ``cost_per_1k=None`` (deployment/vendor-specific, not fabricated).
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
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "EndpointEntityExtractor requires a non-empty 'labels' list in its config "
                "(zero-shot NER: the label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._label_set = set(self._labels)
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._temperature = float(config.get("temperature", 0.0))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        self._policy = RetryPolicy(
            attempts=int(config.get("retry_attempts", 4)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
        )
        self._rng = random.Random(f"{self._base_url}|{self._model}")
        self._lease = _POOL.lease(f"{self._base_url}|{self._model}")
        self._client: Any = None
        self.endpoint_retries = 0

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'openai' package is required to run EndpointEntityExtractor. "
                "Install it with: pip install latence-ner-endpoint"
            )
            raise ImportError(msg) from exc
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    # -- extraction ---------------------------------------------------------

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        chunk_list = [c for c in chunks if len(c.content.strip()) >= self._min_chars]
        if not chunk_list:
            # Every chunk empty/near-empty (or no chunks): no mentions, no endpoint call (S4 AC).
            return
        client = self._get_client()
        # Ref-counted lease: activate the shared endpoint once for this Stage's whole chunk stream,
        # tear it down once when done — the ported lifecycle, generalized off RunPod. The default
        # activate/deactivate are no-ops for an always-on endpoint, so this is free there.
        self._lease.acquire()
        try:
            for chunk in chunk_list:
                yield from self._extract_one(client, chunk)
        finally:
            self._lease.release()

    def _extract_one(self, client: Any, chunk: ChunkRecord) -> Iterator[EntityMention]:
        raw = self._call_model(client, chunk.content)
        # Deterministic ordering: by chunk-local start, then label.
        raw.sort(key=lambda e: (int(e.get("start", 0)), str(e.get("label", ""))))
        # Resolve chunk-local model offsets to TRUE original offsets through the chunk's
        # stripped→original offset map (markup stripping makes char_start+local only a
        # lower bound — ADR-0031); a pre-v11 chunk with no map degrades to that shift.
        offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        chunk_char_start = chunk.provenance.char_start or 0
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
        emitted = 0
        for ent in raw:
            resolved = self._valid_span(ent, chunk.content)
            if resolved is None:
                continue
            local_start, local_end, label, confidence = resolved
            if offset_index is not None:
                doc_char_start, doc_char_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_char_start = chunk_char_start + local_start
                doc_char_end = chunk_char_start + local_end
            provenance = self._mention_provenance(
                chunk.provenance,
                doc_char_start=doc_char_start,
                doc_char_end=doc_char_end,
                page_index=page_index,
            )
            yield EntityMention(
                record_id=f"{chunk.record_id}#entity-{emitted}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=emitted,
                label=label,
                text=chunk.content[local_start:local_end],
                confidence=confidence,
                provenance=provenance,
                classification=chunk.classification,
            )
            emitted += 1

    def _valid_span(
        self, ent: dict[str, Any], content: str
    ) -> tuple[int, int, str, float] | None:
        """Validate one model-returned entity: in-range span, known label, above threshold.

        A hallucinated span (out-of-range offsets, inverted span, unknown label, or below the
        confidence threshold) is dropped — a bad reply never becomes an un-resolvable record.
        """
        label = str(ent.get("label", ""))
        if label not in self._label_set:
            return None
        try:
            start = int(ent["start"])
            end = int(ent["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= start < end <= len(content)):
            return None
        try:
            confidence = float(ent.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = min(max(confidence, 0.0), 1.0)
        if confidence < self._threshold:
            return None
        return start, end, label, confidence

    def _call_model(self, client: Any, text: str) -> list[dict[str, Any]]:
        def do_call() -> list[dict[str, Any]]:
            return self._one_call(client, text)

        result = call_with_retry(
            do_call,
            self._policy,
            rng=self._rng.random,
            sleep=lambda _d: None,
            on_retry=self._count_retry,
        )
        return list(result) if isinstance(result, list) else []

    def _count_retry(self) -> None:
        self.endpoint_retries += 1

    def _one_call(self, client: Any, text: str) -> list[dict[str, Any]]:
        user = (
            f"Entity types: {', '.join(self._labels)}\n\n"
            f"Text:\n{text}\n\n"
            "Return the JSON object of entities now."
        )
        try:
            response = client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalized into a classed EndpointError below
            raise _classify_exception(exc) from exc
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # A non-JSON reply yields no entities rather than crashing the run.
            return []
        entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
        return [e for e in entities if isinstance(e, dict)]

    @staticmethod
    def _mention_provenance(
        chunk_prov: Provenance,
        doc_char_start: int,
        doc_char_end: int,
        page_index: PageOffsetIndex | None,
    ) -> Provenance:
        """Provenance for a mention: chunk lineage + the mention's own span AND page.

        Identical to the CPU learned Provider: the char span is narrowed to the mention's own
        ``[start, end)`` in the parent document's assembled markdown; the page span is re-resolved
        from that offset through the parent's page map (so a mention on a later page cites its own
        page), degrading to the chunk's inherited page range when no page map is present.
        """
        update: dict[str, int] = {"char_start": doc_char_start, "char_end": doc_char_end}
        if page_index is not None:
            update["page_start"] = page_index.page_for_offset(doc_char_start)
            update["page_end"] = page_index.page_for_offset(max(doc_char_end - 1, doc_char_start))
        return chunk_prov.model_copy(update=update)


def _classify_exception(exc: Exception) -> EndpointError:
    """Normalize any client exception into a classed :class:`EndpointError` for the ported retry."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return EndpointError(str(exc), status=status)
    return EndpointError(str(exc))

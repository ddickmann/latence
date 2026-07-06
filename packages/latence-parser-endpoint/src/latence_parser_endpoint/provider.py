"""Endpoint-backed Parser Provider — the Parse performance path (S12, ADR-0007).

The heavy-model Parse Provider: it turns a raw :class:`~latence_core.contracts.ParserInput` into a
markdown :class:`~latence_core.contracts.DocumentRecord` by calling any **OpenAI-compatible /
vLLM / Triton** chat endpoint serving a document-understanding model (e.g. PaddleOCR-VL, a
vision-LLM). It is the *same Capability* as the in-core ``parser.document`` reference — a Pipeline
swaps CPU↔endpoint with **one config change** (point ``base_url`` at the endpoint), no code edit
and no pipeline change (ADR-0007: "switching is a config/URL swap, never a pipeline change").

Fault tolerance is the ported endpoint machinery (:mod:`latence_core.endpoint`): every call goes
through :func:`~latence_core.endpoint.call_with_retry` with exponential-backoff-with-jitter, so a
429/5xx/transport blip is retried and a terminal 4xx fails fast; the count of retries a run
absorbed is exposed on ``endpoint_retries`` and folded into the Quality Report by the Runner (S12
AC: endpoint failures retried/surfaced). An unrecoverable failure yields a ``PARSE_ERROR`` record
(carrying Provenance + the reason) rather than aborting the run — identical graceful-degradation
contract to the CPU reference, so the output is identical modulo the model.

Licensing: this Provider is framework code (Apache-2.0) and calls a user-supplied endpoint; it
bundles no model weights. The ``openai`` client is Apache-2.0 (code). No key or endpoint is
embedded — both come from config/environment. The client import is deferred to first use so
importing this module (for Provider discovery) never needs the client installed.
"""

from __future__ import annotations

import base64
import os
import random
from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, DocumentRecord, PageMap
from latence_core.endpoint import EndpointError, RetryPolicy, call_with_retry
from latence_core.language import detect_language

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ParserInput

_DEFAULT_MODEL = "paddleocr-vl"
_SYSTEM_PROMPT = (
    "You are a document-understanding engine. Transcribe the given document faithfully into "
    "clean GitHub-flavored Markdown, preserving headings, lists, and tables. Return ONLY the "
    "Markdown transcription, no commentary."
)
# The page separator the transcription uses so a multi-page document maps back to per-page
# boundaries — the model is asked to delimit pages with this token; a single-page document (plain
# text) simply has none.
_PAGE_SEPARATOR = "\n\n---PAGE---\n\n"


class EndpointDocumentParser:
    """Parser Provider that transcribes documents via an OpenAI-compatible endpoint.

    Config:
        model: str  — the served model id (default ``paddleocr-vl``). For vLLM/Triton use the id
            the endpoint registers; nothing is hardcoded to a vendor.
        base_url: str | None  — the OpenAI-compatible endpoint. ``None`` uses the client default.
            Set it to a vLLM / Triton / hosted URL to move the Parse Stage to the endpoint with no
            pipeline change (ADR-0007 performance path).
        api_key: str | None  — the key; falls back to ``$OPENAI_API_KEY``. A local server accepts
            any placeholder.
        temperature: float  — sampling temperature (default 0.0 for determinism).
        retry_attempts / retry_base_delay / retry_max_delay  — the ported retry policy knobs.
        category / sensitivity  — Classification attached at Parse (category from config else the
            input's; sensitivity inherited else config else ``"unknown"``), matching the CPU
            reference so the swap is output-identical modulo the model.

    The client is constructed lazily on first ``parse`` so building the Provider (which the Runner
    does for every Stage) never needs the endpoint reachable.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._temperature = float(config.get("temperature", 0.0))
        self._category = config.get("category")
        self._sensitivity = config.get("sensitivity")
        self._policy = RetryPolicy(
            attempts=int(config.get("retry_attempts", 4)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
        )
        # Deterministic-but-jittered retry: a seeded RNG so a test run is byte-stable while a real
        # run still de-correlates a burst of retries. Seed derived from the endpoint identity.
        self._rng = random.Random(f"{self._base_url}|{self._model}")
        self._client: Any = None
        # S12 observability: how many transient endpoint failures this Provider retried through.
        # The Runner reads this after the Stage runs and folds it into the Quality Report.
        self.endpoint_retries = 0

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'openai' package is required to run EndpointDocumentParser. "
                "Install it with: pip install latence-parser-endpoint"
            )
            raise ImportError(msg) from exc
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    # -- parse --------------------------------------------------------------

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        client = None
        for raw in inputs:
            if client is None:
                client = self._get_client()
            yield self._parse_one(client, raw)

    def _parse_one(self, client: Any, raw: ParserInput) -> DocumentRecord:
        try:
            markdown = self._transcribe(client, raw)
        except EndpointError as exc:
            # Unrecoverable endpoint failure (terminal 4xx, or retries exhausted): emit a
            # PARSE_ERROR record carrying Provenance + reason, exactly like the CPU reference's
            # corrupt-file path — the run continues and the failure is auditable, never a crash.
            return DocumentRecord(
                record_id=raw.provenance.document_id,
                provenance=raw.provenance,
                classification=raw.classification,
                content="",
                disposition=Disposition.PARSE_ERROR,
                error=f"endpoint parse failed: {exc}",
            )
        # Split the transcription back into per-page spans on the page separator, so any downstream
        # offset resolves to its original source page (Provenance integrity, S2) — the same page-map
        # contract as the CPU reference.
        page_texts = markdown.split(_PAGE_SEPARATOR)
        assembly = PageMap.from_page_texts(page_texts)
        classification = self._classification(raw, assembly.text)
        provenance = raw.provenance.model_copy(
            update={
                "char_start": 0,
                "char_end": assembly.page_map.total_chars,
                "page_start": 1,
                "page_end": max(len(page_texts), 1),
            }
        )
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=provenance,
            classification=classification,
            content=assembly.text,
            media_type="text/markdown",
            page_map=assembly.page_map,
        )

    def _classification(self, raw: ParserInput, text: str) -> Any:
        """Classification attached at Parse: language from the decoded text, category/sensitivity.

        Mirrors the CPU reference so an endpoint swap is output-identical modulo the model: language
        auto-detected from the transcription via the shared core heuristic (not forked), category
        from config else the input's, sensitivity inherited else config else ``"unknown"``.
        """
        base = raw.classification
        category = self._category if self._category is not None else base.category
        sensitivity = (
            base.sensitivity
            if base.sensitivity not in (None, "", "unknown")
            else (self._sensitivity if self._sensitivity is not None else "unknown")
        )
        return base.model_copy(
            update={
                "language": detect_language(text),
                "category": category,
                "sensitivity": str(sensitivity),
            }
        )

    def _transcribe(self, client: Any, raw: ParserInput) -> str:
        """Call the endpoint (through the ported retry) and return the markdown transcription."""

        def do_call() -> str:
            return self._call_model(client, raw)

        result = call_with_retry(
            do_call,
            self._policy,
            rng=self._rng.random,
            sleep=_no_sleep,
            on_retry=self._count_retry,
        )
        return str(result)

    def _count_retry(self) -> None:
        self.endpoint_retries += 1

    def _call_model(self, client: Any, raw: ParserInput) -> str:
        """One chat-completions call transcribing the document; classes HTTP errors for retry.

        A text input is sent as text; a binary input (PDF/image bytes) is sent as a base64 data
        URL so a vision endpoint can read it. Any client exception carrying an HTTP ``status_code``
        is re-raised as a classed :class:`EndpointError` so the ported retry decides retry-vs-fail;
        a bare transport error is classed retryable.
        """
        content = raw.content
        if isinstance(content, str):
            user_content: Any = f"Transcribe this document to Markdown:\n\n{content}"
        else:
            b64 = base64.b64encode(content).decode("ascii")
            mime = _mime_for(raw.provenance.file_type)
            user_content = [
                {"type": "text", "text": "Transcribe this document to Markdown."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        try:
            response = client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalized into a classed EndpointError below
            raise _classify_exception(exc) from exc
        return response.choices[0].message.content or ""


def _mime_for(file_type: str) -> str:
    return {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(file_type.lower(), "application/octet-stream")


def _no_sleep(_delay: float) -> None:
    """Retries do not block the Runner thread on wall-clock sleep in this reference path.

    The ported backoff still de-correlates by *count* (each retry is a fresh endpoint attempt); a
    deployment that wants real backoff pacing injects a sleeping policy. Kept non-sleeping here so a
    seeded test run is instant and deterministic.
    """
    return None


def _classify_exception(exc: Exception) -> EndpointError:
    """Normalize any client exception into a classed :class:`EndpointError` for the ported retry.

    An exception exposing an HTTP ``status_code`` (the OpenAI client's ``APIStatusError`` shape) is
    classed by that status (429/5xx retryable, other 4xx terminal); anything else is a transport
    error → retryable.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return EndpointError(str(exc), status=status)
    return EndpointError(str(exc))

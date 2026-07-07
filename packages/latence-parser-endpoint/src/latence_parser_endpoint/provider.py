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
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

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

# Defensive input size cap (issue #44.1). Intake Screening's ``SignatureSizeIntakeScreener`` caps
# raw input at 25 MiB before Parse, but this Provider must not TRUST that Screening ran ahead of it
# (it can be invoked directly, or a zip that expanded downstream of Screening's central-directory
# check could still reach it): base64-encoding an unbounded input into a data URL inflates it ~1.33x
# in memory and again on the wire, a DoS/OOM vector on an oversized or zip-bomb-expanded input. So
# the Provider guards its OWN input, defense-in-depth, degrading an over-cap input to a PARSE_ERROR
# record (the same graceful-degradation contract as a corrupt file / a terminal endpoint failure)
# rather than encoding it. Default matches Screening's ``max_bytes`` so a Screening-passed input is
# never rejected here; override with ``max_input_bytes``.
_DEFAULT_MAX_INPUT_BYTES = 25 * 1024 * 1024


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
        max_input_bytes: int  — defensive cap on the raw input this Provider will encode/send
            (default 25 MiB, matching Intake Screening's ``max_bytes``); an over-cap input degrades
            to a PARSE_ERROR without an endpoint call. ``<= 0`` disables the guard.
        category / sensitivity  — Classification attached at Parse (category from config else the
            input's; sensitivity inherited else config else ``"unknown"``), matching the CPU
            reference so the swap is output-identical modulo the model.

    The client is constructed lazily on first ``parse`` so building the Provider (which the Runner
    does for every Stage) never needs the endpoint reachable.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): an OpenAI-compatible endpoint adapter.

        ``compute="either"``: the adapter itself is a thin HTTP client that runs on the local
        CPU (no local CUDA required) — the served model lives behind the endpoint (a vLLM /
        Triton / hosted URL), so the device seam does not skip it on a CPU-only host. Its real
        quality/throughput needs a live rig (#64); conformance verifies it against a stub.

        ``model_id=None``: this package ships **no** model weights — the served model id is
        config-supplied (``config['model']``, default ``paddleocr-vl`` is a served-model id, not
        a bundled checkpoint). License is framework code (Apache-2.0, verified per ADR-0012;
        the ``openai`` client is Apache-2.0). ``deterministic=False``: a served LLM/VLM is not
        guaranteed byte-identical (temperature/model/version), so it does not overclaim.
        ``cost_per_1k=None``: the per-call cost is deployment/vendor-specific (a self-hosted
        vLLM has no vendor cost), so a number is left undeclared rather than fabricated.
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
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._temperature = float(config.get("temperature", 0.0))
        self._category = config.get("category")
        self._sensitivity = config.get("sensitivity")
        # Defensive cap on the raw input this Provider will encode/send (issue #44.1). <= 0 disables
        # the guard (an adopter who has proven Screening runs ahead and wants no second cap).
        self._max_input_bytes = int(config.get("max_input_bytes", _DEFAULT_MAX_INPUT_BYTES))
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
        # Defensive size guard (issue #44.1): reject an over-cap input BEFORE base64-encoding it
        # into a data URL, so an oversized / zip-bomb-expanded input can't inflate memory/request.
        # Degrades to PARSE_ERROR (same graceful path as a corrupt file / terminal endpoint failure)
        # — never encoded, never sent. No endpoint call is made for an over-cap input.
        oversize = self._oversize_reason(raw)
        if oversize is not None:
            return self._parse_error(raw, oversize)
        try:
            markdown = self._transcribe(client, raw)
        except EndpointError as exc:
            # Unrecoverable endpoint failure (terminal 4xx, or retries exhausted): emit a
            # PARSE_ERROR record carrying Provenance + reason, exactly like the CPU reference's
            # corrupt-file path — the run continues and the failure is auditable, never a crash.
            return self._parse_error(raw, f"endpoint parse failed: {exc}")
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

    def _oversize_reason(self, raw: ParserInput) -> str | None:
        """Reason string if the raw input is over the size cap, else ``None`` (issue #44.1).

        Measures the true byte length of the undecoded input — the number of bytes that would be
        base64-encoded into the data URL (a ``str`` input is measured by its UTF-8 encoding, since
        that is what inflates the prompt). ``max_input_bytes <= 0`` disables the guard.
        """
        if self._max_input_bytes <= 0:
            return None
        content = raw.content
        n_bytes = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
        if n_bytes > self._max_input_bytes:
            return (
                f"input is {n_bytes} bytes, over the {self._max_input_bytes}-byte parser cap "
                "(rejected before base64 encoding to avoid request/memory inflation)"
            )
        return None

    def _parse_error(self, raw: ParserInput, reason: str) -> DocumentRecord:
        """A PARSE_ERROR record carrying Provenance + reason (the graceful-degradation contract).

        Identical shape whether the cause is an oversized input, a terminal endpoint error, or
        exhausted retries — the run continues and the failure is auditable, never a crash.
        """
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=raw.provenance,
            classification=raw.classification,
            content="",
            disposition=Disposition.PARSE_ERROR,
            error=reason,
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

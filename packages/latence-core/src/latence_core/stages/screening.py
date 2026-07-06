"""Built-in reference Screening Providers (intake + content).

The S3 Screening reference Providers (ADR-0004; ship in core so the demo runs end
to end; permissively licensed — pure-Python signature/size heuristics and a
lightweight keyword/regex injection classifier, no model weights, ADR-0012). Two
checkpoints (CONTEXT ``Screening``):

* **Intake Screening** — before Parse, over the raw :class:`ParserInput` bytes:
  oversized files, file-type spoofing (declared extension vs magic bytes), and zip
  bombs (a small archive whose declared/again-inspected uncompressed size explodes
  the compression ratio). A dangerous document is **Quarantined** — turned into a
  :class:`QuarantineRecord`, removed from the pipeline, never reaching Parse — with
  the reason recorded.

* **Content Screening** — after Chunk, over each :class:`ChunkRecord`'s text:
  prompt-injection patterns (e.g. "ignore previous instructions", "system prompt")
  and a small harmful-content keyword set. A hit **flags** the chunk with a
  :class:`RiskMarker` that survives into the corpus (so a RAG consumer can exclude
  it) rather than quarantining it — a legitimate document may quote an injection
  string, and dropping it silently would be worse than marking it. A configurable
  ``quarantine_threshold`` escalates a sufficiently strong hit to Quarantine.

Neither Screener uses a network, a model download, or unsafe deserialization — a
zip bomb is inspected via the stdlib ``zipfile`` central directory WITHOUT
extracting any member (no decompression, no path traversal). Both are deterministic.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import TYPE_CHECKING, Any

from latence_core.capability import ScreeningOutcome
from latence_core.contracts import (
    ParserInput,
    QuarantineRecord,
    RiskMarker,
    ScreeningCheckpoint,
    ScreeningFinding,
    ScreeningVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord

# Magic-byte signatures used for the type-spoof check: a file declaring one of
# these extensions whose leading bytes do not match is a spoof. Only a small,
# high-value set (the formats a Parser will try to decode as structured docs).
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF-",),
    "zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    # Office OpenXML (docx/xlsx/pptx) are ZIP containers.
    "docx": (b"PK\x03\x04",),
    "xlsx": (b"PK\x03\x04",),
    "pptx": (b"PK\x03\x04",),
    "gz": (b"\x1f\x8b",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
}
# The window scanned for a magic signature (a valid PDF tolerates a small leading
# BOM/junk window, per the spec — mirrors the document Parser's guard).
_MAGIC_WINDOW = 1024
# ZIP-container extensions whose uncompressed size we probe for a zip bomb.
_ZIP_EXTENSIONS = frozenset({"zip", "docx", "xlsx", "pptx"})


class SignatureSizeIntakeScreener:
    """Intake Screener: size cap, file-type spoof, and zip-bomb detection.

    Config:
        max_bytes: int  — oversized cap; a raw document larger than this is
            Quarantined (default 25 MiB).
        max_zip_ratio: float  — zip-bomb cap; a ZIP-container whose total
            uncompressed size divided by its on-disk size exceeds this ratio is
            Quarantined (default 100.0).
        max_zip_uncompressed: int  — absolute cap on a ZIP's total uncompressed
            size (default 512 MiB) — catches a bomb whose ratio alone is modest.
        check_spoof: bool  — enable the magic-byte spoof check (default True).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._max_bytes = int(config.get("max_bytes", 25 * 1024 * 1024))
        self._max_zip_ratio = float(config.get("max_zip_ratio", 100.0))
        self._max_zip_uncompressed = int(config.get("max_zip_uncompressed", 512 * 1024 * 1024))
        self._check_spoof = bool(config.get("check_spoof", True))

    def screen_intake(self, inputs: Iterable[ParserInput]) -> ScreeningOutcome[ParserInput]:
        passed: list[ParserInput] = []
        quarantined: list[QuarantineRecord] = []
        findings: list[ScreeningFinding] = []
        for raw in inputs:
            reason = self._inspect(raw)
            if reason is None:
                passed.append(raw)
                findings.append(self._finding(raw, ScreeningVerdict.PASS, "clean", "ok"))
                continue
            category, detail = reason
            quarantined.append(self._quarantine(raw, category, detail))
            findings.append(self._finding(raw, ScreeningVerdict.QUARANTINE, category, detail))
        return ScreeningOutcome(passed=passed, quarantined=quarantined, findings=findings)

    def _inspect(self, raw: ParserInput) -> tuple[str, str] | None:
        """Return ``(category, reason)`` if the raw document is dangerous, else None."""
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        if len(data) > self._max_bytes:
            return ("oversized", f"file is {len(data)} bytes, over the {self._max_bytes} cap")
        ext = raw.provenance.file_type.lower()
        if self._check_spoof and (spoof := self._spoof_reason(ext, data)) is not None:
            return ("type_spoof", spoof)
        if ext in _ZIP_EXTENSIONS and (bomb := self._zip_bomb_reason(data)) is not None:
            return ("zip_bomb", bomb)
        return None

    @staticmethod
    def _spoof_reason(ext: str, data: bytes) -> str | None:
        signatures = _MAGIC_SIGNATURES.get(ext)
        if signatures is None:
            return None  # no known signature for this type — not a spoof we can judge
        head = data[:_MAGIC_WINDOW]
        # For PDF the spec tolerates a leading window; for the binary container
        # formats the signature must be at offset 0.
        if ext == "pdf":
            if not any(sig in head for sig in signatures):
                return f"declared .{ext} but missing its {signatures[0]!r} signature"
        elif not any(head.startswith(sig) for sig in signatures):
            return f"declared .{ext} but missing its {signatures[0]!r} signature"
        return None

    def _zip_bomb_reason(self, data: bytes) -> str | None:
        """Inspect a ZIP's central directory for a decompression bomb — no extract.

        Reads only the metadata (``ZipInfo.file_size`` / ``compress_size``) from the
        central directory; it never decompresses a member, so a bomb cannot be
        triggered by the inspection itself. Flags on either an excessive total
        uncompressed size or an excessive uncompressed/on-disk ratio.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                infos = zf.infolist()
        except zipfile.BadZipFile:
            # Declared as a ZIP container but unreadable as one — a spoof/corrupt;
            # the spoof check already ran, so treat an unreadable archive as clean
            # here and let Parse record the decode failure.
            return None
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > self._max_zip_uncompressed:
            return (
                f"zip uncompresses to {total_uncompressed} bytes, "
                f"over the {self._max_zip_uncompressed} cap"
            )
        on_disk = max(len(data), 1)
        ratio = total_uncompressed / on_disk
        if ratio > self._max_zip_ratio:
            return f"zip compression ratio {ratio:.1f}x exceeds the {self._max_zip_ratio}x cap"
        return None

    @staticmethod
    def _quarantine(raw: ParserInput, category: str, reason: str) -> QuarantineRecord:
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        return QuarantineRecord(
            record_id=raw.provenance.document_id,
            provenance=raw.provenance,
            classification=raw.classification,
            checkpoint=ScreeningCheckpoint.INTAKE,
            detector="screening.intake_signature",
            category=category,
            reason=reason,
            quarantined_bytes=len(data),
        )

    @staticmethod
    def _finding(
        raw: ParserInput, verdict: ScreeningVerdict, category: str, reason: str
    ) -> ScreeningFinding:
        return ScreeningFinding(
            checkpoint=ScreeningCheckpoint.INTAKE,
            verdict=verdict,
            document_id=raw.provenance.document_id,
            detector="screening.intake_signature",
            category=category,
            reason=reason,
        )


# Prompt-injection patterns: lowercase substrings/regexes that strongly indicate an
# attempt to subvert a downstream LLM. Deliberately small and high-precision — the
# reference Provider marks (not drops), so a false positive costs a filterable flag,
# not lost data. A heavier ML classifier is a drop-in second Provider (ADR-0004).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions"),
    re.compile(r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above)"),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\b"),
    re.compile(r"system\s+prompt"),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)"),
    re.compile(r"override\s+(your|the)\s+(safety|guardrails|instructions)"),
    re.compile(r"do\s+anything\s+now"),
)
_HARMFUL_KEYWORDS: tuple[str, ...] = (
    "build a bomb",
    "make a weapon",
)


class KeywordContentScreener:
    """Content Screener: prompt-injection + harmful-content flagging on chunks.

    Config:
        quarantine_threshold: float  — a chunk whose risk score reaches this is
            Quarantined instead of flagged (default 1.01 — i.e. never, flag-only by
            default; set below 1.0 to escalate any hit to Quarantine).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        # Default > 1.0 so the reference Provider flags rather than drops (a single
        # hit scores 1.0). Lowering it escalates strong hits to Quarantine.
        self._quarantine_threshold = float(config.get("quarantine_threshold", 1.01))

    def screen_content(self, chunks: Iterable[ChunkRecord]) -> ScreeningOutcome[ChunkRecord]:
        passed: list[ChunkRecord] = []
        quarantined: list[QuarantineRecord] = []
        findings: list[ScreeningFinding] = []
        for chunk in chunks:
            category, score, reason = self._score(chunk.content)
            if category is None:
                passed.append(chunk)
                findings.append(
                    self._finding(chunk, ScreeningVerdict.PASS, "clean", "ok", None)
                )
                continue
            if score >= self._quarantine_threshold:
                quarantined.append(self._quarantine(chunk, category, reason))
                findings.append(
                    self._finding(chunk, ScreeningVerdict.QUARANTINE, category, reason, None)
                )
                continue
            marker = RiskMarker(
                category=category,
                detector="screening.content_keyword",
                score=min(score, 1.0),
                reason=reason,
            )
            # Attach the marker so it PROPAGATES into the corpus (S3 AC): the chunk
            # keeps flowing but a RAG consumer can filter on ``risk_markers``.
            flagged = chunk.model_copy(update={"risk_markers": [*chunk.risk_markers, marker]})
            passed.append(flagged)
            findings.append(self._finding(chunk, ScreeningVerdict.FLAG, category, reason, marker))
        return ScreeningOutcome(passed=passed, quarantined=quarantined, findings=findings)

    @staticmethod
    def _score(text: str) -> tuple[str | None, float, str]:
        """Return ``(category|None, score, reason)`` for a chunk's text."""
        lowered = text.lower()
        for pattern in _INJECTION_PATTERNS:
            if (match := pattern.search(lowered)) is not None:
                return ("prompt_injection", 1.0, f"matched injection pattern {match.group(0)!r}")
        for keyword in _HARMFUL_KEYWORDS:
            if keyword in lowered:
                return ("harmful", 1.0, f"matched harmful keyword {keyword!r}")
        return (None, 0.0, "no risk pattern matched")

    @staticmethod
    def _quarantine(chunk: ChunkRecord, category: str, reason: str) -> QuarantineRecord:
        return QuarantineRecord(
            record_id=chunk.record_id,
            provenance=chunk.provenance,
            classification=chunk.classification,
            checkpoint=ScreeningCheckpoint.CONTENT,
            detector="screening.content_keyword",
            category=category,
            reason=reason,
            quarantined_bytes=len(chunk.content.encode("utf-8")),
        )

    @staticmethod
    def _finding(
        chunk: ChunkRecord,
        verdict: ScreeningVerdict,
        category: str,
        reason: str,
        marker: RiskMarker | None,
    ) -> ScreeningFinding:
        return ScreeningFinding(
            checkpoint=ScreeningCheckpoint.CONTENT,
            verdict=verdict,
            document_id=chunk.provenance.document_id,
            detector="screening.content_keyword",
            category=category,
            reason=reason,
            marker=marker,
        )

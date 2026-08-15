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
  and a small harmful-content keyword set, PLUS a cross-chunk-boundary pass that
  catches an injection deliberately split across two adjacent chunks of the same
  document (which per-chunk scanning would miss). A hit **flags** the chunk(s) with a
  :class:`RiskMarker` that survives into the corpus (so a RAG consumer can exclude
  it) rather than quarantining it — a legitimate document may quote an injection
  string, and dropping it silently would be worse than marking it. A configurable
  ``quarantine_threshold`` escalates a sufficiently strong in-chunk hit to Quarantine.

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
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord

# Magic-byte signatures used for the type-spoof check: a file declaring one of
# these extensions whose leading bytes do not match is a spoof. Only a small,
# high-value set (the formats a Parser will try to decode as structured docs).
# NB: ``pdf`` is handled by :func:`_pdf_spoof_reason` (offset-0 + structural), not
# by a plain magic match — see below — so it is intentionally NOT listed here.
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
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
_PDF_MAGIC = b"%PDF-"
# A real PDF's header sits at byte 0. Some producers emit a UTF-8/UTF-16 BOM or a
# few stray whitespace bytes before it; pypdf itself reads the header at offset 0,
# so we allow only a tiny, bounded leading window of BOM/whitespace — NOT the old
# "anywhere in the first 1 KB" substring, which let an adversarial file smuggle a
# ``%PDF-`` substring past the check (#62). Kept minimal so the screener is at
# least as strict as the reader it protects.
_PDF_LEADING_SLACK = 8
# Structural markers that a genuine PDF file body contains. The header alone is
# cheap to forge; requiring at least one real structural token (a cross-reference,
# a trailer, or an object) raises the bar to "actually looks like a PDF".
_PDF_STRUCTURE_MARKERS: tuple[bytes, ...] = (b"%%EOF", b"startxref", b"xref", b"trailer", b" obj")
# The window scanned for the magic signature of the OTHER binary formats (all of
# which are offset-0 signatures; the window just bounds the slice we inspect).
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

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The intake Screener is a pure-Python part of ``latence-core`` (Apache-2.0,
        verified) — signature/size/zip-bomb heuristics over the raw bytes, no model weights
        (``model_id=None``), deterministic, CPU-only, no per-call cost. It screens a stream of
        inputs (``batch=True``).
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=True,
            cost_per_1k=None,
        )

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
        # PDF has its own offset-0 + structural check (harder to spoof, #62).
        if ext == "pdf":
            return SignatureSizeIntakeScreener._pdf_spoof_reason(data)
        signatures = _MAGIC_SIGNATURES.get(ext)
        if signatures is None:
            return None  # no known signature for this type — not a spoof we can judge
        head = data[:_MAGIC_WINDOW]
        # The binary container formats: the signature must be at offset 0.
        if not any(head.startswith(sig) for sig in signatures):
            return f"declared .{ext} but missing its {signatures[0]!r} signature"
        return None

    @staticmethod
    def _pdf_spoof_reason(data: bytes) -> str | None:
        """Reject a ``.pdf`` whose header is not at offset 0 or that lacks PDF structure.

        Hardening for #62: the old check matched ``%PDF-`` as a substring anywhere in
        the first 1 KB, so an adversarial file could smuggle the substring (e.g. inside
        a comment or a benign-looking string) to evade quarantine. This requires the
        header at byte 0 — allowing only a tiny bounded leading BOM/whitespace window,
        matching how pypdf itself locates the header — AND at least one genuine PDF
        structural marker (a cross-reference, trailer, or object) in the body, so a bare
        forged header is not enough. The oversized-file check runs before this.
        """
        # Skip a bounded leading window of BOM / ASCII whitespace only.
        offset = 0
        limit = min(len(data), _PDF_LEADING_SLACK)
        while offset < limit and data[offset] in b"\x00\r\n\t \xef\xbb\xbf\xff\xfe":
            offset += 1
        if data[offset : offset + len(_PDF_MAGIC)] != _PDF_MAGIC:
            return (
                f"declared .pdf but the {_PDF_MAGIC!r} header is not at the start of the "
                f"file (offset-0 check; a smuggled %PDF- substring does not count)"
            )
        # A genuine PDF version digit follows the header (e.g. %PDF-1.7).
        version_pos = offset + len(_PDF_MAGIC)
        if version_pos >= len(data) or not chr(data[version_pos]).isdigit():
            return "declared .pdf but the %PDF- header is not followed by a version digit"
        # Require at least one real structural marker somewhere in the file.
        if not any(marker in data for marker in _PDF_STRUCTURE_MARKERS):
            return (
                "declared .pdf with a %PDF- header but no PDF structural marker "
                "(xref/trailer/obj/%%EOF) — likely a spoofed header"
            )
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
# not lost data. A CPU obfuscation-tolerant second Provider ships behind the same seam
# (FuzzyInjectionContentScreener, #77/ADR-0035); a heavier ML classifier remains a
# further drop-in behind it (ADR-0004).
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
# Width of the seam window scanned across an adjacent-chunk boundary: the tail of
# the previous chunk + the head of the next. Bounded (not the whole chunks) so the
# boundary pass stays linear in the number of chunks and cannot be turned into a
# quadratic blow-up by a pathological corpus. 240 chars comfortably spans the
# longest injection pattern even when the split lands mid-phrase.
_BOUNDARY_WINDOW = 240


class KeywordContentScreener:
    """Content Screener: prompt-injection + harmful-content flagging on chunks.

    Detects both **in-chunk** injection (a pattern wholly inside one chunk) and
    **cross-chunk-boundary** injection (a pattern split across two adjacent chunks of
    the same document, which evades per-chunk scanning — issue #62): the seam of each
    consecutive same-document pair is re-scanned over a bounded window, and a hit that
    fires only across the boundary flags BOTH chunks so the corpus carries the marker.

    Config:
        quarantine_threshold: float  — a chunk whose risk score reaches this is
            Quarantined instead of flagged (default 1.01 — i.e. never, flag-only by
            default; set below 1.0 to escalate any hit to Quarantine).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The keyword Content Screener catches prompt-injection/harmful markers by scan.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        # Default > 1.0 so the reference Provider flags rather than drops (a single
        # hit scores 1.0). Lowering it escalates strong hits to Quarantine.
        self._quarantine_threshold = float(config.get("quarantine_threshold", 1.01))

    def screen_content(self, chunks: Iterable[ChunkRecord]) -> ScreeningOutcome[ChunkRecord]:
        # Materialise so adjacent chunks can be inspected for a boundary-straddling
        # injection; output order is the input order (unchanged contract).
        chunk_list = list(chunks)
        passed: list[ChunkRecord | None] = []
        quarantined: list[QuarantineRecord] = []
        findings: list[ScreeningFinding] = []
        # Whether each chunk already fired an in-chunk injection (so the boundary pass
        # does not double-flag it) and its slot in ``passed`` (None once quarantined).
        fired_injection: list[bool] = []
        for chunk in chunk_list:
            category, score, reason = self._score(chunk.content)
            fired_injection.append(category == "prompt_injection")
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
                passed.append(None)
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

        self._flag_cross_chunk_injection(chunk_list, passed, fired_injection, findings)
        return ScreeningOutcome(
            passed=[c for c in passed if c is not None],
            quarantined=quarantined,
            findings=findings,
        )

    def _flag_cross_chunk_injection(
        self,
        chunk_list: list[ChunkRecord],
        passed: list[ChunkRecord | None],
        fired_injection: list[bool],
        findings: list[ScreeningFinding],
    ) -> None:
        """Second pass: flag an injection split across an adjacent same-document seam.

        For each consecutive pair of chunks from the SAME document (in ``index`` order)
        where neither already fired an in-chunk injection, scan the bounded seam window
        (tail of the previous chunk + head of the next). A pattern that matches only
        across the boundary — but not inside either chunk alone — is a smuggled
        injection; both chunks are flagged with a cross-boundary ``prompt_injection``
        marker so the corpus carries it, and a FLAG finding is recorded for each.
        """
        for i in range(len(chunk_list) - 1):
            prev_chunk, next_chunk = chunk_list[i], chunk_list[i + 1]
            # Only consecutive chunks of the same document form a real reading seam.
            if prev_chunk.document_record_id != next_chunk.document_record_id:
                continue
            if next_chunk.index != prev_chunk.index + 1:
                continue
            # If either chunk already carries an in-chunk injection, the seam is
            # already covered — do not double-flag.
            if fired_injection[i] or fired_injection[i + 1]:
                continue
            # Chunks already quarantined (removed) are not re-flagged.
            if passed[i] is None or passed[i + 1] is None:
                continue
            match = self._boundary_injection_match(prev_chunk.content, next_chunk.content)
            if match is None:
                continue
            reason = f"matched injection pattern {match!r} split across a chunk boundary"
            marker = RiskMarker(
                category="prompt_injection",
                detector="screening.content_keyword",
                score=1.0,
                reason=reason,
            )
            for slot in (i, i + 1):
                current = passed[slot]
                if current is None:
                    continue
                passed[slot] = current.model_copy(
                    update={"risk_markers": [*current.risk_markers, marker]}
                )
                findings.append(
                    self._finding(
                        chunk_list[slot],
                        ScreeningVerdict.FLAG,
                        "prompt_injection",
                        reason,
                        marker,
                    )
                )

    @staticmethod
    def _boundary_injection_match(prev_text: str, next_text: str) -> str | None:
        """Return the matched phrase if an injection straddles the ``prev|next`` seam.

        Scans a bounded window (tail of ``prev`` + head of ``next``). A match counts
        only if it actually crosses the boundary — i.e. begins in ``prev``'s tail and
        continues into ``next`` — so an injection wholly inside one side (already
        caught by the per-chunk pass) is not re-reported here.

        A chunker typically splits on and consumes the whitespace between two chunks,
        so the raw tail and head often abut with no separator (``…previous`` +
        ``instructions``). When neither side already carries boundary whitespace, a
        single reconstructed space is inserted at the seam so a phrase split exactly on
        a word gap is still recognised; the space's presence is accounted for when
        testing that the match straddles the boundary.
        """
        tail = prev_text[-_BOUNDARY_WINDOW:]
        head = next_text[:_BOUNDARY_WINDOW]
        # Reconstruct the likely-consumed word separator so a split-on-whitespace
        # boundary does not glue two tokens into one and hide the phrase.
        separator = ""
        if tail and head and not tail[-1].isspace() and not head[0].isspace():
            separator = " "
        seam = (tail + separator + head).lower()
        boundary = len(tail)
        next_start = len(tail) + len(separator)
        for pattern in _INJECTION_PATTERNS:
            for m in pattern.finditer(seam):
                # Straddles the seam: begins in the tail (at/before the boundary) and
                # ends in the head (past the reconstructed separator).
                if m.start() < boundary and m.end() > next_start:
                    return m.group(0)
        return None

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


# Compacted injection signatures for the fuzzy Content Screener: the same intents as the
# reference keyword screener's patterns, reduced to a canonical alphanumeric-only form so
# an obfuscated variant ("i-g-n-o-r-e p.r.e.v.i.o.u.s") collapses onto the signature after
# the same normalisation is applied to the input. Deliberately small + high-precision.
_FUZZY_INJECTION_SIGNATURES: tuple[str, ...] = (
    "ignoreallpreviousinstructions",
    "ignorepreviousinstructions",
    "ignorepriorinstructions",
    "ignoreaboveinstructions",
    "disregardpreviousinstructions",
    "disregardallprevious",
    "disregardpriorinstructions",
    "systemprompt",
    "revealyourprompt",
    "revealthesystemprompt",
    "revealyourinstructions",
    "overridethesafety",
    "overrideyourinstructions",
    "overridethesafetyinstructions",
    "doanythingnow",
    "youarenowin",
)
_FUZZY_HARMFUL_SIGNATURES: tuple[str, ...] = (
    "buildabomb",
    "makeaweapon",
)
# Hard cap on the number of ALPHANUMERIC characters kept per chunk — i.e. on the length of
# the COMPACTED form, not on the raw prefix. Compaction is a single linear pass (no regex, no
# backtracking — ReDoS-proof by construction) and stops early once this many signal characters
# have been collected, so the compacted string a signature is matched against is bounded and
# the match cost is O(cap) regardless of chunk size. Capping the compacted length rather than a
# raw prefix is deliberate: cost-free separators (whitespace/punctuation) do NOT consume the
# budget, so an attacker cannot pad an injection past the window with leading separators — the
# exact evasion a raw-prefix cap would let through (#77 security review). An injection signature
# is at most a few dozen chars, so a window this wide cannot hide one that matters. The raw scan
# is additionally bounded by ``_FUZZY_RAW_SCAN_CAP`` so a megabyte of pure separators (which
# never fills the alphanumeric budget) still costs a bounded constant, not O(chunk length).
_FUZZY_SCAN_CAP = 8192
# Absolute ceiling on raw characters iterated, independent of how many are alphanumeric. Guards
# the pathological all-separator chunk that would otherwise never reach ``_FUZZY_SCAN_CAP`` kept
# characters and so scan the whole (arbitrarily large) input. Set well above ``_FUZZY_SCAN_CAP``
# so a realistically-obfuscated injection (a bounded run of interleaved separators around a
# few-dozen-char phrase) is never truncated before its signal characters are collected.
_FUZZY_RAW_SCAN_CAP = 1 << 20  # 1 MiB of raw characters


def _compact(text: str, max_kept: int = _FUZZY_SCAN_CAP) -> str:
    """Collapse ``text`` to lowercase alphanumerics only — the fuzzy-match canonical form.

    A single linear pass that keeps only ``[a-z0-9]`` (lowercasing as it goes) and drops
    EVERYTHING else — whitespace, punctuation, zero-width and combining characters, and any
    other separator an adversary interleaves to break a literal keyword match. There is no
    regex and no backtracking, so it cannot be driven into super-linear time (ReDoS-proof).

    The pass stops once ``max_kept`` alphanumeric characters have been collected — a cap on the
    COMPACTED output, so free separators do not consume the budget and cannot push an injection
    past the window (a raw-prefix cap could be defeated by leading separator-padding — #77). A
    secondary ``_FUZZY_RAW_SCAN_CAP`` bounds the raw characters iterated so a pathological
    all-separator chunk (which never fills the alphanumeric budget) is still O(1)-bounded.

    So ``i g n o r e   p r e v i o u s`` and ``ig.no.re/pre.vi.ous`` both collapse onto the
    same ``ignoreprevious…`` signature form.
    """
    out: list[str] = []
    for scanned, ch in enumerate(text):
        if scanned >= _FUZZY_RAW_SCAN_CAP:
            break
        if ch.isalnum():
            # ASCII alphanumerics are the signature alphabet. A non-ASCII alphanumeric (e.g. a
            # full-width or Cyrillic look-alike) is kept in lowercased form so a homoglyph-padded
            # token is not silently dropped, but it only matches a signature if it lands on the
            # same ASCII letters — which it will not for a genuine homoglyph, so this neither
            # over- nor under-matches the ASCII set.
            out.append(ch.lower())
            if len(out) >= max_kept:
                break
    return "".join(out)


class FuzzyInjectionContentScreener:
    """Second reference Content Screener Provider (#77): obfuscation-tolerant injection detector.

    An alternative CPU Content Screener behind the same
    :class:`~latence_core.capability.ContentScreener` seam as :class:`KeywordContentScreener`,
    satisfying the identical typed contract — it returns the passed chunks (some carrying a
    propagated :class:`~latence_core.contracts.RiskMarker`), any Quarantined chunks, and the
    per-chunk :class:`~latence_core.contracts.ScreeningFinding` log. Pure Python, deterministic,
    zero extra deps (ADR-0007 CPU-first, ADR-0016 thin core); no model, so no license note
    (ADR-0012).

    The difference vs the reference is **recall against obfuscation**: the keyword screener
    matches injection phrases as literal (whitespace-tolerant) regexes, so an attacker who
    interleaves separators — ``i g n o r e   p r e v i o u s   i n s t r u c t i o n s`` or
    ``ignore/previous.instructions`` — slips past it. This Provider reduces the chunk text to a
    canonical alphanumeric-only form (:func:`_compact`) and matches a small set of compacted
    injection signatures, so a separator-padded variant still fires. That is the measurable
    benchmark vs the existing one (``test_screening_second_provider``): a corpus of obfuscated
    injections the keyword screener misses and this one catches.

    **Security / DoS.** Compaction is a single linear pass with no regex and no backtracking
    (ReDoS-proof by construction). The pass stops after collecting ``_FUZZY_SCAN_CAP``
    *alphanumeric* characters — a cap on the compacted output, so cost-free separators do not
    consume the budget and an attacker cannot pad an injection past the window with leading
    separators (a raw-prefix cap could be defeated that way — #77 security review); a secondary
    ``_FUZZY_RAW_SCAN_CAP`` bounds the raw characters iterated, so a pathological all-punctuation
    megabyte chunk still costs a bounded constant, not a blow-up. A hit **flags** (propagates a
    marker) by default rather than dropping, mirroring the reference (a legitimate document may
    quote an injection string); a ``quarantine_threshold`` below 1.0 escalates a hit to
    Quarantine.

    Config:
        quarantine_threshold: float  — a chunk whose score reaches this is Quarantined instead
            of flagged (default 1.01 — flag-only; set below 1.0 to escalate any hit).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The fuzzy-injection Content Screener is a bounded obfuscation-tolerant scorer.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._quarantine_threshold = float(config.get("quarantine_threshold", 1.01))

    def screen_content(self, chunks: Iterable[ChunkRecord]) -> ScreeningOutcome[ChunkRecord]:
        passed: list[ChunkRecord] = []
        quarantined: list[QuarantineRecord] = []
        findings: list[ScreeningFinding] = []
        for chunk in chunks:
            category, score, reason = self._score(chunk.content)
            if category is None:
                passed.append(chunk)
                findings.append(self._finding(chunk, ScreeningVerdict.PASS, "clean", "ok", None))
                continue
            if score >= self._quarantine_threshold:
                quarantined.append(self._quarantine(chunk, category, reason))
                findings.append(
                    self._finding(chunk, ScreeningVerdict.QUARANTINE, category, reason, None)
                )
                continue
            marker = RiskMarker(
                category=category,
                detector="screening.content_fuzzy",
                score=min(score, 1.0),
                reason=reason,
            )
            flagged = chunk.model_copy(update={"risk_markers": [*chunk.risk_markers, marker]})
            passed.append(flagged)
            findings.append(self._finding(chunk, ScreeningVerdict.FLAG, category, reason, marker))
        return ScreeningOutcome(passed=passed, quarantined=quarantined, findings=findings)

    @staticmethod
    def _score(text: str) -> tuple[str | None, float, str]:
        """Return ``(category|None, score, reason)`` for a chunk's obfuscation-compacted text."""
        # Cap the COMPACTED length inside _compact (free separators do not consume the budget),
        # not a raw prefix — a raw-prefix cap is defeated by leading separator-padding (#77).
        compacted = _compact(text)
        for signature in _FUZZY_INJECTION_SIGNATURES:
            if signature in compacted:
                return (
                    "prompt_injection",
                    1.0,
                    f"matched obfuscation-normalised injection signature {signature!r}",
                )
        for signature in _FUZZY_HARMFUL_SIGNATURES:
            if signature in compacted:
                return (
                    "harmful",
                    1.0,
                    f"matched obfuscation-normalised harmful signature {signature!r}",
                )
        return (None, 0.0, "no risk signature matched")

    @staticmethod
    def _quarantine(chunk: ChunkRecord, category: str, reason: str) -> QuarantineRecord:
        return QuarantineRecord(
            record_id=chunk.record_id,
            provenance=chunk.provenance,
            classification=chunk.classification,
            checkpoint=ScreeningCheckpoint.CONTENT,
            detector="screening.content_fuzzy",
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
            detector="screening.content_fuzzy",
            category=category,
            reason=reason,
            marker=marker,
        )

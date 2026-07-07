"""Built-in reference PIIDetector Provider (deterministic, hybrid rule, zero-dep).

The S6 Redaction reference Provider (ADR-0004; ships in core like the S3
Chunker/Screeners, the S4 gazetteer and the S5 trigger extractor so the demo runs
end to end, ADR-0007 CPU-first, ADR-0016 no extra deps). It is the **rule half** of
the "hybrid rule+model" reference the issue calls for: a library of well-known-PII
and secret recognisers (email, phone, SSN, credit card, IP, IBAN, AWS access-key id,
labelled API token — bare ``token:``/``jwt:`` and the session/refresh variants
included, structural JWT, and a password/secret assignment that also fires inside a
compound identifier such as ``aws_secret_access_key``) enabled comprehensively by
default, plus a config-supplied gazetteer and custom-pattern set. The **model half** is the drop-in,
separately-packaged ``latence-pii-gliner`` (a permissively-licensed GLiNER-PII
checkpoint via plain ``transformers``, ADR-0012/0016) behind this same
:class:`~latence_core.capability.PIIDetector` seam — shipping the deterministic
rule reference in core keeps first contact (``git clone`` → working redaction) free
of torch and byte-stable for the Baseline bar.

Redaction is **metadata-aware** (S6 AC): the Provider keys its policy off each
document's ``Classification.sensitivity``. A ``policies`` map says, per sensitivity
value, which :class:`~latence_core.contracts.RedactionAction` to take (mask / replace
/ hash / tag) and, optionally, which PII types to act on — so a ``public`` document
can be masked hard while an ``internal`` one is only tagged for review, from the same
Pipeline with no code change.

Each emitted :class:`~latence_core.contracts.RedactionRecord` carries:

* ``masked_content`` — the document's assembled markdown with every non-``TAG`` PII
  span replaced by its deterministic placeholder;
* ``pii_spans`` — the structured, auditable spans (type, **document offset**, action,
  placeholder, resolved source page) — never the raw PII value, so the audit trail is
  not itself a PII leak (S6 AC: no raw PII written);
* ``policy`` — the applied policy name (the sensitivity value, or ``default``);
* Provenance covering ``[0, len)`` of the document, the parent's Classification
  inherited unchanged.

A document with no detected PII yields a RedactionRecord whose masked text equals the
source and whose span list is empty (never an error), so the corpus always has one
PII-handled variant per document.

Determinism: within a document, overlapping matches are resolved by (start, longer
span, type) and non-``TAG`` replacements are applied right-to-left so earlier offsets
never shift; documents are emitted in first-seen order. A ``replace``/``hash`` action
derives its surrogate from a config ``salt`` + the value, so a seeded run is
byte-identical without ever persisting the value.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, PIISpan, RedactionRecord
from latence_core.pagemap import PageOffsetIndex
from latence_core.redaction_policy import (
    RedactionPolicy,
    apply_masking,
    build_span,
    parse_action,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord

# The built-in rule library: well-known PII/secret types with conservative,
# whole-value recognisers. Selectable by name via config ``detectors`` (default:
# all — comprehensive coverage). These are literal, offline, dependency-free — the
# deterministic rule half of the hybrid. Every pattern is linear-time (no nested
# quantifier over an overlapping class), so a pathological input cannot induce
# catastrophic backtracking (ReDoS).
_BUILTIN_PATTERNS: dict[str, str] = {
    # RFC-5322-ish email (deliberately conservative, not exhaustive). Linear-time:
    # a naive ``\b[local]+@[domain]+\.tld`` backtracks catastrophically on a long
    # dotted run because the local class ``[A-Za-z0-9._%+-]`` and the domain class
    # ``[A-Za-z0-9.-]`` both contain ``.`` and letters — and ``\b`` matches at every
    # word/non-word transition in ``x.x.x…``, so ``finditer`` relaunches a full
    # greedy scan from each of O(n) starts → O(n²). Fixed structurally: a negative
    # lookbehind pins the local part to its true start (no re-entry mid-run, since the
    # preceding char is never a local-part char), a possessive ``++`` local part never
    # backtracks, and the domain is a run of ``label.`` units whose label class
    # ``[A-Za-z0-9-]`` excludes ``.`` (no overlap). The single greedy outer ``+`` on
    # the domain backtracks only to peel a trailing sentence dot off the last label,
    # which is O(labels) once per (unique) ``@`` — linear overall.
    "email": (
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@(?:[A-Za-z0-9-]++\.)+[A-Za-z]{2,}\b"
    ),
    # US SSN NNN-NN-NNNN.
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    # 13–16 digit card number, optional single space/dash between digits.
    "credit_card": r"\b\d(?:[ -]?\d){12,15}\b",
    # International-ish phone: optional +, 7–15 digits with separators.
    "phone": r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)",
    # IPv4.
    "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # IBAN: 2 letters, 2 check digits, up to 30 alnum.
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
    # AWS access key id: literal AKIA/ASIA/AGPA/… prefix + 16 uppercase alnum.
    "aws_key": r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|A3T[A-Z0-9])[A-Z0-9]{16}\b",
    # Generic long high-entropy API token: a labelled key/token (including a bare
    # ``token:``/``jwt:`` and the session/refresh variants) followed by a 20+ char
    # base64/hex-ish credential (bounded, linear-time). The label set is a fixed
    # alternation of whole words — no nested quantifier over the value class.
    "api_key": (
        r"(?i:\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|session[_-]?token"
        r"|refresh[_-]?token|bearer|token|jwt|secret[_-]?key)\b)"
        r"['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+_.=-]{20,})"
    ),
    # A structural JSON Web Token: three base64url segments joined by dots, the header
    # segment starting ``eyJ`` (base64 of ``{"``). Catches a **bare** ``token: <JWT>``
    # even when the label is unrecognised, absent, or on another line — the value's own
    # shape is the recogniser. Linear-time: each segment is a single character class
    # under a fixed `+`, the segments separated by literal dots (no overlap).
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b",
    # A ``password``/``passwd``/``pwd``/``secret`` assignment — including when the
    # keyword is a **component of a compound identifier** (``aws_secret_access_key``,
    # ``client_secret``, ``db_password``): the plain ``\bsecret\b`` word boundary fails
    # inside ``…_secret_…`` because ``_`` is a word char, so match an optional single
    # leading identifier component and any number of trailing ones around the keyword.
    # Linear-time: the leading component ends in a separator and each trailing component
    # starts with one (both disjoint from the alnum run's edge), so there is no
    # catastrophic backtracking on a pathological run.
    "secret": (
        r"(?i:\b(?:[a-z0-9]+[_.-])?(?:password|passwd|pwd|secret)(?:[_.-][a-z0-9]+)*)"
        r"['\"]?\s*[:=]\s*['\"]?(\S{3,})"
    ),
}
# Recognisers whose credential value is captured in group(1) (a labelled
# assignment): the span must cover only the secret, not the label. All other
# recognisers match the whole PII value as group(0).
_CAPTURE_GROUP_DETECTORS: frozenset[str] = frozenset({"api_key", "secret"})

# Specificity rank used only to break ties when two recognisers claim the SAME
# span (same start, same length): the more specific, structured recogniser wins its
# label so a per-type recall/precision benchmark is not polluted by the loose phone
# recogniser mislabelling an SSN. Lower rank = higher priority. Config-supplied
# gazetteer/pattern recognisers and unlisted types fall back to the default rank, so
# they never lose their own exact spans to a built-in (they only ever tie on an
# identical span, and a stable, type-alphabetical order then decides — unchanged).
_SPECIFICITY_RANK: dict[str, int] = {
    "ssn": 0,
    "credit_card": 0,
    "iban": 0,
    "aws_key": 0,
    "jwt": 0,  # structural, whole-value token — wins a tie over a loose label capture
    "email": 1,
    "api_key": 1,
    "secret": 1,
    "ip": 1,
    "phone": 5,  # loosest built-in: loses a tie to any structured recogniser above
}
_DEFAULT_RANK = 3


class _Detector:
    """One compiled PII recogniser: a type label + a pattern + which engine fired.

    ``group`` is the capture group whose span is masked: 0 for a whole-value
    recogniser, 1 for a labelled-assignment recogniser (``password: <value>``)
    where only the credential — not the label — must be redacted.
    """

    __slots__ = ("pii_type", "pattern", "detector", "confidence", "group")

    def __init__(
        self,
        pii_type: str,
        pattern: re.Pattern[str],
        detector: str,
        confidence: float,
        group: int = 0,
    ) -> None:
        self.pii_type = pii_type
        self.pattern = pattern
        self.detector = detector
        self.confidence = confidence
        self.group = group


# A raw match before it becomes a PIISpan: type, document-offset span, the surface
# value (used only transiently to derive a surrogate — never persisted), the
# detector name, and the confidence.
_Match = tuple[str, int, int, str, str, float]


class HybridPIIRedactor:
    """Zero-dep hybrid-rule PIIDetector Provider, policy-keyed off Classification.

    The PII type set and how to recognise each type come from the built-in rule
    library plus ``config`` — nothing about *which* document gets *what* treatment is
    hardcoded: that is the sensitivity-keyed ``policies`` map.

    Config:
        detectors: list[str]  — which built-in recognisers to enable (subset of
            ``email``/``ssn``/``credit_card``/``phone``/``ip``/``iban``/``aws_key``/
            ``api_key``/``jwt``/``secret``). Default: **all** — comprehensive PII+secret
            coverage, so a corpus is PII-handled by default (do not silently narrow
            this to a partial set in production configs).
        gazetteer: dict[str, list[str]]  — extra literal terms to redact, keyed by PII
            type (e.g. ``{"name": ["Alice", "Bob"]}``) — whole-word, case-insensitive
            by default. The config-supplied half of the rule set.
        patterns: dict[str, list[str]]  — extra custom regexes keyed by PII type.
        default_action: str  — action when a policy names none: ``mask`` (default),
            ``replace``, ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map. Each key is a
            ``Classification.sensitivity`` value (``public``/``internal``/… or the
            literal ``default`` fallback); each value may set:
              * ``action``: str  — the action for this sensitivity (overrides
                ``default_action``).
              * ``types``: list[str]  — only act on these PII types (default: all
                detected). Types not listed are ignored (not even tagged).
              * ``skip``: bool  — if true, this sensitivity is passed through with an
                empty span list (redaction disabled for that class).
        salt: str  — salt mixed into ``replace``/``hash`` surrogates so equal values
            map to equal (but non-reversible) placeholders deterministically.
        case_sensitive: bool  — match gazetteer terms case-sensitively (default False).
        min_chars: int  — a document whose assembled text is shorter than this is
            emitted as a clean (no-PII) RedactionRecord (default 1).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._case_sensitive = bool(config.get("case_sensitive", False))
        self._min_chars = int(config.get("min_chars", 1))
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        self._detectors = self._compile_detectors(config)

    # -- config compilation -------------------------------------------------

    def _compile_detectors(self, config: dict[str, Any]) -> list[_Detector]:
        """Compile the enabled built-in rules + config gazetteer/patterns into recognisers."""
        flags = 0 if self._case_sensitive else re.IGNORECASE
        enabled = config.get("detectors")
        if enabled is not None and not isinstance(enabled, list):
            msg = "'detectors' config must be a list of built-in recogniser names."
            raise TypeError(msg)
        names = list(enabled) if enabled is not None else list(_BUILTIN_PATTERNS)
        detectors: list[_Detector] = []
        for name in names:
            builtin = _BUILTIN_PATTERNS.get(str(name))
            if builtin is None:
                available = ", ".join(sorted(_BUILTIN_PATTERNS))
                msg = f"Unknown built-in PII detector {name!r}. Available: {available}."
                raise ValueError(msg)
            # Built-in patterns are ASCII-cased already; IGNORECASE is harmless.
            group = 1 if name in _CAPTURE_GROUP_DETECTORS else 0
            detectors.append(
                _Detector(str(name), re.compile(builtin, flags), f"regex:{name}", 0.9, group)
            )
        gazetteer = config.get("gazetteer", {})
        if not isinstance(gazetteer, dict):
            msg = "'gazetteer' config must be a mapping of pii_type -> list[str]."
            raise TypeError(msg)
        for pii_type, terms in gazetteer.items():
            alternatives = [rf"\b{re.escape(str(term))}\b" for term in terms]
            if not alternatives:
                continue
            pattern = re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
            detectors.append(
                _Detector(str(pii_type), pattern, f"gazetteer:{pii_type}", 0.99)
            )
        patterns = config.get("patterns", {})
        if not isinstance(patterns, dict):
            msg = "'patterns' config must be a mapping of pii_type -> list[str]."
            raise TypeError(msg)
        for pii_type, raw_patterns in patterns.items():
            alternatives = [str(pat) for pat in raw_patterns]
            if not alternatives:
                continue
            pattern = re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
            detectors.append(
                _Detector(str(pii_type), pattern, f"pattern:{pii_type}", 0.9)
            )
        return detectors

    # -- detection ----------------------------------------------------------

    def redact(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """Detect PII per document; each :class:`DocumentRecord` supplies text + page map.

        A detector scans the document's assembled markdown (``record.content``) at its
        original-document offsets — the coordinate system the masked variant and every
        :class:`PIISpan` use — and resolves each span's page through ``record.page_map``.
        A ``PARSE_ERROR`` document (no usable content) is skipped, mirroring Chunk.
        Records are emitted in first-seen document order for determinism.
        """
        for document in documents:
            if document.disposition is Disposition.PARSE_ERROR:
                continue
            yield self._redact_one_document(document)

    def _redact_one_document(self, document: DocumentRecord) -> RedactionRecord:
        doc_id = document.record_id
        text = document.content
        page_map = document.page_map
        classification = document.classification
        base_prov = document.provenance
        action, allowed, skip, policy_name = self._policy.resolve(classification.sensitivity)

        spans: list[PIISpan] = []
        masked = text
        if not skip and len(text.strip()) >= self._min_chars:
            matches = self._find_matches(text, allowed)
            page_index = PageOffsetIndex(page_map) if page_map is not None else None
            spans = [
                build_span(
                    pii_type=pii_type,
                    detector=detector,
                    start=start,
                    end=end,
                    value=value,
                    action=action,
                    salt=self._salt,
                    page_index=page_index,
                    confidence=confidence,
                )
                for pii_type, start, end, value, detector, confidence in matches
            ]
            masked = apply_masking(text, spans)

        # A RedactionRecord's Provenance covers the whole document [0, len). Source
        # identity is inherited from the DocumentRecord's Provenance; the offset span is
        # set to the document extent.
        provenance = base_prov.model_copy(
            update={
                "char_start": 0,
                "char_end": len(text),
                "page_start": None,
                "page_end": None,
            }
        )
        return RedactionRecord(
            record_id=f"{doc_id}#redaction",
            document_record_id=doc_id,
            masked_content=masked,
            media_type="text/markdown",
            policy=policy_name,
            pii_spans=spans,
            provenance=provenance,
            classification=classification,
        )

    def _find_matches(self, text: str, allowed: set[str] | None) -> list[_Match]:
        """All non-overlapping PII matches across enabled detectors, deterministically.

        Each detector scans independently; a type not in ``allowed`` (when a policy
        restricts types) is skipped entirely. Overlaps between detectors are resolved
        by (earlier start, longer span, more specific type, type name) so a stable
        non-overlapping set results — the specificity rank ensures a structured
        recogniser (SSN, credit card, …) claims a span the loose phone recogniser also
        matched, so the per-type label is correct.
        """
        raw: list[_Match] = []
        for detector in self._detectors:
            if allowed is not None and detector.pii_type not in allowed:
                continue
            for m in detector.pattern.finditer(text):
                # For a labelled-assignment recogniser, redact only the captured
                # credential (group 1) — never the ``password:`` label; otherwise
                # the whole matched value (group 0).
                group = detector.group
                start, end = m.start(group), m.end(group)
                if start < 0 or end <= start:
                    continue
                raw.append(
                    (detector.pii_type, start, end, m.group(group), detector.detector,
                     detector.confidence)
                )
        raw.sort(
            key=lambda t: (
                t[1],
                -(t[2] - t[1]),
                _SPECIFICITY_RANK.get(t[0], _DEFAULT_RANK),
                t[0],
            )
        )
        accepted: list[_Match] = []
        last_end = -1
        for match in raw:
            _, start, end, _, _, _ = match
            if start >= last_end:
                accepted.append(match)
                last_end = end
        return accepted

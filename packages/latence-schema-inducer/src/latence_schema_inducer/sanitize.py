"""Label sanitization for induced labels (W4 safety net #5 — anti-injection / anti-bloat).

An induced label is UNTRUSTED LLM output that becomes data for a downstream zero-shot extractor
prompt — never executed, but it MUST be bounded so a malicious or degenerate model cannot bloat the
extractor prompt (thousands of labels), smuggle megabytes (a single enormous label), or inject
control characters / prompt-injection payloads into the label stream. This module is the pure,
deterministic sanitizer the Provider runs over every list the model returns, before it ever reaches
an :class:`~latence_core.contracts.InducedLabels`:

  * coerce each item to ``str`` (a model returning an int/None for a label yields its str form or is
    dropped if empty);
  * strip surrounding whitespace and remove control characters (C0/C1 + DEL) — the anti-injection
    + anti-control-char rule (a newline-laden label can't reshape the extractor prompt);
  * collapse internal runs of whitespace to a single space (a label is a type name, not a passage);
  * drop empty / whitespace-only labels;
  * cap each label's length at :data:`MAX_LABEL_LEN` (64) characters — the anti-smuggle bound;
  * de-duplicate **case-insensitively** while preserving first-seen order (``Person`` and ``person``
    collapse to the first);
  * cap the COUNT per type at :data:`MAX_LABELS` (40) — the anti-bloat bound; the overflow is
    reported (not silently truncated) so the drop is auditable.

Deterministic + order-stable (first-seen order), so a seeded run is byte-reproducible (Baseline
bar). Nothing here raises on bad input — a wholly-invalid list sanitizes to ``[]`` (the Provider
then falls back to config labels), never a crash (the fail-open discipline).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: The per-label character cap (anti-smuggle): a label longer than this is truncated.
MAX_LABEL_LEN = 64
#: The per-type label-count cap (anti-bloat): labels beyond this are dropped and reported.
MAX_LABELS = 40

# Control characters to strip: Unicode "Cc" (C0/C1 control + DEL) and the zero-width /
# bidi-override formatting chars ("Cf") a prompt-injection payload can hide in. A label is a plain
# type name, so any of these is noise to remove, not preserve.
_CONTROL_CATEGORIES = frozenset({"Cc", "Cf"})
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class SanitizeResult:
    """The sanitized labels plus the count dropped by the count cap (auditable, W4 §5).

    ``labels`` is the bounded, deduped, order-stable list an extractor can safely prompt with.
    ``dropped_for_count`` is how many otherwise-valid labels the :data:`MAX_LABELS` cap removed — a
    non-zero value is the "no silent truncation" audit signal the Provider logs + surfaces.
    """

    labels: list[str]
    dropped_for_count: int = 0


def _strip_controls(text: str) -> str:
    """Remove NON-whitespace control / format characters, preserving printable + whitespace chars.

    Whitespace control chars (tab, newline, CR) are KEPT here so the subsequent whitespace-run
    collapse turns them into a single space (``"card\\tnumber"`` → ``"card number"``, not
    ``"cardnumber"``); only non-whitespace controls (NUL, BEL, zero-width/bidi format chars) are
    removed. ``str.isspace()`` covers every Unicode whitespace, so a whitespace-category control is
    preserved for the collapse step.
    """
    return "".join(
        ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in _CONTROL_CATEGORIES
    )


def sanitize_label(raw: object) -> str | None:
    """Sanitize ONE candidate label; return the cleaned string, or ``None`` if it is not usable.

    Coerces to ``str``, strips control chars + surrounding whitespace, collapses internal whitespace
    runs, and caps the length at :data:`MAX_LABEL_LEN`. Returns ``None`` for an empty / whitespace-
    only / non-coercible candidate so the caller drops it (never a fabricated placeholder).
    """
    try:
        text = raw if isinstance(raw, str) else str(raw)
    except Exception:  # noqa: BLE001 - a pathological __str__ never crashes sanitization
        return None
    text = _strip_controls(text)
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    if not text:
        return None
    if len(text) > MAX_LABEL_LEN:
        text = text[:MAX_LABEL_LEN].strip()
    return text or None


def sanitize_labels(raw_labels: object) -> SanitizeResult:
    """Sanitize a whole candidate list into a bounded, deduped, order-stable label set (W4 §5).

    A non-list ``raw_labels`` (the model returned a string / dict / null for a slot) yields an empty
    result — the schema-validation safety net rejects that shape upstream, but sanitizing defends in
    depth. Each item is cleaned by :func:`sanitize_label`; usable items are de-duplicated
    case-insensitively (first-seen casing wins) and the count is capped at :data:`MAX_LABELS`, with
    the overflow reported in :attr:`SanitizeResult.dropped_for_count`.
    """
    if not isinstance(raw_labels, list):
        return SanitizeResult(labels=[], dropped_for_count=0)
    seen_lower: set[str] = set()
    kept: list[str] = []
    dropped_for_count = 0
    for raw in raw_labels:
        cleaned = sanitize_label(raw)
        if cleaned is None:
            continue
        key = cleaned.casefold()
        if key in seen_lower:
            continue
        if len(kept) >= MAX_LABELS:
            # Anti-bloat: cap the count, but count the drop (no silent truncation) so the Provider
            # can log + surface how many labels the cap removed.
            dropped_for_count += 1
            continue
        seen_lower.add(key)
        kept.append(cleaned)
    return SanitizeResult(labels=kept, dropped_for_count=dropped_for_count)

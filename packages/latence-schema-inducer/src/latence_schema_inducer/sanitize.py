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
  * emit ONE **canonical normal form** per label (W12): ``_`` and ``-`` are treated as
    whitespace-equivalent separators (a run of any of them collapses to a single space) and the
    result is ``casefold()``-ed, so ``payment_product`` / ``payment-product`` / ``payment product``
    and ``Date`` / ``date`` all become the SAME string — the fix for corpus-wide label fragmentation
    (case + format variants splitting one type into many). This is a PURE per-string transform (no
    cross-call state), so every variant maps to the same normal form regardless of streaming order;
  * drop empty / whitespace-only labels;
  * cap each label's length at :data:`MAX_LABEL_LEN` (64) characters — the anti-smuggle bound;
  * de-duplicate on the canonical form while preserving first-seen order (``Person``, ``person`` and
    ``PERSON`` all collapse to one ``person``);
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
#: The character cap on the operator's ``domain`` hint (anti-bloat): a hint longer than this is
#: truncated. 512 chars is ample for a real description ("German customs law: tariff classification,
#: binding tariff information, export-control rulings") while bounding what an operator — or a
#: config an operator was handed — can push into every induction prompt.
MAX_DOMAIN_LEN = 512

# Control characters to strip: Unicode "Cc" (C0/C1 control + DEL) and the zero-width /
# bidi-override formatting chars ("Cf") a prompt-injection payload can hide in. A label is a plain
# type name, so any of these is noise to remove, not preserve.
_CONTROL_CATEGORIES = frozenset({"Cc", "Cf"})
# A run of separators — Unicode whitespace OR ``_`` / ``-`` — collapses to a single space in the
# canonical normal form (W12). Treating ``_``/``-`` as space-equivalent is what makes
# ``payment_product`` and ``payment product`` (and ``payment-product``) the SAME label corpus-wide.
_SEPARATOR_RUN = re.compile(r"[\s_-]+")


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
    """Sanitize ONE candidate label into its canonical normal form, or ``None`` if not usable (W12).

    Coerces to ``str``, strips control chars, collapses every separator run (whitespace / ``_`` /
    ``-``) to a single space, ``casefold()``-s the result, and caps the length at
    :data:`MAX_LABEL_LEN`. The casefold + separator collapse is the canonical normal form: ``Date``,
    ``date``, ``payment_product`` and ``payment product`` all map to one string, so case/format
    variants no longer fragment a type corpus-wide. Returns ``None`` for an empty / whitespace-only
    / non-coercible candidate so the caller drops it (never a fabricated placeholder). A pure
    function of ``raw`` alone — no cross-call state — so it holds under per-doc / streaming
    induction.
    """
    try:
        text = raw if isinstance(raw, str) else str(raw)
    except Exception:  # noqa: BLE001 - a pathological __str__ never crashes sanitization
        return None
    text = _strip_controls(text)
    # Collapse every separator run (whitespace + ``_``/``-``) to one space BEFORE casefolding, so a
    # trailing/leading separator is trimmed and internal ones become a single space.
    text = _SEPARATOR_RUN.sub(" ", text).strip()
    if not text:
        return None
    # Casefold LAST — once separators are spaces — for a single case-insensitive normal form.
    text = text.casefold()
    if len(text) > MAX_LABEL_LEN:
        text = text[:MAX_LABEL_LEN].strip()
    return text or None


def sanitize_domain(raw: object) -> str:
    """Sanitize the operator's free-text ``domain`` hint into one bounded, fence-safe line.

    The hint is config-supplied prose that the Provider folds into every induction prompt
    (``DOMAIN_CONTEXT_TEMPLATE``), so it gets the same "untrusted text becomes prompt data" bounding
    the induced labels get:

      * coerce to ``str``; a non-coercible value yields ``""`` (the no-hint path);
      * strip control / zero-width / bidi-override characters (``Cc``/``Cf``) — the anti-injection,
        anti-hidden-payload rule;
      * drop ``<`` and ``>`` so the hint can NEVER close or forge the ``<domain_context>`` fence it
        is placed inside. A domain description is prose; angle brackets carry no meaning it needs,
        and removing them makes fence-escape structurally impossible rather than merely discouraged;
      * collapse every whitespace run to a single space, so the hint stays one line (it cannot
        fabricate a new "message" with blank lines) and is stable in a comment;
      * cap the length at :data:`MAX_DOMAIN_LEN` — the anti-bloat bound.

    Pure + deterministic, and never raises: a wholly-unusable hint sanitizes to ``""``, which the
    Provider treats exactly as "no hint given" (the byte-identical backward-compatible path).
    """
    try:
        text = raw if isinstance(raw, str) else str(raw)
    except Exception:  # noqa: BLE001 - a pathological __str__ never crashes sanitization
        return ""
    text = _strip_controls(text)
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_DOMAIN_LEN:
        text = text[:MAX_DOMAIN_LEN].strip()
    return text


def sanitize_labels(raw_labels: object) -> SanitizeResult:
    """Sanitize a whole candidate list into a bounded, deduped, order-stable label set (W4 §5).

    A non-list ``raw_labels`` (the model returned a string / dict / null for a slot) yields an empty
    result — the schema-validation safety net rejects that shape upstream, but sanitizing defends in
    depth. Each item is cleaned by :func:`sanitize_label` into its canonical normal form; usable
    items are de-duplicated on that form (so ``Person`` / ``person`` / ``PERSON`` collapse to one
    ``person``) preserving first-seen order, and the count is capped at :data:`MAX_LABELS`, with the
    overflow reported in :attr:`SanitizeResult.dropped_for_count`.
    """
    if not isinstance(raw_labels, list):
        return SanitizeResult(labels=[], dropped_for_count=0)
    seen: set[str] = set()
    kept: list[str] = []
    dropped_for_count = 0
    for raw in raw_labels:
        cleaned = sanitize_label(raw)
        if cleaned is None:
            continue
        # ``cleaned`` is already the canonical normal form, so it IS the dedup key.
        if cleaned in seen:
            continue
        if len(kept) >= MAX_LABELS:
            # Anti-bloat: cap the count, but count the drop (no silent truncation) so the Provider
            # can log + surface how many labels the cap removed.
            dropped_for_count += 1
            continue
        seen.add(cleaned)
        kept.append(cleaned)
    return SanitizeResult(labels=kept, dropped_for_count=dropped_for_count)

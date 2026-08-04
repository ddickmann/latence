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
  * cap the COUNT per type at the caller-supplied ``max_labels`` (default :data:`MAX_LABELS`, 40) —
    the anti-bloat bound; the overflow is reported (not silently truncated) so the drop is
    auditable;
  * OPTIONALLY (``reject_value_shapes=True``) reject **value-shaped** label candidates — the S3
    value-shape guard (see below).

**The value-shape guard (S3).** The live S3 ohr_bench campaign showed the per-chunk LLM induction
letting VALUE-shaped strings leak in as entity/relation TYPE names — ``"1q22"``, ``"10.3"``,
``".net"``, ``"1024 fulton street"``, ``"15 to 24 years"``, ``"october 27, 1986"``: each an
*instance*, not a *type*, each stamped once, and together bloating the type vocabulary at corpus
scale. :func:`is_value_shaped` is the pure, deterministic, Unicode-safe shape gate that rejects a
canonical-normal-form label when it is:

  1. purely numeric / decimal / percentage (``"10.3"``, ``"42"``, ``"15%"``);
  2. digit-dominated — digit ratio > 0.4 of its non-space characters — or a quarter/fiscal shape
     (``"1q22"``, ``"q3 2024"``, ``"fy2023"``);
  3. date-shaped — a month name alongside a digit (``"october 27, 1986"``), an ISO-style date, or a
     ``"15 to 24 years"``-style numeric range;
  4. address-shaped — a digit AND a street/location suffix token (``"1024 fulton street"``,
     ``"10 chestnut st."``);
  5. a file/tech-artifact fragment — a leading ``.`` (``".net"``) or a single character — or a
     measurement-prefixed spec (``"1.2 mb floppy disc"``, ``"20 mb hard disc"``: a number + unit
     head names one artifact, not a type). Labels merely *containing* dots inside stay.

Borderline rulings are decided by the digit-ratio rule and pinned in tests: ``"2008 farm bill"``
(ratio 4/12 ≈ 0.33) is KEPT; ``"1933 act"`` (ratio 4/7 ≈ 0.57) is REJECTED. The guard is OPT-IN
(default off) because these functions also canonicalize hand-authored config label sets (the
lexicon inducer's candidate space, the label-profiler's fixed-point contract) — an operator's
deliberate type name is trusted; only INDUCED (untrusted-LLM) labels pass through the guard, on the
Provider's sanitize path. Rejections ride the same auditable drop accounting as the count cap
(:attr:`SanitizeResult.dropped_for_shape` — never a silent drop path).

**The count cap is a caller decision, because it means two different things** (the extraction
redesign, T1). At *corpus* granularity the cap is a **salience cap**: the merged corpus union is
ranked by document frequency and truncated to the most-agreed-on :data:`MAX_LABELS` types. At
*chunk* granularity there is no union and no salience question — a chunk's own induced set is
naturally small — so capping at 40 would be a silent recall loss on a chunk that legitimately
evidences more. The chunk path therefore passes :data:`CHUNK_LABEL_CEILING` (64), a **safety
ceiling** against a degenerate reply, not a salience cap, and :data:`MAX_LABEL_CEILING` (256) is the
absolute anti-bloat bound no configuration may exceed.

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
#: The DEFAULT per-type label-count cap (anti-bloat): labels beyond this are dropped and reported.
#: This is the CORPUS-salience cap — the bound on the document-frequency-ranked merged schema.
MAX_LABELS = 40
#: The default per-type count cap on the PER-CHUNK induction path (T1). A chunk's own induced set
#: is naturally small, so this is a **safety ceiling** against a degenerate/hostile reply, NOT a
#: salience cap: a chunk that legitimately evidences 50 types keeps all 50. Deliberately generous.
CHUNK_LABEL_CEILING = 64
#: The absolute per-type count ceiling no configuration may exceed (the hard anti-bloat bound). A
#: configured ``label_ceiling`` above this is clamped down to it — an operator cannot turn the
#: safety ceiling off, only raise it within a bound that keeps the extractor prompt finite.
MAX_LABEL_CEILING = 256
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

# ---------------------------------------------------------------------------
# The value-shape guard (S3) — shape patterns over the CANONICAL normal form.
# Every pattern below sees a label AFTER sanitize_label ran: casefolded, separator runs
# (whitespace/``_``/``-``) collapsed to single spaces. Dots, commas and ``%`` survive
# canonicalization, so they are matched literally. All pure, compiled once, deterministic.
# ---------------------------------------------------------------------------

#: Rule 2's digit-dominance threshold: a label whose non-space characters are more than this
#: fraction digits is an instance (a value), not a type. 0.4 keeps "2008 farm bill" (0.33) and
#: rejects "1933 act" (0.57) — the two pinned borderline rulings.
DIGIT_RATIO_THRESHOLD = 0.4

# Rule 1: purely numeric / decimal / thousands-separated / percentage ("10.3", "42", "1,000",
# "15%", "15 %" — the canonical form can keep one space before the ``%``).
_PURE_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)*\s?%?$")
# Rule 2: quarter / fiscal-period shapes ("q3", "q3 2024", "1q22", "fy2023", "fy 2023").
_FISCAL_RE = re.compile(r"^(?:q\s?[1-4](?:\s\d{2,4})?|[1-4]\s?q\s?\d{2,4}|fy\s?\d{2,4})$")
# Rule 3a: ISO-style dates. Canonicalization turns "2024-01-15" into "2024 01 15", so the
# separator class covers the space it becomes as well as "." and "/" (which survive).
_ISO_DATE_RE = re.compile(r"^\d{4}[\s./]\d{1,2}[\s./]\d{1,2}$")
# Rule 3b: "15 to 24 years"-style numeric ranges — two numbers joined by a range word, with up to
# two trailing tokens for the unit ("years", "per cent", ...).
_NUMERIC_RANGE_RE = re.compile(r"^\d[\d.,]*\s(?:to|through|until)\s\d[\d.,]*(?:\s\S+){0,2}$")
# Rule 3c: English month tokens (full + common abbreviations). A month token is only date evidence
# TOGETHER with a digit ("october 27, 1986"), so a bare "may" or "march" as part of a real type
# name is never rejected by this rule alone.
_MONTH_TOKENS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
)
# Rule 4: street/location suffix tokens. Address evidence only TOGETHER with a digit, so
# "main street" (a legitimate place-type-ish label) is not rejected while "1024 fulton street" is.
_ADDRESS_SUFFIX_TOKENS = frozenset(
    {
        "st", "street", "ave", "avenue", "road", "rd", "blvd", "drive", "lane",
        "suite", "floor",
    }
)
# Rule 5b: a number + measurement-unit HEAD names one artifact/spec, not a type ("1.2 mb floppy
# disc", "20 mb hard disc"). The unit list is deliberately tight (storage / frequency / physical
# units) so "2008 farm bill" ("farm" is no unit) and "3 body problem" stay.
_MEASUREMENT_HEAD_RE = re.compile(
    r"^\d+(?:[.,]\d+)?\s"
    r"(?:kb|mb|gb|tb|pb|kib|mib|gib|tib|bit|bits|byte|bytes"
    r"|hz|khz|mhz|ghz"
    r"|mm|cm|km|kg|mg|ml|kwh|mph|kmh)"
    r"(?:\s|$)"
)
# Token-edge punctuation stripped before token-set membership checks ("st." → "st", "27," → "27").
_TOKEN_EDGE_PUNCT = ".,;:"


def _digit_ratio(label: str) -> float:
    """The fraction of ``label``'s NON-SPACE characters that are digits (0.0 for an empty label).

    Non-space, so the separator spaces canonicalization introduced do not dilute the ratio
    ("1933 act" measures 4/7, not 4/8 — the ruling must not depend on how many separators the model
    happened to emit). ``str.isdigit`` covers Unicode digits, keeping the rule Unicode-safe.
    """
    chars = [ch for ch in label if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isdigit() for ch in chars) / len(chars)


def _tokens(label: str) -> list[str]:
    """The label's space-split tokens, edge punctuation stripped ("st." → "st", "27," → "27")."""
    return [token.strip(_TOKEN_EDGE_PUNCT) for token in label.split(" ")]


def is_value_shaped(label: str) -> bool:
    """True iff ``label`` (canonical normal form) is a VALUE shape, not a type name (S3 guard).

    The deterministic shape gate for induced labels — see the module docstring for the rule list
    and its S3 ohr_bench provenance. Expects the canonical normal form :func:`sanitize_label`
    emits (casefolded, separator runs collapsed to single spaces); pure and Unicode-safe (digit
    checks use Unicode digit semantics; the month/address/unit token lists are English, matched
    whole-token only so they never fire inside a longer word).
    """
    # Rule 5: file/tech-artifact fragments — a leading dot (".net") or a single character. A dot
    # INSIDE a longer label ("asp.net framework", "u.s. government") is untouched.
    if len(label) == 1 or label.startswith("."):
        return True
    # Rule 1: purely numeric / decimal / percentage.
    if _PURE_NUMBER_RE.match(label):
        return True
    # Rule 2: digit-dominated, or a quarter/fiscal period.
    if _digit_ratio(label) > DIGIT_RATIO_THRESHOLD or _FISCAL_RE.match(label):
        return True
    # Rule 3: date shapes — ISO-style, numeric range, or month token + digit.
    if _ISO_DATE_RE.match(label) or _NUMERIC_RANGE_RE.match(label):
        return True
    has_digit = any(ch.isdigit() for ch in label)
    tokens = _tokens(label)
    if has_digit and any(token in _MONTH_TOKENS for token in tokens):
        return True
    # Rule 4: address shapes — a digit AND a street/location suffix token.
    if has_digit and any(token in _ADDRESS_SUFFIX_TOKENS for token in tokens):
        return True
    # Rule 5b: a number + measurement-unit head ("1.2 mb floppy disc").
    return _MEASUREMENT_HEAD_RE.match(label) is not None


@dataclass(frozen=True)
class SanitizeResult:
    """The sanitized labels plus the auditable drop tallies (W4 §5 — no silent drop path).

    ``labels`` is the bounded, deduped, order-stable list an extractor can safely prompt with.
    ``dropped_for_count`` is how many otherwise-valid labels the :data:`MAX_LABELS` cap removed — a
    non-zero value is the "no silent truncation" audit signal the Provider logs + surfaces.
    ``dropped_for_shape`` is how many DISTINCT candidates the S3 value-shape guard rejected
    (``reject_value_shapes=True`` only; always 0 with the guard off) — the same auditable-drop
    accounting, so a junk-type rejection is visible in the Provider's log, never a silent path.
    """

    labels: list[str]
    dropped_for_count: int = 0
    dropped_for_shape: int = 0


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


def sanitize_labels(
    raw_labels: object, *, max_labels: int = MAX_LABELS, reject_value_shapes: bool = False
) -> SanitizeResult:
    """Sanitize a whole candidate list into a bounded, deduped, order-stable label set (W4 §5).

    A non-list ``raw_labels`` (the model returned a string / dict / null for a slot) yields an empty
    result — the schema-validation safety net rejects that shape upstream, but sanitizing defends in
    depth. Each item is cleaned by :func:`sanitize_label` into its canonical normal form; usable
    items are de-duplicated on that form (so ``Person`` / ``person`` / ``PERSON`` collapse to one
    ``person``) preserving first-seen order, and the count is capped at ``max_labels``, with the
    overflow reported in :attr:`SanitizeResult.dropped_for_count`.

    ``max_labels`` defaults to :data:`MAX_LABELS` (the corpus-salience cap, the pre-T1 behaviour);
    the per-chunk induction path passes :data:`CHUNK_LABEL_CEILING` instead, because on that path
    the count bound is a safety ceiling and not a salience decision (see the module docstring).

    ``reject_value_shapes=True`` additionally runs the S3 value-shape guard
    (:func:`is_value_shaped`) over each canonical candidate BEFORE it can occupy a ceiling slot:
    a value-shaped candidate ("10.3", "1q22", "october 27, 1986") is rejected and tallied on
    :attr:`SanitizeResult.dropped_for_shape` — the same auditable drop accounting the count cap
    uses, never a silent path. Each DISTINCT junk shape counts once (duplicates dedup first).
    OFF by default: hand-authored config label sets (the lexicon candidate space) are trusted and
    must NOT pass through the guard — only the Provider's induced-label path opts in.
    """
    if not isinstance(raw_labels, list):
        return SanitizeResult(labels=[], dropped_for_count=0)
    ceiling = max(1, min(int(max_labels), MAX_LABEL_CEILING))
    seen: set[str] = set()
    kept: list[str] = []
    dropped_for_count = 0
    dropped_for_shape = 0
    for raw in raw_labels:
        cleaned = sanitize_label(raw)
        if cleaned is None:
            continue
        # ``cleaned`` is already the canonical normal form, so it IS the dedup key.
        if cleaned in seen:
            continue
        if reject_value_shapes and is_value_shaped(cleaned):
            # The S3 value-shape guard: an instance-shaped candidate never becomes a type. Marking
            # it seen dedups repeats, so the tally counts DISTINCT junk shapes — and the reject
            # happens BEFORE the ceiling check, so junk never consumes a ceiling slot.
            seen.add(cleaned)
            dropped_for_shape += 1
            continue
        if len(kept) >= ceiling:
            # Anti-bloat: cap the count, but count the drop (no silent truncation) so the Provider
            # can log + surface how many labels the cap removed.
            dropped_for_count += 1
            continue
        seen.add(cleaned)
        kept.append(cleaned)
    return SanitizeResult(
        labels=kept, dropped_for_count=dropped_for_count, dropped_for_shape=dropped_for_shape
    )

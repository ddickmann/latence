"""Shared, reusable Redaction machinery (policy, masking, placeholders) — S6.

The parts of Redaction that every :class:`~latence_core.capability.PIIDetector`
Provider needs *regardless of how it detects PII* live here, in the thin core, so a
rule Provider (``redaction.hybrid_rule``) and a learned one (``redaction.gliner_pii``)
share one implementation and cannot drift:

* :func:`parse_action` — validate a config action string into a
  :class:`~latence_core.contracts.RedactionAction`.
* :class:`RedactionPolicy` — compile + resolve the sensitivity-keyed policy map (the
  metadata-aware core of S6: which action / which types / skip, per
  ``Classification.sensitivity``).
* :func:`placeholder_for` — the deterministic surrogate a span's action writes into
  the masked variant (mask token / salted pseudonym / salted hash / empty for tag),
  derived from the value but **never storing it**.
* :func:`apply_masking` — rewrite text replacing non-tag spans right-to-left.

A detector scans a parsed :class:`~latence_core.contracts.DocumentRecord`'s assembled
markdown directly (``record.content`` + ``record.page_map``) — the original-document
coordinate system every PIISpan offset and the page resolution use. Redaction does NOT
reconstruct text from the markup-stripped chunk stream: chunk content is de-marked-up
with offsets remapped to the original, so ``len(content) != char_end - char_start`` and
reassembling it would fabricate a broken coordinate system.

Keeping only *detection* in each Provider is the seam: the framework owns policy and
masking; a Provider owns "where is the PII".
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from latence_core.contracts import PIISpan, RedactionAction

if TYPE_CHECKING:
    from latence_core.pagemap import PageOffsetIndex

# The mask token per PII type used by a MASK action: ``[EMAIL]``, ``[PERSON]``, …
_MASK_TEMPLATE = "[{type}]"


def parse_action(value: Any) -> RedactionAction:
    """Validate a config action string into a :class:`RedactionAction` (clear error)."""
    try:
        return RedactionAction(str(value))
    except ValueError as exc:
        valid = ", ".join(a.value for a in RedactionAction)
        msg = f"Unknown redaction action {value!r}. Valid: {valid}."
        raise ValueError(msg) from exc


class RedactionPolicy:
    """The sensitivity-keyed redaction policy — the metadata-aware core of S6.

    Compiled once from a Pipeline's ``policies`` config; :meth:`resolve` maps a
    document's ``Classification.sensitivity`` to the action to take, an optional
    PII-type allow-list, whether to skip the document entirely, and the policy name
    recorded on the RedactionRecord. Looks up the exact sensitivity, then a
    ``default`` fallback, then the Provider's ``default_action`` over all types.
    """

    def __init__(self, policies: dict[str, Any] | None, default_action: RedactionAction) -> None:
        self._default_action = default_action
        self._policies = self._compile(policies or {})

    @staticmethod
    def _compile(policies: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not isinstance(policies, dict):
            msg = "'policies' config must be a mapping of sensitivity -> policy."
            raise TypeError(msg)
        compiled: dict[str, dict[str, Any]] = {}
        for sensitivity, spec in policies.items():
            if not isinstance(spec, dict):
                msg = f"Policy for sensitivity {sensitivity!r} must be a mapping."
                raise TypeError(msg)
            entry: dict[str, Any] = {}
            if "action" in spec:
                entry["action"] = parse_action(spec["action"])
            if "types" in spec:
                types = spec["types"]
                if not isinstance(types, list):
                    msg = f"Policy 'types' for {sensitivity!r} must be a list of PII types."
                    raise TypeError(msg)
                entry["types"] = {str(t) for t in types}
            entry["skip"] = bool(spec.get("skip", False))
            compiled[str(sensitivity)] = entry
        return compiled

    def resolve(self, sensitivity: str) -> tuple[RedactionAction, set[str] | None, bool, str]:
        """Return ``(action, allowed_types, skip, policy_name)`` for a sensitivity value."""
        entry = self._policies.get(sensitivity)
        name = sensitivity
        if entry is None:
            entry = self._policies.get("default")
            name = "default"
        if entry is None:
            return self._default_action, None, False, "default"
        action = entry.get("action", self._default_action)
        allowed = entry.get("types")
        skip = bool(entry.get("skip", False))
        return action, allowed, skip, name


def placeholder_for(
    pii_type: str, value: str, action: RedactionAction, salt: str
) -> str:
    """The deterministic replacement text for a span — never stores the raw value.

    ``MASK`` → ``[TYPE]``; ``REPLACE`` → ``TYPE_<8hex>`` (a stable pseudonym so equal
    values read consistently); ``HASH`` → ``<TYPE:12hex>`` (a stable salted digest,
    joinable across the corpus without exposing the value); ``TAG`` → empty (surface
    text left intact, detect-only). The value is mixed into the salted digest but is
    not returned in the clear and is never persisted on the PIISpan.
    """
    if action is RedactionAction.TAG:
        return ""
    if action is RedactionAction.MASK:
        return _MASK_TEMPLATE.format(type=pii_type.upper())
    digest = hashlib.sha256(f"{salt}:{pii_type}:{value}".encode()).hexdigest()
    if action is RedactionAction.REPLACE:
        return f"{pii_type.upper()}_{digest[:8]}"
    return f"<{pii_type.upper()}:{digest[:12]}>"


def apply_masking(text: str, spans: list[PIISpan]) -> str:
    """Rewrite ``text`` replacing each non-TAG span with its placeholder.

    Applied right-to-left so replacing a span never shifts the offsets of an earlier
    one (the spans are non-overlapping and ordered by the caller).
    """
    result = text
    for span in reversed(spans):
        if span.action is RedactionAction.TAG:
            continue
        result = result[: span.char_start] + span.placeholder + result[span.char_end :]
    return result


def build_span(
    pii_type: str,
    detector: str,
    start: int,
    end: int,
    value: str,
    action: RedactionAction,
    salt: str,
    page_index: PageOffsetIndex | None,
    confidence: float,
) -> PIISpan:
    """Assemble one auditable :class:`PIISpan` (placeholder derived, raw value dropped).

    The single place a detected match becomes a contract span: it derives the
    deterministic placeholder, resolves the source page(s) from the parent page map
    (if any), and records type/detector/offset/action/confidence — never the raw
    surface value.
    """
    placeholder = placeholder_for(pii_type, value, action, salt)
    page_start: int | None = None
    page_end: int | None = None
    if page_index is not None:
        page_start = page_index.page_for_offset(start)
        page_end = page_index.page_for_offset(max(end - 1, start))
    return PIISpan(
        pii_type=pii_type,
        detector=detector,
        char_start=start,
        char_end=end,
        action=action,
        placeholder=placeholder,
        page_start=page_start,
        page_end=page_end,
        confidence=confidence,
    )

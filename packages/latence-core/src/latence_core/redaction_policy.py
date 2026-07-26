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

Since W16 (ADR-0042) Redaction is a **chunk→chunk transform**: a detector scans each
:class:`~latence_core.contracts.ChunkRecord`'s ``content`` (each chunk ≤ the model window, so no
PII is truncated) and the shared :func:`plan_chunk_redaction` / :func:`finalize_redacted_chunk`
tail produces the redacted chunk (``masked_content`` + chunk-local ``pii_spans``, ``content``
unchanged). A span's chunk-local offset resolves to its ORIGINAL source page through the chunk's
``offset_map`` + ``page_map`` (the ADR-0031 round trip). The deprecated document-level helpers
(:func:`build_span`, the :class:`RedactionPolicy` resolve, :func:`apply_masking`) remain for the
superseded ``RedactionRecord`` path.

Keeping only *detection* in each Provider is the seam: the framework owns policy and
masking; a Provider owns "where is the PII".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from latence_core.contracts import PIISpan, RedactionAction

if TYPE_CHECKING:
    from latence_core.contracts import ChunkRecord
    from latence_core.offsetmap import OffsetIndex
    from latence_core.page_index import PageIndexResolver
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


# -- W16 chunk-level redaction (ADR-0042) ------------------------------------
#
# Redaction is a **chunk→chunk transform** (the fix for the document-level 768-token truncation
# leak): each :class:`~latence_core.contracts.ChunkRecord`'s ``content`` (≤ the model window) is
# PII-scanned and masked, and the chunk is re-emitted with ``masked_content`` + ``pii_spans``
# populated (``content`` stays UNMASKED). The parts every Provider shares — policy resolution off
# the chunk's Classification, the H-C1 §4 no-op floor, chunk-local→original page resolution, span
# building + masking, and the ``model_copy`` that produces the redacted chunk — live here so a rule
# Provider and a learned one cannot drift. A Provider owns ONLY detection: it hands this helper the
# per-chunk candidate matches (chunk-content offsets), and gets back the finished ChunkRecord.


@dataclass(frozen=True)
class ChunkRedactionPlan:
    """The per-chunk, model-independent redaction context resolved before any detection.

    Policy action/allow-list, the skip/near-empty decision, the §4 no-op-floor disposition, and the
    chunk-local→original page/offset resolvers are all pure functions of the chunk + the compiled
    policy, so a Provider resolves them once (e.g. before a batched model call) via
    :func:`plan_chunk_redaction`, then feeds detection candidates to
    :func:`finalize_redacted_chunk`.
    """

    action: RedactionAction
    allowed: set[str] | None
    policy_name: str
    needs_detection: bool
    disabled_for_sensitive: bool
    chunk: ChunkRecord
    resolver: PageIndexResolver
    offset_index: OffsetIndex | None
    base_char_start: int


def plan_chunk_redaction(
    chunk: ChunkRecord,
    policy: RedactionPolicy,
    *,
    detector_types: set[str],
    sensitive_set: set[str],
    min_chars: int,
    resolver: PageIndexResolver | None = None,
) -> ChunkRedactionPlan:
    """Resolve everything a chunk needs BEFORE detection (policy, no-op floor, page/offset maps).

    ``detector_types`` is the set of ``pii_type`` values SOME enabled detector can produce (the
    coverage the applied policy narrows) — used by the §4 no-op floor to tell "control ran no
    detectors" (empty intersection with the policy allow-list) apart from "genuinely no PII found".
    A chunk classified in ``sensitive_set`` whose control runs no detector carries
    ``disabled_for_sensitive=True`` (the LOUD H-C1 §4 signal). ``offset_index`` is built from the
    chunk's offset_map (chunk-scoped, so genuinely per chunk); page resolution goes through
    ``resolver``, the DOCUMENT-scoped :class:`~latence_core.page_index.PageIndexResolver`, so a
    chunk-content span resolves to its ORIGINAL source page.

    **Pass a resolver that outlives the chunk loop.** A Provider redacting a stream of chunks
    should build ONE :class:`~latence_core.page_index.PageIndexResolver` per ``redact`` call and
    hand it to every ``plan_chunk_redaction``: since v18 a document's page map rides its FIRST
    chunk only, so a per-chunk resolver can only resolve pages for that first chunk. Omitting it
    creates a throwaway resolver — correct, and byte-identical on a pre-v18 stream where every
    chunk carried the map, but it forfeits page resolution on later chunks of a v18 stream.
    """
    from latence_core.offsetmap import OffsetIndex
    from latence_core.page_index import PageIndexResolver

    sensitivity = chunk.classification.sensitivity
    action, allowed, skip, policy_name = policy.resolve(sensitivity)
    needs_detection = not skip and len(chunk.content.strip()) >= min_chars
    would_run_detectors = (not skip) and bool(
        detector_types if allowed is None else detector_types & allowed
    )
    disabled_for_sensitive = not would_run_detectors and sensitivity in sensitive_set
    offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
    base_char_start = chunk.provenance.char_start or 0
    return ChunkRedactionPlan(
        action=action,
        allowed=allowed,
        policy_name=policy_name,
        needs_detection=needs_detection,
        disabled_for_sensitive=disabled_for_sensitive,
        chunk=chunk,
        resolver=resolver if resolver is not None else PageIndexResolver(),
        offset_index=offset_index,
        base_char_start=base_char_start,
    )


def build_chunk_span(
    pii_type: str,
    detector: str,
    local_start: int,
    local_end: int,
    value: str,
    action: RedactionAction,
    salt: str,
    plan: ChunkRedactionPlan,
    confidence: float,
) -> PIISpan:
    """One chunk-carried :class:`PIISpan`: chunk-LOCAL offsets, ORIGINAL-resolved source page.

    ``local_start``/``local_end`` are offsets into the chunk's ``content`` (the coordinate system
    :func:`apply_masking` masks in, so they are stored verbatim on the span). The source page(s) are
    resolved by mapping the local offset back to its ORIGINAL-markdown offset through the chunk's
    :class:`~latence_core.offsetmap.OffsetIndex` (identity ``base_char_start + local`` when the
    chunk carries no offset_map) and then to a page through the plan's document-scoped
    :class:`~latence_core.page_index.PageIndexResolver`. Both pages are ``None`` when no page map
    is available for the document — the same degradation the doc-level path used, now counted on
    the resolver rather than silently branched on here. The raw ``value`` derives the
    deterministic placeholder and is never stored (S6 AC).
    """
    placeholder = placeholder_for(pii_type, value, action, salt)
    if plan.offset_index is not None:
        orig_start = plan.offset_index.resolve(local_start)
        orig_end = plan.offset_index.resolve(max(local_end - 1, local_start))
    else:
        orig_start = plan.base_char_start + local_start
        orig_end = plan.base_char_start + max(local_end - 1, local_start)
    page_start = plan.resolver.page_for(plan.chunk, orig_start)
    page_end = plan.resolver.page_for(plan.chunk, orig_end)
    return PIISpan(
        pii_type=pii_type,
        detector=detector,
        char_start=local_start,
        char_end=local_end,
        action=action,
        placeholder=placeholder,
        page_start=page_start,
        page_end=page_end,
        confidence=confidence,
    )


# -- W18 universal financial-PII redaction floor (ADR-0044) -------------------
#
# A small, high-harm set of FINANCIAL / national-identity PII types that MUST be handled no matter
# which detection backend a stack configures. These deterministic, zero-dep recognisers run at the
# shared :func:`finalize_redacted_chunk` choke point — UNIONed with whatever the Provider detected —
# so a learned Redactor whose model MISSES (or is misconfigured away from) a financial identifier
# still cannot leak it into the exported corpus. Before W18, only the in-core
# ``redaction.hybrid_rule`` recognised these; the three learned Providers
# (``gliner_pii``/``gliner2``/``presidio``) inherited only their model's coverage, so a model that
# missed an IBAN/SSN/card leaked it. Routing the floor through the single seam every Provider
# already calls makes it NON-BYPASSABLE by model/backend choice, impossible to forget per-Provider.
#
# Scope of the guarantee (deliberately narrow + honest, H-C1): the floor guarantees DETECTION of the
# financial subset regardless of backend; it does NOT override an operator's explicit policy
# `action` or `skip`. A floor span is handled with the chunk's resolved policy action
# (mask/replace/hash/tag), and the floor runs ONLY when redaction actually runs on the chunk
# (``plan.needs_detection`` — an explicit ``skip:true`` or a near-empty chunk still skips). This
# mirrors the H-C1 §4 sensitive-doc floor philosophy: the disposition is guaranteed-visible and
# operator intent is preserved.
#
# The patterns are byte-identical to the ``redaction.hybrid_rule`` built-ins for these types (a
# drift-guard test pins them equal), so the floor never disagrees with the in-core rule Redactor and
# a hybrid_rule stack is byte-identical (every floor span exactly overlaps a Provider span and the
# Provider wins the tie — see :func:`_merge_floor`). ``ssn_nodash`` reports as ``ssn`` so the
# per-type count stays honest (the same remap ``redaction.hybrid_rule`` uses).
FINANCIAL_FLOOR_PATTERNS: dict[str, tuple[str, str]] = {
    # detector_name: (reported pii_type, regex) — byte-identical to the hybrid_rule built-ins.
    "ssn": ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    "ssn_nodash": ("ssn", r"(?<!\d)(?!000|666|9\d\d)\d{3} ?(?!00)\d{2} ?(?!0000)\d{4}(?!\d)"),
    "credit_card": ("credit_card", r"\b\d(?:[ -]?\d){12,15}\b"),
    "iban": ("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
}

# The reported pii_type set the floor GUARANTEES is handled (``ssn``/``credit_card``/``iban``).
FINANCIAL_FLOOR_TYPES: frozenset[str] = frozenset(
    pii_type for pii_type, _ in FINANCIAL_FLOOR_PATTERNS.values()
)

# Compiled once. IGNORECASE mirrors the hybrid_rule default (``case_sensitive: False``) so the
# floor's IBAN recogniser matches a lower-cased prefix exactly as the in-core rule Redactor does.
_FLOOR_DETECTORS: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (pii_type, f"floor:{name}", re.compile(regex, re.IGNORECASE))
    for name, (pii_type, regex) in FINANCIAL_FLOOR_PATTERNS.items()
)
# Rule-certain: the floor is a deterministic recogniser, not a probabilistic model.
_FLOOR_CONFIDENCE = 1.0


def floor_candidates(text: str) -> list[tuple[int, int, str, str, str, float]]:
    """Every financial-floor match in ``text`` as ``(start, end, pii_type, value, detector, conf)``.

    The always-on half of the universal floor: it scans a chunk's ``content`` for the fixed
    FINANCIAL_PII set independently of any Provider or policy allow-list. Offsets are
    chunk-``content`` coordinates (the system :func:`apply_masking` masks in). Empty when the chunk
    carries no financial PII (the common case), so a chunk with no floor hit is byte-identical to
    the pre-W18 path.
    """
    out: list[tuple[int, int, str, str, str, float]] = []
    for pii_type, detector, pattern in _FLOOR_DETECTORS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if end > start:
                out.append((start, end, pii_type, match.group(0), detector, _FLOOR_CONFIDENCE))
    return out


# -- T2 universal GUARDRAIL safety floor -------------------------------------
#
# The learned-guardrail twin of the ADR-0044 financial-PII floor above, and it exists for exactly
# the same reason. The blessed Redaction checkpoint (``fastino/GLiNER2-Guardrails-PII-Multi``) is a
# GUARDRAILS+PII model, so a guardrails-capable Redaction Provider prompts it with guard categories
# as well as PII types. Those guard categories follow the per-chunk rule like every other LEARNED
# label set — a chunk is screened for the categories induced FOR IT — EXCEPT for this small set,
# which is unioned onto every chunk and can never be subsetted: a prompt injection or harmful
# passage hidden in a chunk whose induced guard categories happen to omit it is precisely the case
# a per-chunk-only screen would miss, and it is a security invariant, not a relevance judgement.
#
# Scope, stated honestly: this is a LABEL floor (which categories the model is always asked about),
# not a detection guarantee — the model may still miss a hit, exactly as the PII model may. The
# DETERMINISTIC keyword pre-filter (``screening.content_keyword``) is a separate, earlier Stage and
# is untouched by this: it carries no learned label count, so the redesign exempts it by nature.
#
# The mapping is label → the canonical :class:`~latence_core.contracts.RiskMarker` ``category`` the
# hit is recorded under, deliberately the SAME vocabulary ``screening.content_keyword`` emits
# (``prompt_injection`` / ``harmful``) so a RAG consumer filters on ONE category set regardless of
# which control caught it. The label surface forms are natural language because GLiNER2-family
# zero-shot models score natural-language type names best.
GUARD_SAFETY_FLOOR: dict[str, str] = {
    "prompt injection": "prompt_injection",
    "harmful content": "harmful",
}


def guard_category(label: str) -> str:
    """The canonical :class:`RiskMarker` ``category`` for a guardrail ``label``.

    A floor label maps to the screening vocabulary (``prompt injection`` → ``prompt_injection``,
    ``harmful content`` → ``harmful``); any other (config- or induction-supplied) guard label
    normalizes deterministically to a snake_case category, so an operator-chosen or induced
    category is still a stable, filterable value rather than raw model text. Never empty for a
    non-blank label — the caller drops blank labels before it gets here.
    """
    normalized = " ".join(label.split()).casefold()
    return GUARD_SAFETY_FLOOR.get(normalized, normalized.replace(" ", "_"))


_T = TypeVar("_T")


def _resolve_clipping(intervals: list[tuple[int, int, _T]]) -> list[tuple[int, int, _T]]:
    """Greedy non-overlapping resolution that CLIPS an overlapping span to its uncovered tail.

    ``intervals`` MUST already be sorted by ``start`` ascending (callers append their tie-breakers
    after it). The classic greedy rule — accept iff ``start >= last_end``, otherwise DROP — silently
    UN-MASKS text: a span overlapping an accepted one by a single character is discarded wholesale,
    so the part of it nothing else covers is emitted verbatim (audit R6-invariants-0: 12 digits of
    an already-masked card re-exposed). Clipping instead of dropping makes the resolution
    COVERAGE-PRESERVING — every character covered by ANY input interval stays covered.

    Clipping to ``last_end`` (the max end so far) is sound precisely because the input is
    start-sorted: for any point ``p`` in ``[start, last_end)`` the accepted span that set
    ``last_end`` began at or before ``start``, hence already covers ``p``. Truncated-to-empty spans
    (fully covered, e.g. the floor duplicate of a coincident Provider span) drop out, so a
    disjoint or coincident input is byte-identical to the pre-fix greedy output.
    """
    result: list[tuple[int, int, _T]] = []
    last_end = -1
    for start, end, payload in intervals:
        clipped = max(start, last_end)
        if clipped < end:
            result.append((clipped, end, payload))
            last_end = end
    return result


def _subtract_covered(
    start: int, end: int, covered: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """``[start, end)`` minus an ordered, non-overlapping ``covered`` set → surviving fragments.

    Used to keep a non-mutating TAG span's audit record wherever a masking span did not already
    claim the region, without ever letting the TAG span overlap (and so shadow) a mask.
    """
    fragments: list[tuple[int, int]] = []
    cursor = start
    for c_start, c_end in covered:
        if c_end <= cursor:
            continue
        if c_start >= end:
            break
        if c_start > cursor:
            fragments.append((cursor, c_start))
        cursor = max(cursor, c_end)
        if cursor >= end:
            return fragments
    if cursor < end:
        fragments.append((cursor, end))
    return fragments


def _merge_floor(
    accepted: list[tuple[int, int, str, str, str, float]],
    floor: list[tuple[int, int, str, str, str, float]],
) -> list[tuple[int, int, str, str, str, float]]:
    """Union a Provider's accepted spans with floor candidates → ordered, non-overlapping.

    Deterministic resolution (matching the hybrid_rule overlap policy): a longer span wins first
    (max masking coverage), and on an identical span the PROVIDER wins the tie (source rank 0 < 1)
    so its richer detector/label provenance is kept — a floor span survives only where the Provider
    left a gap, which is exactly the guarantee. When the two sets fully coincide (e.g. the in-core
    ``redaction.hybrid_rule``, whose built-ins ARE these patterns) the Provider wins every span and
    the output is byte-identical.

    "Where the Provider left a gap" is enforced by CLIPPING, not by dropping (audit
    R6-invariants-0): a Provider span that overlaps a floor span by one character used to discard
    the floor span whole, re-exposing the rest of a card/IBAN/SSN the ADR-0044 floor had matched —
    on the plain SINGLE-Redactor path, no chaining required. The clipped remainder keeps the floor's
    ``value`` so its placeholder stays the deterministic one derived from the full match.
    """
    tagged: list[tuple[tuple[int, int, str, str, str, float], int]] = [
        *((span, 0) for span in accepted),
        *((span, 1) for span in floor),
    ]
    tagged.sort(key=lambda item: (item[0][0], -(item[0][1] - item[0][0]), item[1], item[0][2]))
    resolved = _resolve_clipping([(span[0], span[1], span) for span, _source in tagged])
    return [(start, end, span[2], span[3], span[4], span[5]) for start, end, span in resolved]


def _compose_prior_spans(spans: list[PIISpan], prior: list[PIISpan]) -> list[PIISpan]:
    """Union THIS Stage's spans with those an UPSTREAM Redaction Stage already applied.

    Redaction composes (audit R5-12). Both span sets are chunk-local offsets into the SAME clean
    ``content`` — every Redactor detects on ``content``, never on a predecessor's
    ``masked_content`` — so the two sets share one coordinate system and can be unioned directly.
    Without this, a second Redaction Stage rebuilds ``masked_content`` from the clean text and
    CLOBBERS every mask its predecessor applied: the upstream Stage still reports success, still
    contributes its counts, and none of its masks reach the exported corpus.

    Resolution mirrors :func:`_merge_floor` — longer span wins first (max masking coverage), and on
    an identical span THIS Stage wins the tie (source rank 0 < 1) so the Provider that just ran
    keeps its own provenance. A prior span survives wherever this Stage left a gap, which is the
    whole guarantee, and the resolution is COVERAGE-PRESERVING in both directions: composing can
    only ever ADD masked characters, never remove one (audit R6-invariants-0). Two rules make that
    true, and the pre-fix greedy union broke both:

    * **Masking spans outrank TAG.** ``TAG`` is the one action :func:`apply_masking` skips (the
      surface text stays readable), so a TAG span that merely STARTS EARLIER must never displace a
      ``MASK``/``REPLACE``/``HASH`` span — least of all the ADR-0044 financial floor's. Pre-fix an
      upstream TAG person span swallowed the downstream floor's ``credit_card`` span outright and
      the full card number was exported verbatim. TAG spans are therefore resolved SECOND and
      subtracted against the accepted masking coverage, keeping their audit record only where no
      mask claimed the region.
    * **Overlaps CLIP, never drop.** A learned Redactor's boundaries need not coincide with its
      predecessor's; discarding a partly-overlapping prior span whole re-exposed the uncovered tail
      of text the pipeline had already masked.

    ``prior`` is empty for every single-Redactor Pipeline (nothing upstream populated
    ``pii_spans``), so this is a no-op there and the output stays byte-identical.
    """
    if not prior:
        return spans
    tagged: list[tuple[PIISpan, int]] = [*((s, 0) for s in spans), *((s, 1) for s in prior)]
    tagged.sort(key=lambda item: (item[0].char_start, -item[0].length, item[1], item[0].pii_type))
    masking = [
        (s.char_start, s.char_end, s) for s, _r in tagged if s.action is not RedactionAction.TAG
    ]
    resolved = _resolve_clipping(masking)
    covered = [(start, end) for start, end, _s in resolved]
    out: list[tuple[int, int, PIISpan]] = list(resolved)
    for span, _rank in tagged:
        if span.action is not RedactionAction.TAG:
            continue
        out.extend(
            (start, end, span)
            for start, end in _subtract_covered(span.char_start, span.char_end, covered)
        )
    out.sort(key=lambda item: (item[0], item[1], item[2].pii_type))
    return [
        span
        if (start, end) == (span.char_start, span.char_end)
        else span.model_copy(update={"char_start": start, "char_end": end})
        for start, end, span in out
    ]


def finalize_redacted_chunk(
    chunk: ChunkRecord,
    plan: ChunkRedactionPlan,
    accepted: list[tuple[int, int, str, str, str, float]],
    salt: str,
) -> ChunkRecord:
    """Build the redacted chunk from an ORDERED, NON-OVERLAPPING candidate list (the shared tail).

    ``accepted`` is ``[(local_start, local_end, pii_type, value, detector, confidence), ...]`` in
    chunk-``content`` coordinates, already filtered to the policy allow-list and resolved to a clean
    non-overlapping set by the Provider's detection half. This shared tail derives each
    :class:`PIISpan` (chunk-local offset, original-resolved page, derived placeholder), masks the
    chunk text right-to-left, and returns ``chunk.model_copy`` with ``masked_content`` +
    ``pii_spans`` + ``redaction_disabled_for_sensitive`` populated — ``content`` UNCHANGED
    (extraction still sees full entities). A chunk with no accepted spans gets ``masked_content ==
    content`` and
    an empty span list (never an error), so every redacted chunk carries a PII-handled variant.

    W18 (ADR-0044): before building spans, the shared UNIVERSAL FINANCIAL-PII FLOOR is UNIONed in —
    a deterministic zero-dep recogniser for the high-harm financial subset (``ssn``/``credit_card``/
    ``iban``) that runs regardless of which detection backend produced ``accepted``. Because EVERY
    Provider funnels through this one seam, the floor is non-bypassable by model/backend choice: a
    learned Redactor that missed a card/IBAN/SSN still cannot leak it. The floor honours the chunk's
    resolved policy action and only runs when redaction runs on the chunk
    (``plan.needs_detection``), so an explicit ``skip`` / near-empty chunk is unchanged and a
    hybrid_rule stack (whose built-ins ARE these patterns) is byte-identical.
    """
    if plan.needs_detection:
        floor = floor_candidates(chunk.content)
        if floor:
            accepted = _merge_floor(accepted, floor)
    spans = [
        build_chunk_span(
            pii_type=pii_type,
            detector=detector,
            local_start=start,
            local_end=end,
            value=value,
            action=plan.action,
            salt=salt,
            plan=plan,
            confidence=confidence,
        )
        for start, end, pii_type, value, detector, confidence in accepted
    ]
    # Compose with an UPSTREAM Redaction Stage's spans (audit R5-12). Unconditional — a chunk this
    # Stage's policy SKIPS must still keep the masks a predecessor already applied, or a chained
    # Redactor un-redacts it. Empty (hence byte-identical) for every single-Redactor Pipeline.
    spans = _compose_prior_spans(spans, list(chunk.pii_spans))
    masked = apply_masking(chunk.content, spans) if spans else chunk.content
    return chunk.model_copy(
        update={
            "masked_content": masked,
            "pii_spans": spans,
            "redaction_disabled_for_sensitive": plan.disabled_for_sensitive,
        }
    )

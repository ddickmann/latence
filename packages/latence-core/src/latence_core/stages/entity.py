"""Built-in reference EntityExtractor Provider (deterministic, zero-shot, zero-dep).

The S4 Entity Extraction reference Provider (ADR-0004; ships in core like the S3
Chunker/Screeners so the demo runs end to end, ADR-0007 CPU-first, ADR-0016 no
extra deps). It is a **zero-shot** gazetteer/pattern extractor: the entity types
and how to recognise them are supplied entirely through Stage ``config`` — the
Capability names no labels and core hardcodes none. A Pipeline says "find these
labels with these seed terms / these regexes" and the Provider tags every
occurrence.

This is the CPU-viable, dependency-free, hash-stable default the Baseline bar
needs (no model download, so a seeded run is byte-identical). It is not a weaker
reinvention of a learned NER model — the real learned Provider is the separately
packaged ``latence-ner-gliner`` (a permissively-licensed GLiNER checkpoint via
plain ``transformers``, ADR-0016 heavy-dep isolation), a drop-in behind this same
seam. Shipping the deterministic reference in core keeps first contact
(``git clone`` → working extraction) free of torch.

Each emitted :class:`~latence_core.contracts.EntityMention` carries:

* its ``label`` (from the configured zero-shot label set) and a ``confidence``;
* Provenance ``char_start``/``char_end`` — the mention's half-open span **in the
  parent document's assembled markdown**, resolved from the chunk-local match
  position through the chunk's ``offset_map`` (the stripped-``content``→original
  offset map carried since v11) via :class:`~latence_core.offsetmap.OffsetIndex`, so
  the mention resolves back to its EXACT source offset even when markdown markup was
  stripped before it within the chunk. (Chunk ``content`` is markup-stripped, so the
  naive ``char_start + local`` shift is only a lower bound — ADR-0022 corrected,
  ADR-0031.) A chunk without an offset map (pre-v11) degrades to that shift;
* Provenance ``page_start``/``page_end`` — the source page(s) that span falls on,
  resolved from the mention's *own* document offset through the parent's page map
  (carried on the chunk) via :class:`~latence_core.pagemap.PageOffsetIndex` — the
  S4 round-trip (mention → offset → original page). A mention on a later page of a
  multi-page chunk resolves to that page, not the chunk's first page. When the
  chunk carries no page map, it degrades to the chunk's inherited page range;
* the parent chunk's Classification, inherited unchanged.

An empty or near-empty chunk yields no mentions (never an error) — the S4
acceptance criterion.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.contracts import EntityMention
from latence_core.errors import ConfigError
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, Provenance

# A single non-overlapping match found in a chunk's text, before it is turned into
# a contract: the label it matched, the surface string, the chunk-local half-open
# span, and the confidence assigned.
_Match = tuple[str, str, int, int, float]

# -- ReDoS guard for operator-supplied config patterns (issue #28) -----------
#
# S4 is the first Stage that compiles arbitrary operator-supplied config regex
# (``labels[...]["patterns"]``) and runs it over untrusted chunk.content. A
# pathological pattern (nested unbounded quantifiers like ``(\w+)+``, an outer
# ``{m,}`` on a quantified group like ``(a+){2,}``, an overlapping alternation under
# a quantifier like ``(a|a)+`` / ``(a|ab)+``, or a run of adjacent unbounded
# quantifiers with no group at all like ``a*a*a*…b``) plus a crafted document can
# drive Python's backtracking engine into exponential CPU — a DoS lever even though
# the dangerous pattern is operator-authored. We keep heavy deps out of core
# (ADR-0016), so instead of a re2 backend we statically reject the
# catastrophic-backtracking signature at compile time, before any untrusted content
# is matched. Literal ``terms`` are escaped and never reach this guard; only raw
# ``patterns`` do.

# The compiled config-pattern alternation is refused above this character length: a
# runaway pattern string is itself a red flag and the bound keeps compilation cheap.
_MAX_PATTERN_LENGTH = 1000

# A quantifier that can match unboundedly (``*``, ``+``, ``{m,}``) — the amplifier in
# every catastrophic-backtracking pattern. ``?`` and bounded ``{m,n}`` cannot blow up.
# Used to scan a group *body* for an inner unbounded quantifier (``(X+)+``); the outer
# quantifier is parsed exactly by ``_quantifier_at`` so ``{N,}`` is never truncated.
_UNBOUNDED_QUANTIFIER = re.compile(r"[*+]|\{\d*,\}")


def _top_level_alternatives(body: str) -> list[str]:
    """Split a group body on top-level ``|`` (ignoring ``|`` inside nested groups/classes)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_class = False
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(body[i : i + 2])
            i += 2
            continue
        if in_class:
            buf.append(ch)
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


# Metaclass first-tokens that match many chars. Two *different* metaclasses can be
# disjoint (``\w`` vs ``\s``), so this set is only used by the (liberal) alternation
# heuristic, never by the (strict) adjacency check.
_METACLASS_FIRSTS = {rf"\{c}" for c in "wWdDsS"}


def _first_token(atom: str) -> str | None:
    """The leading token of a sub-expression, for first-char overlap comparison.

    A literal char, an escaped class like ``\\w`` (returned as ``\\w``), a full ``[...]``
    class (returned verbatim, e.g. ``[a-z]``), or ``.``. ``None`` for an empty string.
    Returning the class text verbatim lets the adjacency check tell ``[a-z]`` from
    ``[0-9]`` (disjoint) instead of collapsing every class to one wildcard.
    """
    atom = atom.strip()
    if not atom:
        return None
    if atom[0] == "\\" and len(atom) >= 2:
        return atom[:2]
    if atom[0] == "[":
        end = atom.find("]", 1)
        return atom if end == -1 else atom[: end + 1]
    return atom[0]


def _tokens_definitely_overlap(a: str | None, b: str | None) -> bool:
    """True only when two first-tokens *provably* share an input character.

    Stricter than the alternation heuristic: it drives the group-free adjacency check,
    where over-approximating would falsely reject ubiquitous, safe operator patterns
    like ``\\w+\\s+\\w+`` (``\\w`` and ``\\s`` are DISJOINT) or ``[a-z]+[0-9]+`` (disjoint
    classes). So metaclasses and char classes overlap only their identical selves; only
    ``.`` (true wildcard) overlaps everything. Identical tokens overlap outright.
    """
    if a is None or b is None:
        return False
    if a == "." or b == ".":
        return True
    return a == b


def _alternation_branches_overlap(alternatives: list[str]) -> bool:
    """True if two branches can begin with the same input character.

    The classic ``(a|a)+`` / ``(a|ab)+`` exponential-backtracking shape: a quantified
    alternation whose branches share a first character, so the engine has multiple
    equally-valid ways to consume the same input. Compared by first *token* (a literal
    char, an escaped class like ``\\w``, or a ``[...]`` class treated as a wildcard for
    overlap purposes) — a conservative over-approximation, which is the safe direction
    for a DoS guard.
    """
    firsts = [t for alt in alternatives if (t := _first_token(alt)) is not None]
    seen: set[str] = set()
    for f in firsts:
        # A metaclass, a char class, or ``.`` overlaps liberally in an alternation:
        # here over-rejecting an overlapping quantified alternation is the safe DoS
        # direction (these shapes are rare and suspicious in operator config).
        if f in _METACLASS_FIRSTS or f == "." or f.startswith("["):
            return True
        if f in seen:
            return True
        seen.add(f)
    return False


def _quantifier_at(pattern: str, k: int) -> tuple[str, bool] | None:
    """Parse a full quantifier starting at ``pattern[k]``, if one is present.

    Returns ``(quantifier_text, is_unbounded)`` or ``None`` if no quantifier starts
    at ``k``. A quantifier is ``*``/``+``/``?`` (optionally followed by a lazy/possessive
    ``?``/``+``) or a complete ``{m}`` / ``{m,}`` / ``{m,n}`` run — parsed in full so an
    outer ``{N,}`` is never truncated (the bug that let ``(a+){2,}`` slip past). It is
    *unbounded* when the engine can repeat without a fixed cap: ``*``, ``+``, or a
    ``{m,}`` with no upper bound. ``?`` and every ``{m}`` / ``{m,n}`` is bounded.
    """
    if k >= len(pattern):
        return None
    ch = pattern[k]
    if ch in "*+?":
        end = k + 1
        # Absorb a trailing lazy (``?``) or possessive (``+``) modifier.
        if end < len(pattern) and pattern[end] in "?+":
            end += 1
        return pattern[k:end], ch in "*+"
    if ch == "{":
        close = pattern.find("}", k)
        if close == -1:
            return None  # a stray ``{`` is a literal, not a quantifier
        inner = pattern[k + 1 : close]
        # Must look like {m}, {m,} or {m,n} to be a quantifier; else treat as literal.
        if not re.fullmatch(r"\d*(?:,\d*)?", inner) or inner in ("", ","):
            return None
        end = close + 1
        if end < len(pattern) and pattern[end] in "?+":
            end += 1
        # Unbounded ⇔ a comma with no digits after it ({m,}); {m} and {m,n} are bounded.
        is_unbounded = "," in inner and inner.split(",")[1] == ""
        return pattern[k:end], is_unbounded
    return None


def _has_adjacent_unbounded_quantifiers(pattern: str) -> bool:
    """True if two consecutive unbounded-quantified atoms can match the same input.

    The group-free exponential shape ``a*a*…b`` / ``\\w*\\w*…x``: unbounded quantifiers
    on *overlapping* atoms in sequence, with no enclosing group, so the
    ``(...)``-scanning check never sees it. When the engine hits a non-matching tail it
    redistributes the same input across those repeats combinatorially → exponential.

    The overlap condition is essential: ``a+b+c`` is linear (disjoint atoms, nothing to
    redistribute) and must NOT be flagged, whereas ``a*a*…b`` / ``\\w*\\w*…`` is
    exponential. We compare the first token of each unbounded-quantified atom (via
    :func:`_tokens_overlap`) and flag only when two adjacent ones can consume the same
    character — conservative (a safe operator pattern is rewritten without the
    redundant overlapping repeats).

    Atoms counted: a single char / escape / ``.`` / char-class / group immediately
    followed by an *unbounded* quantifier. Bounded quantifiers (``?``, ``{m}``,
    ``{m,n}``) break the run.
    """
    i = 0
    n = len(pattern)
    prev_unbounded_first: str | None = None  # first-token of the last unbounded atom
    while i < n:
        ch = pattern[i]
        # Determine the atom spanning [i, atom_end).
        if ch == "\\":
            atom_end = min(i + 2, n)
        elif ch == "[":
            # Consume a char class as one atom.
            j = i + 1
            if j < n and pattern[j] == "]":  # a literal ']' right after '[' is a member
                j += 1
            while j < n and pattern[j] != "]":
                if pattern[j] == "\\":
                    j += 2
                    continue
                j += 1
            atom_end = min(j + 1, n)  # include the closing ']'
        elif ch == "(":
            # Consume the whole group as one atom.
            depth = 1
            j = i + 1
            while j < n and depth:
                if pattern[j] == "\\":
                    j += 2
                    continue
                if pattern[j] == "(":
                    depth += 1
                elif pattern[j] == ")":
                    depth -= 1
                j += 1
            atom_end = j
        elif ch in "*+?{|":
            # A stray quantifier / alternation bar breaks the adjacency run.
            i += 1
            prev_unbounded_first = None
            continue
        else:
            atom_end = i + 1
        atom = pattern[i:atom_end]
        quant = _quantifier_at(pattern, atom_end)
        if quant is not None and quant[1]:  # this atom carries an unbounded quantifier
            this_first = _first_token(atom)
            if prev_unbounded_first is not None and _tokens_definitely_overlap(
                prev_unbounded_first, this_first
            ):
                return True
            prev_unbounded_first = this_first
        else:
            prev_unbounded_first = None
        # Advance past the atom and any quantifier it carries.
        i = atom_end + (len(quant[0]) if quant is not None else 0)
    return False


def _has_catastrophic_backtracking(pattern: str) -> bool:
    """True if ``pattern`` has a shape that can drive exponential backtracking.

    Flags the dominant catastrophic-backtracking shapes:
      * an unbounded-quantified group whose body itself contains an unbounded
        quantifier — ``(X+)+`` / ``(X*)*`` / ``(a+){2,}`` (any unbounded outer form);
      * an unbounded-quantified group that is a top-level alternation with overlapping
        branches — ``(a|a)+`` / ``(a|ab)+``;
      * a run of adjacent unbounded quantifiers with no enclosing group — ``a*a*…b``.

    Scans each ``(`` … matching ``)`` group, then the group-free adjacency case.
    Conservative: it may over-reject an exotic-but-safe pattern, which is the right
    trade for an operator-config guard — the operator rewrites it without the
    nesting/overlap/redundant repeats.
    """
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\":  # escaped char — skip the pair
            i += 2
            continue
        if ch == "(":
            depth = 1
            j = i + 1
            while j < n and depth:
                if pattern[j] == "\\":
                    j += 2
                    continue
                if pattern[j] == "(":
                    depth += 1
                elif pattern[j] == ")":
                    depth -= 1
                j += 1
            body = pattern[i + 1 : j - 1] if depth == 0 else pattern[i + 1 :]
            # Parse the FULL outer quantifier at the group's close (``{N,}`` is 4+ chars
            # and must not be truncated — the bug that let ``(a+){2,}`` slip past).
            quant = _quantifier_at(pattern, j)
            group_is_unbounded = quant is not None and quant[1]
            if group_is_unbounded:
                if _UNBOUNDED_QUANTIFIER.search(body):
                    return True
                alts = _top_level_alternatives(body)
                if len(alts) > 1 and _alternation_branches_overlap(alts):
                    return True
            i = j + (len(quant[0]) if quant is not None else 0)
            continue
        i += 1
    return _has_adjacent_unbounded_quantifiers(pattern)


def _guard_config_pattern(pattern: str) -> None:
    """Reject an operator config pattern that can cause catastrophic backtracking.

    Raises :class:`~latence_core.errors.ConfigError` (a ``ValueError`` subclass, so the
    pre-taxonomy ``except ValueError`` callers keep catching it — #73) for an over-long
    pattern or a catastrophic-backtracking ReDoS signature (issue #28): a
    nested/outer-unbounded-quantified group (``(\\w+)+``, ``(a+){2,}``), an overlapping
    quantified alternation (``(a|a)+``), or a run of adjacent unbounded quantifiers
    (``a*a*…b``). Only raw ``patterns`` are guarded; escaped literal ``terms`` are always
    safe.
    """
    if len(pattern) > _MAX_PATTERN_LENGTH:
        msg = (
            f"Config pattern is too long ({len(pattern)} > {_MAX_PATTERN_LENGTH} chars); "
            "refuse to compile a potential ReDoS pattern."
        )
        raise ConfigError(msg)
    if _has_catastrophic_backtracking(pattern):
        msg = (
            f"Config pattern {pattern!r} has a catastrophic-backtracking signature "
            r"(e.g. nested/outer unbounded quantifiers '(\w+)+' / '(a+){2,}', an "
            r"overlapping alternation '(a|a)+', or adjacent unbounded quantifiers "
            "'a*a*...b') that can cause exponential backtracking (ReDoS); rewrite it "
            "without the nesting/redundant repeats."
        )
        raise ConfigError(msg)


class GazetteerEntityExtractor:
    """Zero-shot gazetteer/regex EntityExtractor Provider.

    The label set and its recognisers come entirely from ``config`` — nothing is
    hardcoded in core, so the same Provider extracts ``PERSON``/``ORG`` for one
    Pipeline and ``GENE``/``DRUG`` for another (zero-shot, S4 AC).

    Config:
        labels: dict[str, dict]  — the zero-shot label set. Each key is an entity
            type; each value configures how to find it:
              * ``terms``: list[str]  — literal surface strings (case-insensitive,
                whole-word) to tag as this type. A gazetteer.
              * ``patterns``: list[str]  — regexes whose whole match is tagged as
                this type (e.g. an email or an id format). Rejected at compile time
                (``ValueError``) if a pattern has a catastrophic-backtracking ReDoS
                signature (nested/outer unbounded quantifiers, an overlapping quantified
                alternation, or adjacent unbounded quantifiers) or is absurdly long —
                the framework compiles operator config regex over untrusted content, so
                the guard keeps a bad pattern from becoming a DoS lever (issue #28).
              * ``confidence``: float  — the score to stamp on mentions of this
                type (default ``default_confidence``).
            ``labels`` may also be given as ``list[str]`` for pattern-free label
            declaration (produces the empty extractor for those types — useful when
            only the label vocabulary matters and recognisers arrive via config).
        default_confidence: float  — confidence when a label sets none (default 0.9).
        case_sensitive: bool  — match ``terms`` case-sensitively (default False).
        min_chars: int  — a chunk whose stripped length is below this is treated as
            near-empty and yields no mentions (default 1).

    Determinism: matches are emitted in ``(char_start, label)`` order, so identical
    input + config yields byte-identical mentions.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        A gazetteer/regex extractor ships no model weights (``model_id=None``) and is a
        pure-Python part of ``latence-core`` (Apache-2.0, verified — the package's own
        license, weights-and-code trivially the same code). Deterministic by construction
        (matches emitted in ``(char_start, label)`` order) and CPU-only; it does not batch
        (a per-chunk scan), and has no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._default_confidence = float(config.get("default_confidence", 0.9))
        self._case_sensitive = bool(config.get("case_sensitive", False))
        self._min_chars = int(config.get("min_chars", 1))
        self._recognisers = self._compile_labels(config.get("labels", {}))

    # -- config compilation -------------------------------------------------

    def _compile_labels(
        self, labels: dict[str, Any] | list[str]
    ) -> list[tuple[str, re.Pattern[str], float]]:
        """Compile the zero-shot label set into ``(label, pattern, confidence)`` rules.

        A list-of-labels form declares the vocabulary with no recognisers (no
        mentions until terms/patterns are given). The dict form compiles each
        label's literal ``terms`` (word-boundary, escaped) and raw ``patterns`` into
        one alternation per label, so recognition is a single scan per label.
        """
        rules: list[tuple[str, re.Pattern[str], float]] = []
        flags = 0 if self._case_sensitive else re.IGNORECASE
        if isinstance(labels, list):
            # Vocabulary-only declaration: valid, but recognises nothing.
            return rules
        for label, spec in labels.items():
            if not isinstance(spec, dict):
                msg = f"Label {label!r} config must be a mapping (got {type(spec).__name__})."
                raise TypeError(msg)
            confidence = float(spec.get("confidence", self._default_confidence))
            alternatives: list[str] = []
            for term in spec.get("terms", []):
                # Whole-word, literal: escape the term and bound it so "Amazon" does
                # not match inside "Amazonian".
                alternatives.append(rf"\b{re.escape(str(term))}\b")
            for pat in spec.get("patterns", []):
                pat_str = str(pat)
                # ReDoS guard (issue #28): refuse a catastrophic-backtracking pattern
                # before it is ever compiled or run over untrusted content.
                _guard_config_pattern(pat_str)
                alternatives.append(pat_str)
            if not alternatives:
                continue
            pattern = re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
            rules.append((label, pattern, confidence))
        return rules

    # -- extraction ---------------------------------------------------------

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        for chunk in chunks:
            yield from self._extract_one(chunk)

    def _extract_one(self, chunk: ChunkRecord) -> Iterator[EntityMention]:
        text = chunk.content
        # Empty / near-empty chunk: no mentions, no error (S4 AC).
        if len(text.strip()) < self._min_chars:
            return
        matches = self._find_matches(text)
        # A mention's document-level offset is its chunk-local match offset resolved
        # through the chunk's stripped→original offset map: chunk ``content`` is
        # markup-stripped, so ``char_start + local`` is only a lower bound once markup
        # was stripped before the match. Absent an offset map (pre-v11 chunk) fall back
        # to that shift; the chunk always carries its own start (S3 invariant), 0 only
        # defensively so a mention always has an offset.
        offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        chunk_char_start = chunk.provenance.char_start or 0
        # Resolve each mention's OWN page from the parent's page map (carried on the
        # chunk), not the chunk's whole page range: a mention on a later page of a
        # multi-page chunk must cite that page (S4 "correct source page"). Built once
        # per chunk. Absent (pre-S4 chunk / no page map) → inherit the chunk range.
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
        for i, (label, surface, local_start, local_end, confidence) in enumerate(matches):
            if offset_index is not None:
                doc_char_start, doc_char_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_char_start = chunk_char_start + local_start
                doc_char_end = chunk_char_start + local_end
            provenance = self._mention_provenance(
                chunk.provenance,
                doc_char_start=doc_char_start,
                doc_char_end=doc_char_end,
                page_index=page_index,
            )
            yield EntityMention(
                record_id=f"{chunk.record_id}#entity-{i}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=i,
                label=label,
                text=surface,
                confidence=confidence,
                provenance=provenance,
                classification=chunk.classification,
            )

    def _find_matches(self, text: str) -> list[_Match]:
        """All non-overlapping matches across every label, deterministically ordered.

        Each label is scanned independently; overlaps *between* labels are resolved
        by preferring the earlier start, then the longer span, then the label name —
        so a fixed, deterministic set of non-overlapping mentions results (identical
        input ⇒ identical output). Within a label ``finditer`` is already
        non-overlapping.
        """
        raw: list[_Match] = []
        for label, pattern, confidence in self._recognisers:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if end > start:
                    raw.append((label, m.group(0), start, end, confidence))
        # Deterministic ordering: by start, then longer span first, then label.
        raw.sort(key=lambda t: (t[2], -(t[3] - t[2]), t[0]))
        # Drop any match that overlaps one already accepted (first-wins under the
        # sort above), so the mention set is non-overlapping and stable.
        accepted: list[_Match] = []
        last_end = -1
        for match in raw:
            _, _, start, end, _ = match
            if start >= last_end:
                accepted.append(match)
                last_end = end
        return accepted

    @staticmethod
    def _mention_provenance(
        chunk_prov: Provenance,
        doc_char_start: int,
        doc_char_end: int,
        page_index: PageOffsetIndex | None,
    ) -> Provenance:
        """Provenance for a mention: chunk lineage + the mention's own span AND page.

        The char span is narrowed to the mention's own ``[doc_char_start,
        doc_char_end)`` in the parent document. The page span is re-resolved from
        that offset through the parent's page map (``page_index``) so a mention on a
        later page of a multi-page chunk cites *its own* page — the ``page_end`` uses
        the last included character (``doc_char_end - 1``) so a mention straddling a
        page boundary records both pages. When no page map is available the mention
        degrades to inheriting the chunk's page range.
        """
        update: dict[str, int] = {
            "char_start": doc_char_start,
            "char_end": doc_char_end,
        }
        if page_index is not None:
            update["page_start"] = page_index.page_for_offset(doc_char_start)
            update["page_end"] = page_index.page_for_offset(max(doc_char_end - 1, doc_char_start))
        return chunk_prov.model_copy(update=update)

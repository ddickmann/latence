"""Candidate mining: what types could this corpus possibly want, and how much mass does each carry.

Three slots, three miners, all deterministic and all offline:

``entity_types``
    RAKE-style candidate phrases (Rose, Engel, Cramer & Cowley, 2010) — maximal runs of content
    words between stopwords, punctuation and bare numbers. Every contiguous sub-sequence of a phrase
    is a candidate TYPE, and a type covers a phrase when its tokens are a contiguous sub-sequence of
    that phrase's tokens. That is not a modelling choice made for convenience: it is exactly the
    matching rule ``LexiconLabelInducer`` will apply downstream (``\\b(?:cue)\\b``), so the coverage
    number this module feeds the selector is the coverage the pipeline will actually get.

``relation_types``
    ReVerb-style (Fader, Soderland & Etzioni, 2011): the relation is the ``V | VP | VW*P`` material
    between two argument phrases. With no POS tagger the verb test is a closed set plus conservative
    suffix morphology (:func:`~latence_label_profiler.text.is_verb_like`), which is the weakest part
    of this package — so the artifact carries an explicit note rather than letting a thin relation
    pool read as "this corpus has few relations".

``pii_types``
    A declared table of PII types, each with structural detectors and contextual cues. Two rules
    make it safe: a detected value is **masked out of the text before phrase mining**, so no
    person's name or account number can become an entity type; and a detected value is **never**
    echoed into a cue list — the emitted cues are contextual keywords only. When a type's whole mass
    came from the detector, it is flagged ``detector_only`` so nobody reads its cue list as evidence
    that per-chunk induction will fire on a bare value. (For those, the non-bypassable ADR-0044
    redaction floor is the mechanism, not this pool.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from latence_label_profiler.text import (
    CLOSED_VERBS,
    COPULA_VERBS,
    DEFAULT_STOPWORDS,
    RELATION_TAIL_WORDS,
    Token,
    canonical_label,
    is_verb_like,
    subsequences,
    tokenize,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: The three slots an ``InducedLabels`` carries, in the order ``LexiconLabelInducer`` declares them.
SLOTS: tuple[str, ...] = ("entity_types", "relation_types", "pii_types")

DEFAULT_MAX_PHRASE_TOKENS = 4
DEFAULT_MAX_RELATION_TOKENS = 4
DEFAULT_MAX_CUES = 8


@dataclass(frozen=True)
class PiiSpec:
    """One declared PII type: how to DETECT it, and what to tell the inducer to look for.

    ``detector`` measures mass and drives masking. ``cues`` are the literal, value-free strings that
    go into the emitted candidate space — the inducer escapes them, so they are never a regex.
    """

    label: str
    cues: tuple[str, ...]
    detector: re.Pattern[str] | None = None


# The detectors are deliberately conservative: a false PII hit costs a masked-out phrase (a small
# recall loss in the entity slot), while a miss costs nothing here because ADR-0044's redaction
# floor is the real safety net. None of these carry a value into the output.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Separator-delimited on purpose: a bare digit run is an identifier, not a phone number, and a
# greedy phone pattern that swallowed one would mask a document's own reference numbers out of the
# corpus before the entity miner ever saw them.
_PHONE = re.compile(r"\b(?:\+\d{1,3}[ /-]?)?(?:\(\d{2,5}\)|\d{2,5})(?:[ /-]\d{2,6}){1,4}\b")

PII_TABLE: tuple[PiiSpec, ...] = (
    PiiSpec("email address", ("email", "e mail", "email address", "mail address"), _EMAIL),
    PiiSpec("iban", ("iban", "bank account", "account number"), _IBAN),
    PiiSpec("credit card number", ("credit card", "card number", "kreditkarte"), _CARD),
    PiiSpec("ip address", ("ip address", "ip adresse"), _IPV4),
    PiiSpec("phone number", ("phone", "telephone", "telefon", "mobile", "fax"), _PHONE),
    PiiSpec("date of birth", ("date of birth", "dob", "geburtsdatum", "born on"), None),
    PiiSpec("tax identification number", ("vat number", "tax id", "steuernummer"), None),
    PiiSpec(
        "national identification number",
        ("passport number", "social security number", "ssn", "personalausweis"),
        None,
    ),
    PiiSpec("postal address", ("street", "strasse", "postal code", "zip code", "postleitzahl"),
            None),
    PiiSpec("person name", ("mr", "mrs", "ms", "dr", "herr", "frau"), None),
)


@dataclass(frozen=True)
class Candidate:
    """One candidate type: what it covers, how much mass that is, and how to look for it."""

    label: str
    covers: frozenset[str]
    mass: int
    document_frequency: int
    cues: tuple[str, ...]
    detector_only: bool = False


@dataclass
class SlotMining:
    """One slot's mined universe: surface mass, and the candidates over it."""

    surface_mass: dict[str, int] = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)

    @property
    def total_mass(self) -> int:
        return sum(self.surface_mass.values())


@dataclass
class MinedCorpus:
    """Every slot's mining result, plus what was actually read."""

    slots: dict[str, SlotMining]
    texts_mined: int


def _cue_for(tokens: tuple[str, ...]) -> str:
    return " ".join(tokens)


def _mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace every structurally-detected PII value with a space; return the counts it removed.

    Masking BEFORE phrase mining is the whole reason no PII value can reach the emitted type space.
    Replacing with a space (not a placeholder token) also makes the masked span a phrase boundary,
    so ``Contact jane.doe@example.com for details`` cannot mine ``jane`` or ``doe`` as a type.
    """
    counts: dict[str, int] = {}
    masked = text
    for spec in PII_TABLE:
        if spec.detector is None:
            continue
        hits = 0

        def _blank(match: re.Match[str]) -> str:
            nonlocal hits
            hits += 1
            return " " * (match.end() - match.start())

        masked = spec.detector.sub(_blank, masked)
        if hits:
            counts[spec.label] = counts.get(spec.label, 0) + hits
    return masked, counts


def argument_spans(
    tokens: tuple[Token, ...], runs: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    """Trim each content run down to its ARGUMENT core: strip verb-like tokens off both ends.

    A verb that is also a content word ("maps", "recorded", "wurde") does not break a RAKE run,
    because it is not a stopword. Untrimmed, ``Each commodity code maps to a tariff heading`` mines
    the entity type ``commodity code maps`` — and never mines the relation ``maps to`` at all,
    because the verb is sitting inside an argument instead of in the gap between two of them.
    Trimming both ends fixes both halves at once: the entity slot loses the verb and the relation
    slot gains it. A run that trims to nothing was pure relation material and is not an argument.
    """
    spans: list[tuple[int, int]] = []
    for start, end in runs:
        left, right = start, end
        while left < right and _is_leading_verb(tokens, left):
            left += 1
        while right > left and is_verb_like(tokens[right - 1]):
            right -= 1
        if left < right:
            spans.append((left, right))
    return tuple(spans)


def _is_leading_verb(tokens: tuple[Token, ...], index: int) -> bool:
    """Is the token at the START of a run a verb, or a participial ADJECTIVE modifying the head?

    ``combined nomenclature`` and ``is classified under`` share a suffix and nothing else. The
    signal that separates them is the token immediately before: a participle after a copula is a
    verb, a participle after a determiner is an adjective. So an unambiguous closed-set verb
    ("maps", "wurde") is trimmed on sight, while a suffix-derived one is trimmed only when a copula
    precedes it. Without this, ``combined`` is cut out of its own noun phrase and mined as a
    relation type — measured, not hypothetical: it is what this miner did before the check existed.
    """
    token = tokens[index]
    if token.text in CLOSED_VERBS:
        return True
    if not is_verb_like(token):
        return False
    return index > 0 and tokens[index - 1].text in COPULA_VERBS


def _phrase_surface(tokens: tuple[Token, ...], *, max_phrase_tokens: int) -> tuple[str, ...] | None:
    """One trimmed argument span reduced to its surface token tuple."""
    if not tokens:
        return None
    words = tuple(token.text for token in tokens)
    if len(words) > max_phrase_tokens:
        # English and German noun phrases are head-final, so the tail carries the head. Truncating
        # to the last max_phrase_tokens keeps the type and drops the modifiers.
        words = words[-max_phrase_tokens:]
    return words


def _relation_surface(
    gap: Sequence[Token], *, max_relation_tokens: int
) -> tuple[str, ...] | None:
    """ReVerb's ``V | VP | VW*P``: from the first verb-like token to the end of the span."""
    first_verb = next((i for i, token in enumerate(gap) if is_verb_like(token)), None)
    if first_verb is None:
        return None
    kept = [
        token.text
        for token in gap[first_verb:]
        if is_verb_like(token) or token.text in RELATION_TAIL_WORDS or token.kind == "word"
    ]
    if not kept:
        return None
    return tuple(kept[:max_relation_tokens])


def _relation_surfaces(
    tokens: tuple[Token, ...],
    arguments: tuple[tuple[int, int], ...],
    *,
    max_relation_tokens: int,
) -> list[tuple[str, ...]]:
    """Every relation surface in one sentence: the material between consecutive argument spans."""
    surfaces: list[tuple[str, ...]] = []
    for left, right in zip(arguments, arguments[1:], strict=False):
        gap = tokens[left[1] : right[0]]
        surface = _relation_surface(gap, max_relation_tokens=max_relation_tokens)
        if surface is not None:
            surfaces.append(surface)
    return surfaces


def mine_corpus(
    texts: Iterable[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    max_phrase_tokens: int = DEFAULT_MAX_PHRASE_TOKENS,
    max_relation_tokens: int = DEFAULT_MAX_RELATION_TOKENS,
    min_mention_mass: int = 3,
    min_document_frequency: int = 2,
    max_cues: int = DEFAULT_MAX_CUES,
) -> MinedCorpus:
    """Mine every slot's candidate type space from ``texts``. Pure, offline, order-deterministic."""
    surface_mass: dict[str, dict[tuple[str, ...], int]] = {
        "entity_types": {},
        "relation_types": {},
    }
    # Document IDs per surface, not a count: a candidate's document frequency is the size of the
    # UNION over the surfaces it covers, and a union cannot be rebuilt from per-surface counts.
    surface_docs: dict[str, dict[tuple[str, ...], set[int]]] = {
        "entity_types": {},
        "relation_types": {},
    }
    pii_detected: dict[str, int] = {}
    pii_cue_hits: dict[str, int] = {}
    pii_df: dict[str, int] = {}
    texts_mined = 0

    cue_patterns = {
        spec.label: tuple(
            re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE) for cue in spec.cues
        )
        for spec in PII_TABLE
    }

    for doc_id, raw_text in enumerate(texts):
        texts_mined += 1
        masked, detected = _mask_pii(raw_text)
        seen_pii: set[str] = set()
        for label, hits in detected.items():
            pii_detected[label] = pii_detected.get(label, 0) + hits
            seen_pii.add(label)
        for spec in PII_TABLE:
            hits = sum(
                len(pattern.findall(raw_text)) for pattern in cue_patterns[spec.label]
            )
            if hits:
                pii_cue_hits[spec.label] = pii_cue_hits.get(spec.label, 0) + hits
                seen_pii.add(spec.label)
        for label in sorted(seen_pii):
            pii_df[label] = pii_df.get(label, 0) + 1

        tokenized = tokenize(masked, stopwords=stopwords)
        seen: dict[str, set[tuple[str, ...]]] = {"entity_types": set(), "relation_types": set()}
        for sentence in tokenized.sentences:
            arguments = argument_spans(sentence.tokens, sentence.runs)
            for span in arguments:
                words = _phrase_surface(
                    sentence.run_tokens(span), max_phrase_tokens=max_phrase_tokens
                )
                if words is None:
                    continue
                surface_mass["entity_types"][words] = (
                    surface_mass["entity_types"].get(words, 0) + 1
                )
                seen["entity_types"].add(words)
            for words in _relation_surfaces(
                sentence.tokens, arguments, max_relation_tokens=max_relation_tokens
            ):
                surface_mass["relation_types"][words] = (
                    surface_mass["relation_types"].get(words, 0) + 1
                )
                seen["relation_types"].add(words)
        for slot, surfaces in seen.items():
            for words in surfaces:
                surface_docs[slot].setdefault(words, set()).add(doc_id)

    slots: dict[str, SlotMining] = {}
    for slot in ("entity_types", "relation_types"):
        slots[slot] = _build_slot(
            surface_mass[slot],
            surface_docs[slot],
            max_phrase_tokens=(
                max_phrase_tokens if slot == "entity_types" else max_relation_tokens
            ),
            min_mention_mass=min_mention_mass,
            min_document_frequency=min_document_frequency,
            max_cues=max_cues,
            vacuous_tokens=(
                # ReVerb's lexical constraint (Fader et al. 2011), applied to the relation slot
                # only. A phrase built entirely from copulas, prepositions and function words says
                # nothing — and greedy max-coverage ranks exactly those FIRST, because a copula
                # appears in every relation phrase it occurs in. "is" would win the top slot and be
                # useless to the extractor; "is classified under" and "maps to" survive.
                COPULA_VERBS | RELATION_TAIL_WORDS | stopwords
                if slot == "relation_types"
                else None
            ),
        )
    slots["pii_types"] = _build_pii_slot(
        pii_detected,
        pii_cue_hits,
        pii_df,
        min_mention_mass=min_mention_mass,
        min_document_frequency=min_document_frequency,
    )
    return MinedCorpus(slots=slots, texts_mined=texts_mined)


def _build_slot(
    surface_mass: dict[tuple[str, ...], int],
    surface_docs: dict[tuple[str, ...], set[int]],
    *,
    max_phrase_tokens: int,
    min_mention_mass: int,
    min_document_frequency: int,
    max_cues: int,
    vacuous_tokens: frozenset[str] | None = None,
) -> SlotMining:
    """Turn surface counts into the candidate universe: every sub-sequence, with its covered mass.

    Keyed by the CANONICAL label rather than by the token tuple, so two spellings that normalise to
    one type (``e-mail`` and ``e mail``) merge their coverage instead of the second silently losing
    to the first.
    """
    covers: dict[str, set[tuple[str, ...]]] = {}
    for words in sorted(surface_mass):
        for sub in subsequences(words, max_len=max_phrase_tokens):
            label = canonical_label(" ".join(sub))
            if label is None:
                continue
            covers.setdefault(label, set()).add(words)

    mass_by_surface = {" ".join(words): count for words, count in surface_mass.items()}
    candidates: dict[str, Candidate] = {}
    for label in sorted(covers):
        if vacuous_tokens is not None and all(
            token in vacuous_tokens for token in label.split(" ")
        ):
            continue
        covered = sorted(covers[label])
        mass = sum(surface_mass[words] for words in covered)
        documents: set[int] = set()
        for words in covered:
            documents |= surface_docs.get(words, set())
        if mass < min_mention_mass or len(documents) < min_document_frequency:
            continue
        # Cues: the type's own name first (so a cue list is never empty and never instance-shaped),
        # then the observed surfaces it covers, heaviest first.
        ranked = sorted(covered, key=lambda w: (-surface_mass[w], w))
        cue_list = [label] + [_cue_for(words) for words in ranked]
        candidates[label] = Candidate(
            label=label,
            covers=frozenset(" ".join(words) for words in covered),
            mass=mass,
            document_frequency=len(documents),
            cues=tuple(dict.fromkeys(cue_list))[:max_cues],
        )
    return SlotMining(surface_mass=mass_by_surface, candidates=candidates)


def _build_pii_slot(
    detected: dict[str, int],
    cue_hits: dict[str, int],
    document_frequency: dict[str, int],
    *,
    min_mention_mass: int,
    min_document_frequency: int,
) -> SlotMining:
    """PII candidates are DECLARED, not mined — only their mass is measured.

    Their coverage sets are disjoint singletons (one type, one element), which makes greedy exactly
    optimal on this slot and the certificate exact. That is stated in the artifact rather than left
    for a reader to infer from a suspiciously perfect ratio.
    """
    surface_mass: dict[str, int] = {}
    candidates: dict[str, Candidate] = {}
    for spec in PII_TABLE:
        mass = detected.get(spec.label, 0) + cue_hits.get(spec.label, 0)
        if mass <= 0:
            continue
        surface_mass[spec.label] = mass
        if mass < min_mention_mass:
            continue
        if document_frequency.get(spec.label, 0) < min_document_frequency:
            continue
        candidates[spec.label] = Candidate(
            label=spec.label,
            covers=frozenset({spec.label}),
            mass=mass,
            document_frequency=document_frequency.get(spec.label, 0),
            cues=spec.cues,
            detector_only=detected.get(spec.label, 0) > 0 and cue_hits.get(spec.label, 0) == 0,
        )
    return SlotMining(surface_mass=surface_mass, candidates=candidates)

"""``label_inducer.lexicon`` — the non-frontier per-chunk inducer: a zero-shot tagger, no LLM (T5).

Per-chunk induction (T1) is one induction per distinct chunk text, and that is the right shape for
extraction quality — but "one frontier-LLM call per distinct chunk across your corpus" is a number
an enterprise evaluator will rightly balk at. T5's answer is layered: chunk-text dedup and a call
budget bound the LLM path (both in :mod:`latence_schema_inducer.provider`), and the *task itself* —
"which of these types are present in this one short chunk?" — does not need a frontier model. This
module is the seam's second real adapter: a deterministic, CPU-only, **zero-network** tagger over an
operator-supplied CANDIDATE TYPE SPACE.

Why a candidate-space tagger rather than a bundled small model
--------------------------------------------------------------
The three ways to make the per-chunk pass cheap are (a) a smaller served model, (b) a distilled
classifier, (c) a deterministic tagger over a candidate space. (a) needs no new Provider at all —
:class:`~latence_schema_inducer.provider.LLMLabelInducer` is already model-agnostic, so pointing it
at a local vLLM / Ollama / llama.cpp box is a ``base_url`` change and nothing else (see
"Pointing the LLM inducer at a local model" below). (b) would mean vendoring weights, which this
framework does not do (ADR-0012/0016). (c) is the one that adds real capability: it costs nothing,
never leaves the host, is byte-reproducible by construction, and is exactly right for the corpora
that most want per-chunk induction — a known domain where the operator can NAME the type space but
cannot say *which chunk* carries *which* type. It is also the honest offline adapter for the seam:
it makes the swap real (two adapters with genuinely different behaviour and different failure
modes), not speculative.

What it does
------------
The operator declares a candidate type space per slot — either a plain list of type names, or a
mapping of type name → the surface cues that evidence it::

    provider: label_inducer.lexicon
    config:
      granularity: chunk
      candidates:
        entity_types:
          tariff heading: ["heading", "HS code", "CN code"]
          commodity: ["commodity", "goods", "product"]
        relation_types: ["classified under"]        # list form: the type name IS its own cue
        pii_types:
          customer reference: ["customer ref", "cust. ref"]

For each chunk, a candidate type is induced iff at least ``min_hits`` of its cues occur in the
chunk's text (case-insensitive, on word boundaries so ``iban`` does not fire inside ``Caribbean``).
Types keep the operator's declaration order, so the emitted schema is a byte-stable function of the
config and the text — a seeded run is reproducible without a seed.

The failure posture is the same fail-open contract the LLM inducer honours: a chunk that matches
nothing gets ``induced_labels = None`` and the extractor falls back to its config labels. Nothing
here can raise mid-run — the only errors are typed CONFIGURATION errors raised at construction
(:class:`ValueError`), because an unusable candidate space is an operator mistake to fix, not a
runtime condition to fail open on.

Cost reporting is the SAME seam the Runner reads for the LLM inducer, so a Quality Report is
comparable across the swap: ``induction_calls`` is 0 (there is no endpoint), the token counters are
0 (an honest "no tokens were spent", not an unknown), and ``induction_cache_hits`` still records
what the chunk-text dedup answered for free.

Pointing the LLM inducer at a local model
-----------------------------------------
The cheap path does not have to be this Provider. ``label_inducer.llm`` speaks the OpenAI
chat-completions API, so any local server that does too is a config-only swap — no code, no new
package::

    provider: label_inducer.llm
    config:
      granularity: chunk
      base_url: http://localhost:8000/v1     # vLLM / SGLang / llama.cpp / Ollama (/v1)
      model: Qwen3-4B-Instruct               # whatever id your server registers
      api_key: EMPTY                         # a local server accepts a placeholder
      max_concurrency: 16                    # local boxes take more in-flight than a hosted API
      dedup: true                            # induce once per DISTINCT chunk text
      max_induction_calls: 50000             # optional: a hard per-run bound, loudly reported

Both paths keep every safety net (structured output, truncation/repair retries, empty-recovery,
sanitization, fail-open) — they are the same Provider with a different endpoint.

Licensing: framework code, Apache-2.0. No weights, no network, no third-party runtime dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.contracts import InducedLabels
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

from latence_schema_inducer.sanitize import (
    CHUNK_LABEL_CEILING,
    MAX_LABEL_CEILING,
    sanitize_label,
    sanitize_labels,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

_LOG = logging.getLogger("latence.schema_induction")

# The three slots an InducedLabels carries — the same shape the LLM inducer produces, so the two
# Providers are genuinely interchangeable behind the Capability.
_SLOTS = ("entity_types", "relation_types", "pii_types")
# The model id this Provider stamps on its InducedLabels. It names the METHOD, not a checkpoint —
# there is no model. Stamping something honest here keeps the provenance column readable when a
# corpus was induced by the cheap path rather than by an LLM.
_METHOD_ID = "lexicon"
# A chunk shorter than this (stripped) is skipped, matching the LLM inducer's ``min_chars`` guard.
_DEFAULT_MIN_CHARS = 1


@dataclass(frozen=True)
class _Candidate:
    """One candidate type + the compiled cue patterns that evidence it, in declaration order."""

    label: str
    patterns: tuple[re.Pattern[str], ...]

    def hits(self, text: str) -> int:
        """How many DISTINCT cues of this type occur in ``text`` (the ``min_hits`` gauge).

        Distinct cues, not total occurrences: a chunk that repeats one cue ten times is still one
        piece of evidence, while a chunk matching two different cues is genuinely stronger. That
        makes ``min_hits`` a precision knob rather than a frequency threshold.
        """
        return sum(1 for pattern in self.patterns if pattern.search(text) is not None)


class LexiconLabelInducer:
    """LabelInducer Provider that tags each chunk against an operator-declared candidate type space.

    Config:
        candidates: mapping — the candidate type space, keyed by slot (``entity_types`` /
            ``relation_types`` / ``pii_types``). Each slot is either a LIST of type names (each
            name is its own cue) or a MAPPING of type name → list of surface cues. At least one
            usable type must be declared, else construction raises :class:`ValueError` — an inducer
            with an empty candidate space could only ever induce nothing, which is a config mistake
            worth failing loudly on rather than a run-time fall-back to pretend about.
        granularity: "chunk" | "doc" — ``chunk`` (the default, and the point of this Provider):
            tag each chunk against its own text. ``doc``: tag each document's concatenated text and
            broadcast to its chunks. There is deliberately NO ``corpus`` mode: a corpus-merged
            schema is the very thing the redesign removed, and this Provider has nothing to merge.
        min_hits: int — how many DISTINCT cues must match before a type is induced (default 1).
            Raise it to 2 for a noisy corpus where single-word cues over-fire.
        whole_word: bool — match cues on word boundaries (default True), so ``iban`` does not fire
            inside ``Caribbean``. Set False for languages/corpora where compounding makes boundary
            matching too strict (German ``Rechnungsnummernkreis`` contains ``Rechnungsnummer``).
        label_ceiling: int — the per-slot count safety ceiling (default 64, clamped to 256). Same
            role as on the LLM path: a bound against a degenerate config, never a salience cap.
        dedup: bool — tag once per DISTINCT chunk text and fan the result to duplicates (default
            True). Lossless here by construction (this Provider is a pure function of the text), so
            the only reason to disable it is to make the cost columns comparable to a run that did.
        min_chars: int — a chunk/document shorter than this (stripped) is skipped (default 1).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor: pure-CPU, deterministic, zero-cost, no model.

        ``deterministic=True`` — unlike the LLM inducer, this one genuinely IS byte-identical run to
        run, because it is a pure function of (text, config). ``cost_per_1k=0.0`` is a measured
        fact, not an estimate: it makes no calls.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=8,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-25",
            deterministic=True,
            batch=False,
            cost_per_1k=0.0,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        granularity = str(config.get("granularity", "chunk")).lower()
        if granularity not in ("doc", "chunk"):
            msg = (
                "LexiconLabelInducer 'granularity' must be 'chunk' or 'doc' (there is no 'corpus' "
                f"mode — a corpus-merged schema is what the per-chunk redesign removed), got "
                f"{granularity!r}."
            )
            raise ValueError(msg)
        self._granularity = granularity
        self._min_hits = max(1, int(config.get("min_hits", 1)))
        self._whole_word = bool(config.get("whole_word", True))
        self._label_ceiling = max(
            1, min(int(config.get("label_ceiling", CHUNK_LABEL_CEILING)), MAX_LABEL_CEILING)
        )
        self._dedup = bool(config.get("dedup", True))
        self._min_chars = max(0, int(config.get("min_chars", _DEFAULT_MIN_CHARS)))
        self._candidates = _compile_candidates(
            config.get("candidates"), whole_word=self._whole_word, ceiling=self._label_ceiling
        )
        # The SAME observability attributes the Runner drains off the LLM inducer, so swapping the
        # Provider does not silently blank the Quality Report's induction columns.
        self.endpoint_retries = 0
        self.induction_fallbacks = 0
        self.induction_documents = 0
        self.induction_empty_after_sanitization = 0
        # Zero by construction and reported as zero — an operator comparing this run's report with
        # an LLM run's reads "0 calls, 0 tokens" as the measured saving, not as missing data.
        self.induction_calls = 0
        self.induction_cache_hits = 0
        self.induction_prompt_tokens = 0
        self.induction_completion_tokens = 0
        self.induction_budget_exhausted = False

    # -- induction ----------------------------------------------------------

    def induce(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Annotate each chunk with the candidate types evidenced in it (chunk→chunk, 1:1, ordered).

        Every chunk is yielded exactly once in input order. A chunk that is too small, or that
        matches no candidate, flows through unchanged (``induced_labels`` stays ``None``) so the
        extractor uses its config labels — the same fail-open contract the LLM inducer honours,
        reached here without any possibility of a network or parse failure.
        """
        chunk_list = list(chunks)
        if self._granularity == "doc":
            yield from self._induce_per_document(chunk_list)
            return
        jobs = [
            (chunk.record_id, chunk.content.strip())
            for chunk in chunk_list
            if len(chunk.content.strip()) >= self._min_chars
        ]
        induced = self._induce_all(jobs)
        for chunk in chunk_list:
            labels = induced.get(chunk.record_id)
            yield chunk if labels is None else chunk.model_copy(update={"induced_labels": labels})

    def _induce_per_document(self, chunk_list: list[ChunkRecord]) -> Iterator[ChunkRecord]:
        by_doc: dict[str, list[ChunkRecord]] = {}
        order: list[str] = []
        for chunk in chunk_list:
            if chunk.document_record_id not in by_doc:
                by_doc[chunk.document_record_id] = []
                order.append(chunk.document_record_id)
            by_doc[chunk.document_record_id].append(chunk)
        jobs: list[tuple[str, str]] = []
        for doc_id in order:
            text = "\n".join(c.content for c in by_doc[doc_id]).strip()
            if len(text) >= self._min_chars:
                jobs.append((doc_id, text))
        induced = self._induce_all(jobs)
        for chunk in chunk_list:
            labels = induced.get(chunk.document_record_id)
            yield chunk if labels is None else chunk.model_copy(update={"induced_labels": labels})

    def _induce_all(self, jobs: list[tuple[str, str]]) -> dict[str, InducedLabels | None]:
        """Tag every job's text once per DISTINCT text, keyed back to every job that carried it.

        The same dedup layer the LLM path uses (T5) — kept here so the reported cache-hit rate means
        the same thing on both Providers, and because tagging a corpus's boilerplate once is free
        speed even when each tagging is cheap.
        """
        if not jobs:
            return {}
        self.induction_documents += len(jobs)
        distinct: list[str] = []
        slot_of_text: dict[str, int] = {}
        slot_of_job: list[int] = []
        for _key, text in jobs:
            slot = slot_of_text.get(text) if self._dedup else None
            if slot is None:
                slot = len(distinct)
                distinct.append(text)
                slot_of_text[text] = slot
            slot_of_job.append(slot)
        self.induction_cache_hits += len(jobs) - len(distinct)
        tagged = [self._tag(text) for text in distinct]
        # A text that matched nothing IS a fall-back to the config-label floor — tallied on the same
        # Quality-Report column the LLM path uses, per DISTINCT text (as there), so a high count
        # reads as "your candidate space does not cover this corpus" rather than as a defect.
        self.induction_fallbacks += sum(1 for item in tagged if item is None)
        return {key: tagged[slot] for (key, _text), slot in zip(jobs, slot_of_job, strict=True)}

    def _tag(self, text: str) -> InducedLabels | None:
        """The whole model: the candidate types whose cues are evidenced in ``text``.

        Returns ``None`` when nothing matched — the fail-open signal the caller turns into "leave
        ``induced_labels`` unset", so the extractor uses its config labels rather than an empty
        schema. Order is the operator's declaration order in every slot.
        """
        found: dict[str, list[str]] = {}
        for slot in _SLOTS:
            matched = [
                c.label
                for c in self._candidates.get(slot, ())
                if c.hits(text) >= self._min_hits
            ]
            if matched:
                # Re-run the shared sanitizer so both Providers' output obeys the identical bounds
                # (canonical form, dedup, the count ceiling) — the candidate space was already
                # sanitized at construction, so this only ever enforces the ceiling.
                found[slot] = sanitize_labels(matched, max_labels=self._label_ceiling).labels
        if not any(found.values()):
            return None
        return InducedLabels(
            entity_types=found.get("entity_types", []),
            relation_types=found.get("relation_types", []),
            pii_types=found.get("pii_types", []),
            model_id=_METHOD_ID,
        )


def _compile_candidates(
    raw: object, *, whole_word: bool, ceiling: int
) -> dict[str, tuple[_Candidate, ...]]:
    """Validate + compile the operator's candidate type space; raise on an unusable one.

    Accepts, per slot, either a list of type names or a ``{type: [cue, ...]}`` mapping (a type with
    an empty cue list falls back to its own name as the cue, which is what the list form means).
    Type names go through the SAME :func:`~latence_schema_inducer.sanitize.sanitize_label` canonical
    normal form the LLM path uses, so a lexicon-induced ``Payment_Product`` and an LLM-induced
    ``payment product`` are the same type downstream.

    Every rejection is a TYPED configuration error raised HERE, at construction — not a fail-open at
    run time. A candidate space that cannot induce anything is an operator mistake; failing open on
    it would produce a silently label-free run that looks like a working one.
    """
    if not isinstance(raw, dict) or not raw:
        msg = (
            "LexiconLabelInducer requires a non-empty 'candidates' mapping "
            f"(keys: {', '.join(_SLOTS)}); got {type(raw).__name__}."
        )
        raise ValueError(msg)
    unknown = sorted(str(k) for k in raw if k not in _SLOTS)
    if unknown:
        msg = f"LexiconLabelInducer 'candidates' has unknown slot(s) {unknown}; expected {_SLOTS}."
        raise ValueError(msg)
    compiled: dict[str, tuple[_Candidate, ...]] = {}
    total = 0
    for slot in _SLOTS:
        if slot not in raw:
            continue
        entries = _slot_entries(slot, raw[slot])
        seen: set[str] = set()
        candidates: list[_Candidate] = []
        for name, cues in entries:
            label = sanitize_label(name)
            if label is None or label in seen:
                continue
            seen.add(label)
            if cues:
                patterns = tuple(
                    _compile_cue([cue], whole_word=whole_word) for cue in cues if str(cue).strip()
                )
            else:
                # No explicit cues ⇒ the type name IS the cue — in BOTH its written form and its
                # canonical form, as ONE alternation pattern. A config that writes
                # ``Payment_Product`` means the type, and the text says "Payment Product"; matching
                # only the literal written form would make the list form silently useless for
                # exactly the multi-word types it is most convenient for. One pattern (not two)
                # because they are two spellings of ONE cue — otherwise ``min_hits: 2`` could be
                # satisfied by a single piece of evidence spelled twice.
                patterns = (_compile_cue([name, label], whole_word=whole_word),)
            if not patterns:
                continue
            candidates.append(_Candidate(label=label, patterns=patterns))
            if len(candidates) >= ceiling:
                _LOG.info(
                    "lexicon inducer capped slot %s at the per-type safety ceiling (%d)",
                    slot,
                    ceiling,
                )
                break
        if candidates:
            compiled[slot] = tuple(candidates)
            total += len(candidates)
    if total == 0:
        msg = (
            "LexiconLabelInducer 'candidates' declares no usable types — every entry sanitized "
            "away. It could only ever induce nothing; fix the config rather than shipping a "
            "label-free run."
        )
        raise ValueError(msg)
    return compiled


def _slot_entries(slot: str, value: object) -> list[tuple[str, list[str]]]:
    """One slot's ``[(type, cues)]`` pairs from either the list form or the mapping form."""
    if isinstance(value, list):
        return [(str(name), []) for name in value]
    if isinstance(value, dict):
        entries: list[tuple[str, list[str]]] = []
        for name, cues in value.items():
            if cues is None:
                entries.append((str(name), []))
            elif isinstance(cues, list):
                entries.append((str(name), [str(c) for c in cues]))
            elif isinstance(cues, str):
                entries.append((str(name), [cues]))
            else:
                msg = (
                    f"LexiconLabelInducer candidate {slot}.{name!r} must map to a list of cue "
                    f"strings (or be omitted to use the type name itself), got "
                    f"{type(cues).__name__}."
                )
                raise ValueError(msg)
        return entries
    msg = (
        f"LexiconLabelInducer 'candidates.{slot}' must be a list of type names or a mapping of "
        f"type name -> cue list, got {type(value).__name__}."
    )
    raise ValueError(msg)


def _compile_cue(spellings: list[str], *, whole_word: bool) -> re.Pattern[str]:
    """Compile ONE surface cue — given as its interchangeable spellings — into a matcher.

    Every spelling is matched LITERALLY (``re.escape``) — an operator's config is not a regex
    surface, and treating it as one would turn a stray ``(`` in a real-world cue into a run-time
    error and a hand-written ``.*`` into an accidental catastrophic backtrack. The spellings are
    joined into ONE alternation so the result counts as a single piece of evidence for
    ``min_hits``. ``whole_word`` wraps the alternation in ``\\b`` boundaries, which is what stops a
    short cue like ``iban`` firing inside ``Ibanez``.
    """
    literals = [re.escape(s.strip()) for s in dict.fromkeys(spellings) if s.strip()]
    alternation = "|".join(literals)
    pattern = rf"\b(?:{alternation})\b" if whole_word else f"(?:{alternation})"
    return re.compile(pattern, re.IGNORECASE)

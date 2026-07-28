"""The corpus profiler: sample -> mine -> select -> report, with the uncertainty attached.

This is the piece D5 says is missing (DECISIONS.md:316-327). The architecture it slots into has
three steps and only this one did not exist:

1. **THIS** — sample the corpus, derive a corpus-level candidate type POOL by greedy submodular
   max-coverage over mention mass, emit it as a ``candidates`` mapping with surface cues. No LLM,
   one pass over a sample, zero network.
2. ``label_inducer.lexicon`` narrows that pool **per chunk**, so a chunk carries only the handful of
   types its own text evidences. This is what keeps a run inside GLiNER2's 20-30 uni-encoder ceiling
   (D5b) instead of firing all 40 labels at every chunk.
3. Extraction unions each chunk's induced set onto the config floor, with ADR-0044's financial-PII
   floor still non-bypassable.

The profiler emits a POOL, never a per-chunk schema. Emitting a corpus-level top-40 and applying it
to every chunk is precisely the defect the T1-T7 extraction redesign removed (D5a), so the artifact
says what N is a budget for, and the narrowing stays the inducer's job.

**What this artifact will NOT claim.** There is no published standard for a "90% coverage" or
"marginal gain < x%" label-budget cutoff. When the fitted rank exponent sits near 1, coverage grows
logarithmically in N and no sharp knee exists at all; in that regime the artifact states outright
that N is a budget decision rather than reporting a discovered boundary. Every number that would let
a reader disagree with that call - the gain curve, C(N), the certificate, the Zipf fit and the
Good-Turing residual - is in the output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from latence_label_profiler.artifact import render_artifact, to_json
from latence_label_profiler.coverage import (
    CoverageProblem,
    GreedyResult,
    greedy_max_coverage,
)
from latence_label_profiler.mining import (
    DEFAULT_MAX_CUES,
    DEFAULT_MAX_PHRASE_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    SLOTS,
    Candidate,
    mine_corpus,
)
from latence_label_profiler.stats import (
    Knee,
    ZipfFit,
    fit_zipf,
    good_turing_missing_mass,
    kneedle,
)
from latence_label_profiler.text import DEFAULT_STOPWORDS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: The label-count cap. 40 is the request's ceiling, NOT a measured optimum: GLiNER2 is a
#: uni-encoder whose author recommends 20 labels and whose docs say quality degrades past 30
#: (DECISIONS.md D5b), and no controlled accuracy-vs-label-count ablation exists in the literature.
#: T-LABEL-2 measures the real N on the pod; until then this is a budget, and the artifact says so.
DEFAULT_MAX_TYPES = 40


@dataclass(frozen=True)
class ProfilerConfig:
    """Every knob, all offline. ``seed`` fixes the sample; nothing else here is stochastic."""

    seed: int = 0
    #: Texts to mine. A profiler samples by construction — reading a 300 GB corpus to pick 40 labels
    #: would be a bad trade. Reservoir sampling keeps the draw uniform without a length probe.
    #: ``0`` or negative means NO cap: mine every text handed in.
    sample_size: int = 5000
    max_types: int = DEFAULT_MAX_TYPES
    #: Per-slot override of ``max_types``, e.g. ``{"relation_types": 20}``.
    slot_budgets: Mapping[str, int] = field(default_factory=dict)
    max_phrase_tokens: int = DEFAULT_MAX_PHRASE_TOKENS
    max_relation_tokens: int = DEFAULT_MAX_RELATION_TOKENS
    min_mention_mass: int = 3
    min_document_frequency: int = 2
    max_cues: int = DEFAULT_MAX_CUES
    stopwords: frozenset[str] = DEFAULT_STOPWORDS

    def budget_for(self, slot: str) -> int:
        return max(1, int(self.slot_budgets.get(slot, self.max_types)))


@dataclass(frozen=True)
class CorpusStats:
    """What was actually read, so a coverage number can never be read out of context."""

    texts_seen: int
    texts_sampled: int
    sample_seed: int
    sample_strategy: str


@dataclass(frozen=True)
class MethodStats:
    """The cost seam an operator compares against an LLM-induction run: all zeros, measured."""

    name: str = "latence-label-profiler"
    network_calls: int = 0
    model_id: None = None
    cost_usd: float = 0.0
    references: tuple[str, ...] = (
        "Rose, Engel, Cramer & Cowley (2010) - RAKE candidate phrases",
        "Fader, Soderland & Etzioni (2011) - ReVerb relation phrases",
        "Nemhauser, Wolsey & Fisher (1978) - greedy submodular (1 - 1/e) guarantee",
        "Feige (1998) - (1 - 1/e) is optimal for max-coverage unless P = NP",
        "Satopaa, Albrecht, Irwin & Raghavan (2011) - Kneedle knee detection",
        "Good (1953) - missing-mass estimate",
        "Clauset, Shalizi & Newman (2009) - discrete power-law MLE",
    )


@dataclass(frozen=True)
class SlotProfile:
    """One slot's selection plus everything needed to disbelieve it."""

    slot: str
    budget: int
    problem: CoverageProblem
    result: GreedyResult
    candidates: dict[str, Candidate]
    zipf: ZipfFit
    knee: Knee
    good_turing_unseen_mass: float
    notes: tuple[str, ...]

    @property
    def selected(self) -> tuple[Candidate, ...]:
        return tuple(self.candidates[label] for label in self.result.selected)

    @property
    def empirical_uncovered_mass(self) -> float:
        return max(0.0, 1.0 - self.result.coverage_fraction)

    @property
    def estimated_future_uncovered_mass(self) -> float:
        """What share of the NEXT corpus slice these labels would miss.

        ``p0`` of future mentions belong to types never observed (Good-Turing) and are uncovered by
        construction; of the remaining ``1 - p0``, the empirically uncovered share is missed too.
        Stated as a model rather than a measurement, because that is what it is.
        """
        p0 = self.good_turing_unseen_mass
        return min(1.0, p0 + (1.0 - p0) * self.empirical_uncovered_mass)


@dataclass(frozen=True)
class CorpusProfile:
    """The artifact: the candidate pool, and the profiler's own uncertainty about it."""

    corpus: CorpusStats
    method: MethodStats
    slots: dict[str, SlotProfile]
    notes: tuple[str, ...]
    config: ProfilerConfig

    def problem(self, slot: str) -> CoverageProblem:
        return self.slots[slot].problem

    def lexicon_candidates(self) -> dict[str, dict[str, list[str]]]:
        """The ``candidates`` block of a ``label_inducer.lexicon`` config, and nothing else.

        Exactly the shape ``LexiconLabelInducer`` parses: ``{slot: {type name: [surface cues]}}``,
        with every type name already in the sanitizer's canonical normal form so nothing is renamed
        between here and the inducer. A slot that selected nothing is omitted rather than emitted
        empty - an empty slot is a config error there, and a silently label-free run is worse than
        a loud one.
        """
        out: dict[str, dict[str, list[str]]] = {}
        for slot in SLOTS:
            profile = self.slots.get(slot)
            if profile is None or not profile.result.selected:
                continue
            out[slot] = {
                candidate.label: list(candidate.cues) for candidate in profile.selected
            }
        return out

    def to_dict(self) -> dict[str, object]:
        return render_artifact(self)

    def to_json(self) -> str:
        return to_json(self)


class CorpusProfiler:
    """Sample a corpus, mine a candidate type space, and select it under a cardinality budget."""

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self._config = config or ProfilerConfig()

    @property
    def config(self) -> ProfilerConfig:
        return self._config

    def profile(self, texts: Iterable[str]) -> CorpusProfile:
        config = self._config
        sample, seen = _reservoir_sample(texts, config.sample_size, config.seed)
        mined = mine_corpus(
            sample,
            stopwords=config.stopwords,
            max_phrase_tokens=config.max_phrase_tokens,
            max_relation_tokens=config.max_relation_tokens,
            min_mention_mass=config.min_mention_mass,
            min_document_frequency=config.min_document_frequency,
            max_cues=config.max_cues,
        )

        slots: dict[str, SlotProfile] = {}
        for slot in SLOTS:
            mining = mined.slots[slot]
            budget = config.budget_for(slot)
            problem = CoverageProblem(
                weights={surface: float(mass) for surface, mass in mining.surface_mass.items()},
                sets={
                    label: frozenset(candidate.covers)
                    for label, candidate in mining.candidates.items()
                },
            )
            result = greedy_max_coverage(problem, budget=budget)
            zipf = fit_zipf(mining.surface_mass)
            knee = kneedle(
                [step.rank for step in result.steps],
                [step.coverage_fraction for step in result.steps],
            )
            slots[slot] = SlotProfile(
                slot=slot,
                budget=budget,
                problem=problem,
                result=result,
                candidates=dict(mining.candidates),
                zipf=zipf,
                knee=knee,
                good_turing_unseen_mass=good_turing_missing_mass(mining.surface_mass),
                notes=_slot_notes(slot, result, zipf, knee),
            )

        corpus = CorpusStats(
            texts_seen=seen,
            texts_sampled=len(sample),
            sample_seed=config.seed,
            sample_strategy="reservoir" if seen > len(sample) else "full",
        )
        return CorpusProfile(
            corpus=corpus,
            method=MethodStats(),
            slots=slots,
            notes=_corpus_notes(slots),
            config=config,
        )


def _reservoir_sample(texts: Iterable[str], size: int, seed: int) -> tuple[list[str], int]:
    """Algorithm R, seeded. Uniform over the stream without needing its length in advance.

    Deterministic for a fixed ``(stream order, seed)``, which is the whole determinism contract:
    the same corpus in the same order with the same seed yields the same sample, hence the same
    bytes out.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for text in texts:
        seen += 1
        if size <= 0 or len(reservoir) < size:
            reservoir.append(text)
            continue
        index = rng.randrange(seen)
        if index < size:
            reservoir[index] = text
    return reservoir, seen


def _slot_notes(slot: str, result: GreedyResult, zipf: ZipfFit, knee: Knee) -> tuple[str, ...]:
    notes: list[str] = []
    if not result.selected:
        notes.append(
            f"{slot}: no candidate type cleared the mass / document-frequency floors, so this slot "
            "contributes nothing to the induction pool."
        )
        return tuple(notes)
    if not knee.sharp:
        notes.append(
            f"{slot}: no sharp knee (Kneedle difference peak "
            f"{knee.max_difference:.3f} < 0.5). N is a budget decision, not a discovered boundary."
        )
    if zipf.log_growth:
        notes.append(
            f"{slot}: the fitted rank exponent is {zipf.rank_slope:.2f}, i.e. approximately "
            "Zipfian with alpha ~ 1, so coverage grows only logarithmically in N. Doubling N "
            "buys a constant, shrinking increment; there is no N at which the pool suffices."
        )
    if result.certificate_ratio >= 0.999:
        notes.append(
            f"{slot}: the certificate is exact - this selection is provably optimal at N = "
            f"{len(result.selected)} on this corpus, not merely within (1 - 1/e) of it."
        )
    if slot == "relation_types":
        notes.append(
            "relation_types: mined by a POS-tagger-free ReVerb heuristic (closed verb set + "
            "conservative suffix morphology). Recall in this slot is bounded by that heuristic, so "
            "a thin pool here is evidence about the miner, not necessarily about the corpus."
        )
    if slot == "pii_types":
        notes.append(
            "pii_types: candidates are DECLARED (a fixed table), only their mass is measured, and "
            "their coverage sets are disjoint - so greedy is exactly optimal on this slot by "
            "construction and its certificate ratio of 1.0 is not evidence of a good corpus fit."
        )
    return tuple(notes)


def _corpus_notes(slots: dict[str, SlotProfile]) -> tuple[str, ...]:
    notes: list[str] = []
    if not any(profile.result.selected for profile in slots.values()):
        notes.append(
            "no candidate types were selected in any slot: the corpus sample was empty or every "
            "candidate fell below the mass / document-frequency floors."
        )
    if any(not profile.knee.sharp and profile.result.selected for profile in slots.values()):
        notes.append(
            "at least one slot has no sharp knee in its coverage curve, so N there is a budget "
            "decision, not a discovered boundary. There is no published standard for a coverage or "
            "marginal-gain cutoff; the gain curve and C(N) are reported so the budget can be set "
            "deliberately rather than justified after the fact."
        )
    detector_only = sorted(
        candidate.label
        for profile in slots.values()
        for candidate in profile.selected
        if candidate.detector_only
    )
    if detector_only:
        notes.append(
            "detector-only PII types selected ("
            + ", ".join(detector_only)
            + "): their mention mass was measured by a structural detector, but the emitted cues "
            "are contextual keywords with ZERO observed support in this sample. Per-chunk "
            "induction will fire on a chunk that TALKS about the type, not on one that merely "
            "carries a bare value. Bare values are covered by the ADR-0044 redaction floor, not by "
            "this pool."
        )
    notes.append(
        "the pool is CORPUS-level. Narrowing it per chunk is label_inducer.lexicon's job; applying "
        "the whole pool to every chunk is the defect the per-chunk extraction redesign removed."
    )
    notes.append(
        "the Good-Turing residual estimates the share of FUTURE mentions belonging to types this "
        "sample never observed. A high C(N) over a sample with a high residual is not a high "
        "coverage of the corpus."
    )
    return tuple(notes)

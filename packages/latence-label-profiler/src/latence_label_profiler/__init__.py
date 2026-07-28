"""latence-label-profiler: the non-LLM corpus profiler behind ``label_inducer.lexicon``.

The request that produced this package forbids LLM-induced labels. The naive way to satisfy that -
sample the corpus, take the top 40 types, hand all 40 to every chunk - is *the same defect* the
per-chunk extraction redesign removed, because GLiNER2 is a uni-encoder whose author recommends 20
labels and whose docs report degradation past 30 (DECISIONS.md D5a/D5b).

So this package does exactly half the job, on purpose. It emits a corpus-level candidate **pool**;
the already-shipped ``label_inducer.lexicon`` narrows that pool **per chunk**. One module produces a
type space, the other decides which of those types a given chunk actually evidences, and neither
ever sends a token anywhere.

Everything here is pure-Python, dependency-free, deterministic and offline:

* :class:`~latence_label_profiler.profiler.CorpusProfiler` - the entry point.
* :func:`~latence_label_profiler.profiler.CorpusProfile.lexicon_candidates` - the ``candidates``
  block of a ``label_inducer.lexicon`` config, ready to paste into a stack.
* :func:`~latence_label_profiler.profiler.CorpusProfile.to_json` - the artifact, which reports the
  marginal-gain curve, the per-corpus optimality certificate, ``C(N)``, the fitted Zipf exponent and
  the Good-Turing residual, and which says plainly when N is a budget decision rather than a
  discovered boundary.
"""

from __future__ import annotations

from latence_label_profiler.artifact import ARTIFACT_VERSION
from latence_label_profiler.coverage import (
    WORST_CASE_RATIO,
    CoverageProblem,
    GreedyResult,
    GreedyStep,
    greedy_max_coverage,
)
from latence_label_profiler.profiler import (
    DEFAULT_MAX_TYPES,
    CorpusProfile,
    CorpusProfiler,
    ProfilerConfig,
    SlotProfile,
)

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_MAX_TYPES",
    "WORST_CASE_RATIO",
    "CorpusProfile",
    "CorpusProfiler",
    "CoverageProblem",
    "GreedyResult",
    "GreedyStep",
    "ProfilerConfig",
    "SlotProfile",
    "greedy_max_coverage",
]
__version__ = "0.1.0"

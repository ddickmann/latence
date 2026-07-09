"""``latence-linkpred-ultra`` — the learned graph-completion Provider (``graph_completion.ultra``).

A drop-in on the Slice-1 :class:`~latence_core.capability.GraphCompleter` seam (ADR-0037): an
inductive **ULTRA** (ICLR'24, MIT) zero-shot scorer + a **PyKEEN** (MIT) graceful fallback behind a
:class:`~latence_linkpred_ultra.scorer.Scorer` abstraction, a type-aware tiered 2-hop
:class:`~latence_linkpred_ultra.candidates.CandidateGenerator` (bounded, rare-relation recall), a
temperature-scaling + precision-targeted :class:`~latence_linkpred_ultra.calibrator.Calibrator`, and
held-out MRR / Hits@k with leak detection (:mod:`~latence_linkpred_ultra.evaluation`). It emits
inferred edges via the Slice-1 inferred-edge contract EXACTLY.

The provider (:class:`~latence_linkpred_ultra.provider.UltraGraphCompleter`) is registered under the
``latence.providers`` entry-point group as ``graph_completion.ultra``. Heavy deps
(torch / torch-geometric / torch-scatter / easydict / pykeen) are isolated in this package
(ADR-0016)
and lazily imported, so provider discovery stays torch-free.
"""

from latence_linkpred_ultra.calibrator import Calibration, Calibrator, CalibratorConfig
from latence_linkpred_ultra.candidates import (
    Candidate,
    CandidateConfig,
    CandidateGenerator,
)
from latence_linkpred_ultra.candidates import ObservedEdge as CandidateObservedEdge
from latence_linkpred_ultra.evaluation import (
    LeakDetectionError,
    RankingMetrics,
    assert_no_leak,
    metrics_from_ranks,
    split_edges,
)
from latence_linkpred_ultra.scorer import (
    BuiltKG,
    PykeenScorer,
    Scorer,
    ScorerUnavailableError,
    UltraScorer,
)

__all__ = [
    "BuiltKG",
    "Calibration",
    "Calibrator",
    "CalibratorConfig",
    "Candidate",
    "CandidateConfig",
    "CandidateGenerator",
    "CandidateObservedEdge",
    "LeakDetectionError",
    "PykeenScorer",
    "RankingMetrics",
    "Scorer",
    "ScorerUnavailableError",
    "UltraScorer",
    "assert_no_leak",
    "metrics_from_ranks",
    "split_edges",
]

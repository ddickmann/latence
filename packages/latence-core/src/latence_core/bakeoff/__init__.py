"""Bake-off / Provider Matrix harness (ADR-0036 §4, P3-F5) — Goal B, the Wave-0 capstone.

The per-Stage **bake-off**: hold every OTHER Stage at ``stacks/default.yaml`` and run the SAME
bundled corpus (+ gold-set) through each candidate Provider for ONE Stage, producing a comparison
an enterprise can act on — quality, latency/throughput, peak memory (measured vs declared), cost,
license, determinism, error rate (ADR-0036 §4/§6). This retires the last audit thread: it is the
real consumer of ``ProviderProfile.memory_mb`` / ``cost_per_1k`` / ``deterministic``, so those
fields stop being speculative.

Three cohesive pieces, all dependency-light where they can be (the contracts import no
runner/stages; the harness lazy-imports the heavy runner at call time so ``import latence_core``
stays thin):

* :mod:`~latence_core.bakeoff.matrix` — the :class:`~latence_core.bakeoff.matrix.MatrixSpec`
  contract that loads ``matrix/<stage>.yaml`` (target Stage + candidate Providers).
* :mod:`~latence_core.bakeoff.report` — the :class:`~latence_core.bakeoff.report.MatrixResult` +
  :class:`~latence_core.bakeoff.report.CandidateResult` result contracts + their signed-style
  ``MATRIX-RESULTS-<stage>.md`` rendering + JSON companion.
* :mod:`~latence_core.bakeoff.harness` — :func:`~latence_core.bakeoff.harness.run_bakeoff`: the
  composition over the F4 stack machinery + the S10 gold-set scorer + the H-E1 equality helper +
  the ``select_device`` seam that runs each candidate and measures the columns.
"""

from __future__ import annotations

from latence_core.bakeoff.harness import BakeOffHarness, run_bakeoff
from latence_core.bakeoff.matrix import (
    MATRIX_SCHEMA_VERSION,
    MatrixCandidate,
    MatrixSpec,
)
from latence_core.bakeoff.report import (
    BAKEOFF_SCHEMA_VERSION,
    CandidateResult,
    DeterminismColumn,
    LicenseColumn,
    MatrixResult,
    MemoryMeasurement,
    QualityMetric,
)

__all__ = [
    "BAKEOFF_SCHEMA_VERSION",
    "MATRIX_SCHEMA_VERSION",
    "BakeOffHarness",
    "CandidateResult",
    "DeterminismColumn",
    "LicenseColumn",
    "MatrixCandidate",
    "MatrixResult",
    "MatrixSpec",
    "MemoryMeasurement",
    "QualityMetric",
    "run_bakeoff",
]

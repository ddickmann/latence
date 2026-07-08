"""Stacks: full Pipeline configs + the ``latence stack validate`` harness (ADR-0036 §3, P3-F4).

A *stack* is a complete, runnable Pipeline config for the whole spine
(Source → Intake-Screening → Parse → Chunk → Content-Screening → NER → RE → Redaction →
Profiling → Disambiguation → Graph-Assembly → Export), distinct from the per-stage
``examples/*.yaml`` slices. The blessed default (``stacks/default.yaml``) is CPU-first,
offline, permissive, and deterministic (ADR-0007/0012); ``stacks/gpu-sota.yaml`` names the
learned/endpoint Providers that are **skipped-with-flag** on a CPU-only host (the P3-F1 device
posture) rather than faked.

This package holds the **harness**, not new pipeline logic — it is composition over the existing
``LocalRunner`` + ``QualityReportBuilder`` + conformance seams. The two public entry points:

* :func:`~latence_core.stacks.corpus.stage_validation_corpus` — write the small, bundled,
  deterministic messy corpus (entity-rich + deliberately dangerous fixtures) onto any Storage.
* :func:`~latence_core.stacks.validate.validate_stack` — run a named stack end-to-end and assert
  the G1-style checks (contracts/KG/RAG/report + determinism + resume + graceful-failure +
  device-honesty), returning a :class:`~latence_core.stacks.validate.StackValidationResult`.

The heavy work is lazy-imported so ``import latence_core`` stays thin (the harness pulls in the
runner + stages only when a validation actually runs).
"""

from __future__ import annotations

from latence_core.stacks.corpus import (
    ValidationCorpusManifest,
    build_validation_corpus_bytes,
    stage_validation_corpus,
)
from latence_core.stacks.validate import (
    CheckResult,
    StackValidationResult,
    validate_stack,
)

__all__ = [
    "CheckResult",
    "StackValidationResult",
    "ValidationCorpusManifest",
    "build_validation_corpus_bytes",
    "stage_validation_corpus",
    "validate_stack",
]

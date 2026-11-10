"""The PHASE-BOUNDARY invariant: the front half is chunk-level, corpus-level work comes after (T1).

The extraction pipeline has two phases and they are not interchangeable:

    source → parse → chunk → [content_screen] → induce → extract → redact     ← the FRONT half
    ────────────────────────── every stage here is per-CHUNK or per-DOCUMENT ──────────────────
    profiling → consolidate_types → disambiguate → graph → complete → export  ← the BACK half
    ─────────────────────── the first corpus-wide operations, and the only ones ───────────────

**``redact`` is the LAST chunk-level stage.** Everything before it sees one chunk (or one document)
at a time; the first corpus-level operation is the resolution / corpus-building phase.

Why this is an invariant and not a style preference: a corpus-level stage in the front half has to
COLLAPSE the corpus into one shared artifact and stamp it back onto every chunk, and the only way to
bound that artifact is to CAP it. That cap is where recall goes to die — the type a corpus mentions
in exactly one chunk ranks below the salience cut and is dropped from the one chunk that needed it,
invisibly. It is also where zero-shot precision goes: a chunk scored against a ~40-type corpus union
competes against ~35 types that are not in it. (And it is where the gliner2 CUDA-OOM came from: the
span-scoring tensor is ``batch × tokens × labels``, and ``labels`` was corpus-wide.) The consistency
such a stage buys is real, but it belongs DOWNSTREAM — earned in the resolution phase by canonical-
izing entities and the type vocabulary corpus-wide, where it costs nothing. Consistency is earned
downstream, never imposed upstream by capping.

The check is the mechanical statement of that: **no chunk-level stage may transitively depend on a
corpus-level stage.** Since ``redact`` is chunk-level, this is exactly "no corpus-level operation
runs before redaction" — expressed over the DAG rather than over a topological schedule, so it holds
for every valid execution order rather than for the one the runner happens to pick.

It is a pure, static, dependency-free predicate over a :class:`~latence_core.pipeline.Pipeline`: no
run, no providers, no I/O. ``latence stack validate`` reports it as a check and ``latence setup``'s
dry validation refuses to emit a config that violates it.

**Where the levels come from.** This module used to keep its own two hand-written kind sets, and
they drifted: ``TYPE_CONSOLIDATION`` was added to the Runner's and the DeltaRunner's corpus-level
tuples but to neither set here, so :func:`is_corpus_level` answered *False* for a Stage that
collapses the whole run's mentions into one canonical type vocabulary and this check waved a
front-half ``consolidate_types`` straight through. (The chunk-level set was worse: no code read it
at all — documentation posing as a guard.) A Capability's level is a fact about the **Capability**,
so it is declared once on its :class:`~latence_core.capability_descriptor.CapabilityDescriptor` and
imported here. A new Capability that does not declare its level now fails at import instead of
silently defaulting to chunk-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from latence_core.capability_descriptor import (
    CORPUS_LEVEL_KINDS,
    DOCUMENT_LEVEL_KINDS,
    is_corpus_level,
)

if TYPE_CHECKING:
    from latence_core.pipeline import CapabilityKind, Pipeline, Stage

__all__ = [
    "CHUNK_LEVEL_KINDS",
    "CORPUS_LEVEL_KINDS",
    "PhaseBoundaryViolation",
    "check_phase_boundary",
    "is_corpus_level",
]

#: The Capability kinds that are CHUNK- or DOCUMENT-level: they see one unit at a time and never
#: aggregate. These are the front half, and the invariant is about what may feed them. Kept as an
#: alias of the descriptor-derived :data:`~latence_core.capability_descriptor.DOCUMENT_LEVEL_KINDS`,
#: which is the exact COMPLEMENT of :data:`CORPUS_LEVEL_KINDS` over the DAG kinds — so no kind can
#: land in neither, which is precisely how the corpus-collapsing Stage slipped through.
#: ``SCHEMA_INDUCTION`` is in it because per-chunk / per-document induction is chunk-level — it is
#: the ONE kind whose level depends on its config (``granularity: corpus`` makes that single stage a
#: corpus-level operation), which is why :func:`is_corpus_level` reads the Stage, not the kind.
CHUNK_LEVEL_KINDS: frozenset[CapabilityKind] = DOCUMENT_LEVEL_KINDS


@dataclass(frozen=True)
class PhaseBoundaryViolation:
    """One chunk-level Stage that (transitively) consumes a corpus-level Stage's output.

    ``path`` is the dependency chain from the corpus-level stage down to the chunk-level one
    (``("induce", "extract", "redact")`` reads "redact depends on extract depends on induce"), so
    the message points at the wiring to fix rather than only at the two endpoints.
    """

    stage: str
    capability: str
    corpus_stage: str
    corpus_capability: str
    path: tuple[str, ...]

    def describe(self) -> str:
        """A single-line, actionable message naming both stages and the chain between them."""
        chain = " -> ".join(self.path)
        return (
            f"chunk-level stage {self.stage!r} ({self.capability}) depends on corpus-level stage "
            f"{self.corpus_stage!r} ({self.corpus_capability}) via {chain} — the front half must "
            f"stay chunk/document-level; move corpus-wide work after redaction"
        )


def check_phase_boundary(pipeline: Pipeline) -> list[PhaseBoundaryViolation]:
    """Every corpus-level Stage scheduled BEFORE a chunk-level one; empty ⇒ the DAG holds.

    Walks each chunk-level Stage's transitive dependencies and reports the corpus-level ancestors
    it finds. Because it reasons over the DAG (not over one topological order), a reported
    violation is a violation under EVERY valid execution order, and a clean result means no
    scheduler can put corpus-level work in the front half.

    Deterministic: violations come back in Stage-declaration order, then by the ancestor's own
    declaration order, and the reported ``path`` is the shortest dependency chain (breadth-first),
    so the same Pipeline always yields byte-identical messages.
    """
    by_name = {stage.name: stage for stage in pipeline.stages}
    order = {stage.name: index for index, stage in enumerate(pipeline.stages)}
    violations: list[PhaseBoundaryViolation] = []
    for stage in pipeline.stages:
        if is_corpus_level(stage):
            continue
        found: list[PhaseBoundaryViolation] = []
        for ancestor, path in _shortest_paths_to_ancestors(stage, by_name):
            if not is_corpus_level(ancestor):
                continue
            found.append(
                PhaseBoundaryViolation(
                    stage=stage.name,
                    capability=str(stage.capability),
                    corpus_stage=ancestor.name,
                    corpus_capability=str(ancestor.capability),
                    path=path,
                )
            )
        violations.extend(sorted(found, key=lambda v: order.get(v.corpus_stage, 0)))
    return violations


def _shortest_paths_to_ancestors(
    stage: Stage, by_name: dict[str, Stage]
) -> list[tuple[Stage, tuple[str, ...]]]:
    """Every transitive dependency of ``stage`` with the shortest dependency chain reaching it.

    Breadth-first over ``depends_on`` so the reported chain is the shortest one (the most readable
    explanation of the wiring). A ``depends_on`` naming a stage that does not exist is ignored here
    — the Pipeline's own DAG validation owns that error, and this check must not shadow it with a
    confusing phase-boundary message.
    """
    reached: dict[str, tuple[str, ...]] = {}
    frontier: list[tuple[str, tuple[str, ...]]] = [
        (dep, (dep, stage.name)) for dep in stage.depends_on
    ]
    while frontier:
        name, path = frontier.pop(0)
        if name in reached or name not in by_name:
            continue
        reached[name] = path
        for dep in by_name[name].depends_on:
            frontier.append((dep, (dep, *path)))
    return [(by_name[name], path) for name, path in reached.items()]

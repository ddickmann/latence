"""StageInputs — one Stage's input, resolved once, behind one small interface (ADR-0033).

"What does Graph Assembly receive?" used to be a four-place question. The executor's ``_run``
loop had a four-way branch that picked the ``incoming`` shape and the ``records_in`` counting rule;
two hand-maintained frozensets that the :class:`~latence_core.capability_descriptor.\
CapabilityDescriptor` did not derive named which kinds went down which arm; the dispatch then
re-decided the same shape a SECOND time (``gathered = [] if capability in _STREAMED else
list(incoming)``) so it would not double-read a one-shot generator; and the
wiring rules lived in a family of ~10 near-identical ``_require_*`` methods, each transcribing its
own error text — three of which were dead.

The shape and the rules are facts about the Capability, so they are declared with it
(:class:`~latence_core.capability_descriptor.StageInputPlan`). This module is the other half: the
object the Runner builds from that plan and the dispatch *queries*.

    inputs = runner.stage_inputs(stage, outputs, storage=…, pipeline=…, run_id=…)
    records_in = inputs.count()            # per the plan's shape, never a materialised list
    …
    inputs.validate()                      # the plan's wiring rules, the plan's exact messages
    chunks = inputs.stream_of(ChunkRecord) # one lazy carrier stream, overlay folded per ADR-0061

What this buys, beyond one home for the table:

* **the dispatch stops re-deciding.** A branch asks for the shape it needs (:meth:`gathered`,
  :meth:`stream`, :meth:`stream_of`, :meth:`corpus_records_of`) and gets it. The one-shot
  double-read hazard the second decision existed to prevent is structurally gone: every accessor
  re-reads from the record source, so :meth:`gathered` is idempotent rather than destructive.
* **the seam narrows.** :class:`~latence_core.stage_execution.StageContext` used to carry the
  Runner's whole live ``outputs: dict[str, list[StageOutput]]`` — executor-internal state a Stage
  has no business seeing — purely so six dispatch branches could re-open per-carrier streams from
  it. It now carries this object; the substrate lives behind :class:`StageRecordSource`.
* **it is testable without a run.** A fake :class:`StageRecordSource` (or a fake ``StageInputs``)
  drives any dispatch branch, which the old form — reachable only by executing a Pipeline — could
  not offer.

Scope: this module resolves and validates a Stage's input. It performs no I/O of its own (the
source does), constructs no Provider, and decides nothing the descriptor does not declare. The
cross-run reuse fold is likewise not re-decided here — WHERE the overlay folds, in what order and
what happens to an unclaimed one is :class:`~latence_core.reuse_overlay.ReuseOverlay`'s rule
(Wave 3-G); this module supplies each site's enumeration and its trailer policy, exactly as the
Runner's six hand-written gathers did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from latence_core.capability_descriptor import StageInputShape, input_plan_for
from latence_core.reuse_overlay import OverlayTrailer, ReuseOverlay

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.capability_descriptor import StageInputPlan
    from latence_core.contracts import Record
    from latence_core.pipeline import Pipeline, Stage
    from latence_core.runner import StageOutput

#: One inter-Stage carrier a caller narrows to (the same bound the Runner's helpers used).
_T = TypeVar("_T", bound="StageOutput")

__all__ = ["StageInputs", "StageRecordSource"]


class StageRecordSource(Protocol):
    """Where a run's per-Stage records come from — the substrate half of :class:`StageInputs`.

    Deliberately three methods and no more. The local Runner's implementation reads the live
    retained ``outputs`` dict and falls back to the Stage's on-disk checkpoint (#69); a Runner on a
    materialised substrate (Airflow's XCom) reads its own map. Neither shape reaches
    :class:`StageInputs`, which is why the same input resolution serves both Runners.

    ``records`` and ``count`` are the read/count pair over ONE Stage's output — ``count`` exists
    separately because a released Stage can be counted from its checkpoint's LINES without
    decoding (and therefore without re-validating) a single record. ``carrier`` is the record type
    a Stage checkpoints, needed by the wiring check to see a carrier whose fresh output is empty
    because every one of its documents was reused.
    """

    def records(self, stage_name: str) -> Iterator[StageOutput]:
        """One Stage's output records, lazily, in emission order."""
        ...  # pragma: no cover - Protocol

    def count(self, stage_name: str) -> int:
        """How many records one Stage produced — without decoding them where possible."""
        ...  # pragma: no cover - Protocol

    def carrier(self, stage: Stage) -> type[StageOutput] | type[Record]:
        """The record type a Stage checkpoints (``Record`` for a mixed-carrier Stage)."""
        ...  # pragma: no cover - Protocol


class StageInputs:
    """One Stage's input: what it is, how much of it there is, and whether it is wired right.

    Built by the Runner (one per Stage, per run) and queried by the dispatch. Every accessor
    honours the Stage's declared :class:`~latence_core.capability_descriptor.StageInputPlan`, so
    the Capability decides the shape and the dispatch never re-derives it.
    """

    def __init__(
        self,
        stage: Stage,
        pipeline: Pipeline,
        source: StageRecordSource,
        *,
        overlay: ReuseOverlay | None = None,
        plan: StageInputPlan | None = None,
    ) -> None:
        self._stage = stage
        self._pipeline = pipeline
        self._source = source
        # W17/ADR-0043: empty for a non-delta run, which makes every fold below the identity.
        self._overlay = overlay if overlay is not None else ReuseOverlay()
        self._plan = plan if plan is not None else input_plan_for(stage.capability)
        self._by_name = {s.name: s for s in pipeline.stages}
        self._gathered: list[StageOutput] | None = None

    @property
    def stage(self) -> Stage:
        """The Stage whose input this is."""
        return self._stage

    @property
    def plan(self) -> StageInputPlan:
        """The Capability's declared input plan — the shape and the wiring rules, as data."""
        return self._plan

    @property
    def source(self) -> StageRecordSource:
        """The record substrate this reads through — a Runner's own, never the dispatch's."""
        return self._source

    @property
    def materialised(self) -> int | None:
        """How many records this has materialised into a list, or ``None`` if it never has.

        The memory-shaping fact ADR-0033 bounds, readable rather than inferred: a STREAMED /
        ACCUMULATOR Stage must never reach an ``int`` here, because no list of its input exists
        at any corpus size.
        """
        return None if self._gathered is None else len(self._gathered)

    # -- the shapes ---------------------------------------------------------

    def gathered(self) -> list[StageOutput]:
        """The parents' records concatenated into one list, in declared ``depends_on`` order.

        What a :attr:`~latence_core.capability_descriptor.StageInputShape.GATHERED` Stage receives
        — the Capabilities whose Provider materialises the input anyway. Deterministic (a
        converging DAG ``{a,b} -> c`` sees a's records then b's) and release-transparent: a parent
        whose retained output was already dropped is streamed back from its checkpoint (#69).

        Cached, so the count and the dispatch share one list; idempotent, so a second caller can
        never read an exhausted stream (which is precisely what the dispatch's second shape
        decision existed to prevent).
        """
        if self._gathered is None:
            gathered: list[StageOutput] = []
            for parent in self._stage.depends_on:
                gathered.extend(self._source.records(parent))
            self._gathered = gathered
        return self._gathered

    def stream(self) -> Iterator[StageOutput]:
        """The same records as :meth:`gathered`, lazily — what a STREAMED Stage receives.

        Same records, same order; nothing is concatenated. The Stage's Provider pulls one record at
        a time, so the input side costs O(1) records instead of O(corpus-slice) (audit R2). ONE
        SHOT by nature: a Provider consumes it exactly once, which is the property that makes the
        shape sound.

        Overlay-blind on purpose: the W17 reuse overlay is folded into the corpus-level gathers
        (:meth:`stream_of`, :meth:`corpus_records_of`), never into a doc-level Stage's input — a
        reused document must not be re-parsed or re-extracted.
        """
        for parent in self._stage.depends_on:
            yield from self._source.records(parent)

    def stream_of(self, expected: type[_T]) -> Iterator[_T]:
        """One carrier of this Stage's parents' output, lazily, with the reuse overlay folded in.

        The per-carrier read an ACCUMULATOR (Profiling/Type-Consolidation/Disambiguation/Graph)
        does instead of receiving one gathered list: the
        parents are re-read in declared ``depends_on`` order and only ``expected`` is yielded. A
        checkpoint is re-readable, so N carriers cost N O(1)-memory passes rather than one
        O(corpus) list, and the order is exactly the gathered order filtered — byte-identical
        records in a byte-identical sequence.

        W17/ADR-0061: the overlay of this carrier is MERGED document-interleaved into the FIRST
        doc-level parent stream that produces it, so the consumer folds ``fresh ∪ reused`` in the
        full rebuild's arrival order. The trailer policy for an overlay no parent stream claimed
        follows the shape:

        * ACCUMULATOR → ``APPEND_IF_CARRIER_UNSEEN``. An accumulator's contract is union semantics,
          so an overlay carrier no parent produces is still delivered — unless a CORPUS-LEVEL
          parent produces it, whose checkpoint already contains the union.
        * any other shape → ``NEVER``. A doc-level Stage's input is its parents' records, never
          records no parent produced.
        """
        trailer = (
            OverlayTrailer.APPEND_IF_CARRIER_UNSEEN
            if self._plan.shape is StageInputShape.ACCUMULATOR
            else OverlayTrailer.NEVER
        )
        sources = (
            (self._by_name.get(parent), self._records_of_type(parent, expected))
            for parent in self._stage.depends_on
        )
        yield from self._overlay.fold_into(
            expected, sources, self._pipeline, trailer=trailer
        )

    def corpus_records_of(self, expected: type[_T]) -> Iterator[_T]:
        """Every Stage's output narrowed to one carrier — corpus-wide, not just parents (#94).

        A corpus-level Stage sometimes needs a carrier that is not one of its ``depends_on``
        parents: a context-embedding Disambiguator needs the run's ChunkRecords, produced by the
        Chunk Stage — a *mention ancestor*, not a parent. This streams every Stage's output in
        topological order (deterministic), one record at a time, yielding only ``expected``; a run
        with no producing Stage yields nothing.

        W17/ADR-0061: the trailer is ``ALWAYS`` here — the enumeration spans the whole pipeline, so
        an overlay no stream claimed IS the corpus's only records of that carrier.
        """
        # Imported here rather than at module scope: ``topological_order`` lives with the executor
        # that owns scheduling, and the executor imports this module. Deferring the import keeps
        # the dependency one-directional at import time without duplicating Kahn's algorithm.
        from latence_core.runner import topological_order

        sources = (
            (stage, self._records_of_type(stage.name, expected))
            for stage in topological_order(list(self._pipeline.stages))
        )
        yield from self._overlay.fold_into(
            expected, sources, self._pipeline, trailer=OverlayTrailer.ALWAYS
        )

    # -- the count ----------------------------------------------------------

    def count(self) -> int:
        """``records_in`` for this Stage — measured the way its shape allows, never by holding it.

        The metric the span, the lineage START event and the ``StageMetrics`` row all report is
        needed BEFORE the Stage executes, so it cannot be a tally taken while the stream flows.
        Per shape:

        * GATHERED → the length of the list the Stage is about to receive anyway.
        * STREAMED → the parents' checkpoint LINE counts (one record per line by
          construction of the checkpoint writer), so the count is exact without decoding — and
          therefore without re-validating — a single record.
        * ACCUMULATOR → the parents' records counted by streaming, plus the reuse overlay's
          contribution where the fold rule delivers it: once per carrier, at the first doc-level
          parent producing it (a corpus-level parent's checkpoint already contains the union, so
          its line count needs no top-up), with a mixed-carrier parent counting both mention
          overlays. That re-derivation is the overlay's own (:meth:`~latence_core.reuse_overlay.\
ReuseOverlay.count_for_stage`).
        """
        shape = self._plan.shape
        if shape is StageInputShape.GATHERED:
            return len(self.gathered())
        if shape is StageInputShape.ACCUMULATOR:
            return self._accumulator_count()
        return sum(self._source.count(parent) for parent in self._stage.depends_on)

    def _accumulator_count(self) -> int:
        total = 0
        counted: set[type[Record]] = set()
        for parent in self._stage.depends_on:
            for _ in self._source.records(parent):
                total += 1
            parent_stage = self._by_name.get(parent)
            if parent_stage is not None:
                total += self._overlay.count_for_stage(
                    parent_stage, self._pipeline, already_counted=counted
                )
        return total

    # -- the wiring check ---------------------------------------------------

    def validate(self) -> None:
        """Raise the plan's typed wiring error if this Stage's input does not satisfy its rules.

        The ~10 ``_require_*`` methods, as one traversal over declared data. Same verdicts, same
        messages, same reads:

        * a plan with no rules reads NOTHING (most Capabilities: the carrier narrowing is their
          only check), so calling this uniformly from every dispatch branch costs them nothing;
        * an EMPTY input is legitimately empty — an empty run has no documents, therefore no
          chunks and no mentions, and that is not a wiring mistake;
        * required carriers are checked in declaration order, so a Stage missing both sides names
          the same one it always did;
        * an :class:`~latence_core.capability_descriptor.ExclusiveCarrier` rule is checked last and
          rejects every other carrier present — the streamed form of the eager ``_as_type``
          narrowing it replaced.
        """
        plan = self._plan
        if not plan.validates:
            return
        present, names = self.carriers_present()
        if not present:
            return
        for requirement in plan.requires:
            if not requirement.satisfied_by(present):
                raise requirement.error(self._stage, names)
        if plan.exclusive is not None:
            wrong = plan.exclusive.violations(present)
            if wrong:
                raise plan.exclusive.error(self._stage, wrong)

    def carriers_present(self) -> tuple[frozenset[type], frozenset[str]]:
        """The carrier TYPES this Stage's input carries (and their names) — bounded, never a list.

        O(distinct carriers) memory rather than O(records): the wiring check needs to know *which
        carriers arrived*, not to hold them. A GATHERED Stage reads the list it already has; every
        other shape streams the parents once.

        One detail is load-bearing — **the overlay top-up.** A parent whose fresh output is empty
        because every one of its documents was reused still "produces" its carrier through the
        overlay, so an all-reused delta is not misread as a mis-wired empty parent (W17).
        """
        if self._plan.shape is StageInputShape.GATHERED:
            records = self.gathered()
            return (
                frozenset(type(record) for record in records),
                frozenset(type(record).__name__ for record in records),
            )
        present: set[type[StageOutput]] = set()
        names: set[str] = set()
        for parent in self._stage.depends_on:
            for record in self._source.records(parent):
                present.add(type(record))
                names.add(type(record).__name__)
            parent_stage = self._by_name.get(parent)
            if parent_stage is not None:
                parent_type = self._source.carrier(parent_stage)
                if self._overlay.has_carrier(parent_type):
                    present.add(parent_type)  # type: ignore[arg-type]
                    names.add(parent_type.__name__)
        return frozenset(present), frozenset(names)

    # -- internals ----------------------------------------------------------

    def _records_of_type(self, stage_name: str, expected: type[_T]) -> Iterator[_T]:
        """One Stage's output narrowed to one carrier, lazily — the streamed ``isinstance`` filter.

        The per-Stage stream every overlay-folding gather hands to
        :meth:`~latence_core.reuse_overlay.ReuseOverlay.fold_into`: identical records in identical
        order to ``[r for r in gathered if isinstance(r, expected)]``, without the list.
        """
        return (
            record
            for record in self._source.records(stage_name)
            if isinstance(record, expected)
        )

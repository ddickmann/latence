"""QualityReportBuilder — the Quality Report half of the Runner, as its own unit (H-F1).

The Quality Report is a first-class run artifact (ADR-0008): per-Stage metrics, corpus-level
KG statistics, contract-completeness checks, and optional gold-set precision/recall — the
substantiation of the "AI-ready" claim. Assembling it used to live inside
:class:`~latence_core.runner.LocalRunner` as ~830 lines of ``_build_report`` + ``_*_quality``
methods, so a second Runner (the Airflow adapter) could only reuse it by reaching **past the
interface** into six private ``LocalRunner`` methods (``# noqa: SLF001``). The audit's THEME F
verdict: *"the interface is the reuse surface — if a second impl must reach past it, the shape is
wrong."*

This module lifts that half out as a cohesive unit that depends only on what it needs, not the
whole executor:

* the stage ``outputs`` mapping + a :class:`StageRecordReader` seam that yields a Stage's records
  (from the live dict or, once released for memory bounding — #69 — its on-disk checkpoint), and
* the run identity / :class:`~latence_core.pipeline.Pipeline` /
  :class:`~latence_core.storage.Storage` it reads, plus the relation fan-out-guard tally the
  executor captured while Stages ran.

:class:`LocalRunner` now *composes* a :class:`QualityReportBuilder` (its ``_build_report`` /
``_final_records`` are thin delegators), and the Airflow adapter constructs the **public**
:class:`QualityReportBuilder` directly against the engine's public
:meth:`~latence_core.runner.LocalRunner.stage_records` reader — so neither Runner reaches past the
interface. This is a **pure structural refactor: the Quality Report is byte-identical** to the
pre-extraction output; the report-building logic below moved verbatim, only its collaborators are
now injected instead of ``self``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol

from latence_core.contracts import (
    ChunkRecord,
    DisambiguationRecord,
    DisambiguationScope,
    Disposition,
    DocumentRecord,
    EntityMention,
    FeatureRecord,
    FeatureScope,
    GraphRecord,
    GraphScope,
    Record,
    RedactionRecord,
    RelationMention,
    ScreeningCheckpoint,
    ScreeningFinding,
)
from latence_core.errors import ConfigError
from latence_core.goldset import (
    ConfusionCounts,
    GoldSet,
    score_entities,
    score_relations,
)
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import CapabilityKind
from latence_core.quality import (
    ChunkQuality,
    CompletenessCheck,
    ContextEnrichmentQuality,
    DisambiguationQuality,
    DriftDiagnostics,
    EntityQuality,
    GoldSetEvaluation,
    GraphQuality,
    LabelScore,
    ParseQuality,
    ProfilingQuality,
    QualityReport,
    RedactionQuality,
    RelationQuality,
    ScreeningQuality,
    StageMetrics,
    StageScore,
    TypeConsolidationQuality,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from pydantic import BaseModel

    from latence_core.pipeline import Pipeline, Stage
    from latence_core.runner import StageOutput
    from latence_core.storage import Storage


# -- co-occurrence fingerprint (relocated from runner.py, re-exported there) ----------------
#
# The Quality Report is a shareable artifact; a co-occurring entity pair drawn from a
# ``confidential`` source could carry PII in its raw surface. These helpers key each pair by a
# salted sha256 fingerprint instead of the raw ``left|right`` surface. They lived in runner.py
# before H-F1; the report code that uses them moved here, so they moved too — runner.py re-exports
# them for backwards compatibility (some tests import them from there).
_CO_OCCURRENCE_FINGERPRINT_LEN = 16
_DEFAULT_FINGERPRINT_SALT = "latence"


def _co_occurrence_fingerprint(left: str, right: str, salt: str) -> str:
    """A deterministic, salted fingerprint of a co-occurrence pair.

    The Quality Report is a shareable artifact; a co-occurring entity pair drawn from a
    ``confidential`` source could carry PII in its raw surface. Rather than emit
    ``"{left}|{right}"`` (the raw surfaces) into the report, we key the co-occurrence
    entry by ``"pair:" + <sha256 hex prefix>`` over ``"{salt}:{left}|{right}"`` — the
    same salted-digest construction S6 uses in
    :func:`~latence_core.redaction_policy.placeholder_for` (``salt`` resolved from the
    pipeline's Redaction Stage config, default ``"latence"``).

    This is a **fingerprint, not encryption**: entity surfaces are low-entropy, so a
    reader who can enumerate candidate names could recover a pair by dictionary attack.
    The salt makes that attack per-deployment (the attacker must know the deployment's
    salt) and defeats precomputed rainbow tables, matching S6's threat posture — it does
    **not** make the digest irreversible. It removes the raw surface from the shareable
    report; it does not make the report safe to publish to an untrusted audience. The
    raw pair remains only in the exported, sensitivity-stamped ``CorpusFeatures`` data
    artifact. Deterministic (byte-stable across processes, not the seed-salted builtin
    ``hash``) so a seeded run renders identically, and stable per pair so the recurrence
    signal survives.
    """
    digest = hashlib.sha256(f"{salt}:{left}|{right}".encode()).hexdigest()
    return f"pair:{digest[:_CO_OCCURRENCE_FINGERPRINT_LEN]}"


def _fingerprint_salt(pipeline: Pipeline) -> str:
    """Resolve the co-occurrence fingerprint salt from the Redaction Stage config.

    Uses the identical key + default as S6's Redaction Stage
    (``config["salt"]``, default ``"latence"``) so a deployment that sets a custom salt
    gets the same salt applied to both PII surrogates and co-occurrence fingerprints. If
    the pipeline has no Redaction Stage, the default applies.
    """
    for stage in pipeline.stages:
        if stage.capability is CapabilityKind.REDACTION:
            return str(stage.config.get("salt", _DEFAULT_FINGERPRINT_SALT))
    return _DEFAULT_FINGERPRINT_SALT


class StageRecordReader(Protocol):
    """The run-state read seam the report builder needs from a Runner.

    Assembling the Quality Report reads two kinds of persisted run state, both of which are the
    Runner's file-layout concern, not the builder's:

    * :meth:`stage_records` — a Stage's output records. A Runner threads Stage outputs through a
      per-Stage ``outputs`` dict, releasing a doc-level Stage's entry once its last consumer ran
      (#69) — after which the record set is read back from the on-disk checkpoint, one batch at a
      time, to bound corpus-level memory. The builder does not care which path a Stage's records
      come from; it only needs them yielded in byte-identical order.
    * :meth:`read_findings` — a Screening Stage's persisted findings sidecar (written atomically
      beside the checkpoint), so the report's Screening rollup is identical on fresh and resumed
      runs.

    :class:`~latence_core.runner.LocalRunner` satisfies this Protocol via its public
    :meth:`~latence_core.runner.LocalRunner.stage_records` and
    :meth:`~latence_core.runner.LocalRunner.read_findings`, so the builder composes with the engine
    without naming its concrete type — and a second Runner reuses the builder the same way, through
    the public interface, never a private reach-through (H-F1).
    """

    def stage_records(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        """Yield one Stage's output — from the live dict if retained, else its checkpoint (#69)."""
        ...

    def read_findings(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[ScreeningFinding]:
        """Read a Screening Stage's persisted findings sidecar (empty if none written)."""
        ...


@dataclass(frozen=True)
class QualityRunContext:
    """Everything an aggregator may need to decide whether it applies, and to configure itself.

    Handed to each registered factory exactly once per report. It carries the *run*, never the
    records: an aggregator that needs a Stage's config (the Context-Enrichment caps), the Redaction
    supersession topology, the run's fan-out tally, or the Storage-backed screening sidecars reads
    them from here at construction, and everything else arrives one record at a time through
    :meth:`QualityAggregator.observe`.
    """

    pipeline: Pipeline
    ordered: list[Stage]
    run_id: str
    storage: Storage
    reader: StageRecordReader
    fanout: tuple[int, int, int, int]


class QualityAggregator(Protocol):
    """One Quality-Report section, computed as a FOLD over a record stream (C5).

    The :class:`StageRecordReader` seam is lazy by construction — ``stage_records`` yields one
    :class:`~latence_core.runner.StageOutput` at a time precisely so a Runner can release a
    doc-level Stage's retained output and stream it back from its checkpoint (#69, ADR-0033). The
    report builder used to defeat that: each per-kind gather did ``result.extend(...)`` and
    ``build()`` held fourteen whole-corpus lists live at once (the entity-mention stream — the
    largest in any corpus — twice). This Protocol is the consumer that matches the producer.

    An implementation declares:

    * ``section_field`` — the :class:`~latence_core.quality.QualityReport` field it fills. One
      aggregator owns one field; the report keeps its typed, named sections (a discriminated list
      would be worse for every consumer of the artifact).
    * ``kinds`` — the :class:`~latence_core.pipeline.CapabilityKind`\\ s whose Stage output it
      wants. The builder streams each subscribed Stage's records ONCE and fans them out to every
      subscriber, so two sections reading the same stream cost one decode, not two.

    and implements:

    * :meth:`observe` — fold one record into bounded state. The ``stage`` is passed because a fold
      legitimately depends on WHICH Stage produced the record: a section that subscribes to two
      kinds (asserted vs inferred graph edges), or one that must ignore a superseded Redaction
      Stage, decides here rather than in ``build()``.
    * :meth:`finish` — return the section, or ``None`` when the run has nothing to report (the
      honest "no such Stage ran" answer, never an all-zero section).

    **Invariants an implementation must hold.**

    * *Bounded state.* ``observe`` must not accumulate per-record objects. Counters, sums, sets of
      ids, and small sorted maps are the vocabulary. (:class:`_GoldSetAggregator` is the one
      documented exception — the scorer indexes both whole sets by construction — and it is opt-in,
      constructed only when the Pipeline declares a ``gold_set_uri``.)
    * *Order-stability.* Records arrive in the DAG's topological Stage order, and within a Stage in
      the reader's yield order — the same order the old per-kind gather concatenated. Any
      accumulation that is order-sensitive (float sums, last-write-wins alias maps) therefore
      reproduces the pre-C5 bytes exactly; any map that reaches the report is sorted before it is
      returned. ADR-0008's H-F1 amendment makes the report's BYTES the contract.
    * *No side effects.* ``finish`` returns a value; it never writes.

    Adding a Stage's quality metric is one module implementing this Protocol plus one entry in
    :data:`QUALITY_AGGREGATORS` — ``build()`` is not edited. Lane L2's ``CapabilityDescriptor``
    will own the per-Capability factory reference; because
    :class:`QualityReportBuilder` takes the factory tuple as a dependency, that wiring replaces the
    default tuple and changes nothing else here.
    """

    section_field: ClassVar[str]
    kinds: ClassVar[tuple[CapabilityKind, ...]]

    def observe(self, stage: Stage, record: StageOutput) -> None:
        """Fold one record of one subscribed Stage into this section's bounded state."""
        ...

    def finish(self) -> BaseModel | None:
        """The finished section, or ``None`` when the run has nothing to report."""
        ...


#: What the builder needs to obtain an aggregator for a run: a callable that inspects the run and
#: either returns its aggregator or declines (``None``) because the Pipeline has no such Stage. The
#: decline is load-bearing — a declined aggregator never subscribes, so its Capability's records are
#: never decoded at all.
QualityAggregatorFactory = Callable[[QualityRunContext], "QualityAggregator | None"]


class QualityReportBuilder:
    """Assembles a run's :class:`~latence_core.quality.QualityReport` from its Stage outputs.

    Depends on a :class:`StageRecordReader` (to read each Stage's records back from the live
    ``outputs`` dict or its checkpoint) plus the relation fan-out-guard tally the executor
    captured — **not** the whole executor. Both Runners construct it against their engine's public
    reader and call :meth:`build`; the report is byte-identical to the pre-H-F1 in-runner build.

    The relation fan-out-guard tally is passed as a live mapping (not a snapshot): the local
    Runner mutates ``LocalRunner._fanout_by_stage`` while Stages run and reads it at report time —
    the builder reads the same mapping through :meth:`_fanout_total`. A Runner that captures no
    fan-out (the Airflow in-process path resets the engine's tally per Stage) passes an empty
    mapping, so every Stage contributes ``(0, 0, 0, 0)`` — the honest default.
    """

    def __init__(
        self,
        reader: StageRecordReader,
        *,
        fanout_by_stage: Mapping[str, tuple[int, int, int, int]] | None = None,
        aggregators: tuple[QualityAggregatorFactory, ...] | None = None,
    ) -> None:
        self._reader = reader
        self._fanout_by_stage: Mapping[str, tuple[int, int, int, int]] = (
            fanout_by_stage if fanout_by_stage is not None else {}
        )
        # The registry is a DEPENDENCY, not a hard-wired table (C5): the default is the built-in
        # section set, and lane L2's CapabilityDescriptor will supply the per-Capability factories
        # instead. Every caller that does not care passes nothing and learns nothing.
        self._aggregators: tuple[QualityAggregatorFactory, ...] = (
            aggregators if aggregators is not None else QUALITY_AGGREGATORS
        )

    # -- public surface ----------------------------------------------------

    def final_records(
        self,
        ordered: list[Stage],
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> list[StageOutput]:
        """Records for the Quality Report: the outputs of the DAG's sink Stages.

        A sink is a Stage nothing depends on. Reporting on sinks (rather than a
        single threaded ``current``) keeps the report correct for a fan-out or
        converging DAG, where the last topological Stage is not the whole story.
        A sink whose retained output was released (#69) is streamed back from its checkpoint.
        """
        depended_on = {p for s in ordered for p in s.depends_on}
        sinks = [s.name for s in ordered if s.name not in depended_on]
        final: list[StageOutput] = []
        for name in sinks:
            final.extend(
                self._reader.stage_records(
                    name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            )
        return final

    def build(
        self,
        pipeline: Pipeline,
        run_id: str,
        storage: Storage,
        ordered: list[Stage],
        stage_metrics: list[StageMetrics],
        outputs: Mapping[str, list[StageOutput]],
        final_records: Iterable[Record | StageOutput],
    ) -> QualityReport:
        """Assemble the run's report by FOLDING each Stage's records through its aggregators (C5).

        One pass over the DAG. For every Stage that at least one aggregator subscribed to, the
        reader's stream is opened ONCE and each record is fanned out to that Stage's subscribers,
        then dropped — so peak memory is ``O(aggregators x bounded state)`` rather than
        ``O(corpus x sections)``, and the entity-mention stream (previously decoded twice, once as
        ``mentions`` and once as the byte-identical ``disambig_mentions_in``) is decoded once.

        ``final_records`` is consumed lazily too: the sink records are folded into the completeness
        check and the distinct-source-document count in a single pass, never copied into a list. A
        caller that hands over an iterator therefore never materialises the sinks either.

        The report is byte-identical to the pre-C5 gather. Records reach each aggregator in the same
        order the per-kind gather concatenated them (topological Stage order, reader order within a
        Stage), which is what makes the order-sensitive accumulations — float averages, last-wins
        alias maps — reproduce exactly (ADR-0008 H-F1).
        """
        context = QualityRunContext(
            pipeline=pipeline,
            ordered=ordered,
            run_id=run_id,
            storage=storage,
            reader=self._reader,
            fanout=self._fanout_total(ordered),
        )
        aggregators: list[QualityAggregator] = []
        for factory in self._aggregators:
            aggregator = factory(context)
            if aggregator is not None:
                aggregators.append(aggregator)

        # Subscription plan: which aggregators want each Capability's records. An aggregator that
        # declined (no such Stage in the Pipeline) subscribes to nothing, so its Capability's
        # records are never decoded — the cheapest possible answer for a narrow stack.
        subscribers: dict[CapabilityKind, list[QualityAggregator]] = {}
        for aggregator in aggregators:
            for kind in aggregator.kinds:
                subscribers.setdefault(kind, []).append(aggregator)

        for stage in ordered:
            listeners = subscribers.get(stage.capability)
            if not listeners:
                continue
            for record in self._reader.stage_records(
                stage.name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                for aggregator in listeners:
                    aggregator.observe(stage, record)

        sections: dict[str, BaseModel] = {}
        for aggregator in aggregators:
            section = aggregator.finish()
            if section is not None:
                sections[aggregator.section_field] = section

        completeness, document_count = _fold_final_records(final_records)
        return QualityReport(
            pipeline_name=pipeline.name,
            run_id=run_id,
            document_count=document_count,
            total_duration_seconds=round(sum(m.duration_seconds for m in stage_metrics), 6),
            stages=stage_metrics,
            completeness=completeness,
            contracts_complete=completeness.passed,
            **sections,  # type: ignore[arg-type]
        )

    def _fanout_total(self, ordered: list[Stage]) -> tuple[int, int, int, int]:
        """Sum the per-Stage relation fan-out-guard tallies across the run (ADR-0033 / #63, #68).

        Folds every relation/fused Stage's ``(output-capped-docs, dropped-relations,
        input-capped-docs, dropped-mentions)`` tally (captured in the executor while Stages ran)
        into one corpus-wide 4-tuple for the Quality Report. A Stage restored from a checkpoint on
        resume never re-ran its Provider, so it contributes nothing here — the audited tally
        reflects Providers that actually executed this run (a resumed relation Stage's caps were
        already reflected in its committed output).
        """
        out_capped = 0
        dropped_rel = 0
        in_capped = 0
        dropped_men = 0
        for stage in ordered:
            oc, dr, ic, dm = self._fanout_by_stage.get(stage.name, (0, 0, 0, 0))
            out_capped += oc
            dropped_rel += dr
            in_capped += ic
            dropped_men += dm
        return out_capped, dropped_rel, in_capped, dropped_men


    # -- backwards-compatible shim ----------------------------------------

    @staticmethod
    def _drift_diagnostics(chunked: list[StageOutput]) -> DriftDiagnostics | None:
        """Fold a materialised chunk list through the drift aggregator (compat shim).

        The drift roll-up itself is now :class:`_DriftAggregator`, folded from the streamed chunk
        records like every other section. This staticmethod survives because
        :meth:`~latence_core.runner.LocalRunner._drift_diagnostics` forwards to it and the existing
        unit tests exercise it directly with a hand-built list; it is a thin re-entry into the same
        fold, so the two can never disagree.
        """
        aggregator = _DriftAggregator()
        for record in chunked:
            aggregator.audit(record)
        return aggregator.finish()


def _fold_final_records(
    final_records: Iterable[Record | StageOutput],
) -> tuple[CompletenessCheck, int]:
    """One lazy pass over the sink records: contract-completeness + distinct source documents.

    Both answers used to be read off ``records = list(final_records)`` — a whole-corpus copy on top
    of the thirteen per-kind lists. They are folds, so they are computed together in a single pass
    and the sink stream is never materialised here. A caller that passes an iterator therefore pays
    ``O(1)``; a caller that passes a list keeps only its own list.

    **Completeness** (S10): every record must carry Provenance and Classification
    (contract-guaranteed, audited here), and the offsets must be *aligned* — a record that carries a
    char span must carry a **well-ordered** one, and if it also carries page coordinates those must
    be well-ordered too. A record with no char span at all is vacuously aligned and not counted.
    Page *presence* is not re-required: the sub-document carriers whose contracts require a page
    span (EntityMention/RelationMention) already guarantee it at construction, while a
    whole-document carrier (DocumentRecord/RedactionRecord/FeatureRecord) legitimately holds a
    ``[0, len)`` char span with no page span.

    **Document count**: distinct source documents seen across the run. Chunks fan out from a
    document (many chunks share one ``document_id``), so counting distinct Provenance document ids
    is correct whether the sinks are DocumentRecords or ChunkRecords. A Profiling Stage's
    CORPUS-scope FeatureRecord carries a synthetic corpus-scope ``document_id`` (it is not a source
    document), so it is excluded.
    """
    check = CompletenessCheck()
    doc_ids: set[str] = set()
    for record in final_records:
        check.records_checked += 1
        if record.provenance is not None:
            check.records_with_provenance += 1
        if record.classification is not None:
            check.records_with_classification += 1
        prov = record.provenance
        if not (isinstance(record, FeatureRecord) and record.scope is FeatureScope.CORPUS):
            doc_ids.add(prov.document_id)
        if prov.char_start is None and prov.char_end is None:
            # No char span to audit — vacuously aligned.
            continue
        check.offset_bearing_records += 1
        span_ok = (
            prov.char_start is not None
            and prov.char_end is not None
            and prov.char_end >= prov.char_start
        )
        pages_ok = (
            prov.page_start is None or prov.page_end is None or prov.page_end >= prov.page_start
        )
        if span_ok and pages_ok:
            check.records_with_aligned_offsets += 1
    return check, len(doc_ids)


# ==========================================================================================
# The built-in aggregators. One module-private class per Quality-Report section; each declares
# the Capabilities whose records it folds and the report field it fills, and NONE of them is
# named in :meth:`QualityReportBuilder.build` — they reach the report through
# :data:`QUALITY_AGGREGATORS` alone.
# ==========================================================================================


_MENTION_KINDS = (CapabilityKind.ENTITY_EXTRACTION, CapabilityKind.FUSED_ENTITY_RELATION)
_RELATION_KINDS = (CapabilityKind.RELATION_EXTRACTION, CapabilityKind.FUSED_ENTITY_RELATION)
_SCREENING_KINDS = (CapabilityKind.INTAKE_SCREENING, CapabilityKind.CONTENT_SCREENING)


def _has_kind(pipeline: Pipeline, kinds: tuple[CapabilityKind, ...]) -> bool:
    """True when the Pipeline declares at least one Stage fulfilling any of ``kinds``."""
    return any(s.capability in kinds for s in pipeline.stages)


class _ParseAggregator:
    """Parse-Stage quality: page-map drift + parse-error counts (S2).

    Constructed only when the Pipeline actually has a Parse Stage, so a Source->Export pipeline
    reports ``parse=None`` rather than a misleading all-zero section. For each parsed
    :class:`DocumentRecord` that carries a page map, a :class:`PageOffsetIndex` is built and audited
    — resolving each page's boundary offsets — so the drift diagnostics the ported asset produces
    surface in the Quality Report (S2: "drift diagnostics in the Quality Report"). A clean parse
    resolves every offset exactly (``page_maps_clean``). Each parsed record's assembled-markdown
    byte size is summed into ``bytes_after_parse`` — the observable rollup of the Provenance
    'size-after-parse'. The page index is built and discarded per record, so the fold's state is the
    :class:`~latence_core.quality.ParseQuality` counters and nothing else.
    """

    section_field: ClassVar[str] = "parse"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (CapabilityKind.PARSE,)

    def __init__(self) -> None:
        self._quality = ParseQuality()

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _ParseAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if not isinstance(record, DocumentRecord):
            return
        quality = self._quality
        if record.disposition is Disposition.PARSE_ERROR:
            quality.parse_errors += 1
            return
        quality.documents_parsed += 1
        # Provenance 'size-after-parse', aggregated: the assembled-markdown byte size of each
        # parsed record (mirrors the parser's per-record ``size_after_parse`` helper).
        quality.bytes_after_parse += len(record.content.encode("utf-8"))
        page_map = record.page_map
        if page_map is None:
            return
        quality.page_map_present += 1
        quality.pages_total += len(page_map.pages)
        index = PageOffsetIndex(page_map)
        # Audit: resolve each page's start and last-char offset. With exact boundaries every lookup
        # is exact; any drift the map carries is detected here and accumulated.
        for span in page_map.pages:
            index.page_for_offset(span.char_start)
            index.page_for_offset(max(span.char_end - 1, span.char_start))
        diag = index.diagnostics
        quality.page_drift_resolutions += diag.resolutions
        quality.page_drift_recovered += diag.recovered
        quality.page_drift_max_chars = max(quality.page_drift_max_chars, diag.max_drift_chars)

    def finish(self) -> ParseQuality | None:
        return self._quality


class _ChunkAggregator:
    """Chunk-Stage quality: chunk counts, tokens, offset integrity (S3).

    Constructed only when the Pipeline has a Chunk Stage. The Chunk Stage's own output is the
    canonical chunk set from which counts, tokens, and the offset-preservation invariant are
    measured: every :class:`ChunkRecord`'s Provenance ``char_start``/``char_end`` must be a valid
    span and its ``page_start``/``page_end`` populated — the S3 round-trip invariant, so
    ``offset_preserving`` is True only if every chunk carried a resolvable span.

    ``chunks_flagged`` reads the POST-Content-Screening corpus, because Content Screening is what
    attaches the risk markers (the raw Chunk output has none yet), falling back to the chunk output
    when the stack declares no Content-Screening Stage. The gather expressed that as
    ``post_content if post_content else chunked`` over two materialised lists; the fold keeps both
    tallies plus the "did any screened record arrive at all" bit, which is exactly the same
    predicate — non-empty ``post_content`` — decided without holding either list.
    """

    section_field: ClassVar[str] = "chunk"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (
        CapabilityKind.CHUNK,
        CapabilityKind.CONTENT_SCREENING,
    )

    def __init__(self) -> None:
        self._quality = ChunkQuality()
        self._flagged_in_chunks = 0
        self._flagged_after_screening = 0
        self._saw_screened_record = False

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _ChunkAggregator | None:
        if not _has_kind(context.pipeline, (CapabilityKind.CHUNK,)):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if stage.capability is CapabilityKind.CONTENT_SCREENING:
            self._saw_screened_record = True
            if isinstance(record, ChunkRecord) and record.flagged:
                self._flagged_after_screening += 1
            return
        if not isinstance(record, ChunkRecord):
            return
        quality = self._quality
        quality.chunks_total += 1
        quality.tokens_total += record.token_count
        prov = record.provenance
        has_span = (
            prov.char_start is not None
            and prov.char_end is not None
            and prov.char_end >= prov.char_start
        )
        has_pages = prov.page_start is not None and prov.page_end is not None
        if has_span and has_pages:
            quality.chunks_offset_preserving += 1
        if record.flagged:
            self._flagged_in_chunks += 1

    def finish(self) -> ChunkQuality | None:
        self._quality.chunks_flagged = (
            self._flagged_after_screening
            if self._saw_screened_record
            else self._flagged_in_chunks
        )
        return self._quality


class _DriftAggregator:
    """Corpus-level Provenance-integrity drift roll-up across the chunk offsets (S10).

    Audits every :class:`ChunkRecord` that carries its parent document's page map by resolving the
    chunk's ``[char_start, char_end)`` endpoints through a :class:`PageOffsetIndex` and accumulating
    the fuzzy recoveries — a finer drift signal than the Parse-only page-boundary audit. Only
    ChunkRecords are audited: mentions, relations, and redaction spans inherit the chunk offsets the
    Chunk Stage already validated against the page map, so they carry no independent offset->page
    drift. Returns ``None`` when there were no page-map-bearing chunk records to audit (a
    Parse-only or chunk-free run), so a pipeline with nothing to audit reports no drift section
    rather than a misleading all-zero one.

    Subscribes to the same chunk stream as :class:`_ChunkAggregator`; the builder decodes it once
    and hands each record to both.
    """

    section_field: ClassVar[str] = "drift"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (CapabilityKind.CHUNK,)

    def __init__(self) -> None:
        self._audited = 0
        self._resolutions = 0
        self._recovered = 0
        self._max_drift = 0

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _DriftAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        self.audit(record)

    def audit(self, record: Record | StageOutput) -> None:
        """Fold one record into the drift tally (the Stage is irrelevant — see the compat shim)."""
        if not isinstance(record, ChunkRecord) or record.page_map is None:
            return
        prov = record.provenance
        if prov.char_start is None or prov.char_end is None:
            return
        index = PageOffsetIndex(record.page_map)
        index.page_for_offset(prov.char_start)
        index.page_for_offset(max(prov.char_end - 1, prov.char_start))
        diag = index.diagnostics
        self._audited += 1
        self._resolutions += diag.resolutions
        self._recovered += diag.recovered
        self._max_drift = max(self._max_drift, diag.max_drift_chars)

    def finish(self) -> DriftDiagnostics | None:
        if self._audited == 0:
            return None
        return DriftDiagnostics(
            resolutions=self._resolutions,
            recovered=self._recovered,
            max_drift_chars=self._max_drift,
            records_audited=self._audited,
        )


class _ScreeningAggregator:
    """Screening findings across both checkpoints (S3) — the one section that reads a SIDECAR.

    Screening's report input is not a record stream: each Screening Stage's findings + quarantine
    dispositions are persisted to a sidecar beside its checkpoint (so the numbers are identical on
    fresh and resumed runs) and read back through
    :meth:`StageRecordReader.read_findings`. It therefore subscribes to no
    :class:`~latence_core.pipeline.CapabilityKind` at all and does all its work in :meth:`finish`.

    Keeping it in the same registry as the record-stream sections is deliberate: "how do I add a
    Quality-Report section?" has exactly one answer, and the odd one out is odd inside its own
    module rather than as a special case in ``build()``. Every Quarantine disposition and reason and
    every propagated flag is auditable in the report (S3 AC).
    """

    section_field: ClassVar[str] = "screening"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = ()

    def __init__(self, context: QualityRunContext) -> None:
        self._context = context

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _ScreeningAggregator | None:
        if not _has_kind(context.pipeline, _SCREENING_KINDS):
            return None
        return cls(context)

    def observe(self, stage: Stage, record: StageOutput) -> None:  # pragma: no cover - no kinds
        """Never called: this section subscribes to no record stream."""

    def finish(self) -> ScreeningQuality | None:
        ctx = self._context
        quality = ScreeningQuality()
        for stage in ctx.ordered:
            if stage.capability not in _SCREENING_KINDS:
                continue
            findings = ctx.reader.read_findings(
                ctx.storage, ctx.pipeline, ctx.run_id, stage.name
            )
            quality.findings.extend(findings)
            for finding in findings:
                if finding.checkpoint is ScreeningCheckpoint.INTAKE:
                    quality.intake_screened += 1
                else:
                    quality.content_screened += 1
                if finding.verdict.value == "quarantine":
                    quality.quarantined += 1
                elif finding.verdict.value == "flag":
                    quality.flagged += 1
                else:
                    quality.passed += 1
        return quality


class _EntityAggregator:
    """Entity-Extraction quality: per-type counts + confidence distribution (S4).

    Constructed only when the Pipeline has an Entity-Extraction (or fused) Stage. Folds the mention
    set into per-label counts, the confidence histogram, and the offset-resolution invariant: every
    :class:`~latence_core.contracts.EntityMention`'s Provenance must carry a valid char span AND a
    page span — the S4 round-trip (mention -> offset -> original page) — so ``offset_resolving`` is
    True only when every mention carried a resolvable span. The confidence distribution + per-type
    counts are the S4 acceptance criterion.

    A fused Entity+Relation Stage emits mentions and relations together, so this aggregator and
    :class:`_RelationAggregator` both subscribe to ``FUSED_ENTITY_RELATION`` and each isinstance-
    filters the mixed stream to its own carrier — one decode, two sections.

    Bounded state: counters, a ten-bin histogram, a per-label map, and the set of chunk ids that
    produced a mention (``chunks_with_mentions``) — bounded by the corpus's chunk count, not its
    mention count, and the same set the gather built.
    """

    section_field: ClassVar[str] = "entities"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = _MENTION_KINDS

    def __init__(self) -> None:
        self._quality = EntityQuality()
        self._chunk_ids_with_mentions: set[str] = set()

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _EntityAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if not isinstance(record, EntityMention):
            return
        quality = self._quality
        quality.mentions_total += 1
        quality.per_type[record.label] = quality.per_type.get(record.label, 0) + 1
        quality.confidence.add(record.confidence)
        self._chunk_ids_with_mentions.add(record.chunk_record_id)
        prov = record.provenance
        has_span = (
            prov.char_start is not None
            and prov.char_end is not None
            and prov.char_end >= prov.char_start
        )
        has_pages = prov.page_start is not None and prov.page_end is not None
        if has_span and has_pages:
            quality.mentions_offset_resolving += 1

    def finish(self) -> EntityQuality | None:
        quality = self._quality
        quality.chunks_with_mentions = len(self._chunk_ids_with_mentions)
        # Sort per_type for a stable, deterministic report.
        quality.per_type = dict(sorted(quality.per_type.items()))
        return quality


class _RelationAggregator:
    """Relation-Extraction quality: per-type counts + confidence distribution (S5).

    Constructed only when the Pipeline has a Relation-Extraction (or fused) Stage. Folds the
    relation set into per-label counts, the confidence histogram, and the offset-resolution
    invariant: every :class:`~latence_core.contracts.RelationMention`'s Provenance must carry a
    valid char span AND a page span — the S5 round-trip — so ``offset_resolving`` is True only when
    every relation carried a resolvable span.

    The run's ``(output-capped-docs, dropped-relations, input-capped-docs, dropped-mentions)``
    fan-out-guard tally (ADR-0033 / #63 output cap; #68 input mention cap) is read from the
    :class:`QualityRunContext` at construction and surfaced as the audited "capped" marker, so a
    bounded pathological corpus is never silently truncated.
    """

    section_field: ClassVar[str] = "relations"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = _RELATION_KINDS

    def __init__(self, fanout: tuple[int, int, int, int]) -> None:
        self._quality = RelationQuality()
        (
            self._quality.fanout_capped_documents,
            self._quality.fanout_dropped_relations,
            self._quality.mention_capped_documents,
            self._quality.mention_dropped_mentions,
        ) = fanout
        self._docs_with_relations: set[str] = set()

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _RelationAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls(context.fanout)

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if not isinstance(record, RelationMention):
            return
        quality = self._quality
        quality.relations_total += 1
        quality.per_type[record.label] = quality.per_type.get(record.label, 0) + 1
        quality.confidence.add(record.confidence)
        self._docs_with_relations.add(record.document_record_id)
        prov = record.provenance
        has_span = (
            prov.char_start is not None
            and prov.char_end is not None
            and prov.char_end >= prov.char_start
        )
        has_pages = prov.page_start is not None and prov.page_end is not None
        if has_span and has_pages:
            quality.relations_offset_resolving += 1

    def finish(self) -> RelationQuality | None:
        quality = self._quality
        quality.documents_with_relations = len(self._docs_with_relations)
        quality.per_type = dict(sorted(quality.per_type.items()))
        return quality


def _terminal_redaction_stages(pipeline: Pipeline) -> set[str]:
    """Names of the Redaction Stages whose output actually reaches the rest of the Pipeline.

    A Redaction Stage is SUPERSEDED when some other Redaction Stage depends on it, directly or
    transitively: since W16 Redaction is a chunk->chunk transform and
    :func:`~latence_core.redaction_policy._compose_prior_spans` folds the upstream spans into the
    downstream Stage's own, the downstream Stage re-emits the same chunks carrying every mask the
    chain applied. Only that downstream version flows on to the Export, so only it may be tallied.
    Parallel Redaction branches supersede nothing and are all terminal.
    """
    by_name = {s.name: s for s in pipeline.stages}
    redaction_stages = {s.name for s in pipeline.stages if s.capability is CapabilityKind.REDACTION}
    superseded: set[str] = set()
    for name in redaction_stages:
        seen: set[str] = set()
        frontier = list(by_name[name].depends_on)
        while frontier:
            dep = frontier.pop()
            if dep in seen or dep not in by_name:
                continue
            seen.add(dep)
            if dep in redaction_stages:
                superseded.add(dep)
            frontier.extend(by_name[dep].depends_on)
    return redaction_stages - superseded


class _RedactionAggregator:
    """Redaction quality: per-PII-type + per-action detection counts (S6/W16).

    Constructed only when the Pipeline has a Redaction Stage. Since W16 (ADR-0042) Redaction is a
    chunk->chunk transform, so the stream is the set of redacted
    :class:`~latence_core.contracts.ChunkRecord`s (each carrying ``masked_content`` +
    ``pii_spans``). Per-type/per-action span counts are summed across chunks; documents-redacted and
    documents-with-PII are rolled up to the DOCUMENT level (distinct ``document_record_id``s) so the
    report reads the same as the old one-record-per-document shape — the S6 acceptance criterion.
    Only counts and type/action names cross into the report; **no raw PII value** ever does (a
    PIISpan never stores the surface string, so this rollup cannot leak one). The deprecated
    doc-level :class:`~latence_core.contracts.RedactionRecord` path is still tallied for backward
    compatibility.

    Only the TERMINAL Redaction Stages are folded, and every one of their records is summed with
    FULL MULTIPLICITY. Both halves matter, and they used to be got wrong in opposite directions:

    * Folding EVERY Redaction Stage's output double-counts a CHAINED topology — exactly what the
      shipped ``gpu-learned`` / ``full-stack`` stacks declare since audit R5-12
      (``redact_financial.depends_on: [redact_pii]``). The same chunk appears twice: once as the
      upstream Stage emitted it, once as the downstream Stage re-emitted it after
      :func:`~latence_core.redaction_policy._compose_prior_spans` folded the upstream spans in. That
      inflates every upstream span and can name a ``pii_type`` that exists nowhere in the exported
      corpus. Dropping the superseded Stages at the source fixes this: the terminal Stage's chunks
      already carry every mask the chain applied, because composition is coverage-preserving.
    * De-duplicating by ``record_id`` instead (audit R6) fixes the chain but UNDER-counts the plain
      single-Redactor path, because ``record_id``s are content-addressed: two byte-identical
      documents share one ``document_id`` and hence one chunk ``record_id``, yet both are separately
      exported and both carry real PII. Halving that corpus's tally is a silent regression (audit
      R7), so no de-duplication by ``record_id`` happens here.

    The gather expressed the terminal-stage rule by handing ``_records_of_kind`` a pre-filtered
    Stage list; the fold decides per record from the ``stage`` it was produced by — the same rule,
    now owned by the section that depends on it. The report is the operator's redaction audit trail
    (ADR-0034 counts-only observability), so it must describe the EXPORTED CORPUS.
    """

    section_field: ClassVar[str] = "redaction"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (CapabilityKind.REDACTION,)

    def __init__(self, terminal: set[str]) -> None:
        self._terminal = terminal
        self._quality = RedactionQuality()
        self._redacted_docs: set[str] = set()
        self._docs_with_pii: set[str] = set()
        self._disabled_docs: set[str] = set()

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _RedactionAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls(_terminal_redaction_stages(context.pipeline))

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if stage.name not in self._terminal:
            return
        if isinstance(record, ChunkRecord):
            # Chunk-level (W16): masked_content is set on every chunk the Redaction Stage produced.
            if record.masked_content is None:
                return
            doc_id = record.document_record_id
            self._redacted_docs.add(doc_id)
            if record.pii_spans:
                self._docs_with_pii.add(doc_id)
            if record.redaction_disabled_for_sensitive:
                self._disabled_docs.add(doc_id)
            spans = record.pii_spans
        elif isinstance(record, RedactionRecord):
            # Deprecated doc-level path: one record per document.
            self._redacted_docs.add(record.document_record_id)
            if record.pii_spans:
                self._docs_with_pii.add(record.document_record_id)
            if record.redaction_disabled_for_sensitive:
                self._disabled_docs.add(record.document_record_id)
            spans = record.pii_spans
        else:
            return
        quality = self._quality
        for span in spans:
            quality.pii_total += 1
            quality.per_type[span.pii_type] = quality.per_type.get(span.pii_type, 0) + 1
            action = span.action.value
            quality.per_action[action] = quality.per_action.get(action, 0) + 1

    def finish(self) -> RedactionQuality | None:
        quality = self._quality
        quality.documents_redacted = len(self._redacted_docs)
        quality.documents_with_pii = len(self._docs_with_pii)
        # H-C1 no-op floor: a sensitive doc whose control ran no detectors. LOUD, first-class —
        # while this is > 0, documents_with_pii:0 is NOT a clean bill.
        quality.redaction_disabled_for_sensitive = len(self._disabled_docs)
        quality.per_type = dict(sorted(quality.per_type.items()))
        quality.per_action = dict(sorted(quality.per_action.items()))
        return quality


class _ProfilingAggregator:
    """Profiling quality: per-document feature averages + corpus aggregates (S7).

    Constructed only when the Pipeline has a Profiling Stage. The stream is the mix of
    DOCUMENT-scope and the single CORPUS-scope
    :class:`~latence_core.contracts.FeatureRecord`s. The document-scope records are averaged into a
    compact readability/density/diversity summary, and the corpus-scope record's aggregates
    (word/vocabulary counts, corpus Zipf alpha, source coverage, entity-type frequency, top
    co-occurrences) are surfaced verbatim — the S7 acceptance criterion. Counts and averages only
    (no raw document text).

    The gather materialised the document-scope records so it could take ``len`` once and five
    ``sum``s over the same list; the fold keeps a count and five running totals instead. The
    accumulation order is the stream order, which is the order the gather's list was in, so the
    float sums are bit-identical — the byte-identity constraint bites hardest here.

    The co-occurrence list is rendered as ``pair-fingerprint`` -> document-count: the raw entity
    surfaces (possible PII from a ``confidential`` source that would otherwise land in an
    ``internal``-stamped report) are replaced by a deterministic, **salted** sha256 fingerprint of
    the ``left|right`` pair (the same salted-digest construction S6 uses; salt from the Redaction
    Stage config). The fingerprint keeps the raw surface out of the report and defeats rainbow-table
    recovery, but it is not irreversible — a reader who can enumerate candidate entity names and
    knows the salt could still recover a pair by dictionary attack, so the report is *not* safe to
    publish to an untrusted audience. The raw surfaces remain only in the exported,
    sensitivity-stamped ``CorpusFeatures`` data artifact.
    """

    section_field: ClassVar[str] = "profiling"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (CapabilityKind.PROFILING,)

    def __init__(self, salt: str) -> None:
        self._salt = salt
        self._documents = 0
        self._readability = 0.0
        self._density = 0.0
        self._type_token_ratio = 0.0
        self._zipf_alpha = 0.0
        self._compression_ratio = 0.0
        self._corpus: FeatureRecord | None = None

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _ProfilingAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls(_fingerprint_salt(context.pipeline))

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if not isinstance(record, FeatureRecord):
            return
        if record.scope is FeatureScope.DOCUMENT and record.document is not None:
            doc = record.document
            self._documents += 1
            self._readability += doc.readability
            self._density += doc.density
            self._type_token_ratio += doc.type_token_ratio
            self._zipf_alpha += doc.zipf_alpha
            self._compression_ratio += doc.compression_ratio
        elif (
            record.scope is FeatureScope.CORPUS
            and record.corpus is not None
            and self._corpus is None
        ):
            # The FIRST corpus-scope record wins, exactly as the gather's ``next(...)`` did. This is
            # the one whole record the fold retains, and there is at most one per run.
            self._corpus = record

    def finish(self) -> ProfilingQuality | None:
        quality = ProfilingQuality()
        n = self._documents
        quality.documents_profiled = n
        if n:
            quality.avg_readability = round(self._readability / n, 6)
            quality.avg_density = round(self._density / n, 6)
            quality.avg_type_token_ratio = round(self._type_token_ratio / n, 6)
            quality.avg_zipf_alpha = round(self._zipf_alpha / n, 6)
            quality.avg_compression_ratio = round(self._compression_ratio / n, 6)
        corpus = self._corpus
        if corpus is not None and corpus.corpus is not None:
            cf = corpus.corpus
            quality.corpus_total_words = cf.total_words
            quality.corpus_unique_words = cf.total_unique_words
            quality.corpus_zipf_alpha = cf.corpus_zipf_alpha
            quality.language_coverage = dict(cf.language_coverage)
            quality.category_coverage = dict(cf.category_coverage)
            quality.sensitivity_coverage = dict(cf.sensitivity_coverage)
            quality.distinct_entities = len(cf.entity_frequency)
            quality.entity_type_frequency = dict(cf.entity_type_frequency)
            quality.top_co_occurrences = {
                _co_occurrence_fingerprint(co.left, co.right, self._salt): co.document_count
                for co in cf.co_occurrences
            }
        return quality


@dataclass
class _TypeDriftFold:
    """How one carrier's label vocabulary changed across a Type-Consolidation Stage (T3).

    ``raw_label`` is set if and ONLY if the Stage rewrote that record's label, so the raw vocabulary
    is ``{raw_label or label}``, the canonical vocabulary is ``{label}``, and the alias map is
    exactly the rewrites applied. A run that consolidated nothing therefore reports equal counts and
    an empty map — the honest reading of "nothing happened", never a fabricated fold. State is
    bounded by the TYPE vocabulary (tens of labels), not by the mention count.
    """

    raw: set[str] = field(default_factory=set)
    canonical: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    remapped: int = 0

    def add(self, record: EntityMention | RelationMention) -> None:
        self.canonical.add(record.label)
        self.raw.add(record.raw_label or record.label)
        if record.raw_label is not None:
            self.aliases[record.raw_label] = record.label
            self.remapped += 1

    def alias_map(self) -> dict[str, str]:
        return dict(sorted(self.aliases.items()))


class _TypeConsolidationAggregator:
    """The TYPE-drift collapse the run achieved (T3, ADR-0054).

    Constructed only when the Pipeline has a Type-Consolidation Stage. The stream is that Stage's
    mixed output — the run's :class:`~latence_core.contracts.EntityMention`s and
    :class:`~latence_core.contracts.RelationMention`s after the remap, each carrying its
    pre-canonicalization type on ``raw_label`` when (and only when) it was rewritten.

    Everything is read off those records, so the section is a MEASUREMENT of what happened to the
    data rather than a restatement of what the Provider intended — a Stage that silently failed to
    remap would report ``N raw -> N canonical`` and an empty alias map, which is exactly the honest
    answer.
    """

    section_field: ClassVar[str] = "type_consolidation"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (CapabilityKind.TYPE_CONSOLIDATION,)

    def __init__(self) -> None:
        self._entities = _TypeDriftFold()
        self._relations = _TypeDriftFold()

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _TypeConsolidationAggregator | None:
        if not _has_kind(context.pipeline, cls.kinds):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if isinstance(record, EntityMention):
            self._entities.add(record)
        elif isinstance(record, RelationMention):
            self._relations.add(record)

    def finish(self) -> TypeConsolidationQuality | None:
        return TypeConsolidationQuality(
            raw_entity_types=len(self._entities.raw),
            canonical_entity_types=len(self._entities.canonical),
            mentions_remapped=self._entities.remapped,
            entity_alias_map=self._entities.alias_map(),
            raw_relation_types=len(self._relations.raw),
            canonical_relation_types=len(self._relations.canonical),
            relations_remapped=self._relations.remapped,
            relation_alias_map=self._relations.alias_map(),
        )


class _DisambiguationAggregator:
    """Disambiguation merge/link/normalisation statistics (S8).

    Constructed only when the Pipeline has a Disambiguation Stage. It folds TWO streams: the
    Disambiguation Stage's own mix of ENTITY-scope (canonical entities) and RELATION-scope
    (normalized relations) :class:`~latence_core.contracts.DisambiguationRecord`s, and the
    Entity-Extraction (or fused) output the Stage consumed — of which it needs only a COUNT
    (``mentions_in``, the number it resolved from).

    That second subscription is the duplicate the gather paid for twice: ``mentions`` and
    ``disambig_mentions_in`` were byte-identical ``_records_of_kinds`` calls over the same
    Capabilities, so the largest stream in any corpus (59,450 records for 200 documents per
    PERF-RESULTS) was decoded and retained twice. Sharing the stream with
    :class:`_EntityAggregator` makes it one decode and no retention.

    Rolls up: mentions in vs canonical entities out; the created/merged/singleton breakdown; applied
    vs below-policy merges (the no-silent-over-merge audit) broken down by strategy; KB-linked vs
    unlinked entities; and relations normalized (with the inverted count). Counts only (no raw
    text) — the S8 acceptance criterion, safe to persist.
    """

    section_field: ClassVar[str] = "disambiguation"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (
        CapabilityKind.DISAMBIGUATION,
        *_MENTION_KINDS,
    )

    def __init__(self) -> None:
        self._quality = DisambiguationQuality()
        self._entities = 0
        self._confidence_sum = 0.0

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _DisambiguationAggregator | None:
        if not _has_kind(context.pipeline, (CapabilityKind.DISAMBIGUATION,)):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        quality = self._quality
        if stage.capability in _MENTION_KINDS:
            if isinstance(record, EntityMention):
                quality.mentions_in += 1
            return
        if not isinstance(record, DisambiguationRecord):
            return
        if record.scope is DisambiguationScope.ENTITY and record.entity is not None:
            entity = record.entity
            self._entities += 1
            if entity.mention_count > 1:
                quality.entities_merged += 1
            else:
                quality.entities_singleton += 1
            if entity.linked:
                quality.entities_linked += 1
            else:
                quality.entities_unlinked += 1
            self._confidence_sum += entity.evidence.confidence
            for decision in entity.merge_decisions:
                if decision.applied:
                    quality.merges_applied += 1
                    key = decision.strategy.value
                    quality.merges_by_strategy[key] = quality.merges_by_strategy.get(key, 0) + 1
                else:
                    quality.merges_below_policy += 1
        elif record.scope is DisambiguationScope.RELATION and record.relation is not None:
            quality.relations_normalized += 1
            if record.relation.inverted:
                quality.relations_inverted += 1

    def finish(self) -> DisambiguationQuality | None:
        quality = self._quality
        quality.entities_out = self._entities
        if self._entities:
            quality.avg_entity_confidence = round(self._confidence_sum / self._entities, 6)
        quality.merges_by_strategy = dict(sorted(quality.merges_by_strategy.items()))
        return quality


class _GraphAggregator:
    """Graph-Assembly KG statistics (S9) + the separately-counted inferred edges (W2-linkpred).

    Constructed only when the Pipeline has a Graph Assembly Stage. The ASSERTED stream is that
    Stage's mix of NODE-scope and EDGE-scope :class:`~latence_core.contracts.GraphRecord`s:
    node/edge counts; nodes by entity type and edges by relation label; KB-linked node count; and
    — the S9
    acceptance criterion — evidence coverage, the fraction of edges whose Evidence names at least
    one source mention (1.0 = every edge is justified).

    The OPTIONAL Graph Completion Stage's PREDICTED EDGE-scope records (ADR-0037) are a SECOND
    subscription, counted into ``inferred_edge_count`` / ``inferred_edges_by_scorer`` and NEVER
    folded into ``edges_total`` / ``edges_by_label`` / ``evidence_coverage``. An inferred edge must
    never inflate the asserted-edge statistics — the enterprise-honesty crux — which is exactly why
    the two streams stay distinguishable in the fold: the ``stage`` says which one a record came
    from. A run with no Graph Completion Stage has ``inferred_edge_count == 0``. Only edges that
    actually carry ``properties["inferred"] is True`` are tallied (a defensive guard so nothing but
    a genuinely-inferred edge is counted). Counts only (no raw text); all maps sorted.
    """

    section_field: ClassVar[str] = "graph"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (
        CapabilityKind.GRAPH_ASSEMBLY,
        CapabilityKind.GRAPH_COMPLETION,
    )

    def __init__(self) -> None:
        self._quality = GraphQuality()
        self._nodes = 0
        self._edges = 0
        self._node_confidence = 0.0
        self._edge_confidence = 0.0

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _GraphAggregator | None:
        if not _has_kind(context.pipeline, (CapabilityKind.GRAPH_ASSEMBLY,)):
            return None
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if not isinstance(record, GraphRecord):
            return
        quality = self._quality
        if stage.capability is CapabilityKind.GRAPH_COMPLETION:
            if record.scope is not GraphScope.EDGE or record.edge is None:
                return
            if record.edge.properties.get("inferred") is not True:
                return
            quality.inferred_edge_count += 1
            scorer = str(record.edge.properties.get("scorer", "unknown"))
            quality.inferred_edges_by_scorer[scorer] = (
                quality.inferred_edges_by_scorer.get(scorer, 0) + 1
            )
            return
        if record.scope is GraphScope.NODE and record.node is not None:
            node = record.node
            self._nodes += 1
            quality.nodes_by_type[node.label] = quality.nodes_by_type.get(node.label, 0) + 1
            if node.linked:
                quality.nodes_linked += 1
            self._node_confidence += node.confidence
        elif record.scope is GraphScope.EDGE and record.edge is not None:
            edge = record.edge
            self._edges += 1
            quality.edges_by_label[edge.label] = quality.edges_by_label.get(edge.label, 0) + 1
            if edge.evidence.mention_ids:
                quality.edges_with_evidence += 1
            self._edge_confidence += edge.confidence

    def finish(self) -> GraphQuality | None:
        quality = self._quality
        quality.nodes_total = self._nodes
        quality.edges_total = self._edges
        if self._nodes:
            quality.avg_node_confidence = round(self._node_confidence / self._nodes, 6)
        if self._edges:
            quality.avg_edge_confidence = round(self._edge_confidence / self._edges, 6)
            quality.evidence_coverage = round(quality.edges_with_evidence / self._edges, 6)
        quality.nodes_by_type = dict(sorted(quality.nodes_by_type.items()))
        quality.edges_by_label = dict(sorted(quality.edges_by_label.items()))
        quality.inferred_edges_by_scorer = dict(sorted(quality.inferred_edges_by_scorer.items()))
        return quality


class _ContextEnrichmentAggregator:
    """The OPTIONAL Context Enrichment Stage's coherence roll-up (W13, ADR-0039).

    Constructed only when the Pipeline has a ``context_enrichment`` Stage (else the section is
    ``None`` — byte-identical to a stack without it). Counts-only (no content).
    ``chunks_enriched`` + ``avg_neighbors_per_chunk`` are read off the enriched
    :class:`~latence_core.contracts.ChunkRecord`s directly (a chunk with a non-empty
    :class:`~latence_core.contracts.ContextHeader` counts; its ``neighbor_node_ids`` are the
    query-time KG-expansion keys averaged).

    ``hub_entities_capped`` is the anti-bloat witness: the number of KG entities whose gated
    neighbor DEGREE exceeds the Stage's ``max_neighbors_per_entity`` cap — those entities contribute
    only their TOP-k relations wherever they appear (the hub-entity guard). It is computed from the
    assembled graph (the second subscription, shared with :class:`_GraphAggregator` at no extra
    decode) plus the Stage's config, so it is honest and reproducible. The degree map is bounded by
    the node count, not the edge count.
    """

    section_field: ClassVar[str] = "context_enrichment"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (
        CapabilityKind.CONTEXT_ENRICHMENT,
        CapabilityKind.GRAPH_ASSEMBLY,
    )

    def __init__(self, max_neighbors: int, min_edge_confidence: float) -> None:
        self._max_neighbors = max_neighbors
        self._min_edge_confidence = min_edge_confidence
        self._headers = 0
        self._neighbors = 0
        self._degree: dict[str, int] = {}

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _ContextEnrichmentAggregator | None:
        stage = next(
            (
                s
                for s in context.pipeline.stages
                if s.capability is CapabilityKind.CONTEXT_ENRICHMENT
            ),
            None,
        )
        if stage is None:
            return None
        return cls(
            int(stage.config.get("max_neighbors_per_entity", 4)),
            float(stage.config.get("min_edge_confidence", 0.0)),
        )

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if stage.capability is CapabilityKind.GRAPH_ASSEMBLY:
            # Degree = distinct gated edges touching the node (each edge is a neighbor from both
            # endpoints). Mirrors the Provider's edge-confidence gate.
            if self._max_neighbors <= 0:
                return
            if not isinstance(record, GraphRecord) or record.scope is not GraphScope.EDGE:
                return
            edge = record.edge
            if edge is None or edge.confidence < self._min_edge_confidence:
                return
            self._degree[edge.source_node_id] = self._degree.get(edge.source_node_id, 0) + 1
            self._degree[edge.target_node_id] = self._degree.get(edge.target_node_id, 0) + 1
            return
        if isinstance(record, ChunkRecord) and record.context_header is not None:
            self._headers += 1
            self._neighbors += len(record.context_header.neighbor_node_ids)

    def finish(self) -> ContextEnrichmentQuality | None:
        quality = ContextEnrichmentQuality()
        quality.chunks_enriched = self._headers
        if self._headers:
            quality.avg_neighbors_per_chunk = round(self._neighbors / self._headers, 6)
        if self._max_neighbors > 0:
            quality.hub_entities_capped = sum(
                1 for d in self._degree.values() if d > self._max_neighbors
            )
        return quality


class _GoldSetAggregator:
    """The gold-set harness result (S10) — the one aggregator that legitimately retains records.

    Constructed ONLY when the Pipeline declares a ``gold_set_uri``: the harness is opt-in (an
    adopter drops in 20-50 annotated documents), so a normal run constructs nothing here and pays
    nothing. When it is present, :func:`~latence_core.goldset.score_entities` /
    :func:`~latence_core.goldset.score_relations` index BOTH whole sets against the annotations —
    that is the algorithm, not a consumer defect — so this fold keeps the typed mentions and typed
    relations. It is the documented exception to the bounded-state rule, and it is bounded by an
    operator decision (a gold set is a few dozen documents' worth of extractions, not the corpus).

    Which stream a record came from matters and is decided from the ``stage``: mentions are the
    Entity-Extraction (or fused) output and relations the Relation-Extraction (or fused) output,
    exactly as the two gathers scoped them, so a relation Stage that also emitted mentions does not
    silently contribute them to the entity score.

    The mention side is required for both Stages (a relation's endpoints resolve through the mention
    spans), so a relations-only gold set with no Entity-Extraction Stage simply yields no relation
    matches — never an error.
    """

    section_field: ClassVar[str] = "goldset"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (
        CapabilityKind.ENTITY_EXTRACTION,
        CapabilityKind.RELATION_EXTRACTION,
        CapabilityKind.FUSED_ENTITY_RELATION,
    )

    def __init__(self, gold_set_uri: str, storage: Storage) -> None:
        self._gold_set_uri = gold_set_uri
        self._storage = storage
        self._mentions: list[EntityMention] = []
        self._relations: list[RelationMention] = []

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _GoldSetAggregator | None:
        if not context.pipeline.gold_set_uri:
            return None
        return cls(context.pipeline.gold_set_uri, context.storage)

    def observe(self, stage: Stage, record: StageOutput) -> None:
        if stage.capability in _MENTION_KINDS and isinstance(record, EntityMention):
            self._mentions.append(record)
        if stage.capability in _RELATION_KINDS and isinstance(record, RelationMention):
            self._relations.append(record)

    def finish(self) -> GoldSetEvaluation | None:
        gold = _load_gold_set(self._gold_set_uri, self._storage)
        typed_mentions = self._mentions
        typed_relations = self._relations
        # Map each mention/relation's parent document to its source file name so the gold
        # annotations (keyed by file name) bind to the run's records.
        file_name_of: dict[str, str] = {}
        for m in typed_mentions:
            file_name_of.setdefault(m.document_record_id, m.provenance.file_name)
        for r in typed_relations:
            file_name_of.setdefault(r.document_record_id, r.provenance.file_name)

        gold_files = set(gold.by_file_name())
        matched_files = {
            file_name_of[doc_id] for doc_id in file_name_of if file_name_of[doc_id] in gold_files
        }
        evaluation = GoldSetEvaluation(
            documents_annotated=len(gold.documents),
            documents_matched=len(matched_files),
            match_min_overlap=gold.match_min_overlap,
        )
        entity_overall, entity_labels = score_entities(typed_mentions, gold, file_name_of)
        evaluation.stages.append(_stage_score("entities", entity_overall, entity_labels))
        if gold.total_relations > 0:
            relation_overall, relation_labels = score_relations(
                typed_relations, typed_mentions, gold, file_name_of
            )
            evaluation.stages.append(_stage_score("relations", relation_overall, relation_labels))
        return evaluation


def _load_gold_set(uri: str, storage: Storage) -> GoldSet:
    """Load + validate a gold set from Storage (JSON or YAML), by file extension."""
    text = storage.read_text(uri)
    if uri.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        msg = f"Gold set at {uri!r} must be a JSON/YAML mapping at the top level."
        # A malformed operator-supplied gold set is a config error (#73); ConfigError is a
        # ValueError subclass, so pre-taxonomy ``except ValueError`` callers keep catching it.
        raise ConfigError(msg)
    return GoldSet.from_mapping(data)


def _stage_score(
    stage: str,
    overall: ConfusionCounts,
    per_label: dict[str, ConfusionCounts],
) -> StageScore:
    """Assemble a :class:`StageScore` from the confusion counts (overall + per label)."""
    return StageScore(
        stage=stage,
        true_positives=overall.true_positives,
        false_positives=overall.false_positives,
        false_negatives=overall.false_negatives,
        precision=overall.precision,
        recall=overall.recall,
        f1=overall.f1,
        per_label={
            label: LabelScore(
                true_positives=c.true_positives,
                false_positives=c.false_positives,
                false_negatives=c.false_negatives,
                precision=c.precision,
                recall=c.recall,
                f1=c.f1,
            )
            for label, c in per_label.items()
        },
    )


#: The built-in Quality-Report sections, in the order they are constructed and finished.
#:
#: This tuple is the ONLY place ``build()`` learns which sections exist. Adding a Stage's quality
#: metric is: one class implementing :class:`QualityAggregator`, one entry here. It used to be five
#: edits — a new model, a new optional field on :class:`~latence_core.quality.QualityReport`, an
#: entry in ``render()``'s section map, a bespoke render block, and a ``_x_quality`` method plus a
#: gather and a kwarg inside ``build()``. Two of those five are gone entirely (the render section
#: list is now derived from the report model's own section-marked fields) and the ``build()`` edits
#: collapsed into this line.
#:
#: Order is fixed rather than incidental: it decides the order records are fanned out to
#: subscribers, which is irrelevant to every section's arithmetic but is one less thing that can
#: drift between runs.
#:
#: Lane L2's ``CapabilityDescriptor`` will own a per-Capability ``quality_aggregator`` factory; the
#: wiring is to pass ``aggregators=`` to :class:`QualityReportBuilder` with the descriptors'
#: factories instead of this default. Nothing else in this module changes.
QUALITY_AGGREGATORS: tuple[QualityAggregatorFactory, ...] = (
    _ParseAggregator.for_run,
    _ChunkAggregator.for_run,
    _ScreeningAggregator.for_run,
    _EntityAggregator.for_run,
    _RelationAggregator.for_run,
    _RedactionAggregator.for_run,
    _ProfilingAggregator.for_run,
    _TypeConsolidationAggregator.for_run,
    _DisambiguationAggregator.for_run,
    _GraphAggregator.for_run,
    _ContextEnrichmentAggregator.for_run,
    _DriftAggregator.for_run,
    _GoldSetAggregator.for_run,
)

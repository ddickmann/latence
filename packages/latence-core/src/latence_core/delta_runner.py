"""DeltaRunner — the incremental-delta orchestration over the local Runner (S11, ADR-0018).

The DeltaRunner wraps :class:`~latence_core.runner.LocalRunner` and adds the lean-yet-enterprise
delta machinery on top of it, without forking the Runner (ADR-0018 reuses the existing seams):

* **A delta run** (:meth:`apply_delta`) executes the Pipeline once on the current source,
  computes a content-hash :class:`~latence_core.contracts.Delta` of the current manifest against
  the last committed Corpus Version (ports ``delta_detector``), feeds the parent + current
  corpus-level records through the :class:`~latence_core.capability.DeltaProcessor`
  (blocking-neighborhood affected-set recompute + retract-by-provenance, the merge-on-add /
  split-on-delete), applies the drift-triggered Reconciliation decision, and commits an
  immutable Version N+1 via the WAL-backed
  :class:`~latence_core.state.corpus_store.CorpusStore`. The delta churn — including the
  affected-set size vs the whole corpus — lands in the run's Quality Report.

  **What is affected-set-scoped, and the extraction boundary** (issue #42 → resolved by W17,
  ADR-0043). The DeltaProcessor's *committed corpus-level output* is affected-set-scoped: only
  the records in the blocking neighborhoods the delta touched are re-resolved and spliced, every
  untouched cluster is carried forward verbatim (byte-identical), and the churn records the
  affected-set size as the perf witness. **Since W17 the extraction pass is incremental too.** A
  delta REUSES each unchanged document's persisted doc-level records (parse/chunk/extract/redact)
  from the prior Version, excluding those documents at Source so only NEW/CHANGED documents run
  the doc-level Stages — which is also why only the new/changed chunks are ever INDUCED. The
  corpus-level Stages
  (disambiguate/graph/…) then run over the UNION of reused + freshly-extracted records — via the
  Runner's reuse overlay (:meth:`LocalRunner.run`'s ``reuse``) — so the committed corpus is
  byte-identical to a from-scratch run of the whole source, while the *extraction cost* is
  O(changed documents), not O(corpus). The churn's ``documents_reused`` / ``documents_extracted``
  are the measurable witness.

  **How a delta stays consistent with the base run (T4 replaces the frozen schema).** ADR-0043
  originally froze the ONE induced corpus schema and stamped it on the delta's chunks. Per-chunk
  induction (T1) produces no such schema — each chunk induces its own labels from its own text —
  so what a delta carries across runs is the **canonical TYPE vocabulary** (ADR-0054): the Corpus
  Version persists it, and a delta SEEDS its ``type_consolidation`` Stage with it so the new
  chunks' induced types canonicalize AGAINST it (a matching type aliases into the existing
  canonical entry; a genuinely new type appends one; **existing canonical ids/labels never
  churn**, so the KG's type space grows monotonically without rewriting history). The churn's
  ``canonical_types_added`` / ``type_aliases_added`` are the witness. The frozen schema survives
  only for an explicit ``granularity: corpus`` induction Stage, where the single shared vocabulary
  IS a genuine cross-document coupling. **``--reinduce``** now means *re-canonicalize from
  scratch*: no seed, no schema reuse, no doc-level reuse — the full O(corpus) rebuild, the rare
  path.
* **Retraction** (:meth:`retract`) is the default soft deletion: the deleted documents' derived
  records are tombstoned (excluded from the live set, retained on Storage for audit and
  rollback), affected clusters re-resolved, a new Version committed.
* **Purge** (:meth:`purge`) is the hard GDPR erasure: source + all derived records + Evidence
  physically removed, then the affected set reconciled — nothing derived survives.

Every operation is a Corpus Version transaction: re-running the same delta is idempotent (the
:class:`~latence_core.state.corpus_store.CorpusStore` fingerprint check), a failed run leaves
Version N intact (the WAL), and any prior Version is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from latence_core.capability import DeltaProcessor, DocumentEnumerator, ProviderRegistry
from latence_core.capability_descriptor import CORPUS_LEVEL_KINDS
from latence_core.contracts import (
    ChunkRecord,
    DeletionMode,
    Delta,
    DeltaChurn,
    DeltaEntry,
    DeltaKind,
    DisambiguationRecord,
    DocumentRecord,
    EntityMention,
    GraphRecord,
    InducedLabels,
    RelationMention,
    TypeKind,
    TypeVocabulary,
)
from latence_core.observability import OpenLineageEmitter
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.run_store import RunLayout, RunStore, runs_root_for
from latence_core.runner import LocalRunner, StageOutput, topological_order
from latence_core.state.corpus_store import CorpusStore
from latence_core.state.delta_detector import DeltaDetector, ManifestEntry, content_hash
from latence_core.state.incremental import compute_drift
from latence_core.storage import Storage
from latence_core.tracing import SpanSink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.contracts import Record
    from latence_core.quality import QualityReport

# The Provider that fulfils the DeltaProcessor Capability by default (the in-core reference).
DEFAULT_DELTA_PROVIDER = "delta.affected_set"

# The corpus-level record kinds a Corpus Version persists (the delta operates over these).
_CORPUS_LEVEL = (DisambiguationRecord, GraphRecord)

# W17 (ADR-0043): the per-document doc-level carriers the reuse store persists + overlays. These
# are reproducible from a document's own bytes + the frozen schema (parse/chunk/extract/redact),
# so an unchanged document's records are reused across a delta instead of re-extracted.
_DOC_LEVEL = (DocumentRecord, ChunkRecord, EntityMention, RelationMention)

# The Stage kinds whose checkpoint the reuse store must NOT read (T4): every CORPUS-LEVEL kind.
# A corpus-level Stage can ride a doc-level carrier — Type Consolidation re-emits the whole run's
# mentions/relations relabelled, and an Export sink re-emits its input verbatim — so the
# distinction has to be drawn by Stage KIND, not by carrier type (see
# :meth:`DeltaRunner._doc_level_records`).
#
# Derived from the Capability descriptors rather than hand-listed: this tuple and the
# phase-boundary check are the SAME fact ("does this Stage aggregate across documents?"), and when
# they were two hand-written lists they disagreed — ``TYPE_CONSOLIDATION`` was here but missing
# there. Sorted by CapabilityKind declaration order so the tuple is deterministic.
_CORPUS_LEVEL_KINDS: tuple[CapabilityKind, ...] = tuple(
    kind for kind in CapabilityKind if kind in CORPUS_LEVEL_KINDS
)


@dataclass(frozen=True)
class DeltaPreview:
    """The churn a delta WOULD apply, classified by content-addressed id (ADR-0061 dry-run).

    Produced by :meth:`DeltaRunner.preview` without running a single Stage: ``unchanged`` ids
    would be reused (extraction skipped), ``new_or_changed`` ids would run the doc-level Stages,
    ``missing`` ids would be classified DELETED. Ids are sorted for stable output.
    """

    parent_version: int | None
    unchanged: list[str]
    new_or_changed: list[str]
    missing: list[str]


class DeltaRunner:
    """Drives incremental deltas, retraction, and purge as Corpus Version transactions."""

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        *,
        drift_threshold: float = 0.5,
        delta_provider: str = DEFAULT_DELTA_PROVIDER,
        span_sink: SpanSink | None = None,
        lineage: OpenLineageEmitter | None = None,
    ) -> None:
        self._registry = registry or ProviderRegistry()
        # C3: the delta's own Runner is a fully-configured Runner, not a stripped one. Before this
        # the DeltaRunner constructed ``LocalRunner(registry)`` with no way to pass either
        # observability seam, so ``latence delta`` — the DAILY enterprise path — structurally could
        # not emit OpenLineage or per-Stage spans while ``latence process`` could. Both default to
        # ``None`` (off), so an un-wired delta run is byte- and behaviour-identical to before.
        self._runner = LocalRunner(self._registry, span_sink=span_sink, lineage=lineage)
        self._drift_threshold = drift_threshold
        self._delta_provider = delta_provider

    # -- public API ---------------------------------------------------------

    def preview(self, pipeline: Pipeline) -> DeltaPreview:
        """What :meth:`apply_delta` WOULD do to this corpus — no Stage runs, no commit (ADR-0061).

        The churn dry-run behind ``latence delta --dry-run``: reads the committed head Version's
        manifest, enumerates the current source with the SAME cheap, content-fetch-free pass the
        delta's reuse decision uses (:meth:`_current_document_ids`), and classifies by
        content-addressed id:

        * ``unchanged`` — present at the source AND committed (would be reused, not re-extracted);
        * ``new_or_changed`` — present at the source, not committed (would run the doc-level
          Stages; an in-place edit mints a new content id, so it lands here);
        * ``missing`` — committed but not found at the source (would be classified DELETED).

        Honest limits, inherited from the enumeration: a Source that can only *partially* name
        its documents without fetching them under-reports ``unchanged`` and over-reports
        ``missing`` (the delta itself uses the full union manifest, so its committed counts are
        exact) — the preview is a planning signal, never the commit. Durably deleted ids (prior
        retract/purge) are excluded exactly as the delta excludes them.
        """
        pipeline.require_runnable()
        store = CorpusStore(Storage.for_uri(pipeline.storage_uri), pipeline.storage_uri)
        parent_version = store.head_version()
        prior_deletions = self._prior_deletions(store, parent_version)
        previous_manifest = self._manifest_of(parent_version, store)
        current_ids = self._current_document_ids(pipeline, sorted(prior_deletions))
        committed = set(previous_manifest)
        return DeltaPreview(
            parent_version=parent_version,
            unchanged=sorted(current_ids & committed),
            new_or_changed=sorted(current_ids - committed),
            missing=sorted(committed - current_ids),
        )

    def apply_delta(
        self,
        pipeline: Pipeline,
        run_id: str,
        *,
        deletion_mode: DeletionMode = DeletionMode.RETRACTION,
        reinduce: bool = False,
    ) -> QualityReport:
        """Run the Pipeline on the current source and commit the resulting Corpus Version.

        The Runner executes the DAG over the current source with **incremental extraction** (W17,
        ADR-0043): each UNCHANGED document (its content-addressed id matches a document committed
        in the prior Version) is excluded at Source and its persisted doc-level records are REUSED,
        so only NEW/CHANGED documents run parse/chunk/extract/redact — and, since ``induce`` is one
        of those doc-level Stages, only their chunks are induced. The corpus-level Stages then run
        over the UNION (reused ∪ freshly-extracted) via the Runner's reuse overlay, with the
        ``type_consolidation`` Stage SEEDED from the Corpus Version's persisted canonical type
        vocabulary (T4) so the fresh chunks' induced types canonicalize against the committed type
        space instead of re-electing it. This method then computes the
        :class:`~latence_core.contracts.Delta` of the current (union) manifest against the last
        committed Version, applies the blocking-neighborhood affected-set recompute +
        drift-triggered Reconciliation to produce the *committed* corpus-level set (untouched
        clusters carried forward verbatim), commits Version N+1 with the refreshed doc-level reuse
        store + the EXTENDED type vocabulary, and folds the
        :class:`~latence_core.contracts.DeltaChurn` (with the ``documents_reused`` /
        ``documents_extracted`` and ``canonical_types_added`` / ``type_aliases_added`` witnesses)
        into the Quality Report. Returns that report.

        ``reinduce`` (the ``--reinduce`` escape hatch) is the **full rebuild**: re-induce every
        chunk AND re-canonicalize the type vocabulary from scratch (no seed), re-extracting every
        document. It is the correct-by-construction fallback for when the operator wants the
        vocabulary to move with the data — accepting that canonical type labels may be re-elected,
        which is exactly why it is not the default. It reuses nothing (``documents_reused == 0``).
        """
        pipeline.require_runnable()
        store = CorpusStore(Storage.for_uri(pipeline.storage_uri), pipeline.storage_uri)
        parent_version = store.head_version()
        # A document deleted (retracted or purged) in an earlier Version stays deleted: exclude it
        # at Source so a re-run over the still-present file does not silently resurrect it. (Adds
        # of NEW documents are unaffected — only the previously-deleted ids are excluded.)
        prior_deletions = self._prior_deletions(store, parent_version)
        parent_records = store.head_records()
        previous_manifest = self._manifest_of(parent_version, store)

        # W17/ADR-0061: decide cross-run reuse. Reuse engages from a committed parent whenever not
        # reinducing — for EVERY pipeline (the Runner's
        # document-ordered overlay fold makes the union reach every corpus-level Stage exactly as
        # a rebuild would deliver it). The frozen schema + the prior Version's doc-level store are
        # read once; the reusable documents are those whose current content-addressed id matches a
        # committed one (content-addressing ⇒ an id match is byte-identical ⇒ unchanged).
        reuse_on = parent_version is not None and not reinduce
        frozen_schema = store.head_schema() if reuse_on else None
        # T4: the canonical TYPE vocabulary the committed corpus already uses. It is read
        # independently of ``reuse_on`` — vocabulary stability is not a function of doc-level
        # reuse: even a run that reuses nothing (a bootstrap re-run, an all-changed delta) must
        # not rename the canonical types its committed KG is built from. Only ``--reinduce``
        # drops it, which is precisely what "re-canonicalize from scratch" means.
        seed_vocabulary = (
            store.head_type_vocabulary() if parent_version is not None and not reinduce else None
        )
        committed_doclevel = store.head_doclevel() if reuse_on else []
        committed_doc_ids = {r.provenance.document_id for r in committed_doclevel}
        current_doc_ids = self._current_document_ids(pipeline, sorted(prior_deletions))
        reusable_ids = current_doc_ids & committed_doc_ids
        reuse_records = [
            r for r in committed_doclevel if r.provenance.document_id in reusable_ids
        ]

        # Run only the NEW/CHANGED documents through the doc-level Stages: exclude the reused
        # documents (and the durable prior deletions) at Source, freeze the induced schema so the
        # fresh documents extract under the SAME labels as the reused ones, and hand the Runner the
        # reused doc-level records as the corpus-level overlay. T4: the new/changed documents'
        # chunks are induced PER CHUNK by the induce Stage itself (the unchanged ones are excluded
        # at Source, so they are never re-induced), and the Type-Consolidation Stage is seeded with
        # the committed vocabulary so those fresh types canonicalize AGAINST it.
        run_excluded = sorted(prior_deletions | reusable_ids)
        effective = self._with_source_exclusions(pipeline, run_excluded)
        effective = self._apply_frozen_schema(effective, frozen_schema)
        effective = self._apply_seed_vocabulary(effective, seed_vocabulary)
        report = self._runner.run(effective, run_id=run_id, reuse=reuse_records)
        outputs = self._collect_outputs(effective, run_id)

        # The union manifest: the freshly-extracted documents' records (from the run's checkpoints)
        # plus the reused documents' records — content-hashed by the same detector, so a reused
        # document lands UNCHANGED and only genuine adds/edits/deletes are classified.
        current_manifest = self._build_manifest(
            [*outputs, *cast("list[StageOutput]", reuse_records)]
        )

        diff = DeltaDetector.diff(previous_manifest, current_manifest)
        # The refreshed reuse store the next delta reads: every live document's doc-level records =
        # the freshly-extracted ones ∪ the reused ones carried forward. The frozen schema is
        # reused when present, else captured from this run's fresh chunks (first run / --reinduce).
        new_doclevel = [*self._doc_level_records(effective, run_id), *reuse_records]
        new_schema = frozen_schema or self._captured_schema(effective, run_id)
        # T4: the EXTENDED canonical type vocabulary this run emitted (the seed merged with the new
        # chunks' induced types, append-don't-churn). A pipeline with no Type-Consolidation Stage
        # emits none — carry the seed forward untouched rather than dropping it, so adding, then
        # removing, then re-adding the Stage never loses the corpus's committed type space.
        new_vocabulary = self._captured_type_vocabulary(effective, run_id) or seed_vocabulary
        churn = self._commit(
            store=store,
            parent_version=parent_version,
            parent_records=parent_records,
            current_records=self._corpus_level(outputs),
            diff_added=diff.added,
            diff_updated=diff.updated,
            diff_deleted=diff.deleted,
            diff_unchanged=diff.unchanged,
            current_manifest=current_manifest,
            deletion_mode=deletion_mode,
            doclevel_records=new_doclevel,
            induced_schema=new_schema,
            parent_vocabulary=seed_vocabulary,
            type_vocabulary=new_vocabulary,
            documents_reused=len(reusable_ids),
            # Derived from the documents this run ACTUALLY produced (the union manifest), not
            # from the discovery pass: a Source whose cheap enumeration is partial — SharePoint
            # Online can only content-address the items Graph reports a ``sha256Hash`` for —
            # under-reports ``current_doc_ids``, and the old
            # ``len(current_doc_ids) - len(reusable_ids)`` then reported 0 extracted for a run
            # that re-extracted the whole library (audit R3-connectors_ingestion-8). The manifest
            # counts every live document, so manifest − reused is exactly the freshly-extracted
            # set. Identical on a Source that enumerates completely.
            documents_extracted=len(current_manifest) - len(reusable_ids),
        )
        return self._attach_churn(pipeline, run_id, report, churn)

    def retract(
        self, pipeline: Pipeline, run_id: str, document_ids: list[str]
    ) -> QualityReport:
        """Soft-delete ``document_ids``: tombstone their derived records, re-resolve, commit.

        The default deletion disposition (CONTEXT ``Retraction``): the documents' derived
        records are excluded from the live set and the corpus, but **retained** on Storage
        (inside the retracted Version's committed record log) for audit and rollback, and any
        cluster whose merge was justified solely by a deleted document is split. Returns the
        delta run's Quality Report (with the retraction churn).
        """
        return self._delete(pipeline, run_id, document_ids, DeletionMode.RETRACTION)

    def purge(
        self, pipeline: Pipeline, run_id: str, document_ids: list[str]
    ) -> QualityReport:
        """Hard-delete ``document_ids`` (GDPR erasure): remove source + derived + Evidence.

        The hard deletion operation (CONTEXT ``Purge``): the deleted documents' derived records
        are physically removed (not retained), the affected set reconciled, and — so nothing
        derived survives — the retracted records are discarded rather than committed to the
        Version's record log. Returns the delta run's Quality Report (with the purge churn).
        """
        return self._delete(pipeline, run_id, document_ids, DeletionMode.PURGE)

    # -- deletion path ------------------------------------------------------

    def _delete(
        self,
        pipeline: Pipeline,
        run_id: str,
        document_ids: list[str],
        deletion_mode: DeletionMode,
    ) -> QualityReport:
        pipeline.require_runnable()
        store = CorpusStore(Storage.for_uri(pipeline.storage_uri), pipeline.storage_uri)
        parent_version = store.head_version()
        if parent_version is None:
            msg = "Cannot delete from an empty corpus (no committed Corpus Version)."
            raise ValueError(msg)
        parent_records = store.head_records()
        parent_version_obj = store.read_version(parent_version)
        previous_manifest = self._manifest_of(parent_version, store)

        deleted = sorted(set(document_ids) & set(previous_manifest))
        # The current manifest is the parent minus the deleted documents (a deletion changes no
        # other document's content).
        current_manifest = {
            doc_id: entry
            for doc_id, entry in previous_manifest.items()
            if doc_id not in deleted
        }
        diff = DeltaDetector.diff(previous_manifest, current_manifest)
        # Re-run the pipeline with the deleted documents EXCLUDED at Source: doc-level Stages never
        # see them and the corpus-level Stages re-resolve the survivors — a genuine split-on-delete
        # through the real pipeline (a merged cluster justified solely by a deleted document splits
        # because its bridging mentions are gone). The re-resolved corpus-level records are the
        # "current" set the DeltaProcessor patches against the parent. Prior deletions are carried
        # forward so sequential deletes are cumulative (a doc retracted at v2 stays gone at v3).
        # A Purge is a hard GDPR erasure — the source file(s) must be physically removed so
        # nothing derived survives. Resolve the purged document ids to their on-disk Source URIs
        # BEFORE the re-run (the files still exist now; the exclusion below only hides them from
        # this run) and delete them, then exclude them at Source too as a belt-and-braces guard.
        if deletion_mode is DeletionMode.PURGE:
            self._purge_source_files(pipeline, set(deleted))
            # ...and the same erasure has to reach the DERIVED tables the Export publishes outside
            # this storage tree, or the purged text stays readable there (audit R7-invariants-1).
            # Also before the re-run, so the re-run's own registration republishes the survivors.
            self._purge_derived_tables(pipeline)
        all_deleted = sorted(set(deleted) | self._prior_deletions(store, parent_version))
        excluded_pipeline = self._with_source_exclusions(pipeline, all_deleted)
        # T4: a RETRACTION seeds the survivors' re-canonicalization with the committed vocabulary
        # — the surviving KG keeps its type names, so a soft delete never retypes nodes it did not
        # touch. A PURGE deliberately does NOT seed: a hard GDPR erasure re-derives the survivors'
        # schema from the survivors alone, so no cluster the purged documents created can persist
        # by carry-forward. Both then commit the vocabulary the re-run actually emitted.
        parent_vocabulary = (
            store.read_type_vocabulary(parent_version)
            if deletion_mode is DeletionMode.RETRACTION
            else None
        )
        excluded_pipeline = self._apply_seed_vocabulary(excluded_pipeline, parent_vocabulary)
        report = self._runner.run(excluded_pipeline, run_id=run_id)
        current_records = self._corpus_level(self._collect_outputs(excluded_pipeline, run_id))
        # W17: the deletion path re-extracts the survivors from scratch (it does not reuse — a
        # split-on-delete must re-resolve the survivors through the real pipeline), so the refreshed
        # reuse store is exactly the survivors' fresh doc-level records; the deleted documents'
        # records never enter it (excluded at Source). The frozen schema is carried forward from
        # the parent so a subsequent add can still reuse against a stable vocabulary. On a Purge the
        # expunge below sweeps every prior Version's doc-level store too (nothing survives).
        surviving_doclevel = self._doc_level_records(excluded_pipeline, run_id)
        churn = self._commit(
            store=store,
            parent_version=parent_version,
            parent_records=parent_records,
            current_records=current_records,
            diff_added=[],
            diff_updated=[],
            diff_deleted=diff.deleted,
            diff_unchanged=diff.unchanged,
            current_manifest=current_manifest,
            deletion_mode=deletion_mode,
            document_count=parent_version_obj.document_count - len(deleted),
            intentional_deletion=True,
            doclevel_records=surviving_doclevel,
            induced_schema=store.read_schema(parent_version),
            parent_vocabulary=parent_vocabulary,
            type_vocabulary=(
                self._captured_type_vocabulary(excluded_pipeline, run_id) or parent_vocabulary
            ),
            documents_extracted=len(current_manifest),
        )
        if deletion_mode is DeletionMode.PURGE and deleted:
            # Expunge the purged documents' derived records from EVERY records.jsonl under the
            # corpus — committed Versions AND abandoned WAL staging dirs a crash left behind
            # (H-D1) — re-fingerprinting each rewritten Version's manifest so the idempotency
            # check is exact, and wipe the run checkpoints that hold their raw text. After this
            # neither an audit read of a prior Version, a checkpoint scan, nor a filesystem/backup
            # read of an abandoned staging file can resurrect the erased content. The head Version
            # the commit above just wrote already omits them.
            store.expunge_records(set(deleted))
            # ...and what every Version's manifest still SAYS about them: the file name and the
            # content hash go, and only the content-addressed id stays, as the suppression
            # tombstone that keeps the erased document from being re-ingested.
            store.expunge_document_listings(set(deleted))
            self._purge_run_artifacts(pipeline)
        return self._attach_churn(pipeline, run_id, report, churn)

    def _purge_source_files(self, pipeline: Pipeline, purged_ids: set[str]) -> None:
        """Physically delete the on-disk Source files whose content-addressed id is purged.

        Walks each local-folder Source's path, content-hashes each file to its document id (the
        same sha256-of-bytes id the Source stamps), and removes the ones being purged — the
        physical half of the GDPR erasure (the logical half is the exclusion + expunge).
        """
        import hashlib

        storage = Storage.for_uri(pipeline.storage_uri)
        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.SOURCE:
                continue
            path = (stage.config or {}).get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                files = list(storage.walk_files(path))
            except FileNotFoundError:
                continue
            for uri in files:
                try:
                    data = storage.read_bytes(uri)
                except FileNotFoundError:
                    continue
                doc_id = "sha256:" + hashlib.sha256(data).hexdigest()
                if doc_id in purged_ids:
                    storage.remove(uri)

    @staticmethod
    def _purge_derived_tables(pipeline: Pipeline) -> None:
        """Erase every DERIVED catalog table this Pipeline's Exports publish into (#184).

        A Purge that stops at this storage tree is not an erasure. The optional ``iceberg`` block
        (:mod:`latence_core.iceberg`) copies the whole corpus — ``content`` included — into the
        catalog's own warehouse, which is *outside* ``storage_uri`` and therefore untouched by
        :meth:`_purge_source_files`, :meth:`~...CorpusStore.expunge_records` and
        :meth:`_purge_run_artifacts` alike. In the DEFAULT ``mode: append`` the post-purge re-export
        merely appends a second data file, leaving the purged rows in the table's CURRENT snapshot:
        an ordinary ``table.scan()`` returned erased text while the tutorial promised removal
        "everywhere" (audit R7-invariants-1).

        Called BEFORE the survivor re-run, exactly like :meth:`_purge_source_files`: the re-run's
        own registration then republishes the surviving corpus onto an emptied table, so ``append``
        converges on the survivors without the operator having to reconfigure the mode.

        Off-by-default is preserved end-to-end: an Export with no ``iceberg`` block builds no target
        (``build_iceberg_output`` returns ``None``), so a purge on the default stack does exactly
        what it did before and never imports ``pyiceberg`` (ADR-0016). When the block IS present,
        every failure — an unreachable catalog, a missing optional dependency, a refused commit — is
        a typed error that FAILS THE PURGE (ADR-0034), because a right-to-be-forgotten command must
        never exit 0 over a warehouse it could not reach.
        """
        from latence_core.iceberg import build_iceberg_output

        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.EXPORT:
                continue
            config = stage.config or {}
            target = build_iceberg_output(
                config.get("iceberg"), default_table=str(config.get("basename", "records"))
            )
            if target is not None:
                target.erase()

    def _purge_run_artifacts(self, pipeline: Pipeline) -> None:
        """Wipe every run's artifact tree under ``_latence/runs`` (the Purge erasure step).

        A run directory holds a purged document's raw text in several re-derivable artifacts — the
        per-Stage checkpoints (raw chunk/mention text), the Export sink's ``records.jsonl`` (the
        derived records with their Evidence), the run manifest (source file names), quarantine and
        screening logs. None of it is durable audit state (the committed Corpus Version is the
        truth), so a right-to-be-forgotten erasure removes the whole tree. The current purge run's
        own tree is wiped too; its Quality Report (counts only, no raw text) is re-written straight
        after by :meth:`_attach_churn`.
        """
        storage = Storage.for_uri(pipeline.storage_uri)
        runs_root = runs_root_for(pipeline.storage_uri)
        if not storage.exists(runs_root):
            return
        for run_entry in storage.ls(runs_root):
            storage.remove_tree(run_entry)

    @staticmethod
    def _with_source_exclusions(pipeline: Pipeline, excluded_document_ids: list[str]) -> Pipeline:
        """A copy of the Pipeline with the deleted document ids excluded at every Source Stage.

        The Retraction/Purge re-run reads everything BUT the deleted documents (ADR-0018), so the
        exclusion is injected into each Source Stage's ``exclude_document_ids`` config — the
        Source seam already supports it (ADR-0014 thin Source config), so no Runner fork is needed.
        """
        data = pipeline.model_dump()
        for stage in data["stages"]:
            if stage["capability"] == CapabilityKind.SOURCE.value:
                config = dict(stage.get("config") or {})
                existing = list(config.get("exclude_document_ids") or [])
                config["exclude_document_ids"] = sorted(set(existing) | set(excluded_document_ids))
                stage["config"] = config
        return Pipeline.model_validate(data)

    # -- transaction --------------------------------------------------------

    def _commit(
        self,
        *,
        store: CorpusStore,
        parent_version: int | None,
        parent_records: list[Record],
        current_records: list[Record],
        diff_added: list[str],
        diff_updated: list[str],
        diff_deleted: list[str],
        diff_unchanged: list[str],
        current_manifest: dict[str, ManifestEntry],
        deletion_mode: DeletionMode,
        document_count: int | None = None,
        intentional_deletion: bool = False,
        doclevel_records: list[Record] | None = None,
        induced_schema: InducedLabels | None = None,
        parent_vocabulary: TypeVocabulary | None = None,
        type_vocabulary: TypeVocabulary | None = None,
        documents_reused: int = 0,
        documents_extracted: int = 0,
    ) -> DeltaChurn:
        """Compute drift, run the DeltaProcessor, and commit the next Corpus Version."""
        total_docs = len(current_manifest) if document_count is None else document_count
        changed = len(diff_added) + len(diff_updated) + len(diff_deleted)
        drift = compute_drift(changed, max(total_docs, changed), self._drift_threshold)

        entries = [
            *(_entry(DeltaKind.ADDED, d, current_manifest) for d in diff_added),
            *(_entry(DeltaKind.UPDATED, d, current_manifest) for d in diff_updated),
            *(DeltaEntry(kind=DeltaKind.DELETED, document_id=d) for d in diff_deleted),
            *(_entry(DeltaKind.UNCHANGED, d, current_manifest) for d in diff_unchanged),
        ]
        delta = Delta(
            parent_version=parent_version,
            deletion_mode=deletion_mode,
            intentional_deletion=intentional_deletion,
            entries=entries,
            drift=drift.drift,
            triggered_reconciliation=drift.triggered,
        )

        processor = self._load_processor()
        outcome = processor.apply_delta(parent_records, current_records, delta)

        version = store.commit(
            parent_version=parent_version,
            records=outcome.records,
            delta_json=delta.model_dump_json(),
            document_count=total_docs,
            reconciled=drift.triggered,
            doclevel_records=doclevel_records,
            induced_schema=induced_schema,
            type_vocabulary=type_vocabulary,
        )

        vocab_churn = _type_vocabulary_churn(parent_vocabulary, type_vocabulary)
        records_purged = len(outcome.retracted) if deletion_mode is DeletionMode.PURGE else 0
        records_retracted = (
            len(outcome.retracted) if deletion_mode is DeletionMode.RETRACTION else 0
        )
        return DeltaChurn(
            from_version=parent_version,
            to_version=version.version,
            documents_added=len(diff_added),
            documents_updated=len(diff_updated),
            documents_deleted=len(diff_deleted),
            documents_unchanged=len(diff_unchanged),
            documents_reused=documents_reused,
            documents_extracted=documents_extracted,
            canonical_types_added=vocab_churn.canonical_added,
            type_aliases_added=vocab_churn.aliases_added,
            canonical_types_total=vocab_churn.canonical_total,
            entities_created=outcome.entities_created,
            entities_merged=outcome.entities_merged,
            entities_split=outcome.entities_split,
            edges_added=outcome.edges_added,
            edges_retracted=outcome.edges_retracted,
            records_retracted=records_retracted,
            records_purged=records_purged,
            affected_set_size=outcome.affected_set_size,
            corpus_record_count=outcome.corpus_record_count,
            drift=drift.drift,
            reconciliation_triggered=drift.triggered,
            deletion_mode=deletion_mode,
        )

    def _load_processor(self) -> DeltaProcessor:
        provider_cls = self._registry.load(self._delta_provider)
        provider = provider_cls({})
        if not isinstance(provider, DeltaProcessor):
            msg = (
                f"Provider {self._delta_provider!r} does not satisfy the DeltaProcessor capability."
            )
            raise TypeError(msg)
        return provider

    # -- Runner output collection ------------------------------------------

    def _collect_outputs(self, pipeline: Pipeline, run_id: str) -> list[StageOutput]:
        """Re-read the run's per-Stage checkpoints so the delta sees the extracted records.

        The Runner persists each Stage's output as a checkpoint; a delta reads the Disambiguation
        + Graph + Document + mention/relation checkpoints back to build the manifest and the
        corpus-level record set. Reusing the checkpoints (rather than re-running) keeps the delta
        cheap and is exact, since the checkpoints are the run's committed output.

        An EXPORT Stage is a sink that passes its input records through unchanged, so its
        checkpoint is a verbatim copy of an upstream producer's — reading it too would
        double-count every record. Export checkpoints are therefore skipped; the producing Stage
        (Disambiguation/Graph/Parse/…) is the single source of truth. A record is de-duplicated by
        ``record_id`` defensively so a fan-out DAG never double-feeds the delta.

        **A Type-Consolidation Stage is skipped for the same reason, and one more** (T4). It also
        re-emits its input carrier — but over ``fresh ∪ reused``, so its checkpoint carries the
        REUSED documents' mentions as well. Those reused mentions are added to the manifest a
        second time by the caller (which appends ``reuse_records``), which doubled every reused
        document's derived texts, flipped its content hash, and got it classified UPDATED on every
        delta: ``documents_unchanged`` went to zero and the affected set became the whole corpus,
        silently undoing the ADR-0018 bound for any stack wiring the Stage (the enterprise-SOTA
        one does). Skipping it is also the better manifest: change detection then hashes the
        extractor's RAW labels, which are a function of the document alone, rather than canonical
        labels that legitimately move as the corpus grows and would flip an untouched document's
        hash.
        """
        store = RunStore(Storage.for_uri(pipeline.storage_uri), pipeline, run_id)
        seen: set[str] = set()
        collected: list[StageOutput] = []
        for stage in pipeline.stages:
            if stage.capability in (
                CapabilityKind.EXPORT,
                CapabilityKind.TYPE_CONSOLIDATION,
            ):
                continue
            # C6: STREAM the checkpoint rather than buffering it. The buffered read returned
            # the whole Stage's records as a list before this line even began filtering them, so
            # peak memory was O(largest checkpoint) on top of the collected set — in the one
            # operation whose entire purpose is to be O(changed). ``RunStore.read`` decodes one
            # line at a time (a missing checkpoint yields nothing) and yields byte-identical
            # records in the same order — through the owned interface, no private reach-through.
            for record in store.read(stage):
                # A ParserInput (Source/Intake output) carries no record_id; key it by its
                # source document id. Everything else is a Record keyed by record_id.
                rid = getattr(record, "record_id", None) or record.provenance.document_id
                key = f"{type(record).__name__}:{rid}"
                if key in seen:
                    continue
                seen.add(key)
                collected.append(record)
        return collected

    @staticmethod
    def _corpus_level(outputs: list[StageOutput]) -> list[Record]:
        """The corpus-level derived records (Disambiguation + Graph) a Version holds."""
        return [r for r in outputs if isinstance(r, _CORPUS_LEVEL)]

    # -- W17 incremental extraction (frozen schema + doc-level reuse) --------

    def _current_document_ids(self, pipeline: Pipeline, excluded: list[str]) -> set[str]:
        """The content-addressed ids of the current live source documents (minus ``excluded``).

        Enumerates each Source Stage and collects every ``provenance.document_id`` the Source
        would emit, dropping the durably-deleted ids in ``excluded``. Content-addressing makes an
        id match a byte-identical match, so intersecting these with a prior Version's committed
        doc ids yields exactly the UNCHANGED (reusable) documents; the rest are new/changed and
        get re-extracted.

        **This pass must not fetch content.** It runs BEFORE the reuse exclusions are known, so
        anything it transfers is transferred unconditionally — a discovery pass that downloaded
        every document would make a delta run over an unchanged remote library cost a full run
        (in fact more: the changed documents would then be fetched a second time by the run
        itself), erasing the whole saving ADR-0018 promises. A Source that can name its documents
        without fetching them therefore implements
        :class:`~latence_core.capability.DocumentEnumerator` and that is used here; only a Source
        that cannot falls back to ``produce`` — correct, just not free. Either way the durable
        deletions are pushed into the Source's ``exclude_document_ids`` first, so even the
        fallback never re-fetches a document this run is going to drop anyway.

        An enumeration may be **partial** (a Source that can name only some of its documents
        without fetching them omits the rest — see ``DocumentEnumerator``). The returned set is
        therefore a lower bound on the live documents: it is used only to pick the *reusable*
        ones, so a missing id costs a reuse and nothing else. Nothing downstream may treat it as
        the corpus census — that is the union manifest.
        """
        storage = Storage.for_uri(pipeline.storage_uri)
        excluded_set = set(excluded)
        enumeration_pipeline = self._with_source_exclusions(pipeline, sorted(excluded_set))
        ids: set[str] = set()
        for stage in enumeration_pipeline.stages:
            if stage.capability is not CapabilityKind.SOURCE:
                continue
            provider_cls = self._registry.load(stage.provider)
            provider = provider_cls(stage.config or {})
            if isinstance(provider, DocumentEnumerator):
                candidates: Iterator[str] = provider.document_ids(storage)
            else:
                produce = getattr(provider, "produce", None)
                if produce is None:  # pragma: no cover - a Source always produces
                    continue
                candidates = (i.provenance.document_id for i in produce(storage))
            for doc_id in candidates:
                if doc_id not in excluded_set:
                    ids.add(doc_id)
        return ids

    @staticmethod
    def _apply_frozen_schema(
        pipeline: Pipeline, schema: InducedLabels | None
    ) -> Pipeline:
        """Stamp the FROZEN CORPUS schema on a ``granularity: corpus`` induction Stage (W17/T4).

        The freeze exists to break ONE coupling: at ``granularity: corpus`` a single vocabulary is
        induced from ALL chunks, so adding a document could shift it and invalidate the unchanged
        documents' extraction. Freezing it makes the fresh documents extract under the SAME labels
        as the reused ones. ``frozen_induced_labels`` short-circuits induction entirely (no LLM
        call, :meth:`LLMLabelInducer.induce`).

        **T4 scopes the freeze to exactly that coupling.** Under per-chunk induction (the default
        since T1) a chunk's labels are a function of its own text alone, so there is no corpus
        coupling to break: an unchanged document's records are byte-identical to re-extracting it
        with no freeze at all, and freezing would be actively wrong — it would stamp the parent's
        corpus union on the delta's chunks and reintroduce the very label broadcast the redesign
        removed. Cross-run consistency for the per-chunk path comes from the PERSISTED CANONICAL
        TYPE VOCABULARY instead (:meth:`_apply_seed_vocabulary`), which earns it downstream rather
        than imposing it upstream. So a non-corpus induction Stage is left untouched even when the
        parent Version froze a corpus schema (a pipeline can hold both kinds of Stage).

        ``schema is None`` (first run, ``--reinduce``, a per-chunk-only or induction-free pipeline)
        returns the Pipeline unchanged, so induction runs normally.
        """
        if schema is None:
            return pipeline
        data = pipeline.model_dump()
        touched = False
        for stage in data["stages"]:
            if stage["capability"] != CapabilityKind.SCHEMA_INDUCTION.value:
                continue
            config = dict(stage.get("config") or {})
            if str(config.get("granularity", "doc")).lower() != "corpus":
                continue
            config["frozen_induced_labels"] = schema.model_dump(mode="json")
            stage["config"] = config
            touched = True
        return Pipeline.model_validate(data) if touched else pipeline

    @staticmethod
    def _apply_seed_vocabulary(
        pipeline: Pipeline, vocabulary: TypeVocabulary | None
    ) -> Pipeline:
        """Seed every ``type_consolidation`` Stage with the committed type vocabulary (T4).

        The per-chunk replacement for "freeze one schema". A delta induces the new/changed chunks'
        labels freshly (per chunk — that is the whole point of the redesign), which means the
        delta's raw type surfaces are NOT drawn from the parent run's vocabulary. Canonicalizing
        them from scratch would re-elect canonical labels over the delta's own salience and could
        RENAME a type the committed KG is already built from. Seeding forbids that: the Stage
        merges its fresh vocabulary INTO the committed one (append-don't-churn), so a fresh type
        that matches an existing cluster aliases to it, a genuinely new type appends a new
        canonical entry, and no existing canonical id/label moves. The corpus's type space
        therefore grows monotonically without rewriting history.

        Injected as Stage config (rather than handed to the Provider object) because the Pipeline
        is the run's whole input: the persisted manifest then records exactly which vocabulary the
        run canonicalized against, and replaying it reproduces the run.

        ``vocabulary is None`` (first run, ``--reinduce`` ⇒ re-canonicalize from scratch, or a
        parent that committed none) returns the Pipeline unchanged.
        """
        if vocabulary is None:
            return pipeline
        data = pipeline.model_dump()
        touched = False
        for stage in data["stages"]:
            if stage["capability"] == CapabilityKind.TYPE_CONSOLIDATION.value:
                config = dict(stage.get("config") or {})
                config["seed_vocabulary"] = vocabulary.model_dump(mode="json")
                stage["config"] = config
                touched = True
        return Pipeline.model_validate(data) if touched else pipeline

    def _doc_level_records(self, pipeline: Pipeline, run_id: str) -> list[Record]:
        """The run's per-document Stage outputs, for the next delta's reuse store (W17).

        Reads each doc-level Stage's checkpoint and collects the per-document carriers
        (parse→DocumentRecord, chunk/redact→ChunkRecord, entity→EntityMention,
        relation→RelationMention). A ChunkRecord is de-duplicated **last-writer-wins in
        topological order**, so a chunk→…→redact chain persists the FINAL (masked) chunk variant —
        the one that flows to a doc-level Export — never the earlier raw one. DocumentRecords /
        mentions / relations have unique ids across Stages, so the order is immaterial for them.

        **Only doc-level Stages are read** (``_CORPUS_LEVEL_KINDS`` are skipped). An Export sink
        re-emits its input verbatim, so reading it would double-count. More subtly, a corpus-level
        accumulator can re-emit a DOC-LEVEL CARRIER: Type Consolidation consumes ``fresh ∪ reused``
        mentions and returns all of them relabelled, so its checkpoint holds the REUSED documents'
        mentions too — collecting from it would put every reused document's records into the store
        twice (once here, once via the caller's ``reuse_records``), and the next delta would then
        feed the resolver duplicate mentions. Skipping it is also the semantically right line: the
        reuse store substitutes for RE-RUNNING the doc-level Stages, and a corpus-level decision is
        not a doc-level Stage output — it is recomputed (over the seeded vocabulary) every run.
        """
        store = RunStore(Storage.for_uri(pipeline.storage_uri), pipeline, run_id)
        collected: dict[str, Record] = {}
        for stage in topological_order(list(pipeline.stages)):
            if stage.capability in _CORPUS_LEVEL_KINDS:
                continue
            # C6: streamed, not buffered — same reason as ``_collect_outputs``.
            for record in store.read(stage):
                if not isinstance(record, _DOC_LEVEL):
                    continue
                collected[f"{type(record).__name__}:{record.record_id}"] = record
        return list(collected.values())

    def _captured_schema(self, pipeline: Pipeline, run_id: str) -> InducedLabels | None:
        """The induced CORPUS schema this run produced, to freeze in the Corpus Version (W17).

        For a ``schema_induction`` Stage running at ``granularity: corpus`` (the one shared
        vocabulary for the whole corpus — the coupling), every chunk carries the SAME
        :class:`InducedLabels`, so the first populated one IS the corpus schema. Returns it so the
        commit freezes it and the next delta reuses it. A non-corpus granularity (per-doc/per-chunk
        schema rides in each document's reused chunk records, no corpus-level freeze), a pipeline
        with no induction, or an all-fail-open run yields ``None`` — no corpus schema to freeze.
        """
        store = RunStore(Storage.for_uri(pipeline.storage_uri), pipeline, run_id)
        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.SCHEMA_INDUCTION:
                continue
            if str((stage.config or {}).get("granularity", "doc")).lower() != "corpus":
                continue
            # A corpus-granularity induction Stage checkpoints ChunkRecords; the explicit
            # ``record_type`` keeps the decode contract this reader has always used.
            for record in store.read(stage, record_type=ChunkRecord):
                if isinstance(record, ChunkRecord) and record.induced_labels is not None:
                    return record.induced_labels
        return None

    def _captured_type_vocabulary(
        self, pipeline: Pipeline, run_id: str
    ) -> TypeVocabulary | None:
        """The canonical TYPE vocabulary this run emitted, to commit in the Corpus Version (T4).

        Read from the Type-Consolidation Stage's persisted sidecar (ADR-0054 §6 / ADR-0017: files
        are the source of truth) rather than recomputed, so what is committed is exactly what the
        Stage decided and what the records were remapped onto. When the run was SEEDED the sidecar
        already holds seed ⊕ fresh — the merge happens inside the Provider — so this is the
        EXTENDED vocabulary, not the delta's own.

        A pipeline with several Type-Consolidation Stages (an unusual but legal DAG) commits the
        union in Stage order, since each Stage canonicalizes a different slice of the same corpus
        and the committed vocabulary must describe all of it. ``None`` when the pipeline has no
        such Stage — the caller then carries the parent's vocabulary forward untouched.
        """
        storage = Storage.for_uri(pipeline.storage_uri)
        captured: TypeVocabulary | None = None
        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.TYPE_CONSOLIDATION:
                continue
            emitted = self._runner.read_type_vocabulary(storage, pipeline, run_id, stage.name)
            if emitted is None:
                continue
            captured = emitted if captured is None else captured.merge(emitted)
        return captured

    # -- manifest -----------------------------------------------------------

    @staticmethod
    def _build_manifest(outputs: list[StageOutput]) -> dict[str, ManifestEntry]:
        """Build the current content-hash manifest from the run's documents + derived records.

        The ported detector hashes each document's markdown plus its derived entity texts and
        relation labels (:func:`content_hash`), so a re-extraction that changes the derived
        records flips the hash. Documents come from the Parse checkpoint; the derived texts/labels
        from the mention/relation checkpoints, grouped by ``document_id``.
        """
        markdown: dict[str, str] = {}
        file_names: dict[str, str] = {}
        entity_texts: dict[str, list[str]] = {}
        relation_labels: dict[str, list[str]] = {}
        for record in outputs:
            if isinstance(record, DocumentRecord):
                doc_id = record.provenance.document_id
                markdown[doc_id] = record.content
                file_names[doc_id] = record.provenance.file_name
            elif isinstance(record, EntityMention):
                entity_texts.setdefault(record.provenance.document_id, []).append(record.text)
            elif isinstance(record, RelationMention):
                relation_labels.setdefault(record.provenance.document_id, []).append(record.label)
        manifest: dict[str, ManifestEntry] = {}
        for doc_id, text in markdown.items():
            manifest[doc_id] = ManifestEntry(
                document_id=doc_id,
                content_hash=content_hash(
                    doc_id, text, entity_texts.get(doc_id), relation_labels.get(doc_id)
                ),
                file_name=file_names.get(doc_id, ""),
            )
        return manifest

    def _manifest_of(
        self, parent_version: int | None, store: CorpusStore
    ) -> dict[str, ManifestEntry]:
        """Reconstruct the parent Version's manifest (its committed doc-hash map), or empty.

        The manifest is persisted with the Corpus Version's delta (its entries carry the content
        hash), so the parent manifest is read straight off the committed :class:`Delta` rather
        than re-hashed — exact and cheap. On bootstrap (no parent) the manifest is empty.

        A bootstrap Version's own entries list every document it committed; a subsequent delta
        lists only that delta's view, so the walk reads the chain newest-first and a nearer
        Version's entry always wins (``setdefault``). **A DELETED entry is a tombstone, not a
        skip** (ticket-03): editing a file retires its old content-addressed id as DELETED, and
        when that tombstone was merely skipped the ancestor chain resurrected the retired id into
        the reconstructed manifest — every later delta then re-classified it DELETED
        (``documents_deleted`` over-counted forever after an edit). A tombstone therefore shadows
        the id for all OLDER Versions; a NEWER re-add (edit-then-revert mints the same
        content-addressed id again) is seen first in the walk and wins over the older tombstone,
        so reverted content still counts as live.
        """
        manifest: dict[str, ManifestEntry] = {}
        tombstoned: set[str] = set()
        current = parent_version
        while current is not None:
            version = store.read_version(current)
            for entry in version.delta.entries:
                if entry.kind is DeltaKind.DELETED:
                    if entry.document_id not in manifest:
                        tombstoned.add(entry.document_id)
                    continue
                if entry.content_hash is None or entry.document_id in tombstoned:
                    continue
                manifest.setdefault(
                    entry.document_id,
                    ManifestEntry(
                        document_id=entry.document_id,
                        content_hash=entry.content_hash,
                        file_name=entry.file_name,
                    ),
                )
            current = version.parent_version
        return manifest

    @staticmethod
    def _prior_deletions(store: CorpusStore, version: int | None) -> set[str]:
        """Every *intentionally* deleted document id across the committed Version chain.

        Walks from ``version`` back to the root, unioning the DELETED entries of only those
        Versions whose Delta was an explicit user-requested retract/purge
        (``delta.intentional_deletion``). Those tombstones are durable: the deleted document stays
        excluded at Source on every subsequent re-run so it cannot silently resurrect.

        Crucially it does NOT union a DELETED entry that is merely the delete-half of an ordinary
        in-place content edit. Document ids are content-addressed (sha256 of the file bytes), so
        editing a file mints a new id and retires the old one; the detector classifies that retired
        content-id DELETED even though the user retracted nothing. Blacklisting it would silently
        drop the content forever if the file were later reverted to it (edit-then-revert). Only an
        ``intentional_deletion`` DELETED id is carried forward. ``None`` (empty corpus) yields no
        deletions.
        """
        deleted: set[str] = set()
        current = version
        while current is not None:
            corpus_version = store.read_version(current)
            if corpus_version.delta.intentional_deletion:
                for entry in corpus_version.delta.entries:
                    if entry.kind is DeltaKind.DELETED:
                        deleted.add(entry.document_id)
            current = corpus_version.parent_version
        return deleted

    # -- report -------------------------------------------------------------

    def _attach_churn(
        self, pipeline: Pipeline, run_id: str, report: QualityReport, churn: DeltaChurn
    ) -> QualityReport:
        """Fold the delta churn into the Quality Report and re-persist it (S11 AC)."""
        report.delta = churn
        storage = Storage.for_uri(pipeline.storage_uri)
        report_uri = RunLayout(pipeline.storage_uri, run_id).report_uri
        storage.write_atomic(report_uri, report.to_json().encode("utf-8"))
        return report


@dataclass(frozen=True)
class _TypeVocabularyChurn:
    """How a delta EXTENDED the committed canonical type vocabulary (T4 churn, counts only)."""

    canonical_added: int = 0
    aliases_added: int = 0
    canonical_total: int = 0


def _type_vocabulary_churn(
    parent: TypeVocabulary | None, current: TypeVocabulary | None
) -> _TypeVocabularyChurn:
    """Diff the committed type vocabularies into the delta's churn counters (T4).

    ``canonical_added`` is the number of canonical entries in ``current`` whose ``(kind,
    canonical)`` key the parent did not have — the genuinely NEW types this delta introduced.
    ``aliases_added`` is the number of raw labels that are new to the vocabulary but landed in a
    cluster that already existed — the append-don't-churn half, i.e. the drifted surfaces that
    aliased INTO the committed type space instead of renaming anything. Measured from the two
    committed vocabularies, so a Stage that silently failed to extend reports zeros rather than
    claiming an extension the corpus does not show.
    """
    if current is None:
        return _TypeVocabularyChurn()
    parent_canonicals = (
        set() if parent is None else {(t.kind, t.canonical) for t in parent.types}
    )
    parent_labels: set[tuple[TypeKind, str]] = (
        set()
        if parent is None
        else {(t.kind, label) for t in parent.types for label in t.labels}
    )
    canonical_added = sum(
        1 for t in current.types if (t.kind, t.canonical) not in parent_canonicals
    )
    aliases_added = sum(
        1
        for t in current.types
        if (t.kind, t.canonical) in parent_canonicals
        for label in t.labels
        if (t.kind, label) not in parent_labels
    )
    return _TypeVocabularyChurn(
        canonical_added=canonical_added,
        aliases_added=aliases_added,
        canonical_total=len(current.types),
    )


def _entry(
    kind: DeltaKind, doc_id: str, manifest: dict[str, ManifestEntry]
) -> DeltaEntry:
    entry = manifest.get(doc_id)
    return DeltaEntry(
        kind=kind,
        document_id=doc_id,
        file_name=entry.file_name if entry else "",
        content_hash=entry.content_hash if entry else "sha256:unknown",
    )

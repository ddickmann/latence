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
  delta FREEZES the induced corpus schema in the Corpus Version (so extraction labels are stable
  across the delta) and REUSES each unchanged document's persisted doc-level records
  (parse/chunk/extract/redact) from the prior Version, excluding those documents at Source so
  only NEW/CHANGED documents run the doc-level Stages. The corpus-level Stages
  (disambiguate/graph/…) then run over the UNION of reused + freshly-extracted records — via the
  Runner's reuse overlay (:meth:`LocalRunner.run`'s ``reuse``) — so the committed corpus is
  byte-identical to a from-scratch run of the whole source, while the *extraction cost* is
  O(changed documents), not O(corpus). The churn's ``documents_reused`` / ``documents_extracted``
  are the measurable witness. **The one caveat**: an explicit ``--reinduce`` (or a first run with
  no frozen schema) re-runs induction over the full current corpus and re-extracts every document
  — the correct-by-construction fallback to the old O(corpus) behavior, by design. Reuse also
  falls back to full extraction for a pipeline whose stages are not purely per-document
  reproducible (a ``context_enrichment`` Stage — its per-chunk output depends on the mutable
  graph — see :meth:`DeltaRunner._reuse_supported`).
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

from typing import TYPE_CHECKING, cast

from latence_core.capability import DeltaProcessor, ProviderRegistry
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
)
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.runner import LocalRunner, StageOutput, topological_order
from latence_core.state.corpus_store import CorpusStore
from latence_core.state.delta_detector import DeltaDetector, ManifestEntry, content_hash
from latence_core.state.incremental import compute_drift
from latence_core.storage import Storage

if TYPE_CHECKING:
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


class DeltaRunner:
    """Drives incremental deltas, retraction, and purge as Corpus Version transactions."""

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        *,
        drift_threshold: float = 0.5,
        delta_provider: str = DEFAULT_DELTA_PROVIDER,
    ) -> None:
        self._registry = registry or ProviderRegistry()
        self._runner = LocalRunner(self._registry)
        self._drift_threshold = drift_threshold
        self._delta_provider = delta_provider

    # -- public API ---------------------------------------------------------

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
        ADR-0043): each UNCHANGED document (its content-addressed id matches a document committed in
        the prior Version, and the induced schema is frozen) is excluded at Source and its
        persisted doc-level records are REUSED, so only NEW/CHANGED documents run
        parse/chunk/extract/redact; the corpus-level Stages then run over the UNION (reused ∪
        freshly-extracted) via the Runner's reuse overlay. This method then computes the
        :class:`~latence_core.contracts.Delta` of the current (union) manifest against the last
        committed Version, applies the blocking-neighborhood affected-set recompute +
        drift-triggered Reconciliation to produce the *committed* corpus-level set (untouched
        clusters carried forward verbatim), commits Version N+1 with the refreshed doc-level reuse
        store + frozen schema, and folds the :class:`~latence_core.contracts.DeltaChurn` (with the
        ``documents_reused`` / ``documents_extracted`` witness) into the Quality Report. Returns
        that report.

        ``reinduce`` (the ``--reinduce`` escape hatch): re-run schema induction over the full
        current corpus and re-extract every document — the correct-by-construction fallback to the
        old O(corpus) behavior when the operator wants the induced vocabulary to move with the
        data. It reuses nothing (``documents_reused == 0``).
        """
        store = CorpusStore(Storage.for_uri(pipeline.storage_uri), pipeline.storage_uri)
        parent_version = store.head_version()
        # A document deleted (retracted or purged) in an earlier Version stays deleted: exclude it
        # at Source so a re-run over the still-present file does not silently resurrect it. (Adds
        # of NEW documents are unaffected — only the previously-deleted ids are excluded.)
        prior_deletions = self._prior_deletions(store, parent_version)
        parent_records = store.head_records()
        previous_manifest = self._manifest_of(parent_records, parent_version, store)

        # W17: decide cross-run reuse. Reuse engages only from a committed parent, when not
        # reinducing, and for a pipeline whose doc-level Stages are purely per-document reproducible
        # (:meth:`_reuse_supported`). The frozen schema + the prior Version's doc-level store are
        # read once; the reusable documents are those whose current content-addressed id matches a
        # committed one (content-addressing ⇒ an id match is byte-identical ⇒ unchanged).
        reuse_on = (
            parent_version is not None and not reinduce and self._reuse_supported(pipeline)
        )
        frozen_schema = store.head_schema() if reuse_on else None
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
        # reused doc-level records as the corpus-level overlay.
        run_excluded = sorted(prior_deletions | reusable_ids)
        effective = self._with_source_exclusions(pipeline, run_excluded)
        effective = self._apply_frozen_schema(effective, frozen_schema)
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
            documents_reused=len(reusable_ids),
            documents_extracted=len(current_doc_ids) - len(reusable_ids),
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
        store = CorpusStore(Storage.for_uri(pipeline.storage_uri), pipeline.storage_uri)
        parent_version = store.head_version()
        if parent_version is None:
            msg = "Cannot delete from an empty corpus (no committed Corpus Version)."
            raise ValueError(msg)
        parent_records = store.head_records()
        parent_version_obj = store.read_version(parent_version)
        previous_manifest = self._manifest_of(parent_records, parent_version, store)

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
        all_deleted = sorted(set(deleted) | self._prior_deletions(store, parent_version))
        excluded_pipeline = self._with_source_exclusions(pipeline, all_deleted)
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
        runs_root = f"{pipeline.storage_uri.rstrip('/')}/_latence/runs"
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
        )

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
        """
        storage = Storage.for_uri(pipeline.storage_uri)
        seen: set[str] = set()
        collected: list[StageOutput] = []
        for stage in pipeline.stages:
            if stage.capability is CapabilityKind.EXPORT:
                continue
            ckpt = (
                f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
                f"/checkpoints/{stage.name}.jsonl"
            )
            if not storage.exists(ckpt):
                continue
            record_type = self._runner._output_type(stage, pipeline)  # noqa: SLF001
            for record in self._runner._read_checkpoint(storage, ckpt, record_type):  # noqa: SLF001
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

    @staticmethod
    def _reuse_supported(pipeline: Pipeline) -> bool:
        """True if this pipeline's doc-level Stages are purely per-document reproducible (W17).

        Cross-run doc-level reuse is sound when a document's reused records are byte-identical to
        re-extracting it — which holds for the per-document Stages (parse/chunk/extract/redact)
        under a frozen schema. A :class:`CapabilityKind.CONTEXT_ENRICHMENT` Stage is NOT
        per-document — a chunk's ``context_header`` summarizes the mutable assembled graph — so a
        reused document's enriched output could go stale across a delta. Rather than hazard a
        fragile partial reuse (which could corrupt the exported corpus), reuse is disabled for such
        a pipeline and it falls back to full extraction (correct by construction). Every other
        reference pipeline (the KG spine, the RAG chunk/redact/export spine) is supported.
        """
        return not any(
            stage.capability is CapabilityKind.CONTEXT_ENRICHMENT for stage in pipeline.stages
        )

    def _current_document_ids(self, pipeline: Pipeline, excluded: list[str]) -> set[str]:
        """The content-addressed ids of the current live source documents (minus ``excluded``).

        Enumerates each Source Stage (reading + hashing the source bytes, the cheap content-address
        pass — NOT the expensive per-document extraction the reuse skips) and collects every
        ``provenance.document_id`` the Source would emit, dropping the durably-deleted ids in
        ``excluded``. Content-addressing makes an id match a byte-identical match, so intersecting
        these with a prior Version's committed doc ids yields exactly the UNCHANGED (reusable)
        documents; the rest are new/changed and get re-extracted.
        """
        storage = Storage.for_uri(pipeline.storage_uri)
        excluded_set = set(excluded)
        ids: set[str] = set()
        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.SOURCE:
                continue
            provider_cls = self._registry.load(stage.provider)
            provider = provider_cls(stage.config or {})
            produce = getattr(provider, "produce", None)
            if produce is None:  # pragma: no cover - a Source always produces
                continue
            for parser_input in produce(storage):
                doc_id = parser_input.provenance.document_id
                if doc_id not in excluded_set:
                    ids.add(doc_id)
        return ids

    @staticmethod
    def _apply_frozen_schema(
        pipeline: Pipeline, schema: InducedLabels | None
    ) -> Pipeline:
        """Return a Pipeline whose ``schema_induction`` Stage stamps the FROZEN schema (W17).

        When a frozen corpus schema is being reused, every ``schema_induction`` Stage's config is
        given ``frozen_induced_labels`` so its Provider stamps that single schema on every chunk
        and makes NO LLM call (:meth:`LLMLabelInducer.induce`) — the freshly-extracted documents
        then use the SAME labels as the reused ones, keeping the unchanged documents' extraction
        valid (the coupling ADR-0043 breaks). ``schema is None`` (first run, ``--reinduce``, or a
        pipeline with no induction) returns the Pipeline unchanged, so induction runs normally.
        """
        if schema is None:
            return pipeline
        data = pipeline.model_dump()
        touched = False
        for stage in data["stages"]:
            if stage["capability"] == CapabilityKind.SCHEMA_INDUCTION.value:
                config = dict(stage.get("config") or {})
                config["frozen_induced_labels"] = schema.model_dump(mode="json")
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
        Export checkpoints are skipped (a sink re-emits its input verbatim — double-counting).
        """
        storage = Storage.for_uri(pipeline.storage_uri)
        collected: dict[str, Record] = {}
        for stage in topological_order(list(pipeline.stages)):
            if stage.capability is CapabilityKind.EXPORT:
                continue
            ckpt = (
                f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
                f"/checkpoints/{stage.name}.jsonl"
            )
            if not storage.exists(ckpt):
                continue
            record_type = self._runner._output_type(stage, pipeline)  # noqa: SLF001
            for record in self._runner._read_checkpoint(storage, ckpt, record_type):  # noqa: SLF001
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
        storage = Storage.for_uri(pipeline.storage_uri)
        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.SCHEMA_INDUCTION:
                continue
            if str((stage.config or {}).get("granularity", "doc")).lower() != "corpus":
                continue
            ckpt = (
                f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
                f"/checkpoints/{stage.name}.jsonl"
            )
            if not storage.exists(ckpt):
                continue
            for record in self._runner._read_checkpoint(storage, ckpt, ChunkRecord):  # noqa: SLF001
                if isinstance(record, ChunkRecord) and record.induced_labels is not None:
                    return record.induced_labels
        return None

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
        self, parent_records: list[Record], parent_version: int | None, store: CorpusStore
    ) -> dict[str, ManifestEntry]:
        """Reconstruct the parent Version's manifest (its committed doc-hash map), or empty.

        The manifest is persisted with the Corpus Version's delta (its entries carry the content
        hash), so the parent manifest is read straight off the committed :class:`Delta` rather
        than re-hashed — exact and cheap. On bootstrap (no parent) the manifest is empty.
        """
        if parent_version is None:
            return {}
        version = store.read_version(parent_version)
        manifest: dict[str, ManifestEntry] = {}
        for entry in version.delta.entries:
            if entry.kind is DeltaKind.DELETED or entry.content_hash is None:
                continue
            manifest[entry.document_id] = ManifestEntry(
                document_id=entry.document_id,
                content_hash=entry.content_hash,
                file_name=entry.file_name,
            )
        # A bootstrap parent Version's own manifest lists every document it committed; but a
        # subsequent delta only lists that delta's changed docs, so carry forward the parent's
        # unchanged docs by re-hashing from the parent records is not possible (records are
        # corpus-level). The delta entries above are authoritative for the docs this parent saw;
        # its own parent chain fills the rest — read transitively.
        if version.parent_version is not None:
            for doc_id, ancestor_entry in self._manifest_of(
                parent_records, version.parent_version, store
            ).items():
                manifest.setdefault(doc_id, ancestor_entry)
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
        report_uri = (
            f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}/quality-report.json"
        )
        storage.write_atomic(report_uri, report.to_json().encode("utf-8"))
        return report


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

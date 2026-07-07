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

  **What is affected-set-scoped today, and what is not** (issue #42, honest boundary). The
  DeltaProcessor's *committed corpus-level output* is genuinely affected-set-scoped: only the
  records in the blocking neighborhoods the delta touched are re-resolved and spliced, every
  untouched cluster is carried forward verbatim (byte-identical), and the churn records the
  affected-set size as the perf witness. What is NOT yet incremental is the **extraction pass**:
  each delta uses a fresh ``run_id``, so the Runner re-parses / re-chunks / re-extracts the WHOLE
  current source from scratch (there is no cross-run doc-level checkpoint reuse), then the
  DeltaProcessor scopes the committed set. So the *stored* corpus honours ADR-0018's "recompute
  only the affected set" property, but the *extraction cost* of a delta is still O(corpus).
  Cross-run doc-level incremental execution (skipping unchanged documents' doc-level Stages) is a
  Runner-core change tracked as follow-up work, not delivered here.
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

from typing import TYPE_CHECKING

from latence_core.capability import DeltaProcessor, ProviderRegistry
from latence_core.contracts import (
    DeletionMode,
    Delta,
    DeltaChurn,
    DeltaEntry,
    DeltaKind,
    DisambiguationRecord,
    DocumentRecord,
    EntityMention,
    GraphRecord,
    RelationMention,
)
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.runner import LocalRunner, StageOutput
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
    ) -> QualityReport:
        """Run the Pipeline on the current source and commit the resulting Corpus Version.

        The Runner executes the whole DAG over the current source (extraction is not yet
        cross-run incremental — see the module docstring); this method then computes the
        :class:`~latence_core.contracts.Delta` of the current manifest against the last committed
        Version, applies the blocking-neighborhood affected-set recompute + drift-triggered
        Reconciliation to produce the *committed* corpus-level set (affected-set-scoped: untouched
        clusters carried forward verbatim), commits Version N+1, and folds the
        :class:`~latence_core.contracts.DeltaChurn` (with the affected-set size) into the Quality
        Report. Returns that report.
        """
        store = CorpusStore(Storage.for_uri(pipeline.storage_uri), pipeline.storage_uri)
        parent_version = store.head_version()
        # A document deleted (retracted or purged) in an earlier Version stays deleted: exclude it
        # at Source so a re-run over the still-present file does not silently resurrect it. (Adds
        # of NEW documents are unaffected — only the previously-deleted ids are excluded.)
        prior_deletions = self._prior_deletions(store, parent_version)
        effective = self._with_source_exclusions(pipeline, sorted(prior_deletions))

        report = self._runner.run(effective, run_id=run_id)
        outputs = self._collect_outputs(effective, run_id)

        parent_records = store.head_records()
        current_manifest = self._build_manifest(outputs)
        previous_manifest = self._manifest_of(parent_records, parent_version, store)

        diff = DeltaDetector.diff(previous_manifest, current_manifest)
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

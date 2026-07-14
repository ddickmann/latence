"""CorpusStore — the WAL-backed Corpus Version transaction (S11, ADR-0018).

Composes the ported :class:`~latence_core.state.write_ahead_log.WriteAheadLog` into the
read-N → write-N+1 transaction the S11 issue's first sub-task calls for: read the last
committed Corpus Version, produce the next one, and commit it atomically so that

* **re-running the same delta is idempotent** — a delta whose fingerprint matches the head
  Version is a no-op (the store returns the existing head rather than minting a duplicate);
* **a failed run leaves Version N intact** — the next version's records are staged under the
  WAL and only become live when the manifest commits last; a crash before commit is an
  abandoned transaction (WAL recovery), never a torn head;
* **any prior Version is inspectable** — every committed Version's manifest and its derived
  records are retained on Storage and readable by number for audit and rollback.

Layout under ``<storage_uri>/_latence/corpus/``:

* ``wal/v<N>/MANIFEST.json`` — the WAL commit manifest (the linearisation point);
* ``wal/v<N>/records.jsonl`` — the committed derived record set for Version N;
* ``wal/v<N>/version.json`` — the :class:`~latence_core.contracts.CorpusVersion` contract;
* ``wal/HEAD`` — a pointer to the highest committed version (a fast-path read; the WAL scan
  is authoritative and used to rebuild HEAD on recovery).

Records are (de)serialised as one JSON object per line (like the Runner's checkpoints), tagged
with a ``_kind`` discriminator so the polymorphic derived set (canonical entities, normalized
relations, graph nodes/edges) round-trips losslessly.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from latence_core.contracts import (
    ChunkRecord,
    CorpusVersion,
    DisambiguationRecord,
    DocumentRecord,
    EntityMention,
    GraphRecord,
    InducedLabels,
    Record,
    RelationMention,
)
from latence_core.errors import StorageError
from latence_core.state.write_ahead_log import WriteAheadLog

if TYPE_CHECKING:
    from latence_core.storage import Storage

_RECORDS_NAME = "records.jsonl"
_VERSION_NAME = "version.json"
_HEAD_NAME = "HEAD"
# W17 (ADR-0043): the cross-run doc-level reuse store + the frozen induced schema, staged
# alongside the corpus-level records under the SAME WAL transaction (so a delta commits them
# atomically with Version N+1). ``doclevel.jsonl`` holds every LIVE document's doc-level Stage
# outputs (parse/chunk/extract/redact records) keyed by their content-addressed
# ``provenance.document_id``, so a subsequent ``apply_delta`` can reuse an unchanged document's
# records instead of re-parsing/-chunking/-extracting it. ``schema.json`` holds the induced
# corpus schema (:class:`InducedLabels`) the run FROZE — reused by default so extraction labels
# stay stable across a delta (the coupling ADR-0043 breaks); absent ⇒ no induction / no freeze.
_DOCLEVEL_NAME = "doclevel.jsonl"
_SCHEMA_NAME = "schema.json"

# The record kinds a Corpus Version persists. Discriminated on read by a ``_kind`` tag so the
# polymorphic derived set round-trips. Doc-level carriers (chunks/mentions/relations/redactions)
# are re-derivable from the run and are NOT persisted in the corpus version — the corpus version
# holds the corpus-level derived truth (canonical entities, normalized relations, graph).
_RECORD_TYPES: dict[str, type[Record]] = {
    "disambiguation": DisambiguationRecord,
    "graph": GraphRecord,
}
_KIND_BY_TYPE: dict[type[Record], str] = {v: k for k, v in _RECORD_TYPES.items()}

# W17: the doc-level carrier kinds the reuse store persists — the per-document Stage outputs a
# delta reuses for an unchanged document (parse→DocumentRecord, chunk/redact→ChunkRecord,
# entity→EntityMention, relation→RelationMention). Discriminated on read by the SAME ``_kind``
# tag convention as the corpus-level records. These carry raw text / masked content, so the
# Purge sweep (:meth:`expunge_records`) neutralises them by ``document_id`` reach exactly like the
# corpus-level log — nothing derived (or raw) from a purged document survives.
_DOCLEVEL_TYPES: dict[str, type[Record]] = {
    "document": DocumentRecord,
    "chunk": ChunkRecord,
    "entity_mention": EntityMention,
    "relation_mention": RelationMention,
}
_DOCLEVEL_KIND_BY_TYPE: dict[type[Record], str] = {v: k for k, v in _DOCLEVEL_TYPES.items()}


class CorpusStoreError(StorageError):
    """Raised on an inconsistent Corpus Version operation (e.g. inspecting an uncommitted one).

    A :class:`~latence_core.errors.StorageError` (a Corpus-Version/Storage inconsistency, #73)
    — and, transitively, still a ``RuntimeError``, so the pre-taxonomy ``except RuntimeError`` /
    ``except CorpusStoreError`` call sites keep catching it.
    """


class CorpusStore:
    """The read-N → write-N+1 Corpus Version transaction over a Storage root (WAL-backed)."""

    def __init__(self, storage: Storage, storage_uri: str) -> None:
        self._storage = storage
        self._root = f"{storage_uri.rstrip('/')}/_latence/corpus"
        self._wal = WriteAheadLog(storage, f"{self._root}/wal")

    # -- head / inspection --------------------------------------------------

    def head_version(self) -> int | None:
        """The highest committed Corpus Version number, or ``None`` before the first commit."""
        recovery = self._wal.recover()
        return recovery.committed

    def read_version(self, version: int) -> CorpusVersion:
        """Read a committed Corpus Version's manifest contract (inspectable for audit/rollback)."""
        uri = self._wal.staged_uri(version, _VERSION_NAME)
        if not self._wal.is_committed(version) or not self._storage.exists(uri):
            msg = f"Corpus Version {version} is not committed / not inspectable."
            raise CorpusStoreError(msg)
        return CorpusVersion.model_validate_json(self._storage.read_text(uri))

    def read_records(self, version: int) -> list[Record]:
        """Read a committed Corpus Version's live derived record set (polymorphic, tagged)."""
        uri = self._wal.staged_uri(version, _RECORDS_NAME)
        if not self._wal.is_committed(version) or not self._storage.exists(uri):
            msg = f"Corpus Version {version} has no committed records."
            raise CorpusStoreError(msg)
        return self._decode_records(self._storage.read_text(uri))

    def head_records(self) -> list[Record]:
        """The live derived record set of the head Version (empty before the first commit)."""
        head = self.head_version()
        if head is None:
            return []
        return self.read_records(head)

    # -- doc-level reuse store + frozen schema (W17, ADR-0043) ---------------

    def read_doclevel(self, version: int) -> list[Record]:
        """Read a committed Version's doc-level reuse store (every live doc's Stage outputs).

        The per-document parse/chunk/extract/redact records the delta persisted for reuse, keyed
        (in each record's ``provenance.document_id``) by the content-addressed document id. Empty
        for a Version committed before W17 (its ``doclevel.jsonl`` is absent) or a genuinely empty
        corpus — either way there is simply nothing to reuse and the next delta re-extracts.
        """
        uri = self._wal.staged_uri(version, _DOCLEVEL_NAME)
        if not self._wal.is_committed(version) or not self._storage.exists(uri):
            return []
        return self._decode_doclevel(self._storage.read_text(uri))

    def head_doclevel(self) -> list[Record]:
        """The head Version's doc-level reuse store (empty before the first commit / pre-W17)."""
        head = self.head_version()
        if head is None:
            return []
        return self.read_doclevel(head)

    def read_schema(self, version: int) -> InducedLabels | None:
        """Read a committed Version's frozen induced corpus schema, or ``None`` if it froze none.

        ``None`` when the Version ran no schema induction (or predates W17), so the next delta has
        no schema to reuse and either induces fresh or runs with static config labels — the
        byte-identical backward-compatible path.
        """
        uri = self._wal.staged_uri(version, _SCHEMA_NAME)
        if not self._wal.is_committed(version) or not self._storage.exists(uri):
            return None
        text = self._storage.read_text(uri).strip()
        if not text:
            return None
        return InducedLabels.model_validate_json(text)

    def head_schema(self) -> InducedLabels | None:
        """The head Version's frozen induced corpus schema (``None`` before the first commit)."""
        head = self.head_version()
        if head is None:
            return None
        return self.read_schema(head)

    # -- transaction --------------------------------------------------------

    def commit(
        self,
        *,
        parent_version: int | None,
        records: list[Record],
        delta_json: str,
        document_count: int,
        reconciled: bool,
        doclevel_records: list[Record] | None = None,
        induced_schema: InducedLabels | None = None,
    ) -> CorpusVersion:
        """Commit the next Corpus Version transactionally (read-N → write-N+1).

        ``records`` is the live derived set after the delta; ``delta_json`` the serialised
        :class:`~latence_core.contracts.Delta` that produced it. The next version number is
        ``head + 1`` (0 on bootstrap). The records + version manifest are staged under the WAL,
        then the WAL manifest is written LAST (the atomic swap): a crash before that leaves the
        prior head intact. Returns the committed :class:`~latence_core.contracts.CorpusVersion`.

        W17 (ADR-0043): ``doclevel_records`` (every live document's doc-level Stage outputs, for
        cross-run reuse) and ``induced_schema`` (the frozen corpus schema) are staged in the SAME
        WAL transaction, so a Version's reuse store + schema commit atomically with its
        corpus-level records — a crash leaves the prior head (and its reuse store) intact. Both
        are optional (``None`` ⇒ an empty doclevel / no frozen schema) so a caller that does not
        yet persist them is unaffected. On the idempotent no-op re-run below the head is returned
        as-is (its reuse store already stands), so nothing is re-staged.
        """
        head = self.head_version()
        version = 0 if head is None else head + 1
        fingerprint = self._fingerprint(records)

        # Idempotent re-run: re-applying the same corpus state (adding/updating/deleting nothing)
        # yields a live set whose fingerprint matches the current head — the delta was a no-op, so
        # return the head rather than mint a duplicate Version. The fingerprint (a content hash
        # over the committed record ids, timestamp excluded) is the exact "same corpus" signal;
        # the parent_version differs on a re-run (the head advanced), so it is deliberately NOT
        # part of the equality — only the resulting corpus content is.
        if head is not None:
            current = self.read_version(head)
            if current.fingerprint == fingerprint:
                return current

        from latence_core.contracts import Delta

        delta = Delta.model_validate_json(delta_json)
        corpus_version = CorpusVersion(
            version=version,
            parent_version=parent_version,
            delta=delta,
            record_count=len(records),
            document_count=document_count,
            fingerprint=fingerprint,
            reconciled=reconciled,
            created_ts=round(time.time(), 6),
        )

        self._wal.begin(version)
        records_uri = self._wal.stage_text(version, _RECORDS_NAME, self._encode_records(records))
        version_uri = self._wal.stage_text(version, _VERSION_NAME, corpus_version.model_dump_json())
        # W17: stage the doc-level reuse store + frozen schema under the same transaction. Always
        # staged (an empty doclevel is the empty string) so every committed Version has a
        # well-defined reuse store the next delta reads back without a presence heuristic.
        doclevel_uri = self._wal.stage_text(
            version, _DOCLEVEL_NAME, self._encode_doclevel(doclevel_records or [])
        )
        schema_uri = self._wal.stage_text(
            version,
            _SCHEMA_NAME,
            "" if induced_schema is None else induced_schema.model_dump_json(),
        )
        self._wal.commit(
            version,
            files={
                "records": records_uri,
                "version": version_uri,
                "doclevel": doclevel_uri,
                "schema": schema_uri,
            },
            metadata={
                "parent": "none" if parent_version is None else str(parent_version),
                "fingerprint": fingerprint,
                "reconciled": str(reconciled).lower(),
            },
        )
        # Fast-path HEAD pointer (the WAL scan remains authoritative; this is a convenience read).
        self._storage.write_atomic(
            f"{self._root}/wal/{_HEAD_NAME}", str(version).encode("utf-8")
        )
        return corpus_version

    # -- purge (physical erasure) ------------------------------------------

    def expunge_records(self, document_ids: set[str]) -> int:
        """Physically remove every derived record reaching ``document_ids`` from EVERY
        ``records.jsonl`` under the corpus — committed AND abandoned (the Purge / GDPR-erasure
        step, ADR-0018 / ADR-0029; hardened by H-D1).

        Corpus Versions are normally immutable audit history, but a Purge is a hard
        right-to-be-forgotten erasure that trumps that immutability: the S11 acceptance criterion
        requires that *nothing derived survives* — so a purged document's canonical entities,
        relations, graph nodes/edges, and their Evidence must not linger in any ``records.jsonl``
        anywhere under the corpus root, whether or not the version that staged it ever committed.
        A record is expunged when its Evidence-provenance reach touches any purged document
        (:func:`~latence_core.state.graph_patcher._record_reach`). Each rewritten record log is
        written atomically. Returns the number of records physically removed.

        Two residue surfaces are swept:

        * **Committed Versions** — every ``v<N>/records.jsonl`` with a ``MANIFEST.json``. For each
          one that actually loses a record, its ``version.json`` is **re-fingerprinted and
          re-counted** (:meth:`_reconcile_manifest`) so the stored
          :class:`~latence_core.contracts.CorpusVersion` describes the *post-purge* record set, not
          the pre-purge one. This is required for correctness, not just tidiness: the head
          Version's stored fingerprint drives :meth:`commit`'s idempotency check, so leaving it
          stale makes re-applying the identical delta mint a spurious duplicate Version instead of
          the documented no-op (audit correctness S3). The recomputed value reuses
          :meth:`_fingerprint` — the same helper ``commit`` uses — so the idempotency compare is
          exact. Erasure already trumps immutability; the manifest is rewritten to reflect the
          erased reality (see the ADR-0018/0029 H-D1 note).

        * **Abandoned transactions** — a crash mid-commit leaves a ``v<N>/`` staging dir with a
          fully-staged ``records.jsonl`` and **no** manifest (:meth:`WriteAheadLog.recover` reports
          these as ``abandoned`` and never deletes them). That staged file is invisible to
          :meth:`read_records` (so the corpus *looks* erased) yet holds a purged document's records
          in cleartext on disk — exactly the GDPR-erasure threat model (security S2). Each
          abandoned ``records.jsonl`` is expunged the **same** reach-filtered way as a committed one
          (shape (a) of the H-D1 spec). We rewrite it in place via
          :meth:`~latence_core.storage.Storage.write_atomic` rather than deleting the whole dir
          (shape (b)) precisely to stay race-safe against a *concurrent* run: an atomic in-place
          rewrite of a not-yet-committed staging file cannot corrupt a live recovery — a concurrent
          reader either sees valid JSONL before or after the swap, and if that transaction later
          commits it simply becomes a committed Version a subsequent Purge sweep would re-scan —
          whereas a bare ``remove_tree`` of a staging dir a live run is mid-commit on could tear its
          transaction. No abandoned version's ``version.json`` is touched: an uncommitted staging
          area has no authoritative manifest to reconcile, and it is never read back.
        """
        from latence_core.state.graph_patcher import _record_reach

        def _sweep(uri: str, *, doclevel: bool) -> int:
            if not self._storage.exists(uri):
                return 0
            decode = self._decode_doclevel if doclevel else self._decode_records
            encode = self._encode_doclevel if doclevel else self._encode_records
            records = decode(self._storage.read_text(uri))
            kept = [r for r in records if not (set(_record_reach(r)) & document_ids)]
            if len(kept) == len(records):
                return 0
            self._storage.write_atomic(uri, encode(kept).encode("utf-8"))
            return len(records) - len(kept)

        removed = 0
        # 1. Committed Versions: expunge the corpus-level log AND the W17 doc-level reuse store
        #    (which holds the purged document's raw text / masked content — a GDPR erasure must
        #    neutralise it too), then reconcile the manifest to the erased reality so the
        #    idempotency fingerprint / audit record_count stop describing the pre-purge set. The
        #    fingerprint is over the corpus-level records only, so the doc-level sweep does not
        #    perturb it; reconciliation still runs whenever either log actually lost a record.
        for version in self._wal.committed_versions():
            purged = _sweep(self._wal.staged_uri(version, _RECORDS_NAME), doclevel=False)
            purged += _sweep(self._wal.staged_uri(version, _DOCLEVEL_NAME), doclevel=True)
            if purged:
                removed += purged
                self._reconcile_manifest(version)
        # 2. Abandoned staging dirs: neutralise both logs the same reach-filtered way (no manifest
        #    exists to reconcile; the rewrite is race-safe against a concurrent run's in-flight
        #    transaction).
        for version in self._wal.abandoned_versions():
            removed += _sweep(self._wal.staged_uri(version, _RECORDS_NAME), doclevel=False)
            removed += _sweep(self._wal.staged_uri(version, _DOCLEVEL_NAME), doclevel=True)
        return removed

    def _reconcile_manifest(self, version: int) -> None:
        """Rewrite ``version``'s ``version.json`` to match its post-purge ``records.jsonl``.

        After :meth:`expunge_records` drops the purged records from a committed Version's log, the
        stored :class:`~latence_core.contracts.CorpusVersion` still carries the *pre-purge*
        ``fingerprint`` and ``record_count`` — a stale manifest. This recomputes both from the now-
        live records (reusing :meth:`_fingerprint`, the exact value :meth:`commit` compares against
        for idempotency) and atomically rewrites the manifest, so the head Version's idempotency
        check is correct and every audit read reflects the erased reality. Only the two erasure-
        derived fields change; the immutable delta/parent/timestamp lineage is preserved verbatim.
        A missing ``version.json`` (a torn manifest) is left untouched.
        """
        version_uri = self._wal.staged_uri(version, _VERSION_NAME)
        if not self._storage.exists(version_uri):
            return
        corpus_version = CorpusVersion.model_validate_json(self._storage.read_text(version_uri))
        records = self.read_records(version)
        reconciled = corpus_version.model_copy(
            update={
                "fingerprint": self._fingerprint(records),
                "record_count": len(records),
            }
        )
        self._storage.write_atomic(version_uri, reconciled.model_dump_json().encode("utf-8"))

    # -- serialisation ------------------------------------------------------

    @staticmethod
    def _encode_records(records: list[Record]) -> str:
        lines: list[str] = []
        for record in records:
            kind = _KIND_BY_TYPE.get(type(record))
            if kind is None:
                msg = (
                    f"CorpusStore cannot persist a {type(record).__name__} — only corpus-level "
                    f"derived records (DisambiguationRecord/GraphRecord) are versioned."
                )
                raise CorpusStoreError(msg)
            obj = record.model_dump(mode="json")
            obj["_kind"] = kind
            lines.append(json.dumps(obj, sort_keys=True, ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _decode_records(text: str) -> list[Record]:
        records: list[Record] = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            kind = obj.pop("_kind", None)
            record_type = _RECORD_TYPES.get(kind) if isinstance(kind, str) else None
            if record_type is None:
                msg = f"CorpusStore read a record with unknown _kind {kind!r}."
                raise CorpusStoreError(msg)
            records.append(record_type.model_validate(obj))
        return records

    @staticmethod
    def _encode_doclevel(records: list[Record]) -> str:
        """Serialise the doc-level reuse store: one tagged JSON object per line, id-sorted.

        Sorted by ``record_id`` so the persisted store is byte-stable (determinism), and tagged
        with the SAME ``_kind`` discriminator as the corpus-level log so the polymorphic doc-level
        carriers (DocumentRecord/ChunkRecord/EntityMention/RelationMention) round-trip losslessly.
        """
        lines: list[str] = []
        for record in sorted(records, key=lambda r: r.record_id):
            kind = _DOCLEVEL_KIND_BY_TYPE.get(type(record))
            if kind is None:
                msg = (
                    f"CorpusStore cannot persist a {type(record).__name__} in the doc-level reuse "
                    f"store — only DocumentRecord/ChunkRecord/EntityMention/RelationMention."
                )
                raise CorpusStoreError(msg)
            obj = record.model_dump(mode="json")
            obj["_kind"] = kind
            lines.append(json.dumps(obj, sort_keys=True, ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _decode_doclevel(text: str) -> list[Record]:
        records: list[Record] = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            kind = obj.pop("_kind", None)
            record_type = _DOCLEVEL_TYPES.get(kind) if isinstance(kind, str) else None
            if record_type is None:
                msg = f"CorpusStore read a doc-level record with unknown _kind {kind!r}."
                raise CorpusStoreError(msg)
            records.append(record_type.model_validate(obj))
        return records

    @staticmethod
    def _fingerprint(records: list[Record]) -> str:
        """Deterministic content fingerprint over the committed records (G1 determinism).

        A sha256 over each record's canonical (sorted-key) JSON dump, taken in ``record_id``
        order: identical input + config ⇒ identical fingerprint, so two runs of the same delta
        produce the same Version fingerprint and the idempotent-re-run check is exact. Hashing the
        full record content — not just the ``record_id`` — is required because a merge can change
        a cluster's membership while keeping its lexicographic-min canonical id (``entity:ibm``
        absorbs ``International Business Machines`` but stays ``entity:ibm``): an id-only
        fingerprint would miss that content change and wrongly treat the delta as a no-op. The
        commit timestamp is excluded (it is on the CorpusVersion manifest, not the records).
        """
        hasher = hashlib.sha256()
        # Canonical (sorted-key) JSON so an open ``properties`` bag whose keys were inserted in a
        # different order between two otherwise-identical runs hashes the same — the equality is
        # over content, not serialisation order.
        dumps = sorted(
            (r.record_id, json.dumps(r.model_dump(mode="json"), sort_keys=True))
            for r in records
        )
        for record_id, blob in dumps:
            hasher.update(record_id.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(blob.encode("utf-8"))
            hasher.update(b"\n")
        return "sha256:" + hasher.hexdigest()

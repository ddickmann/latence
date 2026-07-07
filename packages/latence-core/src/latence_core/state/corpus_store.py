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
    CorpusVersion,
    DisambiguationRecord,
    GraphRecord,
    Record,
)
from latence_core.errors import StorageError
from latence_core.state.write_ahead_log import WriteAheadLog

if TYPE_CHECKING:
    from latence_core.storage import Storage

_RECORDS_NAME = "records.jsonl"
_VERSION_NAME = "version.json"
_HEAD_NAME = "HEAD"

# The record kinds a Corpus Version persists. Discriminated on read by a ``_kind`` tag so the
# polymorphic derived set round-trips. Doc-level carriers (chunks/mentions/relations/redactions)
# are re-derivable from the run and are NOT persisted in the corpus version — the corpus version
# holds the corpus-level derived truth (canonical entities, normalized relations, graph).
_RECORD_TYPES: dict[str, type[Record]] = {
    "disambiguation": DisambiguationRecord,
    "graph": GraphRecord,
}
_KIND_BY_TYPE: dict[type[Record], str] = {v: k for k, v in _RECORD_TYPES.items()}


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

    # -- transaction --------------------------------------------------------

    def commit(
        self,
        *,
        parent_version: int | None,
        records: list[Record],
        delta_json: str,
        document_count: int,
        reconciled: bool,
    ) -> CorpusVersion:
        """Commit the next Corpus Version transactionally (read-N → write-N+1).

        ``records`` is the live derived set after the delta; ``delta_json`` the serialised
        :class:`~latence_core.contracts.Delta` that produced it. The next version number is
        ``head + 1`` (0 on bootstrap). The records + version manifest are staged under the WAL,
        then the WAL manifest is written LAST (the atomic swap): a crash before that leaves the
        prior head intact. Returns the committed :class:`~latence_core.contracts.CorpusVersion`.
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
        self._wal.commit(
            version,
            files={"records": records_uri, "version": version_uri},
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
        """Physically remove every derived record reaching ``document_ids`` from ALL committed
        Versions' record logs (the Purge / GDPR-erasure step, ADR-0018).

        Corpus Versions are normally immutable audit history, but a Purge is a hard
        right-to-be-forgotten erasure that trumps that immutability: the S11 acceptance criterion
        requires that *nothing derived survives* — so a purged document's canonical entities,
        relations, graph nodes/edges, and their Evidence must not linger in any prior Version's
        ``records.jsonl`` (where :meth:`read_records` would otherwise return them verbatim). A
        record is expunged when its Evidence-provenance reach touches any purged document
        (:func:`~latence_core.state.graph_patcher._record_reach`). Each rewritten record log is
        written atomically. Returns the number of records physically removed across all Versions.

        This intentionally does NOT re-fingerprint or re-resolve the prior Versions — they are
        being erased, not re-derived; the head Version the purge just committed is the authoritative
        live set. Only committed Versions are touched; the WAL manifests are left intact.
        """
        from latence_core.state.graph_patcher import _record_reach

        removed = 0
        for version in self._wal.committed_versions():
            uri = self._wal.staged_uri(version, _RECORDS_NAME)
            if not self._storage.exists(uri):
                continue
            records = self._decode_records(self._storage.read_text(uri))
            kept = [r for r in records if not (set(_record_reach(r)) & document_ids)]
            if len(kept) == len(records):
                continue
            removed += len(records) - len(kept)
            self._storage.write_atomic(uri, self._encode_records(kept).encode("utf-8"))
        return removed

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

"""Write-Ahead Log — the ported crash-safe atomic-commit engine (PORT-AS-IS, S11).

Ported faithfully — as-is — from
``latenceai-dataset-intelligence/state/write_ahead_log.py``, re-homed on the framework's
:class:`~latence_core.storage.Storage` seam (ADR-0009) rather than raw ``os`` calls, so it
works over local disk, S3, GCS, or Azure with no change. The algorithmic value is preserved
verbatim: **temp-write → manifest-write → atomic rename**, with crash recovery driven by a
single ``MANIFEST.json``.

The discipline (the ported ``begin`` / ``stage`` / ``commit`` / ``recover`` flow):

1. **begin** — open a transaction targeting a numbered version; a ``*.wal.json`` intent
   record is written first (so a crash between here and commit is detectable).
2. **stage** — every payload is written to a temp file under the version's staging area
   (never over the live data), each landing atomically via
   :meth:`~latence_core.storage.Storage.write_atomic` (its own temp-file→rename), so a
   reader never sees a torn payload.
3. **commit** — the manifest is written last, atomically, listing exactly the staged files
   that make up the committed version. The manifest write is the linearisation point: the
   version does not exist until (and exists the instant) the manifest lands.
4. **recover** — on restart, an intent record with no matching committed manifest is an
   incomplete transaction; its staging area is abandoned (the previous committed version is
   left intact — the S11 "a failed run leaves Version N intact" guarantee).

This is the same WAL discipline the Runner already relies on for per-Stage checkpoints
(``Storage.write_atomic``); S11 lifts it to a whole **Corpus Version** transaction so
read-N → write-N+1 is atomic, idempotent, and rollback-safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from latence_core.storage import Storage

# The WAL manifest and intent record file names, kept fixed so recovery finds them without a
# directory scan heuristic (the ported ``MANIFEST.json`` convention).
_MANIFEST_NAME = "MANIFEST.json"
_INTENT_SUFFIX = ".wal.json"


@dataclass(frozen=True)
class WalRecovery:
    """The outcome of a crash-recovery scan (the ported ``recover`` result).

    ``committed`` is the highest version whose ``MANIFEST.json`` exists (the last durable
    state — ``None`` before the first commit). ``abandoned`` are versions that opened a
    transaction (wrote an intent record) but never committed — incomplete runs whose staging
    is discarded, leaving ``committed`` intact.
    """

    committed: int | None
    abandoned: tuple[int, ...]


class WriteAheadLog:
    """Crash-safe versioned commit over a Storage root (ported WAL, re-homed).

    A WAL is bound to a ``root`` URI on a :class:`~latence_core.storage.Storage`. Under it,
    each transaction owns a ``v<N>/`` staging directory; ``v<N>/MANIFEST.json`` is the
    linearisation point (present ⇔ committed). ``v<N>.wal.json`` is the intent record written
    at :meth:`begin`. Every write goes through ``Storage.write_atomic``, so the WAL never
    forks the framework's IO seam — it composes it.
    """

    def __init__(self, storage: Storage, root: str) -> None:
        self._storage = storage
        self._root = root.rstrip("/")

    # -- URIs --------------------------------------------------------------

    def _version_dir(self, version: int) -> str:
        return f"{self._root}/v{version}"

    def _manifest_uri(self, version: int) -> str:
        return f"{self._version_dir(version)}/{_MANIFEST_NAME}"

    def _intent_uri(self, version: int) -> str:
        return f"{self._root}/v{version}{_INTENT_SUFFIX}"

    def staged_uri(self, version: int, name: str) -> str:
        """The URI a payload named ``name`` is staged at for ``version`` (under its dir)."""
        return f"{self._version_dir(version)}/{name}"

    # -- transaction -------------------------------------------------------

    def begin(self, version: int) -> None:
        """Open a transaction for ``version``: write the intent record first.

        The intent record is the crash-detection anchor — a version with an intent record but
        no committed manifest is an incomplete transaction (:meth:`recover`). Writing it
        before any payload is the ported ``begin`` step.
        """
        self._storage.makedirs(self._version_dir(version))
        payload = json.dumps({"version": version, "committed": False}, sort_keys=True)
        self._storage.write_atomic(self._intent_uri(version), payload.encode("utf-8"))

    def stage(self, version: int, name: str, data: bytes) -> str:
        """Stage one payload under ``version``'s directory (atomic temp→rename). Returns its URI.

        Staged writes never touch the live/previous version's data — they land in ``v<N>/``,
        which only becomes the committed version once :meth:`commit` writes the manifest.
        """
        uri = self.staged_uri(version, name)
        self._storage.write_atomic(uri, data)
        return uri

    def stage_text(self, version: int, name: str, text: str) -> str:
        return self.stage(version, name, text.encode("utf-8"))

    def commit(self, version: int, files: Mapping[str, str], metadata: Mapping[str, str]) -> None:
        """Commit ``version`` by writing its manifest LAST (the linearisation point).

        ``files`` maps a logical payload name to its staged URI (the manifest of what the
        version is made of); ``metadata`` is arbitrary string-keyed audit info (delta kind,
        parent version, churn summary). The manifest write is atomic, so the version does not
        exist until — and exists the instant — the manifest lands. The intent record is then
        flipped to committed (best-effort audit; the manifest, not the intent, is
        authoritative).
        """
        manifest = {
            "version": version,
            "committed": True,
            "files": {k: files[k] for k in sorted(files)},
            "metadata": {k: metadata[k] for k in sorted(metadata)},
        }
        self._storage.write_atomic(
            self._manifest_uri(version),
            json.dumps(manifest, sort_keys=True).encode("utf-8"),
        )
        # Flip the intent record (audit only; the manifest is authoritative for recovery).
        self._storage.write_atomic(
            self._intent_uri(version),
            json.dumps({"version": version, "committed": True}, sort_keys=True).encode("utf-8"),
        )

    def is_committed(self, version: int) -> bool:
        """True if ``version``'s manifest exists (the version is durably committed)."""
        return self._storage.exists(self._manifest_uri(version))

    def read_manifest(self, version: int) -> dict[str, object]:
        """Read a committed version's manifest. Raises if the version is not committed."""
        uri = self._manifest_uri(version)
        if not self._storage.exists(uri):
            msg = f"No committed manifest for version {version} at {uri!r}."
            raise FileNotFoundError(msg)
        data: dict[str, object] = json.loads(self._storage.read_text(uri))
        return data

    def committed_versions(self) -> list[int]:
        """Every committed version number, ascending (a manifest-bearing ``v<N>/``)."""
        if not self._storage.exists(self._root):
            return []
        found: list[int] = []
        for entry in self._storage.ls(self._root):
            name = entry.rstrip("/").rsplit("/", 1)[-1]
            if not name.startswith("v") or _INTENT_SUFFIX in name:
                continue
            try:
                version = int(name[1:])
            except ValueError:
                continue
            if self.is_committed(version):
                found.append(version)
        return sorted(found)

    def abandoned_versions(self) -> tuple[int, ...]:
        """Every abandoned (intent-record-but-no-manifest) version number, ascending.

        A convenience view over :meth:`recover` for callers — such as the Purge residue sweep
        (:meth:`~latence_core.state.corpus_store.CorpusStore.expunge_records`) — that must reach
        into never-committed staging directories a crash left behind. These ``v<N>/`` dirs may
        still hold a fully-staged ``records.jsonl`` (invisible to :meth:`is_committed`/
        :meth:`read_manifest`), which is exactly the abandoned-WAL residue a GDPR erasure must
        neutralise (audit THEME D / H-D1).
        """
        return self.recover().abandoned

    def recover(self) -> WalRecovery:
        """Scan for the last committed version and any abandoned (uncommitted) transactions.

        The ported crash-recovery step: an intent record (``v<N>.wal.json``) with no matching
        committed manifest is an incomplete transaction — reported as ``abandoned`` and never
        promoted, so the last committed version is left intact (rollback-on-failure).
        """
        committed = self.committed_versions()
        highest = committed[-1] if committed else None
        abandoned: list[int] = []
        if self._storage.exists(self._root):
            for entry in self._storage.ls(self._root):
                name = entry.rstrip("/").rsplit("/", 1)[-1]
                if not name.endswith(_INTENT_SUFFIX):
                    continue
                stem = name[: -len(_INTENT_SUFFIX)]
                if not stem.startswith("v"):
                    continue
                try:
                    version = int(stem[1:])
                except ValueError:
                    continue
                if not self.is_committed(version):
                    abandoned.append(version)
        return WalRecovery(committed=highest, abandoned=tuple(sorted(abandoned)))

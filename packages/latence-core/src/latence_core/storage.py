"""fsspec-backed Storage seam (ADR-0009).

All framework IO — input documents, inter-Stage records, checkpoints, exports,
the Quality Report — flows through this one abstraction. Local disk, S3, GCS,
Azure Blob, and in-memory are addressed by URI scheme with no framework code
change; credentials resolve through fsspec's own default chains, so the
framework never stores secrets.
"""

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING, Any

import fsspec

from latence_core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from contextlib import AbstractContextManager

# Storage schemes the framework permits by default. Every entry is a
# cloud-agnostic file/object store fsspec fulfils (ADR-0009); network-fetch
# schemes (http, ftp, sftp, …) are deliberately excluded so a Source URI can
# never be turned into an SSRF vector or an arbitrary remote read. Advanced
# adopters can widen this via the ``allowed_schemes`` argument, but the default
# is deny — a malicious pipeline YAML cannot smuggle in ``http://`` on its own.
DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset(
    {"file", "memory", "s3", "gs", "gcs", "abfs", "az"}
)


class SchemeNotAllowedError(ConfigError):
    """Raised when a Storage is built for a URI scheme not on the allowlist.

    A :class:`~latence_core.errors.ConfigError` (a disallowed scheme is an operator config
    mistake, #73) — and, transitively, still a ``ValueError``, so the pre-taxonomy
    ``except ValueError`` / ``except SchemeNotAllowedError`` guards keep catching it.
    """


class Storage:
    """Thin, URI-scheme-agnostic IO surface over fsspec.

    A :class:`Storage` is bound to a single filesystem (derived from the scheme
    of the URIs it is asked to operate on). Mixing schemes on one instance is a
    programming error and raises ``ValueError``.
    """

    def __init__(
        self,
        protocol: str = "file",
        *,
        allowed_schemes: Iterable[str] | None = None,
        **storage_options: Any,
    ) -> None:
        allowed = (
            frozenset(allowed_schemes) if allowed_schemes is not None else DEFAULT_ALLOWED_SCHEMES
        )
        if protocol not in allowed:
            permitted = ", ".join(sorted(allowed))
            msg = (
                f"Storage scheme {protocol!r} is not permitted. "
                f"Allowed schemes: {permitted}. "
                f"Pass allowed_schemes=[...] to widen this at your own risk."
            )
            raise SchemeNotAllowedError(msg)
        self._protocol = protocol
        self._allowed_schemes = allowed
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(protocol, **storage_options)

    @classmethod
    def for_uri(
        cls,
        uri: str,
        *,
        allowed_schemes: Iterable[str] | None = None,
        **storage_options: Any,
    ) -> Storage:
        """Build a Storage bound to the filesystem implied by ``uri``'s scheme."""
        protocol = cls._protocol_of(uri)
        return cls(protocol=protocol, allowed_schemes=allowed_schemes, **storage_options)

    @staticmethod
    def _protocol_of(uri: str) -> str:
        if "://" not in uri:
            return "file"
        return uri.split("://", 1)[0]

    @property
    def protocol(self) -> str:
        return self._protocol

    def _check(self, uri: str) -> str:
        proto = self._protocol_of(uri)
        if proto != self._protocol:
            msg = f"Storage bound to {self._protocol!r} cannot operate on {proto!r} URI: {uri!r}"
            raise ValueError(msg)
        # Strip the scheme for filesystems (like local) that want a bare path.
        return uri

    def exists(self, uri: str) -> bool:
        return bool(self._fs.exists(self._check(uri)))

    def makedirs(self, uri: str) -> None:
        self._fs.makedirs(self._check(uri), exist_ok=True)

    def read_bytes(self, uri: str) -> bytes:
        with self._fs.open(self._check(uri), "rb") as fh:
            data: bytes = fh.read()
        return data

    def read_text(self, uri: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(uri).decode(encoding)

    def write_bytes(self, uri: str, data: bytes) -> None:
        parent = uri.rsplit("/", 1)[0] if "/" in uri else uri
        if parent and parent != uri:
            self.makedirs(parent)
        with self._fs.open(self._check(uri), "wb") as fh:
            fh.write(data)

    def write_text(self, uri: str, text: str, encoding: str = "utf-8") -> None:
        self.write_bytes(uri, text.encode(encoding))

    def write_atomic(self, uri: str, data: bytes) -> None:
        """Write ``data`` to ``uri`` so a reader never sees a partial file.

        The buffered face of :meth:`atomic_writer` — same publish-or-nothing guarantee, same
        temp-file sweep on failure, for a payload that already fits in one ``bytes`` buffer.
        This is the WAL discipline the Runner relies on for durable, resumable checkpoints
        (ADR-0010): the destination either does not exist or is the complete previous/next
        version, never a torn write, and a failure leaves no ``.tmp`` sibling behind.

        See :meth:`atomic_writer` for what "atomic" does and does not mean per backend.
        """
        with self.atomic_writer(uri) as fh:
            fh.write(data)

    @contextmanager
    def atomic_writer(
        self,
        uri: str,
        *,
        destination: Callable[[], str | None] | None = None,
    ) -> Iterator[IO[bytes]]:
        """Stream bytes into ``uri``: publish on clean exit, sweep the partial write on failure.

        The one-call streaming write-atomic primitive (ADR-0010/0033). It opens a unique temp
        sibling of ``uri``, creates ``uri``'s parent directory, yields the open binary handle,
        and on clean exit closes it and renames it onto ``uri``. A reader never sees the temp
        path under ``uri`` and never sees a torn ``uri`` — the same guarantee as
        :meth:`write_atomic`, without materialising the payload in memory first::

            with storage.atomic_writer(uri) as fh:
                for batch in batches:
                    fh.write(batch)

        **The cleanup obligation lives here, not in the caller.** The two-call
        :meth:`temp_uri` / :meth:`finalize_atomic` protocol this replaces pushed onto every
        caller a duty its signature never stated: sweep the temp path when the write dies
        between the two calls. One of three callers actually did it, and the cost of the other
        two was silent: each temp name carries a fresh random token, so on a flaky store *every
        retry stranded another full-size copy* — unbounded growth of unreferenced, billable
        objects that no lifecycle policy and no ``expire_snapshots`` knows about, still holding
        the run's (possibly sensitive) bytes. On any exception the temp file is removed
        best-effort and the **original error is re-raised untouched** (ADR-0034: the failure is
        never masked by its own cleanup).

        The sweep is *reported, not silent*: the propagating exception carries a PEP-678 note
        (``exc.__notes__``) saying either that the partial write was removed and nothing was
        published, or — when the store refused the delete — naming the exact path that still
        holds the bytes so an operator can delete it by hand. A caller that wraps the failure in
        its own typed error can fold those notes into its message.

        ``destination`` is for a **content-addressed** artifact whose final name is only knowable
        once its bytes have been written (the Iceberg data file, named after the sha256 of its own
        content). It is called after the handle is closed and its return value replaces ``uri`` as
        the publish target; returning ``None`` publishes nothing and sweeps the temp file (the
        "an identical file is already there, leave it alone" case). ``uri`` still names the temp
        sibling, so the in-flight write lands next to where it is going.

        **What "atomic" means per backend** — this is interface, not trivia:

        * *Local / POSIX*: fsspec's rename maps to ``os.replace``, a genuinely atomic
          same-filesystem operation.
        * *Object stores (S3, GCS, Azure)*: there is no rename. fsspec's ``mv`` is a server-side
          **copy followed by a delete — two operations, not one atomic one.** What still holds is
          the guarantee callers actually depend on: a single-object write is atomic and
          read-after-write consistent on all three stores, so a reader of ``uri`` sees either the
          old object or the whole new one, never a torn one. What does *not* hold is the rest of
          the rename contract: a crash between the copy and the delete leaves the temp object
          behind (this method's sweep is best-effort and cannot run if the process is killed), and
          two publishers racing onto one ``uri`` both succeed with a last-writer-wins result rather
          than one losing. Sizes matter too: S3 caps a single ``CopyObject`` at 5 GB, so a larger
          artifact is copied as a multipart upload whose failure can leave an incomplete multipart
          upload the bucket's lifecycle policy — not this method — has to reap.
        """
        with self._atomic_publication([uri], [destination]) as handles:
            yield handles[0]

    def atomic_writers(self, uris: Sequence[str]) -> AbstractContextManager[tuple[IO[bytes], ...]]:
        """Stream a **set** of sibling files, publishing all of them only once all are complete.

        The multi-file face of :meth:`atomic_writer`, for a deliverable that is a *group* of files
        which must not appear one-by-one — the Export's ``records.jsonl`` + ``records.parquet`` (+
        the optional multi-vector and BM25 sidecars), which a downstream reader treats as one
        corpus (ADR-0033/0050). Every handle is written to its own temp sibling; not one byte is
        published until *every* handle has closed cleanly, so a write error part-way through the
        corpus can no longer leave a published ``records.parquet`` next to a missing
        ``records.jsonl``. On any exception every temp file is swept, exactly as
        :meth:`atomic_writer` sweeps its one.

        Handles are yielded in the order of ``uris`` and must all be used inside the block::

            with storage.atomic_writers([jsonl_uri, parquet_uri]) as (jsonl, parquet):
                ...

        Honest limit: the publish phase is still N renames, not one transaction. No store the
        framework targets offers a cross-object commit, so a failure *during* the publish loop
        leaves the already-renamed siblings in place and sweeps the rest; the exception's notes
        name what landed. This is strictly stronger than publishing as you go and is the most any
        of file/S3/GCS/Azure can actually give — a retried run rewrites the whole set.
        """
        return self._atomic_publication(list(uris), [None] * len(uris))

    @contextmanager
    def _atomic_publication(
        self,
        uris: Sequence[str],
        destinations: Sequence[Callable[[], str | None] | None],
    ) -> Iterator[tuple[IO[bytes], ...]]:
        """The shared engine behind :meth:`atomic_writer` / :meth:`atomic_writers`.

        Internal, deliberately: the two public faces are the whole interface, and the temp-name
        minting, the parent ``makedirs`` (a streaming ``open`` does not create one, unlike
        :meth:`write_bytes`), the publish loop and the sweep are implementation.
        """
        tmp_uris = [self.temp_uri(uri) for uri in uris]
        for uri in uris:
            parent = uri.rsplit("/", 1)[0] if "/" in uri else ""
            if parent and parent != uri:
                self.makedirs(parent)
        try:
            with contextlib.ExitStack() as stack:
                handles = tuple(stack.enter_context(self.open(tmp, "wb")) for tmp in tmp_uris)
                yield handles
            # Past here every handle is closed and every temp file is complete on the store.
        except BaseException as exc:
            self._sweep(tmp_uris, exc)
            raise
        published = 0
        try:
            for tmp, uri, destination in zip(tmp_uris, uris, destinations, strict=True):
                target = uri if destination is None else destination()
                if target is None:
                    self.remove(tmp)
                else:
                    self.finalize_atomic(tmp, target)
                published += 1
        except BaseException as exc:
            # Whatever already landed stays landed (there is no cross-object rollback); sweep
            # only the siblings this run never published.
            self._sweep(tmp_uris[published:], exc)
            raise

    def _sweep(self, tmp_uris: Iterable[str], exc: BaseException) -> None:
        """Best-effort remove the temp files of a failed publication, reporting on ``exc``.

        Never raises and never masks ``exc`` — a cleanup failure is appended as a PEP-678 note
        (ADR-0034: no silent errors, and the operator is told the exact path still holding the
        bytes rather than being left to discover it on a storage bill).
        """
        for tmp in tmp_uris:
            try:
                existed = self.exists(tmp)
                self.remove(tmp)
            except Exception as cleanup_exc:  # noqa: BLE001 - never mask the failure being reported
                exc.add_note(
                    f"latence.Storage: the partial write {tmp!r} could not be removed "
                    f"({cleanup_exc}); it still holds this run's bytes — delete it manually."
                )
            else:
                if existed:
                    exc.add_note(
                        f"latence.Storage: nothing was published; the partial write {tmp!r} "
                        f"was removed."
                    )

    def temp_uri(self, uri: str) -> str:
        """A unique sibling temp path for ``uri`` (the write side of :meth:`finalize_atomic`).

        .. deprecated:: superseded by :meth:`atomic_writer` / :meth:`atomic_writers`.

           This is half of a two-call protocol that leaks a cleanup obligation the interface
           never states — the caller, not this module, is left holding the duty to sweep the temp
           path when the write between the two calls fails. :meth:`atomic_writer` owns that duty.
           Retained only until the last remaining caller (the Runner's checkpoint writer) moves
           across; it is not part of the interface a new caller should learn.
        """
        return f"{uri}.{_token()}.tmp"

    def finalize_atomic(self, tmp_uri: str, uri: str) -> None:
        """Atomically publish a fully-written temp file (from :meth:`temp_uri`) as ``uri``.

        The rename half of the two-call streamed write. See :meth:`atomic_writer` for what
        "atomic" actually means on each backend (on an object store this is a copy **and** a
        delete — two operations — not a rename).

        .. deprecated:: superseded by :meth:`atomic_writer` / :meth:`atomic_writers`, which own
           the cleanup obligation this pairing pushes onto its callers.
        """
        self._fs.mv(self._check(tmp_uri), self._check(uri))

    def remove(self, uri: str) -> None:
        """Physically delete a single file (the Purge / GDPR-erasure primitive, ADR-0018).

        Idempotent: removing a non-existent path is a no-op, so a purge that has already erased a
        file (or a file the caller only *believes* exists) does not raise. Unlike
        :meth:`write_atomic` this is a destructive, non-recoverable delete — used only on the
        Purge path where the S11 acceptance criterion requires that *nothing derived survives*
        (the source file, and the derived records expunged from prior Versions and checkpoints).
        """
        checked = self._check(uri)
        if self._fs.exists(checked):
            self._fs.rm_file(checked) if hasattr(self._fs, "rm_file") else self._fs.rm(checked)

    def remove_tree(self, uri: str) -> None:
        """Recursively delete a directory and everything under it (the Purge erasure primitive).

        Idempotent: a missing path is a no-op. Destructive and non-recoverable — used on the Purge
        path to wipe re-derivable run artifact trees (checkpoints, export outputs, run manifests)
        that would otherwise retain a purged document's raw text (the S11 "nothing derived
        survives" acceptance criterion).
        """
        checked = self._check(uri)
        if self._fs.exists(checked):
            self._fs.rm(checked, recursive=True)

    def open(self, uri: str, mode: str = "rb") -> IO[Any]:
        opened: IO[Any] = self._fs.open(self._check(uri), mode)
        return opened

    def ls(self, uri: str) -> list[str]:
        entries: list[str] = self._fs.ls(self._check(uri), detail=False)
        return sorted(entries)

    def modified(self, uri: str) -> datetime | None:
        """The UTC last-modified time of ``uri``, or ``None`` when the store does not expose one.

        A purely read-only metadata probe (#186): the run console reads *when* a run's manifest
        and Quality Report landed straight off Storage, so a run list carries real timestamps
        without the framework persisting any new state (ADR-0010 — run state is the files that
        are already there). fsspec backends spell the field differently (``mtime`` on local,
        ``LastModified`` on S3, ``updated`` on GCS, ``last_modified`` on Azure) and an in-memory
        store exposes none at all, so this normalises the known spellings and returns ``None``
        rather than fabricating a time when the backend has nothing to give. A missing path is
        ``None`` too — the caller renders "unknown", it is never an error.
        """
        try:
            info = self._fs.info(self._check(uri))
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return None
        if not isinstance(info, dict):
            return None
        for key in ("mtime", "LastModified", "last_modified", "updated", "created"):
            value = info.get(key)
            stamp = _as_utc(value)
            if stamp is not None:
                return stamp
        return None

    def walk_files(self, uri: str, follow_symlinks: bool = False) -> Iterator[str]:
        """Yield every file (not directory) under ``uri``, recursively, sorted.

        Every yielded path is a **scheme-bearing URI** the same Storage can read
        back: ``fsspec``'s ``walk`` strips the protocol and yields bare
        ``bucket/key`` (object stores) or bare filesystem paths (local), so each
        result is re-decorated with this Storage's protocol via
        ``unstrip_protocol`` before being yielded. Without this, feeding a
        ``walk_files`` result straight back into :meth:`read_bytes` on an object
        store raised ``ValueError`` (a bare ``bucket/key`` has no ``://``, so
        :meth:`_protocol_of` mis-reads it as ``file`` and :meth:`_check` rejects
        it against the ``s3``/``memory`` binding) — the H-E1 s3:// e2e blocker.

        With ``follow_symlinks=False`` (the secure default) a file whose real,
        symlink-resolved path escapes ``uri``'s own real path is dropped, so a
        symlink planted inside the folder cannot be used to read files outside
        it. fsspec's own ``walk`` follows symlinks (its ``ls`` does), so the
        escape check is applied here rather than relying on a walk flag. Only the
        local filesystem has symlinks; on object/in-memory stores the filter is a
        no-op because ``os.path.realpath`` leaves non-local paths unchanged.

        **Memory characteristic — this is interface, not an implementation detail.** Despite
        returning an ``Iterator``, this is NOT a streaming listing: the whole recursive listing is
        accumulated and sorted before the first path is yielded, so peak memory is O(number of
        files under ``uri``) and the first result costs a full prefix walk. That is deliberate and
        load-bearing rather than an oversight: the framework's determinism invariant requires a
        stable document order, and ``fsspec``'s ``walk`` inherits its per-directory order from
        ``ls`` (``os.listdir`` order locally, the store's paging order remotely), which is not
        stable across hosts or runs. A globally sorted order cannot be produced without seeing
        every path, and a per-directory sort would yield a *different* (if deterministic) order
        than the global one, silently changing every downstream artifact. Callers pointing this at
        a bucket prefix with millions of objects should scope the prefix; a paged listing would be
        a different, order-weakening method, not a change to this one.
        """
        checked = self._check(uri)
        # fsspec's local walk strips the URI scheme and yields bare filesystem
        # paths; resolve the root the same way so the escape check compares like
        # with like (a raw ``os.path.realpath('file://...')`` would be garbage).
        is_local = self._protocol == "file"
        root_bare = checked.split("://", 1)[1] if is_local and "://" in checked else checked
        root_real = os.path.realpath(root_bare) if is_local else checked
        found: list[str] = []
        for root, _dirs, files in self._fs.walk(checked):
            for name in files:
                path = f"{root.rstrip('/')}/{name}"
                # The symlink-escape check runs on the bare local path (that is
                # what ``os.path.realpath`` understands); re-decorate afterwards.
                if not follow_symlinks and is_local and self._escapes(path, root_real):
                    continue
                # Yield a scheme-bearing URI so the caller can round-trip it back
                # through read_bytes/open under this Storage's binding (s3://,
                # memory://, file://) — not a bare, binding-mismatched path.
                found.append(self._fs.unstrip_protocol(path))
        yield from sorted(found)

    @staticmethod
    def _escapes(path: str, root_real: str) -> bool:
        """True if ``path``'s symlink-resolved location is outside ``root_real``."""
        real = os.path.realpath(path)
        root = root_real.rstrip(os.sep) + os.sep
        return real != root_real and not real.startswith(root)


def _as_utc(value: Any) -> datetime | None:
    """Normalise an fsspec ``info`` timestamp (datetime / epoch seconds) to an aware UTC datetime.

    Returns ``None`` for anything that is not a usable timestamp, so :meth:`Storage.modified`
    degrades to "unknown" instead of guessing. ``bool`` is excluded explicitly — it is an ``int``
    subclass and ``True`` would otherwise become 1970-01-01.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _token() -> str:
    """A per-write unique suffix for the atomic temp file (not security-sensitive)."""
    import uuid

    return uuid.uuid4().hex

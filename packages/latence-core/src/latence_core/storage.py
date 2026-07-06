"""fsspec-backed Storage seam (ADR-0009).

All framework IO — input documents, inter-Stage records, checkpoints, exports,
the Quality Report — flows through this one abstraction. Local disk, S3, GCS,
Azure Blob, and in-memory are addressed by URI scheme with no framework code
change; credentials resolve through fsspec's own default chains, so the
framework never stores secrets.
"""

from __future__ import annotations

import os
from typing import IO, TYPE_CHECKING, Any

import fsspec

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# Storage schemes the framework permits by default. Every entry is a
# cloud-agnostic file/object store fsspec fulfils (ADR-0009); network-fetch
# schemes (http, ftp, sftp, …) are deliberately excluded so a Source URI can
# never be turned into an SSRF vector or an arbitrary remote read. Advanced
# adopters can widen this via the ``allowed_schemes`` argument, but the default
# is deny — a malicious pipeline YAML cannot smuggle in ``http://`` on its own.
DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset(
    {"file", "memory", "s3", "gs", "gcs", "abfs", "az"}
)


class SchemeNotAllowedError(ValueError):
    """Raised when a Storage is built for a URI scheme not on the allowlist."""


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
        """Write via temp-file + rename so a reader never sees a partial file.

        This is the WAL discipline the Runner relies on for durable, resumable
        checkpoints (ADR-0010): the destination either does not exist or is the
        complete previous/next version, never a torn write.
        """
        tmp = f"{uri}.{_token()}.tmp"
        self.write_bytes(tmp, data)
        # fsspec's local filesystem rename maps to os.replace (atomic on POSIX);
        # object stores implement move as an atomic server-side copy+delete.
        self._fs.mv(self._check(tmp), self._check(uri))

    def open(self, uri: str, mode: str = "rb") -> IO[Any]:
        opened: IO[Any] = self._fs.open(self._check(uri), mode)
        return opened

    def ls(self, uri: str) -> list[str]:
        entries: list[str] = self._fs.ls(self._check(uri), detail=False)
        return sorted(entries)

    def walk_files(self, uri: str, follow_symlinks: bool = False) -> Iterator[str]:
        """Yield every file (not directory) under ``uri``, recursively, sorted.

        With ``follow_symlinks=False`` (the secure default) a file whose real,
        symlink-resolved path escapes ``uri``'s own real path is dropped, so a
        symlink planted inside the folder cannot be used to read files outside
        it. fsspec's own ``walk`` follows symlinks (its ``ls`` does), so the
        escape check is applied here rather than relying on a walk flag. Only the
        local filesystem has symlinks; on object/in-memory stores the filter is a
        no-op because ``os.path.realpath`` leaves non-local paths unchanged.
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
                if not follow_symlinks and is_local and self._escapes(path, root_real):
                    continue
                found.append(path)
        yield from sorted(found)

    @staticmethod
    def _escapes(path: str, root_real: str) -> bool:
        """True if ``path``'s symlink-resolved location is outside ``root_real``."""
        real = os.path.realpath(path)
        root = root_real.rstrip(os.sep) + os.sep
        return real != root_real and not real.startswith(root)


def _token() -> str:
    """A per-write unique suffix for the atomic temp file (not security-sensitive)."""
    import uuid

    return uuid.uuid4().hex

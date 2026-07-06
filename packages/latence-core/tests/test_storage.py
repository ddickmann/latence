"""Storage seam: round-trip on memory:// and file://, atomic write, scheme guard,
scheme allowlist (SSRF guard), and symlink-escape handling in walk_files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from latence_core.storage import SchemeNotAllowedError, Storage


def test_memory_roundtrip() -> None:
    s = Storage.for_uri("memory://bucket/x")
    uri = "memory://bucket/dir/file.txt"
    assert not s.exists(uri)
    s.write_text(uri, "hello")
    assert s.exists(uri)
    assert s.read_text(uri) == "hello"


def test_file_roundtrip(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/out/file.txt"
    s = Storage.for_uri(uri)
    s.write_text(uri, "world")
    assert s.read_text(uri) == "world"


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/a/records.jsonl"
    s = Storage.for_uri(uri)
    s.write_atomic(uri, b"line\n")
    assert s.read_bytes(uri) == b"line\n"
    # No leftover *.tmp sibling.
    leftovers = list((tmp_path / "a").glob("*.tmp"))
    assert leftovers == []


def test_scheme_mismatch_raises(tmp_path: Path) -> None:
    s = Storage.for_uri("memory://bucket/x")
    with pytest.raises(ValueError, match="cannot operate on"):
        s.write_text(f"file://{tmp_path}/y.txt", "nope")


def test_walk_files_sorted_and_recursive() -> None:
    s = Storage.for_uri("memory://w/root")
    for name in ["memory://w/root/b.txt", "memory://w/root/sub/a.txt", "memory://w/root/c.md"]:
        s.write_text(name, "x")
    files = list(s.walk_files("memory://w/root"))
    assert files == sorted(files)
    assert any(f.endswith("sub/a.txt") for f in files)
    assert len(files) == 3


# -- scheme allowlist (SSRF / arbitrary-fetch guard) ------------------------


@pytest.mark.parametrize("uri", ["file:///tmp/x", "memory://bucket/x"])
def test_allowed_schemes_with_local_backend_construct(uri: str) -> None:
    # file:// and memory:// have in-tree fsspec backends: they fully construct.
    s = Storage.for_uri(uri)
    assert s.protocol == uri.split("://", 1)[0]


def test_allowed_scheme_without_backend_passes_allowlist() -> None:
    # s3:// is on the allowlist, so it is NOT rejected by the allowlist; it only
    # fails later if the optional s3fs backend is missing. The distinction proves
    # the guard is a scheme allowlist, not a "backends installed" check.
    try:
        Storage.for_uri("s3://bucket/key")
    except SchemeNotAllowedError:  # pragma: no cover - would be a real bug
        pytest.fail("s3:// must pass the allowlist")
    except ImportError:
        pass  # s3fs not installed in the dev env — expected, allowlist passed.


@pytest.mark.parametrize("uri", ["http://evil/x", "https://evil/x", "ftp://h/x", "sftp://h/x"])
def test_disallowed_schemes_rejected(uri: str) -> None:
    with pytest.raises(SchemeNotAllowedError, match="not permitted"):
        Storage.for_uri(uri)


def test_allowlist_is_overridable_for_advanced_users() -> None:
    # Default-deny.
    with pytest.raises(SchemeNotAllowedError):
        Storage.for_uri("memory://b/x", allowed_schemes=["file"])
    # An explicit opt-in widens the set — memory:// now constructs.
    s = Storage.for_uri("memory://b/x", allowed_schemes=["file", "memory"])
    assert s.protocol == "memory"


# -- symlink escape handling ------------------------------------------------


def test_symlink_escaping_source_not_read_by_default(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("TOP SECRET")

    src = tmp_path / "src"
    src.mkdir()
    (src / "inside.txt").write_text("ok")
    # A symlink planted inside the source folder that points OUTSIDE it.
    os.symlink(secret, src / "leak.txt")

    s = Storage.for_uri(f"file://{src}")
    files = list(s.walk_files(f"file://{src}"))
    names = {f.rsplit("/", 1)[-1] for f in files}
    # The in-folder file is read; the escaping symlink is NOT.
    assert "inside.txt" in names
    assert "leak.txt" not in names


def test_symlink_escape_followed_when_opted_in(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP SECRET")
    src = tmp_path / "src"
    src.mkdir()
    os.symlink(outside / "secret.txt", src / "leak.txt")

    s = Storage.for_uri(f"file://{src}")
    files = list(s.walk_files(f"file://{src}", follow_symlinks=True))
    names = {f.rsplit("/", 1)[-1] for f in files}
    assert "leak.txt" in names


def test_symlink_inside_source_is_still_read(tmp_path: Path) -> None:
    # A symlink whose target stays INSIDE the source root is fine by default.
    src = tmp_path / "src"
    (src / "real").mkdir(parents=True)
    (src / "real" / "a.txt").write_text("hi")
    os.symlink(src / "real" / "a.txt", src / "link.txt")

    s = Storage.for_uri(f"file://{src}")
    names = {f.rsplit("/", 1)[-1] for f in s.walk_files(f"file://{src}")}
    assert {"a.txt", "link.txt"} <= names

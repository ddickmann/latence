# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The file-identity-keyed read cache (the ADR-0056-amendment pattern, generalised).

The lifetime policy is the point under test, because the pattern's first instance
(``hypergraph._INDEX_CACHE``) had none: (1) identity invalidation — any rewrite changes the
``(dev, ino, size, mtime_ns)`` signature, so stale content is never served; (2)
superseded-signature eviction — decoding a NEW version of a file drops every cached decode of its
OLD versions, so refresh churn never accumulates; (3) bounded LRU — a hard entry cap with
recency-ordered eviction, and an evicted entry rebuilds to an equal value (derived, rebuildable —
eviction changes cost, never results). Plus the ``variant`` sub-key that keeps two decodings of
one file version apart, and the stat-failure passthrough.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from latence_retrieval.read_cache import FileIdentityCache


def _write(path: Path, content: str) -> str:
    path.write_text(content)
    return str(path)


def _rewrite(path: Path, content: str) -> str:
    """Rewrite ``path`` and force a distinct mtime_ns even on coarse filesystem clocks."""
    path.write_text(content)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    return str(path)


def _reader(path: str) -> list[str]:
    return [Path(path).read_text()]


def test_builds_once_per_file_version_and_shares_the_value(tmp_path: Path) -> None:
    cache: FileIdentityCache[list[str]] = FileIdentityCache()
    path = _write(tmp_path / "a.bin", "alpha")
    builds: list[int] = []

    def build() -> list[str]:
        builds.append(1)
        return _reader(path)

    first = cache.get_or_build(path, build)
    second = cache.get_or_build(path, build)
    assert first is second  # shared, not merely equal
    assert len(builds) == 1


def test_a_rewrite_invalidates_and_rebuilds(tmp_path: Path) -> None:
    cache: FileIdentityCache[list[str]] = FileIdentityCache()
    path = _write(tmp_path / "a.bin", "alpha")
    stale = cache.get_or_build(path, lambda: _reader(path))
    _rewrite(tmp_path / "a.bin", "beta")
    fresh = cache.get_or_build(path, lambda: _reader(path))
    assert stale == ["alpha"]  # the old value was never mutated...
    assert fresh == ["beta"]  # ...and the new version is served from a re-decode, never stale


def test_a_rewrite_evicts_the_superseded_version_all_variants(tmp_path: Path) -> None:
    cache: FileIdentityCache[list[str]] = FileIdentityCache()
    path = _write(tmp_path / "a.bin", "alpha")
    cache.get_or_build(path, lambda: _reader(path), variant="upper")
    cache.get_or_build(path, lambda: _reader(path), variant="lower")
    assert len(cache) == 2
    _rewrite(tmp_path / "a.bin", "beta")
    cache.get_or_build(path, lambda: _reader(path), variant="upper")
    # Both decodings of the OLD version are gone; only the new version's entry remains. A process
    # that refreshes one export forever holds at most one version of it.
    assert len(cache) == 1


def test_variants_are_distinct_entries_over_one_file_version(tmp_path: Path) -> None:
    cache: FileIdentityCache[str] = FileIdentityCache()
    path = _write(tmp_path / "a.bin", "alpha")
    upper = cache.get_or_build(path, lambda: "ALPHA", variant="upper")
    lower = cache.get_or_build(path, lambda: "alpha", variant="lower")
    assert (upper, lower) == ("ALPHA", "alpha")
    assert cache.get_or_build(path, lambda: "NEVER BUILT", variant="upper") == "ALPHA"


def test_the_entry_bound_evicts_least_recently_used_first(tmp_path: Path) -> None:
    cache: FileIdentityCache[list[str]] = FileIdentityCache(max_entries=2)
    a = _write(tmp_path / "a.bin", "alpha")
    b = _write(tmp_path / "b.bin", "beta")
    c = _write(tmp_path / "c.bin", "gamma")
    kept = cache.get_or_build(a, lambda: _reader(a))
    cache.get_or_build(b, lambda: _reader(b))
    cache.get_or_build(a, lambda: _reader(a))  # touch a: b becomes the LRU victim
    cache.get_or_build(c, lambda: _reader(c))
    assert len(cache) == 2
    assert cache.get_or_build(a, lambda: ["NEVER BUILT"]) is kept  # a survived (recently used)
    rebuilt = cache.get_or_build(b, lambda: _reader(b))  # b was evicted...
    assert rebuilt == ["beta"]  # ...and rebuilds to an equal value: eviction never changes results


def test_clear_drops_everything(tmp_path: Path) -> None:
    cache: FileIdentityCache[list[str]] = FileIdentityCache()
    path = _write(tmp_path / "a.bin", "alpha")
    first = cache.get_or_build(path, lambda: _reader(path))
    cache.clear()
    assert len(cache) == 0
    assert cache.get_or_build(path, lambda: _reader(path)) is not first


def test_a_stat_failure_propagates_untouched(tmp_path: Path) -> None:
    cache: FileIdentityCache[list[str]] = FileIdentityCache()
    with pytest.raises(FileNotFoundError):
        cache.get_or_build(str(tmp_path / "missing.bin"), lambda: ["NEVER BUILT"])


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "8"])
def test_a_non_positive_entry_bound_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        FileIdentityCache(max_entries=bad)  # type: ignore[arg-type]

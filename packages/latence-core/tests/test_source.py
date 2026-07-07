"""LocalFolderSource reference Provider (ADR-0014): ingest + Provenance invariants.

Unit tests for the reference Source's contracts, including the file-name uniqueness the
downstream gold-set harness and Provenance keys depend on (issue #39.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.stages.source import LocalFolderSource
from latence_core.storage import Storage


def test_duplicate_basename_across_subfolders_is_rejected(tmp_path: Path) -> None:
    # Regression for issue #39.3: two source files with the SAME basename in different
    # subfolders would both stamp Provenance.file_name == "report.txt" and silently merge
    # under the gold-set harness's file-name key. The reference Source enforces the
    # uniqueness GoldDocument documents: a basename collision raises at ingest.
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "b").mkdir(parents=True)
    (docs / "a" / "report.txt").write_text("alpha\n", encoding="utf-8")
    (docs / "b" / "report.txt").write_text("beta\n", encoding="utf-8")

    storage = Storage.for_uri(f"file://{docs}")
    src = LocalFolderSource({"path": f"file://{docs}", "extensions": ["txt"]})
    with pytest.raises(ValueError, match="duplicate file name 'report.txt'"):
        list(src.produce(storage))


def test_same_basename_in_flat_corpus_still_unique(tmp_path: Path) -> None:
    # The guard fires only on a genuine collision: distinct basenames coexist freely, even
    # across subfolders, so a well-formed corpus is unaffected.
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "b").mkdir(parents=True)
    (docs / "a" / "report.txt").write_text("alpha\n", encoding="utf-8")
    (docs / "b" / "memo.txt").write_text("beta\n", encoding="utf-8")

    storage = Storage.for_uri(f"file://{docs}")
    src = LocalFolderSource({"path": f"file://{docs}", "extensions": ["txt"]})
    names = sorted(pi.provenance.file_name for pi in src.produce(storage))
    assert names == ["memo.txt", "report.txt"]


def test_duplicate_basename_only_counts_accepted_extensions(tmp_path: Path) -> None:
    # A basename collision on a file the extension filter SKIPS is not a collision: only
    # produced files share the downstream key, so a .log twin of a .txt is ignored.
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "b").mkdir(parents=True)
    (docs / "a" / "report.txt").write_text("alpha\n", encoding="utf-8")
    (docs / "b" / "report.log").write_text("beta\n", encoding="utf-8")

    storage = Storage.for_uri(f"file://{docs}")
    src = LocalFolderSource({"path": f"file://{docs}", "extensions": ["txt"]})
    names = [pi.provenance.file_name for pi in src.produce(storage)]
    assert names == ["report.txt"]

"""The manifest measures what landed, and reports what the corpus cannot support a claim about.

The interesting assertions here are the *negative* ones: that a corpus with 77,000 ``.doc`` files
and 164 ``.docx`` files is reported as having essentially no modern-OOXML coverage, and that the
manifest says so in its own `limits` section rather than leaving a reader to infer it from a
histogram. That is the difference between a manifest and a brochure.
"""

from __future__ import annotations

import json
from pathlib import Path

from manifest import (
    SourceManifest,
    build_manifest,
    diversity_summary,
    render_markdown,
    scan_tree,
    write_manifest,
)


def _tree(root: Path, spec: dict[str, int], *, size: int = 16) -> None:
    """Materialise ``{extension: count}`` as real files, so the walk measures a real tree."""
    root.mkdir(parents=True, exist_ok=True)
    for ext, count in spec.items():
        for i in range(count):
            (root / f"f{i}.{ext}").write_bytes(b"x" * size)


def test_scan_tree_counts_files_bytes_and_extensions_in_one_walk(tmp_path: Path) -> None:
    _tree(tmp_path / "c", {"pdf": 3, "doc": 2, "html": 1}, size=10)
    count, total, hist = scan_tree(tmp_path / "c")

    assert count == 6
    assert total == 60
    assert dict(hist) == {"pdf": 3, "doc": 2, "html": 1}


def test_a_file_with_no_extension_is_counted_not_dropped(tmp_path: Path) -> None:
    """GovDocs1 files are named by hash; a missing extension is data, not an error."""
    d = tmp_path / "c"
    d.mkdir()
    (d / "no_ext_file").write_bytes(b"x")
    (d / "a.pdf").write_bytes(b"x")

    _count, _total, hist = scan_tree(d)
    assert hist["(none)"] == 1
    assert hist["pdf"] == 1


def test_scan_tree_survives_an_unreadable_entry(tmp_path: Path) -> None:
    """A partial measurement of a 100k-file tree beats no measurement."""
    d = tmp_path / "c"
    d.mkdir()
    (d / "good.pdf").write_bytes(b"x")
    (d / "dangling").symlink_to(tmp_path / "does-not-exist")

    count, _total, hist = scan_tree(d)
    assert count == 1
    assert hist["pdf"] == 1


def test_extension_case_is_normalised(tmp_path: Path) -> None:
    """`.PDF` and `.pdf` are the same claim; splitting them would inflate the diversity count."""
    d = tmp_path / "c"
    d.mkdir()
    (d / "a.PDF").write_bytes(b"x")
    (d / "b.pdf").write_bytes(b"x")

    _count, _total, hist = scan_tree(d)
    assert dict(hist) == {"pdf": 2}


def test_legacy_ole_and_ooxml_are_reported_separately() -> None:
    """The load-bearing split. Summed, 77k .doc + 164 .docx reads as '78k office documents'."""
    summary = diversity_summary({"doc": 76605, "xls": 62876, "ppt": 51292, "docx": 164, "xlsx": 39})

    assert summary["office_legacy_ole"] == 190773
    assert summary["office_ooxml"] == 203
    assert summary["ooxml_share_of_office"] < 0.002


def test_the_manifest_declares_the_ooxml_limit_in_its_own_words() -> None:
    """A reader must not have to derive the limit from the histogram — it is stated."""
    source = SourceManifest(
        name="GovDocs1",
        licence="CC0",
        publishable=True,
        root="/corpus/govdocs1",
        file_count=90000,
        total_bytes=45_000_000_000,
        by_extension={"pdf": 21000, "doc": 7000, "xls": 5700, "docx": 15},
    )
    manifest = build_manifest([source])

    assert any("OOXML is UNTESTED AT VOLUME" in limit for limit in manifest["limits"])
    assert any("SAM.gov" in limit for limit in manifest["limits"])


def test_no_ooxml_limit_is_claimed_when_ooxml_is_actually_present() -> None:
    """Negative control: the limit must be driven by the measurement, not always emitted."""
    source = SourceManifest(
        name="Hypothetical",
        licence="CC0",
        publishable=True,
        root="/x",
        file_count=5000,
        total_bytes=1,
        by_extension={"docx": 3000, "xlsx": 2000},
    )
    manifest = build_manifest([source])

    assert not any("UNTESTED AT VOLUME" in limit for limit in manifest["limits"])


def test_an_unpublishable_source_is_named_as_such() -> None:
    """Fair-use sources must be flagged: the manifest is what stops one reaching an artifact."""
    manifest = build_manifest(
        [
            SourceManifest("GovDocs1", "CC0", True, "/a", 10, 1, {"pdf": 10}),
            SourceManifest("IIT-CDIP", "fair use", False, "/b", 10, 1, {"tif": 10}),
        ]
    )

    limit = next(limit for limit in manifest["limits"] if "Not publishable" in limit)
    assert "IIT-CDIP" in limit
    assert "GovDocs1" not in limit


def test_a_wholly_unpublishable_corpus_says_nothing_may_be_published() -> None:
    manifest = build_manifest(
        [SourceManifest("IIT-CDIP", "fair use", False, "/b", 10, 1, {"tif": 10})]
    )
    assert any("NO acquired source permits redistribution" in x for x in manifest["limits"])


def test_manifest_round_trips_to_disk_and_the_markdown_carries_the_limits(tmp_path: Path) -> None:
    """The limits section is not optional decoration — assert it reaches the rendered file."""
    sources = [
        SourceManifest("GovDocs1", "CC0", True, "/a", 90000, 45_000_000_000, {"pdf": 21000}),
        SourceManifest("SEC VPRR", "SEC public information", True, "/b", 8000, 3e9, {"pdf": 8000}),
    ]
    manifest = write_manifest(sources, tmp_path)

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    assert on_disk["totals"]["file_count"] == 98000

    rendered = (tmp_path / "MANIFEST.md").read_text()
    assert "Limits this corpus cannot support a claim about" in rendered
    assert "UNTESTED AT VOLUME" in rendered
    assert "SEC VPRR" in rendered


def test_render_is_deterministic() -> None:
    """Same measurement ⇒ same document, so a manifest diff means the corpus changed."""
    sources = [SourceManifest("A", "CC0", True, "/a", 5, 10, {"pdf": 3, "doc": 2})]
    assert render_markdown(build_manifest(sources)) == render_markdown(build_manifest(sources))


def test_totals_aggregate_across_sources() -> None:
    manifest = build_manifest(
        [
            SourceManifest("A", "CC0", True, "/a", 10, 100, {"pdf": 10}),
            SourceManifest("B", "CC-BY-4.0", True, "/b", 5, 50, {"doc": 5}),
        ]
    )
    assert manifest["totals"] == {"file_count": 15, "total_bytes": 150}
    assert manifest["diversity"]["distinct_extensions"] == 2

"""Dataset pinning, manifest verification and the recorded license facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from latence_benchmark.datasets import (
    DATASETS,
    OLMOCR_BENCH,
    OMNIDOCBENCH,
    DatasetSpec,
    DatasetUnavailableError,
    LocalDatasetSource,
    get_dataset,
)
from latence_benchmark.licensing import DatasetLicense

if TYPE_CHECKING:
    from pathlib import Path

_LICENCE = DatasetLicense(
    dataset_license="ODC-BY-1.0",
    code_license="Apache-2.0",
    commercial_use=True,
    attribution_required=True,
    verified_on="2026-07-18",
    source_url="https://example.invalid",
)


def _spec(revision: str, manifest: tuple[str, ...] = ("data.jsonl",)) -> DatasetSpec:
    return DatasetSpec(
        name="fixture",
        repo_id="owner/name",
        revision=revision,
        homepage="https://example.invalid",
        license=_LICENCE,
        manifest=manifest,
    )


@pytest.mark.parametrize(
    "revision",
    ["main", "v1.0", "54a96a6", "54A96A6FB6A2BD3B297E59869491DB4D3625B711", ""],
)
def test_a_moving_revision_is_refused(revision: str) -> None:
    """A branch, a tag or an abbreviated SHA cannot pin a reproducible number."""
    with pytest.raises(ValueError, match="not a full 40-character"):
        _spec(revision)


def test_a_full_sha_is_accepted() -> None:
    assert _spec("54a96a6fb6a2bd3b297e59869491db4d3625b711").revision.startswith("54a96a6")


def test_an_empty_manifest_is_refused() -> None:
    with pytest.raises(ValueError, match="empty manifest"):
        _spec("54a96a6fb6a2bd3b297e59869491db4d3625b711", manifest=())


def test_a_partial_download_is_refused_rather_than_partially_scored(tmp_path: Path) -> None:
    """Scoring only the splits that arrived would publish a number that is not the benchmark."""
    spec = _spec("54a96a6fb6a2bd3b297e59869491db4d3625b711", manifest=("a.jsonl", "b.jsonl"))
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DatasetUnavailableError, match="missing 1 of 2"):
        LocalDatasetSource(tmp_path).materialise(spec)


def test_a_complete_local_root_materialises(tmp_path: Path) -> None:
    spec = _spec("54a96a6fb6a2bd3b297e59869491db4d3625b711", manifest=("a.jsonl",))
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    assert LocalDatasetSource(tmp_path).materialise(spec) == tmp_path


def test_a_missing_local_root_names_the_pinned_revision(tmp_path: Path) -> None:
    spec = _spec("54a96a6fb6a2bd3b297e59869491db4d3625b711")
    with pytest.raises(DatasetUnavailableError, match=spec.revision):
        LocalDatasetSource(tmp_path / "nope").materialise(spec)


def test_every_registered_dataset_is_pinned_and_license_verified() -> None:
    for name, spec in DATASETS.items():
        assert spec.name == name
        assert len(spec.revision) == 40
        assert spec.license.verified, f"{name} has an unverified license record"
        assert spec.license.source_url.startswith("https://")


def test_omnidocbench_is_recorded_as_non_commercial_despite_the_apache_toolkit() -> None:
    """The trap this harness exists to avoid.

    OmniDocBench's repository badge is Apache-2.0 — that is the *evaluation toolkit*. The
    corpus copyright statement restricts it to research use. If someone ever "tidies" this
    record to a single Apache-2.0 field, the dataset would silently become fetchable by default
    and its numbers publishable, which the project has no right to do.
    """
    assert OMNIDOCBENCH.license.code_license == "Apache-2.0"
    assert OMNIDOCBENCH.license.commercial_use is False
    assert "research" in OMNIDOCBENCH.license.dataset_license


def test_the_default_parse_dataset_permits_commercial_use() -> None:
    assert OLMOCR_BENCH.license.dataset_license == "ODC-BY-1.0"
    assert OLMOCR_BENCH.license.commercial_use is True
    assert OLMOCR_BENCH.license.attribution_required is True
    assert OLMOCR_BENCH.citation


def test_unknown_dataset_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="olmocr-bench"):
        get_dataset("nope")

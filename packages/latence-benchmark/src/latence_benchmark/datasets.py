"""Pinned public benchmark datasets and the two-adapter acquisition seam.

Reproducibility is the whole point of this harness, so a dataset is identified by a **full
40-hex commit SHA**, never by ``main`` or a tag: :class:`DatasetSpec` rejects anything else at
construction, which makes "we pinned it" a structural property rather than a habit. A spec also
carries the file ``manifest`` it must see on disk, so a truncated download fails loudly instead
of silently scoring fewer tests, and the :class:`~latence_benchmark.licensing.DatasetLicense`
record that :func:`~latence_benchmark.licensing.require_usable` gates on before a byte is
fetched.

The acquisition seam has one implementation per real situation, and no speculative ones:

* :class:`LocalDatasetSource` — the dataset is already on disk (an air-gapped host, a CI cache,
  a reviewer who downloaded it once). Runs with no network and no extra dependency.
* :class:`HuggingFaceDatasetSource` — fetch it at the pinned revision. Rides the ``[hub]`` extra
  (ADR-0016), so the base install stays weightless and offline.
* :class:`ShippedCorpusSource` — the corpus travels **inside this package**, which is the only
  honest way to benchmark redaction: a PII corpus must be synthetic, and a synthetic corpus has
  no upstream repository to pin a commit against. Its ``revision`` is therefore a *content*
  digest (:func:`content_digest`), verified on every run, so editing the corpus without
  re-pinning fails loudly instead of silently republishing an old number against new data.

All three funnel through :func:`verify_manifest`, so "materialised" means the same thing however
the bytes arrived.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from latence_benchmark.licensing import DatasetLicense

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_REVISION_KINDS = frozenset({"commit", "content-digest"})


class DatasetUnavailableError(RuntimeError):
    """The dataset could not be materialised at its pinned revision."""


@dataclass(frozen=True)
class DatasetSpec:
    """One public benchmark dataset, pinned to an exact revision.

    :param name: Harness-local identifier (``--dataset``).
    :param repo_id: Upstream repository id (e.g. a Hugging Face ``owner/name``).
    :param revision: Full 40-hex commit SHA — or, for a corpus shipped inside this package, the
        first 40 hex of its content digest (see ``revision_kind``). Branch names and tags are
        rejected either way: they move, and a moving dataset makes every published number
        unreproducible.
    :param revision_kind: ``"commit"`` (an upstream commit SHA) or ``"content-digest"`` (a
        digest over the shipped bytes, checked by :class:`ShippedCorpusSource`). Recorded on the
        results page so a reader knows which kind of pin they are trusting.
    :param homepage: Human-facing page for the dataset.
    :param license: The separately verified corpus + toolkit license record (ADR-0012).
    :param manifest: Repo-relative paths that MUST exist once materialised.
    :param citation: How the dataset asks to be cited / attributed.
    """

    name: str
    repo_id: str
    revision: str
    homepage: str
    license: DatasetLicense
    manifest: tuple[str, ...]
    citation: str = ""
    revision_kind: str = "commit"

    def __post_init__(self) -> None:
        if self.revision_kind not in _REVISION_KINDS:
            msg = (
                f"Dataset {self.name!r} declares revision_kind {self.revision_kind!r}; expected "
                f"one of {', '.join(sorted(_REVISION_KINDS))}."
            )
            raise ValueError(msg)
        if not _FULL_SHA.match(self.revision):
            msg = (
                f"Dataset {self.name!r} is pinned to {self.revision!r}, which is not a full "
                "40-character lowercase hex commit SHA. Branches and tags move, so a run "
                "against them is not reproducible and is refused."
            )
            raise ValueError(msg)
        if not self.manifest:
            msg = f"Dataset {self.name!r} declares an empty manifest; nothing could be verified."
            raise ValueError(msg)


def verify_manifest(root: Path, spec: DatasetSpec) -> Path:
    """Return ``root`` if every manifest entry exists under it, else raise.

    A partially-downloaded dataset is the classic false-green: the harness would happily score
    the splits it *did* get and publish a number that means something else.
    """
    missing = [entry for entry in spec.manifest if not (root / entry).exists()]
    if missing:
        msg = (
            f"Dataset {spec.name!r} at {root} is incomplete — missing {len(missing)} of "
            f"{len(spec.manifest)} manifest entries (first: {missing[0]!r}). Scoring a partial "
            "dataset would publish a number that is not the benchmark, so this is refused."
        )
        raise DatasetUnavailableError(msg)
    return root


def content_digest(root: Path, spec: DatasetSpec) -> str:
    """First 40 hex of a sha256 over ``spec``'s manifest files, in manifest order.

    The content-addressed counterpart of a commit SHA, for a corpus that has no upstream commit
    because it ships inside this package.
    """
    digest = hashlib.sha256()
    for entry in spec.manifest:
        digest.update((root / entry).read_bytes())
    return digest.hexdigest()[:40]


@runtime_checkable
class DatasetSource(Protocol):
    """Makes a pinned :class:`DatasetSpec` available as a local directory."""

    def materialise(self, spec: DatasetSpec) -> Path:
        """Return a local directory holding ``spec`` at its pinned revision."""
        ...


@dataclass(frozen=True)
class LocalDatasetSource:
    """A dataset already on disk. No network, no optional dependency.

    :param root: Directory that directly contains the dataset's manifest paths.
    """

    root: Path

    def materialise(self, spec: DatasetSpec) -> Path:
        if not self.root.is_dir():
            msg = (
                f"Dataset {spec.name!r} was requested from local root {self.root}, which is not "
                "a directory. Download it once at the pinned revision "
                f"({spec.revision}) from {spec.homepage} and point --dataset-root at it."
            )
            raise DatasetUnavailableError(msg)
        return verify_manifest(self.root, spec)


@dataclass(frozen=True)
class HuggingFaceDatasetSource:
    """Fetch the dataset from the Hugging Face Hub **at the pinned revision**.

    Rides the ``[hub]`` extra (ADR-0016): ``huggingface_hub`` is imported lazily, so the base
    install and every offline test are unaffected by its absence.

    :param cache_dir: Optional download cache directory.
    :param allow_patterns: Optional subset of the repo to fetch (still manifest-verified).
    """

    cache_dir: Path | None = None
    allow_patterns: tuple[str, ...] | None = None

    def materialise(self, spec: DatasetSpec) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - exercised via the message, not the import
            msg = (
                "Remote dataset acquisition needs the optional [hub] extra "
                "(`pip install latence-benchmark[hub]`). To run fully offline, download "
                f"{spec.repo_id} at revision {spec.revision} once and use --dataset-root."
            )
            raise DatasetUnavailableError(msg) from exc
        local = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            allow_patterns=list(self.allow_patterns) if self.allow_patterns else None,
        )
        return verify_manifest(Path(local), spec)


@dataclass(frozen=True)
class ShippedCorpusSource:
    """A corpus that travels inside this package, verified against its pinned content digest.

    :param root: Directory holding the corpus files (defaults to ``latence_benchmark/corpora``).

    The digest check is the point. A shipped corpus is trivially editable by anyone with a text
    editor, so without it a one-character change to the gold set would silently move every
    published number while the results page still claimed the same "revision".
    """

    root: Path | None = None

    def materialise(self, spec: DatasetSpec) -> Path:
        root = self.root if self.root is not None else Path(__file__).parent / "corpora"
        if not root.is_dir():
            msg = f"Shipped corpus directory {root} is missing from the installed package."
            raise DatasetUnavailableError(msg)
        verify_manifest(root, spec)
        actual = content_digest(root, spec)
        if actual != spec.revision:
            msg = (
                f"Shipped corpus {spec.name!r} has content digest {actual} but is pinned to "
                f"{spec.revision}. The corpus on disk is not the corpus the published numbers "
                "were produced against, so the run is refused. If the change is intended, "
                "re-pin the DatasetSpec revision and re-publish the results."
            )
            raise DatasetUnavailableError(msg)
        return root


# -- The registry ------------------------------------------------------------------------------
#
# Every entry's licenses were verified by hand, separately for corpus and toolkit, on the date
# recorded below. Adding an entry without doing that is what `UNVERIFIED` is for.

_OLMOCR_SPLITS = (
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
)

OLMOCR_BENCH = DatasetSpec(
    name="olmocr-bench",
    repo_id="allenai/olmOCR-bench",
    # Pinned SHA read from the Hub API on 2026-07-18. `main` would move under us.
    revision="54a96a6fb6a2bd3b297e59869491db4d3625b711",
    homepage="https://huggingface.co/datasets/allenai/olmOCR-bench",
    license=DatasetLicense(
        # Corpus: the dataset card's own `license:` field.
        dataset_license="ODC-BY-1.0",
        # Toolkit: the olmocr evaluation code, a SEPARATE artifact under a SEPARATE license.
        code_license="Apache-2.0",
        commercial_use=True,
        attribution_required=True,
        verified_on="2026-07-18",
        source_url="https://huggingface.co/datasets/allenai/olmOCR-bench",
        notes=(
            "Corpus license read from the dataset card front-matter (`license: odc-by`), which "
            "states: 'This dataset is licensed under ODC-BY-1.0. It is intended for research and "
            "educational use in accordance with AI2's Responsible Use Guidelines.' ODC-BY-1.0 is "
            "permissive-with-attribution and does not restrict commercial use. The evaluation "
            "toolkit (github.com/allenai/olmocr) is Apache-2.0, verified separately per ADR-0012."
        ),
    ),
    manifest=tuple(f"bench_data/{split}.jsonl" for split in _OLMOCR_SPLITS),
    citation="olmOCR-bench, Allen Institute for AI, ODC-BY-1.0 — https://huggingface.co/datasets/allenai/olmOCR-bench",
)

OMNIDOCBENCH = DatasetSpec(
    name="omnidocbench",
    repo_id="opendatalab/OmniDocBench",
    # Pinned SHA read from the Hub API on 2026-07-18.
    revision="aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec",
    homepage="https://huggingface.co/datasets/opendatalab/OmniDocBench",
    license=DatasetLicense(
        # THE TRAP: the GitHub repo badge says Apache-2.0 — that is the TOOLKIT. The corpus
        # copyright statement is research-only, so this dataset is gated, never a default.
        dataset_license="research-only (non-commercial)",
        code_license="Apache-2.0",
        commercial_use=False,
        attribution_required=True,
        verified_on="2026-07-18",
        source_url="https://github.com/opendatalab/OmniDocBench",
        notes=(
            "The evaluation toolkit is Apache-2.0, but the CORPUS is not: the project's Copyright "
            "Statement says 'The dataset is for research purposes only and not for commercial "
            "use.' The Hugging Face dataset card carries no SPDX license field. Verified "
            "separately per ADR-0012 — reading the repository's Apache-2.0 badge as the dataset "
            "license would be wrong."
        ),
    ),
    manifest=("OmniDocBench.json",),
    citation="OmniDocBench (CVPR 2025), OpenDataLab — research use only",
)

_DATASETS: dict[str, DatasetSpec] = {
    OLMOCR_BENCH.name: OLMOCR_BENCH,
    OMNIDOCBENCH.name: OMNIDOCBENCH,
}

#: Every dataset the harness knows, keyed by ``DatasetSpec.name``. Datasets defined by a suite
#: module register themselves here at import time (see :func:`register_dataset`), so the
#: license/pinning invariants hold over all of them and not just the ones declared above.
DATASETS: Mapping[str, DatasetSpec] = _DATASETS


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Add ``spec`` to :data:`DATASETS`, refusing to redefine a name with a different spec."""
    existing = _DATASETS.get(spec.name)
    if existing is not None and existing != spec:
        msg = f"A different dataset named {spec.name!r} is already registered."
        raise ValueError(msg)
    _DATASETS[spec.name] = spec
    return spec


def dataset_names() -> Sequence[str]:
    """Registered dataset names, sorted."""
    return sorted(DATASETS)


def get_dataset(name: str) -> DatasetSpec:
    """Return the registered :class:`DatasetSpec` called ``name``."""
    try:
        return DATASETS[name]
    except KeyError:
        available = ", ".join(dataset_names())
        msg = f"Unknown benchmark dataset {name!r}. Available: {available}"
        raise KeyError(msg) from None

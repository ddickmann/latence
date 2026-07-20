"""The Delta win, measured through the REAL DeltaRunner — not a hand-injected exclusion.

``test_sharepoint_source.py`` proves the Provider skips an excluded document without
downloading it, but it injects ``exclude_document_ids`` into the Provider config by hand. That
bypasses the runner, so it cannot see whether the DeltaRunner ever *asks* the Source to skip
anything, nor what the runner itself transfers while enumerating the current documents. This
module drives the real :class:`~latence_core.delta_runner.DeltaRunner` over ONE shared fake
transport and counts the byte-transferring calls — the only honest witness that "an unchanged
document is skipped without transferring a byte" holds end to end.

The Provider is wired by the ``"module:factory"`` transport reference (the same mechanism a
YAML stack uses), because a Pipeline config must stay serialisable: the factory hands back the
one shared transport so every run in a test accumulates onto the same call log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from latence_core.capability import DocumentEnumerator
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
from latence_core.storage import Storage
from latence_source_sharepoint.provider import SharePointSource
from latence_source_sharepoint.testing import (
    DOWNLOAD_HOST,
    GRAPH_HOST,
    FakeFile,
    FakeGraphApi,
    FakeGraphTransport,
)

if TYPE_CHECKING:
    from pathlib import Path

TOKEN_ENV = "LATENCE_SHAREPOINT_TOKEN"
TRANSPORT_FACTORY = f"{__name__}:shared_transport"

_LIBRARY = (
    FakeFile(path="/a.md", content=b"# Alpha\n\nContoso signed with Fabrikam.\n"),
    FakeFile(path="/b.md", content=b"# Bravo\n\nFabrikam renewed the lease.\n"),
    FakeFile(path="/c.md", content=b"# Charlie\n\nContoso opened an office.\n"),
)

# The one transport every Provider instance in the current test resolves to (set by the
# ``drive`` fixture). A module global is what a ``"module:factory"`` reference can reach.
_SHARED: FakeGraphTransport | None = None


def shared_transport() -> FakeGraphTransport:
    """The transport factory the pipeline config names."""
    if _SHARED is None:  # pragma: no cover - a test always installs one first
        msg = "no shared transport installed"
        raise RuntimeError(msg)
    return _SHARED


@pytest.fixture()
def drive(monkeypatch: pytest.MonkeyPatch) -> FakeGraphTransport:
    monkeypatch.setenv(TOKEN_ENV, "valid-token")
    transport = FakeGraphTransport(FakeGraphApi(files=list(_LIBRARY)))
    monkeypatch.setattr(f"{__name__}._SHARED", transport, raising=False)
    return transport


def _pipeline(store: str, drive_id: str) -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "sharepoint-delta",
            "storage_uri": store,
            "stages": [
                {
                    "name": "source",
                    "capability": "source",
                    "provider": "source.sharepoint",
                    "config": {
                        "drive_id": drive_id,
                        "graph_base_url": GRAPH_HOST,
                        "transport": TRANSPORT_FACTORY,
                        "auth": {"mode": "env_token", "token_env": TOKEN_ENV},
                    },
                },
                {
                    "name": "parse",
                    "capability": "parse",
                    "provider": "parser.plaintext",
                    "depends_on": ["source"],
                },
                {
                    "name": "chunk",
                    "capability": "chunk",
                    "provider": "chunk.markdown",
                    "depends_on": ["parse"],
                },
                {
                    "name": "export_corpus",
                    "capability": "export",
                    "provider": "export.jsonl_parquet",
                    "depends_on": ["chunk"],
                    "config": {"basename": "corpus"},
                },
            ],
        }
    )


def _transfers(transport: FakeGraphTransport) -> int:
    """Byte-transferring fetches only (the pre-authenticated download leg)."""
    return sum(1 for call in transport.calls if call.url.startswith(DOWNLOAD_HOST))


def test_an_unchanged_library_transfers_no_bytes_on_a_delta_run(
    tmp_path: Path, drive: FakeGraphTransport
) -> None:
    """Two DeltaRunner runs over a drive that did not change: the second transfers nothing.

    Counted on the shared transport, so the discovery pass the DeltaRunner makes to enumerate
    the current document ids is counted too — an enumeration that downloads the whole library
    makes the headline saving zero no matter what the run itself skips.
    """
    store = f"file://{tmp_path}/state"
    pipeline = _pipeline(store, drive.api.drive_id)
    runner = DeltaRunner()

    first = runner.apply_delta(pipeline, run_id="delta-1")
    assert first.delta is not None
    # A first run fetches each document exactly ONCE — enumeration must not double-download.
    assert _transfers(drive) == len(_LIBRARY)

    baseline = _transfers(drive)
    second = runner.apply_delta(pipeline, run_id="delta-2")
    assert second.delta is not None
    assert second.delta.documents_reused == len(_LIBRARY)
    assert second.delta.documents_extracted == 0
    # THE claim under test: nothing changed, so not one byte moves.
    assert _transfers(drive) - baseline == 0


def test_only_the_changed_document_is_transferred_on_a_delta_run(
    tmp_path: Path, drive: FakeGraphTransport
) -> None:
    store = f"file://{tmp_path}/state"
    pipeline = _pipeline(store, drive.api.drive_id)
    runner = DeltaRunner()

    runner.apply_delta(pipeline, run_id="delta-1")
    baseline = _transfers(drive)

    drive.api.files = [
        FakeFile(path="/a.md", content=_LIBRARY[0].content),
        FakeFile(path="/b.md", content=b"# Bravo\n\nFabrikam ENDED the lease.\n"),
        FakeFile(path="/c.md", content=_LIBRARY[2].content),
    ]
    report = runner.apply_delta(pipeline, run_id="delta-2")
    assert report.delta is not None
    assert report.delta.documents_reused == 2
    assert report.delta.documents_extracted == 1
    # An incremental run must cost LESS than a full run, not more.
    assert _transfers(drive) - baseline == 1


# ---------------------------------------------------------------------------
# The backend the connector is NAMED for (audit R3-connectors_ingestion-8).
#
# Every test above runs on ``FakeFile``'s DEFAULT ``hash_algorithm="sha256Hash"`` — which Graph
# reports on OneDrive Personal. SharePoint Online reports ``quickXorHash``, which is not a
# reproducible SHA-256 content address, so the listing cannot name the document and the
# zero-transfer skip is unavailable there. What must STILL hold on that backend is the weaker
# but non-negotiable bound: an incremental run never costs MORE than a full run. It used to
# cost exactly twice as much — the discovery pass downloaded the whole library to hash it, and
# ``produce`` (whose exclusion guard only ran on a listing-derived id) downloaded it all again.
# ---------------------------------------------------------------------------

_SPO_LIBRARY = tuple(
    FakeFile(path=f"/{name}.md", content=body, hash_algorithm="quickXorHash")
    for name, body in (
        ("a", b"# Alpha\n\nContoso signed with Fabrikam.\n"),
        ("b", b"# Bravo\n\nFabrikam renewed the lease.\n"),
        ("c", b"# Charlie\n\nContoso opened an office.\n"),
        ("d", b"# Delta\n\nContoso hired in Berlin.\n"),
    )
)


@pytest.fixture()
def sharepoint_online(monkeypatch: pytest.MonkeyPatch) -> FakeGraphTransport:
    """A drive that reports ``quickXorHash``, exactly like a real SharePoint Online library."""
    monkeypatch.setenv(TOKEN_ENV, "valid-token")
    transport = FakeGraphTransport(FakeGraphApi(files=list(_SPO_LIBRARY)))
    monkeypatch.setattr(f"{__name__}._SHARED", transport, raising=False)
    return transport


def test_a_quickxorhash_library_never_costs_more_than_a_full_run(
    tmp_path: Path, sharepoint_online: FakeGraphTransport
) -> None:
    """No run — first, no-op delta, or incremental — may fetch a document more than once.

    The connector cannot content-address a ``quickXorHash`` item from the listing, so it must
    fetch it in the run. It must NOT also fetch it in the DeltaRunner's discovery pass: that
    pass runs before any exclusion exists, so whatever it transfers is transferred
    unconditionally, on top of the run's own fetch.
    """
    store = f"file://{tmp_path}/state"
    pipeline = _pipeline(store, sharepoint_online.api.drive_id)
    runner = DeltaRunner()
    library = len(_SPO_LIBRARY)

    first = runner.apply_delta(pipeline, run_id="spo-1")
    assert first.delta is not None
    assert _transfers(sharepoint_online) == library, "the first run double-downloaded"

    baseline = _transfers(sharepoint_online)
    second = runner.apply_delta(pipeline, run_id="spo-2")
    assert second.delta is not None
    # Nothing changed, so every document is still classified UNCHANGED — re-extracting a
    # document must not churn the corpus...
    assert second.delta.documents_unchanged == library
    assert second.delta.documents_added == 0
    assert second.delta.documents_deleted == 0
    # ...and the honest cost witness: at most one pass over the library, never two.
    assert _transfers(sharepoint_online) - baseline == library
    # The extraction accounting names what actually happened, rather than deriving it from a
    # partial enumeration.
    assert second.delta.documents_extracted == library
    assert second.delta.documents_reused == 0

    baseline = _transfers(sharepoint_online)
    sharepoint_online.api.files = [
        FakeFile(path="/a.md", content=_SPO_LIBRARY[0].content, hash_algorithm="quickXorHash"),
        FakeFile(
            path="/b.md",
            content=b"# Bravo\n\nFabrikam ENDED the lease.\n",
            hash_algorithm="quickXorHash",
        ),
        FakeFile(path="/c.md", content=_SPO_LIBRARY[2].content, hash_algorithm="quickXorHash"),
        FakeFile(path="/d.md", content=_SPO_LIBRARY[3].content, hash_algorithm="quickXorHash"),
    ]
    third = runner.apply_delta(pipeline, run_id="spo-3")
    assert third.delta is not None
    assert _transfers(sharepoint_online) - baseline == library


def test_a_hash_less_listing_is_not_enumerated_rather_than_downloaded(
    tmp_path: Path, sharepoint_online: FakeGraphTransport
) -> None:
    """The discovery pass fetches NOTHING — the whole library is fetched by the run alone.

    Split out from the cost bound above so the mechanism itself is pinned: were discovery to
    resume hashing the bytes, this asserts on the download count of the enumeration in
    isolation, not on a total that a second bug could coincidentally rebalance.
    """
    stage_config = _pipeline(f"file://{tmp_path}/state", sharepoint_online.api.drive_id)
    source_config = stage_config.stages[0].config or {}
    provider = SharePointSource(dict(source_config))
    assert isinstance(provider, DocumentEnumerator)

    storage = Storage.for_uri(f"file://{tmp_path}/state")
    enumerated = list(provider.document_ids(storage))

    assert enumerated == [], "a quickXorHash listing cannot be content-addressed"
    assert _transfers(sharepoint_online) == 0, "discovery must not fetch content"

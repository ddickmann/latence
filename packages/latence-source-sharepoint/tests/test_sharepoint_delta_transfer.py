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
from latence_core.delta_runner import DeltaRunner
from latence_core.pipeline import Pipeline
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

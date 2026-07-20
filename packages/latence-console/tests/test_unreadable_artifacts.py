"""Audit R7 — a file the console could not read is never rendered as "the run holds nothing".

``RunArtifacts._read_jsonl`` degrades an IO failure to an exhausted iterator, which is right: the
console must stay legible over whatever is on Storage, and a browse page is not the place for a
native traceback (ADR-0034). What was wrong is that it degraded *silently*. An unreadable
checkpoint was indistinguishable from an absent one, so the browse views turned it into an
affirmative claim about the run — "This run checkpointed no chunks.", "This run assembled no
graph nodes.", "No record in this run's corpus carries a risk marker." — while the files on
Storage said the opposite (ADR-0017/0021: the files are the source of truth).

The realistic triggers are ordinary ops, not exotica: differential ACLs on the ``checkpoints``
tree (it holds UNREDACTED ``content``, so restricting it while the Quality Report stays readable
is a sensible posture), an S3 bucket policy or a transient 403/5xx on that prefix, or a file a
concurrent ADR-0018 Purge removed between the ``ls`` and the ``open``.

These tests drive the real :class:`~latence_console.server.ConsoleApp` over the real fixture run
(4 chunks, entities, graph nodes and edges, and a ``prompt_injection`` risk marker on Storage) and
make the store refuse exactly the way a permission boundary does.
"""

from __future__ import annotations

from typing import IO, Any

import pytest
from latence_console.artifacts import RunArtifacts
from latence_console.server import ConsoleApp
from latence_core.storage import Storage

RUN = "run-0001"

#: Every affirmative absence claim the browse views can make about a run's contents. Not one of
#: them may appear on a page whose reader could not read the files that would refute it.
ABSENCE_CLAIMS = (
    "This run checkpointed no chunks.",
    "This run assembled no graph nodes.",
    "This run assembled no graph edges.",
    "No record in this run's corpus carries a risk marker.",
    "No relation in this graph touches this entity.",
    "Screening held nothing back in this run.",
)


@pytest.fixture
def app(browsable: str) -> ConsoleApp:
    return ConsoleApp.over_storage(Storage.for_uri(browsable), browsable)


def _refuse(monkeypatch: pytest.MonkeyPatch, fragment: str) -> None:
    """Make ``Storage.open`` refuse every URI containing ``fragment``, as an ACL would."""
    real = Storage.open

    def _guarded(self: Storage, uri: str, mode: str = "rb") -> IO[Any]:
        if fragment in uri:
            msg = f"[Errno 13] Permission denied: {uri!r}"
            raise PermissionError(msg)
        return real(self, uri, mode)

    monkeypatch.setattr(Storage, "open", _guarded)


def _body(app: ConsoleApp, path: str) -> str:
    response = app.respond("GET", path)
    assert response.status == 200, (path, response.status)
    return response.body.decode("utf-8")


# -- AC1/AC2: an unreadable file never becomes an affirmative "there is nothing" ---------------


@pytest.mark.parametrize("view", ["chunks", "graph", "quarantine"])
def test_an_unreadable_checkpoint_is_not_rendered_as_an_empty_run(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    """The defect, end to end: every checkpoint refused, page still swore the run was empty."""
    _refuse(monkeypatch, "/checkpoints/")

    page = _body(app, f"/run/{RUN}/{view}")

    assert "Partial view" in page, f"{view} claimed a complete read it never made"
    assert "could not read part of this run" in page
    for claim in ABSENCE_CLAIMS:
        assert claim not in page, f"{view} asserted {claim!r} over files it could not read"


def test_an_unreadable_screening_sidecar_never_denies_the_run_holds_a_risk_marker(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline case: the corpus demonstrably carries a ``prompt_injection`` marker."""
    _refuse(monkeypatch, "/screening/")

    page = _body(app, f"/run/{RUN}/quarantine")

    assert "Partial view" in page
    assert "No record in this run's corpus carries a risk marker." not in page


def test_an_unreadable_quarantine_sidecar_never_says_screening_held_nothing_back(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse(monkeypatch, "/quarantine/")

    page = _body(app, f"/run/{RUN}/quarantine")

    assert "Partial view" in page
    assert "Screening held nothing back in this run." not in page


def test_a_partial_read_never_shows_a_bare_zero_tally(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Entities (0)`` reads as a fact. On a failed read the honest tally is a qualified one."""
    _refuse(monkeypatch, "/checkpoints/")

    page = _body(app, f"/run/{RUN}/graph")

    assert "Entities (0)" not in page
    assert "Relations (0)" not in page
    assert "Entities (0 read so far)" in page
    assert "Relations (0 read so far)" in page


def test_the_notice_names_the_files_that_would_not_read(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader must be able to act on it — "something failed" is not an actionable page."""
    _refuse(monkeypatch, "/checkpoints/")

    page = _body(app, f"/run/{RUN}/chunks")

    assert "graph.jsonl" in page
    assert "chunk.jsonl" in page


# -- AC3: a stream that dies mid-file leaves a PREFIX, which is also a partial read ------------


class _TornStream:
    """A handle that hands back small slices and then fails — a 5xx part-way through a GET."""

    def __init__(self, payload: bytes, *, blocks: int) -> None:
        self._payload = payload
        self._blocks = blocks
        self._served = 0
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if self._served >= self._blocks:
            msg = "connection reset while streaming the checkpoint"
            raise OSError(msg)
        self._served += 1
        window = self._payload[self._offset : self._offset + 200]
        self._offset += len(window)
        return window

    def __enter__(self) -> _TornStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _tear(monkeypatch: pytest.MonkeyPatch, fragment: str, *, blocks: int) -> None:
    real = Storage.open

    def _guarded(self: Storage, uri: str, mode: str = "rb") -> Any:  # noqa: ANN401
        if fragment in uri:
            with real(self, uri, mode) as handle:
                payload: bytes = handle.read()
            return _TornStream(payload, blocks=blocks)
        return real(self, uri, mode)

    monkeypatch.setattr(Storage, "open", _guarded)


def test_a_stream_that_fails_mid_file_is_recorded_not_served_as_the_whole_file(
    browsable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worse half of the defect: a silent PREFIX rendered as a complete list."""
    storage = Storage.for_uri(browsable)
    run_dir = f"{browsable}/_latence/runs/{RUN}"
    whole = RunArtifacts(storage, run_dir)
    complete_edges = len(whole.graph_edges())
    assert complete_edges > 0, "fixture must hold edges for this regression to mean anything"

    _tear(monkeypatch, "/checkpoints/graph.jsonl", blocks=1)
    torn = RunArtifacts(storage, run_dir)
    partial_edges = len(torn.graph_edges())

    assert partial_edges < complete_edges, "the tear must actually cut the file short"
    assert not torn.truncated(), "the CAP was never hit — only the stream failed"
    assert torn.partial(), "a torn read is a partial read"
    assert any(u.endswith("/checkpoints/graph.jsonl") for u in torn.unreadable())


def test_a_torn_graph_read_shows_the_partial_notice_on_the_entity_page(
    browsable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entity page resolves from the surviving prefix, so it must say the read was short."""
    _tear(monkeypatch, "/checkpoints/graph.jsonl", blocks=1)
    app = ConsoleApp.over_storage(Storage.for_uri(browsable), browsable)

    listing = _body(app, f"/run/{RUN}/graph")
    assert "Partial view" in listing


# -- AC4/AC5: the reader's own vocabulary, and the healthy run left alone ----------------------


def test_unreadable_is_empty_and_partial_is_false_on_a_healthy_run(browsable: str) -> None:
    artifacts = RunArtifacts(Storage.for_uri(browsable), f"{browsable}/_latence/runs/{RUN}")

    assert artifacts.chunks()
    assert artifacts.graph_nodes()
    assert artifacts.unreadable() == ()
    assert artifacts.partial() is False
    assert artifacts.truncated() is False


@pytest.mark.parametrize("view", ["chunks", "graph", "quarantine"])
def test_a_healthy_run_carries_no_partial_notice(app: ConsoleApp, view: str) -> None:
    """Off-by-default means unchanged: nothing failed, so nothing is qualified."""
    page = _body(app, f"/run/{RUN}/{view}")

    assert "Partial view" not in page
    assert "could not read part of this run" not in page


def test_a_genuinely_empty_run_still_states_its_emptiness_plainly(populated: str) -> None:
    """The qualification is driven by a FAILED read, never by an empty one.

    ``run-0002`` has a Quality Report and no artifacts at all: every checkpoint directory is
    absent, which ``_jsonl_files`` answers with an empty list without ever opening a file. That
    is a complete read of nothing, and the page is entitled to say so.
    """
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)

    page = _body(app, "/run/run-0002/chunks")

    assert "Partial view" not in page
    assert "This run checkpointed no chunks." in page


def test_unreadable_survives_the_reader_being_asked_twice(
    browsable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memoised answers must not memoise away the failure that produced them."""
    _refuse(monkeypatch, "/checkpoints/")
    artifacts = RunArtifacts(Storage.for_uri(browsable), f"{browsable}/_latence/runs/{RUN}")

    assert artifacts.chunks() == []
    assert artifacts.chunks() == []
    assert artifacts.partial() is True
    assert len(artifacts.unreadable()) >= 2


def _claims_in(page: str) -> list[str]:
    return [claim for claim in ABSENCE_CLAIMS if claim in page]


def test_no_view_makes_an_absence_claim_while_anything_was_unreadable(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One sweep over every view and every claim, so a new view cannot regress this quietly."""
    _refuse(monkeypatch, "/_latence/runs/")
    paths: list[str] = [f"/run/{RUN}/{v}" for v in ("chunks", "graph", "quarantine")]
    for path in paths:
        page = _body(app, path)
        assert _claims_in(page) == [], (path, _claims_in(page))


def test_the_partial_notice_keeps_the_console_offline(
    app: ConsoleApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new notice embeds Storage URIs — it must not turn one into an off-machine reference."""
    from latence_console.render import external_references  # noqa: PLC0415 - local to this check

    _refuse(monkeypatch, "/checkpoints/")

    assert external_references(_body(app, f"/run/{RUN}/graph")) == []

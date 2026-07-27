"""The debugging loop (#188): quarantine browser, chunk inspector, KG browser with Evidence.

Every assertion here is about **observable behaviour through the console's own interface** —
the bytes a request returns, or the resolved values the browse derivations produce. Nothing
asserts a flag that merely records intent: "the chunk inspector resolves offsets" is tested
by demanding the resolved numbers, computed by the framework's own resolvers over a chunk
whose offset map is deliberately NOT the identity, so a naive ``char_start + len(content)``
implementation fails.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import cast

import pytest
from latence_console import browse
from latence_console.artifacts import RunArtifacts
from latence_console.render import external_references, render_chunk
from latence_console.server import ConsoleApp
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    PageSlice,
    PIISpan,
    RedactionAction,
)
from latence_core.storage import Storage
from reports import (
    _PAGE_MAP,
    CHUNK_ID,
    DANGLING_MENTION_ID,
    DOC_ID,
    MASKED_CHUNK_TEXT,
    MENTION_ID,
    NODE_ORG,
    NODE_PERSON,
    SENSITIVE_CHUNK_ID,
    _provenance,
    write_artifacts,
    write_run,
)

RUN = "run-0001"


def _tree_digest(root: Path) -> str:
    """A digest over every file's path, bytes, and mtime under ``root``."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
        h.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
    return h.hexdigest()


@pytest.fixture
def app(browsable: str) -> ConsoleApp:
    return ConsoleApp.over_storage(Storage.for_uri(browsable), browsable)


def _page(app: ConsoleApp, path: str) -> str:
    response = app.respond("GET", path)
    assert response.status == 200, path
    return response.body.decode("utf-8")


def _chunk_url(app: ConsoleApp, chunk_id: str) -> str:
    """The console's own link to a chunk — found in the page, not hand-built."""
    listing = _page(app, f"/run/{RUN}/chunks")
    hrefs: list[str] = re.findall(r'href="(/run/[^"]*?/chunk\?[^"]+)"', listing)
    for href in hrefs:
        if app.respond("GET", href).status == 200 and chunk_id.rsplit("#", 1)[-1] in href:
            return href.replace("&amp;", "&")
    raise AssertionError(f"no console link to {chunk_id}")


# -- AC 1: the Quarantine browser -------------------------------------------


def test_quarantine_page_lists_what_was_held_back_with_its_recorded_reason(
    app: ConsoleApp,
) -> None:
    page = _page(app, f"/run/{RUN}/quarantine")
    assert "archive.zip" in page
    assert "zip_bomb" in page
    assert "screening.intake_signature" in page
    assert "zip compression ratio 966.9x exceeds the 50.0x cap" in page
    assert "intake" in page
    assert "2,169" in page  # the bytes actually withheld


def test_quarantine_page_lists_risk_flagged_records_with_their_markers(
    app: ConsoleApp,
) -> None:
    page = _page(app, f"/run/{RUN}/quarantine")
    assert "prompt_injection" in page
    assert "screening.content_keyword" in page
    assert "matched injection pattern" in page
    # The marker is attributed to the checkpoint that raised it, joined from the sidecar.
    assert "content" in page


def test_flag_entries_come_from_the_markers_that_survived_into_the_corpus(
    browsable: str,
) -> None:
    artifacts = ConsoleApp.over_storage(
        Storage.for_uri(browsable), browsable
    )._index.artifacts(RUN)
    entries = browse.flag_entries(artifacts)
    assert [(e.category, e.checkpoint, e.chunk_id) for e in entries] == [
        ("prompt_injection", "content", CHUNK_ID)
    ]


# -- AC 2: the chunk inspector ----------------------------------------------


def test_chunk_inspector_resolves_provenance_to_document_page_and_offsets(
    app: ConsoleApp,
) -> None:
    page = _page(app, _chunk_url(app, CHUNK_ID))
    assert "memo.md" in page
    assert "file:///corpus/memo.md" in page
    # Provenance's own span and page range.
    assert "10–91" in page
    assert "1–2" in page
    # Resolved through the chunk's OWN offset map: the second run shifts by 40, so the
    # chunk's 51 stripped characters cover original 10..91 — NOT 10..61 (char_start+len).
    assert "10–91" in page
    assert "10–61" not in page
    # The source document it resolved against, as the run actually parsed it.
    assert "2 page(s), 120 characters" in page


def test_chunk_inspector_shows_classification_and_redaction_disposition(
    app: ConsoleApp,
) -> None:
    page = _page(app, _chunk_url(app, CHUNK_ID))
    assert "report" in page  # Classification.category
    assert "internal" in page  # Classification.sensitivity
    assert "Redaction ran and handled 1 PII span(s)" in page
    assert "person" in page
    assert "[PERSON]" in page
    assert "mask" in page
    # The PII span, located BOTH in the chunk and in the source document (0–12 chunk-local
    # resolves to 10–22 in the original markdown through the same offset map).
    assert "0–12" in page
    assert "10–22" in page


def _chunk_without_offset_map(char_start: int | None) -> ChunkRecord:
    """A chunk with PII spans but NO offset map — a third-party or pre-v11 record."""
    return ChunkRecord(
        record_id="sha256:" + "c" * 58,
        provenance=_provenance(char_start=char_start, char_end=120),
        classification=Classification(language="en", category="report", sensitivity="confidential"),
        document_record_id=DOC_ID,
        index=1,
        content="Ada Lovelace signs the appendix.",
        token_count=7,
        # v19: the chunk carries the spans it overlaps, cut from the document's map.
        page_slice=PageSlice.for_span(_PAGE_MAP, char_start or 0, 120),
        offset_map=None,
        masked_content="[PERSON] signs the appendix.",
        pii_spans=[
            PIISpan(
                pii_type="person",
                detector="regex.person",
                char_start=0,
                char_end=12,
                action=RedactionAction.MASK,
                placeholder="[PERSON]",
                confidence=0.9,
            )
        ],
    )


class _NoDocuments:
    """Artifacts that hold no parsed document — the chunk's own page_map is all there is."""

    def document(self, _document_id: str) -> None:
        return None


def _no_documents() -> RunArtifacts:
    """The double, as the parameter type ``inspect_chunk`` declares.

    ``inspect_chunk`` reaches exactly one method on its artifacts (``document``), which is all
    this double provides; the cast states that narrowness rather than widening the real
    signature to a structural type nothing else needs.
    """
    return cast("RunArtifacts", _NoDocuments())


def test_a_pii_span_without_an_offset_map_is_shifted_by_the_chunks_document_anchor() -> None:
    """R2-console_readonly-12: chunk-LOCAL offsets must never be shown as document offsets.

    Without an offset map the framework's documented degrade is ``base_char_start + local``
    (contracts.py, ``redaction_policy._span_for``). Reporting the raw local numbers — and
    looking THEM up in the document page map — made the inspector contradict itself: the
    same page reported the enclosing chunk at 91–120 on page 2 while the PII row inside it
    claimed 0–12 on page 1 (ADR-0021/0031, Provenance is true).
    """
    inspection = browse.inspect_chunk(_no_documents(), _chunk_without_offset_map(91))

    row = inspection.redaction.spans[0]
    assert row.chunk_span == "0–12"  # the chunk-local span still reads chunk-local
    assert row.source_span == "91–103"  # shifted by the chunk's anchor, not the raw local span
    assert row.pages == "2"  # page 2 holds 50..120 — the local numbers would have said 1
    # And the enclosing chunk on the very same page agrees.
    assert inspection.resolved_span == "91–120"
    assert inspection.resolved_pages == "2"


def test_a_pii_span_with_no_document_anchor_at_all_reports_unknown_not_chunk_local() -> None:
    """No offset map AND no ``provenance.char_start`` = the span cannot be placed at all."""
    inspection = browse.inspect_chunk(_no_documents(), _chunk_without_offset_map(None))

    row = inspection.redaction.spans[0]
    assert row.chunk_span == "0–12"
    assert row.source_span == "—"
    assert row.pages == "—"


def test_chunk_inspector_shows_the_pii_safe_text_never_the_raw_content(
    app: ConsoleApp,
) -> None:
    page = _page(app, _chunk_url(app, CHUNK_ID))
    assert MASKED_CHUNK_TEXT in page
    assert "Ada Lovelace works at" not in page


def test_a_chunk_whose_redaction_control_was_off_says_so(app: ConsoleApp) -> None:
    """An empty PII list must never be rendered as "clean" when the control never ran."""
    page = _page(app, _chunk_url(app, SENSITIVE_CHUNK_ID))
    assert "the control was OFF" in page
    assert "confidential" in page


def _floor_masked_sensitive_chunk() -> ChunkRecord:
    """A confidential chunk whose POLICY ran no detectors but whose ADR-0044 floor fired.

    The real shape produced by ``latence run`` with ``policies: {confidential: {action: mask,
    types: [passport_number]}}`` over a document containing an IBAN: the allow-list intersects
    no enabled detector (so the H-C1 §4 no-op flag is set) yet the universal financial-PII
    floor detected and masked the IBAN regardless of backend.
    """
    content = "Please wire the retainer to IBAN DE44500105175407324931 before the deadline."
    return ChunkRecord(
        record_id="sha256:" + "f" * 58,
        provenance=_provenance(page_start=2, page_end=2, char_start=0, char_end=len(content)),
        classification=Classification(language="en", category="report", sensitivity="confidential"),
        document_record_id=DOC_ID,
        index=1,
        content=content,
        token_count=14,
        page_slice=PageSlice.for_span(_PAGE_MAP, 0, len(content)),
        masked_content="Please wire the retainer to IBAN [IBAN] before the deadline.",
        pii_spans=[
            PIISpan(
                pii_type="iban",
                detector="floor:iban",
                char_start=33,
                char_end=55,
                action=RedactionAction.MASK,
                placeholder="[IBAN]",
                confidence=1.0,
            )
        ],
        redaction_disabled_for_sensitive=True,
    )


def test_a_floor_masked_chunk_is_never_reported_as_a_control_that_was_off() -> None:
    """R3-console_readonly-11: "the control was OFF" above the IBAN the floor masked.

    ``redaction_disabled_for_sensitive`` and a non-empty ``pii_spans`` are not mutually
    exclusive since ADR-0044 — the universal financial-PII floor runs wherever redaction runs.
    The inspector took the no-op branch unconditionally, so the page told an auditor the PII
    control was OFF and warned about an "empty PII list" directly above a table listing the
    masked IBAN: a false statement about a redaction control on the PII-audit surface.
    """
    chunk = _floor_masked_sensitive_chunk()
    inspection = browse.inspect_chunk(_no_documents(), chunk)

    # The span the floor handled is really there — the premise of the false claim.
    assert [s.pii_type for s in inspection.redaction.spans] == ["iban"]
    assert inspection.redaction.spans[0].detector == "floor:iban"

    summary = inspection.redaction.summary
    assert "the control was OFF" not in summary
    assert "empty PII list" not in summary
    # It still says the CONFIGURED policy ran nothing, and that the floor is all that covered it.
    assert "ran no detectors" in summary
    assert "handled the 1 span(s)" in summary

    # And the rendered page an auditor actually reads carries the same truth.
    page = render_chunk(RUN, inspection)
    assert "the control was OFF" not in page
    assert "floor:iban" in page


def test_a_run_without_redaction_reports_that_rather_than_inventing_a_disposition(
    storage: Storage, storage_uri: str
) -> None:
    from reports import healthy_report  # noqa: PLC0415 - test helper

    write_run(storage, storage_uri, RUN, healthy_report(RUN))
    write_artifacts(storage, storage_uri, RUN, redact=False)
    app = ConsoleApp.over_storage(Storage.for_uri(storage_uri), storage_uri)
    page = _page(app, _chunk_url(app, CHUNK_ID))
    assert "No Redaction Stage produced a masked variant" in page
    assert "Redaction ran and handled" not in page


def test_the_most_downstream_version_of_a_chunk_is_the_one_shown(app: ConsoleApp) -> None:
    """Redaction rewrites the chunk Chunk produced; the console must not show the older one."""
    page = _page(app, _chunk_url(app, CHUNK_ID))
    assert "Corpus text (PII-handled)" in page
    assert "Redaction ran and handled 1 PII span(s)" in page


def test_a_chunk_id_that_does_not_exist_is_a_404_not_a_file_read(app: ConsoleApp) -> None:
    for hostile in (
        f"/run/{RUN}/chunk?id=../../../../etc/passwd",
        f"/run/{RUN}/chunk?id=",
        f"/run/{RUN}/chunk",
        f"/run/{RUN}/chunk?id=nope",
    ):
        response = app.respond("GET", hostile)
        assert response.status == 404, hostile
        assert b"root:" not in response.body


# -- AC 3: the KG browser ----------------------------------------------------


def test_graph_page_lists_navigable_entities(app: ConsoleApp) -> None:
    page = _page(app, f"/run/{RUN}/graph")
    assert "Ada Lovelace" in page
    assert "Acme Corp" in page
    assert "Q12345" in page  # the external KB link
    assert f'href="/run/{RUN}/entity?id={NODE_PERSON}"' in page
    assert app.respond("GET", f"/run/{RUN}/entity?id={NODE_ORG}").status == 200


def test_every_edge_displays_its_evidence(app: ConsoleApp) -> None:
    page = _page(app, f"/run/{RUN}/graph")
    assert "works_for" in page
    # The Evidence itself: confidence, justification, source-document reach.
    assert "Evidence — confidence 0.87" in page
    assert "works_for holds between Ada Lovelace and Acme Corp" in page
    assert "1 source document(s)" in page
    # And the source mentions it rests on, located in their document.
    assert MENTION_ID in page
    assert "Acme Corp" in page
    assert "62–71" in page  # the mention's character offsets
    assert "0.91" in page  # the mention's own confidence


def test_an_evidence_mention_the_run_no_longer_holds_is_shown_not_dropped(
    app: ConsoleApp,
) -> None:
    page = _page(app, f"/run/{RUN}/graph")
    assert DANGLING_MENTION_ID in page
    assert "mention not in this run" in page


def test_every_edge_view_carries_evidence(browsable: str) -> None:
    """Structural, not incidental: Evidence is required on GraphEdge, so no view lacks it."""
    artifacts = ConsoleApp.over_storage(
        Storage.for_uri(browsable), browsable
    )._index.artifacts(RUN)
    views = browse.edge_views(artifacts)
    assert views
    for view in views:
        assert view.mentions
        assert 0.0 <= view.evidence_confidence <= 1.0


def test_the_entity_page_shows_the_edges_touching_that_entity_with_evidence(
    app: ConsoleApp,
) -> None:
    page = _page(app, f"/run/{RUN}/entity?id={NODE_PERSON}")
    assert "Ada Lovelace" in page
    assert "Relations (1)" in page
    assert "Evidence — confidence 0.87" in page


# -- AC 4: still read-only, still offline ------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/run/run-0001/quarantine",
        "/run/run-0001/chunks",
        "/run/run-0001/graph",
        f"/run/run-0001/entity?id={NODE_PERSON}",
    ],
)
def test_every_new_page_is_fully_offline(app: ConsoleApp, path: str) -> None:
    assert external_references(_page(app, path)) == []


def test_browsing_every_new_view_leaves_storage_byte_identical(
    browsable: str, tmp_path: Path
) -> None:
    """The load-bearing read-only proof, extended to the deepened views (#188 AC 4)."""
    root = tmp_path / "corpus"
    before = _tree_digest(root)
    app = ConsoleApp.over_storage(Storage.for_uri(browsable), browsable)
    paths = [
        f"/run/{RUN}",
        f"/run/{RUN}/quarantine",
        f"/run/{RUN}/chunks",
        f"/run/{RUN}/graph",
        f"/run/{RUN}/entity?id={NODE_PERSON}",
        f"/run/{RUN}/entity?id={NODE_ORG}",
        _chunk_url(app, CHUNK_ID),
        _chunk_url(app, SENSITIVE_CHUNK_ID),
    ]
    for path in paths:
        assert app.respond("GET", path).status == 200, path
    for path in paths:
        assert app.respond("POST", path).status == 405, path
    assert _tree_digest(root) == before


def test_a_new_view_reflects_a_checkpoint_that_appeared_since_the_last_request(
    browsable: str, storage: Storage
) -> None:
    """No index, no cache: the console re-reads the files on every request."""
    app = ConsoleApp.over_storage(Storage.for_uri(browsable), browsable)
    run_dir = f"{browsable.rstrip('/')}/_latence/runs/{RUN}"
    assert b"never-seen-before.zip" not in app.respond("GET", f"/run/{RUN}/quarantine").body
    existing = storage.read_text(f"{run_dir}/quarantine/intake_screen.jsonl")
    storage.write_atomic(
        f"{run_dir}/quarantine/late_screen.jsonl",
        existing.replace("archive.zip", "never-seen-before.zip").encode("utf-8"),
    )
    assert b"never-seen-before.zip" in app.respond("GET", f"/run/{RUN}/quarantine").body


def test_a_run_with_no_artifacts_renders_the_views_empty_rather_than_failing(
    populated: str,
) -> None:
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    for suffix in ("/quarantine", "/chunks", "/graph"):
        response = app.respond("GET", f"/run/run-0002{suffix}")
        assert response.status == 200, suffix


def test_an_unknown_view_of_a_real_run_is_a_404(app: ConsoleApp) -> None:
    assert app.respond("GET", f"/run/{RUN}/nonsense").status == 404


def test_the_run_page_links_to_all_four_views(app: ConsoleApp) -> None:
    page = _page(app, f"/run/{RUN}")
    for suffix in ("/quarantine", "/chunks", "/graph"):
        assert f'href="/run/{RUN}{suffix}"' in page

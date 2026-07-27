"""EndpointEntityExtractor: the NER performance path over an OpenAI-compatible endpoint (S12).

The real ``openai`` client is intentionally NOT installed (ADR-0007/0016: offline, deterministic).
A fake ``openai`` module returns canned JSON so the Provider's span validation, offset+page
Provenance (the S4 round-trip), zero-shot config, retry accounting, lease acquisition, and
hallucination-dropping are exercised offline.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSlice,
    PageSpan,
    Provenance,
)
from latence_ner_endpoint import EndpointEntityExtractor
from latence_ner_endpoint.provider import _POOL


def _chunk(
    text: str,
    *,
    char_start: int = 0,
    char_end: int | None = None,
    page_map: PageMap | None = None,
    offset_map: OffsetMap | None = None,
) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:d1",
        char_start=char_start,
        char_end=char_start + len(text) if char_end is None else char_end,
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id="d1#chunk-0",
        document_record_id="d1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
        # v19: the chunk carries its OWN page spans, cut from the parent document's map.
        # Absent a map the fixture is a one-page document, so the slice is that page.
        page_slice=(
            PageSlice.for_span(
                page_map, char_start, char_start + len(text) if char_end is None else char_end
            )
            if page_map is not None
            else PageSlice(
                pages=[
                    PageSpan(
                        page_number=1,
                        char_start=char_start,
                        char_end=char_start + len(text) if char_end is None else char_end,
                    )
                ],
                total_chars=char_start + len(text) if char_end is None else char_end,
            )
        ),
        offset_map=offset_map,
    )


class _FakeCompletions:
    reply: str = "{}"
    error: Exception | None = None
    fail_times: int = 0
    calls: int = 0

    def create(self, **kwargs: Any) -> Any:
        type(self).calls += 1
        err = type(self).error
        if err is not None and type(self).calls <= type(self).fail_times:
            raise err
        msg = types.SimpleNamespace(content=type(self).reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = _FakeChat()


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _FakeCompletions.reply = "{}"
    _FakeCompletions.error = None
    _FakeCompletions.fail_times = 0
    _FakeCompletions.calls = 0
    # A fresh pool each test so lease ref-counts don't leak across tests.
    _POOL._leases.clear()  # noqa: SLF001


def test_requires_labels_config() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        EndpointEntityExtractor({})
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        EndpointEntityExtractor({"labels": []})


def test_extracts_and_resolves_offsets(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps(
        {"entities": [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]}
    )
    chunk = _chunk("Alice works at Acme.", char_start=10)
    out = list(EndpointEntityExtractor({"labels": ["person"]}).extract([chunk]))
    assert len(out) == 1
    m = out[0]
    assert isinstance(m, EntityMention)
    assert m.text == "Alice"
    assert m.label == "person"
    # Chunk-local [0,5) shifted by the chunk's char_start=10 into document coordinates.
    assert m.provenance.char_start == 10
    assert m.provenance.char_end == 15
    assert m.provenance.page_start == 1
    assert m.classification.sensitivity == "internal"


def _two_chunk_document() -> tuple[ChunkRecord, ChunkRecord]:
    """A v19 document stream: each chunk carries the page spans it overlaps.

    Page 1 is ``[0,16)``, page 2 is ``[16,33)``. Chunk 0 lies wholly on page 1; chunk 1
    STRADDLES the boundary (doc ``[13,33)``, inherited page range ``1..2``) so its own slice
    holds BOTH spans. A chunk-local ``[3,6)`` span is ``"ce "`` on page 1 in chunk 0 and
    ``"Bob"`` at doc ``[16,19)`` — wholly on page 2 — in chunk 1.
    """
    assembly = PageMap.from_page_texts(["Alice is here.", "Bob is over here."])

    def _mk(
        *, index: int, char_start: int, char_end: int, pages: tuple[int, int]
    ) -> ChunkRecord:
        return ChunkRecord(
            record_id=f"d1#chunk-{index}",
            document_record_id="d1",
            index=index,
            provenance=Provenance(
                source_uri="file:///c/a.txt",
                file_name="a.txt",
                file_type="txt",
                file_size=1000,
                document_id="sha256:d1",
                char_start=char_start,
                char_end=char_end,
                page_start=pages[0],
                page_end=pages[1],
            ),
            classification=Classification(language="en", sensitivity="internal"),
            content=assembly.text[char_start:char_end],
            token_count=4,
            page_slice=PageSlice.for_span(assembly.page_map, char_start, char_end),
            offset_map=None,
        )

    first = _mk(index=0, char_start=0, char_end=14, pages=(1, 1))
    second = _mk(index=1, char_start=13, char_end=33, pages=(1, 2))
    return first, second


def test_mention_page_resolves_on_a_page_straddling_later_chunk(
    fake_openai: None,
) -> None:
    """A later, page-straddling chunk must resolve its mention's OWN page (ADR-0021/0031).

    Inheriting the chunk's page range degrades "Bob" to ``(1, 2)`` — a plausible-looking
    wrong page. The second half proves the resolution survives ``min_chars`` dropping the
    document's first chunk, which under v18 took the whole document's map with it.
    """
    first, second = _two_chunk_document()
    _FakeCompletions.reply = json.dumps(
        {"entities": [{"start": 3, "end": 6, "label": "person", "confidence": 0.9}]}
    )
    out = list(EndpointEntityExtractor({"labels": ["person"]}).extract([first, second]))
    by_chunk = {m.chunk_record_id: m for m in out}
    assert set(by_chunk) == {"d1#chunk-0", "d1#chunk-1"}
    assert by_chunk["d1#chunk-1"].text == "Bob"
    # The whole point: page 2, NOT the chunk's inherited (1, 2).
    pages = by_chunk["d1#chunk-1"].provenance
    assert (pages.page_start, pages.page_end) == (2, 2)

    # min_chars drops the document's FIRST chunk — under v18 that lost its map.
    assert len(first.content.strip()) == 14
    filtered = list(
        EndpointEntityExtractor({"labels": ["person"], "min_chars": 15}).extract([first, second])
    )
    assert [m.chunk_record_id for m in filtered] == ["d1#chunk-1"]
    assert (filtered[0].provenance.page_start, filtered[0].provenance.page_end) == (2, 2)


def test_mention_offset_resolves_through_offset_map_past_stripped_markup(
    fake_openai: None,
) -> None:
    # Original "**Note:** Alice joined Globex." strips to "Note: Alice joined Globex.":
    # the stripped ``**`` before the prose shifts every later mention right in the
    # original, so char_start+local is a lower bound. The carried offset_map corrects it
    # — the learned/cloud-NER blind spot issue #48 names (this path is a one-config swap
    # for the in-core gazetteer, so it must resolve identically).
    original = "**Note:** Alice joined Globex."
    content = "Note: Alice joined Globex."
    offset_map = OffsetMap(
        segments=[
            OffsetSegment(local_start=0, doc_start=2),
            OffsetSegment(local_start=5, doc_start=9),
        ],
        length=len(content),
    )
    _FakeCompletions.reply = json.dumps(
        {
            "entities": [
                {"start": 6, "end": 11, "label": "person", "confidence": 0.9},
                {"start": 19, "end": 25, "label": "organization", "confidence": 0.9},
            ]
        }
    )
    chunk = _chunk(content, char_start=2, char_end=len(original), offset_map=offset_map)
    out = list(
        EndpointEntityExtractor({"labels": ["person", "organization"]}).extract([chunk])
    )
    by_text = {m.text: m for m in out}
    assert set(by_text) == {"Alice", "Globex"}
    for m in out:
        # TRUE original span (not the char_start+local lower bound) — fails on the shift.
        assert original[m.provenance.char_start : m.provenance.char_end] == m.text
    assert by_text["Alice"].provenance.char_start == 10  # not 2 + 6 = 8
    assert by_text["Globex"].provenance.char_start == 23  # not 2 + 19 = 21


def test_hallucinated_spans_dropped(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps(
        {
            "entities": [
                {"start": 0, "end": 999, "label": "person"},  # out of range
                {"start": 5, "end": 2, "label": "person"},  # inverted
                {"start": 0, "end": 5, "label": "bogus"},  # unknown label
                {"start": 0, "end": 5, "label": "person", "confidence": 0.1},  # below threshold
                {"start": 0, "end": 5, "label": "person", "confidence": 0.9},  # valid
            ]
        }
    )
    chunk = _chunk("Alice works at Acme.")
    out = list(EndpointEntityExtractor({"labels": ["person"], "threshold": 0.5}).extract([chunk]))
    assert [m.confidence for m in out] == [0.9]
    assert out[0].index == 0  # re-indexed after drops


def test_empty_chunk_skips_model(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps({"entities": [{"start": 0, "end": 5, "label": "person"}]})
    chunk = _chunk("   ")
    out = list(EndpointEntityExtractor({"labels": ["person"], "min_chars": 2}).extract([chunk]))
    assert out == []
    assert _FakeCompletions.calls == 0


def test_transient_failure_retried_and_counted(fake_openai: None) -> None:
    _FakeCompletions.error = _StatusError("429", status_code=429)
    _FakeCompletions.fail_times = 1
    _FakeCompletions.reply = json.dumps(
        {"entities": [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]}
    )
    provider = EndpointEntityExtractor({"labels": ["person"], "retry_attempts": 3})
    out = list(provider.extract([_chunk("Alice works at Acme.")]))
    assert len(out) == 1
    assert provider.endpoint_retries == 1


def test_terminal_failure_surfaces(fake_openai: None) -> None:
    from latence_core.endpoint import EndpointError

    _FakeCompletions.error = _StatusError("bad request", status_code=400)
    _FakeCompletions.fail_times = 99
    provider = EndpointEntityExtractor({"labels": ["person"], "retry_attempts": 3})
    with pytest.raises(EndpointError, match="bad request"):
        list(provider.extract([_chunk("Alice works at Acme.")]))


def test_lease_released_after_stage(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps({"entities": []})
    provider = EndpointEntityExtractor({"labels": ["person"], "base_url": "http://vllm/v1"})
    list(provider.extract([_chunk("Alice works at Acme.")]))
    # The ref-counted lease is back to 0 (activated once, released once) after the Stage's stream.
    assert provider._lease.refcount == 0  # noqa: SLF001
    assert provider._lease.live is False  # noqa: SLF001


def test_missing_openai_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    _POOL._leases.clear()  # noqa: SLF001
    provider = EndpointEntityExtractor({"labels": ["person"]})
    with pytest.raises(ImportError, match="pip install latence-ner-endpoint"):
        list(provider.extract([_chunk("Alice works at Acme.")]))

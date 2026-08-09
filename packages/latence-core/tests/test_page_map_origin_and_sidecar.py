"""`PageMapOrigin`, the page-map sidecar reader, and the Source that finds it (ADR-0060).

Three things are pinned here, all of them about the same defect: a page map that was ASSUMED
used to be indistinguishable from one that was MEASURED, so an entire OCR'd corpus cited
``page 1`` with no way to tell. The origin makes the assumption a value; the sidecar reader
refuses a map it cannot vouch for; the Source pairs the file with its document without ever
turning the sidecar into a document of its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from latence_core.contracts import (
    PAGE_MAP_SIDECAR_SUFFIX,
    PageMap,
    PageMapOrigin,
    PageSlice,
)
from latence_core.errors import ContractError, PageMapSidecarError
from latence_core.pagemap import PageOffsetIndex
from latence_core.stages.source import LocalFolderSource
from latence_core.storage import Storage

PAGES = ["alpha page", "bravo page", "charlie page"]


def _sidecar_bytes(page_map: PageMap, *, total_chars: int | None = None) -> bytes:
    return json.dumps(
        {
            "pages": [span.model_dump() for span in page_map.pages],
            "total_chars": page_map.total_chars if total_chars is None else total_chars,
        }
    ).encode("utf-8")


# -- origin -----------------------------------------------------------------------------------


def test_origin_defaults_to_parser_so_v20_records_still_validate() -> None:
    revived = PageMap.model_validate_json(
        '{"pages":[{"page_number":1,"char_start":0,"char_end":4}],"total_chars":4}'
    )
    assert revived.origin is PageMapOrigin.PARSER


def test_an_assumed_map_and_a_real_one_page_map_are_now_different_values() -> None:
    """The heart of the defect: these used to be the same object."""
    real = PageMap.from_page_texts(["only page"]).page_map
    assumed = PageMap.from_page_texts(
        ["only page"], origin=PageMapOrigin.ASSUMED_SINGLE_PAGE
    ).page_map

    assert real.pages == assumed.pages  # identical boundaries...
    assert real != assumed  # ...distinguishable claims
    assert assumed.origin is PageMapOrigin.ASSUMED_SINGLE_PAGE


def test_a_slice_inherits_its_parent_maps_origin() -> None:
    assembled = PageMap.from_page_texts(
        list(PAGES), origin=PageMapOrigin.ASSUMED_SINGLE_PAGE
    )
    sliced = PageSlice.for_span(assembled.page_map, 0, 5)
    assert sliced.origin is PageMapOrigin.ASSUMED_SINGLE_PAGE

    measured = PageSlice.for_span(PageMap.from_page_texts(list(PAGES)).page_map, 0, 5)
    assert measured.origin is PageMapOrigin.PARSER


def test_origin_does_not_change_offset_resolution() -> None:
    """Honesty is additive: the same offsets resolve to the same pages as before."""
    assembled = PageMap.from_page_texts(list(PAGES))
    labelled = PageMap.from_page_texts(list(PAGES), origin=PageMapOrigin.SIDECAR)
    plain = PageOffsetIndex(assembled.page_map)
    tagged = PageOffsetIndex(labelled.page_map)
    for offset in range(len(assembled.text) + 2):
        assert plain.page_for_offset(offset) == tagged.page_for_offset(offset)


# -- the sidecar reader -----------------------------------------------------------------------


def test_sidecar_round_trips_a_multi_page_map_and_is_stamped_sidecar() -> None:
    assembled = PageMap.from_page_texts(list(PAGES))
    revived = PageMap.from_sidecar_json(
        _sidecar_bytes(assembled.page_map), text_length=len(assembled.text)
    )
    assert revived.pages == assembled.page_map.pages
    assert revived.origin is PageMapOrigin.SIDECAR


def test_a_sidecar_cannot_vouch_for_its_own_origin() -> None:
    assembled = PageMap.from_page_texts(list(PAGES))
    payload = json.loads(_sidecar_bytes(assembled.page_map))
    payload["origin"] = "parser"
    revived = PageMap.from_sidecar_json(json.dumps(payload), text_length=len(assembled.text))
    assert revived.origin is PageMapOrigin.SIDECAR


def test_a_sidecar_that_disagrees_with_the_document_length_is_refused() -> None:
    assembled = PageMap.from_page_texts(list(PAGES))
    with pytest.raises(PageMapSidecarError, match="stale"):
        PageMap.from_sidecar_json(
            _sidecar_bytes(assembled.page_map), text_length=len(assembled.text) + 1
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"{not json", "not valid JSON"),
        (b"[1, 2, 3]", "must be a JSON object"),
        (b"\xff\xfe not utf8", "not valid UTF-8"),
        (b'{"pages":[],"total_chars":0}', "not a valid PageMap"),
        (
            b'{"pages":[{"page_number":1,"char_start":0,"char_end":4},'
            b'{"page_number":2,"char_start":9,"char_end":12}],"total_chars":12}',
            "not a valid PageMap",  # non-contiguous spans
        ),
        (
            b'{"schema_version":9999,"pages":[{"page_number":1,"char_start":0,"char_end":4}],'
            b'"total_chars":4}',
            "newer than this build",
        ),
    ],
)
def test_every_malformation_raises_the_one_typed_error(payload: bytes, match: str) -> None:
    with pytest.raises(PageMapSidecarError, match=match):
        PageMap.from_sidecar_json(payload, text_length=4)


def test_the_sidecar_error_is_a_contract_error_so_one_except_clause_covers_it() -> None:
    assert issubclass(PageMapSidecarError, ContractError)


# -- Source-side discovery ---------------------------------------------------------------------


def _write_corpus(tmp_path: Path) -> tuple[str, PageMap]:
    assembled = PageMap.from_page_texts(list(PAGES))
    (tmp_path / "doc.md").write_text(assembled.text, encoding="utf-8")
    (tmp_path / ("doc.md" + PAGE_MAP_SIDECAR_SUFFIX)).write_bytes(
        _sidecar_bytes(assembled.page_map)
    )
    (tmp_path / "plain.md").write_text("no pages here", encoding="utf-8")
    return f"file://{tmp_path}", assembled.page_map


def test_source_pairs_a_document_with_its_sidecar_and_never_emits_the_sidecar(
    tmp_path: Path,
) -> None:
    uri, page_map = _write_corpus(tmp_path)
    inputs = list(LocalFolderSource({"path": uri}).produce(Storage.for_uri(uri)))

    by_name = {raw.provenance.file_name: raw for raw in inputs}
    # The sidecar is a property of a document, not a document.
    assert set(by_name) == {"doc.md", "plain.md"}
    assert by_name["doc.md"].page_map_sidecar is not None
    assert (
        PageMap.from_sidecar_json(
            by_name["doc.md"].page_map_sidecar or b"",
            text_length=page_map.total_chars,
        ).pages
        == page_map.pages
    )
    # A document with no sidecar simply has none — absence is not an error at the Source.
    assert by_name["plain.md"].page_map_sidecar is None


def test_widening_extensions_to_json_still_does_not_ingest_sidecars(tmp_path: Path) -> None:
    """The skip runs BEFORE the extension filter, so page boundaries never become a document."""
    uri, _ = _write_corpus(tmp_path)
    source = LocalFolderSource({"path": uri, "extensions": ["md", "json"]})
    names = {raw.provenance.file_name for raw in source.produce(Storage.for_uri(uri))}
    assert names == {"doc.md", "plain.md"}


def test_sidecars_are_keyed_on_the_full_file_name_not_the_stem(tmp_path: Path) -> None:
    """``report.md`` and ``report.txt`` in one folder must not contend for one sidecar."""
    assembled = PageMap.from_page_texts(list(PAGES))
    (tmp_path / "report.md").write_text(assembled.text, encoding="utf-8")
    (tmp_path / ("report.md" + PAGE_MAP_SIDECAR_SUFFIX)).write_bytes(
        _sidecar_bytes(assembled.page_map)
    )
    (tmp_path / "report.txt").write_text("a different, page-less file", encoding="utf-8")

    uri = f"file://{tmp_path}"
    inputs = {
        raw.provenance.file_name: raw
        for raw in LocalFolderSource({"path": uri}).produce(Storage.for_uri(uri))
    }
    assert inputs["report.md"].page_map_sidecar is not None
    assert inputs["report.txt"].page_map_sidecar is None

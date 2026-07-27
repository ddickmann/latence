"""Page resolution must be ORDER-independent and DROP-independent (the C1 sidecar).

The defect this pins is a *shape* defect, not a call-site defect. Until the run-scoped page-map
sidecar, a document's :class:`~latence_core.contracts.PageMap` — DOCUMENT-scoped state — rode ONE
record of the chunk stream (the document's first chunk, schema v18) and the resolver adopted it
from whichever record it happened to see first. Two consequences follow directly from that shape,
and both silently produce a *plausible-looking wrong page* rather than an error:

* **Order dependence.** A Provider that buffers chunks and resolves them in any order other than
  stream order (every batching extractor does: ``latence_extract_gliner2`` clamps ``batch_size``
  to 64, the PII Providers clamp it not at all) asks for a page for chunk *N* before the resolver
  has seen chunk 0. Resolution degrades to the chunk's inherited page range, so a mention on a
  later page of a multi-page chunk cites the chunk's FIRST page.
* **Drop dependence.** Any Stage that REMOVES chunks can remove the one carrying the map, taking
  the whole document's page resolution with it. The Runner grew a ``rehome_page_maps()`` patch for
  the Content-Screening path — but a chunk→chunk Capability that filters (schema induction,
  redaction, any third-party chunk transform) was never covered by it.

Both are gone by construction once the map is a run-scoped, disk-backed sidecar the resolver reads
lazily by ``document_record_id``: there is no first-record-wins state left to depend on.

**Anti-tautology.** Nothing here asserts against the resolver in isolation. Every case runs a real
Pipeline through the real :class:`~latence_core.runner.LocalRunner` and then checks each emitted
mention's recorded ``page_start``/``page_end`` against the page the *parent document's own page
map* (read back from the Parse checkpoint) resolves that mention's character offset to. That is
the ADR-0021/0031 provenance guarantee itself, measured end to end. :func:`_assert_spans_pages`
additionally refuses to let a case pass vacuously: the fixture is asserted to actually contain
mentions whose true page differs from their chunk's inherited ``page_start``, which is the only
situation in which a degraded resolution is distinguishable from an exact one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from latence_core.capability import ProviderRegistry, ScreeningOutcome
from latence_core.contracts import (
    ChunkRecord,
    Disposition,
    DocumentRecord,
    EntityMention,
    PageMap,
    QuarantineRecord,
    ScreeningCheckpoint,
    ScreeningFinding,
    ScreeningVerdict,
)
from latence_core.offsetmap import OffsetIndex
from latence_core.page_index import PageIndexResolver
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

#: The marker the probe extractor recognises. One per page, so every page of the fixture
#: document carries a mention whose own page is checkable.
_MARKER = "TARGETWORD"

#: Words of filler per page. Small pages + a 64-token chunk budget make every chunk span
#: several pages, which is the only shape in which "inherit the chunk's page" is wrong.
_PAGE_WORDS = 12

#: Pages in the fixture document.
_PAGES = 12


def _page_text(page_number: int) -> str:
    filler = " ".join(f"w{page_number}x{i}" for i in range(_PAGE_WORDS))
    return f"{filler} {_MARKER} page {page_number} end"


def _corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    # Form feed separates source pages; the fake Parser below turns them into a real PageMap.
    body = "\f".join(_page_text(n) for n in range(1, _PAGES + 1))
    (docs / "contract.txt").write_text(body, encoding="utf-8")
    return docs


class _MultiPageParser:
    """Parser Provider: split the decoded text on form feeds into REAL source pages.

    The in-core plain-text Parser assembles every document as a single page, which cannot
    express the defect (a one-page document's inherited page range is always right). This
    is the smallest honest multi-page Parser: it uses the same
    :meth:`~latence_core.contracts.PageMap.from_page_texts` constructor every real Parser
    uses, so the page map is exact by construction.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def parse(self, inputs: Iterable[Any]) -> Iterator[DocumentRecord]:
        for raw in inputs:
            text = raw.content if isinstance(raw.content, str) else raw.content.decode("utf-8")
            assembly = PageMap.from_page_texts(text.split("\f"), separator="\n\n")
            provenance = raw.provenance.model_copy(
                update={
                    "char_start": 0,
                    "char_end": assembly.page_map.total_chars,
                    "page_start": 1,
                    "page_end": len(assembly.page_map.pages),
                }
            )
            yield DocumentRecord(
                record_id=raw.provenance.document_id,
                provenance=provenance,
                classification=raw.classification,
                content=assembly.text,
                media_type="text/markdown",
                page_map=assembly.page_map,
                disposition=Disposition.PARSED,
            )


class _MarkerExtractor:
    """EntityExtractor Provider: one mention per marker, page resolved through the seam.

    Deliberately shaped like a **batching** Provider: it drains the whole chunk stream before
    resolving anything, then walks the buffer in ``reverse`` order when configured to. That is
    the only thing that varies between the order-dependent and order-independent cases — the
    resolution call itself is the ordinary
    :meth:`~latence_core.page_index.PageIndexResolver.pages_for_span`, identical to every
    shipped extractor's.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._reverse = bool((config or {}).get("reverse", False))

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        resolver = PageIndexResolver()
        buffered = list(chunks)
        order = list(reversed(buffered)) if self._reverse else buffered
        for chunk in order:
            yield from self._one(chunk, resolver)

    @staticmethod
    def _one(chunk: ChunkRecord, resolver: PageIndexResolver) -> Iterator[EntityMention]:
        offsets = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        base = chunk.provenance.char_start or 0
        cursor = 0
        i = 0
        while True:
            local = chunk.content.find(_MARKER, cursor)
            if local < 0:
                return
            cursor = local + len(_MARKER)
            local_end = local + len(_MARKER)
            if offsets is not None:
                doc_start, doc_end = offsets.resolve_span(local, local_end)
            else:  # pragma: no cover - the reference Chunker always emits an offset map
                doc_start, doc_end = base + local, base + local_end
            pages = resolver.pages_for_span(chunk, doc_start, doc_end)
            update: dict[str, int] = {"char_start": doc_start, "char_end": doc_end}
            if pages is not None:
                update["page_start"], update["page_end"] = pages
            yield EntityMention(
                record_id=f"{chunk.record_id}#marker-{i}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=i,
                label="MARKER",
                text=_MARKER,
                confidence=1.0,
                provenance=chunk.provenance.model_copy(update=update),
                classification=chunk.classification,
            )
            i += 1


class _DropFirstChunkScreener:
    """ContentScreener Provider: quarantine every document's chunk 0, pass the rest.

    The literal case ``rehome_page_maps()`` was written for — a document's opening chunk (title
    page, header block, a pasted instruction) is a likely place for an injection marker, and
    under the carried-map shape quarantining it destroyed page resolution for the whole document.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def screen_content(self, chunks: Iterable[ChunkRecord]) -> ScreeningOutcome[ChunkRecord]:
        outcome: ScreeningOutcome[ChunkRecord] = ScreeningOutcome()
        for chunk in chunks:
            if chunk.index == 0:
                outcome.quarantined.append(
                    QuarantineRecord(
                        record_id=f"{chunk.record_id}#quarantine",
                        provenance=chunk.provenance,
                        classification=chunk.classification,
                        checkpoint=ScreeningCheckpoint.CONTENT,
                        detector="test.drop_first_chunk",
                        category="test",
                        reason="fixture: the document's opening chunk is withheld",
                        quarantined_bytes=len(chunk.content.encode("utf-8")),
                    )
                )
                outcome.findings.append(
                    ScreeningFinding(
                        checkpoint=ScreeningCheckpoint.CONTENT,
                        verdict=ScreeningVerdict.QUARANTINE,
                        document_id=chunk.provenance.document_id,
                        detector="test.drop_first_chunk",
                        category="test",
                        reason="fixture: the document's opening chunk is withheld",
                    )
                )
                continue
            outcome.passed.append(chunk)
        return outcome


class _DropFirstChunkInducer:
    """LabelInducer Provider: a chunk→chunk transform that FILTERS out chunk 0.

    Content Screening is not the only Stage that can remove a chunk — every chunk→chunk
    Capability can (schema induction, redaction, any third-party chunk transform), and the
    Runner's ``rehome_page_maps()`` patch covered only the screening path. This is the hole that
    patch never closed, which is why drop-independence has to be a property of the mechanism
    rather than of one Stage's wiring.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def induce(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        for chunk in chunks:
            if chunk.index != 0:
                yield chunk


def _registry() -> ProviderRegistry:
    """A real registry with the fixture Providers injected by name (no package install)."""
    registry = ProviderRegistry()
    discovered = dict(registry._discover())  # noqa: SLF001 - test seeds the discovery cache

    fakes: dict[str, type] = {
        "parse.multipage_fixture": _MultiPageParser,
        "entity.marker_fixture": _MarkerExtractor,
        "screen.drop_first_chunk_fixture": _DropFirstChunkScreener,
        "induce.drop_first_chunk_fixture": _DropFirstChunkInducer,
    }
    for name, cls in fakes.items():

        class _FakeEntryPoint:
            def __init__(self, target: type) -> None:
                self._target = target

            def load(self) -> type:
                return self._target

        discovered[name] = _FakeEntryPoint(cls)  # type: ignore[assignment]
        registry._loaded[name] = cls  # noqa: SLF001
    registry._entry_points = discovered  # noqa: SLF001
    return registry


_CHUNK_STAGE = """
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config:
      max_tokens: 64
      overlap_tokens: 0
      min_tokens: 4
"""


def _pipeline(store: str, docs: Path, *, middle: str, entity_parent: str, reverse: bool) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: page-map-independence
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parse.multipage_fixture
    depends_on: [source]
{_CHUNK_STAGE}{middle}
  - name: entities
    capability: entity_extraction
    provider: entity.marker_fixture
    depends_on: [{entity_parent}]
    config:
      reverse: {"true" if reverse else "false"}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


_SCREEN_STAGE = """
  - name: screen
    capability: content_screening
    provider: screen.drop_first_chunk_fixture
    depends_on: [chunk]
"""

_INDUCE_STAGE = """
  - name: induce
    capability: schema_induction
    provider: induce.drop_first_chunk_fixture
    depends_on: [chunk]
"""


def _run(
    tmp_path: Path, run_id: str, *, middle: str = "", entity_parent: str = "chunk", reverse: bool
) -> tuple[list[EntityMention], dict[str, DocumentRecord], list[ChunkRecord]]:
    docs = _corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs, middle=middle, entity_parent=entity_parent, reverse=reverse)
    LocalRunner(_registry()).run(pipeline, run_id=run_id)
    storage = Storage.for_uri(store)
    base = f"{store}/_latence/runs/{run_id}"

    rows = [
        json.loads(line)
        for line in storage.read_text(f"{base}/export/records.jsonl").splitlines()
        if line.strip()
    ]
    mentions = [EntityMention.model_validate(r) for r in rows]
    documents = {
        d.record_id: d
        for d in (
            DocumentRecord.model_validate_json(line)
            for line in storage.read_text(f"{base}/checkpoints/parse.jsonl").splitlines()
            if line.strip()
        )
    }
    chunks = [
        ChunkRecord.model_validate_json(line)
        for line in storage.read_text(f"{base}/checkpoints/chunk.jsonl").splitlines()
        if line.strip()
    ]
    return mentions, documents, chunks


def _assert_spans_pages(
    mentions: list[EntityMention],
    chunks: list[ChunkRecord],
    documents: dict[str, DocumentRecord],
) -> None:
    """Refuse to let a case pass vacuously.

    A degraded resolution inherits the chunk's ``page_start``; that is only *distinguishable*
    from an exact one for a mention whose TRUE page is not its chunk's first page. The check is
    made against ground truth (the parent document's own page map), never against what the
    resolver produced — so it says "this fixture can tell the difference", independently of
    whether the run being examined got the answer right.
    """
    by_id = {c.record_id: c for c in chunks}
    assert mentions, "fixture produced no mentions"
    discriminating = 0
    for mention in mentions:
        chunk = by_id.get(str(mention.chunk_record_id))
        parent = documents[mention.document_record_id]
        assert parent.page_map is not None
        assert mention.provenance.char_start is not None
        true_page = PageOffsetIndex(parent.page_map).page_for_offset(mention.provenance.char_start)
        if chunk is not None and true_page != chunk.provenance.page_start:
            discriminating += 1
    assert discriminating, (
        "fixture is vacuous: no mention's TRUE page differs from its chunk's inherited "
        "page_start, so a degraded resolution would be indistinguishable from an exact one"
    )


def _assert_pages_exact(mentions: list[EntityMention], documents: dict[str, DocumentRecord]) -> None:
    """Every mention's recorded page is the page its offset REALLY falls on (ADR-0021/0031)."""
    for mention in mentions:
        parent = documents[mention.document_record_id]
        assert parent.page_map is not None
        index = PageOffsetIndex(parent.page_map)
        assert mention.provenance.char_start is not None
        assert mention.provenance.char_end is not None
        true_start = index.page_for_offset(mention.provenance.char_start)
        true_end = index.page_for_offset(max(mention.provenance.char_end - 1, mention.provenance.char_start))
        assert mention.provenance.page_start == true_start, (
            f"{mention.record_id}: recorded page_start {mention.provenance.page_start} "
            f"but offset {mention.provenance.char_start} is really on page {true_start}"
        )
        assert mention.provenance.page_end == true_end
        # The surface really is at that offset in the parent — offsets and pages agree.
        assert (
            parent.content[mention.provenance.char_start : mention.provenance.char_end] == _MARKER
        )


def test_page_resolution_is_order_and_drop_independent(tmp_path: Path) -> None:
    """The headline: chunk 0 is QUARANTINED and the survivors are resolved in REVERSE order.

    Under the carried-map shape this is doubly unresolvable — the record that carried the map was
    removed from the stream, and the first record the resolver is asked about is the document's
    LAST chunk. Under the run-scoped sidecar neither fact is reachable from the mechanism: the map
    is fetched by ``document_record_id`` on demand.
    """
    mentions, documents, chunks = _run(
        tmp_path, "both", middle=_SCREEN_STAGE, entity_parent="screen", reverse=True
    )
    _assert_spans_pages(mentions, chunks, documents)
    _assert_pages_exact(mentions, documents)


def test_page_resolution_is_order_independent(tmp_path: Path) -> None:
    """A later chunk is resolved BEFORE any earlier chunk of its document. Nothing is dropped."""
    mentions, documents, chunks = _run(tmp_path, "order", reverse=True)
    _assert_spans_pages(mentions, chunks, documents)
    _assert_pages_exact(mentions, documents)


def test_page_resolution_survives_a_quarantined_first_chunk(tmp_path: Path) -> None:
    """Content Screening removes the document's first chunk; every survivor still resolves."""
    mentions, documents, chunks = _run(
        tmp_path, "quarantine", middle=_SCREEN_STAGE, entity_parent="screen", reverse=False
    )
    # The withheld chunk really is gone from the extraction input (the drop is real).
    assert all(not str(m.chunk_record_id).endswith("#chunk-0") for m in mentions)
    _assert_spans_pages(mentions, chunks, documents)
    _assert_pages_exact(mentions, documents)


def test_page_resolution_survives_a_non_screening_chunk_filter(tmp_path: Path) -> None:
    """A chunk→chunk Stage that is NOT Content Screening drops chunk 0; survivors still resolve.

    ``rehome_page_maps()`` was wired for the screening path only, so this hole stayed open for
    every other chunk→chunk Capability. Under the sidecar there is no hole to wire.
    """
    mentions, documents, chunks = _run(
        tmp_path, "filter", middle=_INDUCE_STAGE, entity_parent="induce", reverse=False
    )
    _assert_spans_pages(mentions, chunks, documents)
    _assert_pages_exact(mentions, documents)

"""P3-F2 end-to-end — a reference ``AdapterBase`` subclass runs through the LocalRunner.

Exercises :class:`~latence_core.providers.adapter.AdapterBase` end to end without adding a
Wave-1 heavy dep: a small in-test EntityExtractor **re-expressed as an AdapterBase subclass**
(the issue's "re-express one existing in-core reference provider in a test/fixture") is
registered and run Source→Parse→Chunk→Entity→Export through the LocalRunner on THIS CPU host.

It proves, through the real Runner:

* **device resolution** — the CPU adapter resolves to ``cpu`` and its profile columns surface;
* **batching on/off** — the same adapter yields byte-identical, input-order output whether its
  profile declares ``batch=True`` (grouped) or ``batch=False`` (one-item), and the output
  round-trips through the checkpoint;
* **native ``ValueError`` → ``ProviderError``** — an adapter whose native call raises is mapped
  to a typed ProviderError and FAILs the stage (a FAIL Quality Report, no leaked content);
* **provenance/offset carry** — the mention's Provenance carries the source lineage and its
  span/page resolve through the OffsetIndex/PageOffsetIndex seam (ADR-0031), identical to the
  in-core gazetteer path.

Deterministic + seeded (``LATENCE_CUDA=0``, no model download).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from latence_core import AdapterBase, ProviderProfile, ProviderRegistry
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    PageMap,
    PageSpan,
    Provenance,
)
from latence_core.errors import ProviderError
from latence_core.page_index import PageIndexResolver
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class _AdapterGazetteer(AdapterBase):
    """A reference EntityExtractor re-expressed as an ``AdapterBase`` subclass.

    Composes all four helpers: :meth:`device` (a cpu profile), :meth:`batched` (grouped
    when ``batch=True``, one-item otherwise), :meth:`guard` (mapping a native ``ValueError``
    to a typed ProviderError), and :meth:`carry_provenance` / :meth:`rebase_offsets` /
    :meth:`resolve_pages` (the Provenance/offset carry over the ADR-0031 seam). It satisfies
    the EntityExtractor Capability structurally — ``extract`` — WITHOUT any change to the
    protocol, proving AdapterBase is optional convenience, not a required layer.
    """

    # config_exceptions is left empty, so a native ValueError from the match step maps to a
    # runtime ProviderError (the default) — the E2E asserts exactly that mapping + no leak.
    config_exceptions = ()

    _BATCH = True

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source="LICENSE",
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=cls._BATCH,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self._terms: list[tuple[str, str]] = []  # (label, term)
        for label, spec in (cfg.get("labels") or {}).items():
            for term in spec.get("terms", []):
                self._terms.append((str(label), str(term)))
        # A test knob: when set, the "native" match step raises to exercise the guard.
        self._explode = bool(cfg.get("explode", False))

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        # Batch the chunks through the AdapterBase helper (grouped iff profile.batch),
        # deterministic input order either way. Each batch is mapped by _run_batch.
        # ONE resolver for the whole call: page maps are DOCUMENT-scoped and (since v18) ride
        # a document's FIRST chunk only, so a per-chunk resolver would lose them.
        resolver = PageIndexResolver()

        def run_batch(chunks_in: list[ChunkRecord]) -> list[EntityMention]:
            return self._run_batch(chunks_in, resolver)

        yield from self.batched(list(chunks), run_batch)

    def _run_batch(
        self, chunks: list[ChunkRecord], resolver: PageIndexResolver
    ) -> list[EntityMention]:
        out: list[EntityMention] = []
        for chunk in chunks:
            # Observe EVERY chunk, before the term filter decides it is uninteresting: the
            # map-carrying first chunk of a document is often the one with no match at all.
            resolver.observe(chunk)
            out.extend(self._extract_one(chunk, resolver))
        return out

    def _extract_one(self, chunk: ChunkRecord, resolver: PageIndexResolver) -> list[EntityMention]:
        with self.guard("extract"):
            if self._explode:
                # A "native library" failure: mapped to a typed ProviderError by the guard,
                # its message deliberately echoing the (sensitive) content to prove no leak.
                msg = f"native matcher crashed on content: {chunk.content!r}"
                raise ValueError(msg)
            chunk_char_start = chunk.provenance.char_start or 0
            mentions: list[EntityMention] = []
            # Deterministic: scan terms in declared order, matches by ascending position.
            found: list[tuple[int, int, str, str]] = []
            for label, term in self._terms:
                start = chunk.content.find(term)
                if start >= 0:
                    found.append((start, start + len(term), label, term))
            for idx, (local_start, local_end, label, term) in enumerate(sorted(found)):
                # Rebase the chunk-local span to true original offsets via the ADR-0031 seam
                # (identity when there is no offset map — a markup-free chunk).
                doc_start, doc_end = self.rebase_offsets(
                    local_start,
                    local_end,
                    chunk_char_start=chunk_char_start,
                    offset_map=chunk.offset_map,
                )
                # The mention's OWN page through the document-scoped resolver seam — never a
                # throwaway index over ``chunk.page_map`` (None on every non-first chunk, v18).
                page_start, page_end = self.resolve_pages(
                    chunk, doc_start, doc_end, resolver=resolver
                )
                provenance, classification = self.carry_provenance(
                    chunk,
                    char_start=doc_start,
                    char_end=doc_end,
                    page_start=page_start,
                    page_end=page_end,
                )
                mentions.append(
                    EntityMention(
                        record_id=f"{chunk.record_id}:m{idx}",
                        provenance=provenance,
                        classification=classification,
                        document_record_id=chunk.document_record_id,
                        chunk_record_id=chunk.record_id,
                        index=idx,
                        label=label,
                        text=term,
                        confidence=1.0,
                    )
                )
            return mentions


def _registry_with_adapter(*, batch: bool) -> ProviderRegistry:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access

    class _Bound(_AdapterGazetteer):
        _BATCH = batch

    ep = metadata.EntryPoint(
        name="entity.adapter_ref", value="test:injected", group="latence.providers"
    )
    object.__setattr__(ep, "load", lambda: _Bound)
    discovered["entity.adapter_ref"] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def _pipeline(store: str, docs_dir: Path, *, explode: bool = False) -> Pipeline:
    explode_line = "      explode: true\n" if explode else ""
    return Pipeline.from_yaml_text(
        f"""
name: adapter-base-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.adapter_ref
    depends_on: [chunk]
    config:
      device: auto
      batch_size: 2
{explode_line}      labels:
        ORG:
          terms: [Acme Corp]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text("Acme Corp shipped the order to the depot.\n", encoding="utf-8")
    (docs / "note.txt").write_text("A second doc mentions Acme Corp once more.\n", encoding="utf-8")
    return docs


def test_adapter_reference_runs_e2e_with_batching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner(registry=_registry_with_adapter(batch=True)).run(
        _pipeline(store, docs), run_id="adapter"
    )

    entities = next(m for m in report.stages if m.name == "entities")
    # Device resolution: the CPU adapter resolves to cpu, profile columns surfaced.
    assert entities.skipped is False
    assert entities.compute == "cpu"
    assert entities.device == "auto"
    assert entities.device_selected == "cpu"
    assert entities.license == "Apache-2.0"
    assert entities.records_out >= 2  # one "Acme Corp" per doc

    # Provenance/offset carry: mentions carry source lineage + a resolved span/page.
    s = Storage.for_uri(store)
    rows = [
        EntityMention.model_validate_json(line)
        for line in s.read_text(f"{store}/_latence/runs/adapter/export/records.jsonl").splitlines()
        if line.strip()
    ]
    acme = [m for m in rows if m.text == "Acme Corp"]
    assert len(acme) >= 2
    for m in acme:
        assert m.provenance.document_id  # lineage carried
        assert m.provenance.char_start is not None
        assert m.provenance.page_start == 1  # resolved via the page seam
        assert m.classification.sensitivity == "internal"  # inherited unchanged


def test_adapter_batch_and_nonbatch_are_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The SAME adapter with batch=True vs batch=False produces byte-identical output through
    # the Runner — batching is a memory-shaping knob, never a semantic change (determinism).
    monkeypatch.setenv("LATENCE_CUDA", "0")
    # ONE shared corpus so Source yields identical document order for both runs — the batching
    # contract is "same input ⇒ same output regardless of batch flag", so the corpus is fixed
    # and only the batch flag + the store dir vary.
    docs = _docs(tmp_path)

    def _run(batch: bool, run_id: str) -> list[str]:
        store = f"file://{tmp_path}/store-{run_id}"
        LocalRunner(registry=_registry_with_adapter(batch=batch)).run(
            _pipeline(store, docs), run_id=run_id
        )
        s = Storage.for_uri(store)
        return [
            line
            for line in s.read_text(
                f"{store}/_latence/runs/{run_id}/export/records.jsonl"
            ).splitlines()
            if line.strip()
        ]

    batched = _run(True, "batched")
    unbatched = _run(False, "unbatched")
    assert batched == unbatched
    assert len(batched) >= 2


def test_adapter_native_error_maps_to_provider_error_and_fails_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_registry_with_adapter(batch=True))

    # The native ValueError inside the adapter's match step is mapped to a typed ProviderError
    # and propagates out of the Runner (the stage FAILs — no torn commit).
    with pytest.raises(ProviderError) as excinfo:
        runner.run(_pipeline(store, docs, explode=True), run_id="boom")
    # No record content leaked into the raised message (the native error echoed chunk content).
    assert "Acme Corp shipped" not in str(excinfo.value)
    assert "extract" in str(excinfo.value)

    # A FAIL Quality Report was persisted with the entity stage's typed error category.
    s = Storage.for_uri(store)
    report_uri = f"{store}/_latence/runs/boom/quality-report.json"
    assert s.exists(report_uri)
    assert '"provider"' in s.read_text(report_uri)


# -- the v18 page-map shape through the PUBLIC adapter seam (audit R2) ---------


def _two_chunk_document() -> list[ChunkRecord]:
    """A 2-page document in the SHIPPED v18 chunk shape: the map rides chunk 0 only.

    Page 1 is ``[0, 100)``, page 2 is ``[100, 200)``. Chunk 0 is the map carrier and holds
    no term (a document's first chunk is very often the least interesting one). Chunk 1
    straddles the page boundary (``[90, 119)``, provenance pages 1-2) and carries the term
    at document offset 102 — i.e. on page **2**, not on the first page of its own chunk.
    """
    page_map = PageMap(
        pages=[
            PageSpan(page_number=1, char_start=0, char_end=100),
            PageSpan(page_number=2, char_start=100, char_end=200),
        ],
        total_chars=200,
    )
    classification = Classification(language="en", sensitivity="internal")

    def _prov(char_start: int, char_end: int, page_start: int, page_end: int) -> Provenance:
        return Provenance(
            source_uri="file://x/report.pdf",
            file_name="report.pdf",
            file_type="pdf",
            file_size=200,
            document_id="doc-1",
            char_start=char_start,
            char_end=char_end,
            page_start=page_start,
            page_end=page_end,
        )

    first = ChunkRecord(
        record_id="chunk-0",
        provenance=_prov(0, 12, 1, 1),
        classification=classification,
        document_record_id="doc-rec-1",
        index=0,
        content="Title page.",
        token_count=2,
        page_map=page_map,  # v18: the ONLY carrier of this document's map
    )
    second_content = "pad pad pad Acme Corp tail xx"  # 'Acme Corp' at local 12 -> doc 102
    second = ChunkRecord(
        record_id="chunk-1",
        provenance=_prov(90, 90 + len(second_content), 1, 2),
        classification=classification,
        document_record_id="doc-rec-1",
        index=1,
        content=second_content,
        token_count=6,
        page_map=None,  # v18: every chunk after the first carries no map
    )
    return [first, second]


def test_reference_adapter_cites_the_mentions_own_page_on_a_later_chunk() -> None:
    """A mention on page 2 of a page-straddling, map-less chunk must cite page 2 (ADR-0021/0031).

    The regression this pins is about the PUBLIC seam third parties are pointed at: the
    reference adapter resolves pages exactly the way ``docs/writing-an-adapter.md`` instructs.
    Under schema v18 a document's ``page_map`` rides its FIRST chunk only, so a helper that
    reads ``chunk.page_map`` sees ``None`` on every later chunk and silently degrades the
    mention to its chunk's INHERITED range ``(1, 2)`` — a plausible-looking wrong page, which
    is why no existing test caught it. Going through the ``PageIndexResolver`` seam (which the
    adapter feeds with EVERY chunk, including the term-free carrier) resolves the mention's own
    page: ``(2, 2)``.
    """
    adapter = _AdapterGazetteer({"labels": {"ORG": {"terms": ["Acme Corp"]}}})
    mentions = list(adapter.extract(_two_chunk_document()))

    assert [m.chunk_record_id for m in mentions] == ["chunk-1"]
    mention = mentions[0]
    # The span itself is unchanged by this fix — offset 102 is where 'Acme Corp' starts.
    assert (mention.provenance.char_start, mention.provenance.char_end) == (102, 111)
    # ...and page 2 is where offset 102 lives. Inheriting the chunk's (1, 2) is the defect.
    assert (mention.provenance.page_start, mention.provenance.page_end) == (2, 2)

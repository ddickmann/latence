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
from latence_core.contracts import EntityMention
from latence_core.errors import ProviderError
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord


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
        yield from self.batched(list(chunks), self._run_batch)

    def _run_batch(self, chunks: list[ChunkRecord]) -> list[EntityMention]:
        out: list[EntityMention] = []
        for chunk in chunks:
            out.extend(self._extract_one(chunk))
        return out

    def _extract_one(self, chunk: ChunkRecord) -> list[EntityMention]:
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
                page_start, page_end = self.resolve_pages(
                    doc_start,
                    doc_end,
                    page_map=chunk.page_map,
                    fallback_page_start=chunk.provenance.page_start,
                    fallback_page_end=chunk.provenance.page_end,
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

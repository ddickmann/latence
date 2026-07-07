"""Second reference Profiler Provider (#77): LightweightProfiler parity + benchmark.

The 'providers' hardening bar requires ≥2 reference providers per Capability with parity
tests (same typed contracts) + a benchmark vs the existing one.
:class:`LightweightProfiler` is the second CPU Profiler behind the same seam as
:class:`StatisticalProfiler`.

* **Parity**: both satisfy the ``Profiler`` protocol and emit the same typed contract —
  one DOCUMENT FeatureRecord per parsed document (the fixed feature set, matched
  document-for-document) plus exactly one CORPUS record; streaming; deterministic.
* **Benchmark / difference**: the lightweight Provider skips the O(entities²)
  co-occurrence pairing (empty ``co_occurrences``) while keeping the cheap linear tallies
  (entity/type frequency, consensus). The empty-co-occurrence assertion is the reverting
  regression guard — folding the pairing back in would FAIL it.
"""

from __future__ import annotations

from latence_core.capability import Profiler
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    EntityMention,
    FeatureScope,
    Provenance,
)
from latence_core.stages.profiling import LightweightProfiler, StatisticalProfiler


def _doc(rid: str, text: str) -> DocumentRecord:
    prov = Provenance(
        source_uri=f"file:///{rid}.txt",
        file_name=f"{rid}.txt",
        file_type="txt",
        file_size=len(text),
        document_id=rid,
        char_start=0,
        char_end=len(text),
    )
    return DocumentRecord(
        record_id=rid,
        provenance=prov,
        classification=Classification(language="en", category="news", sensitivity="internal"),
        content=text,
        page_map=None,
        disposition=Disposition.PARSED,
    )


def _mention(mid: str, rid: str, text: str, label: str) -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///{rid}.txt",
        file_name=f"{rid}.txt",
        file_type="txt",
        file_size=1,
        document_id=rid,
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=mid,
        document_record_id=rid,
        chunk_record_id=f"{rid}#chunk-0",
        index=0,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
    )


def _corpus() -> tuple[list[DocumentRecord], list[EntityMention]]:
    docs = [
        _doc("d1", "Acme and Beta merged this year in a large corporate deal."),
        _doc("d2", "Beta and Gamma also partnered recently in the European market."),
    ]
    mentions = [
        _mention("m1", "d1", "Acme", "ORG"),
        _mention("m2", "d1", "Beta", "ORG"),
        _mention("m3", "d2", "Beta", "ORG"),
        _mention("m4", "d2", "Gamma", "ORG"),
    ]
    return docs, mentions


def test_second_profiler_satisfies_the_same_seam() -> None:
    assert isinstance(LightweightProfiler(), Profiler)


def test_second_profiler_emits_one_corpus_record_and_a_document_record_each() -> None:
    docs, mentions = _corpus()
    records = list(LightweightProfiler().profile(docs, mentions))
    doc_records = [r for r in records if r.scope is FeatureScope.DOCUMENT]
    corpus_records = [r for r in records if r.scope is FeatureScope.CORPUS]
    assert len(doc_records) == len(docs)
    assert len(corpus_records) == 1  # exactly one corpus-scope record (contract)
    assert records[-1].scope is FeatureScope.CORPUS  # emitted last (determinism)


def test_second_profiler_per_document_features_match_the_reference() -> None:
    """Parity: the fixed per-document feature set is computed identically to the reference."""
    docs, mentions = _corpus()
    lw = {
        r.record_id: r.document
        for r in LightweightProfiler().profile(docs, mentions)
        if r.scope is FeatureScope.DOCUMENT
    }
    st = {
        r.record_id: r.document
        for r in StatisticalProfiler().profile(docs, mentions)
        if r.scope is FeatureScope.DOCUMENT
    }
    assert lw == st and lw  # identical per-document feature records


def test_second_profiler_matches_reference_tallies_but_skips_co_occurrence() -> None:
    """Benchmark: same cheap tallies, but the O(n²) co-occurrence pairing is skipped."""
    docs, mentions = _corpus()
    lw_corpus = next(
        r.corpus for r in LightweightProfiler().profile(docs, mentions) if r.scope is FeatureScope.CORPUS
    )
    st_corpus = next(
        r.corpus for r in StatisticalProfiler().profile(docs, mentions) if r.scope is FeatureScope.CORPUS
    )
    assert lw_corpus is not None and st_corpus is not None
    # The cheap linear tallies match the reference exactly (parity on the shared features).
    assert lw_corpus.entity_frequency == st_corpus.entity_frequency
    assert lw_corpus.entity_type_frequency == st_corpus.entity_type_frequency
    assert lw_corpus.entity_type_consensus == st_corpus.entity_type_consensus
    # The measurable difference: the reference records co-occurrence pairs; the lightweight
    # Provider records NONE (the scale trade-off). Reverting the skip would fail this.
    assert st_corpus.co_occurrences  # reference DID pair Beta with Acme and Gamma
    assert lw_corpus.co_occurrences == []


def test_second_profiler_skips_parse_error_documents() -> None:
    docs, mentions = _corpus()
    bad = DocumentRecord(
        record_id="bad",
        provenance=docs[0].provenance.model_copy(update={"document_id": "bad"}),
        classification=docs[0].classification,
        content="",
        page_map=None,
        disposition=Disposition.PARSE_ERROR,
        error="unreadable source",
    )
    records = list(LightweightProfiler().profile([bad, *docs], mentions))
    doc_ids = {r.document_record_id for r in records if r.scope is FeatureScope.DOCUMENT}
    assert "bad" not in doc_ids
    assert doc_ids == {"d1", "d2"}


def test_second_profiler_is_deterministic() -> None:
    docs, mentions = _corpus()
    profiler = LightweightProfiler()
    first = [(r.record_id, r.scope) for r in profiler.profile(docs, mentions)]
    second = [(r.record_id, r.scope) for r in profiler.profile(docs, mentions)]
    assert first == second

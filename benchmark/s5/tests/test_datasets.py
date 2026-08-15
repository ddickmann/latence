# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The dataset gold translations — pinned against the real S1 prep functions and the real files.

The tests that matter here are the *agreement* ones: the harness derives corpus filenames from
query metadata, and if that derivation ever drifts from
:mod:`latence_benchmark.campaign.s3_lib`, every recall number built on it silently becomes
meaningless. So the derivation is compared to the prep function's own output, not to a
hard-coded string.
"""

from __future__ import annotations

import json

import pytest
from benchmark.s5.datasets import (
    DATASETS,
    OHR_DOCUMENT_ALIASES,
    SOTA_CORPUS,
    load_multihop_rag,
    load_ohr_bench,
    load_uda,
    load_vidoseek,
    load_wiki2multihop,
    multihop_rag_title_map,
    ohr_bench_exclusion,
    ohr_bench_file_name,
    prep_filename,
    uda_file_name,
    uda_page_from_q_uid,
)
from benchmark.s5.goldmap import ExclusionReason, RefKind
from latence_benchmark.campaign.s3_lib import multihop_rag_articles, wiki2multihop_paragraphs

_MHRAG_CORPUS = SOTA_CORPUS / "multihop_rag" / "raw" / "corpus.json"
_WIKI_DEV = SOTA_CORPUS / "wiki2multihop" / "raw" / "dev.parquet"
_UDA_FIN = SOTA_CORPUS / "uda" / "raw" / "fin" / "test_00000_of_00001.parquet"
_UDA_PAPER = SOTA_CORPUS / "uda" / "raw" / "paper_tab" / "test_00000_of_00001.parquet"
_OHR_QA = SOTA_CORPUS / "ohr_bench" / "raw" / "OHR-Bench.parquet"
_OHR_EXPORT = SOTA_CORPUS / "s3-exports" / "ohr_bench" / "corpus.parquet"
_UDA_EXPORT = SOTA_CORPUS / "s3-exports" / "uda" / "corpus.parquet"


def _export_file_names(corpus_parquet) -> set[str]:  # type: ignore[no-untyped-def]
    """Every ``provenance.file_name`` an export emitted — the set a gold name must land in."""
    import pyarrow.parquet as pq

    column = pq.read_table(corpus_parquet, columns=["provenance"]).column("provenance")  # type: ignore[no-untyped-call]
    return {str(json.loads(raw)["file_name"]) for raw in column.to_pylist()}

requires_corpus = pytest.mark.skipif(
    not _MHRAG_CORPUS.is_file(), reason="the S1 raw corpora are not on this machine"
)


def test_prep_filename_matches_the_real_prep_for_a_synthetic_document() -> None:
    """The one place the harness reproduces pipeline logic, checked against the pipeline."""
    payload = json.dumps([{"title": "A Curious Title: Part 2", "body": "Body text here."}])
    (prep,) = multihop_rag_articles(payload)
    assert prep_filename("A Curious Title: Part 2", "Body text here.") == prep.filename


@requires_corpus
def test_multihop_rag_title_map_covers_every_prepped_article() -> None:
    """All 609 derived names must equal the set the prep function actually wrote."""
    expected = {f.filename for f in multihop_rag_articles(_MHRAG_CORPUS.read_text())}
    assert set(multihop_rag_title_map().values()) == expected


@requires_corpus
def test_multihop_rag_queries_carry_one_chain_slot_per_distinct_article() -> None:
    queries = load_multihop_rag()
    assert len(queries) == 2556
    multi = next(q for q in queries if len(q.refs) > 2)
    assert [ref.slot for ref in multi.refs] == list(range(len(multi.refs)))
    assert len({ref.file_name for ref in multi.refs}) == len(multi.refs)
    # `null_query` items are unanswerable by construction and carry no evidence: they must
    # appear with zero refs (excluded from scoring) rather than with invented gold.
    assert any(q.question_type == "null_query" and not q.refs for q in queries)


@pytest.mark.skipif(not _WIKI_DEV.is_file(), reason="2Wiki dev parquet is not on this machine")
def test_wiki2multihop_paragraph_names_agree_with_the_prep() -> None:
    """A sample of derived paragraph names must exist in the prep's own output set."""
    import pyarrow.parquet as pq

    contexts = pq.read_table(_WIKI_DEV, columns=["context"]).column("context").to_pylist()[:200]  # type: ignore[no-untyped-call]
    expected = {f.filename for f in wiki2multihop_paragraphs(contexts)}
    derived = {ref.file_name for q in load_wiki2multihop()[:50] for ref in q.refs}
    assert derived
    assert derived <= expected


@pytest.mark.skipif(
    not (SOTA_CORPUS / "vidoseek" / "raw" / "vidoseek.json").is_file(),
    reason="ViDoSeek raw queries are not on this machine",
)
def test_vidoseek_gold_is_page_scoped_and_one_based() -> None:
    queries = load_vidoseek()
    assert len(queries) == 1142
    assert all(ref.kind == "page" for q in queries for ref in q.refs)
    assert all(ref.page is not None and ref.page >= 1 for q in queries for ref in q.refs)
    assert all(q.refs[0].file_name.startswith("pdf__") for q in queries if q.refs)


def test_uda_doc_name_is_the_corpus_key_for_both_halves() -> None:
    """The mapping that took UDA from 81% to total: ``doc_name`` IS the filename stem."""
    assert uda_file_name("fin", "ADI_2009") == "fin_docs__ADI_2009.md"
    assert uda_file_name("paper_tab", "1912.01214") == "paper_docs__1912.01214.md"


def test_uda_fin_page_comes_from_the_q_uid() -> None:
    """``ADI/2009/page_49.pdf-1`` → 49; a paper_tab sha1 carries no page and must say so."""
    assert uda_page_from_q_uid("ADI/2009/page_49.pdf-1") == 49
    assert uda_page_from_q_uid("5eda469a8a77f028d0c5f1acd296111085614537") is None


@pytest.mark.skipif(
    not (_UDA_FIN.is_file() and _UDA_PAPER.is_file() and _UDA_EXPORT.is_file()),
    reason="the UDA query parquets / export are not on this machine",
)
def test_every_uda_query_names_a_document_the_export_contains() -> None:
    """Both halves, end to end: 8,190 fin + 393 paper_tab rows, no unmapped name."""
    queries = load_uda()
    assert len(queries) == 8190 + 393
    files = _export_file_names(_UDA_EXPORT)
    unmapped = {ref.file_name for q in queries for ref in q.refs if ref.file_name not in files}
    assert unmapped == set()
    fin = [q for q in queries if q.question_type == "fin"]
    paper = [q for q in queries if q.question_type == "paper_tab"]
    # fin keeps the finer PAGE reference (with a document fallback for the reports whose page
    # spans do not reconstruct); paper_tab has no page to keep and is document-level.
    assert all(
        r.kind is RefKind.PAGE and r.allow_document_fallback for q in fin for r in q.refs
    )
    assert all(r.page is not None for q in fin for r in q.refs)
    assert all(r.kind is RefKind.FILE for q in paper for r in q.refs)


def test_ohr_bench_paper_domain_is_aliased_onto_academic() -> None:
    """QA says ``paper/``; S1 wrote the same 10 documents as ``academic__…``."""
    assert ohr_bench_file_name("paper/2305.12345") == "academic__2305.12345.md"
    assert ohr_bench_file_name("textbook/Foo.pdf_1") == "textbook__Foo.pdf_1.md"
    (long_name, short_file), *_ = OHR_DOCUMENT_ALIASES.items()
    assert ohr_bench_file_name(long_name) == short_file


def test_ohr_bench_notes_domain_is_excluded_with_its_reason() -> None:
    """288 GNHK handwriting documents were never acquired — a coverage fact, not a miss."""
    assert ohr_bench_exclusion("notes/some_page") is ExclusionReason.CORPUS_MISSING_DOMAIN
    assert ohr_bench_exclusion("law/some_contract") is None


@pytest.mark.skipif(
    not (_OHR_QA.is_file() and _OHR_EXPORT.is_file()),
    reason="the OHR-Bench QA parquet / export is not on this machine",
)
def test_ohr_bench_gold_is_either_in_the_export_or_explicitly_excluded() -> None:
    """No third category: every query either maps onto a real file or states why it cannot."""
    queries = load_ohr_bench()
    files = _export_file_names(_OHR_EXPORT)
    assert "notes" not in {f.split("__")[0] for f in files}  # the constant is still true
    orphans = [
        q
        for q in queries
        if q.exclusion is None and any(ref.file_name not in files for ref in q.refs)
    ]
    assert orphans == []
    excluded = [q for q in queries if q.exclusion is not None]
    assert excluded
    assert {q.exclusion for q in excluded} == {ExclusionReason.CORPUS_MISSING_DOMAIN}


def test_the_control_group_is_exactly_the_single_hop_datasets() -> None:
    assert {name for name, spec in DATASETS.items() if spec.single_hop} == {
        "vidoseek",
        "ohr_bench",
        "uda",
    }

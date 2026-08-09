# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
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
    SOTA_CORPUS,
    load_multihop_rag,
    load_uda,
    load_vidoseek,
    load_wiki2multihop,
    multihop_rag_title_map,
    prep_filename,
)
from latence_benchmark.campaign.s3_lib import multihop_rag_articles, wiki2multihop_paragraphs

_MHRAG_CORPUS = SOTA_CORPUS / "multihop_rag" / "raw" / "corpus.json"
_WIKI_DEV = SOTA_CORPUS / "wiki2multihop" / "raw" / "dev.parquet"

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


@pytest.mark.skipif(
    not (SOTA_CORPUS / "uda" / "raw" / "extended_qa_info" / "fin_qa.json").is_file(),
    reason="UDA raw QA info is not on this machine",
)
def test_uda_fin_gold_comes_from_the_q_uid_page_not_the_finqa_evidence_text() -> None:
    fin = [q for q in load_uda() if q.question_type == "fin"]
    assert fin
    sample = next(q for q in fin if "page_" in q.query_id)
    assert [ref.kind for ref in sample.refs] == ["page"]
    assert sample.refs[0].file_name.startswith("fin_docs__")


def test_the_control_group_is_exactly_the_single_hop_datasets() -> None:
    assert {name for name, spec in DATASETS.items() if spec.single_hop} == {
        "vidoseek",
        "ohr_bench",
        "uda",
    }

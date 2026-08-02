# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The ticket-21 labeling harness — format round-trip, resume, coverage, span integrity.

Offline throughout: the corpus fixture reproduces the real S1 export row shape
(``records.jsonl`` lines carrying ``record_id`` / ``content`` / ``masked_content`` /
``provenance.document_id``), and the calibration fixture writes a parquet with the exact
column set of the S9 export's ``graph-hyperedges.parquet``. The HTTP handler is driven by
the stdlib client against a loopback server on an ephemeral port — no browser.
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from latence_benchmark.labeling import (
    JUDGMENTS,
    Corpus,
    GoldQuestion,
    SpanIntegrityError,
    SupportingFact,
    append_judgment,
    append_question,
    coverage,
    load_corpus,
    load_gold,
    load_hyperedges,
    load_judgments,
    question_id,
    validate_calibration,
    validate_gold,
)
from latence_benchmark.labeling.cli import main as label_main
from latence_benchmark.labeling.server import CalibrationSession, GoldSession, make_server

if TYPE_CHECKING:
    from collections.abc import Iterator

# -- fixtures ---------------------------------------------------------------

DOC_A = "doc-aaaa"
DOC_B = "doc-bbbb"
CHUNK_ROWS = [
    {
        "record_id": "chunk-a1",
        "content": "RAW must never be indexed.",
        "masked_content": "Alice founded Acme Corp in 2001.",
        "document_record_id": "docrec-a",
        "index": 0,
        "media_type": "text/markdown",
        "provenance": {
            "document_id": DOC_A,
            "file_name": "acme.md",
            "source_uri": "mem://acme.md",
            "file_type": "md",
            "file_size": 10,
            "char_start": 0,
            "char_end": 32,
        },
    },
    {
        "record_id": "chunk-a2",
        "content": "Acme Corp is headquartered in Berlin.",
        "masked_content": None,
        "document_record_id": "docrec-a",
        "index": 1,
        "media_type": "text/markdown",
        "provenance": {
            "document_id": DOC_A,
            "file_name": "acme.md",
            "source_uri": "mem://acme.md",
            "file_type": "md",
            "file_size": 10,
            "char_start": 33,
            "char_end": 70,
        },
    },
    {
        "record_id": "chunk-b1",
        "content": "Berlin hosted the 2030 expo.",
        "document_record_id": "docrec-b",
        "index": 0,
        "media_type": "text/markdown",
        "provenance": {
            "document_id": DOC_B,
            "file_name": "expo.md",
            "source_uri": "mem://expo.md",
            "file_type": "md",
            "file_size": 5,
            "char_start": 0,
            "char_end": 28,
        },
    },
]


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    root.mkdir()
    with (root / "records.jsonl").open("w", encoding="utf-8") as fh:
        for row in CHUNK_ROWS:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return root


@pytest.fixture
def corpus(export_dir: Path) -> Corpus:
    return load_corpus(export_dir)


def _fact(
    corpus: Corpus, chunk_id: str, start: int, end: int, text: str | None = None
) -> SupportingFact:
    chunk = corpus.chunk(chunk_id)
    return SupportingFact(
        document_id=chunk.document_id,
        chunk_record_id=chunk_id,
        char_start=start,
        char_end=end,
        text=chunk.text[start:end] if text is None else text,
    )


HYPEREDGE_SCHEMA = pa.schema(
    [
        ("hyperedge_id", pa.string()),
        ("entity_node_ids", pa.string()),
        ("relations", pa.string()),
        ("tau_text", pa.string()),
        ("token_cost", pa.int64()),
        ("token_counter", pa.string()),
        ("gamma", pa.float64()),
        ("support", pa.int64()),
        ("has_relations", pa.bool_()),
        ("arity", pa.int64()),
        ("member_mention_ids", pa.string()),
        ("merged_sources", pa.string()),
        ("properties", pa.string()),
        ("evidence", pa.string()),
    ]
)


@pytest.fixture
def hyperedges_parquet(tmp_path: Path) -> Path:
    rows = [
        {
            "hyperedge_id": f"he-{i}",
            "entity_node_ids": json.dumps([f"node-{i}", "node-x"]),
            "relations": json.dumps([{"label": "founded", "confidence": 0.9}]),
            "tau_text": f"Sentence {i} about node-{i} and node-x.",
            "token_cost": 12,
            "token_counter": "whitespace",
            "gamma": 0.25 * i,
            "support": i,
            "has_relations": True,
            "arity": 2,
            "member_mention_ids": "[]",
            "merged_sources": "[]",
            "properties": "{}",
            "evidence": "{}",
        }
        for i in range(3)
    ]
    path = tmp_path / "graph-hyperedges.parquet"
    table = pa.Table.from_pylist(rows, schema=HYPEREDGE_SCHEMA)
    # cast, not type-ignore: pyarrow is dynamically typed in this repo's mypy config, and
    # an ignore would flip to "unused" in environments where pyarrow is absent.
    cast("Any", pq).write_table(table, path)
    return path


# -- corpus loading ---------------------------------------------------------


def test_corpus_loads_chunks_grouped_and_ordered(corpus: Corpus) -> None:
    assert set(corpus.documents) == {DOC_A, DOC_B}
    assert corpus.documents[DOC_A] == ["chunk-a1", "chunk-a2"]
    # masked_content IS the corpus text when present; content otherwise.
    assert corpus.chunk("chunk-a1").text == "Alice founded Acme Corp in 2001."
    assert corpus.chunk("chunk-a2").text == "Acme Corp is headquartered in Berlin."
    assert corpus.chunk("chunk-a1").file_name == "acme.md"


def test_corpus_loads_from_parquet_rows(tmp_path: Path) -> None:
    # The Parquet export JSON-encodes provenance and carries the corpus text in `content`.
    rows = [
        {
            "record_id": "chunk-p1",
            "content": "Parquet chunk text.",
            "media_type": "text/markdown",
            "provenance": json.dumps({"document_id": "doc-p", "file_name": "p.md"}),
            "classification": "{}",
            "schema_version": 1,
        }
    ]
    path = tmp_path / "records.parquet"
    cast("Any", pq).write_table(pa.Table.from_pylist(rows), path)
    corpus = load_corpus(tmp_path)
    assert corpus.chunk("chunk-p1").text == "Parquet chunk text."
    assert corpus.documents == {"doc-p": ["chunk-p1"]}


# -- gold format round-trip + resume ---------------------------------------


def test_question_id_is_deterministic_hash_of_text() -> None:
    assert question_id("Who founded Acme?") == question_id("  Who founded Acme?  ")
    assert question_id("Who founded Acme?") != question_id("Who founded Acme Corp?")
    assert question_id("q").startswith("q-") and len(question_id("q")) == 18


def test_append_load_round_trip_and_resume(tmp_path: Path, corpus: Corpus) -> None:
    gold_path = tmp_path / "gold.jsonl"
    facts = [_fact(corpus, "chunk-a1", 0, 5), _fact(corpus, "chunk-b1", 0, 6)]
    saved = append_question(gold_path, corpus, "Where is the founder's company's city expo?", 2, facts)
    # Round-trip: the file line parses back into the identical object.
    loaded = load_gold(gold_path)
    assert loaded == {saved.question_id: saved}
    line = json.loads(gold_path.read_text(encoding="utf-8").splitlines()[0])
    assert line["question_id"] == saved.question_id
    assert line["hop_count"] == 2
    assert line["supporting_facts"][0] == {
        "document_id": DOC_A,
        "chunk_record_id": "chunk-a1",
        "char_start": 0,
        "char_end": 5,
        "text": "Alice",
    }
    # Resume: a second session appends without clobbering the first question.
    append_question(gold_path, corpus, "Second question about Berlin?", 1, [_fact(corpus, "chunk-b1", 0, 6)])
    resumed = load_gold(gold_path)
    assert len(resumed) == 2
    assert saved.question_id in resumed


def test_duplicate_question_rejected_then_replaced(tmp_path: Path, corpus: Corpus) -> None:
    gold_path = tmp_path / "gold.jsonl"
    facts = [_fact(corpus, "chunk-a1", 0, 5), _fact(corpus, "chunk-a2", 0, 9)]
    append_question(gold_path, corpus, "Who founded what?", 2, facts)
    with pytest.raises(ValueError, match="already labeled"):
        append_question(gold_path, corpus, "Who founded what?", 2, facts)
    # replace=True appends a correction; the loader keeps the LAST occurrence.
    replaced = append_question(gold_path, corpus, "Who founded what?", 3, facts, replace=True)
    loaded = load_gold(gold_path)
    assert loaded[replaced.question_id].hop_count == 3
    assert len(gold_path.read_text(encoding="utf-8").splitlines()) == 2


def test_multi_hop_question_requires_two_facts(tmp_path: Path, corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="supporting facts"):
        append_question(
            tmp_path / "gold.jsonl", corpus, "Two-hop?", 2, [_fact(corpus, "chunk-a1", 0, 5)]
        )
    # A 1-hop question may carry a single fact.
    saved = append_question(
        tmp_path / "gold.jsonl", corpus, "One-hop?", 1, [_fact(corpus, "chunk-a1", 0, 5)]
    )
    assert saved.hop_count == 1


# -- span integrity (write time + validate) ---------------------------------


def test_span_integrity_rejected_at_write_time(tmp_path: Path, corpus: Corpus) -> None:
    gold_path = tmp_path / "gold.jsonl"
    bad_text = _fact(corpus, "chunk-a1", 0, 5, text="Bob")  # verbatim text mismatch
    with pytest.raises(SpanIntegrityError, match="mismatch"):
        append_question(gold_path, corpus, "Q?", 2, [bad_text, _fact(corpus, "chunk-b1", 0, 6)])
    out_of_range = SupportingFact(DOC_A, "chunk-a1", 0, 10_000, "x")
    with pytest.raises(SpanIntegrityError, match="out of range"):
        append_question(gold_path, corpus, "Q?", 2, [out_of_range, _fact(corpus, "chunk-b1", 0, 6)])
    unknown_chunk = SupportingFact(DOC_A, "chunk-zz", 0, 5, "Alice")
    with pytest.raises(SpanIntegrityError, match="unknown chunk_record_id"):
        append_question(gold_path, corpus, "Q?", 2, [unknown_chunk, _fact(corpus, "chunk-b1", 0, 6)])
    wrong_document = SupportingFact(DOC_B, "chunk-a1", 0, 5, "Alice")
    with pytest.raises(SpanIntegrityError, match="belongs to"):
        append_question(gold_path, corpus, "Q?", 2, [wrong_document, _fact(corpus, "chunk-b1", 0, 6)])
    # Nothing hit the file: write-time verification precedes the append.
    assert not gold_path.exists()


def test_validate_gold_accepts_good_file(tmp_path: Path, corpus: Corpus) -> None:
    gold_path = tmp_path / "gold.jsonl"
    facts = [_fact(corpus, "chunk-a1", 6, 13), _fact(corpus, "chunk-b1", 0, 6)]
    append_question(gold_path, corpus, "Good question?", 2, facts)
    assert validate_gold(corpus, gold_path) == []


def test_validate_gold_rejects_corrupted_spans(tmp_path: Path, corpus: Corpus) -> None:
    gold_path = tmp_path / "gold.jsonl"
    facts = [_fact(corpus, "chunk-a1", 6, 13), _fact(corpus, "chunk-b1", 0, 6)]
    saved = append_question(gold_path, corpus, "Good question?", 2, facts)
    # Corrupt the stored span text after the fact — validate must catch it.
    obj = saved.to_json()
    obj["supporting_facts"][0]["text"] = "tampered"
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text(json.dumps(obj) + "\nnot json\n", encoding="utf-8")
    errors = validate_gold(corpus, corrupt)
    assert len(errors) == 2
    assert any("mismatch" in e for e in errors)
    assert any("malformed" in e for e in errors)


def test_validate_gold_rejects_wrong_question_id_and_missing_file(
    tmp_path: Path, corpus: Corpus
) -> None:
    fact = _fact(corpus, "chunk-a1", 0, 5)
    obj = GoldQuestion("q-0000000000000000", "Mismatch?", 1, (fact,)).to_json()
    path = tmp_path / "bad-id.jsonl"
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    errors = validate_gold(corpus, path)
    assert any("does not match" in e for e in errors)
    assert validate_gold(corpus, tmp_path / "absent.jsonl") == [
        f"{tmp_path / 'absent.jsonl'}: gold file does not exist"
    ]


# -- coverage + the 1-hop guard --------------------------------------------


def _gold(hop: int, i: int) -> GoldQuestion:
    fact = SupportingFact("d", "c", 0, 1, "x")
    return GoldQuestion(f"q-{hop}{i:03d}", f"q {hop} {i}", hop, (fact,))


def test_coverage_counts_hop_distribution() -> None:
    cov = coverage([_gold(1, 0), _gold(2, 1), _gold(2, 2), _gold(3, 3)])
    assert cov.total == 4
    assert cov.hop_distribution == {1: 1, 2: 2, 3: 1}
    assert cov.one_hop_share == 0.25
    assert cov.warning is None


def test_coverage_warns_when_one_hop_share_exceeds_40_percent() -> None:
    cov = coverage([_gold(1, 0), _gold(1, 1), _gold(1, 2), _gold(2, 3), _gold(2, 4)])
    assert cov.one_hop_share == 0.6
    assert cov.warning is not None and "1-hop share 60%" in cov.warning


def test_coverage_empty_set_has_no_warning() -> None:
    cov = coverage([])
    assert cov.total == 0 and cov.warning is None and cov.one_hop_share == 0.0


# -- calibration mode -------------------------------------------------------


def test_hyperedges_load_from_export_schema(hyperedges_parquet: Path) -> None:
    rows = load_hyperedges(hyperedges_parquet)
    assert [r.hyperedge_id for r in rows] == ["he-0", "he-1", "he-2"]
    assert rows[1].tau_text == "Sentence 1 about node-1 and node-x."
    assert rows[1].entity_node_ids == ("node-1", "node-x")
    assert rows[1].gamma == 0.25
    assert json.loads(rows[1].relations_json) == [{"label": "founded", "confidence": 0.9}]


def test_judgment_format_round_trip_and_resume(tmp_path: Path, hyperedges_parquet: Path) -> None:
    out = tmp_path / "judgments.jsonl"
    known = {r.hyperedge_id for r in load_hyperedges(hyperedges_parquet)}
    append_judgment(out, known, "he-0", "correct")
    append_judgment(out, known, "he-1", "partial", note="entity set too wide")
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"hyperedge_id": "he-0", "judgment": "correct"}
    assert lines[1] == {"hyperedge_id": "he-1", "judgment": "partial", "note": "entity set too wide"}
    # Resume + correction: re-judging appends and the loader keeps the last word.
    append_judgment(out, known, "he-0", "incorrect")
    judgments = load_judgments(out)
    assert judgments["he-0"].judgment == "incorrect"
    assert len(judgments) == 2


def test_judgment_vocabulary_and_unknown_id_rejected(
    tmp_path: Path, hyperedges_parquet: Path
) -> None:
    out = tmp_path / "judgments.jsonl"
    known = {r.hyperedge_id for r in load_hyperedges(hyperedges_parquet)}
    assert JUDGMENTS == ("correct", "incorrect", "partial")
    with pytest.raises(ValueError, match="judgment must be one of"):
        append_judgment(out, known, "he-0", "maybe")
    with pytest.raises(ValueError, match="unknown hyperedge_id"):
        append_judgment(out, known, "he-99", "correct")
    assert not out.exists()


def test_validate_calibration_accept_and_reject(tmp_path: Path, hyperedges_parquet: Path) -> None:
    out = tmp_path / "judgments.jsonl"
    known = {r.hyperedge_id for r in load_hyperedges(hyperedges_parquet)}
    append_judgment(out, known, "he-0", "correct")
    assert validate_calibration(out, known) == []
    out.write_text(
        out.read_text(encoding="utf-8")
        + json.dumps({"hyperedge_id": "he-77", "judgment": "correct"})
        + "\n"
        + json.dumps({"hyperedge_id": "he-1", "judgment": "sorta"})
        + "\n",
        encoding="utf-8",
    )
    errors = validate_calibration(out, known)
    assert len(errors) == 2
    assert any("unknown hyperedge_id" in e for e in errors)
    assert any("malformed" in e for e in errors)


# -- HTTP handler (stdlib client, no browser) --------------------------------


@pytest.fixture
def gold_server(corpus: Corpus, tmp_path: Path) -> Iterator[tuple[HTTPConnection, Path]]:
    session = GoldSession.open(corpus, tmp_path / "gold.jsonl")
    server = make_server(session, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        yield conn, tmp_path / "gold.jsonl"
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    conn: HTTPConnection, method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    return response.status, json.loads(response.read().decode("utf-8"))


def test_gold_server_serves_page_state_documents_and_saves(
    gold_server: tuple[HTTPConnection, Path], corpus: Corpus
) -> None:
    conn, gold_path = gold_server
    conn.request("GET", "/")
    page = conn.getresponse()
    html = page.read().decode("utf-8")
    assert page.status == 200 and "latence-label" in html and "<script>" in html

    status, state = _request(conn, "GET", "/api/state")
    assert status == 200 and state["mode"] == "gold" and state["coverage"]["total"] == 0

    status, docs = _request(conn, "GET", "/api/documents")
    assert status == 200
    assert {d["document_id"] for d in docs["documents"]} == {DOC_A, DOC_B}

    status, doc = _request(conn, "GET", f"/api/document?document_id={DOC_A}")
    assert status == 200
    assert [c["chunk_record_id"] for c in doc["chunks"]] == ["chunk-a1", "chunk-a2"]
    assert doc["chunks"][0]["text"] == "Alice founded Acme Corp in 2001."

    payload = {
        "question": "Which city hosts the expo of Acme's HQ?",
        "hop_count": 2,
        "supporting_facts": [
            _fact(corpus, "chunk-a2", 30, 36).to_json(),
            _fact(corpus, "chunk-b1", 0, 6).to_json(),
        ],
    }
    status, saved = _request(conn, "POST", "/api/questions", payload)
    assert status == 200
    assert saved["coverage"]["total"] == 1
    assert load_gold(gold_path)[saved["question"]["question_id"]].hop_count == 2

    # A non-round-tripping span is refused with 400 and nothing further is written.
    bad = dict(payload)
    bad["question"] = "Another question?"
    bad["supporting_facts"] = [
        {**_fact(corpus, "chunk-a2", 30, 36).to_json(), "text": "Munich"},
        _fact(corpus, "chunk-b1", 0, 6).to_json(),
    ]
    status, error = _request(conn, "POST", "/api/questions", bad)
    assert status == 400 and "mismatch" in error["error"]
    assert len(load_gold(gold_path)) == 1

    # Duplicate question text → 409 so the UI can tell the labeler explicitly.
    status, error = _request(conn, "POST", "/api/questions", payload)
    assert status == 409 and "already labeled" in error["error"]


def test_calibration_server_lists_and_records(
    hyperedges_parquet: Path, tmp_path: Path
) -> None:
    session = CalibrationSession.open(load_hyperedges(hyperedges_parquet), tmp_path / "j.jsonl")
    server = make_server(session, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        status, state = _request(conn, "GET", "/api/state")
        assert status == 200 and state["mode"] == "calibration" and state["total"] == 3

        status, listing = _request(conn, "GET", "/api/hyperedges")
        assert status == 200 and len(listing["hyperedges"]) == 3

        body = {"hyperedge_id": "he-2", "judgment": "partial", "note": "over-merged"}
        status, saved = _request(conn, "POST", "/api/judgments", body)
        assert status == 200 and saved["judged"] == 1
        assert load_judgments(tmp_path / "j.jsonl")["he-2"].note == "over-merged"

        status, error = _request(conn, "POST", "/api/judgments", {"hyperedge_id": "he-2", "judgment": "meh"})
        assert status == 400 and "judgment must be one of" in error["error"]

        # Gold-only endpoints do not exist in calibration mode.
        status, error = _request(conn, "GET", "/api/documents")
        assert status == 404
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- CLI --------------------------------------------------------------------


def test_cli_validate_gate_accept_and_reject(
    tmp_path: Path, export_dir: Path, corpus: Corpus, capsys: pytest.CaptureFixture[str]
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    facts = [_fact(corpus, "chunk-a1", 0, 5), _fact(corpus, "chunk-b1", 0, 6)]
    append_question(gold_path, corpus, "CLI gate question?", 2, facts)
    assert label_main(["validate", "--export-dir", str(export_dir), "--gold", str(gold_path)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "questions labeled: 1" in out

    tampered = gold_path.read_text(encoding="utf-8").replace('"Alice"', '"Mallory"')
    bad_path = tmp_path / "tampered.jsonl"
    bad_path.write_text(tampered, encoding="utf-8")
    assert label_main(["validate", "--export-dir", str(export_dir), "--gold", str(bad_path)]) == 1
    err = capsys.readouterr().err
    assert "mismatch" in err and "FAIL" in err


def test_cli_validate_covers_calibration_sidecar(
    tmp_path: Path, export_dir: Path, corpus: Corpus, hyperedges_parquet: Path
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    facts = [_fact(corpus, "chunk-a1", 0, 5), _fact(corpus, "chunk-b1", 0, 6)]
    append_question(gold_path, corpus, "With sidecar?", 2, facts)
    judgments = tmp_path / "judgments.jsonl"
    known = {r.hyperedge_id for r in load_hyperedges(hyperedges_parquet)}
    append_judgment(judgments, known, "he-0", "correct")
    argv = [
        "validate",
        "--export-dir", str(export_dir),
        "--gold", str(gold_path),
        "--hyperedges", str(hyperedges_parquet),
        "--calibration", str(judgments),
    ]
    assert label_main(argv) == 0
    judgments.write_text(
        json.dumps({"hyperedge_id": "he-404", "judgment": "correct"}) + "\n", encoding="utf-8"
    )
    assert label_main(argv) == 1


def test_cli_stats_prints_coverage_and_warning(
    tmp_path: Path, corpus: Corpus, capsys: pytest.CaptureFixture[str]
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    append_question(gold_path, corpus, "Only one hop?", 1, [_fact(corpus, "chunk-a1", 0, 5)])
    assert label_main(["stats", "--gold", str(gold_path)]) == 0
    out = capsys.readouterr().out
    assert "questions labeled: 1" in out and "WARNING" in out and "1-hop share 100%" in out

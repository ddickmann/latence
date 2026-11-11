"""What people ask is a signal on the elements they use, and nothing more (ticket 32)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from estate.induce import induce
from estate.reader_duckdb import DuckDBReader
from estate.usage import attach, read_usage

from .conftest import OPTIONS


def _log(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "usage.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _record(rid: str, **body: Any) -> dict[str, Any]:
    return {"format": "usage/1", "source": "golden", "synthetic": True, "id": rid, "language": "en", "answered": True,
            "measures": [], "dimensions": [], "joins": [], "terms": [], **body}


CLAIMS = {"object": "shop.claims", "name": "claim_count"}
MAKE = {"name": "make", "columns": ["shop.claims.make", "shop.parts.make"]}
STATUS = {"name": "status", "columns": ["shop.claims.status"]}
TO_DEALER = {"from": "shop.claims", "columns": ["dealer_id"], "to": "shop.dealers"}


def test_counts_land_on_the_measure_the_value_set_and_the_relation_a_question_used(reader: DuckDBReader, tmp_path: Path) -> None:
    model = induce(reader, **OPTIONS)
    records = read_usage([_log(tmp_path, [
        _record("q1", measures=[CLAIMS], dimensions=[MAKE], joins=[TO_DEALER]),
        _record("q2", measures=[CLAIMS], dimensions=[MAKE]),
        _record("q3", measures=[{"object": "shop.claims", "name": "paid_amount"}], source="turns", synthetic=False),
        _record("q4", measures=[CLAIMS], answered=False),  # asked and not answered: no evidence of use
    ])])
    out, meta = attach(model, records)
    measure = next(m for m in out["measures"] if m["id"] == "measure:shop_claims.claim_count")
    assert measure["usage"] == {"answers": 2, "sources": {"golden": 2}, "synthetic_only": True}
    assert measure["signals"][-1] == {"kind": "used_in_answers", "code": "used_in_answers", "params": {"answers": 2, "sources": ["golden"]}, "ref": ""}
    paid = next(m for m in out["measures"] if m["id"] == "measure:shop_claims.paid_amount")
    assert paid["usage"] == {"answers": 1, "sources": {"turns": 1}, "synthetic_only": False}
    assert next(c for c in out["concepts"] if c["id"] == "concept:value:make")["usage"]["answers"] == 2
    assert next(r for r in out["relations"] if r["id"] == "relation:shop.claims.dealer_id->shop.dealers")["usage"]["answers"] == 1
    assert meta == {"records": 4, "answered": 3, "sources": {"golden": 3, "turns": 1}, "matched": {"measures": 3, "dimensions": 2, "joins": 1, "terms": 0}, "unmatched": []}


def test_what_was_asked_for_and_is_not_in_the_model_is_counted_and_attached_to_nothing(reader: DuckDBReader, tmp_path: Path) -> None:
    model = induce(reader, **OPTIONS)
    before = json.dumps(model, sort_keys=True)
    records = read_usage([_log(tmp_path, [
        _record("q1", measures=[{"object": "shop.claims", "name": "repeat_rate"}], dimensions=[{"name": "paint", "columns": ["shop.vehicles.paint"]}],
                joins=[{"from": "shop.claims", "columns": ["owner_id"], "to": "shop.owners"}]),
        _record("q2", dimensions=[{"name": "paint", "columns": ["shop.vehicles.paint"]}]),
    ])])
    out, meta = attach(model, records)
    assert json.dumps(model, sort_keys=True) == before  # what was passed in is not changed
    assert not any("usage" in el for kind in ("concepts", "relations", "measures") for el in out[kind])
    assert meta["unmatched"] == [
        {"kind": "dimension", "name": "paint", "columns": ["shop.vehicles.paint"], "answers": 2},
        {"kind": "join", "name": "shop.claims(owner_id) -> shop.owners", "answers": 1},
        {"kind": "measure", "name": "shop.claims.repeat_rate", "answers": 1},
    ]


def test_use_changes_no_status_and_a_term_is_a_proposal_only_where_the_record_states_what_it_means(reader: DuckDBReader, tmp_path: Path) -> None:
    model = induce(reader, **OPTIONS)
    said = {"text": "marque", "element": "concept:value:make", "how": "asked back and answered"}
    known = {"text": "brand", "element": "concept:value:make", "how": "stated"}  # the metric view names it as a synonym already
    lost = {"text": "badge", "element": "concept:value:nothing", "how": "stated"}
    out, meta = attach(model, read_usage([_log(tmp_path, [_record("q1", measures=[CLAIMS], dimensions=[MAKE, STATUS], terms=[said, known]),
                                                         _record("q2", dimensions=[MAKE], terms=[said, lost])])]))
    assert {el["id"]: el["status"] for k in ("concepts", "relations", "measures") for el in out[k]} == {el["id"]: el["status"] for k in ("concepts", "relations", "measures") for el in model[k]}
    make = next(c for c in out["concepts"] if c["id"] == "concept:value:make")
    assert make["proposed_synonyms"] == [{"text": "marque", "answers": 2, "how": ["asked back and answered"]}]
    assert make["synonyms"] == next(c for c in model["concepts"] if c["id"] == "concept:value:make")["synonyms"]  # a proposal: the synonyms themselves are untouched
    assert "badge" not in json.dumps(out)  # a term for nothing we hold is attached to nothing, and counted
    assert meta["matched"]["terms"] == 3 and {"kind": "term", "name": "badge -> concept:value:nothing", "answers": 1} in meta["unmatched"]


def test_a_record_in_another_format_or_with_a_value_in_it_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="usage/1"):
        read_usage([_log(tmp_path, [{**_record("q1"), "format": "usage/0"}])])
    with pytest.raises(ValueError, match="values"):
        read_usage([_log(tmp_path, [_record("q1", dimensions=[{**MAKE, "values": ["HONDA"]}])])])


def test_a_dimension_that_is_an_entitys_key_or_a_column_of_one_object_counts_on_that_concept(reader: DuckDBReader, tmp_path: Path) -> None:
    model = induce(reader, **OPTIONS)
    by_key = {"name": "dealer", "columns": ["shop.claims.dealer_id", "shop.dealers.dealer_id", "shop.visits.dealer_id"]}  # the dealers' key, wherever it is carried
    one_object = {"name": "claim_status", "columns": ["shop.claims.status"]}  # a column of claims that is no value set
    out, meta = attach(model, read_usage([_log(tmp_path, [_record("q1", dimensions=[by_key, one_object]), _record("q2", dimensions=[by_key])])]))
    assert next(c for c in out["concepts"] if c["id"] == "concept:shop.dealers")["usage"]["answers"] == 2
    assert next(c for c in out["concepts"] if c["id"] == "concept:shop.claims")["usage"]["answers"] == 1
    assert meta["unmatched"] == [] and meta["matched"]["dimensions"] == 3

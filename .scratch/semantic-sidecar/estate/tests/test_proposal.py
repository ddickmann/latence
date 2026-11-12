"""The proposal file for a host's semantic layer (ticket 40): written from approved decisions
only, never from anything seeded, rejected, undecided or decided on another statement; the
same bytes for the same inputs; and the export says since which release of the host's library
an element is in the layer only where a receipt says it was placed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import jsonschema
import pytest
import yaml

from estate import cli
from estate.cli import build, propose
from estate.ledger import claim_of
from estate.proposal import proposal

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
FILE_SCHEMA = json.loads((SCHEMAS / "proposal-file.schema.json").read_text(encoding="utf-8"))
EXPORT_SCHEMA = json.loads((SCHEMAS / "graph-export-3.schema.json").read_text(encoding="utf-8"))

MAKERS = ("Alfa Works, Inc.", "Beta Motor Corp.", "Delt Cars", "Beta Motor of America", "Gama (GMA US LLC)", "Gama (GMA US, LLC)", "Alfa Works North")
ODD_NAMES = {7: 'Müller & Söhne: "Autohaus" #7', 8: "0123", 9: "Yes"}  # names a careless writer would let a reader take for something else
DEALERS = {f"D{k}": ODD_NAMES.get(k, f"Dealer {k} GmbH") for k in range(1, 31)}
ESTATE = {
    "name": "yard", "title": "Yard", "schemas": ["shop", "partner"], "resolve": ["maker", "market"],
    "seeded_objects": ["partner.accounts"],
    "link": {"duplicate_free": ["dealer master"], "labels": {"fit": 4, "test": 6}, "records": [
        {"object": "shop.dealers", "id": "dealer_id", "name": "name", "source": "dealer master"},
        {"object": "partner.accounts", "id": "account_id", "name": "account_name", "source": "partner portal (seeded)"}]},
    "propose": {"names": ["shop.dealers"]},
}


def _case(column: str, values: dict[int, str] | tuple[str, ...], modulo: int | None = None, otherwise: str | None = None) -> str:
    """A SQL CASE that gives row i the value listed for it (by i, or by i modulo)."""
    def quoted(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    pairs = values.items() if isinstance(values, dict) else enumerate(values)
    whens = " ".join(f"WHEN {k} THEN {quoted(v)}" for k, v in pairs)
    return f"CASE {'i' if modulo is None else f'i % {modulo}'} {whens}{f' ELSE {quoted(otherwise)}' if otherwise else ''} END AS {column}"


@pytest.fixture()
def yard(tmp_path: Path) -> dict[str, Any]:
    """A dealer master with names, a value set of makers in which three pairs of names may be
    the same (the first of each pair carries more rows), a value set of markets, and a seeded
    second source of portal accounts that shares the markets."""
    path = tmp_path / "yard.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA shop")
    con.execute("CREATE SCHEMA partner")
    names = {**{k: f"Dealer {k} GmbH" for k in range(1, 31)}, **ODD_NAMES}
    con.execute(f"CREATE TABLE shop.dealers AS SELECT 'D' || i AS dealer_id, {_case('name', names)}, "
                "CASE WHEN i % 3 = 0 THEN 'South' ELSE 'North Region' END AS market FROM range(1, 31) t(i)")
    con.execute(f"CREATE TABLE shop.recalls AS SELECT 'R' || i AS recall_ref, {_case('maker', MAKERS, 9, MAKERS[0])} FROM range(1, 401) t(i)")
    con.execute(f"CREATE TABLE shop.bulletins AS SELECT 'B' || i AS bulletin_ref, {_case('maker', MAKERS[:6], 6)} FROM range(1, 301) t(i)")
    con.execute(f"CREATE TABLE shop.claims AS SELECT 'C' || i AS claim_ref, 'D' || (1 + i % 30) AS dealer_id, {_case('maker', MAKERS[:3], 3)}, "
                "CASE WHEN i % 3 = 0 THEN 'South' ELSE 'North Region' END AS market FROM range(1, 601) t(i)")
    con.execute("CREATE TABLE partner.accounts AS SELECT 'PA' || i AS account_id, 'Portal account ' || i AS account_name, "
                "CASE WHEN i % 3 = 0 THEN 'South' ELSE 'North Area' END AS market FROM range(1, 46) t(i)")
    con.close()
    config = tmp_path / "yard.yaml"
    config.write_text(yaml.safe_dump(ESTATE), encoding="utf-8")
    return {"config": config, "duckdb_path": path, "views": None, "verified": None, "out": tmp_path / "out"}


def _export(yard: dict[str, Any]) -> dict[str, Any]:
    export: dict[str, Any] = json.loads((yard["out"] / "ontology.json").read_text(encoding="utf-8"))
    return export


def _pair(export: dict[str, Any], word: str) -> dict[str, Any]:
    pair: dict[str, Any] = next(r for r in export["review"] if r["a"].startswith("value:") and word in r["id"])
    return pair


def _concept(export: dict[str, Any], cid: str) -> dict[str, Any]:
    concept: dict[str, Any] = next(c for c in export["concepts"] if c["id"] == cid)
    return concept


def _line(el: dict[str, Any], verdict: str = "approved", by: str = "reviewer-1", at: str | None = "2026-11-12", claim: str | None = "") -> dict[str, Any]:
    """A line of the decisions file on an element of the export, as the page writes it, with the day added."""
    said = el["claim"] if claim == "" else claim
    return {"subject": el["id"], "verdict": verdict, "by": by, **({"claim": said} if said else {}), **({"at": at} if at else {})}


def _decide(yard: dict[str, Any], *lines: dict[str, Any], replace: bool = False) -> None:
    with open(yard["out"] / "decisions.jsonl", "w" if replace else "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _file(yard: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    text = (yard["out"] / "proposal.yaml").read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def _origin(el: dict[str, Any], by: str = "reviewer-1", at: str = "2026-11-12") -> dict[str, Any]:
    return {"element": el["id"], "claim": el["claim"], "decision": {"verdict": "approved", "by": by, "at": at}}


def test_an_approved_pair_and_an_approved_master_list_are_written_with_the_decision_they_rest_on(yard: dict[str, Any]) -> None:
    build(**yard)
    export = _export(yard)
    alfa, dealers = _pair(export, "Alfa"), _concept(export, "concept:shop.dealers")
    assert alfa["rows_a"] != alfa["rows_b"]  # one spelling carries more rows than the other
    _decide(yard, _line(alfa), _line(dealers, by="owner-of-the-list", at="2026-11-13T09:30:00+01:00"))
    report = propose(yard["config"], yard["out"])
    text, doc = _file(yard)
    jsonschema.validate(doc, FILE_SCHEMA)
    assert doc["source"] == "ontology" and len(doc["entries"]) == 2
    # The value kept is the one more rows carry; the object is the one that holds most of the value set, not the first by name.
    assert doc["entries"][0] == {"spellings": {"object": "shop.recalls", "column": "maker", "value": "Alfa Works, Inc.", "forms": ["Alfa Works North"]},
                                 "origin": _origin(alfa)}
    assert doc["entries"][1] == {"names": {"object": "shop.dealers", "column": "dealer_id", "values": DEALERS},
                                 "origin": _origin(dealers, by="owner-of-the-list", at="2026-11-13")}  # the day of a timestamp
    assert all(isinstance(v, str) for v in doc["entries"][1]["names"]["values"].values())  # "0123" and "Yes" are names, not a number and a truth value
    assert report["file"] == str(yard["out"] / "proposal.yaml")
    assert report["entries"] == [
        {"entry": 1, "kind": "spellings", "element": alfa["id"], "claim": alfa["claim"], "decided_by": "reviewer-1", "decided_on": "2026-11-12", "day_stated_by": "decision",
         "object": "shop.recalls", "column": "maker", "value": "Alfa Works, Inc.", "forms": ["Alfa Works North"]},
        {"entry": 2, "kind": "names", "element": "concept:shop.dealers", "claim": dealers["claim"], "decided_by": "owner-of-the-list", "decided_on": "2026-11-13", "day_stated_by": "decision",
         "object": "shop.dealers", "column": "dealer_id", "records": 30}]
    assert report["stale"] == [] and report["not_written"] == [] and report["not_carried"] == {}
    # The estate file may name the object a value set's spellings are stated on.
    yard["config"].write_text(yaml.safe_dump({**ESTATE, "propose": {"names": ["shop.dealers"], "spellings_on": {"maker": "shop.claims"}}}), encoding="utf-8")
    propose(yard["config"], yard["out"])
    assert _file(yard)[1]["entries"][0]["spellings"]["object"] == "shop.claims"
    yard["config"].write_text(yaml.safe_dump({**ESTATE, "propose": {"spellings_on": {"maker": "shop.dealers"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="shop.dealers does not hold the value set maker"):
        propose(yard["config"], yard["out"])


def test_a_rejected_or_undecided_element_is_in_no_proposal(yard: dict[str, Any]) -> None:
    build(**yard)
    export = _export(yard)
    alfa, beta, gama, dealers = _pair(export, "Alfa"), _pair(export, "Beta"), _pair(export, "Gama"), _concept(export, "concept:shop.dealers")
    assert gama["claim"]  # the third pair is listed and nobody decides it
    _decide(yard, _line(beta), _line(alfa), _line(dealers, verdict="rejected"), _line(beta, verdict="rejected"))  # the last decision on an element is the one in force
    for rebuilt in (False, True):  # the same whether or not a build has applied the decisions since
        if rebuilt:
            build(**yard)
        report = propose(yard["config"], yard["out"])
        text, doc = _file(yard)
        assert [e["origin"]["element"] for e in doc["entries"]] == [alfa["id"]]
        assert "Beta" not in text and "Gama" not in text and "Dealer" not in text and "names:" not in text
        assert [e["element"] for e in report["entries"]] == [alfa["id"]] and report["not_written"] == [] and report["stale"] == []


RESOLVED = {"kind": "name", "code": "name.same_core", "params": {}, "ref": ""}


def _resolver_answer() -> dict[str, Any]:
    """What the resolution engine would answer on the yard: three accounts linked to two
    dealers under a line certified on labels the seed's truth gave, one pair left for a person."""
    ev = [RESOLVED, {"kind": "source", "code": "source.two", "params": {"a": "dealer master", "b": "partner portal (seeded)"}, "ref": ""}]
    counted = {"reviewer": 0, "simulated": 3, "open": 0}
    return {
        "format": "resolution/3", "state": "certified", "records": 75, "sources": {"dealer master": 30, "partner portal (seeded)": 45}, "candidate_pairs": 200,
        "labels": {"training": counted, "accept_test": counted, "reject_test": counted, "total": 9}, "truth_stands_in": True, "certified_on": "simulated",
        "requests": {"training": [], "accept_test": [], "reject_test": []},
        "lines": {"accept_at": 0.6, "reject_below": 0.2, "target": 0.95, "delta": 0.05}, "certified": True,
        "duplicate_free": {"sources": ["dealer master"], "pairs_ruled_out": 0}, "counts": {"accept": 3, "review": 1, "reject": 196},
        "clusters": {"shop.dealers:D1": "shop.dealers:D1", "partner.accounts:PA1": "shop.dealers:D1", "partner.accounts:PA2": "shop.dealers:D1",
                     "shop.dealers:D2": "shop.dealers:D2", "partner.accounts:PA3": "shop.dealers:D2"},
        "accepted": [{"a": "shop.dealers:D1", "b": "partner.accounts:PA1", "p": 0.97, "evidence": ev}, {"a": "shop.dealers:D1", "b": "partner.accounts:PA2", "p": 0.91, "evidence": ev},
                     {"a": "shop.dealers:D2", "b": "partner.accounts:PA3", "p": 0.88, "evidence": ev}],
        "review": [{"a": "shop.dealers:D3", "b": "partner.accounts:PA4", "p": 0.55, "evidence": ev}], "review_total": 1,
    }


def test_a_seeded_record_and_a_link_certified_on_simulated_labels_are_in_no_proposal(yard: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_resolver", lambda *args, **kwargs: _resolver_answer())
    # The estate file even names the seeded list among those whose names may go: what is seeded stays out all the same.
    yard["config"].write_text(yaml.safe_dump({**ESTATE, "propose": {"names": ["shop.dealers", "partner.accounts"]}}), encoding="utf-8")
    build(**yard, resolver=tmp_path)
    export = _export(yard)
    links = [e for e in export["edges"] if e["kind"] == "same_as"]
    assert len(links) == 3 and export["meta"]["resolution"]["certified_on"] == "simulated" and export["meta"]["resolution"]["seeded"] is True
    accounts, dealers = _concept(export, "concept:partner.accounts"), _concept(export, "concept:shop.dealers")
    assert accounts["seeded"] is True and sum(n["concept"] == accounts["id"] for n in export["nodes"]) == 45  # the seeded records are in the build
    left = next(r for r in export["review"] if r["id"] == "same:shop.dealers:D3~partner.accounts:PA4")  # a dealer and a seeded account, left for a person
    north = next(r for r in export["review"] if r["id"].startswith("same:market:"))  # two names of a market, one of them only in the seeded source
    _decide(yard, _line(accounts), _line(dealers), _line(left), _line(north))  # a person approves all four
    seen = []
    for rebuilt in (False, True):
        if rebuilt:
            build(**yard, resolver=tmp_path)
            assert any(e["kind"] == "same_as" and e.get("decided_by") == "reviewer-1" for e in _export(yard)["edges"])  # the person's link is in the build
        report = propose(yard["config"], yard["out"])
        text, doc = _file(yard)
        jsonschema.validate(doc, FILE_SCHEMA)
        assert [e["origin"]["element"] for e in doc["entries"]] == ["concept:shop.dealers"]
        assert doc["entries"][0]["names"]["values"] == DEALERS  # each dealer by its own name, no linked account's name beside it
        assert "Portal account" not in text and "partner.accounts" not in text and "PA1" not in text and "North Area" not in text
        said = json.dumps(doc, ensure_ascii=False)  # the entries, without the file's opening comment
        assert "seeded" not in said and "certified_on" not in said and "simulated" not in said  # nothing that would need such a mark is written
        assert report["not_written"] == [{"subject": "concept:partner.accounts", "why": "seeded"}, {"subject": north["id"], "why": "seeded"},
                                         {"subject": left["id"], "why": "seeded"}]
        seen.append(text)
    assert seen[0] == seen[1]


def test_an_entry_whose_decisions_claim_no_longer_matches_the_build_is_listed_as_stale_and_not_written(yard: dict[str, Any]) -> None:
    build(**yard)
    export = _export(yard)
    alfa, beta, dealers = _pair(export, "Alfa"), _pair(export, "Beta"), _concept(export, "concept:shop.dealers")
    # The list was approved when it was keyed on another column; the pair's line carries the claim of another element.
    then = claim_of("concept", {**dealers, "key": ["dealer_no"]})
    assert then != dealers["claim"]
    _decide(yard, _line(dealers, claim=then), _line(alfa, claim=beta["claim"]), _line(beta))
    for rebuilt in (False, True):
        if rebuilt:
            applied: Any = build(**yard)["decisions"]
            assert [s["subject"] for s in applied["stale"]] == ["concept:shop.dealers", alfa["id"]]  # the build says the same
        report = propose(yard["config"], yard["out"])
        text, doc = _file(yard)
        assert [e["origin"]["element"] for e in doc["entries"]] == [beta["id"]]
        assert "Alfa" not in text and "Dealer" not in text
        assert report["stale"] == [{"subject": "concept:shop.dealers", "decided_on": then, "now": dealers["claim"]},
                                   {"subject": alfa["id"], "decided_on": beta["claim"], "now": alfa["claim"]}]
        assert report["not_written"] == []


def test_two_runs_on_the_same_inputs_give_the_same_bytes_whatever_order_the_decisions_were_written_in(yard: dict[str, Any]) -> None:
    build(**yard)
    export = _export(yard)
    lines = [_line(_concept(export, "concept:shop.dealers")), _line(_pair(export, "Gama")), _line(_pair(export, "Alfa")), _line(_pair(export, "Beta"))]
    _decide(yard, *lines)
    propose(yard["config"], yard["out"])
    first = (yard["out"] / "proposal.yaml").read_bytes()
    assert [next(iter(e)) for e in _file(yard)[1]["entries"]] == ["spellings", "spellings", "spellings", "names"]
    propose(yard["config"], yard["out"])
    assert (yard["out"] / "proposal.yaml").read_bytes() == first
    _decide(yard, *reversed(lines), replace=True)
    propose(yard["config"], yard["out"])
    assert (yard["out"] / "proposal.yaml").read_bytes() == first
    build(**yard)  # the build makes each pair one value; the file says what it said before
    propose(yard["config"], yard["out"])
    assert (yard["out"] / "proposal.yaml").read_bytes() == first
    assert first.endswith(b"\n") and b"\r" not in first and first.decode("utf-8")  # one line ending, UTF-8


def test_a_kept_value_never_rests_on_a_decision_that_is_not_written() -> None:
    def pair(a: str, b: str, rows_a: int, rows_b: int, became: str) -> dict[str, Any]:
        el = {"id": f"same:maker:{a}~{b}", "kind": "possibly_same", "concept": "concept:value:maker", "a": f"value:maker:{a}", "b": f"value:maker:{b}",
              "rows_a": rows_a, "rows_b": rows_b, "merged_into": f"value:maker:{became}"}
        return {**el, "claim": claim_of("possibly_same", el)}

    # Three spellings, two approved pairs. A build that applied both made all three one value: the third name, which most rows carry.
    first, second = pair("Ab Cars", "Ab Cars Inc", 5, 4, "Ab Cars International"), pair("Ab Cars Inc", "Ab Cars International", 4, 20, "Ab Cars International")
    export = {"concepts": [{"id": "concept:value:maker", "kind": "reference", "column": "maker", "objects": ["s.a", "s.b"], "claim": "0" * 16}],
              "relations": [], "measures": [], "nodes": [], "review": [first, second]}
    decisions = {first["id"]: _line(first), second["id"]: _line(second, at=None)}  # the second decision states no day
    document, report = proposal(export, decisions)
    # Only the first pair is written, so its value is one of its own two names: nothing in the file says the third is the same.
    assert [e["spellings"] for e in document["entries"]] == [{"object": "s.a", "column": "maker", "value": "Ab Cars", "forms": ["Ab Cars Inc"]}]
    assert "International" not in json.dumps(document) and report["not_written"] == [{"subject": second["id"], "why": "no_day"}]
    # With both written, both say what the build says.
    document, _ = proposal(export, decisions, decided_on="2026-11-12")
    assert [(e["spellings"]["value"], e["spellings"]["forms"]) for e in document["entries"]] == [
        ("Ab Cars International", ["Ab Cars Inc"]), ("Ab Cars International", ["Ab Cars", "Ab Cars Inc"])]


def test_a_decision_that_states_no_day_or_no_claim_is_not_written_and_a_day_is_never_made_up(yard: dict[str, Any]) -> None:
    build(**yard)
    export = _export(yard)
    alfa, dealers = _pair(export, "Alfa"), _concept(export, "concept:shop.dealers")
    _decide(yard, _line(alfa, at=None), _line(dealers, claim=None))  # as the page writes a line today, and as a line written by hand may be
    report = propose(yard["config"], yard["out"])
    assert _file(yard)[1] == {"source": "ontology", "entries": []}
    assert report["entries"] == [] and report["not_written"] == [{"subject": "concept:shop.dealers", "why": "no_claim"}, {"subject": alfa["id"], "why": "no_day"}]
    # Whoever runs the command may state the day for the decisions that state none; the report says where the day came from.
    report = propose(yard["config"], yard["out"], decided_on="2026-11-14")
    assert _file(yard)[1]["entries"][0]["origin"] == _origin(alfa, at="2026-11-14")
    assert report["entries"][0]["day_stated_by"] == "command" and report["not_written"] == [{"subject": "concept:shop.dealers", "why": "no_claim"}]
    with pytest.raises(ValueError, match="no ISO date"):
        propose(yard["config"], yard["out"], decided_on="14 Nov")
    _decide(yard, _line(alfa, at="yesterday"))
    with pytest.raises(ValueError, match="no ISO date"):
        propose(yard["config"], yard["out"])


def test_what_the_estate_file_does_not_name_and_what_the_file_has_no_entry_for_is_counted_and_not_written(yard: dict[str, Any]) -> None:
    yard["config"].write_text(yaml.safe_dump({k: v for k, v in ESTATE.items() if k != "propose"}), encoding="utf-8")
    build(**yard)
    export = _export(yard)
    dealers, relation = _concept(export, "concept:shop.dealers"), next(r for r in export["relations"] if r["status"] == "proposed")
    _decide(yard, _line(dealers), _line(relation), {"subject": "concept:shop.gone", "verdict": "approved", "by": "reviewer-1", "claim": "0" * 16, "at": "2026-11-12"})
    report = propose(yard["config"], yard["out"])
    assert _file(yard)[1]["entries"] == []  # a reviewer's approval of a concept does not by itself send its rows' names anywhere
    assert report["not_carried"] == {"concept": 1, "relation": 1} and report["not_written"] == [{"subject": "concept:shop.gone", "why": "not_in_the_build"}]


def _receipt(path: Path, version: str, placed: list[tuple[dict[str, Any], str | None]], already: tuple[dict[str, Any], ...] = (),
             refused: tuple[dict[str, Any], ...] = (), fmt: str = "layer-receipt/1") -> Path:
    def taken(el: dict[str, Any], claim: str | None = None) -> dict[str, Any]:
        return {"entry": "proposal.yaml#1", "kind": "spellings", "element": el["id"], "claim": claim or el["claim"], "dimension": "maker", "node": "semantics.dimensions.maker.values"}

    path.write_text(json.dumps({"format": fmt, "library": "yard-semantics", "version": version, "placed": [taken(el, c) for el, c in placed],
                                "already_there": [taken(el) for el in already], "refused": [{**taken(el), "reason": "no dimension reads it"} for el in refused]}), encoding="utf-8")
    return path


def test_the_export_says_since_which_release_an_element_is_in_the_layer_only_where_a_receipt_says_it_was_placed(yard: dict[str, Any], tmp_path: Path) -> None:
    build(**yard)
    export = _export(yard)
    alfa, beta, gama, dealers = _pair(export, "Alfa"), _pair(export, "Beta"), _pair(export, "Gama"), _concept(export, "concept:shop.dealers")
    _decide(yard, _line(alfa), _line(beta), _line(gama), _line(dealers))
    build(**yard)
    assert '"in_layer"' not in (yard["out"] / "ontology.json").read_text(encoding="utf-8")  # no receipt: the field is absent, never guessed

    def taken() -> dict[str, Any]:
        after = _export(yard)
        jsonschema.validate(after, EXPORT_SCHEMA)
        return {el["id"]: el["in_layer"] for el in after["concepts"] + after["review"] if "in_layer" in el}

    # Placed: the Alfa pair, and the dealers on a statement the list no longer makes. Already there: the Gama pair. Refused: the Beta pair.
    first = _receipt(tmp_path / "r1.json", "1.1.0", [(alfa, None), (dealers, "0" * 16)], already=(gama,), refused=(beta,))
    summary = build(**yard, receipts=(first,))
    assert taken() == {alfa["id"]: {"library": "yard-semantics", "version": "1.1.0"}} and summary["in_layer"] == 1
    # A later release placed the dealers and holds the Alfa pair still: the pair is in the layer since the first.
    second = _receipt(tmp_path / "r2.json", "1.2.0", [(dealers, None), (alfa, None)])
    build(**yard, receipts=(first, second))
    assert taken() == {alfa["id"]: {"library": "yard-semantics", "version": "1.1.0"}, "concept:shop.dealers": {"library": "yard-semantics", "version": "1.2.0"}}
    # The approval is withdrawn: the element no longer says it is in the layer.
    _decide(yard, _line(alfa, verdict="rejected"))
    build(**yard, receipts=(first, second))
    assert taken() == {"concept:shop.dealers": {"library": "yard-semantics", "version": "1.2.0"}}
    with pytest.raises(ValueError, match="layer-receipt/1"):
        build(**yard, receipts=(_receipt(tmp_path / "r3.json", "1.3.0", [(dealers, None)], fmt="layer-receipt/2"),))


def test_the_command_line_writes_the_file_and_prints_what_it_wrote(yard: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    build(**yard)
    _decide(yard, _line(_pair(_export(yard), "Alfa"), at=None))
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["estate", "propose", "--config", str(yard["config"]), "--out", str(yard["out"]), "--file", str(tmp_path / "for-the-host.yaml"),
                                      "--decided-on", "2026-11-12"])
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["file"] == str(tmp_path / "for-the-host.yaml") and [e["kind"] for e in printed["entries"]] == ["spellings"]
    assert yaml.safe_load((tmp_path / "for-the-host.yaml").read_text(encoding="utf-8"))["entries"][0]["origin"]["decision"]["at"] == "2026-11-12"
    with pytest.raises(SystemExit, match="no build in"):
        propose(yard["config"], tmp_path / "nothing-built")

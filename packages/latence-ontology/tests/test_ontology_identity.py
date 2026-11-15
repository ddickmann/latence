"""Persistent IRIs, the crosswalk to a host's concept keys, and labels per language (ticket 57).

The alias table here is synthetic, in the shape the host's alias table states itself
(``{namespace, keys, platform, renamed}``, the host's 297): its platform ids name the toy
estate's metric view ``semantics.shop_claims`` as an import of it would."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import jsonschema
import pytest
import rdflib
from _ontology_toy import Project
from latence_ontology import SCHEMAS
from latence_ontology.cli import build
from latence_ontology.iris import Iris, confirm, minted

SCHEMA = json.loads((SCHEMAS / "graph-export-3.schema.json").read_text(encoding="utf-8"))
SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

TABLE: dict[str, Any] = {
    "namespace": "shop",
    "keys": {
        "dimension:make": "dimension:shop.make",
        "dimension:market": "dimension:shop.market",
        "measure:claim_count": "measure:shop.claim_count",
    },
    "platform": {
        "dimension:make": ["dimension:demo.semantics.shop_claims.make"],
        "measure:claim_count": ["measure:demo.semantics.shop_claims.claim_count"],
    },
    "renamed": {},
}


def _export(project: Project) -> dict[str, Any]:
    export: dict[str, Any] = json.loads((project["out"] / "ontology.json").read_text())
    return export


def _iris(export: dict[str, Any]) -> dict[str, str]:
    return {
        el["id"]: el["iri"]
        for part in ("concepts", "relations", "measures")
        for el in export[part]
    }


def _rename_dealers(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute("ALTER TABLE shop.dealers RENAME TO dealer_master")
    con.close()


@pytest.fixture()
def renaming(project: Project, tmp_path: Path) -> Project:
    copy = tmp_path / "renaming.duckdb"
    copy.write_bytes(project["duckdb_path"].read_bytes())
    changed: Project = {**project, "duckdb_path": copy}
    return changed


def test_every_element_has_an_iri_kept_in_the_ledger_and_a_rebuild_keeps_it(
    project: Project,
) -> None:
    build(**project)
    first = _export(project)
    jsonschema.validate(first, SCHEMA)
    iris = _iris(first)
    assert iris and all(i.startswith("urn:estate:toy:x") for i in iris.values())
    assert len(set(iris.values())) == len(iris)  # never one IRI for two elements
    ledger = (project["out"] / "iris.jsonl").read_text().splitlines()
    held = {json.loads(line)["element"]: json.loads(line)["iri"] for line in ledger}
    assert all(held[i] == iri.removeprefix("urn:estate:toy:") for i, iri in iris.items())
    build(**project)
    assert _iris(_export(project)) == iris
    assert (project["out"] / "iris.jsonl").read_text().splitlines() == ledger  # nothing minted


def test_a_rename_no_person_confirmed_gets_a_new_iri(renaming: Project) -> None:
    build(**renaming)
    before = _iris(_export(renaming))
    _rename_dealers(renaming["duckdb_path"])
    build(**renaming)
    after = _iris(_export(renaming))
    assert "concept:shop.dealer_master" in after and "concept:shop.dealers" not in after
    assert after["concept:shop.dealer_master"] != before["concept:shop.dealers"]


def test_a_rename_a_person_confirmed_keeps_every_iri(renaming: Project) -> None:
    build(**renaming)
    before = _iris(_export(renaming))
    old_turtle = (renaming["out"] / "ontology.ttl").read_text()
    _rename_dealers(renaming["duckdb_path"])
    confirm(renaming["out"], "concept:shop.dealers", "concept:shop.dealer_master", "steward-1",
            "2026-11-15", "the table was renamed")
    build(**renaming)
    export = _export(renaming)
    after = _iris(export)
    assert after["concept:shop.dealer_master"] == before["concept:shop.dealers"]
    # The relations into the renamed table keep theirs too.
    moved = {i: i.replace("shop.dealers", "shop.dealer_master") for i in before if "shop.dealers" in i}
    assert len(moved) >= 2
    assert all(after[new] == before[old] for old, new in moved.items() if new in after)
    assert export["meta"]["renames"]["refused"] == []
    carried = {c["from"]: c["to"] for c in export["meta"]["renames"]["carried"]}
    assert carried["concept:shop.dealers"] == "concept:shop.dealer_master"
    # The Turtle file names the renamed table by its old IRI, the attributes included.
    local = before["concept:shop.dealers"].removeprefix("urn:estate:toy:")
    assert f"e:{local} a owl:Class" in (renaming["out"] / "ontology.ttl").read_text()
    assert f"e:{local} a owl:Class" in old_turtle
    # A rebuild carries nothing twice and keeps them.
    lines = (renaming["out"] / "iris.jsonl").read_text()
    build(**renaming)
    assert _iris(_export(renaming)) == after
    assert (renaming["out"] / "iris.jsonl").read_text() == lines
    assert "renames" not in _export(renaming)["meta"]


def test_a_rename_is_refused_where_the_old_element_is_still_there(project: Project) -> None:
    build(**project)
    before = _iris(_export(project))
    confirm(project["out"], "concept:shop.dealers", "concept:shop.parts", "steward-1", "2026-11-15")
    build(**project)
    export = _export(project)
    assert _iris(export) == before
    assert export["meta"]["renames"]["refused"] == [
        {"from": "concept:shop.dealers", "to": "concept:shop.parts",
         "reason": "concept:shop.dealers is still in the build"}
    ]


def test_a_minted_name_another_element_holds_is_minted_again(tmp_path: Path) -> None:
    ledger = Iris(tmp_path)
    ledger.settle(["concept:a.old"], [])
    ledger.save()
    held = minted("concept:a.old")
    renames = [{"from": "concept:a.old", "to": "concept:a.new", "by": "p", "at": "2026-11-15"}]
    again = Iris(tmp_path)
    note = again.settle(["concept:a.new"], renames)
    again.save()
    assert note["carried"] == [{"from": "concept:a.old", "to": "concept:a.new", "iri": held}]
    # The old id comes back as another element: it may not take the IRI the renamed one holds.
    third = Iris(tmp_path)
    third.settle(["concept:a.new", "concept:a.old"], renames)
    assert third.of["concept:a.new"] == held
    assert third.of["concept:a.old"] == minted("concept:a.old", 1) != held


def test_the_hosts_key_comes_from_its_alias_table_and_is_never_guessed(
    project: Project, tmp_path: Path
) -> None:
    table = tmp_path / "alias.json"
    table.write_text(json.dumps(TABLE))
    build(**project, alias_tables=(table,))
    export = _export(project)
    jsonschema.validate(export, SCHEMA)
    concepts = {c["id"]: c for c in export["concepts"]}
    measures = {m["id"]: m for m in export["measures"]}
    make = concepts["concept:value:make"]["match"]
    assert make["key"] == "dimension:shop.make" and make["members"] == "code"
    assert make["namespace"] == "shop" and len(make["read_from"]) == 16
    assert measures["measure:shop_claims.claim_count"]["match"]["key"] == "measure:shop.claim_count"
    # `market` is a key of the table and a value set of the estate, by the same name, but no
    # platform id names it: no key.
    assert "concept:value:market" in concepts and "match" not in concepts["concept:value:market"]
    assert "match" not in measures["measure:shop_claims.paid_amount"]
    assert not any("match" in c for c in export["concepts"] if c["kind"] != "reference")
    note = export["meta"]["crosswalk"]
    assert note["matched"] == 2 and note["several_keys"] == []
    assert "measure:shop_claims.paid_amount" in note["without_key"]
    # The Turtle file: the value set is exactly the key, each value names its member key.
    graph = rdflib.Graph().parse(data=(project["out"] / "ontology.ttl").read_text(), format="turtle")
    key = rdflib.URIRef("urn:concept-key:dimension:shop.make")
    assert len(list(graph.subjects(SKOS.exactMatch, key))) == 1
    vocab = rdflib.Namespace("urn:estate:toy:vocabulary/")
    codes = {str(graph.value(m, vocab.code)) for m in graph.objects(None, vocab.member)
             if graph.value(m, vocab.dimension) == key}
    assert codes == {"ALFA", "BETA", "GAMA", "DELT"}


def test_two_keys_for_one_element_give_it_none(project: Project, tmp_path: Path) -> None:
    other = {**TABLE, "namespace": "sales", "keys": {"dimension:brand": "dimension:sales.brand"},
             "platform": {"dimension:brand": ["dimension:prod.semantics.shop_claims.make"]}}
    paths = []
    for n, t in enumerate((TABLE, other)):
        paths.append(tmp_path / f"alias{n}.json")
        paths[-1].write_text(json.dumps(t))
    build(**project, alias_tables=tuple(paths))
    export = _export(project)
    make = next(c for c in export["concepts"] if c["id"] == "concept:value:make")
    assert "match" not in make
    assert export["meta"]["crosswalk"]["several_keys"] == [
        {"element": "concept:value:make", "keys": ["dimension:sales.brand", "dimension:shop.make"]}
    ]


def test_labels_carry_their_language_in_the_export_and_the_turtle_file(project: Project) -> None:
    build(**project)
    export = _export(project)
    make = next(c for c in export["concepts"] if c["id"] == "concept:value:make")
    assert make["labels"]["pref"] == [{"text": make["label"], "lang": "en", "origin": "engine"}]
    assert make["labels"]["alt"] == [{"text": "brand", "lang": "en", "origin": "catalog"}]
    assert all(len(c["labels"]["pref"]) == 1 for c in export["concepts"])
    graph = rdflib.Graph().parse(data=(project["out"] / "ontology.ttl").read_text(), format="turtle")
    alts = {(str(o), o.language) for o in graph.objects(None, SKOS.altLabel) if isinstance(o, rdflib.Literal)}
    assert ("brand", "en") in alts
    prefs = list(graph.objects(None, SKOS.prefLabel))
    assert rdflib.Literal(make["label"], lang="en") in prefs

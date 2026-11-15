"""The crosswalk read from the host's real alias table (ticket 57, the host's ticket 326).

`fixtures/alias/estate-semantics.json` is the table the host's command `metis-dev concepts
alias` writes for the vehicle estate's library, in its published format ``metis.alias/1``
(where it was copied from: `fixtures/alias/SOURCE.md`). Its platform ids are
``<kind>:<source>.<view>.<field>``: the host's name for the catalog source in its layer, the
metric view's name and the field's name.

The offline tests hold the reader and the crosswalk to that table on the vehicle estate's own
view names; the build test runs the whole vehicle estate and needs its mirror
(``LATENCE_ONTOLOGY_VEHICLE_INPUTS``, as in the same-bytes test)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import rdflib
import yaml
from latence_ontology import SCHEMAS
from latence_ontology.cli import build, seed
from latence_ontology.crosswalk import crosswalk, read_alias_tables
from latence_ontology.reader import Field, MetricView
from test_ontology_host_guard import what_a_host_refuses

TABLE_FILE = Path(__file__).resolve().parent / "fixtures" / "alias" / "estate-semantics.json"
HOSTS: dict[str, Any] = json.loads(TABLE_FILE.read_text(encoding="utf-8"))
ESTATE = Path(__file__).resolve().parents[3] / ".scratch" / "semantic-sidecar" / "estate" / "estates"
SCHEMA = json.loads((SCHEMAS / "graph-export-3.schema.json").read_text(encoding="utf-8"))
SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")


def _keys_by_platform_id() -> dict[str, str]:
    """The table's crosswalk, read here from the fixture as the host states it."""
    return {
        pid: HOSTS["keys"][bare]
        for bare, ids in HOSTS["platform"].items()
        if bare in HOSTS["keys"]
        for pid in ids
    }


def _written(tmp_path: Path, document: dict[str, Any], name: str = "alias.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_the_hosts_table_is_read_with_its_version_and_catalog() -> None:
    [table] = read_alias_tables([TABLE_FILE])
    assert table["format"] == "metis.alias/1" and table["namespace"] == "estate"
    assert table["version"] == HOSTS["version"] and table["catalog"] == HOSTS["catalog"]


def test_a_table_of_another_format_is_refused(tmp_path: Path) -> None:
    other = _written(tmp_path, {**HOSTS, "format": "metis.alias/2"})
    with pytest.raises(ValueError, match=r"metis\.alias/2.*metis\.alias/1"):
        read_alias_tables([other])
    # A table that states no format (the shape this side read before the host wrote one).
    unstated = {k: v for k, v in HOSTS.items() if k not in ("format", "catalog", "version")}
    with pytest.raises(ValueError, match=r"metis\.alias/1"):
        read_alias_tables([_written(tmp_path, unstated, "unstated.json")])


def test_a_table_whose_version_is_not_its_digest_is_refused(tmp_path: Path) -> None:
    changed = copy.deepcopy(HOSTS)
    changed["keys"]["dimension:make"] = "dimension:estate.maker"  # the version stays the old one
    with pytest.raises(ValueError, match="version"):
        read_alias_tables([_written(tmp_path, changed)])


def _views() -> list[MetricView]:
    """Two of the vehicle estate's metric views, as its listing names them."""
    make = Field("make", "source.make")
    return [
        MetricView(
            "clean_complaints", "nhtsa_clean.complaints", dimensions=(make,),
            measures=(Field("complaint_count", "COUNT(1)"), Field("complaint_rate", "AVG(1)")),
            schema="nhtsa_semantics",
        ),
        MetricView(
            "internal_claims", "nhtsa_internal.claims", dimensions=(make,),
            measures=(Field("claim_count", "COUNT(1)"),
                      Field("claimed_vehicle_count", "COUNT(DISTINCT source.vin)")),
            schema="nhtsa_semantics",
        ),
    ]


def test_the_vehicle_crosswalk_names_the_hosts_keys_and_reports_what_it_does_not_name() -> None:
    export = {
        "concepts": [
            {"id": "concept:value:make", "kind": "reference", "column": "make",
             "objects": ["nhtsa_clean.complaints", "nhtsa_internal.claims"]},
        ],
        "measures": [
            {"id": f"measure:{v}.{m}", "view": v, "name": m}
            for v, m in (("clean_complaints", "complaint_count"), ("clean_complaints", "complaint_rate"),
                         ("internal_claims", "claim_count"), ("internal_claims", "claimed_vehicle_count"))
        ],
    }
    matches, note = crosswalk(export, _views(), read_alias_tables([TABLE_FILE]))
    # (a) an element with a host key names exactly the key the table gives its platform id.
    by_platform = _keys_by_platform_id()
    assert matches["measure:clean_complaints.complaint_count"]["key"] == by_platform[
        "measure:lakehouse.clean_complaints.complaint_count"] == "measure:estate.complaint_count"
    assert matches["measure:internal_claims.claim_count"]["key"] == "measure:estate.claim_count"
    # (b) the value set is the dimension's key (its members are that key and a code).
    assert matches["concept:value:make"]["key"] == "dimension:estate.make"
    assert {m["namespace"] for m in matches.values()} == {"estate"}
    # (c) the table's version and catalog are recorded.
    assert note["tables"] == [{
        "format": "metis.alias/1", "namespace": "estate", "version": HOSTS["version"],
        "catalog": HOSTS["catalog"], "read_from": matches["concept:value:make"]["read_from"],
    }]
    # (d) not named by a platform id: no key, and reported. `claimed_vehicle_count` is a key of
    # the table by the same name, but no platform id names it: never guessed.
    assert "measure:claimed_vehicle_count" in HOSTS["keys"]
    assert "measure:claimed_vehicle_count" not in HOSTS["platform"]
    assert note["without_key"] == [
        "measure:clean_complaints.complaint_rate", "measure:internal_claims.claimed_vehicle_count"]
    assert note["matched"] == 3 and note["several_keys"] == []


def test_a_view_name_two_schemas_hold_names_no_key() -> None:
    # The platform id names a view by its name only: where the estate holds two views of that
    # name, it cannot say which, and neither is given a key.
    one, _ = _views()
    twin = MetricView(one.name, "partner.complaints", dimensions=one.dimensions,
                      measures=one.measures, schema="partner_semantics")
    export = {"concepts": [], "measures": [
        {"id": "measure:clean_complaints.complaint_count", "view": "clean_complaints",
         "name": "complaint_count"}]}
    matches, note = crosswalk(export, [one, twin], read_alias_tables([TABLE_FILE]))
    assert matches == {}
    assert note["views_named_twice"] == ["clean_complaints"]
    assert note["without_key"] == ["measure:clean_complaints.complaint_count"]


def test_the_vehicle_build_takes_its_keys_from_the_hosts_table(tmp_path: Path) -> None:
    given = os.environ.get("LATENCE_ONTOLOGY_VEHICLE_INPUTS")
    if not given:
        pytest.skip("the vehicle estate's mirror is written by the host's tools and is not in this "
                    "repository: set LATENCE_ONTOLOGY_VEHICLE_INPUTS to run this check")
    inputs, out = Path(given), tmp_path / "vehicles"
    config = ESTATE / "vehicles.yaml"
    seed(config, inputs / "vehicles.duckdb", tmp_path / "seeded.duckdb", out, 11)
    build(config=config, duckdb_path=tmp_path / "seeded.duckdb", views=inputs / "metric_views.txt",
          verified=inputs / "catalog.yaml", out=out,
          class_answer=ESTATE / "vehicles.classes_answer.gpt-6-luna.json",
          usage=(inputs / "usage.jsonl",), alias_tables=(TABLE_FILE,))
    export = json.loads((out / "ontology.json").read_text())
    jsonschema.validate(export, SCHEMA)
    assert what_a_host_refuses(export) == []  # the host's guard (its 238), as this side states it
    by_platform = _keys_by_platform_id()
    measures = [m for m in export["measures"] if "match" in m]
    assert measures, "no measure of the vehicle estate is named by the host's table"
    for m in measures:  # (a)
        assert m["match"]["key"] == by_platform[f"measure:lakehouse.{m['view']}.{m['name']}"]
    # (b) a member is the pair (dimension key, code).
    make = next(c for c in export["concepts"] if c["id"] == "concept:value:make")
    assert make["match"]["key"] == "dimension:estate.make" and make["match"]["members"] == "code"
    graph = rdflib.Graph().parse(data=(out / "ontology.ttl").read_text(), format="turtle")
    key = rdflib.URIRef(str(yaml.safe_load(config.read_text())["export"].get(
        "key_iri", "urn:concept-key:")) + "dimension:estate.make")
    assert len(list(graph.subjects(SKOS.exactMatch, key))) == 1
    members = [m for m in graph.objects(None, None) if isinstance(m, rdflib.BNode)
               and any(o == key for o in graph.objects(m, None))]
    assert members and all(
        any(isinstance(o, rdflib.Literal) for o in graph.objects(m, None)) for m in members)
    # (c) the table's version is recorded in the export.
    [table] = export["meta"]["crosswalk"]["tables"]
    assert (table["version"], table["catalog"]) == (HOSTS["version"], HOSTS["catalog"])
    # (d) a measure or value set the table does not name has no key and is reported: here the
    # value sets no view's dimension reads, and the one two keys claim (`kind`: a campaign's
    # and a claim line's), which is given neither.
    note = export["meta"]["crosswalk"]
    nameable = {e["id"]: e for e in [*export["measures"], *export["concepts"]]
                if e["id"].startswith("measure:") or e.get("kind") == "reference"}
    reported = {*note["without_key"], *(s["element"] for s in note["several_keys"])}
    assert reported == {i for i, e in nameable.items() if "match" not in e}
    assert "concept:value:kind" in {s["element"] for s in note["several_keys"]}
    assert note["without_key"], "the vehicle estate has value sets no view reads"
    assert note["matched"] == sum("match" in e for e in nameable.values())
    unnamed = {rdflib.URIRef(str(e["iri"])) for i, e in nameable.items() if i in reported}
    assert not any(graph.value(u, SKOS.exactMatch) for u in unnamed)

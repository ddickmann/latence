"""The move into this package changed no byte (ticket 56), and what changed since is only what
was meant to. ``PROTOTYPE`` holds the digests recorded from the estate prototype
(`.scratch/semantic-sidecar/estate/estate/`) at the commit before the move; ``TOY`` and
``VEHICLES`` the digests of the files as they are now, re-recorded by the ticket that changed
them: 57 added ``iri``, ``labels`` and ``match`` to the export and labels to the Turtle file, 58
the labels' claims, and nothing else (the export without those three fields is still the prototype's bytes, which the
toy estate's test checks on every run). One change to the vehicle estate's input since: the
seeded partner accounts' English service-department name tail became "Kundendienst" (DE)
and "Servicing" (GB) (ticket 58), which changes 152 account labels of the export and nothing else (compared
line by line when it was made); its prototype digest is re-recorded for that seed, the
prototype's own beside it. Then R5 (decision D-2 (a), graph-export/4): every ``broader`` edge
gained ``member_rows`` and the export's ``schema`` became ``graph-export/4``; the toy estate's
``ontology.json`` digest is re-recorded for that and its three Turtle files did not change. The
comparison with the prototype drops ``member_rows`` and states the prototype's version string,
so it still proves nothing else moved. The vehicle estate's ``ontology.json`` digest below is
still the graph-export/3 one: its inputs are not in this repository, so it is re-recorded where
they are (the host), and that test fails until it is. The toy
estate is built here on every run. The vehicle estate needs the host's mirror, its metric views,
its usage log and its verified joins, which this repository does not hold, so its test runs when
``LATENCE_ONTOLOGY_VEHICLE_INPUTS`` names a folder with ``vehicles.duckdb``,
``metric_views.txt``, ``usage.jsonl`` and ``catalog.yaml`` (the commands that write them are in
the prototype folder's ``HANDOFF.md``) and is skipped with that reason otherwise."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from _ontology_toy import OPTIONS, SENSITIVE, VIEWS
from latence_ontology.cli import build, seed

# The digests are SHA-256 of build outputs, not secrets; the secret scan reads any long hex
# string as one, hence the allowlist mark on each.
FILES = ("ontology.json", "ontology.ttl", "ontology.shacl.ttl", "ontology.prov.ttl")

PROTOTYPE = {
    "toy": "b284d1c462f5de81fdd237e853e2f487182747374b2ebb2fb963dc827ab45b57",  # pragma: allowlist secret
    # The prototype's own was 05163c07...525f, with the seed's earlier name tails (ticket 58).
    "vehicles": "b0067ef0c76435d9270ae4f95b18bc8d90ef29e3f83fb7e1bb21dedba59eb8f7",  # pragma: allowlist secret
}
"""The export (``ontology.json``) of each estate as the prototype wrote it."""

ADDED_SINCE = ("iri", "labels", "match", "member_rows")  # ticket 57; R5 (graph-export/4)
PROTOTYPE_SCHEMA = "graph-export/3"  # the version string the prototype's bytes state

TOY = {
    "ontology.json": "8b502124fbfbb77cc40f64e7d40ccf14e3a6de95651e2427938704b5ec4caa52",  # pragma: allowlist secret
    "ontology.ttl": "31fa82f79d7864c4166fcc8f8eafb25eefafb277720fd8c270f85b9b25a5417b",  # pragma: allowlist secret
    "ontology.shacl.ttl": "45c9e8aeaa75c1032d3a0ea5362950257182daeb90a2fbde8182826e17ce59d9",  # pragma: allowlist secret
    "ontology.prov.ttl": "27eaf6b9ff8eff7809afa8709a76ad9b1338604ea53f2baa7c454c57798c6dd0",  # pragma: allowlist secret
}

VEHICLES = {
    "ontology.json": "aa9229141beae4f5493cba29e257b1a4a45da2545a137a1f79f5fdfd0be857a0",  # pragma: allowlist secret
    "ontology.ttl": "94da7765d4d71f9e745d0d7ad75f7247d3ed4c6ea4fed64fcfb6dc71c95bfd3f",  # pragma: allowlist secret
    "ontology.shacl.ttl": "cd661e923a370da7f7c44120be47618fb98a779e9ad1b269972e4a5a4f0b2afa",  # pragma: allowlist secret
    "ontology.prov.ttl": "3ffe9059465768154e6bcde294997b33d65bdc572c6560225a524bb40dc88038",  # pragma: allowlist secret
}

ESTATE = Path(__file__).resolve().parents[3] / ".scratch" / "semantic-sidecar" / "estate" / "estates"


def _digests(out: Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in FILES}


def _without_what_was_added(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _without_what_was_added(v) for k, v in node.items() if k not in ADDED_SINCE}
    if isinstance(node, list):
        return [_without_what_was_added(v) for v in node]
    return node


def _as_the_prototype_wrote_it(out: Path) -> str:
    export = _without_what_was_added(json.loads((out / "ontology.json").read_text("utf-8")))
    export["schema"] = PROTOTYPE_SCHEMA  # the version the bump to graph-export/4 changed
    text = json.dumps(export, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_toy_estate_is_the_recorded_bytes(tmp_path: Path, toy_path: Path) -> None:
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({
        "name": "toy", "title": "Toy estate", "schema_prefix": "acme_", "schemas": ["shop", "marts"],
        "personal_objects": ["shop.owners", "shop.vehicles"], "resolve": ["make"],
        "sensitive": SENSITIVE, "ignore_columns": list(OPTIONS["ignore_columns"]),
        "derived_schemas": list(OPTIONS["derived_schemas"])}))
    (tmp_path / "views.txt").write_text(VIEWS)
    (tmp_path / "catalog.yaml").write_text(
        "verified:\n  - shop_claims.shop_dealers\n  - shop_claims.marts_claims_by_dealer\n"
        "  - shop_claims.shop_vehicles\n")
    build(config=cfg, duckdb_path=toy_path, views=tmp_path / "views.txt",
          verified=tmp_path / "catalog.yaml", out=tmp_path / "out")
    assert _digests(tmp_path / "out") == TOY
    assert _as_the_prototype_wrote_it(tmp_path / "out") == PROTOTYPE["toy"]


def test_the_vehicle_estate_is_the_recorded_bytes(tmp_path: Path) -> None:
    given = os.environ.get("LATENCE_ONTOLOGY_VEHICLE_INPUTS")
    if not given:
        pytest.skip("the vehicle estate's mirror is written by the host's tools and is not in this "
                    "repository: set LATENCE_ONTOLOGY_VEHICLE_INPUTS to run this check")
    inputs = Path(given)
    out = tmp_path / "vehicles"
    # The estate file as these bytes were recorded: without the outside vocabulary the owner
    # chose later (ticket 55), whose voice changes the class tree by design.
    said = yaml.safe_load((ESTATE / "vehicles.yaml").read_text(encoding="utf-8"))
    said["classes"].pop("vocabulary", None)
    config = tmp_path / "vehicles.yaml"
    config.write_text(yaml.safe_dump(said), encoding="utf-8")
    seed(config, inputs / "vehicles.duckdb", tmp_path / "seeded.duckdb", out, 11)
    build(config=config, duckdb_path=tmp_path / "seeded.duckdb",
          views=inputs / "metric_views.txt", verified=inputs / "catalog.yaml", out=out,
          class_answer=ESTATE / "vehicles.classes_answer.gpt-6-luna.json",
          usage=(inputs / "usage.jsonl",))
    assert _digests(out) == VEHICLES
    assert _as_the_prototype_wrote_it(out) == PROTOTYPE["vehicles"]

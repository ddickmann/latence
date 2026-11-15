"""The move into this package changed no byte (ticket 56). The digests below were recorded from
the estate prototype (`.scratch/semantic-sidecar/estate/estate/`) at the commit before the move:
the toy estate of this package's tests, and the vehicle estate the host's tools write. The toy
estate is built here on every run. The vehicle estate needs the host's mirror, its metric views,
its usage log and its verified joins, which this repository does not hold, so its test runs when
``LATENCE_ONTOLOGY_VEHICLE_INPUTS`` names a folder with ``vehicles.duckdb``,
``metric_views.txt``, ``usage.jsonl`` and ``catalog.yaml`` (the commands that write them are in
the prototype folder's ``HANDOFF.md``) and is skipped with that reason otherwise."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml
from _ontology_toy import OPTIONS, SENSITIVE, VIEWS
from latence_ontology.cli import build, seed

# The digests are SHA-256 of build outputs, not secrets; the secret scan reads any long hex
# string as one, hence the allowlist mark on each.
FILES = ("ontology.json", "ontology.ttl", "ontology.shacl.ttl", "ontology.prov.ttl")

TOY = {
    "ontology.json": "b284d1c462f5de81fdd237e853e2f487182747374b2ebb2fb963dc827ab45b57",  # pragma: allowlist secret
    "ontology.ttl": "31baa509e9683336a130193237a5c505378b401ce9010857cd01cb3e05c0f38a",  # pragma: allowlist secret
    "ontology.shacl.ttl": "45c9e8aeaa75c1032d3a0ea5362950257182daeb90a2fbde8182826e17ce59d9",  # pragma: allowlist secret
    "ontology.prov.ttl": "27eaf6b9ff8eff7809afa8709a76ad9b1338604ea53f2baa7c454c57798c6dd0",  # pragma: allowlist secret
}

VEHICLES = {
    "ontology.json": "05163c0730929426c3f0887315e8ced8eb2bd66fbd3035689b8dc758b2ca525f",  # pragma: allowlist secret
    "ontology.ttl": "c014e310bd9fc68f3b10eae44277dda140c7b6881695e74ed8759a668f43cb7e",  # pragma: allowlist secret
    "ontology.shacl.ttl": "cd661e923a370da7f7c44120be47618fb98a779e9ad1b269972e4a5a4f0b2afa",  # pragma: allowlist secret
    "ontology.prov.ttl": "3ffe9059465768154e6bcde294997b33d65bdc572c6560225a524bb40dc88038",  # pragma: allowlist secret
}

ESTATE = Path(__file__).resolve().parents[3] / ".scratch" / "semantic-sidecar" / "estate" / "estates"


def _digests(out: Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in FILES}


def test_the_toy_estate_is_the_prototypes_bytes(tmp_path: Path, toy_path: Path) -> None:
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


def test_the_vehicle_estate_is_the_prototypes_bytes(tmp_path: Path) -> None:
    given = os.environ.get("LATENCE_ONTOLOGY_VEHICLE_INPUTS")
    if not given:
        pytest.skip("the vehicle estate's mirror is written by the host's tools and is not in this "
                    "repository: set LATENCE_ONTOLOGY_VEHICLE_INPUTS to run this check")
    inputs = Path(given)
    out = tmp_path / "vehicles"
    seed(ESTATE / "vehicles.yaml", inputs / "vehicles.duckdb", tmp_path / "seeded.duckdb", out, 11)
    build(config=ESTATE / "vehicles.yaml", duckdb_path=tmp_path / "seeded.duckdb",
          views=inputs / "metric_views.txt", verified=inputs / "catalog.yaml", out=out,
          class_answer=ESTATE / "vehicles.classes_answer.gpt-6-luna.json",
          usage=(inputs / "usage.jsonl",))
    assert _digests(out) == VEHICLES

"""The vehicle estate's outside vocabulary (ticket 55: schema.org, the owner's choice of 15 Nov)
and its domains (ticket 58: proposed by the builder, the owner confirms them).

The estate files live in the semantic sidecar's estate folder. The vocabulary tests read the
committed extract; the `match` test builds the vehicle estate and needs its mirror, which this
repository does not hold (``LATENCE_ONTOLOGY_VEHICLE_INPUTS``, as in the same-bytes test)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from latence_core.errors import ConfigError
from latence_ontology.cli import build, propose, seed
from latence_ontology.vocabulary import read_vocabulary

ESTATE = Path(__file__).resolve().parents[3] / ".scratch" / "semantic-sidecar" / "estate" / "estates"
CONFIG: dict[str, Any] = yaml.safe_load((ESTATE / "vehicles.yaml").read_text(encoding="utf-8"))
VOCABULARY = ESTATE / str(CONFIG["classes"]["vocabulary"]["file"])


def test_the_schema_org_extract_is_refused_without_the_estates_allowance() -> None:
    with pytest.raises(ConfigError, match="licensed CC-BY-SA-3.0"):
        read_vocabulary(VOCABULARY)
    allowed = tuple(CONFIG["classes"]["vocabulary"]["licences_allowed"])
    vocabulary = read_vocabulary(VOCABULARY, allowed)
    assert vocabulary["licence"] == "CC-BY-SA-3.0"
    assert "terms.html" in vocabulary["licence_source"]
    assert vocabulary["source"]["release"] == "30.1"
    assert all(c["id"].startswith("https://schema.org/") for c in vocabulary["classes"])
    # schema.org's Claim is a factual claim, not a warranty claim: not in the extract.
    assert "https://schema.org/Claim" not in {c["id"] for c in vocabulary["classes"]}


def test_the_third_party_licence_file_records_the_vocabulary() -> None:
    text = (ESTATE.parents[2].parent / "THIRD-PARTY-LICENSES.md").read_text(encoding="utf-8")
    assert "schema.org" in text and "CC BY-SA 3.0" in text


def test_the_domains_cover_the_estates_schemas_with_a_data_owners_role() -> None:
    domains = CONFIG["domains"]
    held = [s for spec in domains.values() for s in spec["schemas"]]
    assert sorted(held) == ["clean", "internal", "partner"] and len(held) == len(set(held))
    assert all(str(spec["data_owner"]).startswith("role:") for spec in domains.values())
    assert set(CONFIG["derived_schemas"]).isdisjoint(held)  # a derived layer belongs to none


def test_a_vehicle_build_writes_a_match_across_two_domains(tmp_path: Path) -> None:
    given = os.environ.get("LATENCE_ONTOLOGY_VEHICLE_INPUTS")
    if not given:
        pytest.skip("the vehicle estate's mirror is written by the host's tools and is not in this "
                    "repository: set LATENCE_ONTOLOGY_VEHICLE_INPUTS to run this check")
    inputs, out = Path(given), tmp_path / "vehicles"
    # The host's alias table of the vehicle estate's library (metis.alias/1, its ticket 326; where
    # it was copied from: fixtures/alias/SOURCE.md): the make is one key of the estate namespace,
    # read from the views of both domains.
    table = Path(__file__).resolve().parent / "fixtures" / "alias" / "estate-semantics.json"
    config = ESTATE / "vehicles.yaml"
    seed(config, inputs / "vehicles.duckdb", tmp_path / "seeded.duckdb", out, 11)
    kwargs: dict[str, Any] = {
        "config": config, "duckdb_path": tmp_path / "seeded.duckdb",
        "views": inputs / "metric_views.txt", "verified": inputs / "catalog.yaml", "out": out,
        "class_answer": ESTATE / "vehicles.classes_answer.gpt-6-luna.json",
        "usage": (inputs / "usage.jsonl",), "alias_tables": (table,),
    }
    build(**kwargs)
    export = json.loads((out / "ontology.json").read_text())
    make = next(c for c in export["concepts"] if c["id"] == "concept:value:make")
    assert make["match"]["key"] == "dimension:estate.make"
    line = {"subject": make["id"], "verdict": "approved", "by": "reviewer-1",
            "claim": make["claim"], "at": "2026-11-15"}
    (out / "decisions.jsonl").write_text(json.dumps(line) + "\n")
    owners = [str(spec["data_owner"]) for d, spec in sorted(CONFIG["domains"].items())
              if d in ("after_sales", "product_safety")]
    (out / "certifications.jsonl").write_text("".join(
        json.dumps({"subject": make["id"], "by": o, "at": "2026-11-15", "claim": make["claim"]})
        + "\n" for o in owners))
    build(**kwargs)
    report = propose(config, out)
    written = yaml.safe_load((out / "proposal.yaml").read_text())
    assert written["format"] == "layer-proposal/3"
    [entry] = [e for e in written["entries"] if "match" in e]
    assert entry["match"]["key"] == "dimension:estate.make" and entry["match"]["close"] is False
    assert [c["domain"] for c in entry["match"]["concepts"]] == ["after_sales", "product_safety"]
    assert all(c["column"] == "make" for c in entry["match"]["concepts"])
    assert [c["by"] for c in entry["origin"]["certified"]] == owners
    assert [e["kind"] for e in report["entries"]].count("match") == 1

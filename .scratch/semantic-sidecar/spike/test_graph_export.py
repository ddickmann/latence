"""The graph export is the run's machine clusters, with counts that add up."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate import generate
from run_a import run


@pytest.fixture(scope="module")
def exports(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict, dict]:
    tmp = tmp_path_factory.mktemp("graph")
    path = tmp / "estate.parquet"
    generate(1500, seed=5).to_parquet(path, index=False)
    review, graph = tmp / "review.json", tmp / "graph.json"
    result = run(
        "estate", path, 800, 1200, 7, baseline_max=0,
        export_review=review, export_groups=30, export_graph=graph, export_graph_entities=120,
    )
    return json.loads(graph.read_text()), json.loads(review.read_text()), result


def test_schema_level_counts_describe_the_whole_estate(exports: tuple[dict, dict, dict]) -> None:
    graph, _, result = exports
    assert graph["schema"] == "graph-export/1"
    assert graph["meta"]["synthetic"] is True and graph["meta"]["schema_origin"] == "declared"
    assert graph["meta"]["records"] == result["records"]
    concepts = {c["id"]: c for c in graph["concepts"]}
    assert set(concepts) == {"organisation", "city", "country", "industry", "source_system"}
    assert concepts["organisation"]["count"] == graph["meta"]["organisations"]
    assert all(r["from"] in concepts and r["to"] in concepts for r in graph["relations"])
    by_type: dict[str, int] = {}
    for n in graph["nodes"]:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
    assert by_type == {k: concepts[k]["count"] for k in ("city", "country", "industry", "source_system")}
    # Every record is recorded in exactly one system.
    assert sum(n["records"] for n in graph["nodes"] if n["type"] == "source_system") == result["records"]


def test_every_organisation_is_one_machine_cluster_with_its_records(exports: tuple[dict, dict, dict]) -> None:
    graph, review, _ = exports
    seen: set[str] = set()
    ids = {e["id"] for e in graph["entities"]}
    nodes = {n["id"] for n in graph["nodes"]}
    assert len(ids) == len(graph["entities"]) == graph["meta"]["organisations_in_sample"]
    for e in graph["entities"]:
        records = [r["unique_id"] for r in e["records"]]
        assert records and not seen & set(records)  # a record supports one organisation only
        seen |= set(records)
        assert sum(v["records"] for v in e["values"]["name"]) <= len(records)
        assert e["systems"] == sorted({r["source_system"] for r in e["records"]})
        for field, prefix in (("city", "city"), ("country", "country"), ("industry", "industry")):
            assert e[field] is None or f"{prefix}:{e[field]}" in nodes
            assert e[field] is None or e["values"][field][0]["value"] == e[field]
    # The organisations the review queue touches are all carried.
    assert {r for g in review["groups"] for r in g["records"]} <= seen


def test_possibly_same_links_are_symmetric_and_point_at_review_items(exports: tuple[dict, dict, dict]) -> None:
    graph, review, _ = exports
    by_id = {e["id"]: e for e in graph["entities"]}
    item_ids = {i["id"] for i in review["items"]}
    links = 0
    for e in graph["entities"]:
        for link in e["possibly_same"]:
            links += 1
            other = by_id[link["entity"]]
            back = [b for b in other["possibly_same"] if b["entity"] == e["id"]]
            assert len(back) == 1 and back[0]["pairs"] == link["pairs"]
            assert 0.0 < link["probability"] < 1.0 and set(link["items"]) <= item_ids
    assert links > 0


def test_display_name_prefers_the_frequent_then_the_mixed_case_spelling() -> None:
    from graph_export import _display_name

    assert _display_name(["DELTASTONE PUBLIC LIMITED COMPANY", "Deltastone Mfg PLC", "Deltastone Manufacturing PLC"]) == "Deltastone Manufacturing PLC"
    assert _display_name(["ACME LTD", "Acme Ltd", "acme ltd", "Acme Limited"]) == "Acme Ltd"
    assert _display_name([]) == "(unnamed)"


def test_no_nan_reaches_the_file(exports: tuple[dict, dict, dict], tmp_path: Path) -> None:
    graph, _, _ = exports
    assert "NaN" not in json.dumps(graph)

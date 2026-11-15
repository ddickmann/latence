"""The builder's side of the ontology hub (ticket 59), on a local bare repository through the
port: no network. Two domains, ``sales`` (schema ``shop``) and ``service`` (schema ``svc``),
each on its branch with its stewards and its data owner; the conformed library's owner beside
them. The host's part (merging, tagging, blocking on a conflict, reverting) is played by the
local backend, as a forge would play it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _ontology_toy import Project
from latence_ontology.hub import pull_certifications, push_reviews, split, submit, views
from latence_ontology.hub_local import LocalGitHost
from latence_ontology.hub_port import HubError, Review

AT = "2026-11-15"
DOMAINS: dict[str, dict[str, Any]] = {
    "sales": {"schemas": ["shop"], "stewards": ["steward-sales"], "data_owner": "owner-sales"},
    "service": {"schemas": ["svc"], "stewards": ["steward-service"],
                "data_owner": "owner-service"},
}
CONFORMED = {"_conformed": {"data_owner": "estate-steward"}}


def _origin(element: str, n: int, by: str = "reviewer-1") -> dict[str, Any]:
    return {"element": element, "claim": f"{n:016x}",
            "decision": {"verdict": "approved", "by": by, "at": "2026-11-14"}}


DOCUMENT: dict[str, Any] = {
    "format": "layer-proposal/3",
    "source": "ontology",
    "entries": [
        {"match": {"key": "dimension:conformed.maker", "close": False, "concepts": [
            {"domain": "sales", "object": "shop.claims", "column": "maker"},
            {"domain": "service", "object": "svc.repairs", "column": "maker"}]},
         "origin": {**_origin("concept:value:maker", 1),
                    "certified": [{"domain": "sales", "by": "owner-sales", "at": AT},
                                  {"domain": "service", "by": "owner-service", "at": AT}]}},
        {"broader": {"object": "shop.claims", "column": "model", "value": "A1",
                     "broader": ["ALFA"], "support": {"rows": 200, "share": 1.0}},
         "origin": _origin("roll_up:value:model->value:maker", 2)},
        {"labels": {"object": "svc.repairs", "column": "maker", "language": "en",
                    "names": "Maker", "synonyms": ["brand"]},
         "origin": _origin("labels:concept:value:maker@en", 3, by="reviewer-2")},
        {"spellings": {"object": "erp.parts", "column": "maker", "value": "ALFA",
                       "forms": ["Alfa"]},
         "origin": _origin("same:maker:ALFA~Alfa", 4)},
    ],
}


@pytest.fixture()
def host(tmp_path: Path) -> LocalGitHost:
    files = {d: {"library.yaml": f"library: {d}\nversion: 1.0.0\n"} for d in DOMAINS}
    return LocalGitHost.create(tmp_path / "estate.git", {**DOMAINS, **CONFORMED}, AT, files)


def _certifications(domain: str, *elements: tuple[str, int]) -> list[dict[str, str]]:
    owner = DOMAINS[domain]["data_owner"]
    return [{"subject": e, "claim": f"{n:016x}", "by": owner, "at": AT} for e, n in elements]


def test_only_a_domains_stewards_commit_to_its_branch(host: LocalGitHost) -> None:
    host.commit("domains/sales", {"library.yaml": "library: sales\nversion: 1.0.1\n"}, "edit",
                "steward-sales", AT)
    for who in ("steward-service", "engine:latence-ontology", "owner-sales"):
        with pytest.raises(HubError, match="no steward of sales"):
            host.commit("domains/sales", {"x": "y"}, "edit", who, AT)


def test_the_builder_proposes_each_domains_share_by_a_merge_request_and_pushes_no_domain(
    host: LocalGitHost,
) -> None:
    heads = host.branches()
    files, unplaced = split(DOCUMENT, DOMAINS)
    assert sorted(files) == ["sales", "service"]
    assert [next(k for k in e if k != "origin") for e in files["sales"]["entries"]] == [
        "match", "broader"]
    assert [next(k for k in e if k != "origin") for e in files["service"]["entries"]] == [
        "match", "labels"]
    assert unplaced == [{"element": "same:maker:ALFA~Alfa", "kind": "spellings"}]
    opened = submit(host, DOCUMENT, "vehicles", DOMAINS, AT)
    assert [(o["domain"], o["entries"]) for o in opened["opened"]] == [("sales", 2), ("service", 2)]
    after = host.branches()
    assert after["domains/sales"] == heads["domains/sales"]  # no domain's branch was pushed
    assert after["domains/service"] == heads["domains/service"]
    requests = host.merge_requests()
    assert [(m.target, m.author, m.state) for m in requests] == [
        ("domains/sales", "engine:latence-ontology", "open"),
        ("domains/service", "engine:latence-ontology", "open")]
    # The same file again: the same branches, no second request.
    assert submit(host, DOCUMENT, "vehicles", DOMAINS, AT) == opened
    assert len(host.merge_requests()) == 2
    shown = views(host, "vehicles")
    assert "+  - match:" in shown["merge_requests"][0]["diff"]
    assert {r["state"] for r in shown["merge_requests"]} == {"proposed"}


def test_one_domain_is_approved_certified_merged_and_released_while_the_others_is_open(
    host: LocalGitHost,
) -> None:
    submit(host, DOCUMENT, "vehicles", DOMAINS, AT)
    sales, service = host.merge_requests()
    with pytest.raises(HubError, match="approval"):
        host.merge(sales.id, "steward-sales", AT, "1.1.0")  # no review yet
    # The builder's review, written through the port: the reviewers' approvals everywhere, the
    # data owner's certification where every element of the request carries it.
    certified = _certifications("sales", ("concept:value:maker", 1),
                                ("roll_up:value:model->value:maker", 2))
    written = push_reviews(host, "vehicles", DOMAINS, certified, AT)
    assert written == [
        {"merge_request": sales.id, "by": "reviewer-1", "role": "reviewer", "verdict": "approved"},
        {"merge_request": sales.id, "by": "owner-sales", "role": "data_owner",
         "verdict": "certified"},
        {"merge_request": service.id, "by": "reviewer-1", "role": "reviewer",
         "verdict": "approved"},
        {"merge_request": service.id, "by": "reviewer-2", "role": "reviewer",
         "verdict": "approved"}]
    assert push_reviews(host, "vehicles", DOMAINS, certified, AT) == []  # nothing twice
    with pytest.raises(HubError, match="no steward"):
        host.merge(sales.id, "engine:latence-ontology", AT, "1.1.0")  # the builder merges nothing
    merged = host.merge(sales.id, "steward-sales", AT, "1.1.0")
    assert merged.tag == "sales/1.1.0"
    release = {"library": "sales", "version": "1.1.0", "digest": "sha256:" + "ab" * 32}
    shown = views(host, "vehicles", {"sales/1.1.0": release})
    by_domain = {r["domain"]: r for r in shown["merge_requests"]}
    assert by_domain["sales"]["state"] == "certified" and by_domain["sales"]["tag"] == "sales/1.1.0"
    assert by_domain["sales"]["release"] == release
    assert by_domain["service"]["state"] == "approved" and "tag" not in by_domain["service"]
    assert [t["tag"] for t in shown["tags"]] == ["sales/1.1.0"]
    # The history shows both: every review with its actor, and the merge on the domain's branch.
    assert [(r["by"], r["role"]) for r in by_domain["sales"]["reviews"]] == [
        ("reviewer-1", "reviewer"), ("owner-sales", "data_owner")]
    assert [(r["by"], r["role"]) for r in by_domain["service"]["reviews"]] == [
        ("reviewer-1", "reviewer"), ("reviewer-2", "reviewer")]
    said = [(c["by"], c["message"].split(":")[0]) for c in shown["history"]["sales"]]
    assert said[0] == ("steward-sales", "Merge !1")
    assert sorted(said[1:]) == [("engine:latence-ontology", "vehicles"), ("steward-sales", "sales")]
    assert host.read("domains/sales", "proposals/vehicles.yaml") is not None
    assert host.read("domains/service", "proposals/vehicles.yaml") is None


def test_a_certification_given_on_the_git_host_is_read_back(host: LocalGitHost) -> None:
    submit(host, DOCUMENT, "vehicles", DOMAINS, AT)
    _sales, service = host.merge_requests()
    with pytest.raises(HubError, match="not the data owner"):
        host.review(service.id, Review("owner-sales", "data_owner", "certified", AT))
    host.review(service.id, Review("owner-service", "data_owner", "certified", "2026-11-16"))
    lines = pull_certifications(host, "vehicles", DOMAINS, [])
    assert lines == [
        {"subject": "concept:value:maker", "claim": f"{1:016x}", "by": "owner-service",
         "at": "2026-11-16"},
        {"subject": "labels:concept:value:maker@en", "claim": f"{3:016x}", "by": "owner-service",
         "at": "2026-11-16"}]
    assert pull_certifications(host, "vehicles", DOMAINS, lines) == []
    assert views(host, "vehicles")["merge_requests"][1]["state"] == "certified"


def test_a_conflict_across_domains_blocks_the_merge_until_the_conformed_owner_resolves_it(
    host: LocalGitHost,
) -> None:
    submit(host, DOCUMENT, "vehicles", DOMAINS, AT)
    sales, _service = host.merge_requests()
    push_reviews(host, "vehicles", DOMAINS, _certifications(
        "sales", ("concept:value:maker", 1), ("roll_up:value:model->value:maker", 2)), AT)
    host.block(sales.id, "dimension:conformed.maker is redefined by service")
    with pytest.raises(HubError, match="blocked"):
        host.merge(sales.id, "steward-sales", AT, "1.1.0")
    shown = views(host, "vehicles")["merge_requests"][0]
    assert shown["blocked"] == "dimension:conformed.maker is redefined by service"
    with pytest.raises(HubError, match="does not own the conformed library"):
        host.resolve(sales.id, "owner-sales", AT)
    host.resolve(sales.id, "estate-steward", AT)
    assert host.merge(sales.id, "steward-sales", AT, "1.1.0").tag == "sales/1.1.0"


def test_a_revert_is_a_new_release_and_rewrites_no_history(host: LocalGitHost) -> None:
    submit(host, DOCUMENT, "vehicles", DOMAINS, AT)
    sales, _service = host.merge_requests()
    push_reviews(host, "vehicles", DOMAINS, _certifications(
        "sales", ("concept:value:maker", 1), ("roll_up:value:model->value:maker", 2)), AT)
    host.merge(sales.id, "steward-sales", AT, "1.1.0")
    before = [c.sha for c in host.history("domains/sales")]
    first = host.tags()[0].commit
    del first
    tag = host.revert("sales", "sales/1.1.0^", "steward-sales", "2026-11-17", "1.1.1")
    after = [c.sha for c in host.history("domains/sales")]
    assert after[1:] == before  # one commit more, none rewritten
    assert host.read("domains/sales", "proposals/vehicles.yaml") is None  # 1.0.0's files again
    assert [t.name for t in host.tags()] == ["sales/1.1.0", "sales/1.1.1"]
    assert '"reverts_to": "sales/1.1.0^"' in tag.message


def test_the_hub_command_proposes_a_build_and_reads_its_release_back(
    project: Project, tmp_path: Path
) -> None:
    import json

    import yaml
    from latence_ontology.cli import build, hub

    cfg = yaml.safe_load(project["config"].read_text())
    cfg["domains"] = {"sales": {"schemas": ["shop"], "stewards": ["steward-sales"],
                                "data_owner": "owner-sales"}}
    project["config"].write_text(yaml.safe_dump(cfg))
    build(**project)
    export = json.loads((project["out"] / "ontology.json").read_text())
    step = next(r for r in export["structure"]["roll_ups"]
                if r["from"].startswith("concept:value:") and r["to"].startswith("concept:value:"))
    (project["out"] / "decisions.jsonl").write_text(json.dumps(
        {"subject": step["id"], "verdict": "approved", "by": "reviewer-1", "claim": step["claim"],
         "at": AT}) + "\n")
    build(**project)
    repo = tmp_path / "toy.git"
    host = LocalGitHost.create(repo, cfg["domains"], AT)
    said = hub(project["config"], project["out"], repo, AT)
    assert [o["domain"] for o in said["opened"]] == ["sales"] and said["certifications_read"] == 0
    assert said["reviews_written"] == [{"merge_request": 1, "by": "reviewer-1",
                                        "role": "reviewer", "verdict": "approved"}]
    # The data owner certifies on the git host; the next run reads it back into the build folder.
    host.review(1, Review("owner-sales", "data_owner", "certified", "2026-11-16"))
    again = hub(project["config"], project["out"], repo, AT)
    assert again["certifications_read"] >= 1 and again["reviews_written"] == []
    lines = (project["out"] / "certifications.jsonl").read_text().splitlines()
    assert {json.loads(x)["subject"] for x in lines} == {step["id"]}
    merged = host.merge(1, "steward-sales", AT, "1.1.0")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"format": "layer-receipt/1", "library": "sales",
                                   "version": "1.1.0", "tag": merged.tag,
                                   "digest": "sha256:" + "cd" * 32, "placed": []}))
    hub(project["config"], project["out"], repo, AT, (receipt,))
    shown = json.loads((project["out"] / "hub.json").read_text())
    [request] = shown["merge_requests"]
    assert request["state"] == "certified" and request["tag"] == "sales/1.1.0"
    assert request["release"] == {"library": "sales", "version": "1.1.0",
                                  "digest": "sha256:" + "cd" * 32}
    assert {e["element"] for e in request["entries"]} == {step["id"]}

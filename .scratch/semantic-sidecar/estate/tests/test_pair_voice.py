"""A hosted model on the pairs the linker left undecided (ticket 48): off unless the estate file
switches it on with an egress rule, only the fields the rule names leave, a cap stops the calls,
each call is recorded, and the model's verdict decides nothing: it is a vote on the review item,
with the model as its actor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from estate.pair_voice import EgressRefused, ask_about_pairs, rule_of
from tests.test_proposal import yard  # noqa: F401  (the fixture)

RECORDS = {
    "shop.dealers:D3": {"name": "Dealer 3 GmbH", "city": "Ulm", "country": "DE", "postcode": "89073"},
    "partner.accounts:PA4": {"name": "Dealer Three", "city": "Ulm", "country": "DE", "postcode": "89075"},
    "shop.dealers:D9": {"name": "Dealer 9 GmbH", "city": "Kiel", "country": "DE", "postcode": "24103"},
    "partner.accounts:PA7": {"name": "Ninth Dealer", "city": "Kiel", "country": "DE", "postcode": "24105"},
}
REVIEW = [{"id": "same:shop.dealers:D3~partner.accounts:PA4", "a": "shop.dealers:D3", "b": "partner.accounts:PA4", "p": 0.55},
          {"id": "same:shop.dealers:D9~partner.accounts:PA7", "a": "shop.dealers:D9", "b": "partner.accounts:PA7", "p": 0.41},
          {"id": "same:make:ALFA~Alfa", "a": "value:make:ALFA", "b": "value:make:Alfa"}]
PRICES = {"input_per_1m": 1.0, "output_per_1m": 4.0}
ON = {"enabled": True, "model": "gpt-6-luna", "fields": ["name", "city"], "prices": PRICES, "max_usd": 1.0, "max_output_tokens": 200}


class StandInClient:
    """Stands in for the provider: answers each request, says what it used, and keeps every
    request it was sent, so that the test can see what left."""

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts, self.sent = list(verdicts), []

    def judge(self, request: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(request)
        return {"verdict": self.verdicts.pop(0), "served_by": "gpt-6-luna-2026-10-01", "usage": {"input_tokens": 300, "output_tokens": 20}}


def test_it_is_off_unless_the_estate_switches_it_on_with_an_egress_rule(tmp_path: Path) -> None:
    client = StandInClient(["same", "same"])
    for link in ({}, {"hosted_model": {}}, {"hosted_model": {**ON, "enabled": False}}):
        review, note = ask_about_pairs(REVIEW, RECORDS, rule_of(link), client, tmp_path / "calls.jsonl")
        assert review == REVIEW and note == {"state": "off"} and client.sent == []
    with pytest.raises(EgressRefused, match="no egress rule"):
        rule_of({"hosted_model": {k: v for k, v in ON.items() if k != "fields"}})
    with pytest.raises(EgressRefused, match="phone, postcode"):
        rule_of({"hosted_model": {**ON, "fields": ["name", "postcode", "phone"]}})  # only what a record of the linker carries, and no key
    with pytest.raises(EgressRefused, match="prices"):
        rule_of({"hosted_model": {k: v for k, v in ON.items() if k != "prices"}})
    assert not (tmp_path / "calls.jsonl").exists()


def test_only_the_fields_the_rule_names_leave_each_call_is_recorded_and_a_verdict_is_a_vote_with_its_actor(tmp_path: Path) -> None:
    client = StandInClient(["same", "different"])
    review, note = ask_about_pairs(REVIEW, RECORDS, rule_of({"hosted_model": ON}), client, tmp_path / "calls.jsonl")
    sent = json.dumps(client.sent)
    assert len(client.sent) == 2  # a pair of names of a value set is not the linker's and is not asked
    assert "Dealer 3 GmbH" in sent and "Ulm" in sent
    assert "89073" not in sent and "D3" not in sent and "PA4" not in sent and "DE" not in sent  # no other field, no key, no id
    assert [r.get("model_vote") for r in review] == [
        {"verdict": "same", "actor": "model:gpt-6-luna-2026-10-01"}, {"verdict": "different", "actor": "model:gpt-6-luna-2026-10-01"}, None]
    assert [r["p"] for r in review[:2]] == [0.55, 0.41] and all("decided_by" not in r for r in review)  # it decides nothing
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(c["pair"], c["answer"], c["model"]) for c in calls] == [(REVIEW[0]["id"], "same", "gpt-6-luna-2026-10-01"),
                                                                     (REVIEW[1]["id"], "different", "gpt-6-luna-2026-10-01")]
    assert calls[0]["sent"] == {"a": {"name": "Dealer 3 GmbH", "city": "Ulm"}, "b": {"name": "Dealer Three", "city": "Ulm"}}
    assert calls[0]["usd"] == round((300 * 1.0 + 20 * 4.0) / 1e6, 6)
    assert note == {"state": "asked", "pairs": 2, "calls": 2, "usd": round(2 * calls[0]["usd"], 6), "stopped_at_cap": 0}


def test_the_cap_stops_the_calls_before_one_that_could_cost_more_than_is_left(tmp_path: Path) -> None:
    client = StandInClient(["same", "same"])
    one = ask_about_pairs(REVIEW[:1], RECORDS, rule_of({"hosted_model": ON}), StandInClient(["same"]), tmp_path / "probe.jsonl")[1]
    bound = json.loads((tmp_path / "probe.jsonl").read_text(encoding="utf-8"))["worst_case_usd"]
    assert one["calls"] == 1 and bound > one["usd"]
    rule = rule_of({"hosted_model": {**ON, "max_usd": bound * 1.5}})  # room for one call's worst case, not for two
    review, note = ask_about_pairs(REVIEW, RECORDS, rule, client, tmp_path / "calls.jsonl")
    assert note["calls"] == 1 and note["stopped_at_cap"] == 1 and len(client.sent) == 1
    assert "model_vote" not in review[1]


def test_a_build_asks_only_when_the_estate_allows_it_and_its_export_carries_the_votes(
        yard: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jsonschema
    import yaml

    from estate import cli
    from tests import test_proposal as tp

    monkeypatch.setattr(cli, "run_resolver", lambda *args, **kwargs: tp._resolver_answer())
    yard["out"].mkdir(parents=True, exist_ok=True)
    (yard["out"] / "seed_truth.json").write_text(json.dumps({"truth": {f"PA{i}": f"other:PA{i}" for i in range(1, 46)}}), encoding="utf-8")
    client = StandInClient(["unsure"])
    cli.build(**yard, resolver=tmp_path, pair_client=client)  # the estate file says nothing: off
    assert client.sent == [] and not (yard["out"] / "pair_model_calls.jsonl").exists()
    estate = yaml.safe_load(yard["config"].read_text(encoding="utf-8"))
    estate["link"]["hosted_model"] = {**ON, "fields": ["name"]}
    yard["config"].write_text(yaml.safe_dump(estate), encoding="utf-8")
    summary = cli.build(**yard, resolver=tmp_path, pair_client=client)
    export = json.loads((yard["out"] / "ontology.json").read_text(encoding="utf-8"))
    jsonschema.validate(export, tp.EXPORT_SCHEMA)
    voted = [r for r in export["review"] if "model_vote" in r]
    assert [(r["id"], r["model_vote"]["verdict"]) for r in voted] == [("same:shop.dealers:D3~partner.accounts:PA4", "unsure")]
    assert client.sent and "Dealer 3 GmbH" in json.dumps(client.sent) and "Portal account 4" in json.dumps(client.sent)
    assert summary["pair_model"]["calls"] == 1 and export["meta"]["resolution"]["pair_model"]["calls"] == 1
    assert not any(e.get("actor", "").startswith("model:") for e in export["edges"])  # a vote links nothing

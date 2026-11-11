"""The class tree: what kind of thing each concept is, as proposals from two voices."""

from __future__ import annotations

from typing import Any

from estate.classes import TOP, HostedClassModel, RecordedClassModel, describe, pattern_voice, prompt_of, propose_classes


def _concept(cid: str, label: str, kind: str, cols: list[tuple[str, str, bool]], key: list[str] | None = None, obj: str | None = None) -> dict[str, Any]:
    return {"id": f"concept:{cid}", "label": label, "kind": kind, "object_name": obj or cid, "key": key or [], "definition": "",
            "attributes": [{"column": n, "type": t, "sensitive": s, "label": n} for n, t, s in cols]}


MODEL = {
    "concepts": [
        _concept("s.dealers", "Dealer", "entity", [("name", "VARCHAR", False), ("city", "VARCHAR", False), ("contact_email", "VARCHAR", True)], ["dealer_id"]),
        _concept("s.owners", "Owner", "entity", [("name", "VARCHAR", True), ("street", "VARCHAR", True)], ["owner_id"]),
        _concept("s.claims", "Claim", "fact", [("opened_date", "DATE", False), ("amount", "DECIMAL", False)], ["claim_ref"]),
        _concept("s.parts", "Part", "entity", [("description", "VARCHAR", False)], ["part_id"]),
        _concept("s.bulletins", "Bulletin", "entity", [("summary", "VARCHAR", False), ("url", "VARCHAR", False)], ["bulletin_no"]),
        _concept("m.claims_by_dealer", "Claims by dealer", "derived", [("claims", "BIGINT", False)], ["dealer_id"]),
        {"id": "concept:value:market", "label": "Market", "kind": "reference", "column": "market", "definition": ""},
        {"id": "concept:value:unit", "label": "Unit", "kind": "reference", "column": "unit", "definition": ""},
    ],
    "relations": [
        {"id": "r1", "from": "concept:s.claims", "to": "concept:s.dealers", "columns": ["dealer_id"]},
        {"id": "r2", "from": "concept:s.claims", "to": "concept:s.parts", "columns": ["part_id"]},
        {"id": "r3", "from": "concept:s.claims", "to": "concept:s.owners", "columns": ["owner_id"]},
    ],
}
STRUCTURE: dict[str, Any] = {"part_of": [], "roll_ups": []}


class FakeModel:
    name = "fake-model"

    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer, self.seen = answer, []

    def propose(self, pack: list[dict[str, Any]], top: dict[str, str]) -> dict[str, Any]:
        self.seen.append((pack, top))
        return self.answer


def test_a_voice_is_shown_names_types_and_relations_and_nothing_else() -> None:
    pack = describe(MODEL, STRUCTURE, personal_objects=("s.owners",))
    dealer = next(c for c in pack if c["id"] == "concept:s.dealers")
    assert dealer["columns"] == [{"name": "name", "type": "VARCHAR", "personal": False, "comment": ""}, {"name": "city", "type": "VARCHAR", "personal": False, "comment": ""},
                                 {"name": "contact_email", "type": "VARCHAR", "personal": True, "comment": ""}]
    assert dealer["referred_to_by"] == ["Claim"] and next(c for c in pack if c["id"] == "concept:s.owners")["about_persons"] is True
    assert set(dealer) == {"id", "label", "kind", "object", "comment", "key", "about_persons", "columns", "refers_to", "referred_to_by", "part_of", "rolls_up_to"}


def test_the_columns_alone_place_the_plain_cases() -> None:
    said = {k.removeprefix("concept:"): v[0] for k, v in pattern_voice(describe(MODEL, STRUCTURE)).items()}
    assert said == {"s.dealers": "organisation", "s.owners": "person", "s.claims": "event", "s.parts": "product", "s.bulletins": "document", "value:market": "place"}
    # a derived table and a unit of measure are none of the six, and nothing is forced on them


def test_without_a_model_only_the_patterns_speak_and_everything_is_a_proposal() -> None:
    out = propose_classes(MODEL, STRUCTURE)
    assert out["voices"] == {"attribute_pattern": True, "language_model": None} and out["classes"] == []
    assert {s["concept"].removeprefix("concept:"): s["parent"] for s in out["subclass_of"]}["s.dealers"] == "class:organisation"
    assert all(s["status"] == "proposed" and not s["contested"] and [x["kind"] for x in s["signals"]] == ["attribute_pattern"] for s in out["subclass_of"])
    assert out["unplaced"] == ["concept:m.claims_by_dealer", "concept:value:unit"]
    assert [t["id"] for t in out["top"]] == [f"class:{k}" for k in TOP] and all(t["status"] == "declared" for t in out["top"])


def test_a_model_adds_definitions_and_classes_of_its_own_and_a_disagreement_is_kept() -> None:
    fake = FakeModel({
        "classes": [{"name": "Service event", "parent": "event", "definition": "Something a workshop did to a vehicle."},
                    {"name": "Orphan class", "parent": "nowhere", "definition": "x"}],
        "concepts": [
            {"id": "concept:s.claims", "parent": "Service event", "definition": "One warranty claim a dealer filed.", "reason": "dated, points at a dealer and a part"},
            {"id": "concept:s.dealers", "parent": "organisation", "definition": "A workshop that sells and repairs.", "reason": "a name with a town"},
            {"id": "concept:s.parts", "parent": "document", "definition": "A spare part.", "reason": "it has a description"},
            {"id": "concept:value:unit", "parent": "none", "definition": "A unit of measure.", "reason": "a pure classification"},
            {"id": "concept:s.bulletins", "parent": "none", "definition": "A table of counts.", "reason": "every column is a count"},
            {"id": "concept:made.up", "parent": "event", "definition": "x", "reason": "x"},
        ]})
    out = propose_classes(MODEL, STRUCTURE, voice=fake)
    by = {s["concept"].removeprefix("concept:"): s for s in out["subclass_of"]}
    assert [c["id"] for c in out["classes"]] == ["class:service_event"] and out["classes"][0]["parent"] == "class:event"  # a class under nothing known is not taken
    claim = by["s.claims"]
    assert claim["parent"] == "class:service_event" and claim["definition"] == [{"text": "One warranty claim a dealer filed.", "lang": "en", "origin": "model:fake-model"}] and not claim["contested"]
    assert [x["kind"] for x in claim["signals"]] == ["language_model", "attribute_pattern"]  # the two voices agree: an event either way
    assert by["s.parts"]["contested"] and by["s.parts"]["parent"] == "class:document" and by["s.parts"]["other_parent"] == "class:product"
    assert "made.up" not in by and "value:unit" not in by  # an unknown concept is ignored; "none" places nothing
    bulletin = by["s.bulletins"]  # the model says none of the classes, the columns say document: placed, and marked as disputed
    assert bulletin["parent"] == "class:document" and bulletin["contested"] and bulletin["signals"][0]["code"] == "language_model.declined" and bulletin["signals"][0]["text"]["text"] == "every column is a count"
    assert by["s.owners"]["parent"] == "class:person" and [x["kind"] for x in by["s.owners"]["signals"]] == ["attribute_pattern"]  # the model was silent
    assert out["voices"]["language_model"] == "fake-model" and all(s["status"] == "proposed" for s in out["subclass_of"])
    pack, top = fake.seen[0]
    assert top == TOP and "rows" not in str(pack) and all("values" not in c for c in pack)


def test_the_hosted_voice_sends_names_only_and_reads_the_answer_as_data() -> None:
    class Block:
        type, text = "text", '{"classes": [], "concepts": [{"id": "concept:s.dealers", "parent": "organisation", "definition": "d", "reason": "r"}]}'

    class Response:
        stop_reason, content, model = "end_turn", [Block()], "served-model"

    class Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kw: Any) -> Response:
            self.calls.append(kw)
            return Response()

    class Client:
        def __init__(self) -> None:
            self.beta = type("Beta", (), {"messages": Messages()})()

    client = Client()
    out = HostedClassModel(client=client).propose(describe(MODEL, STRUCTURE), TOP)
    call = client.beta.messages.calls[0]
    assert out["served_by"] == "served-model" and out["concepts"][0]["parent"] == "organisation"
    assert call["model"] == "claude-opus-5-5" and call["output_config"]["format"]["type"] == "json_schema" and call["fallbacks"] == "default"
    sent = call["messages"][0]["content"]
    assert "contact_email" in sent and "Top classes" in sent  # names are sent
    assert "no rows and no values" in call["system"]


def test_a_recorded_answer_stands_in_for_the_call_and_the_prompt_file_holds_all_that_leaves() -> None:
    import pytest

    pack = describe(MODEL, STRUCTURE)
    prompt = prompt_of(pack, TOP)
    assert set(prompt) == {"system", "top_classes", "concepts", "answer_schema"} and prompt["concepts"] == pack
    voice = RecordedClassModel({"answered_by": "a model session, 2026-11-11", "classes": [],
                                "concepts": [{"id": "concept:s.dealers", "parent": "organisation", "definition": "A workshop.", "reason": "a name with a town"}]})
    out = propose_classes(MODEL, STRUCTURE, voice=voice)
    dealer = next(s for s in out["subclass_of"] if s["concept"] == "concept:s.dealers")
    assert dealer["definition"][0]["text"] == "A workshop." and dealer["definition"][0]["origin"].startswith("model:recorded answer") and dealer["signals"][0]["ref"] == "recorded answer (a model session, 2026-11-11)"
    assert out["voices"]["language_model"] == "recorded answer (a model session, 2026-11-11)"
    with pytest.raises(ValueError, match="answered_by"):
        RecordedClassModel({"classes": [], "concepts": []})


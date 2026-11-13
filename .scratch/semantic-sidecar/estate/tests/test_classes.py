"""The class tree: what kind of thing each concept is, as proposals from two voices."""

from __future__ import annotations

import json
from typing import Any

from estate.classes import TOP, CachedClassModel, OpenAIClassModel, RecordedClassModel, describe, pattern_voice, prompt_of, propose_classes, worst_case_usd


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


class Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _strict(schema: dict[str, Any], path: str = "answer") -> None:
    """What the provider refuses in a strict schema: an object that allows other fields, or
    one that does not require every field it names."""
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{path}: additionalProperties must be false")
        if sorted(schema.get("required", [])) != sorted(schema["properties"]):
            raise ValueError(f"{path}: every field must be required")
        for name, sub in schema["properties"].items():
            _strict(sub, f"{path}.{name}")
    if schema.get("type") == "array":
        _strict(schema["items"], f"{path}[]")


class StandIn:
    """The provider's client, as far as the voice uses it. It refuses what the provider
    refuses, counts its calls, and answers in the form the SDK returns."""

    def __init__(self, answer: dict[str, Any] | None = None, status: str = "completed", refusal: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.answer = answer or {"classes": [], "concepts": [{"id": "concept:s.dealers", "parent": "organisation", "definition": "A workshop.",
                                                                "definition_de": "Eine Werkstatt.", "reason": "a name with a town"}]}
        self.status, self.refusal = status, refusal
        self.responses = self

    def stream(self, **kw: Any) -> "StandIn":
        """The SDK's streaming helper: a context manager that hands out the final response."""
        self._final = self._answer_to(**kw)
        return self

    def __enter__(self) -> "StandIn":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_response(self) -> Obj:
        return self._final

    def _answer_to(self, **kw: Any) -> Obj:
        allowed = {"model", "instructions", "input", "reasoning", "max_output_tokens", "store", "text"}
        if set(kw) - allowed:
            raise TypeError(f"unexpected arguments: {sorted(set(kw) - allowed)}")
        form = kw["text"]["format"]
        if form["type"] != "json_schema" or form["strict"] is not True or not form["name"]:
            raise ValueError("a strict json_schema format with a name is required")
        _strict(form["schema"])
        if kw["reasoning"]["effort"] not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("unknown reasoning effort")
        self.calls.append(kw)
        content = [Obj(type="refusal", refusal=self.refusal)] if self.refusal else [Obj(type="output_text", text=json.dumps(self.answer))]
        return Obj(status=self.status, incomplete_details=Obj(reason="max_output_tokens") if self.status == "incomplete" else None,
                   output=[Obj(type="reasoning"), Obj(type="message", content=content)], output_text="" if self.refusal else json.dumps(self.answer),
                   model="gpt-6-luna-2026-10-05",
                   usage=Obj(input_tokens=4200, input_tokens_details=Obj(cached_tokens=0), output_tokens=1800, output_tokens_details=Obj(reasoning_tokens=600)))


PRICES = {"input_per_1m": 0.10, "output_per_1m": 0.50, "read_on": "2026-11-11", "source": "a test"}


def test_the_hosted_voice_sends_names_only_in_a_strict_form_and_reads_the_answer_as_data() -> None:
    client = StandIn()
    voice = OpenAIClassModel(client=client)
    out = voice.propose(describe(MODEL, STRUCTURE), TOP)
    call = client.calls[0]
    assert out["served_by"] == "gpt-6-luna-2026-10-05" and out["concepts"][0]["parent"] == "organisation"
    assert call["model"] == "gpt-6-luna" and call["store"] is False and call["reasoning"] == {"effort": "low"}
    assert "contact_email" in call["input"] and "Top classes" in call["input"]  # names are sent
    assert "no rows and no values" in call["instructions"] and "English and German" in call["instructions"]
    assert voice.usage == {"input_tokens": 4200, "cached_input_tokens": 0, "output_tokens": 1800, "reasoning_tokens": 600}
    placed = propose_classes(MODEL, STRUCTURE, voice=OpenAIClassModel(client=StandIn()))
    dealer = next(s for s in placed["subclass_of"] if s["concept"] == "concept:s.dealers")
    assert [(t["lang"], t["text"], t["origin"]) for t in dealer["definition"]] == [("en", "A workshop.", "model:gpt-6-luna-2026-10-05"), ("de", "Eine Werkstatt.", "model:gpt-6-luna-2026-10-05")]


def test_an_answer_that_was_cut_off_or_refused_proposes_nothing() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="incomplete .max_output_tokens."):
        OpenAIClassModel(client=StandIn(status="incomplete")).propose(describe(MODEL, STRUCTURE), TOP)
    with pytest.raises(RuntimeError, match="refused: not this"):
        OpenAIClassModel(client=StandIn(refusal="not this")).propose(describe(MODEL, STRUCTURE), TOP)


def test_the_model_is_asked_once_for_the_same_request_and_every_call_is_logged_with_its_cost(tmp_path: Any) -> None:
    client = StandIn()
    pack = describe(MODEL, STRUCTURE)
    first = CachedClassModel(OpenAIClassModel(client=client), tmp_path / "answers", PRICES, 0.25, tmp_path / "calls.jsonl")
    answer = first.propose(pack, TOP)
    assert len(client.calls) == 1 and first.cost["call"] == "made" and first.cost["usd"] == 0.00132  # 4,200 in at 0.10 and 1,800 out at 0.50 per million
    again = CachedClassModel(OpenAIClassModel(client=client), tmp_path / "answers", PRICES, 0.25, tmp_path / "calls.jsonl")
    assert again.propose(pack, TOP) == answer and len(client.calls) == 1  # the same request: no second call
    assert again.cost["usd"] == 0.00132 and again.cost["call"].startswith("none")
    changed = [{**pack[0], "label": "Workshop"}, *pack[1:]]
    again.propose(changed, TOP)
    assert len(client.calls) == 2  # something else would be sent: asked again
    log = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [e["usd"] for e in log] == [0.00132, 0.00132] and log[0]["output_tokens"] == 1800 and log[0]["prices"]["read_on"] == "2026-11-11"


def test_a_call_that_could_cost_more_than_allowed_is_not_made(tmp_path: Any) -> None:
    import pytest

    client = StandIn()
    voice = OpenAIClassModel(client=client)
    bound = worst_case_usd(voice.request(describe(MODEL, STRUCTURE), TOP), PRICES)
    assert 0.006 < bound < 0.01  # the whole output allowance of 12,000 tokens, and every character sent as a token
    with pytest.raises(RuntimeError, match="more than the USD 0.001 allowed"):
        CachedClassModel(voice, tmp_path / "answers", PRICES, 0.001, tmp_path / "calls.jsonl").propose(describe(MODEL, STRUCTURE), TOP)
    assert client.calls == [] and not (tmp_path / "calls.jsonl").exists()


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



def test_a_build_asks_the_hosted_model_once_and_only_where_the_estate_allows_it(tmp_path: Any, toy_path: Any) -> None:
    import pytest
    import yaml

    from estate.cli import build

    from .conftest import OPTIONS, SENSITIVE, VIEWS

    def project(classes: dict[str, Any] | None) -> dict[str, Any]:
        cfg = tmp_path / f"toy{len(list(tmp_path.glob('*.yaml')))}.yaml"
        cfg.write_text(yaml.safe_dump({"name": "toy", "title": "Toy estate", "schema_prefix": "acme_", "schemas": ["shop", "marts"],
                                       "personal_objects": ["shop.owners", "shop.vehicles"], "sensitive": SENSITIVE,
                                       "ignore_columns": list(OPTIONS["ignore_columns"]), "derived_schemas": list(OPTIONS["derived_schemas"]),
                                       **({"classes": classes} if classes is not None else {})}))
        views = tmp_path / "views.txt"
        views.write_text(VIEWS)
        return {"config": cfg, "duckdb_path": toy_path, "views": views, "verified": None, "out": tmp_path / "out"}

    client = StandIn({"classes": [], "concepts": [{"id": "concept:shop.dealers", "parent": "organisation", "definition": "A workshop.",
                                                   "definition_de": "Eine Werkstatt.", "reason": "a name"}]})
    with pytest.raises(SystemExit, match="does not allow"):
        build(**project(None), hosted_model=True, client=client)
    with pytest.raises(SystemExit, match="states no prices"):
        build(**project({"hosted_model": True}), hosted_model=True, client=client)
    assert client.calls == []
    allowed = project({"hosted_model": True, "prices": PRICES})
    first = build(**allowed, hosted_model=True, client=client)
    assert len(client.calls) == 1 and first["classes"]["hosted_call"] == "made"
    written = json.loads((allowed["out"] / "classes_prompt.json").read_text())
    assert written == {k: client.calls[0][k] for k in written} and set(written) == set(client.calls[0])  # the file a person can read is what was sent
    assert "D1" not in client.calls[0]["input"] and "Dealer 1" not in client.calls[0]["input"] and "@example.org" not in client.calls[0]["input"]  # no value leaves
    export = (allowed["out"] / "ontology.json").read_bytes()
    assert json.loads(export)["classes"]["voices"]["cost"]["usd"] == 0.00132
    second = build(**allowed, hosted_model=True, client=client)
    assert len(client.calls) == 1 and second["classes"]["hosted_call"].startswith("none")  # nothing new to send: no call
    assert (allowed["out"] / "ontology.json").read_bytes() == export  # and the same file
    assert len((allowed["out"] / "model_calls.jsonl").read_text().splitlines()) == 1

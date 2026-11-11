"""A class tree above the tables: what kind of thing each concept is.

The structure pass proves what counts can prove. It cannot say that a dealer is an
organisation and a claim an event: that is meaning, and meaning is proposed, never proved.
Two voices propose it here, and a person decides.

- **attribute patterns** (always on, deterministic): the columns a concept carries and the
  way other concepts use it. A non-personal name with a town is an organisation; a personal
  name is a person; a dated row that points at other things is an event.
- **a language model** (only when the estate allows it): it is shown names, types, comments,
  keys and relations of the concepts, and never a row or a value. It names a parent class for
  each concept, may put a class of its own between a concept and a top class, and writes one
  sentence of definition.

There is no outside vocabulary (owner, 2026-11-11): the top classes are the six below. Every
result is ``proposed``. Where the two voices disagree, both are kept and the proposal is
marked contested.

What may be sent to a hosted model is an egress decision per estate (``classes.hosted_model``
in the estate file). Without it only the pattern voice speaks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

TOP: dict[str, str] = {
    "organisation": "A company, a public body or a unit of one that acts under a name.",
    "person": "A human being. Records of persons are never shown.",
    "product": "A thing that is made, sold, owned or repaired, or a kind of such things.",
    "event": "Something that happened at a time and involves other things: a claim, a visit, a recall.",
    "place": "A location or an area: a town, a region, a market.",
    "document": "A text someone wrote and filed: a bulletin, a report, a letter.",
}
PERSONAL_NAME = re.compile(r"^(name|first_?name|last_?name|surname|full_?name|birth|street|address)", re.I)
ORG_NAME = re.compile(r"(^|_)name$", re.I)
PLACE = re.compile(r"^(city|town|region|country|postcode|zip|market|state|county)$", re.I)
TEXT = re.compile(r"(summary|description|text|body|title|url|document|subject)", re.I)
THING = re.compile(r"(part|model|vin|sku|product|item|vehicle)", re.I)
DATED = re.compile(r"(date|time|timestamp)", re.I)


def describe(model: dict[str, Any], structure: dict[str, Any], personal_objects: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """What a voice is shown about each concept: names, types, comments, keys and how the
    concepts refer to one another. No row and no value."""
    label = {c["id"]: c["label"] for c in model["concepts"]}
    out = []
    for c in model["concepts"]:
        refers = sorted({label[r["to"]] for r in model["relations"] if r["from"] == c["id"] and r["to"] in label})
        used_by = sorted({label[r["from"]] for r in model["relations"] if r["to"] == c["id"] and r["from"] in label})
        out.append({
            "id": c["id"], "label": c["label"], "kind": c["kind"], "object": c.get("object_name") or c.get("column"),
            "comment": c.get("definition") or "", "key": c.get("key") or [],
            "about_persons": c.get("object_name") in personal_objects,
            "columns": [{"name": a["column"], "type": a["type"], "personal": bool(a["sensitive"]), "comment": a.get("definition") or ""}
                        for a in c.get("attributes") or []],
            "refers_to": refers, "referred_to_by": used_by,
            "part_of": sorted(label[p["to"]] for p in structure.get("part_of", []) if p["from"] == c["id"]),
            "rolls_up_to": sorted(label[r["to"]] for r in structure.get("roll_ups", []) if r["from"] == c["id"]),
        })
    return out


def pattern_voice(pack: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """concept id -> (top class, why), for the concepts whose columns say something."""
    out: dict[str, tuple[str, str]] = {}
    for c in pack:
        cols = c["columns"]
        names = [x["name"] for x in cols]
        personal_names = [x["name"] for x in cols if x["personal"] and PERSONAL_NAME.search(x["name"])]
        open_names = [n for x in cols if not x["personal"] and ORG_NAME.search(n := x["name"])]
        dated = [x["name"] for x in cols if DATED.search(x["type"]) or DATED.search(x["name"])]
        if c["kind"] == "reference":
            if PLACE.search(str(c["object"])):
                out[c["id"]] = ("place", f"the column is called {c['object']}")
            continue
        if c["kind"] == "derived":
            continue
        if personal_names:
            out[c["id"]] = ("person", f"it carries personal columns that name a human ({', '.join(personal_names)})")
        elif open_names and (any(PLACE.search(n) for n in names) or any(n.startswith("contact_") for n in names)):
            out[c["id"]] = ("organisation", f"it carries a name that is not personal ({open_names[0]}) with a place or a contact")
        elif dated and c["refers_to"] and (c["kind"] == "fact" or len(c["refers_to"]) >= 2):
            out[c["id"]] = ("event", f"it is dated ({dated[0]}) and points at {', '.join(c['refers_to'][:3])}")
        elif any(TEXT.search(n) for n in names) and not c["referred_to_by"]:
            out[c["id"]] = ("document", f"it carries text columns ({', '.join(n for n in names if TEXT.search(n))}) and nothing refers to it")
        elif c["referred_to_by"] and (any(THING.search(k) for k in c["key"]) or THING.search(str(c["object"]))):
            out[c["id"]] = ("product", f"other concepts refer to it ({', '.join(c['referred_to_by'][:3])}) and its key names a thing")
    return out


class ClassModel(Protocol):
    name: str

    def propose(self, pack: list[dict[str, Any]], top: dict[str, str]) -> dict[str, Any]: ...


ANSWER = {
    "type": "object", "additionalProperties": False, "required": ["classes", "concepts"],
    "properties": {
        "classes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "parent", "definition"],
            "properties": {"name": {"type": "string"}, "parent": {"type": "string"}, "definition": {"type": "string"}}}},
        "concepts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "parent", "definition", "reason"],
            "properties": {"id": {"type": "string"}, "parent": {"type": "string"}, "definition": {"type": "string"}, "reason": {"type": "string"}}}},
    },
}
SYSTEM = """You are helping to draft the class tree of an ontology over a data estate. You are shown the concepts that \
were read from the estate's catalog: for each one its label, the object it comes from, its key, its columns with types and \
comments, and how the concepts refer to one another. You are shown no rows and no values.

For every concept, say which class it is a kind of. Use one of the top classes you are given, or a class of your own that \
sits between the concept and a top class when two or more concepts share something more specific (name such a class in \
the "classes" list, with the top class or another of your classes as its parent). Keep the tree shallow: a class of your \
own must have at least two concepts under it. If a concept is none of the top classes (a pure classification such as a \
unit of measure, or an aggregate table), give "none" as its parent and say why.

Write one plain sentence of definition for each concept and each class of your own: what a row of it is, in the words \
of someone who works with this data. State only what the names, comments and relations support. Give a short reason for \
each parent that names the evidence you used. A person will review every line; nothing you write is applied automatically."""


class HostedClassModel:
    """The language-model voice over the Anthropic Messages API. Credentials come from the
    environment; where a network proxy injects the key, any placeholder value will do."""

    def __init__(self, client: Any | None = None, model: str = "claude-opus-5-5") -> None:
        if client is None:
            import anthropic  # imported here so that tests and builds without this voice need no SDK

            client = anthropic.Anthropic()
        self.client, self.name = client, model

    def propose(self, pack: list[dict[str, Any]], top: dict[str, str]) -> dict[str, Any]:
        prompt = ("Top classes:\n" + "\n".join(f"- {k}: {v}" for k, v in top.items())
                  + "\n\nConcepts (JSON):\n" + json.dumps(pack, ensure_ascii=False, indent=1))
        # A refusal on this model is re-run by the service on its default fallback model.
        response = self.client.beta.messages.create(
            model=self.name, max_tokens=16000, system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": ANSWER}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"the model stopped with {response.stop_reason}; no classes proposed")
        text = next(b.text for b in response.content if b.type == "text")
        return {**json.loads(text), "served_by": response.model}


def prompt_of(pack: list[dict[str, Any]], top: dict[str, str]) -> dict[str, Any]:
    """Everything a model is given, as one file: the instruction, the concepts, and the form
    its answer must have. What is in here is what leaves the estate, and nothing else does."""
    return {"system": SYSTEM, "top_classes": top, "concepts": pack, "answer_schema": ANSWER}


class RecordedClassModel:
    """The language-model voice read from a file: an answer some model session gave to the
    prompt of ``prompt_of``. A build with it sends nothing anywhere and gives the same result
    every time. ``name`` says where the answer came from."""

    def __init__(self, answer: dict[str, Any]) -> None:
        missing = {"classes", "concepts", "answered_by"} - set(answer)
        if missing:
            raise ValueError(f"the recorded answer lacks {sorted(missing)}")
        self.answer, self.name = answer, f"recorded answer ({answer['answered_by']})"

    def propose(self, pack: list[dict[str, Any]], top: dict[str, str]) -> dict[str, Any]:
        return {"classes": self.answer["classes"], "concepts": self.answer["concepts"]}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def propose_classes(model: dict[str, Any], structure: dict[str, Any], personal_objects: tuple[str, ...] = (),
                    voice: ClassModel | None = None) -> dict[str, Any]:
    """The class tree as proposals: the six top classes, classes a model put between them and
    the concepts, and for each concept the class it is a kind of, with every voice that said so."""
    pack = describe(model, structure, personal_objects)
    known = {c["id"] for c in pack}
    by_pattern = pattern_voice(pack)
    answer = voice.propose(pack, TOP) if voice is not None else {"classes": [], "concepts": []}
    served = answer.get("served_by") or (voice.name if voice is not None else None)

    mid, parent_of = [], {}
    names = {_slug(k): f"class:{k}" for k in TOP}
    for c in answer["classes"]:
        names[_slug(c["name"])] = f"class:{_slug(c['name'])}"
    for c in answer["classes"]:
        parent = names.get(_slug(c["parent"]))
        if parent is None or _slug(c["name"]) in TOP:
            continue  # a class that hangs under nothing we know is not taken
        parent_of[names[_slug(c["name"])]] = parent
        mid.append({"id": names[_slug(c["name"])], "label": c["name"], "parent": parent, "definition": c["definition"], "status": "proposed",
                    "signals": [{"kind": "language_model", "detail": f"proposed by {served} as a class between concepts and {c['parent']}", "ref": ""}]})

    def top_of(cid: str | None) -> str | None:
        seen: set[str] = set()
        while cid in parent_of and cid not in seen:
            seen.add(cid)
            cid = parent_of[cid]
        return cid if cid and cid.removeprefix("class:") in TOP else None

    said = {c["id"]: c for c in answer["concepts"] if c["id"] in known}
    subclass_of = []
    for c in pack:
        m, p = said.get(c["id"]), by_pattern.get(c["id"])
        m_parent = names.get(_slug(m["parent"])) if m else None
        signals = []
        declined = bool(m and m_parent is None and _slug(m["parent"]) == "none")
        if m and m_parent:
            signals.append({"kind": "language_model", "detail": m["reason"], "ref": str(served)})
        elif declined and p:
            signals.append({"kind": "language_model", "detail": f"none of the classes: {m['reason']}", "ref": str(served)})
        if p:
            signals.append({"kind": "attribute_pattern", "detail": p[1], "ref": ""})
        parent = m_parent or (f"class:{p[0]}" if p else None)
        if parent is None:
            continue
        contested = bool(p and ((m_parent and top_of(m_parent) != f"class:{p[0]}") or declined))
        subclass_of.append({"id": f"subclass:{c['id'].removeprefix('concept:')}", "concept": c["id"], "parent": parent, "status": "proposed",
                            "definition": (m or {}).get("definition", ""), "contested": contested,
                            **({"other_parent": f"class:{p[0]}"} if contested and p and m_parent else {}), "signals": signals})
    used = {s["parent"] for s in subclass_of} | {s.get("other_parent") for s in subclass_of} | set(parent_of.values())
    mid = [m for m in mid if sum(s["parent"] == m["id"] for s in subclass_of) + sum(v == m["id"] for v in parent_of.values()) >= 1]
    return {
        "top": [{"id": f"class:{k}", "label": k.capitalize(), "definition": v, "status": "declared", "in_use": f"class:{k}" in used} for k, v in TOP.items()],
        "classes": mid,
        "subclass_of": subclass_of,
        "voices": {"attribute_pattern": True, "language_model": served},
        "unplaced": sorted(c["id"] for c in pack if c["id"] not in {s["concept"] for s in subclass_of}),
    }

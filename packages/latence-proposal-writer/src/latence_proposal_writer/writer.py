"""The proposal file for a host's semantic layer, written from approved decisions (ADR-0070).

What a review of the ontology approved reaches a host's semantic layer by one file, which the
host's own tool places into its library as a reviewed change. This module writes that file
from a build's export (what every element states now) and the decisions in force (what a
person decided).

Two kinds of entry are written, each from a decision in force that says ``approved``:

- ``spellings``: two values of a value set that a reviewer called the same. The value kept is
  the one more rows carry, by the build's own rule; the other is its other spelling.
- ``names``: the names of the records of a master list, by their key. Only for a list the
  caller names (``names_of``) or the export marks (``"proposal": "names"``, the estate's word
  that its names may go, which also lets a host's page offer the decision): a reviewer's
  approval of a concept does not by itself send its rows' names anywhere.

Each entry says where it goes by a catalog object and a column, never by a name of the host's
layer, and carries the decision it rests on: the element's id, the element's claim, who
approved it and on which day.

Never written:

- a rejected or undecided element (only the last decision on an element is in force);
- anything seeded: a seeded list, a record of one, a pair that holds one, and a pair of values
  of a value set that a seeded object carries, since the export cannot say which object a
  value was read from;
- a link between records, whoever made it: the machine's links were approved by nobody, and a
  host's names entry keeps one origin per record, so a linked record's name has no place in it;
- an entry whose decision was made on another statement than the build now makes (its claim
  differs): it is listed as stale;
- an entry whose decision states no claim, or no day while none was given: nothing is made up.

In version 2 a link between two records that a person approved is written too (``links``): the
two records by object, key column, key and name. Never a link the machine accepted and no
person approved (one certified on simulated labels is such a link), never one that holds a
seeded record, and never one whose pair the build cannot state now (no claim to check it on).

The file is YAML in block style with every text in double quotes, so that no reader takes a
name for a number, a date or a truth value. The same inputs give the same bytes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from latence_core.errors import ConfigError

SOURCE = "ontology"  # the kind of source the entries come from, as the host's layer records it
# The versions of the file this writer writes. Version 1 is the host's proposal shape with the
# two kinds of entry above and no ``version`` key; version 2 states ``version: "2"`` and adds
# ``links``, a link between two records a person decided (ticket 45). A caller asks for 2 only
# once the host's tool reads it.
VERSIONS = (1, 2)
SCHEMA_VERSION = VERSIONS[-1]
COMMAND = "latence-proposal-writer"  # who wrote the file, as its opening comment says

# The groups of the export a person can decide and this file has no entry for, by the key the
# export lists them under, with the kind of element each holds (graph-export/3).
STRUCTURE_KINDS = {"roll_ups": "roll_up", "equivalent": "equivalent", "part_of": "part_of",
                   "describes": "describes"}
CLASS_KINDS = {"classes": "class", "subclass_of": "subclass"}

_PLAIN_KEY = re.compile(r"[a-z_]+")
_READ_AS_SOMETHING_ELSE = {"true", "false", "null", "yes", "no", "on", "off", "y", "n"}
# Characters some YAML readers take for a line break or drop, even inside quotes.
_ESCAPED = ("\x7f", "\x85", " ", " ", "﻿")


def _header(command: str) -> tuple[str, ...]:
    return (
        "What the review of this estate's ontology approved, for a host's semantic layer to "
        "take up.",
        f"Written by `{command}` from the approved decisions of a build. Nothing",
        "rejected, undecided, seeded or decided on another statement is in it.",
        "Do not edit it: decide, and write it again.",
    )


def _quoted(text: str) -> str:
    """A text in double quotes: a JSON string, which is a YAML one too."""
    out = json.dumps(text, ensure_ascii=False)
    for ch in _ESCAPED:
        out = out.replace(ch, f"\\u{ord(ch):04x}")
    return out


def _key(name: str) -> str:
    plain = _PLAIN_KEY.fullmatch(name) and name not in _READ_AS_SOMETHING_ELSE
    return name if plain else _quoted(name)


def _lines(node: Mapping[str, Any] | list[Any], indent: int) -> list[str]:
    """A mapping or a list as YAML lines in block style. A text is always quoted."""
    pad = " " * indent
    out: list[str] = []
    if isinstance(node, Mapping):
        for name, value in node.items():
            if isinstance(value, Mapping | list) and value:
                out += [f"{pad}{_key(name)}:", *_lines(value, indent + 2)]
            elif isinstance(value, Mapping | list):
                out.append(f"{pad}{_key(name)}: {'{}' if isinstance(value, Mapping) else '[]'}")
            else:
                out.append(f"{pad}{_key(name)}: {_quoted(str(value))}")
        return out
    for item in node:
        if isinstance(item, Mapping):
            inner = _lines(item, indent + 2)
            out += [f"{pad}- {inner[0][indent + 2:]}", *inner[1:]]
        else:
            out.append(f"{pad}- {_quoted(str(item))}")
    return out


def to_yaml(document: Mapping[str, Any], estate: str, command: str = COMMAND) -> str:
    """The proposal file's text: a fixed header naming ``command`` as its writer, then the
    document."""
    head = [f"# {line}" for line in (*_header(command), f"Estate: {_quoted(estate)}")]
    return "\n".join([*head, *_lines(document, 0)]) + "\n"


def _day(text: Any, where: str) -> str:
    """The day of an ISO date or timestamp. Anything else is refused, not repaired."""
    try:
        return date.fromisoformat(str(text)[:10]).isoformat()
    except ValueError:
        said = f"{where} states the day {str(text)!r}, which is no ISO date (YYYY-MM-DD)"
        raise ConfigError(said) from None


def _kinds(export: Mapping[str, Any]) -> dict[str, str]:
    """The kind of every element a person can decide and this file has no entry for."""
    kinds = {r["id"]: "relation" for r in export["relations"]}
    kinds.update({m["id"]: "measure" for m in export["measures"]})
    for part, groups in ((export.get("structure") or {}, STRUCTURE_KINDS),
                         (export.get("classes") or {}, CLASS_KINDS)):
        for key, kind in groups.items():
            kinds.update({el["id"]: kind for el in part.get(key, [])})
    return kinds


def _object_of(concept: Mapping[str, Any], relations: list[dict[str, Any]],
               stated: str | None) -> str:
    """The object a value set's spellings are stated on: the one the caller names, else the
    one that holds the most values of the set (the first by name among equals)."""
    objects = [str(o) for o in concept.get("objects") or []]
    if stated is not None:
        if stated not in objects:
            raise ConfigError(f"{stated} does not hold the value set {concept['column']}: "
                              f"the spellings of it may be stated on one of {objects}")
        return stated
    held: dict[str, int] = {}
    for r in relations:
        if r["to"] == concept["id"]:
            for s in r.get("signals", []):
                if s.get("code") == "shared_column.holds":
                    held[str(r["from"]).removeprefix("concept:")] = int(s["params"]["values"])
    return min(objects, key=lambda o: (-held.get(o, 0), o))


def kept_values(pairs: list[Mapping[str, Any]]) -> dict[str, str]:
    """For each pair of values called the same, the value its two names became.

    The rule is the build's own: of two names, the one more rows carry stays (on equal rows,
    the first by id), it carries the rows of both from then on, and a name two pairs share
    joins them; the pairs are taken in the order of their ids. It is applied to the pairs
    given and to no other, so a value never rests on a decision that is not in the file, and
    the answer does not depend on whether a build has applied the decisions since."""
    ordered = sorted(pairs, key=lambda p: str(p["id"]))
    rows: dict[str, int] = {}
    concept: dict[str, str] = {}
    for p in ordered:
        for end, count in ((p["a"], p.get("rows_a")), (p["b"], p.get("rows_b"))):
            rows.setdefault(str(end), int(count or 0))
            concept.setdefault(str(end), str(p["concept"]))
    into: dict[str, str] = {}

    def standing(name: str) -> str:
        while name in into:
            name = into[name]
        return name

    for p in ordered:
        a, b = standing(str(p["a"])), standing(str(p["b"]))
        if a == b or concept[a] != concept[b]:
            continue
        keep, alias = (a, b) if (rows[a], b) >= (rows[b], a) else (b, a)
        rows[keep] += rows.pop(alias)
        into[alias] = keep
    return {str(p["id"]): standing(str(p["a"])) for p in ordered
            if standing(str(p["a"])) == standing(str(p["b"]))}


def _decided_links(export: Mapping[str, Any]) -> dict[frozenset[str], str]:
    """The pairs of records a build linked on a person's decision, with the claim of the pair
    the build states on the edge."""
    return {frozenset((str(e["from"]), str(e["to"]))): str(e["claim"])
            for e in export.get("edges", [])
            if e.get("kind") == "same_as" and e.get("decided_by") and e.get("claim")}


def proposal(export: Mapping[str, Any], decisions: Mapping[str, Mapping[str, Any]],
             names_of: tuple[str, ...] = (), spellings_on: Mapping[str, str] | None = None,
             decided_on: str | None = None,
             version: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    """The proposal file as data, and a report of what was written and what was not.

    ``export`` is a build's ``graph-export/3``; ``decisions`` the decisions in force, by
    element id. ``names_of`` are the master lists whose records' names may be written, beside
    those the export marks; ``spellings_on`` the object to state a value set's spellings on (by
    column); and ``decided_on`` the day of the approved decisions that state none (the day a
    decision states, ``at``, a date or a timestamp, always wins). ``version`` is the file's
    version (``VERSIONS``); links between records are written from version 2 on.

    The report holds ``entries`` (one line per entry written), ``stale`` (approved on another
    statement), ``not_written`` (approved, with the reason: ``seeded``, ``no_claim``,
    ``no_day``, ``not_in_the_build``, ``no_records``) and ``not_carried`` (approved elements of
    a kind this file has no entry for, counted by kind)."""
    if version not in VERSIONS:
        raise ConfigError(f"this writer writes the proposal file in versions {VERSIONS}, "
                          f"not in version {version}")
    given = _day(decided_on, "the command line") if decided_on else None
    linked = _decided_links(export)
    links: dict[str, tuple[str, str]] = {}  # element id -> the pair's two records
    concepts = {c["id"]: c for c in export["concepts"]}
    nodes = {n["id"]: n for n in export["nodes"]}
    seeded = {c["object_name"] for c in concepts.values()
              if c.get("seeded") and c.get("object_name")}
    asked = (export.get("label_requests") or {}).values()
    listed: dict[str, Mapping[str, Any]] = {p["id"]: p for sample in asked for p in sample}
    listed.update({p["id"]: p for p in export["review"]})
    kinds = _kinds(export)
    stale: list[dict[str, str]] = []
    not_written: list[dict[str, str]] = []
    not_carried: dict[str, int] = {}
    ready: dict[str, dict[str, str]] = {}  # element id -> the origin its entry will carry

    def held_back(subject: str, why: str) -> None:
        not_written.append({"subject": subject, "why": why})

    def carried_not(kind: str) -> None:
        not_carried[kind] = not_carried.get(kind, 0) + 1

    def origin_of(subject: str, d: Mapping[str, Any], now: str) -> None:
        """The checks every entry passes: decided on what the element states now, on a day."""
        stated = _day(d["at"], f"the decision on {subject}") if d.get("at") else None
        if not d.get("claim"):
            held_back(subject, "no_claim")
        elif d["claim"] != now:
            stale.append({"subject": subject, "decided_on": str(d["claim"]), "now": now})
        elif stated is None and given is None:
            held_back(subject, "no_day")
        else:
            ready[subject] = {"claim": now, "by": str(d["by"]), "at": stated or given or "",
                              "day_stated_by": "decision" if stated else "command"}

    for subject, d in sorted(decisions.items()):
        if d["verdict"] != "approved":
            continue
        if subject.startswith("same:"):
            pair = listed.get(subject)
            if pair is None:  # a pair a person has answered is no longer listed for review
                a, _, b = subject.removeprefix("same:").partition("~")
                pair = {"a": a, "b": b} if a in nodes and b in nodes else None
            if pair is None:
                held_back(subject, "not_in_the_build")
            elif pair["a"].startswith("value:") and pair["b"].startswith("value:"):
                value_set = concepts.get(pair.get("concept", ""))
                if value_set is None:
                    held_back(subject, "not_in_the_build")
                elif seeded & set(value_set.get("objects") or ()):
                    held_back(subject, "seeded")
                else:
                    origin_of(subject, d, pair["claim"])
            elif any(x.startswith(o + ":") for o in seeded for x in (pair["a"], pair["b"])):
                held_back(subject, "seeded")
            elif version < 2:
                carried_not("pair_of_records")
            else:
                ends = (str(pair["a"]), str(pair["b"]))
                now = pair.get("claim") or linked.get(frozenset(ends))
                if now is None:  # neither listed nor linked on a person's decision in this build
                    held_back(subject, "not_in_the_build")
                else:
                    links[subject] = ends
                    origin_of(subject, d, str(now))
        elif subject in concepts:
            concept = concepts[subject]
            if concept.get("seeded"):
                held_back(subject, "seeded")
            elif concept.get("object_name") not in names_of and concept.get("proposal") != "names":
                carried_not("concept")
            else:
                origin_of(subject, d, concept["claim"])
        elif subject in kinds:
            carried_not(kinds[subject])
        else:
            held_back(subject, "not_in_the_build")

    # (the element, the entry's body under its kind, its line in the report)
    found: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pairs = sorted((p for p in export["review"] if p["id"] in ready and p["id"] not in links),
                   key=lambda p: str(p["id"]))
    became = kept_values(pairs)
    for pair in pairs:
        value_set = concepts[pair["concept"]]
        column = str(value_set["column"])
        names = [str(x).removeprefix(f"value:{column}:") for x in (became.get(pair["id"], ""),
                                                                    pair["a"], pair["b"])]
        kept, forms = names[0], sorted(set(names[1:]) - {names[0]})
        if not kept or not forms:
            held_back(pair["id"], "not_in_the_build")
            continue
        stated_on = _object_of(value_set, export["relations"], (spellings_on or {}).get(column))
        body: dict[str, Any] = {"object": stated_on, "column": column, "value": kept,
                                "forms": forms}
        found.append((pair["id"], {"spellings": body}, {"kind": "spellings", **body}))
    for subject in sorted(s for s in ready if s in concepts):
        concept = concepts[subject]
        rows = {str(n["key"]): str(n["label"]) for n in export["nodes"]
                if n["concept"] == subject and "key" in n}
        if len(concept.get("key") or []) != 1 or not rows:
            held_back(subject, "no_records")
            continue
        body = {"object": concept["object_name"], "column": concept["key"][0]}
        found.append((subject, {"names": {**body, "values": dict(sorted(rows.items()))}},
                      {"kind": "names", **body, "records": len(rows)}))

    for subject in sorted(s for s in ready if s in links):
        records = []
        for end in links[subject]:
            node = nodes.get(end) or {}
            owner = concepts.get(str(node.get("concept")))
            if owner is None or "key" not in node or len(owner.get("key") or []) != 1:
                break
            records.append({"object": str(owner["object_name"]), "column": str(owner["key"][0]),
                            "key": str(node["key"]), "name": str(node["label"])})
        if len(records) != 2:
            held_back(subject, "no_records")
            continue
        found.append((subject, {"links": {"records": records}},
                      {"kind": "links", "records": 2, "objects": [r["object"] for r in records]}))

    entries: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    for number, (subject, entry, line) in enumerate(found, start=1):
        o = ready[subject]
        decision = {"verdict": "approved", "by": o["by"], "at": o["at"]}
        origin = {"element": subject, "claim": o["claim"], "decision": decision}
        entries.append({**entry, "origin": origin})
        written.append({"entry": number, "kind": line["kind"], "element": subject,
                        "claim": o["claim"], "decided_by": o["by"], "decided_on": o["at"],
                        "day_stated_by": o["day_stated_by"],
                        **{k: v for k, v in line.items() if k != "kind"}})
    report = {"entries": written, "stale": sorted(stale, key=lambda s: s["subject"]),
              "not_written": sorted(not_written, key=lambda s: s["subject"]),
              "not_carried": dict(sorted(not_carried.items()))}
    head = {"version": str(version)} if version > 1 else {}
    return {**head, "source": SOURCE, "entries": entries}, report

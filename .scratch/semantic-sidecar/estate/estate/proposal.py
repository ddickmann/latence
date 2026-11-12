"""The proposal file for a host's semantic layer, written from approved decisions (ADR-0070).

What a review of the ontology approved reaches a host's semantic layer by one file, which the
host's own tool places into its library as a reviewed change. This module writes that file
from a build's export (what every element states now) and its decisions file (what a person
decided), and reads the receipt the host's tool writes when it has placed the file.

Two kinds of entry are written, each from a decision in force that says ``approved``:

- ``spellings``: two values of a value set that a reviewer called the same. The value kept is
  the one more rows carry, by the build's own rule; the other is its other spelling.
- ``names``: the names of the records of a master list, by their key. Only for a list the
  estate file names under ``propose.names``: a reviewer's approval of a concept does not by
  itself send its rows' names anywhere.

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

The file is YAML in block style with every text in double quotes, so that no reader takes a
name for a number, a date or a truth value. The same inputs give the same bytes.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .instances import merge_values
from .ledger import CLASS_KINDS, STRUCTURE_KINDS

SOURCE = "ontology"  # the kind of source the entries come from, as the host's layer records it
RECEIPT = "layer-receipt/1"  # the format of the receipt this build reads
HEADER = (
    "What the review of this estate's ontology approved, for a host's semantic layer to take up.",
    "Written by `python -m estate propose` from the approved decisions of a build. Nothing",
    "rejected, undecided, seeded or decided on another statement is in it.",
    "Do not edit it: decide, and write it again.",
)

_PLAIN_KEY = re.compile(r"[a-z_]+")
_READ_AS_SOMETHING_ELSE = {"true", "false", "null", "yes", "no", "on", "off", "y", "n"}
# Characters some YAML readers take for a line break or drop, even inside quotes.
_ESCAPED = ("\x7f", "\x85", " ", " ", "﻿")


def _quoted(text: str) -> str:
    """A text in double quotes: a JSON string, which is a YAML one too."""
    out = json.dumps(text, ensure_ascii=False)
    for ch in _ESCAPED:
        out = out.replace(ch, f"\\u{ord(ch):04x}")
    return out


def _key(name: str) -> str:
    plain = _PLAIN_KEY.fullmatch(name) and name not in _READ_AS_SOMETHING_ELSE
    return name if plain else _quoted(name)


def _lines(node: dict[str, Any] | list[Any], indent: int) -> list[str]:
    """A mapping or a list as YAML lines in block style. A text is always quoted."""
    pad = " " * indent
    out: list[str] = []
    if isinstance(node, dict):
        for name, value in node.items():
            if isinstance(value, dict | list) and value:
                out += [f"{pad}{_key(name)}:", *_lines(value, indent + 2)]
            elif isinstance(value, dict | list):
                out.append(f"{pad}{_key(name)}: {'{}' if isinstance(value, dict) else '[]'}")
            else:
                out.append(f"{pad}{_key(name)}: {_quoted(str(value))}")
        return out
    for item in node:
        if isinstance(item, dict):
            inner = _lines(item, indent + 2)
            out += [f"{pad}- {inner[0][indent + 2:]}", *inner[1:]]
        else:
            out.append(f"{pad}- {_quoted(str(item))}")
    return out


def to_yaml(document: dict[str, Any], estate: str) -> str:
    """The proposal file's text: a fixed header, then the document."""
    head = [f"# {line}" for line in (*HEADER, f"Estate: {_quoted(estate)}")]
    return "\n".join([*head, *_lines(document, 0)]) + "\n"


def _day(text: Any, where: str) -> str:
    """The day of an ISO date or timestamp. Anything else is refused, not repaired."""
    try:
        return date.fromisoformat(str(text)[:10]).isoformat()
    except ValueError:
        said = f"{where} states the day {str(text)!r}, which is no ISO date (YYYY-MM-DD)"
        raise ValueError(said) from None


def _kinds(export: dict[str, Any]) -> dict[str, str]:
    """The kind of every element a person can decide and this file has no entry for."""
    kinds = {r["id"]: "relation" for r in export["relations"]}
    kinds.update({m["id"]: "measure" for m in export["measures"]})
    for part, groups in ((export.get("structure") or {}, STRUCTURE_KINDS),
                         (export.get("classes") or {}, CLASS_KINDS)):
        for key, kind in groups.items():
            kinds.update({el["id"]: kind for el in part.get(key, [])})
    return kinds


def _object_of(concept: dict[str, Any], relations: list[dict[str, Any]], stated: str | None) -> str:
    """The object a value set's spellings are stated on: the one the estate file names, else
    the one that holds the most values of the set (the first by name among equals)."""
    objects = list(concept.get("objects") or [])
    if stated is not None:
        if stated not in objects:
            raise ValueError(f"{stated} does not hold the value set {concept['column']}: "
                             f"propose.spellings_on in the estate file may name one of {objects}")
        return stated
    held: dict[str, int] = {}
    for r in relations:
        if r["to"] == concept["id"]:
            for s in r.get("signals", []):
                if s.get("code") == "shared_column.holds":
                    held[r["from"].removeprefix("concept:")] = int(s["params"]["values"])
    return str(min(objects, key=lambda o: (-held.get(o, 0), o)))


def _became(pairs: list[dict[str, Any]]) -> dict[str, str]:
    """For each pair of values called the same, the value its two names became. The rule is
    the build's own (the name that more rows carry stays, and a name two pairs share joins
    them), applied to the pairs that are written and to no other. So a value never rests on a
    decision that is not in the file, and the answer does not depend on whether a build has
    applied the decisions since."""
    ordered = sorted(pairs, key=lambda p: str(p["id"]))
    names: dict[str, dict[str, Any]] = {}
    for p in ordered:
        for end, rows in ((p["a"], p.get("rows_a")), (p["b"], p.get("rows_b"))):
            names.setdefault(end, {"id": end, "concept": p["concept"], "label": end,
                                   "rows": int(rows or 0)})
    asked = [{k: v for k, v in p.items() if k != "merged_into"} for p in ordered]
    merged = merge_values({"nodes": list(names.values()), "edges": [], "review": asked, "meta": {}},
                          [(p["a"], p["b"]) for p in ordered])
    return {p["id"]: p["merged_into"] for p in merged["review"] if "merged_into" in p}


def proposal(export: dict[str, Any], decisions: dict[str, dict[str, Any]],
             names_of: tuple[str, ...] = (), spellings_on: dict[str, str] | None = None,
             decided_on: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The proposal file as data, and a report of what was written and what was not.

    ``export`` is a build's ``graph-export/3``; ``decisions`` the decisions in force, by
    element id. ``names_of`` are the master lists whose records' names may be written,
    ``spellings_on`` the object to state a value set's spellings on (by column), and
    ``decided_on`` the day of the approved decisions that state none.

    The report holds ``entries`` (one line per entry written), ``stale`` (approved on another
    statement), ``not_written`` (approved, with the reason: ``seeded``, ``no_claim``,
    ``no_day``, ``not_in_the_build``, ``no_records``) and ``not_carried`` (approved elements of
    a kind this file has no entry for, counted by kind)."""
    given = _day(decided_on, "the command line") if decided_on else None
    concepts = {c["id"]: c for c in export["concepts"]}
    nodes = {n["id"]: n for n in export["nodes"]}
    seeded = {c["object_name"] for c in concepts.values()
              if c.get("seeded") and c.get("object_name")}
    asked = (export.get("label_requests") or {}).values()
    listed = {p["id"]: p for sample in asked for p in sample}
    listed.update({p["id"]: p for p in export["review"]})
    kinds = _kinds(export)
    stale: list[dict[str, str]] = []
    not_written: list[dict[str, str]] = []
    not_carried: dict[str, int] = {}
    ready: dict[str, dict[str, Any]] = {}  # element id -> the origin its entry will carry

    def held_back(subject: str, why: str) -> None:
        not_written.append({"subject": subject, "why": why})

    def origin_of(subject: str, d: dict[str, Any], now: str) -> None:
        """The checks every entry passes: decided on what the element states now, on a day."""
        stated = _day(d["at"], f"the decision on {subject}") if d.get("at") else None
        if not d.get("claim"):
            held_back(subject, "no_claim")
        elif d["claim"] != now:
            stale.append({"subject": subject, "decided_on": str(d["claim"]), "now": now})
        elif stated is None and given is None:
            held_back(subject, "no_day")
        else:
            ready[subject] = {"claim": now, "by": str(d["by"]), "at": stated or given,
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
            else:
                not_carried["pair_of_records"] = not_carried.get("pair_of_records", 0) + 1
        elif subject in concepts:
            concept = concepts[subject]
            if concept.get("seeded"):
                held_back(subject, "seeded")
            elif concept.get("object_name") not in names_of:
                not_carried["concept"] = not_carried.get("concept", 0) + 1
            else:
                origin_of(subject, d, concept["claim"])
        elif subject in kinds:
            not_carried[kinds[subject]] = not_carried.get(kinds[subject], 0) + 1
        else:
            held_back(subject, "not_in_the_build")

    # (the element, the entry's body under its kind, its line in the report)
    found: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pairs = sorted((p for p in export["review"] if p["id"] in ready), key=lambda p: str(p["id"]))
    became = _became(pairs)
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
        body = {"object": stated_on, "column": column, "value": kept, "forms": forms}
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

    entries, written = [], []
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
    return {"source": SOURCE, "entries": entries}, report


def read_receipts(paths: tuple[Path, ...]) -> dict[tuple[str, str], dict[str, str]]:
    """What a host's receipts say was placed: (element id, the claim it was placed on) -> the
    library and the release that placed it. Of several receipts that placed the same, the
    first given counts, so give them oldest first. What a receipt lists as already there or
    as refused says nothing about since when, and is not read."""
    taken: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fmt = doc.get("format") if isinstance(doc, dict) else None
        if fmt != RECEIPT:
            raise ValueError(f"{path}: the receipt is {fmt!r}; this build reads {RECEIPT}")
        if not doc.get("library") or not doc.get("version"):
            raise ValueError(f"{path}: a receipt names the library and the release that "
                             "took the entries up")
        release = {"library": str(doc["library"]), "version": str(doc["version"])}
        for line in doc.get("placed") or []:
            taken.setdefault((str(line["element"]), str(line["claim"])), release)
    return taken


def in_layer(export: dict[str, Any],
             taken: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], int]:
    """The export with ``in_layer`` (the library and its release) on each element that a
    receipt says was placed as it states now, and that a person's approval is in force on.
    Returns the export and how many elements say so. Without a receipt nothing is said."""
    def marked(el: dict[str, Any]) -> dict[str, Any]:
        release = taken.get((el["id"], str(el.get("claim"))))
        approved = (el.get("decision") or {}).get("verdict") == "approved"
        return {**el, "in_layer": dict(release)} if release is not None and approved else el

    if not taken:
        return export, 0
    concepts = [marked(c) for c in export["concepts"]]
    review = [marked(r) for r in export["review"]]
    count = sum("in_layer" in el for el in concepts + review)
    return {**export, "concepts": concepts, "review": review}, count

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
- a link between records that no person approved: the machine's links were approved by
  nobody, and one certified on simulated labels is such a link;
- an entry whose decision was made on another statement than the build now makes (its claim
  differs): it is listed as stale;
- an entry whose decision states no claim, or no day while none was given: nothing is made up.

A link between two records that a person approved is written as a ``link`` entry, in the
shape the host's tool reads (its ticket 294): the master list's record (``record``) and the
other source's record (``linked``), each by object, key column and key, the statement codes the
person saw (``evidence``), the pair's probability where the linker gave one, and the decision.
Such a file states ``format: layer-proposal/2``; a file with no link states no format and is
version 1, byte for byte as before. A link is written only where the host would take it:

- neither record seeded, and both stated in full (a single-column key, a key);
- two records of two objects (no self-link, no two records of one object);
- the master record of a list whose names this file carries (so the record is a glossary value
  of the master object and column the names entry states);
- a linked record linked to that master and to no other: not by another approved decision, and
  not by a link of the build;
- the statements the person saw stated (no link without its evidence);
- ``certified_on`` only as ``reviewer``, and only where the build's resolution says so.

Three entries over concepts (ticket 58, the host's ticket 304 as the owner decided it on 15
Nov) make a file ``format: layer-proposal/3``:

- ``match``: an element that says two domains' columns are one concept (a value set whose
  objects lie in two or more domains, or two value sets called equivalent across domains),
  keyed to the host's key the build read from the host's alias table (the export's ``match``;
  never derived from a name), with one ``{domain, object, column}`` per domain. A reviewer
  approved it, and every domain it names is certified by that domain's data owner on the claim
  it states now (``origin.certified``). Written only where the caller names the domains (by
  schema, each with its data owner); a close match is never written (``close: false``).
- ``broader``: an approved step between two value sets (a roll-up the data proved): one entry
  per member of the finer set, naming the codes it rolls up to and what the data showed
  (``support``: the member's rows and the share of them that carry those codes).
- ``labels``: a concept's words in one language that a person approved as such (the element
  ``labels:<concept id>@<language>``, decided on its own claim): its preferred label and its
  other labels, on the object and column the host's dimension reads (a value set's column, or a
  master list's key).

A file with no such entry is version 2 or 1, byte for byte as before.

The file is YAML in block style with every text in double quotes, so that no reader takes a
name for a number, a date or a truth value. The same inputs give the same bytes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from latence_core.errors import ConfigError

SOURCE = "ontology"  # the kind of source the entries come from, as the host's layer records it
# The versions of the file. Version 1 states no ``format`` and holds spellings and names;
# version 2 states ``format: layer-proposal/2`` and adds ``link`` entries (ticket 45, the
# host's ticket 294). A file is version 2 only when it holds a link.
# Version 3 states ``format: layer-proposal/3`` and adds ``match``, ``broader`` and ``labels``
# entries (ticket 58, the host's ticket 304); a file is version 3 only when it holds one.
VERSIONS = (1, 2, 3)
SCHEMA_VERSION = VERSIONS[-1]
FORMAT_2 = "layer-proposal/2"
FORMAT_3 = "layer-proposal/3"
CONCEPT_ENTRIES = ("match", "broader", "labels")
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
    """A mapping or a list as YAML lines in block style. A text is always quoted; a number (a
    link's probability) is written as a number."""
    pad = " " * indent
    out: list[str] = []
    if isinstance(node, Mapping):
        for name, value in node.items():
            if isinstance(value, Mapping | list) and value:
                out += [f"{pad}{_key(name)}:", *_lines(value, indent + 2)]
            elif isinstance(value, Mapping | list):
                out.append(f"{pad}{_key(name)}: {'{}' if isinstance(value, Mapping) else '[]'}")
            elif isinstance(value, bool):  # a truth value (a match's `close`), read as one
                out.append(f"{pad}{_key(name)}: {'true' if value else 'false'}")
            elif isinstance(value, int | float):
                out.append(f"{pad}{_key(name)}: {json.dumps(value)}")  # a number, read as one
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


def _decided_links(export: Mapping[str, Any]) -> dict[frozenset[str], Mapping[str, Any]]:
    """The links a build drew on a person's decision that state the pair's claim, by pair."""
    return {frozenset((str(e["from"]), str(e["to"]))): e for e in export.get("edges", [])
            if e.get("kind") == "same_as" and e.get("decided_by") and e.get("claim")}


def _linked_to(export: Mapping[str, Any]) -> dict[str, set[str]]:
    """Every record the build links, with the records it links it to (either way round)."""
    out: dict[str, set[str]] = {}
    for e in export.get("edges", []):
        if e.get("kind") == "same_as":
            out.setdefault(str(e["from"]), set()).add(str(e["to"]))
            out.setdefault(str(e["to"]), set()).add(str(e["from"]))
    return out


def _domain_of(obj: str, domains: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The domain an object (``<schema>.<table>``) lies in, by its schema."""
    schema = obj.partition(".")[0]
    found = [d for d, spec in sorted(domains.items()) if schema in (spec.get("schemas") or [])]
    return found[0] if len(found) == 1 else None


def _member_code(node_id: str, column: str) -> str:
    return node_id.removeprefix(f"value:{column}:")


def proposal(export: Mapping[str, Any], decisions: Mapping[str, Mapping[str, Any]],
             names_of: tuple[str, ...] = (), spellings_on: Mapping[str, str] | None = None,
             decided_on: str | None = None,
             links: bool = True, concepts_too: bool = True,
             domains: Mapping[str, Mapping[str, Any]] | None = None,
             certifications: Sequence[Mapping[str, Any]] | None = None,
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The proposal file as data, and a report of what was written and what was not.

    ``export`` is a build's ``graph-export/3``; ``decisions`` the decisions in force, by
    element id. ``names_of`` are the master lists whose records' names may be written, beside
    those the export marks; ``spellings_on`` the object to state a value set's spellings on (by
    column); and ``decided_on`` the day of the approved decisions that state none (the day a
    decision states, ``at``, a date or a timestamp, always wins). ``links=False`` writes no
    link, for a host whose tool reads version 1 only; the file is then version 1 whatever was
    approved. ``concepts_too=False`` writes no ``match``, ``broader`` or ``labels`` entry, for a
    host whose tool reads version 2 at most. ``domains`` are the estate's domains (``{name:
    {schemas: [...], data_owner: <id>}}``), without which no ``match`` is written, and
    ``certifications`` the data owners' certifications (``{subject, by, at, claim}``).

    The report holds ``entries`` (one line per entry written), ``stale`` (approved on another
    statement), ``not_written`` (approved, with the reason: ``seeded``, ``no_claim``,
    ``no_day``, ``not_in_the_build``, ``no_records``; for a link also ``one_object``,
    ``master_not_in_the_file``, ``linked_elsewhere``, ``no_evidence``; for a match also
    ``no_key``, ``one_domain``, ``two_columns_in_a_domain``, ``not_certified``; for a step
    ``no_members``) and ``not_carried``
    (approved elements of a kind this file has no entry for, counted by kind)."""
    given = _day(decided_on, "the command line") if decided_on else None
    decided_links = _decided_links(export)
    pending: dict[str, tuple[str, str, Mapping[str, Any]]] = {}  # element id -> the pair's ends
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
    structure = {el["id"]: el for part in (export.get("structure") or {}).values()
                 if isinstance(part, list) for el in part if isinstance(el, Mapping)}
    matched: list[str] = []  # the elements a match entry may be written for
    stepped: list[str] = []  # the steps between value sets a broader entry may be written for
    worded: dict[str, tuple[str, str]] = {}  # labels element -> (concept id, language)
    known = domains or {}

    def seeded_set(value_set: Mapping[str, Any]) -> bool:
        return bool(seeded & set(value_set.get("objects") or ()))

    def matchable(value_sets: list[Mapping[str, Any]]) -> bool:
        """Whether the value sets lie in two domains or more (so that a match may be written)."""
        named = {_domain_of(str(o), known) for v in value_sets for o in v.get("objects") or []}
        return len(named - {None}) >= 2

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
        if subject.startswith("labels:"):
            concept_id, _, lang = subject.removeprefix("labels:").rpartition("@")
            labelled = concepts.get(concept_id)
            now = ((labelled or {}).get("labels") or {}).get("claims", {}).get(lang)
            if labelled is None or now is None:
                held_back(subject, "not_in_the_build")
            elif labelled.get("seeded") or (labelled.get("kind") == "reference"
                                            and seeded_set(labelled)):
                held_back(subject, "seeded")
            elif not concepts_too:
                carried_not("labels")
            else:
                worded[subject] = (concept_id, lang)
                origin_of(subject, d, str(now))
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
            elif not links:
                carried_not("pair_of_records")
            else:
                ends = (str(pair["a"]), str(pair["b"]))
                edge = decided_links.get(frozenset(ends)) or {}
                now = pair.get("claim") or edge.get("claim")
                if now is None:  # neither listed nor linked on a person's decision in this build
                    held_back(subject, "not_in_the_build")
                else:
                    pending[subject] = (ends[0], ends[1], pair if pair.get("claim") else edge)
                    origin_of(subject, d, str(now))
        elif (subject in concepts and concepts[subject].get("kind") == "reference" and known
              and concepts_too and matchable([concepts[subject]])):
            if seeded_set(concepts[subject]):
                held_back(subject, "seeded")
            else:
                matched.append(subject)
                origin_of(subject, d, str(concepts[subject]["claim"]))
        elif subject in concepts:
            concept = concepts[subject]
            if concept.get("seeded"):
                held_back(subject, "seeded")
            elif concept.get("object_name") not in names_of and concept.get("proposal") != "names":
                carried_not("concept")
            else:
                origin_of(subject, d, concept["claim"])
        elif (kinds.get(subject) == "roll_up" and concepts_too
              and str(structure[subject].get("from", "")).startswith("concept:value:")
              and str(structure[subject].get("to", "")).startswith("concept:value:")):
            step = structure[subject]
            sides = [concepts.get(str(step["from"])), concepts.get(str(step["to"]))]
            if any(e is None for e in sides):
                held_back(subject, "not_in_the_build")
            elif any(seeded_set(e) for e in sides if e is not None):
                held_back(subject, "seeded")
            else:
                stepped.append(subject)
                origin_of(subject, d, str(step["claim"]))
        elif (kinds.get(subject) == "equivalent" and known and concepts_too
              and all(str(c) in concepts for c in structure[subject].get("concepts") or [])
              and matchable([concepts[str(c)] for c in structure[subject]["concepts"]])):
            sets = [concepts[str(c)] for c in structure[subject]["concepts"]]
            if any(seeded_set(v) for v in sets):
                held_back(subject, "seeded")
            else:
                matched.append(subject)
                origin_of(subject, d, str(structure[subject]["claim"]))
        elif subject in kinds:
            carried_not(kinds[subject])
        else:
            held_back(subject, "not_in_the_build")

    # (the element, the entry's body under its kind, its line in the report)
    found: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pairs = sorted((p for p in export["review"] if p["id"] in ready and p["id"] not in pending),
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
    for subject in sorted(s for s in ready if s in concepts and s not in matched):
        concept = concepts[subject]
        rows = {str(n["key"]): str(n["label"]) for n in export["nodes"]
                if n["concept"] == subject and "key" in n}
        if len(concept.get("key") or []) != 1 or not rows:
            held_back(subject, "no_records")
            continue
        body = {"object": concept["object_name"], "column": concept["key"][0]}
        found.append((subject, {"names": {**body, "values": dict(sorted(rows.items()))}},
                      {"kind": "names", **body, "records": len(rows)}))

    named = {str(body["object"]) for _s, entry, _l in found if "names" in entry
             for body in (entry["names"],)}
    certified_on = (export.get("meta") or {}).get("resolution") or {}
    written_links = _links(pending, ready, nodes, concepts, named, _linked_to(export), held_back)
    for subject, record, linked, said in written_links:
        body = {"record": record, "linked": linked,
                "evidence": [{"code": str(s["code"]), "params": dict(s.get("params") or {})}
                             for s in said["evidence"]],
                **({"p": round(float(said["p"]), 4)} if said.get("p") is not None else {})}
        found.append((subject, {"link": body},
                      {"kind": "link", "record": record, "linked": linked}))
    reviewer = certified_on.get("certified_on") == "reviewer"

    certified: dict[str, list[Mapping[str, Any]]] = {}
    for c in certifications or []:
        certified.setdefault(str(c["subject"]), []).append(c)
    for subject in sorted(s for s in matched if s in ready):
        sets = ([concepts[subject]] if subject in concepts
                else [concepts[str(c)] for c in structure[subject]["concepts"]])
        keys = sorted({str((v.get("match") or {}).get("key")) for v in sets
                       if (v.get("match") or {}).get("key")})
        per_domain: dict[str, list[dict[str, str]]] = {}
        for v in sets:
            held_in: dict[str, list[str]] = {}
            for o in v.get("objects") or []:
                domain = _domain_of(str(o), known)
                if domain is not None:
                    held_in.setdefault(domain, []).append(str(o))
            for domain, objects in sorted(held_in.items()):
                one = _object_of({**v, "objects": objects}, export["relations"], None)
                per_domain.setdefault(domain, []).append(
                    {"domain": domain, "object": one, "column": str(v["column"])})
        owners = {d: str(spec.get("data_owner") or "") for d, spec in known.items()}
        signed = {str(c["by"]): c for c in certified.get(subject, [])
                  if str(c.get("claim")) == ready[subject]["claim"]}
        if len(keys) != 1:
            held_back(subject, "no_key")
        elif len(per_domain) < 2:
            held_back(subject, "one_domain")
        elif any(len(v) != 1 for v in per_domain.values()):
            held_back(subject, "two_columns_in_a_domain")
        elif any(not owners.get(d) or owners[d] not in signed for d in per_domain):
            held_back(subject, "not_certified")
        else:
            body = {"key": keys[0], "concepts": [v[0] for _d, v in sorted(per_domain.items())],
                    "close": False}
            by_domain = [{"domain": d, "by": owners[d],
                          "at": _day(signed[owners[d]]["at"], f"the certification of {subject}")}
                         for d in sorted(per_domain)]
            found.append((subject, {"match": body},
                          {"kind": "match", "key": keys[0], "domains": sorted(per_domain),
                           "_certified": by_domain}))

    node_rows = {str(n["id"]): int(n.get("rows") or 0) for n in export["nodes"]}
    for subject in sorted(s for s in stepped if s in ready):
        step = structure[subject]
        finer, coarser = concepts[str(step["from"])], concepts[str(step["to"])]
        up: dict[str, dict[str, int]] = {}
        for e in export.get("edges", []):
            if e.get("kind") == "broader" and e.get("relation") == subject:
                up.setdefault(str(e["from"]), {})[str(e["to"])] = int(e.get("rows") or 0)
        if not up:
            held_back(subject, "no_members")
            continue
        column = str(finer["column"])
        stated_on = _object_of(finer, export["relations"], (spellings_on or {}).get(column))
        for member, parents in sorted(up.items()):
            carried = sum(parents.values())
            whole = node_rows.get(member) or carried
            share = round(carried / whole, 4) if whole else 0.0
            body = {"object": stated_on, "column": column, "value": _member_code(member, column),
                    "broader": sorted(_member_code(p, str(coarser["column"])) for p in parents),
                    "support": {"rows": carried, "share": share}}
            found.append((subject, {"broader": body}, {"kind": "broader", **body}))

    for subject in sorted(s for s in worded if s in ready):
        concept_id, lang = worded[subject]
        labelled = concepts[concept_id]
        said = labelled["labels"]
        if labelled.get("kind") == "reference":
            column = str(labelled["column"])
            where = {"object": _object_of(labelled, export["relations"],
                                          (spellings_on or {}).get(column)), "column": column}
        elif len(labelled.get("key") or []) == 1 and labelled.get("object_name"):
            where = {"object": str(labelled["object_name"]), "column": str(labelled["key"][0])}
        else:
            held_back(subject, "no_records")
            continue
        pref = [t["text"] for t in said.get("pref") or [] if t["lang"] == lang]
        if not pref:
            held_back(subject, "not_in_the_build")
            continue
        body = {**where, "language": lang, "names": pref[0],
                "synonyms": sorted(t["text"] for t in said.get("alt") or [] if t["lang"] == lang)}
        found.append((subject, {"labels": body}, {"kind": "labels", **body}))

    entries: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    for number, (subject, entry, line) in enumerate(found, start=1):
        o = ready[subject]
        decision = {"verdict": "approved", "by": o["by"], "at": o["at"]}
        origin: dict[str, Any] = {"element": subject, "claim": o["claim"], "decision": decision}
        if "link" in entry and reviewer:
            origin["certified_on"] = "reviewer"
        if "match" in entry:
            origin["certified"] = line["_certified"]
        entries.append({**entry, "origin": origin})
        written.append({"entry": number, "kind": line["kind"], "element": subject,
                        "claim": o["claim"], "decided_by": o["by"], "decided_on": o["at"],
                        "day_stated_by": o["day_stated_by"],
                        **{k: v for k, v in line.items() if k not in ("kind", "_certified")}})
    report = {"entries": written, "stale": sorted(stale, key=lambda s: s["subject"]),
              "not_written": sorted(not_written, key=lambda s: s["subject"]),
              "not_carried": dict(sorted(not_carried.items()))}
    head = ({"format": FORMAT_3} if any(k in e for e in entries for k in CONCEPT_ENTRIES)
            else {"format": FORMAT_2} if any("link" in e for e in entries) else {})
    return {**head, "source": SOURCE, "entries": entries}, report


def _links(pending: Mapping[str, tuple[str, str, Mapping[str, Any]]], ready: Mapping[str, Any],
           nodes: Mapping[str, Mapping[str, Any]], concepts: Mapping[str, Mapping[str, Any]],
           named: set[str], linked_to: Mapping[str, set[str]],
           held_back: Any) -> list[tuple[str, dict[str, str], dict[str, str], Mapping[str, Any]]]:
    """The links the host would take, each as (element, master record, linked record, what the
    person saw); every other one is held back with the reason."""
    def stated(end: str) -> dict[str, str] | None:
        node = nodes.get(end) or {}
        owner = concepts.get(str(node.get("concept"))) or {}
        if "key" not in node or len(owner.get("key") or []) != 1 or not owner.get("object_name"):
            return None
        return {"object": str(owner["object_name"]), "column": str(owner["key"][0]),
                "key": str(node["key"])}

    candidates: list[tuple[str, str, dict[str, str], str, dict[str, str], Mapping[str, Any]]] = []
    for subject in sorted(s for s in pending if s in ready):
        a, b, said = pending[subject]
        ends = [(end, stated(end)) for end in (a, b)]
        if any(r is None for _e, r in ends):
            held_back(subject, "no_records")
            continue
        (ea, ra), (eb, rb) = ends
        assert ra is not None and rb is not None
        if ra["object"] == rb["object"]:
            held_back(subject, "one_object")
            continue
        masters = [(e, r, o, ro) for (e, r), (o, ro) in (((ea, ra), (eb, rb)), ((eb, rb), (ea, ra)))
                   if r["object"] in named]
        if len(masters) != 1:
            held_back(subject, "master_not_in_the_file")
            continue
        master, record, other, linked = masters[0]
        seen = said.get("reasons") or said.get("evidence") or []
        shown = [s for s in seen if s.get("kind") != "source"]  # the sources are no reason
        if not shown:
            held_back(subject, "no_evidence")
            continue
        candidates.append((subject, master, record, other, linked, {**said, "evidence": shown}))
    twice = {c[3] for c in candidates if sum(d[3] == c[3] for d in candidates) > 1}
    out: list[tuple[str, dict[str, str], dict[str, str], Mapping[str, Any]]] = []
    for subject, master, record, other, linked, said in candidates:
        if other in twice or linked_to.get(other, set()) - {master}:
            held_back(subject, "linked_elsewhere")
            continue
        out.append((subject, record, linked, said))
    return out

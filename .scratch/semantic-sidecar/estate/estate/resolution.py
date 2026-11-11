"""Records of two sources linked by the resolution engine, brought into the export.

The estate build hands the engine one file of records (``write_records``), runs it
(``run_resolver``) and reads back its answer (``resolution/3``): which pairs it accepts under a
certified line, which it leaves for a person, which it asks a person to label, and what each
rests on. ``merge`` turns that into ``same_as`` edges, review items, label requests and a
summary.

The engine needs labelled pairs to fit its scorer and to certify its lines, and it says which.
A reviewer's answers reach it from the decisions file (``pair_answers``): an answer on a pair
is a label where the engine asked for that pair, and it decides the pair in any case. Where
the estate allows it, the truth of a seeded source stands in for the answers that are
missing, and the summary says how many labels came from a person and how many did not.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import duckdb

from .codes import check
from .induce import _q
from .reader import Reader

COLUMNS = ("unique_id", "source_system", "entity_type", "name", "city", "postcode", "country", "industry", "reg_id", "cluster")


def write_records(reader: Reader, link: list[dict[str, Any]], truth: dict[str, str] | None, path: Path) -> dict[str, int]:
    """One row per record of every object in ``link``, in the layout the engine reads. A
    record's id is its node id in the export. The first object is the master: its truth is its
    own key. Every other object's truth comes from ``truth`` (key -> the master key it belongs
    to, or a name of its own for a record that belongs to none)."""
    objects = {o.full_name: o for o in reader.objects()}
    rows: list[tuple[Any, ...]] = []
    counts: dict[str, int] = {}
    for k, spec in enumerate(link):
        obj = objects[spec["object"]]
        for col in (spec["id"], spec["name"], spec.get("city"), spec.get("country")):
            if col and (obj.column(col) is None or obj.column(col).sensitive):  # type: ignore[union-attr]
                raise ValueError(f"{spec['object']}.{col} is missing or personal and cannot be matched on")
        select = ", ".join(f"CAST({_q(spec[c])} AS VARCHAR)" if spec.get(c) else "NULL" for c in ("id", "name", "city", "country"))
        renames = spec.get("country_map", {})
        for key, name, city, country in reader.query(f"SELECT {select} FROM {reader.quote(obj)} ORDER BY 1"):
            # Without a truth every record is its own: nothing then stands in for a reviewer.
            owner = key if k == 0 else f"{spec['object']}:{key}" if truth is None else truth.get(key)
            if owner is None:
                raise ValueError(f"no truth for {spec['object']}:{key}: the engine cannot be given labels")
            rows.append((f"{spec['object']}:{key}", spec["source"], "organization", name, city, None, renames.get(country, country), None, None, owner))
        counts[spec["source"]] = len(rows) - sum(counts.values())
    con = duckdb.connect()
    con.execute(f"CREATE TABLE r ({', '.join(c + ' VARCHAR' for c in COLUMNS)})")
    con.executemany(f"INSERT INTO r VALUES ({', '.join('?' for _ in COLUMNS)})", rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM r ORDER BY unique_id) TO '{path}' (FORMAT parquet)")
    return counts


def pair_answers(decisions: dict[str, dict[str, Any]], objects: list[str]) -> list[dict[str, Any]]:
    """A reviewer's answers on pairs of records, from the decisions in force: approved means
    the two are the same, rejected that they are different. A pair of value names is none."""
    out = []
    for subject, d in sorted(decisions.items()):
        if not subject.startswith("same:") or "~" not in subject:
            continue
        a, b = subject.removeprefix("same:").split("~", 1)
        if any(a.startswith(o + ":") for o in objects) and any(b.startswith(o + ":") for o in objects):
            out.append({"a": a, "b": b, "same": d["verdict"] == "approved", "by": d["by"]})
    return out


def run_resolver(resolver_dir: Path, records: Path, out: Path, n_fit: int, n_test: int, duplicate_free: tuple[str, ...] = (),
                 answers: list[dict[str, Any]] | None = None, truth_stands_in: bool = True) -> dict[str, Any]:
    """Run the engine's linker in its own environment and read its answer."""
    python = resolver_dir / ".venv" / "bin" / "python"
    extra: list[str] = []
    if answers:
        given = out.with_name("pair_answers.json")
        given.write_text(json.dumps(answers, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        extra += ["--labels", str(given.resolve())]
    if not truth_stands_in:
        extra.append("--no-truth")
    done = subprocess.run(
        [str(python), "link_records.py", "--records", str(records.resolve()), "--out", str(out.resolve()), "--n-cal", str(n_fit), "--n-ltt", str(n_test),
         *[x for s in duplicate_free for x in ("--duplicate-free", s)], *extra],
        cwd=resolver_dir, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"the resolver stopped with exit {done.returncode}:\n{done.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def merge(instances: dict[str, Any], resolution: dict[str, Any], link: dict[str, Any], truth: dict[str, str] | None,
          concept_of: dict[str, str], seeded: bool, max_review: int = 200, answers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The engine's answer as part of the instance level.

    - every record the engine put in a cluster with exactly one master record gets a
      ``same_as`` edge to it, with the probability and what the pair rests on;
    - a reviewer's answer on a pair decides it, whatever the engine made of it: "the same"
      links the record to that master record and to no other, "different" takes the link away.
      Such an edge says who decided it;
    - the best pairs the engine left undecided become review items, without those a reviewer
      has answered;
    - the pairs the engine asks a reviewer to label are carried as ``label_requests``;
    - ``meta.resolution`` says what was certified and on whose labels and, where the truth
      is known, how right the machine was.

    The result is a new instances dict; the one passed in is not changed."""
    if resolution.get("format") != "resolution/3":
        raise ValueError(f"the resolver's answer is {resolution.get('format')!r}; this build reads resolution/3")
    asked = [d for sample in resolution["requests"].values() for d in sample]
    for d in resolution["accepted"] + resolution["review"] + asked:
        for e in d["evidence"]:
            check(e)
    master_obj = link["records"][0]["object"]
    prefix = master_obj + ":"
    known = {n["id"] for n in instances["nodes"]}
    accepted = {frozenset((d["a"], d["b"])): d for d in resolution["accepted"]}
    said = {frozenset((d["a"], d["b"])): d for d in answers or []}

    def ends(pair: frozenset[str]) -> tuple[str, str] | None:
        """(master record, other record) of a pair, if it is one of each."""
        masters = [r for r in pair if r.startswith(prefix)]
        others = [r for r in pair if not r.startswith(prefix)]
        return (masters[0], others[0]) if len(masters) == 1 and len(others) == 1 else None

    by_person = {other: master for pair, d in said.items() if d["same"] and (found := ends(pair)) for master, other in [found]}
    members: dict[str, list[str]] = {}
    for record, cluster in resolution["clusters"].items():
        members.setdefault(cluster, []).append(record)
    edges, linked, right, two_masters, taken_away = [], 0, 0, 0, 0
    done: set[str] = set()
    for group in members.values():
        masters = [r for r in group if r.startswith(prefix)]
        if len(masters) > 1:
            two_masters += 1
            continue
        if len(masters) != 1:
            continue
        for r in sorted(group):
            if r == masters[0] or r not in known or r in by_person:
                continue
            answer = said.get(frozenset((r, masters[0])))
            if answer is not None and not answer["same"]:
                taken_away += 1  # the machine linked them and a person says they are different
                continue
            pair = accepted.get(frozenset((r, masters[0])))
            edges.append({"from": r, "to": masters[0], "kind": "same_as", "relation": None,
                          **({"p": pair["p"], "evidence": pair["evidence"]} if pair else {"evidence": []})})
            done.add(r)
    by_machine = len(edges)
    for other, master in sorted(by_person.items()):
        if other in known and master in known:
            pair = accepted.get(frozenset((other, master)))
            edges.append({"from": other, "to": master, "kind": "same_as", "relation": None, "decided_by": said[frozenset((other, master))]["by"],
                          **({"p": pair["p"], "evidence": pair["evidence"]} if pair else {"evidence": []})})
    for e in edges:
        linked += 1
        if truth is not None:
            right += truth.get(e["from"].split(":", 1)[1]) == e["to"].split(":", 1)[1]
    review = []
    for d in resolution["review"]:
        if len(review) >= max_review:
            break
        if d["a"] not in known or d["b"] not in known or frozenset((d["a"], d["b"])) in said:
            continue
        a, b = (d["a"], d["b"]) if d["a"].startswith(prefix) else (d["b"], d["a"])
        review.append({"id": f"same:{a}~{b}", "kind": "possibly_same", "concept": concept_of[master_obj], "a": a, "b": b,
                       "reasons": [e for e in d["evidence"] if e["kind"] != "source"],
                       "strength": "strong" if d["p"] >= 0.5 else "weak", "p": d["p"], "evidence": d["evidence"]})
    requests = {name: [{**d, "kind": "possibly_same", "concept": concept_of[master_obj]} for d in sample if d["a"] in known and d["b"] in known]
                for name, sample in resolution["requests"].items()}
    others = sum(not r.startswith(prefix) for r in resolution["clusters"])
    summary: dict[str, Any] = {
        "seeded": seeded,
        "master": concept_of[master_obj],
        "sources": resolution["sources"],
        "duplicate_free": (resolution.get("duplicate_free") or {}).get("sources", []),
        "records": resolution["records"],
        "candidate_pairs": resolution["candidate_pairs"],
        "state": resolution["state"],
        "labels": {k: resolution["labels"][k] for k in ("training", "accept_test", "reject_test", "total")},
        "truth_stands_in": bool(resolution["truth_stands_in"]),
        "certified_on": resolution["certified_on"],
        "decided_by_a_person": {"linked": len(edges) - by_machine, "links_taken_away": taken_away},
        "target": resolution["lines"]["target"],
        "certified": bool(resolution["certified"]),
        "reject_line_certified": resolution["lines"]["reject_below"] is not None,
        "pairs": resolution["counts"],
        "records_to_link": others,
        "linked": linked,
        "review_pairs": resolution["review_total"],
        "review_pairs_listed": len(review),
        "clusters_with_two_master_records": two_masters,
    }
    if truth is not None:
        belong = sum(not v.startswith("other:") for v in truth.values())
        summary["checked_against_seed"] = {"linked": linked, "linked_right": right, "belong_to_a_master_record": belong,
                                           "share_right": round(right / linked, 4) if linked else None,
                                           "share_found": round(right / belong, 4) if belong else None}
    all_edges = sorted(instances["edges"] + edges, key=lambda e: (e["kind"], e["from"], e["to"], str(e.get("via")), str(e.get("relation"))))
    return {**instances, "edges": all_edges, "review": instances["review"] + review, "label_requests": requests,
            "meta": {**instances["meta"], "edges": len(all_edges), "review_items": len(instances["review"]) + len(review), "resolution": summary}}

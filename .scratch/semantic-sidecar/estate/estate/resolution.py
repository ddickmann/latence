"""Records of two sources linked by the resolution engine, brought into the export.

The estate build hands the engine one file of records (``write_records``), runs it
(``run_resolver``) and reads back its answer (``resolution/1``): which pairs it accepts under a
certified line, which it leaves for a person, and what each rests on. ``merge`` turns that into
``same_as`` edges, review items and a summary.

The engine needs labelled pairs to fit its scorer and to certify its lines. Here they come
from the seed's truth, which stands in for a reviewer; the summary says so. With a real second
source the labels have to come from the review screen; that path is not built.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import duckdb

from .induce import _q
from .reader import Reader

COLUMNS = ("unique_id", "source_system", "entity_type", "name", "city", "postcode", "country", "industry", "reg_id", "cluster")


def write_records(reader: Reader, link: list[dict[str, Any]], truth: dict[str, str], path: Path) -> dict[str, int]:
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
            owner = key if k == 0 else truth.get(key)
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


def run_resolver(resolver_dir: Path, records: Path, out: Path, n_fit: int, n_test: int, duplicate_free: tuple[str, ...] = ()) -> dict[str, Any]:
    """Run the engine's linker in its own environment and read its answer."""
    python = resolver_dir / ".venv" / "bin" / "python"
    done = subprocess.run(
        [str(python), "link_records.py", "--records", str(records.resolve()), "--out", str(out.resolve()), "--n-cal", str(n_fit), "--n-ltt", str(n_test),
         *[x for s in duplicate_free for x in ("--duplicate-free", s)]],
        cwd=resolver_dir, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"the resolver stopped with exit {done.returncode}:\n{done.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def merge(instances: dict[str, Any], resolution: dict[str, Any], link: dict[str, Any], truth: dict[str, str] | None,
          concept_of: dict[str, str], seeded: bool, max_review: int = 200) -> dict[str, Any]:
    """The engine's answer as part of the instance level.

    - every record the engine put in a cluster with exactly one master record gets a
      ``same_as`` edge to it, with the probability and what the pair rests on;
    - the best pairs it left undecided become review items;
    - ``meta.resolution`` says what was certified, what it cost in labels and, where the truth
      is known, how right it was.

    The result is a new instances dict; the one passed in is not changed."""
    master_obj = link["records"][0]["object"]
    prefix = master_obj + ":"
    known = {n["id"] for n in instances["nodes"]}
    accepted = {frozenset((d["a"], d["b"])): d for d in resolution["accepted"]}
    members: dict[str, list[str]] = {}
    for record, cluster in resolution["clusters"].items():
        members.setdefault(cluster, []).append(record)
    edges, linked, right, two_masters = [], 0, 0, 0
    for group in members.values():
        masters = [r for r in group if r.startswith(prefix)]
        if len(masters) > 1:
            two_masters += 1
            continue
        if len(masters) != 1:
            continue
        for r in sorted(group):
            if r == masters[0] or r not in known:
                continue
            pair = accepted.get(frozenset((r, masters[0])))
            edges.append({"from": r, "to": masters[0], "kind": "same_as", "relation": None,
                          **({"p": pair["p"], "evidence": pair["evidence"]} if pair else {"evidence": []})})
            linked += 1
            if truth is not None:
                right += truth.get(r.split(":", 1)[1]) == masters[0].split(":", 1)[1]
    review = []
    for d in resolution["review"][:max_review]:
        if d["a"] not in known or d["b"] not in known:
            continue
        a, b = (d["a"], d["b"]) if d["a"].startswith(prefix) else (d["b"], d["a"])
        review.append({"id": f"same:{a}~{b}", "kind": "possibly_same", "concept": concept_of[master_obj], "a": a, "b": b,
                       "reason": "; ".join(e["detail"] for e in d["evidence"] if e["kind"] != "source"),
                       "strength": "strong" if d["p"] >= 0.5 else "weak", "p": d["p"], "evidence": d["evidence"]})
    others = sum(not r.startswith(prefix) for r in resolution["clusters"])
    summary: dict[str, Any] = {
        "seeded": seeded,
        "master": concept_of[master_obj],
        "sources": resolution["sources"],
        "duplicate_free": (resolution.get("duplicate_free") or {}).get("sources", []),
        "records": resolution["records"],
        "candidate_pairs": resolution["candidate_pairs"],
        "labels": {k: resolution["labels"][k] for k in ("training", "accept_test", "reject_test", "total")},
        "labels_from": "the seed's truth, standing in for a reviewer" if truth is not None else "a reviewer",
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
    return {**instances, "edges": all_edges, "review": instances["review"] + review,
            "meta": {**instances["meta"], "edges": len(all_edges), "review_items": len(instances["review"]) + len(review), "resolution": summary}}

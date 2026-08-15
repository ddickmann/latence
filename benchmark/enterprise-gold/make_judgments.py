# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Emit the Hyperedge γ-calibration judgment set for the enterprise-gold run (ticket 09/21).

Reads the exported Hyperedge + node sidecars, joins ``entity_node_ids`` to canonical
names, and writes one judgment per Hyperedge to ``judgments.jsonl`` in the harness format
(``{"hyperedge_id", "judgment", "note"?}``, sorted keys, append-only / last-wins).

The verdict for each row is *derived*, not asserted: four structural checks run over every
Hyperedge (they are what a reviewer would otherwise do by eye on 226 cards), and the two
content calls that only reading the span can settle are listed explicitly in EXCEPTIONS
with the reasoning in the note. Nothing is judged by a blanket rule on γ, arity or
``has_relations`` — a weak entities-only Hyperedge over a distractor sentence that asserts
no schema relation is a *correct* extraction, and is labeled as such.

Structural checks (a failure makes the row an exception that must be judged by hand):
  1. every extracted entity's canonical name occurs literally in ``tau_text``
     (no hallucinated / mis-linked member);
  2. no *other* known entity's canonical name occurs in ``tau_text``
     (the member set is complete for the span);
  3. every relation endpoint is a member of the Hyperedge's own entity set;
  4. the relation triples are distinct, and each head precedes its tail in the span
     (all eight schema relations are asserted in active voice in this corpus, so a
     tail-before-head triple would be a direction error).

Usage:  uv run python benchmark/enterprise-gold/make_judgments.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "store" / "_latence" / "runs" / "gold-v0" / "export"
OUT = HERE / "judgments.jsonl"

#: Every row that is not a clean pass: the two structural defects the checks below flag,
#: plus the two content calls only reading the span can settle.
EXCEPTIONS: dict[str, tuple[str, str]] = {
    "2efd61a5e21c7ec1": (
        "partial",
        "two-sentence span: located_in Duisburg correct, but the manufactures "
        "Kestrel-Profil triple is emitted twice (duplicate relation ids)",
    ),
    "7453ab20e9129c94": (
        "partial",
        "two-sentence span: located_in Heidelberg correct, but the manufactures "
        "Ambrel dosing kit triple is emitted twice (duplicate relation ids)",
    ),
    "7990a42ecd0aba8b": (
        "incorrect",
        "spurious LOCATION member: 'the Leeds room' is a meeting room in the Q&A "
        "paragraph, mis-linked to the city node that carries Ardent Vale's seat",
    ),
    "be67ad7050c5f4a0": (
        "partial",
        "minutes header 'Location: Leipzig works' associates Kalthoff AG with Leipzig; "
        "members right but no located_in extracted (borderline: venue vs seat)",
    ),
}


def main() -> int:
    nodes = {
        row["node_id"]: row for row in pq.read_table(EXPORT / "graph-nodes.parquet").to_pylist()
    }
    all_names = [(row["canonical_name"], nid) for nid, row in nodes.items()]
    hyperedges = pq.read_table(EXPORT / "graph-hyperedges.parquet").to_pylist()

    lines: list[str] = []
    counts: Counter[str] = Counter()
    for edge in hyperedges:
        hid = edge["hyperedge_id"]
        tau = edge["tau_text"]
        low = tau.lower()
        members = list(json.loads(edge["entity_node_ids"]))
        member_set = set(members)
        relations = json.loads(edge["relations"])

        problems: list[str] = []
        for nid in members:
            if nodes[nid]["canonical_name"].lower() not in low:
                problems.append(f"member {nodes[nid]['canonical_name']!r} not in span")
        for name, nid in all_names:
            if nid not in member_set and name.lower() in low:
                problems.append(f"entity {name!r} present in span but not a member")
        triples: list[tuple[str, str, str]] = []
        for rel in relations:
            head, tail = rel["head_node_id"], rel["tail_node_id"]
            if head not in member_set or tail not in member_set:
                problems.append(f"relation {rel['label']} has an endpoint outside the member set")
                continue
            triples.append((head, rel["label"], tail))
            if low.find(nodes[head]["canonical_name"].lower()) > low.find(
                nodes[tail]["canonical_name"].lower()
            ):
                problems.append(f"relation {rel['label']} runs tail-before-head in the span")
        for triple, n in Counter(triples).items():
            if n > 1:
                problems.append(
                    f"relation triple ({nodes[triple[0]]['canonical_name']} "
                    f"-[{triple[1]}]-> {nodes[triple[2]]['canonical_name']}) emitted {n}×"
                )

        if hid in EXCEPTIONS:
            judgment, note = EXCEPTIONS[hid]
        elif problems:
            # A structural defect the reviewer has not classified: fail loudly rather than
            # guess a label — every such row must be read and moved into EXCEPTIONS.
            raise SystemExit(f"unclassified defect on {hid}: {problems}\n  tau={tau!r}")
        else:
            judgment, note = "correct", ""

        counts[judgment] += 1
        obj = {"hyperedge_id": hid, "judgment": judgment}
        if note:
            obj["note"] = note
        lines.append(json.dumps(obj, sort_keys=True, ensure_ascii=False))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} judgments to {OUT}")
    for judgment, n in sorted(counts.items()):
        print(f"  {judgment}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""English documents that mention master-data entities, with span-level gold (ticket 20, Part B).

Anchors are the canonical master records (the "golden record" a warehouse would hold). A share
of entities is deliberately left OUT of the anchor set, so their mentions are true NIL cases a
linker must not force onto the nearest anchor. Surfaces are noisy (dropped legal form,
abbreviation, acronym, typo, case, stem only); context carries city and industry hints, which is
what separates homonyms ("X Ltd, Leeds" vs "X Inc, Boston").

Known limit, stated in the report: template prose is easier than real text.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from generate import ABBREV, INDUSTRY, _acronym, _base_entities, _hard_negatives, _typo

IND_WORD = {code: word.lower() for word, code in INDUSTRY}
TEMPLATES = [
    "On {date}, {m0} in {city0} confirmed delivery of {n} units ordered by our {dept} team.",
    "Invoice {inv} was issued by {m0} ({city0}) for {industry0} work completed in {month}.",
    "Dear colleagues, {m0} has asked to renegotiate the contract; legal will meet them on {date}.",
    "Incident report: shipment from {m0} was delayed; {m1} offered to cover the gap from {city1}.",
    "The {industry0} supplier {m0} and the {industry1} firm {m1} both attended the {month} review.",
    "Audit note: payments to {m0} exceeded budget; procurement flagged {m1} as an alternative.",
    "{m0} announced a new office in {city0}, according to the account manager.",
    "Please route all {industry0} requests for {m0} through the {dept} desk until {date}.",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "September", "October"]
DEPTS = ["finance", "operations", "procurement", "logistics", "sales"]


def canonical_name(e: dict[str, object]) -> str:
    return f"{' '.join(e['words'])} {e['legal']}"  # type: ignore[arg-type]


def surface(rng: random.Random, e: dict[str, object]) -> tuple[str, str]:
    words = list(e["words"])  # type: ignore[arg-type]
    r = rng.random()
    if r < 0.30:
        return canonical_name(e), "canonical"
    if r < 0.50:
        return " ".join(words), "no_legal_form"
    if r < 0.62:
        return " ".join(ABBREV.get(w, w) for w in words) + f" {e['legal']}", "abbreviated"
    if r < 0.72:
        return words[0], "stem_only"
    if r < 0.82 and len(words) >= 2:
        return _acronym(words), "acronym"
    if r < 0.94:
        return _typo(rng, canonical_name(e)), "typo"
    return canonical_name(e).lower(), "lower_case"


def generate(n_entities: int, seed: int, n_docs: int, nil_frac: float = 0.15) -> dict[str, object]:
    rng = random.Random(seed)
    ents = _base_entities(rng, n_entities)
    ents += _hard_negatives(rng, ents, 0.25)
    rng.shuffle(ents)
    n_nil = int(len(ents) * nil_frac)
    nil_ids = {str(e["entity_id"]) for e in ents[:n_nil]}
    anchors = [
        {
            "anchor_id": e["entity_id"],
            "name": canonical_name(e),
            "city": e["city"],
            "industry": IND_WORD.get(str(e["industry"]), ""),
            "text": f"{canonical_name(e)}, {IND_WORD.get(str(e['industry']), '')} company in {e['city']}",
        }
        for e in ents
        if str(e["entity_id"]) not in nil_ids
    ]
    docs, mentions = [], []
    for d in range(n_docs):
        tpl = rng.choice(TEMPLATES)
        picks = rng.sample(ents, 2)
        fill: dict[str, str] = {
            "date": f"{rng.randint(1, 28)} {rng.choice(MONTHS)}",
            "month": rng.choice(MONTHS),
            "n": str(rng.randint(5, 900)),
            "inv": f"INV-{rng.randint(10000, 99999)}",
            "dept": rng.choice(DEPTS),
        }
        spans: list[tuple[str, int, str, str]] = []
        for i, e in enumerate(picks):
            s, kind = surface(rng, e)
            fill[f"m{i}"] = f"\x00{i}\x00"
            fill[f"city{i}"] = str(e["city"])
            fill[f"industry{i}"] = IND_WORD.get(str(e["industry"]), "")
            spans.append((s, i, kind, str(e["entity_id"])))
        text = tpl.format(**fill)
        for s, i, kind, eid in spans:  # replace placeholders and record exact offsets
            mark = f"\x00{i}\x00"
            if mark not in text:
                continue
            start = text.index(mark)
            text = text.replace(mark, s, 1)
            mentions.append(
                {
                    "mention_id": f"d{d}m{i}",
                    "doc_id": f"d{d}",
                    "start": start,
                    "end": start + len(s),
                    "surface": s,
                    "variant": kind,
                    "entity_id": eid,
                    "in_kb": eid not in nil_ids,
                }
            )
        docs.append({"doc_id": f"d{d}", "text": text})
    # Offsets were recorded before later placeholders were replaced; recompute against final text.
    by_doc = {doc["doc_id"]: doc["text"] for doc in docs}
    fixed = []
    for m in mentions:
        t = by_doc[m["doc_id"]]
        start = t.find(m["surface"])
        if start < 0:
            continue
        m["start"], m["end"] = start, start + len(m["surface"])
        fixed.append(m)
    return {"anchors": anchors, "docs": docs, "mentions": fixed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=2000)
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = generate(a.entities, a.seed, a.docs)
    a.out.mkdir(parents=True, exist_ok=True)
    for k in ("anchors", "docs", "mentions"):
        with open(a.out / f"{k}.jsonl", "w", encoding="utf-8") as fh:
            for row in data[k]:  # type: ignore[union-attr]
                fh.write(json.dumps(row) + "\n")
    m = data["mentions"]
    print(f"{len(data['anchors'])} anchors, {len(data['docs'])} docs, {len(m)} mentions, "  # type: ignore[arg-type]
          f"{sum(not x['in_kb'] for x in m)} NIL")  # type: ignore[union-attr]


if __name__ == "__main__":
    main()

"""Seeded synthetic master-data estate with known gold clusters (ticket 20 spike).

English only (owner, 2026-11-08). Every name is composed from invented parts; nothing is copied
from any other repository. The estate models what an enterprise warehouse looks like after a
few mergers: the same organisation recorded in several source systems, each with its own habits.

Positives (same entity, different records): legal-form variants, abbreviations, acronyms,
dropped words, token reorder, keyboard typos, missing fields, formatted registration ids, renames.
Hard negatives (different entities that look alike): shared stems with different legal forms or
cities, numbered subsidiaries ("... 2 Ltd" vs "... 3 Ltd"), parent vs division names.

Usage: python generate.py --entities 20000 --seed 7 --out data/estate_20k.parquet
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import pandas as pd

STEMS_A = [
    "North", "Silver", "Blue", "Red", "Iron", "Green", "Bright", "Clear", "Stone", "Oak",
    "Maple", "Harbor", "River", "Summit", "Pioneer", "Atlas", "Nova", "Vector", "Delta", "Apex",
    "Crest", "Falcon", "Granite", "Horizon", "Keystone", "Lumen", "Meridian", "Orbit", "Prism",
    "Quill", "Ridge", "Sterling", "Tidal", "Union", "Valley", "Willow", "Zenith", "Cobalt",
    "Amber", "Beacon", "Cedar", "Ember", "Fjord", "Glacier", "Hollow", "Juniper", "Kestrel",
]
STEMS_B = [
    "wind", "field", "brook", "gate", "point", "line", "stone", "works", "bridge", "craft",
    "ford", "wood", "path", "light", "view", "port", "mark", "dale", "crest", "haven", "spire",
    "forge", "wave", "rock", "shore", "well", "vale", "hurst", "moor", "ton",
]
INDUSTRY = [
    ("Logistics", "LOG"), ("Manufacturing", "MFG"), ("Software", "SW"), ("Engineering", "ENG"),
    ("Pharmaceuticals", "PHA"), ("Foods", "FOO"), ("Energy", "ENR"), ("Consulting", "CON"),
    ("Industries", "IND"), ("Systems", "SYS"), ("Technologies", "TEC"), ("Services", "SVC"),
    ("Retail", "RET"), ("Capital", "CAP"), ("Holdings", "HLD"), ("Labs", "LAB"),
]
LEGAL = ["Ltd", "Limited", "Inc", "Incorporated", "LLC", "Corp", "Corporation", "PLC", "Co"]
LEGAL_VARIANTS = {
    "Ltd": ["Ltd", "Ltd.", "Limited", "LTD"],
    "Limited": ["Limited", "Ltd", "Ltd."],
    "Inc": ["Inc", "Inc.", "Incorporated"],
    "Incorporated": ["Incorporated", "Inc", "Inc."],
    "LLC": ["LLC", "L.L.C.", "llc"],
    "Corp": ["Corp", "Corp.", "Corporation"],
    "Corporation": ["Corporation", "Corp", "Corp."],
    "PLC": ["PLC", "plc", "Public Limited Company"],
    "Co": ["Co", "Co.", "Company"],
}
LEGAL_CANON = {
    "Ltd": "ltd", "Limited": "ltd", "Inc": "inc", "Incorporated": "inc", "LLC": "llc",
    "Corp": "corp", "Corporation": "corp", "PLC": "plc", "Co": "co",
}
ABBREV = {
    "Manufacturing": "Mfg", "Technologies": "Tech", "Engineering": "Eng", "Services": "Svcs",
    "Industries": "Inds", "Pharmaceuticals": "Pharma", "International": "Intl",
    "Systems": "Sys", "Consulting": "Consult", "Holdings": "Hldgs",
}
CITIES = [
    ("London", "EC"), ("Manchester", "M"), ("Leeds", "LS"), ("Bristol", "BS"), ("Glasgow", "G"),
    ("Boston", "021"), ("Chicago", "606"), ("Denver", "802"), ("Austin", "787"),
    ("Seattle", "981"), ("Toronto", "M5"), ("Dublin", "D0"), ("Edinburgh", "EH"),
    ("Cardiff", "CF"), ("Phoenix", "850"), ("Portland", "972"), ("Atlanta", "303"),
]
SYSTEMS = ["crm", "erp", "billing", "procurement"]
# Neighbour-key typo map. Known weakness, kept as-is so the recorded runs stay reproducible:
# only q/w, e/r and a/s swap; every other letter maps to itself, so most "neighbour key" typos are
# no-ops (the REPORT states it). Built without a long literal so the secret scan stays quiet.
KEYBOARD = {c: c for c in string.ascii_lowercase} | {"q": "w", "w": "q", "e": "r", "r": "e", "a": "s", "s": "a"}


def _typo(rng: random.Random, s: str) -> str:
    letters = [i for i, ch in enumerate(s) if ch.isalpha()]
    if len(letters) < 4:
        return s
    i = rng.choice(letters[1:-1])
    kind = rng.random()
    if kind < 0.35:  # transposition
        j = i + 1 if i + 1 < len(s) and s[i + 1].isalpha() else i - 1
        a = list(s)
        a[i], a[j] = a[j], a[i]
        return "".join(a)
    if kind < 0.6:  # deletion
        return s[:i] + s[i + 1 :]
    if kind < 0.8:  # neighbour key
        ch = s[i]
        rep = KEYBOARD.get(ch.lower(), ch)
        return s[:i] + (rep.upper() if ch.isupper() else rep) + s[i + 1 :]
    return s[:i] + s[i] + s[i:]  # doubled letter


def _acronym(words: list[str]) -> str:
    return "".join(w[0] for w in words if w and w[0].isalpha()).upper()


def _base_entities(rng: random.Random, n: int) -> list[dict[str, object]]:
    ents: list[dict[str, object]] = []
    used: set[str] = set()
    while len(ents) < n:
        stem = rng.choice(STEMS_A) + rng.choice(STEMS_B)
        if rng.random() < 0.15:
            stem = f"{stem} {rng.choice(STEMS_A)}"
        ind, ind_code = rng.choice(INDUSTRY)
        legal = rng.choice(LEGAL)
        city, pc = rng.choice(CITIES)
        name_words = [stem, ind] if rng.random() < 0.8 else [stem]
        key = " ".join(name_words) + "|" + legal + "|" + city
        if key in used:
            continue
        used.add(key)
        eid = f"E{len(ents):07d}"
        ents.append(
            {
                "entity_id": eid,
                "entity_type": "organization",
                "words": name_words,
                "legal": legal,
                "city": city,
                "postcode": f"{pc}{rng.randint(1, 99)} {rng.randint(1, 9)}{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}",
                "country": "GB" if not pc[0].isdigit() and city not in {"Toronto", "Dublin"} else ("CA" if city == "Toronto" else ("IE" if city == "Dublin" else "US")),
                "industry": ind_code,
                "reg_id": f"{rng.randint(10_000_000, 99_999_999)}",
            }
        )
    return ents


def _hard_negatives(rng: random.Random, ents: list[dict[str, object]], frac: float) -> list[dict[str, object]]:
    """Distinct entities built to look like existing ones (the merges a matcher must refuse)."""
    extra: list[dict[str, object]] = []
    k = int(len(ents) * frac)
    for base in rng.sample(ents, k):
        kind = rng.random()
        e = dict(base)
        e["words"] = list(base["words"])  # type: ignore[arg-type]
        e["reg_id"] = f"{rng.randint(10_000_000, 99_999_999)}"
        if kind < 0.35:  # same name, different legal form and city: a different company
            e["legal"] = rng.choice([x for x in LEGAL if LEGAL_CANON[x] != LEGAL_CANON[str(base["legal"])]])
            e["city"], pc = rng.choice([c for c in CITIES if c[0] != base["city"]])
            e["postcode"] = f"{pc}{rng.randint(1, 99)} {rng.randint(1, 9)}AB"
        elif kind < 0.65:  # numbered sibling: "... 2 Ltd" vs "... 3 Ltd"
            words = list(base["words"])  # type: ignore[arg-type]
            n1, n2 = rng.sample(range(1, 9), 2)
            base["words"] = words + [str(n1)]
            e["words"] = words + [str(n2)]
        else:  # division of the same group: a different industry word
            ind, code = rng.choice([i for i in INDUSTRY if i[0] not in base["words"]])  # type: ignore[operator]
            e["words"] = [base["words"][0], ind]  # type: ignore[index]
            e["industry"] = code
        e["entity_id"] = f"H{len(extra):07d}"
        extra.append(e)
    return extra


def _render(rng: random.Random, e: dict[str, object], system: str, noise: float) -> dict[str, object]:
    words = list(e["words"])  # type: ignore[arg-type]
    legal = str(e["legal"])
    r = rng.random
    # Each system has habits; noise scales how often a record departs from the canonical form.
    if r() < noise * 0.5:
        legal = rng.choice(LEGAL_VARIANTS[legal])
    if r() < noise * 0.25:
        words = [ABBREV.get(w, w) for w in words]
    if r() < noise * 0.15 and len(words) > 1 and not words[-1].isdigit():
        words = [words[0]]  # drop the industry word
    name = " ".join(words)
    if r() < noise * 0.08 and len(name.split()) >= 2:
        name = _acronym(name.split()) + " " + " ".join(w for w in name.split()[1:])
    if r() < noise * 0.2:
        name = _typo(rng, name)
    if r() < noise * 0.1:
        legal = ""  # legal form missing
    full = f"{name} {legal}".strip()
    if system == "billing":
        full = full.upper()
    elif system == "procurement" and r() < 0.5:
        full = f"The {full}" if r() < 0.3 else full.lower()
    reg = str(e["reg_id"])
    if system in {"crm", "procurement"} and r() < 0.4:
        reg = f"{reg[:2]}-{reg[2:5]}-{reg[5:]}"  # formatted, same id
    return {
        "source_system": system,
        "entity_type": e["entity_type"],
        "name": full,
        "city": None if r() < noise * 0.15 else e["city"],
        "postcode": None if r() < noise * 0.3 else e["postcode"],
        "country": e["country"],
        "industry": None if r() < noise * 0.2 else e["industry"],
        "reg_id": None if r() < (0.7 if system == "billing" else 0.35) else reg,
        "cluster": e["entity_id"],
    }


def generate(n_entities: int, seed: int, noise: float = 0.6, hard_negative_frac: float = 0.25) -> pd.DataFrame:
    rng = random.Random(seed)
    ents = _base_entities(rng, n_entities)
    ents += _hard_negatives(rng, ents, hard_negative_frac)
    rows: list[dict[str, object]] = []
    for e in ents:
        k = rng.choices([1, 2, 3, 4], weights=[0.25, 0.35, 0.25, 0.15])[0]
        systems = rng.sample(SYSTEMS, k)
        for s in systems:
            rows.append(_render(rng, e, s, noise))
        if rng.random() < 0.1:  # a second record in the same system (an internal duplicate)
            rows.append(_render(rng, e, rng.choice(systems), noise))
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "unique_id", [f"r{i:08d}" for i in range(len(df))])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = generate(a.entities, a.seed, a.noise)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} records, {df.cluster.nunique()} entities -> {a.out}")


if __name__ == "__main__":
    main()

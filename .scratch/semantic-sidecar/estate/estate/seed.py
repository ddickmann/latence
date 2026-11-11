"""A seeded second source: service accounts of the dealers, as a partner portal would hold them.

The estate has one record per dealer. What a resolution engine is for only shows when a second
system names the same organisations its own way, so this module writes one: several accounts
per dealer, each typed by someone else (legal form left out or spelled differently, a word
abbreviated, a typing slip, the town appended, the town missing), and accounts of workshops
that are not dealers at all, with names made from the same words.

It is generated data and says so everywhere: the table and every column carry a comment, the
concept is flagged ``seeded`` in the export, and the screen shows the flag. The estate's own
tables are not touched: the accounts go into a schema of their own in a copy of the mirror.

Which dealer an account belongs to is the seed's truth. It is written to a file outside the
catalog, is never an input of the matching, and is used for two things only: to stand in for a
reviewer's labels, and to report how right the machine was.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

SCHEMA, TABLE = "partner", "service_accounts"
NOTE = "Seeded by the sidecar for demonstration. Not part of the estate."
LEGAL = ("GmbH & Co. KG", "GmbH", "Limited", "Ltd", "AG", "KG", "PLC")
RESPELL = {"GmbH": ("Gmbh", "G.m.b.H.", "GMBH"), "Ltd": ("Limited", "Ltd.", "LTD"), "Limited": ("Ltd",),
           "GmbH & Co. KG": ("GmbH & Co KG", "GmbH und Co. KG", "GmbH"), "AG": ("A.G.",), "KG": ("K.G.",), "PLC": ("plc",)}
SHORT = {"Autohaus": "AH", "Automobile": "Auto", "Garages": "Garage", "Motors": "Motor", "Group": "Grp"}
TOWN = {"Köln": ("Koeln", "Cologne"), "München": ("Muenchen", "Munich"), "Nürnberg": ("Nuernberg",), "Düsseldorf": ("Duesseldorf",)}
TAIL = {"DE": ("Service", "Werkstatt", "Aftersales"), "GB": ("Service", "Workshop", "Aftersales")}


@dataclass(frozen=True)
class Account:
    account_id: str
    account_name: str
    city: str | None
    country: str
    opened: str


def _key(name: str) -> str:
    """A name without legal form, case and punctuation: what two spellings must not share
    unless they are the same organisation."""
    core = name
    for form in LEGAL:
        core = re.sub(rf"\b{re.escape(form)}\b\.?", " ", core, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", core.lower())


def _split(name: str) -> tuple[str, str]:
    for form in LEGAL:
        if name.endswith(" " + form):
            return name[: -len(form) - 1], form
    return name, ""


def _slip(rng: random.Random, text: str) -> str:
    """One typing slip in a long word: two neighbouring letters swapped, or one left out."""
    words = text.split()
    long = [k for k, w in enumerate(words) if len(w) >= 7 and w.isalpha()]
    if not long:
        return text
    k = rng.choice(long)
    w, i = words[k], rng.randrange(2, len(words[k]) - 2)
    words[k] = w[:i] + w[i + 1] + w[i] + w[i + 2:] if rng.random() < 0.5 else w[:i] + w[i + 1:]
    return " ".join(words)


def spell(rng: random.Random, name: str, city: str | None, market: str) -> tuple[str, str | None]:
    """One way a person at a workshop might type this organisation and its town."""
    core, form = _split(name)
    roll = rng.random()
    if roll < 0.22:
        form = ""
    elif roll < 0.5 and form in RESPELL:
        form = rng.choice(RESPELL[form])
    if rng.random() < 0.25:
        for long, short in SHORT.items():
            if re.search(rf"\b{long}\b", core) and rng.random() < 0.6:
                core = re.sub(rf"\b{long}\b", short, core)
    if rng.random() < 0.12:
        core = _slip(rng, core)
    if rng.random() < 0.18:
        core = f"{core} {rng.choice(TAIL.get(market, TAIL['GB']))}"
    if city and rng.random() < 0.15:
        core = f"{core} {city}"
    out = f"{core} {form}".strip()
    case = rng.random()
    out = out.upper() if case < 0.2 else out.lower() if case < 0.26 else out
    if city is None or rng.random() < 0.1:
        return out, None
    if city in TOWN and rng.random() < 0.3:
        return out, rng.choice(TOWN[city])
    return out, city.upper() if rng.random() < 0.15 else city


def seed_accounts(dealers: list[dict[str, str | None]], seed: int = 11, per_dealer: tuple[int, int] = (2, 8),
                  others: int = 80) -> tuple[list[Account], dict[str, str]]:
    """Accounts for ``dealers`` (each with ``dealer_id``, ``name``, ``city``, ``market``) and for
    ``others`` workshops that are no dealers. Returns the accounts in a shuffled order and the
    truth: account id -> the dealer id, or ``other:<n>`` for a workshop outside the dealer list."""
    rng = random.Random(seed)
    taken: dict[str, set[str]] = {}
    for d in dealers:
        taken.setdefault(_key(str(d["name"])), set()).add(str(d["dealer_id"]))
    drafts: list[tuple[str, str | None, str, str]] = []  # name, city, country, owner

    def add(name: str, city: str | None, market: str, owner: str, namesakes: list[dict[str, str | None]]) -> None:
        """One typed account. A spelling must not read as another organisation's name. Where
        another dealer carries the same name, what tells the two apart is kept as it is: the
        town, and, if they share the town too, the legal form."""
        country = "DE" if market == "DE" else "GB"
        same_town = any(n["city"] == city for n in namesakes)
        form = _split(name)[1]
        for _ in range(8):
            typed, town = spell(rng, name, city, market)
            if namesakes:
                town = city
            if owner not in taken.get(_key(typed), {owner}):
                continue
            if same_town and not typed.lower().endswith(" " + form.lower()):
                continue
            drafts.append((typed, town, country, owner))
            return
        drafts.append((name, city, country, owner))

    for d in sorted(dealers, key=lambda d: str(d["dealer_id"])):
        twins = [o for o in dealers if o is not d and _key(str(o["name"])) == _key(str(d["name"]))]
        for _ in range(rng.randint(*per_dealer)):
            add(str(d["name"]), d["city"], str(d["market"]), str(d["dealer_id"]), twins)
    by_market: dict[str, list[dict[str, str | None]]] = {}
    for d in dealers:
        by_market.setdefault(str(d["market"]), []).append(d)
    made = 0
    while made < others:
        market = rng.choice(sorted(by_market))
        a, b = rng.sample(by_market[market], 2) if len(by_market[market]) > 1 else (by_market[market][0],) * 2
        (core_a, form), (core_b, _) = _split(str(a["name"])), _split(str(b["name"]))
        words = [w for w in core_a.split()[:1] + core_b.split()[-1:] if w]
        name = f"{' '.join(dict.fromkeys(words))} {rng.choice(TAIL.get(market, TAIL['GB']))} {form}".strip()
        if _key(name) in taken:
            continue
        owner = f"other:{made + 1}"
        taken[_key(name)] = {owner}
        town = rng.choice(by_market[market])["city"]
        for _ in range(rng.randint(1, 4)):
            add(name, town, market, owner, [])
        made += 1
    rng.shuffle(drafts)
    accounts, truth = [], {}
    for k, (name, city, country, owner) in enumerate(drafts, start=1):
        aid = f"SA{k:05d}"
        accounts.append(Account(aid, name, city, country, f"{rng.randint(2015, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"))
        truth[aid] = owner
    return accounts, truth


def create_sql(catalog_schema: str = f"{SCHEMA}") -> list[str]:
    """Statements that create the seeded table, with the comments that mark it."""
    t = f"{catalog_schema}.{TABLE}"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {catalog_schema}",
        f"CREATE OR REPLACE TABLE {t} (account_id VARCHAR(16), account_name VARCHAR(200), city VARCHAR(80), country VARCHAR(2), opened DATE)",
        f"COMMENT ON TABLE {t} IS 'Service accounts of a partner portal. {NOTE}'",
        f"COMMENT ON COLUMN {t}.account_id IS 'Account number in the portal. {NOTE}'",
        f"COMMENT ON COLUMN {t}.account_name IS 'Name of the workshop as typed when the account was opened. {NOTE}'",
        f"COMMENT ON COLUMN {t}.city IS 'Town as typed. {NOTE}'",
        f"COMMENT ON COLUMN {t}.country IS 'Country code. {NOTE}'",
        f"COMMENT ON COLUMN {t}.opened IS 'Day the account was opened. {NOTE}'",
    ]


def _lit(v: str | None) -> str:
    return "NULL" if v is None else "'" + v.replace("'", "''") + "'"


def insert_sql(accounts: list[Account], catalog_schema: str = f"{SCHEMA}", batch: int = 200) -> list[str]:
    t, out = f"{catalog_schema}.{TABLE}", []
    for i in range(0, len(accounts), batch):
        rows = ", ".join(f"({_lit(a.account_id)}, {_lit(a.account_name)}, {_lit(a.city)}, {_lit(a.country)}, DATE {_lit(a.opened)})"
                         for a in accounts[i : i + batch])
        out.append(f"INSERT INTO {t} VALUES {rows}")
    return out

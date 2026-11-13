from __future__ import annotations

import duckdb

from estate.seed import NOTE, _key, create_sql, insert_sql, seed_accounts

DEALERS = [
    {"dealer_id": "H1", "name": "Autohaus Elmhaus GmbH", "city": "Bremen", "market": "DE"},
    {"dealer_id": "H2", "name": "Ashvale Automobile GmbH & Co. KG", "city": "Köln", "market": "DE"},
    {"dealer_id": "H3", "name": "Brookmere Motors Ltd", "city": "Leeds", "market": "UK"},
    {"dealer_id": "H4", "name": "Brookmere Motor Group Ltd", "city": "York", "market": "UK"},
    {"dealer_id": "H5", "name": "O'Dell Garages Ltd", "city": "Hull", "market": "UK"},
    # namesakes: told apart by the town, or, in one town, by the legal form alone
    {"dealer_id": "H6", "name": "Autohaus Fairholm GmbH", "city": "Dresden", "market": "DE"},
    {"dealer_id": "H7", "name": "Autohaus Fairholm GmbH & Co. KG", "city": "Kassel", "market": "DE"},
    {"dealer_id": "H8", "name": "Autohaus Ashvale GmbH", "city": "Leipzig", "market": "DE"},
    {"dealer_id": "H9", "name": "Autohaus Ashvale AG", "city": "Leipzig", "market": "DE"},
]


def test_the_same_seed_gives_the_same_accounts_and_another_seed_others() -> None:
    assert seed_accounts(DEALERS, seed=3, others=4) == seed_accounts(DEALERS, seed=3, others=4)
    assert seed_accounts(DEALERS, seed=3, others=4) != seed_accounts(DEALERS, seed=4, others=4)


def test_every_dealer_has_accounts_and_some_accounts_belong_to_no_dealer() -> None:
    accounts, truth = seed_accounts(DEALERS, seed=3, others=4)
    per = {d["dealer_id"]: sum(v == d["dealer_id"] for v in truth.values()) for d in DEALERS}
    assert all(2 <= n <= 8 for n in per.values())
    assert {v for v in truth.values() if v.startswith("other:")} == {"other:1", "other:2", "other:3", "other:4"}
    assert len(accounts) == len(truth) == len({a.account_id for a in accounts})
    assert {a.country for a in accounts} == {"DE", "GB"}


def test_no_spelling_reads_as_another_organisations_name() -> None:
    # "Brookmere Motor Group" shortened to "Brookmere Motors" would be the other dealer's name,
    # and the truth would then call a correct match wrong.
    for seed in range(40):
        accounts, truth = seed_accounts(DEALERS, seed=seed, others=3)
        names: dict[str, set[str]] = {}
        for d in DEALERS:
            names.setdefault(_key(str(d["name"])), set()).add(str(d["dealer_id"]))
        for a in accounts:
            assert truth[a.account_id] in names.get(_key(a.account_name), {truth[a.account_id]}), (seed, a)


def test_an_account_of_a_dealer_with_a_namesake_can_still_be_told_apart() -> None:
    # Seen on the real dealers (2026-11-11): two dealers whose names differ only in the legal
    # form, and accounts typed without it and without a town. No one could say whose they are.
    by_id = {str(d["dealer_id"]): d for d in DEALERS}
    for seed in range(40):
        accounts, truth = seed_accounts(DEALERS, seed=seed, others=3)
        for a in accounts:
            owner = truth[a.account_id]
            if owner in ("H6", "H7"):  # namesake in another town: the town is kept as it is
                assert a.city == by_id[owner]["city"], (seed, a)
            if owner in ("H8", "H9"):  # namesake in the same town: the legal form is kept as it is
                form = str(by_id[owner]["name"]).rsplit(" ", 1)[1]
                assert a.account_name.lower().endswith(" " + form.lower()) and a.city == "Leipzig", (seed, a)


def test_the_table_is_marked_as_seeded_and_holds_no_answer() -> None:
    accounts, truth = seed_accounts(DEALERS, seed=3, others=4)
    con = duckdb.connect()
    for sql in create_sql() + insert_sql(accounts, batch=7):
        con.execute(sql)
    assert con.execute("SELECT count(*) FROM partner.service_accounts").fetchall()[0][0] == len(accounts)
    assert NOTE in con.execute("SELECT comment FROM duckdb_tables() WHERE table_name = 'service_accounts'").fetchall()[0][0]
    cols = con.execute("SELECT column_name, comment FROM duckdb_columns() WHERE table_name = 'service_accounts'").fetchall()
    assert all(NOTE in c for _, c in cols)
    assert not {"dealer_id", "cluster", "truth"} & {n for n, _ in cols}  # which dealer an account belongs to is not in the catalog
    held = con.execute("SELECT account_name FROM partner.service_accounts WHERE account_name ILIKE '%dell%'").fetchall()
    assert held and all("'" in n or "dell" in n.lower() for (n,) in held)  # an apostrophe survives the load

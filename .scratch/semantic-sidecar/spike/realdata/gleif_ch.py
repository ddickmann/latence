"""Build a real two-register estate: GLEIF legal-entity records of UK companies and the
Companies House free company file, linked by the company number both carry.

GLEIF records the register a legal entity is entered in (RA000585 England and Wales, RA000586
Northern Ireland, RA000587 Scotland) and its number there. Companies House is that register.
The two are kept by different organisations, so a GLEIF record and the Companies House record
with the same number are a true match between independent sources.

The output has the columns ``run_a.py --dataset estate`` reads. ``cluster`` is the company
number (the gold). ``reg_id`` is empty unless ``--with-number`` is given: the matcher must not
see the answer, and the run with the number is only the ceiling.

Sources and terms (read on 2026-11-11): GLEIF golden copy, CC0 1.0; Companies House "Free
Company Data Product", provided free of charge, GOV.UK content under the Open Government
Licence v3.0. The data is downloaded where the script runs and is never committed; only
aggregate results leave that machine. No officer or owner data is read. Sole proprietors are
left out.

Usage: python realdata/gleif_ch.py --gleif <lei2.csv> --ch <BasicCompanyData.csv> \
         --distractors 1000000 --out data/real_gleif_ch.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

UK_REGISTERS = ("RA000585", "RA000586", "RA000587")
NUMBER = "upper(regexp_replace(trim({c}), '\\s', '', 'g'))"
# Purely numeric numbers are written with and without leading zeros: pad to eight digits.
PADDED = "CASE WHEN regexp_full_match({n}, '[0-9]+') THEN lpad({n}, 8, '0') ELSE {n} END"


def build(gleif: Path, ch: Path, distractors: int, with_number: bool, out: Path, seed: int = 7) -> dict[str, int]:
    con = duckdb.connect()
    num_g = PADDED.format(n=NUMBER.format(c='"Entity.RegistrationAuthority.RegistrationAuthorityEntityID"'))
    num_c = PADDED.format(n=NUMBER.format(c='"CompanyNumber"'))
    con.execute(f"""
        CREATE TABLE g AS
        SELECT "LEI" AS lei, "Entity.LegalName" AS name, "Entity.LegalAddress.City" AS city,
               "Entity.LegalAddress.PostalCode" AS postcode, "Entity.LegalAddress.Country" AS country,
               {num_g} AS number
        FROM read_csv('{gleif}', all_varchar=true, header=true)
        WHERE "Entity.RegistrationAuthority.RegistrationAuthorityID" IN {UK_REGISTERS}
          AND "Entity.RegistrationAuthority.RegistrationAuthorityEntityID" IS NOT NULL
          AND coalesce("Entity.EntityCategory", '') <> 'SOLE_PROPRIETOR'
    """)
    con.execute(f"""
        CREATE TABLE c AS
        SELECT {num_c} AS number, "CompanyName" AS name, "RegAddress.PostTown" AS city,
               "RegAddress.PostCode" AS postcode, 'GB' AS country,
               nullif(split_part("SICCode.SicText_1", ' ', 1), 'None') AS industry
        FROM read_csv('{ch}', all_varchar=true, header=true)
    """)
    # One GLEIF record per number keeps the gold a clean one-to-one link (duplicates are counted).
    counts = {
        "gleif_uk_records": con.execute("SELECT count(*) FROM g").fetchone()[0],
        "gleif_uk_numbers": con.execute("SELECT count(DISTINCT number) FROM g").fetchone()[0],
        "companies_house_records": con.execute("SELECT count(*) FROM c").fetchone()[0],
        "linked_numbers": con.execute("SELECT count(DISTINCT g.number) FROM g JOIN c USING (number)").fetchone()[0],
    }
    reg = "number" if with_number else "NULL"
    con.execute(f"""
        COPY (
            SELECT 'g:' || lei AS unique_id, 'gleif' AS source_system, 'organization' AS entity_type, name, city,
                   postcode, country, CAST(NULL AS VARCHAR) AS industry, CAST({reg} AS VARCHAR) AS reg_id,
                   number AS cluster
            FROM g
            UNION ALL
            SELECT 'c:' || number, 'companies_house', 'organization', name, city, postcode, country, industry,
                   CAST({reg} AS VARCHAR), number
            FROM c
            WHERE number IN (SELECT number FROM g)
               OR number IN (SELECT number FROM c WHERE number NOT IN (SELECT number FROM g)
                             ORDER BY hash(number || '{seed}') LIMIT {distractors})
        ) TO '{out}' (FORMAT parquet)
    """)
    counts["estate_records"] = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gleif", type=Path, required=True)
    ap.add_argument("--ch", type=Path, required=True)
    ap.add_argument("--distractors", type=int, default=1_000_000)
    ap.add_argument("--with-number", action="store_true", help="give the matcher the company number (the ceiling)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(build(a.gleif, a.ch, a.distractors, a.with_number, a.out))


if __name__ == "__main__":
    main()

"""A toy estate with planted structure: a key, a clean reference, a reference that breaks for
3% of rows, one that breaks for half of them, a personal column, shared value columns and a
metric view that states one of the joins."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estate.reader_duckdb import DuckDBReader  # noqa: E402

SENSITIVE = ["shop.dealers.contact_email", "shop.owners.name", "shop.vehicles.vin"]
OPTIONS = {"ignore_columns": ("source_line",), "derived_schemas": ("marts",)}

VIEWS = """-- semantics.shop_claims
version: 1.1
source: demo.acme_shop.claims
comment: One row per warranty claim.
joins:
  - name: shop_dealers
    source: demo.acme_shop.dealers
    on: source.dealer_id = shop_dealers.dealer_id
dimensions:
  - name: make
    expr: source.make
    display_name: Make
    comment: The vehicle's brand.
    synonyms:
      - brand
  - name: status
    expr: source.status
    display_name: Claim status
measures:
  - name: claim_count
    expr: COUNT(1)
    display_name: Claims
    comment: The number of claims.
  - name: paid_amount
    expr: SUM(source.paid)
"""


@pytest.fixture(scope="session")
def toy_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("toy") / "toy.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA shop")
    con.execute("CREATE TABLE shop.dealers AS SELECT 'D' || i AS dealer_id, 'Dealer ' || i AS name, "
                "CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS market, 'c' || i || '@example.org' AS contact_email "
                "FROM range(1, 41) t(i)")
    con.execute("CREATE TABLE shop.parts AS SELECT 'P' || i AS part_id, CASE i % 3 WHEN 0 THEN 'BRAKES' WHEN 1 THEN 'ENGINE' ELSE 'BODY' END AS system, "
                "CASE i % 4 WHEN 0 THEN 'ALFA' WHEN 1 THEN 'BETA' WHEN 2 THEN 'GAMA' ELSE 'DELT' END AS make, "
                "'M' || (i % 8) AS model FROM range(1, 101) t(i)")
    con.execute("CREATE TABLE shop.claims AS SELECT 'C' || i AS claim_ref, 'D' || (1 + i % 40) AS dealer_id, "
                "CASE WHEN i % 100 < 3 THEN 'P999' || i ELSE 'P' || (1 + i % 100) END AS part_id, "
                "CASE i % 4 WHEN 0 THEN 'ALFA' WHEN 1 THEN 'BETA' WHEN 2 THEN 'GAMA' ELSE 'DELT' END AS make, "
                "'M' || (i % 8) AS model, "
                "CASE WHEN i % 5 = 0 THEN 'open' ELSE 'paid' END AS status, "
                "CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS market, i * 1.5 AS paid FROM range(1, 2001) t(i)")
    con.execute("CREATE TABLE shop.visits AS SELECT i AS visit_no, CASE WHEN i % 2 = 0 THEN 'D' || (1 + i % 40) ELSE 'X' || i END AS dealer_id, "
                "CASE i % 4 WHEN 0 THEN 'ALFA' WHEN 1 THEN 'BETA' WHEN 2 THEN 'GAMA' ELSE 'DELT' END AS make, 'M' || (i % 8) AS model, "
                "CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS market FROM range(1, 501) t(i)")
    con.execute("CREATE TABLE shop.owners AS SELECT 'O' || i AS owner_id, 'Person ' || i AS name, "
                "CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS market FROM range(1, 301) t(i)")
    con.execute("CREATE TABLE shop.vehicles AS SELECT i AS source_line, 'V' || i AS vin, 'O' || i AS owner_id, "
                "CASE i % 4 WHEN 0 THEN 'ALFA' WHEN 1 THEN 'BETA' WHEN 2 THEN 'GAMA' ELSE 'DELT' END AS make, CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS market, "
                "CASE WHEN i % 2 = 0 THEN 'km' ELSE 'mi' END AS unit FROM range(1, 301) t(i)")
    con.execute("CREATE SCHEMA marts")
    con.execute("CREATE TABLE marts.claims_by_dealer AS SELECT dealer_id, count(*) AS claims FROM shop.claims GROUP BY 1")
    con.execute("COMMENT ON TABLE shop.dealers IS 'One row per dealer.'")
    con.close()
    return path


@pytest.fixture()
def reader(toy_path: Path) -> DuckDBReader:
    return DuckDBReader(toy_path, schemas=["shop", "marts"], sensitive=SENSITIVE, metric_views=VIEWS,
                        schema_prefix="acme_", name="toy")

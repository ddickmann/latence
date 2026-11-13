"""Where the true links of a two-register estate sit among the candidate pairs."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

d = Path(sys.argv[1])
con = duckdb.connect()
con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{d}/pairs.parquet')")
con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{d}/recs.parquet')")
con.execute("""CREATE TABLE j AS SELECT p.*, a.source_system sa, b.source_system sb, a.name na, b.name nb,
    a.name_core ca, b.name_core cb, a.legal_canon la, b.legal_canon lb, a.postcode pa, b.postcode pb, a.city_n cia, b.city_n cib,
    upper(regexp_replace(a.name, '[^A-Za-z0-9]', '', 'g')) xa, upper(regexp_replace(b.name, '[^A-Za-z0-9]', '', 'g')) xb
    FROM p JOIN r a ON a.rid = p.l JOIN r b ON b.rid = p.r""")


def show(title: str, sql: str) -> None:
    print("\n==", title)
    for row in con.execute(sql).fetchall():
        print("  ", row)


show("pairs and true pairs by source pair", """SELECT CASE WHEN sa = sb THEN 'within ' || sa ELSE 'cross' END k, count(*), sum(y::INT), round(avg(y::INT), 5), sum(veto::INT)
    FROM j GROUP BY 1 ORDER BY 1""")
show("cross pairs: how alike are the two names", """SELECT xa = xb AS same_raw, ca = cb AS same_core, coalesce(la = lb, false) AS same_legal,
    upper(replace(pa, ' ', '')) = upper(replace(pb, ' ', '')) AS same_postcode, count(*) n, sum(y::INT) t, round(avg(y::INT), 4) rate
    FROM j WHERE sa <> sb GROUP BY ALL ORDER BY t DESC LIMIT 40""")
show("true pairs by first-stage probability", """SELECT round(p, 1) b, count(*) n, sum(y::INT) t FROM j WHERE sa <> sb GROUP BY 1 ORDER BY 1""")
con.execute("""CREATE TABLE g AS SELECT *, CASE WHEN sa = 'gleif' THEN l ELSE r END AS g, CASE WHEN sa = 'gleif' THEN r ELSE l END AS c FROM j WHERE sa <> sb""")
con.execute("""CREATE TABLE rk AS SELECT *, row_number() OVER (PARTITION BY g ORDER BY p DESC, (xa = xb) DESC) rk, count(*) OVER (PARTITION BY g) n FROM g""")
show("rank of the true record among a GLEIF record's candidates (by first-stage p)", "SELECT least(rk, 6) rk, count(*) FROM rk WHERE y GROUP BY 1 ORDER BY 1")
show("candidates per GLEIF record", "SELECT least(n, 20) n, count(DISTINCT g) FROM rk GROUP BY 1 ORDER BY 1")
show("GLEIF records with / without a true candidate", "SELECT has, count(*) FROM (SELECT g, max(y::INT) has FROM rk GROUP BY g) GROUP BY 1")
show("true pairs that differ in the raw name: sample", """SELECT na, nb, pa, pb, round(p, 3) FROM g WHERE y AND xa <> xb ORDER BY hash(na) LIMIT 40""")
show("false cross pairs with the same core name: sample", """SELECT na, nb, pa, pb, round(p, 3) FROM g WHERE NOT y AND ca = cb ORDER BY hash(na) LIMIT 25""")
show("false cross pairs with identical raw name: count and sample", """SELECT count(*) FROM g WHERE NOT y AND xa = xb""")
show("", """SELECT na, nb, pa, pb FROM g WHERE NOT y AND xa = xb ORDER BY hash(na) LIMIT 15""")

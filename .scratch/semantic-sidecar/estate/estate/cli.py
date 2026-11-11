"""Build the ontology of an estate and write what a screen reads.

  python -m estate build --config estates/vehicles.yaml --duckdb <mirror.duckdb> \\
         --metric-views <views.txt> [--verified <host catalog.yaml>] --out out/vehicles
  python -m estate export --out out/vehicles --ui-dir <host ui>/public/data/ontology

``build`` appends to ``assertions.jsonl`` in the output folder, reads ``decisions.jsonl``
there if a person has written one, and writes ``ontology.json`` (graph-export/2).
``export`` copies the files a screen reads into the host's static folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from .export import build_export
from .induce import induce
from .instances import build_instances
from .ledger import append, assertions_of, read_decisions, replay
from .reader_duckdb import DuckDBReader
from .resolution import merge, run_resolver, write_records
from .seed import SCHEMA, create_sql, insert_sql, seed_accounts

UI_FILES = ("ontology.json",)


def read_verified(path: Path, objects: set[str]) -> list[tuple[str, str]]:
    """A host's verified joins, written ``<layer>_<table>.<layer>_<table>``, as object names."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat = {name.replace(".", "_"): name for name in objects}
    out = []
    for item in doc.get("verified", []):
        child, _, parent = str(item).partition(".")
        if child in flat and parent in flat:
            out.append((flat[child], flat[parent]))
    return out


def build(config: Path, duckdb_path: Path, views: Path | None, verified: Path | None, out: Path,
          resolver: Path | None = None) -> dict[str, object]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    reader = DuckDBReader(
        duckdb_path, schemas=cfg["schemas"], sensitive=cfg.get("sensitive", []),
        metric_views=views.read_text(encoding="utf-8") if views else "",
        schema_prefix=cfg.get("schema_prefix", ""), name=cfg["name"],
    )
    names = {o.full_name for o in reader.objects()}
    joins = read_verified(verified, names) if verified else []
    shown = tuple(cfg.get("without_personal_columns", ()))
    model = induce(reader, joins, tuple(cfg.get("ignore_columns", ())), tuple(cfg.get("derived_schemas", ())), shown)
    seeded = set(cfg.get("seeded_objects", ()))
    for c in model["concepts"]:
        if c.get("object_name") in seeded:
            c["seeded"] = True
    link = cfg.get("link")
    if link and not all(spec["object"] in names for spec in link["records"]):
        link = None  # the second source is not in this estate (the mirror without the seed)
    out.mkdir(parents=True, exist_ok=True)
    new = append(out / "assertions.jsonl", assertions_of(model))
    layer = replay(model, read_decisions(out / "decisions.jsonl"))
    instances = build_instances(reader, layer, tuple(cfg.get("personal_objects", ())), tuple(cfg.get("resolve", ())), shown,
                                also={spec["object"]: spec["name"] for spec in link["records"][1:]} if link else None)
    resolution_note = "no second source in this estate" if link is None else "no resolver given: records not linked"
    if link and resolver is not None:
        truth_file = out / "seed_truth.json"
        truth = json.loads(truth_file.read_text(encoding="utf-8"))["truth"] if truth_file.exists() else None
        if truth is None:
            raise SystemExit("linking needs labelled pairs: no seed truth in the output folder, and labels from the review screen are not read yet")
        write_records(reader, link["records"], truth, out / "records.parquet")
        answer = run_resolver(resolver, out / "records.parquet", out / "resolution.json", link["labels"]["fit"], link["labels"]["test"],
                              tuple(link.get("duplicate_free", ())))
        concept_of = {c["object_name"]: c["id"] for c in layer["concepts"] if c.get("object_name")}
        instances = merge(instances, answer, link, truth, concept_of, seeded=any(s["object"] in seeded for s in link["records"]))
        resolution_note = "linked"
    export = build_export(layer, source=cfg["name"], title=cfg.get("title", cfg["name"]), verified=joins, instances=instances)
    (out / "ontology.json").write_text(json.dumps(export, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"new_assertions": new, **export["meta"]["counts"], "decisions": export["meta"]["decisions"],
            "nodes": instances["meta"]["nodes"], "edges": instances["meta"]["edges"], "review_items": instances["meta"]["review_items"],
            "resolution": instances["meta"].get("resolution") or resolution_note}


def seed(config: Path, duckdb_path: Path, out_db: Path, out: Path, seed_no: int) -> dict[str, object]:
    """Copy the mirror and add the seeded second source to the copy, in a schema of its own.
    The truth goes to ``seed_truth.json`` in the output folder, outside the catalog, together
    with the statements that load the same table into a catalog (``seeded_accounts.sql``)."""
    import duckdb

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    spec = cfg["seeded"]
    shutil.copyfile(duckdb_path, out_db)
    con = duckdb.connect(str(out_db))
    schema, table = spec["from"].split(".")
    rows = con.execute(f'SELECT "{spec["id"]}", "{spec["name"]}", "{spec["city"]}", "{spec["market"]}" FROM "{schema}"."{table}" ORDER BY 1').fetchall()
    dealers = [{"dealer_id": r[0], "name": r[1], "city": r[2], "market": r[3]} for r in rows]
    accounts, truth = seed_accounts(dealers, seed=seed_no)
    for sql in create_sql() + insert_sql(accounts):
        con.execute(sql)
    con.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed_truth.json").write_text(json.dumps({"seed": seed_no, "of": spec["from"], "truth": truth}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    target = f"<catalog>.{cfg.get('schema_prefix', '')}{SCHEMA}"
    head = ["-- The seeded second source, for a catalog. Generated; not part of the estate.",
            "-- Replace <catalog>. Not run against a catalog where it was written: read it before you run it.", ""]
    (out / "seeded_accounts.sql").write_text("\n".join(head + [s + ";" for s in create_sql(target) + insert_sql(accounts, target)]) + "\n", encoding="utf-8")
    return {"accounts": len(accounts), "of_dealers": sum(not v.startswith("other:") for v in truth.values()),
            "of_other_workshops": sum(v.startswith("other:") for v in truth.values()), "mirror_with_seed": str(out_db)}


def main() -> None:
    ap = argparse.ArgumentParser(prog="estate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--duckdb", type=Path, required=True)
    b.add_argument("--metric-views", type=Path)
    b.add_argument("--verified", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--resolver", type=Path, help="folder of the resolution engine (its link_records.py and environment)")
    sd = sub.add_parser("seed")
    sd.add_argument("--config", type=Path, required=True)
    sd.add_argument("--duckdb", type=Path, required=True)
    sd.add_argument("--out-duckdb", type=Path, required=True)
    sd.add_argument("--out", type=Path, required=True)
    sd.add_argument("--seed", type=int, default=11)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--ui-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "seed":
        print(json.dumps(seed(a.config, a.duckdb, a.out_duckdb, a.out, a.seed), indent=1))
    elif a.cmd == "build":
        print(json.dumps(build(a.config, a.duckdb, a.metric_views, a.verified, a.out, a.resolver), indent=1))
    else:
        a.ui_dir.mkdir(parents=True, exist_ok=True)
        for name in UI_FILES:
            shutil.copyfile(a.out / name, a.ui_dir / name)
        print(f"copied {', '.join(UI_FILES)} to {a.ui_dir}")


if __name__ == "__main__":
    main()

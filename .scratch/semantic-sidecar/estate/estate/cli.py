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


def build(config: Path, duckdb_path: Path, views: Path | None, verified: Path | None, out: Path) -> dict[str, object]:
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
    out.mkdir(parents=True, exist_ok=True)
    new = append(out / "assertions.jsonl", assertions_of(model))
    layer = replay(model, read_decisions(out / "decisions.jsonl"))
    instances = build_instances(reader, layer, tuple(cfg.get("personal_objects", ())), tuple(cfg.get("resolve", ())), shown)
    export = build_export(layer, source=cfg["name"], title=cfg.get("title", cfg["name"]), verified=joins, instances=instances)
    (out / "ontology.json").write_text(json.dumps(export, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"new_assertions": new, **export["meta"]["counts"], "decisions": export["meta"]["decisions"],
            "nodes": instances["meta"]["nodes"], "edges": instances["meta"]["edges"], "review_items": instances["meta"]["review_items"]}


def main() -> None:
    ap = argparse.ArgumentParser(prog="estate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--duckdb", type=Path, required=True)
    b.add_argument("--metric-views", type=Path)
    b.add_argument("--verified", type=Path)
    b.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--ui-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.config, a.duckdb, a.metric_views, a.verified, a.out), indent=1))
    else:
        a.ui_dir.mkdir(parents=True, exist_ok=True)
        for name in UI_FILES:
            shutil.copyfile(a.out / name, a.ui_dir / name)
        print(f"copied {', '.join(UI_FILES)} to {a.ui_dir}")


if __name__ == "__main__":
    main()

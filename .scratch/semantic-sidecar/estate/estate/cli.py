"""Build the ontology of an estate and write what a screen reads.

  python -m estate build --config estates/vehicles.yaml --duckdb <mirror.duckdb> \\
         --metric-views <views.txt> [--verified <host catalog.yaml>] --out out/vehicles
  python -m estate export --out out/vehicles --ui-dir <host ui>/public/data/ontology

``build`` appends to ``assertions.jsonl`` in the output folder, reads ``decisions.jsonl``
there if a person has written one, and writes ``ontology.json`` (graph-export/3).
``export`` copies the files a screen reads into the host's static folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .classes import TOP, CachedClassModel, OpenAIClassModel, RecordedClassModel, describe, prompt_of, propose_classes
from .export import build_export
from .induce import induce
from .instances import build_instances, merge_values
from .ledger import Decisions, append, assertions_of, read_decisions, replay
from .reader import Reader
from .reader_databricks import DatabricksReader
from .reader_duckdb import DuckDBReader
from .score import score
from .resolution import merge, pair_answers, run_resolver, write_records
from .seed import SCHEMA, create_sql, insert_sql, seed_accounts
from .structure import broader_edges, build_structure, explain_review
from .turtle import to_turtle
from .usage import attach, read_usage

UI_FILES = ("ontology.json", "ontology.ttl")


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


def build(config: Path, duckdb_path: Path | None, views: Path | None, verified: Path | None, out: Path,
          resolver: Path | None = None, hosted_model: bool = False, class_answer: Path | None = None, max_usd: float = 0.25,
          client: object | None = None, usage: tuple[Path, ...] = (), catalog: str | None = None) -> dict[str, object]:
    """The estate is read from a DuckDB mirror (``duckdb_path``) or from Unity Catalog
    (``catalog``; host, warehouse and credentials from the environment). ``hosted_model`` lets
    a hosted language model propose classes, if the estate allows it. The call is made only
    when what would be sent has changed, and only if its worst case stays under ``max_usd``.
    ``client`` stands in for the provider's client in tests."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    reader: Reader
    if catalog:
        # The schemas of the catalog that exist: a layer the mirror adds (a seeded source) may be missing there.
        reader = DatabricksReader.from_env(catalog, cfg["schemas"], schema_prefix=cfg.get("schema_prefix", ""), sensitive=cfg.get("sensitive", []),
                                           sensitive_tags=cfg.get("sensitive_tags", []))
    elif duckdb_path is not None:
        reader = DuckDBReader(
            duckdb_path, schemas=cfg["schemas"], sensitive=cfg.get("sensitive", []),
            metric_views=views.read_text(encoding="utf-8") if views else "",
            schema_prefix=cfg.get("schema_prefix", ""), name=cfg["name"],
        )
    else:
        raise SystemExit("give the estate to read: --duckdb FILE for a mirror, or --catalog NAME for Unity Catalog")
    names = {o.full_name for o in reader.objects()}
    joins = read_verified(verified, names) if verified else []
    shown = tuple(cfg.get("without_personal_columns", ()))
    model = induce(reader, joins, tuple(cfg.get("ignore_columns", ())), tuple(cfg.get("derived_schemas", ())), shown, labels=cfg.get("labels"))
    usage_meta = None
    if usage:  # what people asked, counted on the elements they used; it changes no status
        model, usage_meta = attach(model, read_usage(list(usage)))
        model = {**model, "meta": {**model["meta"], "usage": usage_meta}}
    seeded = set(cfg.get("seeded_objects", ()))
    for c in model["concepts"]:
        if c.get("object_name") in seeded:
            c["seeded"] = True
    link = cfg.get("link")
    if link and not all(spec["object"] in names for spec in link["records"]):
        link = None  # the second source is not in this estate (the mirror without the seed)
    out.mkdir(parents=True, exist_ok=True)
    decisions = Decisions(read_decisions(out / "decisions.jsonl"))
    layer = replay(model, decisions)
    instances = build_instances(reader, layer, tuple(cfg.get("personal_objects", ())), tuple(cfg.get("resolve", ())), shown,
                                also={spec["object"]: spec["name"] for spec in link["records"][1:]} if link else None,
                                label_columns=cfg.get("label_columns"))
    structure = build_structure(reader, layer, tuple(cfg.get("personal_objects", ())), decisions)
    if hosted_model and not cfg.get("classes", {}).get("hosted_model"):
        raise SystemExit("this estate does not allow its names to be sent to a hosted model (classes.hosted_model in the estate file)")
    personal = tuple(cfg.get("personal_objects", ()))
    voice: CachedClassModel | RecordedClassModel | None = None
    if hosted_model:
        spec = cfg["classes"]
        if "prices" not in spec:
            raise SystemExit("the estate file states no prices for the hosted model (classes.prices): the call cannot be held under a limit")
        hosted = OpenAIClassModel(client=client, model=spec.get("model", "gpt-6-luna"), effort=spec.get("effort", "low"),
                                  languages=tuple(spec.get("languages", ("en", "de"))))
        voice = CachedClassModel(hosted, out / "class_answers", spec["prices"], max_usd, out / "model_calls.jsonl")
        leaves: dict[str, object] = hosted.request(describe(layer, structure, personal), TOP)  # exactly what the call sends
    elif class_answer:
        voice = RecordedClassModel(json.loads(class_answer.read_text(encoding="utf-8")))
    if not hosted_model:
        leaves = prompt_of(describe(layer, structure, personal), TOP)  # what a model session would be given
    (out / "classes_prompt.json").write_text(json.dumps(leaves, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    classes = propose_classes(layer, structure, personal, voice, decisions)
    if isinstance(voice, CachedClassModel) and voice.cost is not None:
        # What the answer in this export cost when it was asked for; a rebuild that asks nothing says the same.
        classes = {**classes, "voices": {**classes["voices"], "cost": {k: v for k, v in voice.cost.items() if k != "call"}}}
        call_note = voice.cost["call"]
    else:
        call_note = "no hosted model"
    new = append(out / "assertions.jsonl", assertions_of(model, structure, classes))
    up = broader_edges(reader, structure, {n["id"] for n in instances["nodes"]})
    edges = sorted(instances["edges"] + up, key=lambda e: (e["kind"], e["from"], e["to"], str(e.get("via")), str(e.get("relation"))))
    instances = {**instances, "edges": edges, "review": explain_review(instances["review"], edges, instances["nodes"], layer["concepts"]),
                 "meta": {**instances["meta"], "edges": len(edges)}}
    resolution_note = "no second source in this estate" if link is None else "no resolver given: records not linked"
    if link and resolver is not None:
        truth_file = out / "seed_truth.json"
        truth = json.loads(truth_file.read_text(encoding="utf-8"))["truth"] if truth_file.exists() else None
        # Whether the truth of a seeded source may stand in for a reviewer's labels is the
        # estate's to say. Without a truth nothing can stand in, and every label is a person's.
        stands_in = bool(link["labels"].get("truth_stands_in", True)) and truth is not None
        answers = pair_answers(decisions.by, [spec["object"] for spec in link["records"]])
        write_records(reader, link["records"], truth, out / "records.parquet")
        answer = run_resolver(resolver, out / "records.parquet", out / "resolution.json", link["labels"]["fit"], link["labels"]["test"],
                              tuple(link.get("duplicate_free", ())), answers, stands_in)
        concept_of = {c["object_name"]: c["id"] for c in layer["concepts"] if c.get("object_name")}
        instances = merge(instances, answer, link, truth, concept_of, seeded=any(s["object"] in seeded for s in link["records"]), answers=answers)
        resolution_note = "linked" if answer["certified"] else answer["state"]
    if link and resolver is not None:
        for said in answers:  # every answer on a pair of records was applied, whether the pair is still listed or not
            decisions.verdict("possibly_same", {"id": f"same:{said['a']}~{said['b']}", "a": said["a"], "b": said["b"]})
    # Names of a value set a reviewer called the same are one value from here on.
    same_names = [(item["a"], item["b"]) for item in instances["review"]
                  if item["a"].startswith("value:") and (v := decisions.verdict("possibly_same", item)) and v["verdict"] == "approved"]
    instances = merge_values(instances, same_names)
    instances = {**instances, "review": [decisions.marked("possibly_same", item) for item in instances["review"]],
                 "label_requests": {name: [decisions.marked("possibly_same", item) for item in sample] for name, sample in instances.get("label_requests", {}).items()}}
    layer = {**layer, "meta": {**layer["meta"], "decisions": decisions.summary()}}
    export = build_export(layer, source=cfg["name"], title=cfg.get("title", cfg["name"]), verified=joins, instances=instances, structure=structure, classes=classes,
                          language=cfg.get("language", "en"))
    (out / "ontology.json").write_text(json.dumps(export, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "ontology.ttl").write_text(to_turtle(export, cfg.get("export", {}).get("base_iri", f"urn:estate:{cfg['name']}:")), encoding="utf-8")
    return {"new_assertions": new, **export["meta"]["counts"], "decisions": export["meta"]["decisions"],
            "nodes": instances["meta"]["nodes"], "edges": instances["meta"]["edges"], "review_items": instances["meta"]["review_items"],
            "structure": {k: len(v) for k, v in export["structure"].items()},
            "classes": {"voices": classes["voices"], "placed": len(classes["subclass_of"]), "contested": sum(s["contested"] for s in classes["subclass_of"]),
                        "own_classes": len(classes["classes"]), "unplaced": len(classes["unplaced"]), "hosted_call": call_note},
            "resolution": instances["meta"].get("resolution") or resolution_note}


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """What two exports of one estate hold differently, by element id: what only one holds,
    and for what both hold, whether its key, its ends or its status differ. Counts on the data
    (rows, support) are left out: a mirror and the catalog may be loaded at different times."""
    def elements(e: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
        out: dict[str, tuple[Any, ...]] = {}
        for c in e["concepts"]:
            out[c["id"]] = (c["kind"], list(c.get("key") or ()), c["status"], c["label"])
        for r in e["relations"]:
            out[r["id"]] = (r["from"], r["to"], list(r["columns"]), r["status"])
        for m in e["measures"]:
            out[m["id"]] = (m["concept"], m["expr"])
        for kind in ("roll_ups", "equivalent", "part_of", "describes"):
            for s in e["structure"].get(kind, []):
                out[s["id"]] = (s.get("kind", kind), s.get("from"), s.get("to"))
        return out

    ea, eb = elements(a), elements(b)
    return {"only_in_a": sorted(set(ea) - set(eb)), "only_in_b": sorted(set(eb) - set(ea)),
            "differ": {k: {"a": list(ea[k]), "b": list(eb[k])} for k in sorted(set(ea) & set(eb)) if ea[k] != eb[k]},
            "same": len([k for k in set(ea) & set(eb) if ea[k] == eb[k]])}


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
    b.add_argument("--duckdb", type=Path, help="a DuckDB mirror of the estate")
    b.add_argument("--catalog", help="read Unity Catalog instead: the catalog's name; DATABRICKS_HOST, DATABRICKS_WAREHOUSE_ID and a token or a service principal in the environment")
    b.add_argument("--metric-views", type=Path, help="the metric views' definitions as text (a mirror does not hold them)")
    b.add_argument("--verified", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--resolver", type=Path, help="folder of the resolution engine (its link_records.py and environment)")
    b.add_argument("--hosted-model", action="store_true", help="let a hosted language model propose classes and definitions from names (the estate file must allow it)")
    b.add_argument("--class-answer", type=Path, help="a recorded answer to classes_prompt.json, in place of a call to a hosted model")
    b.add_argument("--usage", type=Path, action="append", default=[], help="a log of answered questions (usage/1); repeat for more")
    b.add_argument("--max-usd", type=float, default=0.25, help="the most one call to the hosted model may cost in the worst case; a call that could cost more is not made")
    sd = sub.add_parser("seed")
    sd.add_argument("--config", type=Path, required=True)
    sd.add_argument("--duckdb", type=Path, required=True)
    sd.add_argument("--out-duckdb", type=Path, required=True)
    sd.add_argument("--out", type=Path, required=True)
    sd.add_argument("--seed", type=int, default=11)
    sc = sub.add_parser("score", help="precision and recall of the proposals against a hand-checked list")
    sc.add_argument("--out", type=Path, required=True, help="the build folder that holds ontology.json")
    sc.add_argument("--gold", type=Path, required=True)
    cp = sub.add_parser("compare", help="what two builds of one estate hold differently (a mirror and the catalog)")
    cp.add_argument("--a", type=Path, required=True)
    cp.add_argument("--b", type=Path, required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--ui-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "seed":
        print(json.dumps(seed(a.config, a.duckdb, a.out_duckdb, a.out, a.seed), indent=1))
    elif a.cmd == "compare":
        print(json.dumps(compare(json.loads((a.a / "ontology.json").read_text(encoding="utf-8")), json.loads((a.b / "ontology.json").read_text(encoding="utf-8"))), indent=1))
    elif a.cmd == "score":
        result = score(json.loads((a.out / "ontology.json").read_text(encoding="utf-8")), yaml.safe_load(a.gold.read_text(encoding="utf-8")))
        (a.out / "score.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=1, ensure_ascii=False))
    elif a.cmd == "build":
        print(json.dumps(build(a.config, a.duckdb, a.metric_views, a.verified, a.out, a.resolver, a.hosted_model, a.class_answer, a.max_usd, usage=tuple(a.usage), catalog=a.catalog), indent=1))
    else:
        a.ui_dir.mkdir(parents=True, exist_ok=True)
        for name in UI_FILES:
            shutil.copyfile(a.out / name, a.ui_dir / name)
        print(f"copied {', '.join(UI_FILES)} to {a.ui_dir}")


if __name__ == "__main__":
    main()

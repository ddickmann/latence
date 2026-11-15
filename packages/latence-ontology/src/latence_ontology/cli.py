"""Build the ontology of an estate and write what a screen reads.

  latence-ontology build --config estates/vehicles.yaml --duckdb <mirror.duckdb> \\
         --metric-views <views.txt> [--verified <host catalog.yaml>] --out out/vehicles
  latence-ontology export --out out/vehicles --ui-dir <host ui>/public/data/ontology
  latence-ontology propose --config estates/vehicles.yaml --out out/vehicles

``build`` appends to ``assertions.jsonl`` in the output folder, reads ``decisions.jsonl``
there if a person has written one, and writes ``ontology.json`` (graph-export/3).
``export`` copies the files a screen reads into the host's static folder.
``propose`` writes, from the approved decisions of that folder, the proposal file a host's
semantic layer may take up (``proposal.yaml``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .classes import (
    CachedClassModel,
    OpenAIClassModel,
    RecordedClassModel,
    describe,
    prompt_of,
    propose_classes,
)
from .codes import TOP
from .crosswalk import crosswalk, read_alias_tables
from .drift import find_drift, prior_statements
from .erasure import erase, turned_sensitive
from .export import build_export, guard
from .hub import pull_certifications, push_reviews
from .hub import submit as submit_to_hub
from .hub import views as hub_views
from .hub_local import LocalGitHost
from .induce import induce
from .instances import build_instances, merge_values
from .iris import Iris, confirm, element_ids, read_renames
from .labels import with_identity
from .ledger import Decisions, append, assertions_of, read_decisions, replay
from .lifecycle import calibrated, engine_approvals, read_certifications, with_states
from .mentions import find_mentions
from .pair_voice import OpenAIPairClient, PairClient, ask_about_pairs, rule_of
from .proposal import in_layer, proposal, read_receipts, to_yaml
from .reader import Reader
from .reader_databricks import DatabricksReader
from .reader_duckdb import DuckDBReader
from .resolution import merge, pair_answers, run_resolver, write_records
from .score import score
from .seed import SCHEMA, create_sql, insert_sql, seed_accounts
from .staged import staged_statements, to_sql
from .standards import to_prov, to_shacl
from .structure import broader_edges, build_structure, explain_review
from .turtle import KEY_BASE, to_turtle
from .usage import attach, read_usage
from .vocabulary import read_vocabulary

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


def build(
    config: Path,
    duckdb_path: Path | None,
    views: Path | None,
    verified: Path | None,
    out: Path,
    resolver: Path | None = None,
    hosted_model: bool = False,
    class_answer: Path | None = None,
    max_usd: float = 0.25,
    client: object | None = None,
    usage: tuple[Path, ...] = (),
    catalog: str | None = None,
    receipts: tuple[Path, ...] = (),
    pair_client: PairClient | None = None,
    documents: Path | None = None,
    alias_tables: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """The estate is read from a DuckDB mirror (``duckdb_path``) or from Unity Catalog
    (``catalog``; host, warehouse and credentials from the environment). ``receipts`` are what
    a host wrote when it placed a proposal file: an approved element they name says in the
    export since which release of the host's library it is in the semantic layer; without
    one no element says so. ``hosted_model`` lets
    a hosted language model propose classes, if the estate allows it. The call is made only
    when what would be sent has changed, and only if its worst case stays under ``max_usd``.
    ``client`` stands in for the provider's client in tests, and ``pair_client`` for the client
    of the hosted model on the pairs the linker leaves undecided (ticket 48), which is asked only
    where the estate file switches it on with an egress rule (``link.hosted_model``).
    ``alias_tables`` are a host's alias tables (ticket 57): an element one of them names by a
    platform id carries the host's key (``match``)."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    reader: Reader
    if catalog:
        # The schemas of the catalog that exist: a layer the mirror adds (a seeded source) may be
        # missing there.
        reader = DatabricksReader.from_env(
            catalog,
            cfg["schemas"],
            schema_prefix=cfg.get("schema_prefix", ""),
            sensitive=cfg.get("sensitive", []),
            sensitive_tags=cfg.get("sensitive_tags", []),
        )
    elif duckdb_path is not None:
        reader = DuckDBReader(
            duckdb_path,
            schemas=cfg["schemas"],
            sensitive=cfg.get("sensitive", []),
            metric_views=views.read_text(encoding="utf-8") if views else "",
            schema_prefix=cfg.get("schema_prefix", ""),
            name=cfg["name"],
        )
    else:
        raise SystemExit(
            "give the estate to read: --duckdb FILE for a mirror, or --catalog NAME for Unity "
            "Catalog"
        )
    read_objects = reader.objects()
    names = {o.full_name for o in read_objects}
    # What the catalog says each object is made from, between the estate's concepts (ticket 41).
    read_lineage = reader.lineage()
    lineage = (
        None
        if read_lineage is None
        else [
            {"from": f"concept:{e.source}", "to": f"concept:{e.target}", "source": "catalog"}
            for e in read_lineage
            if e.inside and e.source in names and e.target in names
        ]
    )
    lineage_outside = 0 if read_lineage is None else sum(not e.inside for e in read_lineage)
    joins = read_verified(verified, names) if verified else []
    shown = tuple(cfg.get("without_personal_columns", ()))
    model = induce(
        reader,
        joins,
        tuple(cfg.get("ignore_columns", ())),
        tuple(cfg.get("derived_schemas", ())),
        shown,
        labels=cfg.get("labels"),
    )
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
    # The engine approves a kind only where a recorded calibration proves it safe (ticket 51);
    # a person's decision outranks it.
    life_cfg = cfg.get("lifecycle") or {}
    safe, calibrations_refused = calibrated(list(life_cfg.get("auto_approve") or []))
    # A column that turned sensitive since an earlier build read it: its values are erased from
    # the folder before anything here reads the folder (ticket 54).
    prior = prior_statements(out / "assertions.jsonl", out / "ledger_builds.jsonl")
    builds_so_far = (
        (out / "ledger_builds.jsonl").read_text(encoding="utf-8").count("\n")
        if (out / "ledger_builds.jsonl").exists()
        else 0
    )
    previous = (
        json.loads((out / "ontology.json").read_text(encoding="utf-8"))
        if (out / "ontology.json").exists()
        else None
    )
    erased = erase(
        out,
        previous,
        turned_sensitive(
            prior, {o.full_name: {c.name: c.sensitive for c in o.columns} for o in read_objects}
        ),
        {str(c["column"]) for c in model["concepts"] if c.get("kind") == "reference"},
        builds_so_far + 1,
    )
    prior = prior_statements(out / "assertions.jsonl", out / "ledger_builds.jsonl")
    by_people = read_decisions(out / "decisions.jsonl")
    # What drifted under a decided element since it was decided (ticket 53): the decision is
    # kept on the record as stale and not applied, so the element is a proposal again.
    drifted = find_drift(
        by_people,
        model,
        prior,
        {o.full_name: {c.name: c.type for c in o.columns} for o in read_objects},
    )
    by_people = {k: v for k, v in by_people.items() if k not in {e["subject"] for e in drifted}}
    decisions = Decisions({**engine_approvals(model, safe, by_people), **by_people})
    layer = replay(model, decisions)
    instances = build_instances(
        reader,
        layer,
        tuple(cfg.get("personal_objects", ())),
        tuple(cfg.get("resolve", ())),
        shown,
        also={spec["object"]: spec["name"] for spec in link["records"][1:]} if link else None,
        label_columns=cfg.get("label_columns"),
    )
    structure = build_structure(reader, layer, tuple(cfg.get("personal_objects", ())), decisions)
    if hosted_model and not cfg.get("classes", {}).get("hosted_model"):
        raise SystemExit(
            "this estate does not allow its names to be sent to a hosted model "
            "(classes.hosted_model in the estate file)"
        )
    personal = tuple(cfg.get("personal_objects", ()))
    voice: CachedClassModel | RecordedClassModel | None = None
    if hosted_model:
        spec = cfg["classes"]
        if "prices" not in spec:
            raise SystemExit(
                "the estate file states no prices for the hosted model (classes.prices): the call "
                "cannot be held under a limit"
            )
        hosted = OpenAIClassModel(
            client=client,
            model=spec.get("model", "gpt-6-luna"),
            effort=spec.get("effort", "low"),
            languages=tuple(spec.get("languages", ("en", "de"))),
        )
        voice = CachedClassModel(
            hosted, out / "class_answers", spec["prices"], max_usd, out / "model_calls.jsonl"
        )
        leaves: dict[str, object] = hosted.request(
            describe(layer, structure, personal), TOP
        )  # exactly what the call sends
    elif class_answer:
        voice = RecordedClassModel(json.loads(class_answer.read_text(encoding="utf-8")))
    if not hosted_model:
        leaves = prompt_of(
            describe(layer, structure, personal), TOP
        )  # what a model session would be given
    (out / "classes_prompt.json").write_text(
        json.dumps(leaves, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # An outside vocabulary as the third voice, its licence checked first (ticket 55); its file
    # is named relative to the estate file.
    said_vocabulary = (cfg.get("classes") or {}).get("vocabulary")
    vocabulary = (
        read_vocabulary(
            config.parent / str(said_vocabulary["file"]),
            tuple(said_vocabulary["licences_allowed"])
            if said_vocabulary.get("licences_allowed")
            else None,
        )
        if said_vocabulary
        else None
    )
    classes = propose_classes(layer, structure, personal, voice, decisions, vocabulary)
    if isinstance(voice, CachedClassModel) and voice.cost is not None:
        # What the answer in this export cost when it was asked for; a rebuild that asks nothing
        # says the same.
        classes = {
            **classes,
            "voices": {
                **classes["voices"],
                "cost": {k: v for k, v in voice.cost.items() if k != "call"},
            },
        }
        call_note = voice.cost["call"]
    else:
        call_note = "no hosted model"
    ledger = out / "assertions.jsonl"
    before = (
        sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())
        if ledger.exists()
        else 0
    )
    new = append(ledger, assertions_of(model, structure, classes))
    if new:  # where this build's statements begin in the ledger, for what drifted since (ticket 53)
        with open(out / "ledger_builds.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"first_line": before, "statements": new}) + "\n")
    up = broader_edges(reader, structure, {n["id"] for n in instances["nodes"]})
    edges = sorted(
        instances["edges"] + up,
        key=lambda e: (e["kind"], e["from"], e["to"], str(e.get("via")), str(e.get("relation"))),
    )
    instances = {
        **instances,
        "edges": edges,
        "review": explain_review(instances["review"], edges, instances["nodes"], layer["concepts"]),
        "meta": {**instances["meta"], "edges": len(edges)},
    }
    resolution_note = (
        "no second source in this estate"
        if link is None
        else "no resolver given: records not linked"
    )
    if link and resolver is not None:
        truth_file = out / "seed_truth.json"
        truth = (
            json.loads(truth_file.read_text(encoding="utf-8"))["truth"]
            if truth_file.exists()
            else None
        )
        # Whether the truth of a seeded source may stand in for a reviewer's labels is the
        # estate's to say. Without a truth nothing can stand in, and every label is a person's.
        stands_in = bool(link["labels"].get("truth_stands_in", True)) and truth is not None
        answers = pair_answers(decisions.by, [spec["object"] for spec in link["records"]])
        write_records(reader, link["records"], truth, out / "records.parquet")
        answer = run_resolver(
            resolver,
            out / "records.parquet",
            out / "resolution.json",
            link["labels"]["fit"],
            link["labels"]["test"],
            tuple(link.get("duplicate_free", ())),
            answers,
            stands_in,
        )
        concept_of = {c["object_name"]: c["id"] for c in layer["concepts"] if c.get("object_name")}
        instances = merge(
            instances,
            answer,
            link,
            truth,
            concept_of,
            seeded=any(s["object"] in seeded for s in link["records"]),
            answers=answers,
        )
        rule = rule_of(link)
        if rule is not None:
            records = {
                str(u): {"name": n, "city": c, "country": k}
                for u, n, c, k in duckdb.execute(
                    "SELECT unique_id, name, city, country FROM "
                    f"read_parquet('{out / 'records.parquet'}')"
                ).fetchall()
            }
            review, pair_model = ask_about_pairs(
                instances["review"],
                records,
                rule,
                pair_client or OpenAIPairClient(),
                out / "pair_model_calls.jsonl",
            )
            instances = {
                **instances,
                "review": review,
                "meta": {
                    **instances["meta"],
                    "resolution": {**instances["meta"]["resolution"], "pair_model": pair_model},
                },
            }
        resolution_note = "linked" if answer["certified"] else answer["state"]
    if link and resolver is not None:
        for said in (
            answers
        ):  # every answer on a pair of records was applied, whether the pair is still listed or not
            decisions.verdict(
                "possibly_same",
                {"id": f"same:{said['a']}~{said['b']}", "a": said["a"], "b": said["b"]},
            )
    # Names of a value set a reviewer called the same are one value from here on.
    same_names = [
        (item["a"], item["b"])
        for item in instances["review"]
        if item["a"].startswith("value:")
        and (v := decisions.verdict("possibly_same", item))
        and v["verdict"] == "approved"
    ]
    instances = merge_values(instances, same_names)
    instances = {
        **instances,
        "review": [decisions.marked("possibly_same", item) for item in instances["review"]],
        "label_requests": {
            name: [decisions.marked("possibly_same", item) for item in sample]
            for name, sample in instances.get("label_requests", {}).items()
        },
    }
    layer = {**layer, "meta": {**layer["meta"], "decisions": decisions.summary()}}
    export = build_export(
        layer,
        source=cfg["name"],
        title=cfg.get("title", cfg["name"]),
        verified=joins,
        instances=instances,
        structure=structure,
        classes=classes,
        language=cfg.get("language", "en"),
        proposal_names=tuple((cfg.get("propose") or {}).get("names", ())),
        lineage=lineage,
        lineage_outside=lineage_outside,
    )
    export, taken_up = in_layer(export, read_receipts(receipts))
    in_export = {
        el["id"]
        for part in (export["concepts"], export["relations"], export["measures"], export["review"])
        for el in part
    }
    export, life = with_states(
        export,
        read_certifications(out / "certifications.jsonl"),
        life_cfg.get("data_owner"),
        [s for s, v in decisions.applied().items() if v == "rejected" and s not in in_export],
    )
    if drifted:
        export = {**export, "drift": drifted}
    mention_counts = None
    if (
        documents is not None
    ):  # the estate's documents, their mentions linked to its rows (ticket 42)
        mentions, mention_counts = find_mentions(export, documents)
        export = {
            **export,
            "mentions": mentions,
            "meta": {**export["meta"], "mentions": mention_counts},
        }
    base = cfg.get("export", {}).get("base_iri", f"urn:estate:{cfg['name']}:")
    # Persistent IRIs, the host's keys and labels per language (ticket 57).
    iris = Iris(out)
    renamed = iris.settle(element_ids(export), read_renames(out))
    iris.save()
    matches: dict[str, dict[str, Any]] = {}
    crosswalk_note = None
    if alias_tables:
        matches, crosswalk_note = crosswalk(
            export, reader.metric_views(), read_alias_tables(alias_tables)
        )
    export = with_identity(export, base, iris.of, matches, cfg.get("language", "en"), decisions)
    export["meta"] = {
        **export["meta"],
        "decisions": decisions.summary(),  # again: the labels were asked about just now
        **({"renames": renamed} if renamed["carried"] or renamed["refused"] else {}),
        **({"crosswalk": crosswalk_note} if crosswalk_note is not None else {}),
    }
    guard(export)  # before any of the four files is written
    (out / "ontology.json").write_text(
        json.dumps(export, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    key_base = cfg.get("export", {}).get("key_iri", KEY_BASE)
    (out / "ontology.ttl").write_text(to_turtle(export, base, iris.of, key_base), encoding="utf-8")
    (out / "ontology.shacl.ttl").write_text(
        to_shacl(export, base, iris.of), encoding="utf-8"
    )  # ticket 42
    (out / "ontology.prov.ttl").write_text(to_prov(export, base, iris.of), encoding="utf-8")
    return {
        "new_assertions": new,
        **export["meta"]["counts"],
        "decisions": export["meta"]["decisions"],
        "nodes": instances["meta"]["nodes"],
        "edges": instances["meta"]["edges"],
        "review_items": instances["meta"]["review_items"],
        "structure": {k: len(v) for k, v in export["structure"].items()},
        "classes": {
            "voices": classes["voices"],
            "placed": len(classes["subclass_of"]),
            "contested": sum(s["contested"] for s in classes["subclass_of"]),
            "own_classes": len(classes["classes"]),
            "unplaced": len(classes["unplaced"]),
            "hosted_call": call_note,
        },
        "resolution": instances["meta"].get("resolution") or resolution_note,
        **({"in_layer": taken_up} if receipts else {}),
        "lifecycle": {**life, "calibrations_refused": calibrations_refused},
        **({"drift": drifted} if drifted else {}),
        **({"erased": erased} if erased else {}),
        **({"mentions": mention_counts} if mention_counts is not None else {}),
        **({"renames": renamed} if renamed["carried"] or renamed["refused"] else {}),
        **({"crosswalk": crosswalk_note} if crosswalk_note is not None else {}),
        **(
            {"pair_model": (instances["meta"].get("resolution") or {}).get("pair_model")}
            if (instances["meta"].get("resolution") or {}).get("pair_model")
            else {}
        ),
    }


def propose(
    config: Path, out: Path, file: Path | None = None, decided_on: str | None = None
) -> dict[str, Any]:
    """Write the proposal file of a build folder: from ``ontology.json`` (what the elements
    state now) and ``decisions.jsonl`` (what a person approved), to ``proposal.yaml`` there or
    to ``file``. ``decided_on`` is the day for the decisions that state none. Returns what was
    written and what was not, each with the reason."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    said = cfg.get("propose") or {}
    if not (out / "ontology.json").exists():
        raise SystemExit(f"no build in {out}: run `build` first, then `propose`")
    export = json.loads((out / "ontology.json").read_text(encoding="utf-8"))
    document, report = proposal(
        export,
        read_decisions(out / "decisions.jsonl"),
        tuple(said.get("names", ())),
        said.get("spellings_on"),
        decided_on,
        links=bool(said.get("links", True)),
        concepts_too=bool(said.get("concepts", True)),
        domains=cfg.get("domains") or None,
        certifications=read_certifications(out / "certifications.jsonl"),
    )
    target = file or out / "proposal.yaml"
    target.write_text(to_yaml(document, cfg["name"]), encoding="utf-8")
    return {"file": str(target), **report}


def hub(
    config: Path,
    out: Path,
    repo: Path,
    at: str,
    receipts: tuple[Path, ...] = (),
    decided_on: str | None = None,
) -> dict[str, Any]:
    """The build's approved entries proposed to the estate's git repository (ticket 59): a
    merge request per domain, the builder's reviews written on them, the data owners'
    certifications given on the git host read back into ``certifications.jsonl``, and what the
    Ontology page shows of it in ``hub.json``. ``repo`` is a bare repository (the local backend
    of the port); ``receipts`` are the host's receipts of the releases built from its tags."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    domains = cfg.get("domains") or {}
    if not domains:
        raise SystemExit("the estate file names no domains (domains: {<name>: {schemas, ...}})")
    said = cfg.get("propose") or {}
    export = json.loads((out / "ontology.json").read_text(encoding="utf-8"))
    certifications = read_certifications(out / "certifications.jsonl")
    document, _report = proposal(
        export,
        read_decisions(out / "decisions.jsonl"),
        tuple(said.get("names", ())),
        said.get("spellings_on"),
        decided_on,
        links=bool(said.get("links", True)),
        concepts_too=bool(said.get("concepts", True)),
        domains=domains,
        certifications=certifications,
    )
    host = LocalGitHost(repo)
    opened = submit_to_hub(host, document, cfg["name"], domains, at)
    pulled = pull_certifications(host, cfg["name"], domains, certifications)
    if pulled:
        with (out / "certifications.jsonl").open("a", encoding="utf-8") as fh:
            for line in pulled:
                fh.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
    written = push_reviews(host, cfg["name"], domains, [*certifications, *pulled], at)
    releases: dict[str, dict[str, str]] = {}
    for path in receipts:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("tag"):
            releases[str(doc["tag"])] = {
                k: str(doc[k]) for k in ("library", "version", "digest") if doc.get(k)
            }
    shown = hub_views(host, cfg["name"], releases)
    (out / "hub.json").write_text(
        json.dumps(shown, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {**opened, "reviews_written": written, "certifications_read": len(pulled),
            "merge_requests": len(shown["merge_requests"])}


def stage(config: Path, out: Path, file: Path | None = None) -> dict[str, Any]:
    """Write the staged proposal for the catalog (``staged-catalog.sql``) from the build's export
    and the people's decisions: statements its owner reviews and applies (ticket 52)."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not (out / "ontology.json").exists():
        raise SystemExit(f"no build in {out}: run `build` first, then `stage`")
    if not cfg.get("catalog"):
        raise SystemExit(
            "the estate file names no catalog (catalog: <name>): the statements could not say "
            "where they go"
        )
    export = json.loads((out / "ontology.json").read_text(encoding="utf-8"))
    statements, report = staged_statements(
        export,
        read_decisions(out / "decisions.jsonl"),
        str(cfg["catalog"]),
        cfg.get("schema_prefix", ""),
    )
    target = file or out / "staged-catalog.sql"
    target.write_text(to_sql(statements), encoding="utf-8")
    return {"file": str(target), **report}


def certify(out: Path, subject: str, by: str, at: str) -> dict[str, Any]:
    """Append the data owner's certification of an element, on the statement the build's export
    makes now (its claim), to ``certifications.jsonl``. Whether it stands (the right person, an
    element approved, the same statement) is the next build's to say."""
    if not (out / "ontology.json").exists():
        raise SystemExit(f"no build in {out}: run `build` first, then `certify`")
    export = json.loads((out / "ontology.json").read_text(encoding="utf-8"))
    groups = [export["concepts"], export["relations"], export["measures"], export["review"]]
    element = next((el for part in groups for el in part if el["id"] == subject), None)
    if element is None:
        raise SystemExit(f"{subject} is no element of the build in {out}")
    day = str(at)[:10]
    line = {"subject": subject, "by": by, "claim": element["claim"], "at": day}
    with open(out / "certifications.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line


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
    return {
        "only_in_a": sorted(set(ea) - set(eb)),
        "only_in_b": sorted(set(eb) - set(ea)),
        "differ": {
            k: {"a": list(ea[k]), "b": list(eb[k])}
            for k in sorted(set(ea) & set(eb))
            if ea[k] != eb[k]
        },
        "same": len([k for k in set(ea) & set(eb) if ea[k] == eb[k]]),
    }


def seed(
    config: Path, duckdb_path: Path, out_db: Path, out: Path, seed_no: int
) -> dict[str, Any]:
    """Copy the mirror and add the seeded second source to the copy, in a schema of its own.
    The truth goes to ``seed_truth.json`` in the output folder, outside the catalog, together
    with the statements that load the same table into a catalog (``seeded_accounts.sql``)."""
    import duckdb

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    spec = cfg["seeded"]
    shutil.copyfile(duckdb_path, out_db)
    con = duckdb.connect(str(out_db))
    schema, table = spec["from"].split(".")
    rows = con.execute(
        f'SELECT "{spec["id"]}", "{spec["name"]}", "{spec["city"]}", "{spec["market"]}" FROM '
        f'"{schema}"."{table}" ORDER BY 1'
    ).fetchall()
    dealers = [{"dealer_id": r[0], "name": r[1], "city": r[2], "market": r[3]} for r in rows]
    accounts, truth = seed_accounts(dealers, seed=seed_no)
    for sql in create_sql() + insert_sql(accounts):
        con.execute(sql)
    con.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed_truth.json").write_text(
        json.dumps({"seed": seed_no, "of": spec["from"], "truth": truth}, indent=1, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    target = f"<catalog>.{cfg.get('schema_prefix', '')}{SCHEMA}"
    head = [
        "-- The seeded second source, for a catalog. Generated; not part of the estate.",
        "-- Replace <catalog>. Not run against a catalog where it was written: read it before you "
        "run it.",
        "",
    ]
    (out / "seeded_accounts.sql").write_text(
        "\n".join(head + [s + ";" for s in create_sql(target) + insert_sql(accounts, target)])
        + "\n",
        encoding="utf-8",
    )
    return {
        "accounts": len(accounts),
        "of_dealers": sum(not v.startswith("other:") for v in truth.values()),
        "of_other_workshops": sum(v.startswith("other:") for v in truth.values()),
        "mirror_with_seed": str(out_db),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="latence-ontology")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--duckdb", type=Path, help="a DuckDB mirror of the estate")
    b.add_argument(
        "--catalog",
        help=(
            "read Unity Catalog instead: the catalog's name; DATABRICKS_HOST, "
            "DATABRICKS_WAREHOUSE_ID and a token or a service principal in the environment"
        ),
    )
    b.add_argument(
        "--metric-views",
        type=Path,
        help="the metric views' definitions as text (a mirror does not hold them)",
    )
    b.add_argument("--verified", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument(
        "--resolver",
        type=Path,
        help="folder of the resolution engine (its link_records.py and environment)",
    )
    b.add_argument(
        "--hosted-model",
        action="store_true",
        help=(
            "let a hosted language model propose classes and definitions from names (the estate "
            "file must allow it)"
        ),
    )
    b.add_argument(
        "--class-answer",
        type=Path,
        help="a recorded answer to classes_prompt.json, in place of a call to a hosted model",
    )
    b.add_argument(
        "--usage",
        type=Path,
        action="append",
        default=[],
        help="a log of answered questions (usage/1); repeat for more",
    )
    b.add_argument(
        "--documents",
        type=Path,
        help=(
            "a folder of the estate's documents (.txt, .md): their mentions of its rows are linked"
        ),
    )
    b.add_argument(
        "--alias-table",
        type=Path,
        action="append",
        default=[],
        help="a host's alias file (metis.alias/1: its keys and the platform ids they were read "
        "from); an element it names carries the host's key; repeat for more",
    )
    b.add_argument(
        "--receipt",
        type=Path,
        action="append",
        default=[],
        help="what a host wrote when it placed a proposal file (layer-receipt/1); "
        "repeat for more, oldest first",
    )
    b.add_argument(
        "--max-usd",
        type=float,
        default=0.25,
        help=(
            "the most one call to the hosted model may cost in the worst case; a call that could "
            "cost more is not made"
        ),
    )
    sd = sub.add_parser("seed")
    sd.add_argument("--config", type=Path, required=True)
    sd.add_argument("--duckdb", type=Path, required=True)
    sd.add_argument("--out-duckdb", type=Path, required=True)
    sd.add_argument("--out", type=Path, required=True)
    sd.add_argument("--seed", type=int, default=11)
    sc = sub.add_parser(
        "score", help="precision and recall of the proposals against a hand-checked list"
    )
    sc.add_argument(
        "--out", type=Path, required=True, help="the build folder that holds ontology.json"
    )
    sc.add_argument("--gold", type=Path, required=True)
    cp = sub.add_parser(
        "compare", help="what two builds of one estate hold differently (a mirror and the catalog)"
    )
    cp.add_argument("--a", type=Path, required=True)
    cp.add_argument("--b", type=Path, required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--ui-dir", type=Path, required=True)
    pr = sub.add_parser(
        "propose",
        help="the proposal file for a host's semantic layer, from "
        "the approved decisions of a build",
    )
    pr.add_argument("--config", type=Path, required=True)
    pr.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the build folder that holds ontology.json and decisions.jsonl",
    )
    pr.add_argument(
        "--file",
        type=Path,
        help="where to write the file (default: proposal.yaml in the build folder)",
    )
    pr.add_argument(
        "--decided-on",
        help="the day (YYYY-MM-DD) of the approved decisions that state none; "
        "without it such a decision is not written",
    )
    st = sub.add_parser(
        "stage",
        help=(
            "the staged proposal for the catalog's owner, from the approved decisions of a build "
            "(ticket 52)"
        ),
    )
    st.add_argument("--config", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True)
    st.add_argument(
        "--file",
        type=Path,
        help="where to write it (default: staged-catalog.sql in the build folder)",
    )
    ce = sub.add_parser(
        "certify", help="the data owner certifies an approved element of a build (ticket 51)"
    )
    ce.add_argument(
        "--out", type=Path, required=True, help="the build folder that holds ontology.json"
    )
    ce.add_argument("--subject", required=True, help="the element's id")
    ce.add_argument(
        "--by", required=True, help="the data owner's id, as the estate file names them"
    )
    ce.add_argument("--at", required=True, help="the day of the certification (YYYY-MM-DD)")
    hb = sub.add_parser(
        "hub",
        help="propose the approved entries to the estate's git repository, a merge request per "
        "domain, and sync the reviews both ways (ticket 59)",
    )
    hb.add_argument("--config", type=Path, required=True)
    hb.add_argument("--out", type=Path, required=True, help="the build folder")
    hb.add_argument("--repo", type=Path, required=True, help="the estate's bare git repository")
    hb.add_argument("--at", required=True, help="the day (YYYY-MM-DD)")
    hb.add_argument("--receipt", type=Path, action="append", default=[],
                    help="the host's receipt of a release built from a tag; repeat for more")
    hb.add_argument("--decided-on", help="the day of the approved decisions that state none")
    rn = sub.add_parser(
        "rename",
        help="a person confirms that an element was renamed: the next build gives the new id "
        "the old IRI (ticket 57)",
    )
    rn.add_argument("--out", type=Path, required=True, help="the build folder")
    rn.add_argument("--from", dest="old", required=True, help="the element's old id")
    rn.add_argument("--to", dest="new", required=True, help="the element's id now")
    rn.add_argument("--by", required=True, help="who confirms it")
    rn.add_argument("--at", required=True, help="the day (YYYY-MM-DD)")
    rn.add_argument("--reason", default="", help="why, in a few words")
    a = ap.parse_args()
    if a.cmd == "seed":
        print(json.dumps(seed(a.config, a.duckdb, a.out_duckdb, a.out, a.seed), indent=1))
    elif a.cmd == "compare":
        print(
            json.dumps(
                compare(
                    json.loads((a.a / "ontology.json").read_text(encoding="utf-8")),
                    json.loads((a.b / "ontology.json").read_text(encoding="utf-8")),
                ),
                indent=1,
            )
        )
    elif a.cmd == "score":
        result = score(
            json.loads((a.out / "ontology.json").read_text(encoding="utf-8")),
            yaml.safe_load(a.gold.read_text(encoding="utf-8")),
        )
        (a.out / "score.json").write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=1, ensure_ascii=False))
    elif a.cmd == "propose":
        report = propose(a.config, a.out, a.file, a.decided_on)
        print(json.dumps(report, indent=1, ensure_ascii=False))
    elif a.cmd == "stage":
        print(json.dumps(stage(a.config, a.out, a.file), indent=1, ensure_ascii=False))
    elif a.cmd == "certify":
        print(json.dumps(certify(a.out, a.subject, a.by, a.at), indent=1, ensure_ascii=False))
    elif a.cmd == "hub":
        result = hub(a.config, a.out, a.repo, a.at, tuple(a.receipt), a.decided_on)
        print(json.dumps(result, indent=1, ensure_ascii=False))
    elif a.cmd == "rename":
        line = confirm(a.out, a.old, a.new, a.by, a.at, a.reason)
        print(json.dumps(line, indent=1, ensure_ascii=False))
    elif a.cmd == "build":
        print(
            json.dumps(
                build(
                    a.config,
                    a.duckdb,
                    a.metric_views,
                    a.verified,
                    a.out,
                    a.resolver,
                    a.hosted_model,
                    a.class_answer,
                    a.max_usd,
                    usage=tuple(a.usage),
                    catalog=a.catalog,
                    receipts=tuple(a.receipt),
                    documents=a.documents,
                    alias_tables=tuple(a.alias_table),
                ),
                indent=1,
            )
        )
    else:
        a.ui_dir.mkdir(parents=True, exist_ok=True)
        for name in UI_FILES:
            shutil.copyfile(a.out / name, a.ui_dir / name)
        print(f"copied {', '.join(UI_FILES)} to {a.ui_dir}")


if __name__ == "__main__":
    main()

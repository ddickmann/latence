# Estate prototype: from a catalog to an ontology

Prototype code for map ticket [*The Ontology page over the estate*](../issues/25-estate-demo-slice.md).
It reads an estate (tables, columns, comments, metric views), induces the schema level of its
ontology with the evidence for every element, keeps assertions and decisions in an append-only
ledger, and writes the file a screen reads (`graph-export/3`). Cut A covers the schema level
only: no rows, no values.

It is not a shipped package. It has its own pinned environment and its own tests.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONHASHSEED=0 .venv/bin/python -m pytest -q tests          # a toy estate with planted structure
.venv/bin/python -m estate build --config estates/vehicles.yaml --duckdb <mirror.duckdb> \
    --metric-views <views.txt> --verified <host catalog.yaml> --out out/vehicles
.venv/bin/python -m estate export --out out/vehicles --ui-dir <host ui>/public/data/ontology
```

| File | What it does |
|---|---|
| `estate/reader.py` | the interface to a warehouse: objects, declared keys, metric views, one query call |
| `estate/reader_duckdb.py` | that interface over a DuckDB mirror |
| `estate/metric_views.py` | reads metric-view definitions |
| `estate/induce.py` | concepts, relations, value sets, hierarchies, attributes, measure bindings, each with signals and support |
| `estate/ledger.py` | assertions and decisions, append-only; the layer is their replay |
| `estate/export.py` | the `graph-export/3` file; its JSON Schema and the table of codes are in `schemas/` |
| `estates/vehicles.yaml` | what the engine is told about the vehicle and warranty estate |

A person's decisions go into `out/<estate>/decisions.jsonl`, one JSON object per line with
`subject`, `verdict` (`approved` or `rejected`), `by` and an optional `reason`. The next
`build` reads them; they survive every rebuild.

How to run it with the host application is in [`HANDOFF.md`](HANDOFF.md).

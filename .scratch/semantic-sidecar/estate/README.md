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
.venv/bin/python -m estate propose --config estates/vehicles.yaml --out out/vehicles     # the proposal file for a host
```

| File | What it does |
|---|---|
| `estate/reader.py` | the interface to a warehouse: objects, declared keys, metric views, one query call |
| `estate/reader_duckdb.py` | that interface over a DuckDB mirror |
| `estate/metric_views.py` | reads metric-view definitions |
| `estate/induce.py` | concepts, relations, value sets, hierarchies, attributes, measure bindings, each with signals and support |
| `estate/ledger.py` | assertions and decisions, append-only; the layer is their replay |
| `estate/export.py` | the `graph-export/3` file; its JSON Schema and the table of codes are in `schemas/` |
| `estate/proposal.py` | the proposal file for a host's semantic layer, written from approved decisions, and the host's receipt read back; the file's shape is `schemas/proposal-file.schema.json` |
| `estates/vehicles.yaml` | what the engine is told about the vehicle and warranty estate |

A person's decisions go into `out/<estate>/decisions.jsonl`, one JSON object per line with
`subject`, `verdict` (`approved` or `rejected`), `by`, and optionally the element's `claim`
(a placement, a class and anything that is to go into a proposal file need it), a `reason`
and `at` (the day of the decision). The next `build` reads them; they survive every rebuild.

## The one seam to a host: the proposal file

What a review approved reaches a host's semantic layer by one file and no other way
(ADR-0070). `propose` writes it as `out/<estate>/proposal.yaml` from the build's
`ontology.json` and `decisions.jsonl`; the host's own tool places it into its library as a
reviewed change and writes a receipt.

- **Two kinds of entry, from approved decisions only.** `spellings`: two values of a value set
  a reviewer called the same (the value more rows carry is kept, the other is its spelling).
  `names`: the names of a master list's records by their key, for a list the estate file
  names under `propose.names`. Each entry says where it goes by a catalog object and a column
  and carries its origin: the element's id, its claim, who approved it and on which day.
- **Never written:** a rejected or undecided element; anything seeded (a seeded list, its
  records, a pair that holds one, a pair of values of a value set a seeded object carries); a
  link between records, whoever made it; an entry whose decision was made on another
  statement than the build now makes (listed as `stale`); an entry whose decision states no
  claim, or no day. The command prints each approved decision it did not write, with the
  reason.
- **The day of a decision** is `at` on its line. A line without one is written only when the
  command is given the day (`--decided-on YYYY-MM-DD`); no day is made up.
- **The same inputs give the same bytes.** The file is YAML in block style with every text in
  double quotes.
- **The receipt.** `build --receipt <file>` (repeat for more, oldest first) reads what the
  host wrote when it placed a file (`layer-receipt/1`). An approved element the receipt lists
  as placed, on the claim it still states, carries `in_layer` in the export: the host's
  library and its version. Without a receipt no element carries it.

How to run it with the host application is in [`HANDOFF.md`](HANDOFF.md).

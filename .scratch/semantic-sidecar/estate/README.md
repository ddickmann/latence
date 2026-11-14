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
| `estate/lifecycle.py` | the five states of an element (draft, proposed, approved, certified, retired): a reviewer approves in `decisions.jsonl`, the data owner the estate file names (`lifecycle.data_owner`) certifies an approved element in `certifications.jsonl` (`python -m estate certify`), and the engine approves a kind only under a calibration record that reaches its target (`lifecycle.auto_approve`) |
| `estate/staged.py` | the staged proposal for the catalog's owner (`python -m estate stage`, `staged-catalog.sql`): an approved key or reference between tables that the catalog does not declare, as an informational constraint statement with the decision it rests on; the engine runs none of it |
| `estate/pair_voice.py` | a hosted model's vote on the pairs of records the linker leaves for a person: off unless the estate file switches it on with an egress rule (`link.hosted_model`), only the fields the rule names leave, a cap, each call recorded; the vote decides nothing |
| `estate/proposal.py` | a thin call into the repository's package `latence-proposal-writer` (`packages/latence-proposal-writer`), which writes the proposal file for a host's semantic layer from approved decisions and reads the host's receipt back; the file's shape is that package's `schema/proposal-file.schema.json` |
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
  names under `propose.names` (the export marks each such list that is not seeded
  `"proposal": "names"`, so a host's page offers a decision on it). Each entry says where it goes by a catalog object and a column
  and carries its origin: the element's id, its claim, who approved it and on which day.
- **Links between records.** A link between two records a person approved goes in as a `link`
  entry, in the shape the host's tool reads (its ticket 294), and the file then states
  `format: layer-proposal/2`; a file with no link is version 1, as before. Only where the host
  takes it: no seeded record, two objects, the master list's names in the same file, a linked
  record linked to no other, the statements the person saw. `propose.links: false` in the estate
  file keeps a file at version 1.
- **Never written:** a rejected or undecided element; anything seeded (a seeded list, its
  records, a pair that holds one, a pair of values of a value set a seeded object carries); a
  link the machine accepted and no person approved (one certified on simulated labels is such a
  link); an entry whose decision was made on another
  statement than the build now makes (listed as `stale`); an entry whose decision states no
  claim, or no day. The command prints each approved decision it did not write, with the
  reason.
- **The day of a decision** is `at` on its line (a date, or a timestamp such as the host's
  page writes since its ticket 282). A line without one is written only when the command is
  given the day (`--decided-on YYYY-MM-DD`), which never replaces a day a line states; no day
  is made up.
- **The same inputs give the same bytes.** The file is YAML in block style with every text in
  double quotes.
- **The receipt.** `build --receipt <file>` (repeat for more, oldest first) reads what the
  host wrote when it placed a file (`layer-receipt/1`). An approved element the receipt lists
  as placed, on the claim it still states, carries `in_layer` in the export: the host's
  library and its version. Without a receipt no element carries it.

How to run it with the host application is in [`HANDOFF.md`](HANDOFF.md).

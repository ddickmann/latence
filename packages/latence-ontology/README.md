# latence-ontology

The ontology build over a data estate. It reads an estate's catalog — a DuckDB mirror, or Unity
Catalog through the Databricks REST API — and induces the schema level of its ontology: concepts,
relations, value sets, hierarchies, attributes and measure bindings, each with the signals and the
support it rests on. Assertions and a person's decisions are kept in an append-only ledger, and the
layer is their replay, so a decision survives every rebuild. The build writes the file a screen
reads (`graph-export/3`, its JSON Schema shipped as package data) and, beside it, Turtle, SHACL
shapes and PROV-O provenance.

```bash
pip install latence-ontology                     # the build, the DuckDB and Unity Catalog readers
pip install "latence-ontology[hosted-model]"     # adds the OpenAI client for the hosted model's voice
```

```bash
latence-ontology build --config estate.yaml --duckdb mirror.duckdb --metric-views views.txt --out out/estate
latence-ontology propose --config estate.yaml --out out/estate      # the proposal file for a host
latence-ontology stage --config estate.yaml --out out/estate        # staged statements for the catalog's owner
latence-ontology certify --out out/estate --subject <id> --by <person> --at <day>
latence-ontology rename --out out/estate --from <old id> --to <new id> --by <person> --at <day>
latence-ontology hub --config estate.yaml --out out/estate --repo estate.git --at <day>
```

`hub` proposes the approved entries to the estate's git repository through a port (one branch
per domain, which its stewards alone commit to): a merge request per domain from a branch of the
builder's own, the reviewers' approvals and the data owners' certifications as its reviews, read
back both ways, and the tag and release of a certified merge (`hub.json`). The backend here is a
local bare repository; a hosted forge is another backend of the same port.

Every element has a persistent IRI, minted once and kept in the build folder's ledger
(`iris.jsonl`); a rename keeps it only once a person confirms it (`rename`). Given a host's
alias table (`build --alias-table <file>`), an element the table names by a platform id carries
the host's concept key (`match`, and `skos:exactMatch` in the Turtle file); a key is never
derived from a name. Labels carry their language (`skos:prefLabel`, `skos:altLabel`).

Its one seam to a host's semantic layer is the proposal file, written by
`latence-proposal-writer` from approved decisions only (ADR-0070). It registers no Provider: the
build is not a Stage of a Pipeline.

The same inputs give the same bytes. A hosted model is called only where the estate file switches
it on with an egress rule, and every call is recorded.

The class tree has up to three voices: attribute patterns, a hosted model (only where the
estate file allows it) and an outside vocabulary (only where the estate file names one, its
licence checked before use and recorded in the export). A voice proposes; a person decides.

# Build: a persistent IRI, the crosswalk to the host's concept keys, and labels per language

Type: build
Mode: AFK
Status: built (15 Nov); decided by the owner (15 Nov): option (a), as built; open: the host's command that writes its alias table, and the host-side test
Blocked by: nothing on this side (56 is done); the host's 296 and 297 are decided and built
Emerges from: the host's drafted ticket B-1 (15 Nov), for its v1; ADR-0073 point 4 (opaque IRIs,
not built); `latence_ontology/turtle.py` (an IRI is a hash of the name-based id);
`latence_ontology/induce.py` (`concept:{schema.table}`)

## Question

Ids are built from catalog names: a rename is drift, and the element goes back to proposed
(`drift.py`); decisions bind to claims (`ledger.py`); a label carries no language tag; there
is no `altLabel` and no `exactMatch`. The host keys its concepts `<kind>:<namespace>.<local>`
(its 296/297: `ConceptKey`, `MemberKey`, the alias table) and wants each element of the
ontology to name the key it is, read from its alias table, never guessed.

## What is asked

- An opaque IRI per element under the estate's `base_iri`, minted once and kept in the ledger;
  carried over a rename a person confirms on the drift proposal.
- `exactMatch` from each element to the host's concept key, read from the host's alias table,
  never guessed; a member of a value set to (dimension key, code).
- `skos:prefLabel` (one per language) and `skos:altLabel`, language-tagged, in the export and
  the Turtle file.
- The host's export guard (its 238) still accepts the export; a test on both sides.

## Done when

A rebuild after a confirmed rename keeps every IRI; every element that has a host key names it;
the host accepts the export.

## Decided by the owner (15 Nov): option (a)

The IRI carries over a rename only after a person confirms it; built so.

1. Does a rename that only drift detected keep the IRI?
   - (a) **Recommended, and built:** no, until a person confirms it.
   - (b) Yes, where the claim's digest is equal.

   Why (a): drift's `renamed` is a guess from types (two columns of one type swapped read as a
   rename), and an IRI carried on a guess is an identity merge no person made, which the map's
   "no silent over-merge" rules out. (b) would keep IRIs without a person in more cases, but a
   renamed column changes the claim of every element that names it, so it would rarely apply.

## Result (15 Nov)

Built in `latence-ontology`, option (a) of the open decision.

- **Persistent IRIs** (`iris.py`). The first build that states an element mints its IRI (the
  opaque name the Turtle file always used: `x` and 14 hex digits of the id's SHA-1) and appends it
  to `iris.jsonl` in the build folder, the ledger; every later build reads it from there. The
  export carries `iri` on concepts, relations, measures and classes, and the Turtle, SHACL and
  PROV-O files use the ledger's IRIs. An IRI is never given to two elements (a name another
  element holds is minted again with a counter).
- **A rename keeps the IRI only when a person confirmed it**: `latence-ontology rename --out
  <build> --from <old id> --to <new id> --by <person> --at <day>` appends to `renames.jsonl`;
  the next build gives the new id the old IRI, and the elements whose ids name a renamed table
  (its relations, its attributes) theirs, where the old id is gone and the renamed one is in the
  build. A confirmation whose old element is still in the build, whose new one is not, or whose
  old id the ledger never held is refused with the reason (`meta.renames.refused`). Drift alone
  carries nothing: an unconfirmed rename gets a new IRI.
- **The crosswalk** (`crosswalk.py`, `build --alias-table <file>`, repeatable) reads the host's
  alias table in the shape it states itself (`{namespace, keys, platform, renamed}`, the host's
  297) and names a key only where a platform id of the table names the element: a measure of a
  metric view, and a value set whose column a view's dimension reads (`source.<column>` on the
  view's source object), by `<schema>.<view>.<field>` (the catalog part is not compared). An
  element no platform id names gets no key, even where a key's bare id has the element's name
  (tested); an element two keys claim gets none and is listed (`meta.crosswalk.several_keys`).
  A matched value set says `members: "code"`: a member is the pair of its key and the value's
  code; the Turtle file writes `skos:exactMatch <urn:concept-key:<key>>` on the set (the base is
  `export.key_iri` in the estate file) and on each value `v:member [ v:dimension <key> ; v:code
  "<code>" ]`, never one string. A metric view now remembers its schema (`MetricView.schema`),
  from the listing's header or the catalog.
- **Labels per language** (`labels.py`): `labels: {pref: [text], alt: [text]}`, each a text
  with its language and who wrote it, as every free text of the export is: the label the build
  gives in the estate's language (`language`, English by default) and a metric view
  dimension's synonyms for a value set. The Turtle file writes `skos:prefLabel` and
  `skos:altLabel` with the language tag. A second language is another text in each list.
- **The export stays `graph-export/3`**, with the new fields optional and stated in its JSON
  Schema, as tickets 41, 51 and 53 added theirs: the host reads that one version and its guard
  refuses any other. The change is additive: the export without `iri`, `labels` and `match` is
  the prototype's bytes (checked on every run for the toy estate, and for the vehicle estate
  with its inputs); the Turtle file only gains lines; SHACL and PROV-O are unchanged.
- **The host's guard accepts it**: the host's own `refusals()` (its 238, run read-only from its
  checkout at its head) returns nothing for the vehicle export built with this change. This
  side's emulation of the guard (`test_ontology_host_guard.py`) passes unchanged.
- **Tests, red first** (`tests/test_ontology_identity.py`, 8; red on the tree of 56: no module
  `latence_ontology.iris`): every element has an IRI kept in the ledger, and a rebuild mints
  nothing; an unconfirmed rename gets a new IRI; a confirmed rename of a table keeps its IRI and
  its relations' and attributes', in the export and the Turtle file, and a rebuild carries
  nothing twice; a confirmation is refused where the old element is still there; a name held by
  a renamed element is minted again for a newcomer with the old id; the key comes from the alias
  table's platform ids and never from a name; two keys give none; labels carry their language
  in the export and the Turtle file. The same-bytes tests were re-recorded (above).

### Not done here, and why

- **The host has no command that writes its alias table** to a file (`AliasTable.to_json()`
  exists; nothing calls it for the builder). Asked of the host: a command that writes the
  alias table of an estate's namespace (with the platform ids of its import), so that the
  vehicle estate's build can take it with `--alias-table`. Until then the crosswalk is tested on
  a synthetic table in the host's shape, and the vehicle export names no key.
- **The host-side test** ("a test on both sides"): the host's guard was run here against the
  export; a test in the host's suite holding an export with `iri`, `labels` and `match` is the
  host's to add (this repository never writes there).

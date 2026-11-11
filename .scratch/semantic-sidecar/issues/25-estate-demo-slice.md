# Build: the Ontology page over the estate

Type: build
Mode: HITL
Status: resolved
Blocked by:

## Question

After TalkToData and TalkToDashboards, the owner wants to open an Ontology entry in the same
application and show the ontology and knowledge graph the engine built over the same data
(owner, 2026-11-11). This is the first vertical slice of the re-scoped engine: read an estate,
induce its schema level, build the instance level, and show both in the host.

**Decided by the owner on 2026-11-11:**

- the estate is the vehicle and warranty estate (the one with metric views);
- built and tested here on the DuckDB mirror; the Databricks run is the owner's;
- the screen is written natively in the host's user interface, right under Dashboards; a
  shared registry package comes later ([ADR-0074](../../../docs/adr/0074-a-host-shows-the-engines-export-in-a-read-only-screen-of-its-own-off-the-answer-path.md));
- files first, a service route later; the files may be served without sign-in as long as they
  hold synthetic and public data only and personal columns are excluded by a test;
- the resolution part rests on real manufacturer names and documents, and on a seeded second
  dealer source that is labelled as seeded;
- a third cut reads the questions people ask and feeds them back as signals.

## Cuts

| Cut | Holds | State |
|---|---|---|
| A, the ontology | reader, metric views, induction with signals and support, ledger, `graph-export/2`, the host's page | built 2026-11-11 |
| B, the knowledge graph | table roles, aggregated fact edges, resolution (manufacturers, seeded dealers), document mentions, node view, Turtle download | **mostly built** (2026-11-11): entities, counted fact edges, review items, the seeded second source linked under a certified line, the host views. Open: labels from the review screen, document mentions, the Turtle download |
| C, usage signals | joins and terms from recorded questions and queries as evidence and proposals | open |
| Later | a reader for Unity Catalog run against a live workspace; a service route in the host; the generic workbench as a registry package | open |

## Cut A, as built

Code: [`../estate/`](../estate/) (a prototype with its own pinned environment and tests; the
shipped contracts are document-shaped, so real packages wait for the contract generalisation).

- **Reader** (`estate/reader.py`, `reader_duckdb.py`): objects with columns, comments and
  sensitivity; declared keys; metric views; one query call. Only the DuckDB reader exists.
- **Metric views** (`estate/metric_views.py`): the YAML the catalog stores, with joins,
  dimensions and measures.
- **Induction** (`estate/induce.py`): concepts, relations, value sets, hierarchies between
  value sets, attributes and measure bindings; each with its signals; each mined relation with
  the rows that hold and break it; mined relations under 90% support are dropped and listed.
  A relation a metric view or a host-verified join states is `declared`, a mined one
  `proposed`. A sensitive column is never queried.
- **Ledger** (`estate/ledger.py`): assertions and decisions as append-only files; a decision
  is keyed on the element's id and survives a rebuild; a rejected element leaves the layer.
- **Export** (`estate/export.py`, `schemas/graph-export-2.schema.json`): schema level only, no
  row values.
- **The host's page:** a draft pull request in the host's repository (its number 5).

**On the mirror of the vehicle and warranty estate** (28 objects, 13 metric views, about a
second, 823 aggregate queries): 38 concepts (15 entities, 6 facts, 10 value sets, 7 derived),
86 relations (8 stated by metric views, 78 proposed), 38 measures, 8 mined relations dropped.
Of the host's twelve verified joins, nine are found; one runs through a personal column and
is not checked; two point at a derived table and are left out by design.

**What the first run on the real mirror got wrong, each now fixed with a test:** a source line
number was taken as a key; the vehicle, whose own key is personal, was keyed by its owner;
relations were mined into the aggregate tables; a hierarchy that follows from two others was
listed; and the metric view's join condition was not read at all, because YAML reads the bare
key `on` as a boolean.

**Limits of cut A.**
- Which schemas are derived and which columns are load columns is told to the engine in
  `estates/vehicles.yaml`. In Unity Catalog the first comes from lineage; that reader is not
  written.
- Value sets are found by shared column names; two columns that mean the same under different
  names are not joined.
- Labels come from table names and metric-view display names; nothing writes definitions.
- No reader for Unity Catalog exists yet, so nothing has run against the estate in Databricks.

## Cut B, as built so far (2026-11-11)

- **Instances** (`estate/instances.py`): every value of a value set and every row of master
  data is a node. Master data is what other objects refer to, with a single-column key and at
  most 2,000 rows. A fact row is never a node: a fact is a count on an edge ("this dealer has
  240 claims", "this dealer and this part meet in 6 claims"), with the object and the columns
  it was counted over.
- **Names that may be the same** are listed for review with the reason (the same first word,
  or a shared rare word). Nothing is merged: without labels there is no certified band, and
  the export says `certified: false`.
- **Personal data:** an object about persons gives no node. An object that only carries a
  contact person's columns gives nodes when the estate file names it under
  `without_personal_columns`; those columns are still never read. The vehicle estate names
  the dealers.
- **On the real mirror:** 1,240 nodes (211 dealers, 464 parts, 76 recall campaigns, 21
  investigation cases, 468 values of ten value sets), 6,417 edges, 7 review items, among them
  two spellings of one manufacturer.
- **The host page** has two more views, entities and names that may be the same; a person's
  answer downloads with the other decisions.

**Seen on the real export and not fixed:**
- The label "Investigation cas": the singular rule cuts "cases" wrongly.
- Recall campaigns have no name column, so their label is their key.
- An investigation case refers to a subject value identical to its own label, because the
  subject column is both its name and a value set.
- Review reasons are English text and show untranslated in the host's German.
- A manufacturer value carried only by fact rows shows no links.

**The seeded second source, linked (2026-11-11).**
- `python -m estate seed` copies the mirror and adds `partner.service_accounts`: 1,274
  accounts, several per dealer, each typed its own way, and accounts of 80 workshops that are
  no dealers. The table, its columns, its concept and the summary all say it is generated.
  Which dealer an account belongs to is kept in a file outside the catalog.
- `python -m estate build --resolver ../spike` hands the dealers and the accounts to the
  resolution engine and brings its answer into the export: `same_as` edges with the
  probability and what each rests on, review items, and `meta.resolution`.
- **Result:** 938 accounts linked to a dealer by the machine under a certified line, 916 of
  them right (97.7%), 85% of the accounts that belong to a dealer; 304 pairs for review. It
  cost 1,300 labels, taken from the seed's truth in place of a reviewer.
- What the dealers showed about the engine is in the
  [spike report](../spike/REPORT.md#how-it-carries-to-new-data).

**Open in cut B.**
- **Labels from the review screen.** The engine's labels come from the seed's truth. A
  reviewer's answers on the screen are downloaded and recorded, and not yet read as labels.
- Mentions of organisations in the estate's documents.
- The Turtle download (OWL 2 QL and SKOS).
- A minimum count below which a fact count is not exported.

## From a flat model to an ontology (owner, 2026-11-11)

The owner's objection: linking records is entity disambiguation; the model above it is flat,
"not ontology nor hierarchical yet". Agreed. The route, in order of certainty:

1. **Structure the data proves** (`estate/structure.py`, built 2026-11-11). No model, no guess:
   - **roll-ups** between value sets, checked in every object that carries both, with how many
     values break them; master data hangs below the finest level it carries;
   - **equivalent value sets** (two columns that cut the rows the same way) count as one level;
   - **part-of** from keys (a claim line of a claim), and derived objects that **describe** an
     entity;
   - `broader` edges at the instance level: which value sits under which, with the contested
     ones marked;
   - names that may be the same are read against the hierarchy: two manufacturer names under
     one make is a reason more, under two makes a reason against.
   - **On the real mirror:** 10 roll-up steps (market ← dealer group ← dealer; make ← model;
     make ← manufacturer ← recall campaign; system ← component; part → make and system), one
     equivalence (market and distance unit), 3 part-of, 3 describes, 390 `broader` edges.
2. **A class tree above the tables** (`estate/classes.py`, built 2026-11-11). **Decided (owner,
   2026-11-11): no external vocabulary at first; a small set of our own top classes**
   (Organisation, Person, Product, Event, Place, Document). Two voices propose, a person
   decides; nothing is accepted by the machine:
   - **attribute patterns**, always on: the columns a concept carries and how others use it;
   - **a language model**, shown names, types, comments, keys and relations, never a row or a
     value. It may put classes of its own between a concept and a top class and writes one
     sentence of definition per concept. The estate file has to allow it
     (`classes.hosted_model`); the build writes everything that would leave to
     `classes_prompt.json`.
   - Where the voices disagree, both are kept and the proposal is marked contested.
   - **On the real mirror:** 28 of 39 concepts placed under 12 classes, six of them the
     model's own (vehicle and vehicle component under product; campaign, defect investigation
     and warranty claim under event; manufacturer communication under document); 6 contested;
     11 left unplaced on purpose (aggregates, a unit of measure, a rule table).
   - **How the model spoke here:** not through the API. The environment this was built in
     accepts no key for that host, so the prompt file was answered by a separate model
     session and the answer is kept as `estates/vehicles.classes_answer.json`
     (`--class-answer`). The direct call (`--hosted-model`) is written and tested against a
     stand-in client only; it has never reached the service.
3. **The entity level hangs off that tree** (resolved dealers, parts and manufacturers as
   instances of classes, with the hierarchy edges between them).
4. **Export in the standard forms** (`estate/turtle.py`, built 2026-11-11): one Turtle file,
   OWL 2 QL and SKOS, parsed back in the tests. Classes and concepts with subclass axioms,
   relations and attributes with domain and range, part-of as a restriction, value sets as
   schemes with `broader`. A proposal is marked, on the element or on the axiom. On the
   mirror: 3,857 triples, 41 classes, 83 object properties, 468 values. **Not in it:** master
   rows, record links, measures, and no SHACL shapes yet.

**The host page** shows all four: a class tree with definitions and the disputed placements,
the hierarchies as levels and as trees of entities (market → dealer group → dealer → linked
accounts), the way up from any concept or entity, and the Turtle download.

**Limits of step 1.**
- A roll-up needs two objects that carry both columns; one seen in a single object is not
  proposed.
- "Subject → manufacturer" holds for 97 of 107 values and is proposed; it is a weak one and a
  person should reject it.
- One part-of reads oddly: campaign completion is called a part of the dealer, because the
  campaign's own key has two columns and is not matched.
- The pattern voice alone is crude: it calls a complaint an organisation and a bulletin an
  event. It is one voice and the check on the other, not a classifier.
- Decisions on hierarchy steps and classes are downloaded from the page but are not yet read
  back by the ledger; only concepts, relations and measures are.
- The top classes are fixed in code. An estate cannot add one yet.

## Answer

The slice is built as far as this ticket goes (2026-11-11): cut A, cut B without document
mentions, and the four steps from a flat model to an ontology. What is open continues as
tickets of its own, in the map's *Route to ontology v1 and the local demonstration*:

- the export's text as codes and a claim on every proposal: [26](26-export-contract-codes-and-claims.md);
- the defects listed under *Limits of step 1*: [27](27-quality-fixes-on-the-estate.md);
- decisions on hierarchy steps and classes read back: [28](28-decisions-on-structure-and-classes.md);
- how right the proposals are: [29](29-hand-checked-list-and-score.md);
- labels from the review screen: [30](30-label-loop.md);
- the model's voice through an API: [31](31-live-class-voice.md);
- cut C, usage signals: [32](32-usage-signals.md);
- the reader for Unity Catalog: [33](33-unity-catalog-reader.md).

Document mentions stay open and are not in v1 (owner, 2026-11-11).

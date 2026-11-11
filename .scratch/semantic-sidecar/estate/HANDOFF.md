# Running the Ontology page on another machine

State on 2026-11-11, after cut A and most of cut B (entities, review items, and a seeded second source linked to the dealers). Two repositories, one branch name in both:
`claude/blissful-maxwell-bpln7a`. The engine side is in this repository (draft pull request 6);
the page is in the host's repository (draft pull request 5).

## What was verified, and where

| Step | On the machine that built it (Linux, no container, no Databricks) | Still to do on the owner's machine |
|---|---|---|
| Estate mirror as DuckDB from the host's committed fixture | done | same commands |
| Metric views printed offline | done, 13 views | same command |
| `estate build` on the mirror | done: 38 concepts, 86 relations, 38 measures; 1,240 entities, 6,417 edges, 7 review items | rerun |
| Engine tests (`estate/tests`, 51) | pass | rerun |
| Seeded second source and its link to the dealers | done on the mirror: 938 of 1,274 accounts linked, 916 right, 304 pairs for review, 1,300 labels from the seed | rerun; needs the resolution engine's environment (`../spike/.venv`) |
| The seeded table in a catalog (`out/vehicles/seeded_accounts.sql`) | **written, never run against a catalog** | read it, then run it |
| Host UI: type check, 190 unit tests, build | pass on Node 24.11.0 | rerun |
| Host UI: browser tests of the page with accessibility checks, English and German | pass against the host's e2e world | rerun |
| Host checks `layout`, `register`, `secrets`, `domain-terms` | pass | rerun |
| Host checks `names`, `client-terms`, `assistant` | **not run: they need the guard key** | run |
| The host's landing tier and full gate | **not run** | run |
| The page next to TalkToData and TalkToDashboards on the Databricks world | **not run** | run |
| Reading the estate from Unity Catalog | **not possible yet: that reader is not written** | waits for the next cut |

## Commands, in order

Paths: `$LATENCE` and `$METIS` are the two checkouts. The host needs `uv` 0.12.21 or newer
and Node 24.11.0.

```bash
# 1. the estate mirror, with the host's own tools
cd $METIS
uv run metis-synth estates load --data tools/synth/estates/nhtsa --duckdb /tmp/vehicles.duckdb
uv run metis-synth estates internal --duckdb /tmp/vehicles.duckdb
uv run metis-dev metric-views --estate vehicles plan --catalog demo > /tmp/metric_views.txt

# 2. the ontology
cd $LATENCE/.scratch/semantic-sidecar/estate
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONHASHSEED=0 .venv/bin/python -m pytest -q tests
.venv/bin/python -m estate build --config estates/vehicles.yaml --duckdb /tmp/vehicles.duckdb \
    --metric-views /tmp/metric_views.txt --verified $METIS/tools/synth/estates/catalog.yaml --out out/vehicles
.venv/bin/python -m estate export --out out/vehicles --ui-dir $METIS/reference-ui/public/data/ontology

# 3. the host's UI with the page
cd $METIS/reference-ui && npm ci && npm run build
cd $METIS && uv run metis-dev ui up --static reference-ui/dist --world dashboards --estate vehicles
# then open /ontology (the entry is right under Dashboards)
```

Step 3's last command is the owner's usual way to start the Dashboards world; add
`--databricks` as usual. The page itself needs no backend: it reads the exported file. The
file has to be exported **before** `npm run build`, because the build copies it.

## Decisions made on the page

Approve or reject on the page, press *Download decisions*, and put the file at
`out/vehicles/decisions.jsonl` (append to it if it exists). Rerun step 2. Approved elements
show as approved, rejected ones are gone from the page and stay in `assertions.jsonl`.

## The seeded second source

After step 1 (the mirror) and before the build:

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
.venv/bin/python -m estate seed --config estates/vehicles.yaml --duckdb <mirror.duckdb> \
    --out-duckdb <mirror_with_seed.duckdb> --out out/vehicles
# then build on the copy, with the resolution engine
.venv/bin/python -m estate build --config estates/vehicles.yaml --duckdb <mirror_with_seed.duckdb> \
    --metric-views <views.txt> --verified $METIS/tools/synth/estates/catalog.yaml \
    --out out/vehicles --resolver ../spike
```

- The resolution engine runs in its own environment: `../spike/.venv` with
  `../spike/requirements` installed (Splink, DuckDB, scikit-learn). The build takes about a
  minute.
- `out/vehicles/seed_truth.json` says which dealer each account belongs to. It stays outside
  the catalog and is never exported.
- `out/vehicles/seeded_accounts.sql` creates the same table in a catalog, with comments that
  mark it as generated. It has not been run against a catalog.
- Without `--resolver`, or on the mirror without the seed, the build is as before and the
  page shows no links.

## Hierarchies and classes

The build always adds the structure the data proves (roll-ups, part-of, which value sits
under which) and the class proposals of the pattern voice. For the language-model voice add
one of:

- `--class-answer estates/vehicles.classes_answer.gpt-6-luna.json`: the answer the small
  hosted model gave on 2026-11-11, kept as a file. Sends nothing anywhere and gives the same
  class tree every time. (`estates/vehicles.classes_answer.json` is an earlier answer from a
  model session without an API; the numbers in ticket 25 were built with it.)
- `--hosted-model [--max-usd 0.05]`: one call to the model the estate file names
  (`classes.model`), on the provider's official SDK (`pip install openai==3.49.0` in the
  estate's environment, `OPENAI_API_KEY` set in the shell). The estate file must allow it
  (`classes.hosted_model: true`) and state the prices. The call is made only when what would
  be sent has changed; the answer is kept in `<out>/class_answers/`, each call is logged in
  `<out>/model_calls.jsonl` with its cost, and a call whose worst case is above `--max-usd`
  is not made. Read `<out>/classes_prompt.json` first: it is exactly what is sent. Run once
  on 2026-11-11 on this estate (ticket 31).

## What to say aloud in a demonstration

- The page shows the **ontology** (concepts, relations, measures) and the **entities** read
  from the data: values, master rows, and facts as counts between them.
- **The second source is seeded.** The service accounts are generated from the dealers; the
  page marks them. They are there to show linking, not as a finding about the estate.
- **What the machine links, it links under a certified line:** 938 accounts, at least 95%
  right by the test, 97.7% right against the seed. The labels for that test came from the
  seed's truth in place of a reviewer; with a real second source a reviewer gives them, and
  reading them from the screen is not built.
- **On an estate this small the labels are most of the work** (1,300 labels for 1,274
  accounts). The number does not grow with the estate: 5,000 were enough for 1.2 million
  real company records.
- **Manufacturer names that may be the same** are still only listed for a person.
- **The class tree and the hierarchies are proposals.** Roll-ups are checked on the data and
  say what breaks them; what kind of thing a concept is was proposed by column patterns and
  by a language model that saw names only. Nothing of it is approved, and decisions on these
  two are downloaded but not yet read back by the engine.
- **Only 95 of 211 dealers carry a dealer group;** the page says so. The other 116 are in no
  tree.
- **The Turtle file** (OWL and SKOS) holds classes, properties and value hierarchies, with
  proposals marked. Master rows, record links and measures are not in it.
- **Known blemishes on the page:** the label "Investigation cas"; recall campaigns labelled
  by their key; review reasons in English on the German page.
- The ontology is built from the **local mirror**, not read live from Unity Catalog.
- **Stated** means a metric view or a verified join says so. **Proposed** means mined from the
  data and not yet reviewed by anyone. Nothing is approved automatically.
- Which tables are derived, and which columns are load columns, was **told** to the engine for
  this estate.
- The file is served **without sign-in**; it holds synthetic and public data, and no value of
  a personal column. An object with a personal column gives no rows at all, unless the estate
  file names it under `without_personal_columns` (here: the dealers, an organisation with a
  contact person); then its rows are exported without those columns. Counts of claims per
  dealer and part are exported; no claim row is.

## Credentials

None are needed for cut A. The Unity Catalog reader of the next cut will read its host,
warehouse id, catalog and client credentials from environment variables set in the shell.
Never put them in a file or a chat.

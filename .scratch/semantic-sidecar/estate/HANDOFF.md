# Running the Ontology page on another machine

State on 2026-11-11, after cut A and the first half of cut B (entities and review items). Two repositories, one branch name in both:
`claude/blissful-maxwell-bpln7a`. The engine side is in this repository (draft pull request 6);
the page is in the host's repository (draft pull request 5).

## What was verified, and where

| Step | On the machine that built it (Linux, no container, no Databricks) | Still to do on the owner's machine |
|---|---|---|
| Estate mirror as DuckDB from the host's committed fixture | done | same commands |
| Metric views printed offline | done, 13 views | same command |
| `estate build` on the mirror | done: 38 concepts, 86 relations, 38 measures; 1,240 entities, 6,417 edges, 7 review items | rerun |
| Engine tests (`estate/tests`, 26) | pass | rerun |
| Host UI: type check, 146 unit tests, build | pass on Node 24.11.0 | rerun |
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

## What to say aloud in a demonstration

- The page shows the **ontology** (concepts, relations, measures) and the **entities** read
  from the data: values, master rows, and facts as counts between them.
- **Nothing is merged by the machine.** Names that may be the same are listed for a person.
  Certified merging of a second source is the next step; the scorer for it is measured (97%
  of true links at 99.7% precision on two real company registers) and not yet connected.
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

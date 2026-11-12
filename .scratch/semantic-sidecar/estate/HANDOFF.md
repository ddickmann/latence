# Running the Ontology page on another machine

State on 2026-11-11, after tickets 26 to 33 (the ontology stream of the map). Two repositories,
one branch name in both: `claude/blissful-maxwell-bpln7a`. The engine side is in this
repository (draft pull request 6); the page is in the host's repository (draft pull request 5).

Added on 2026-11-12 (ticket 40): the proposal file for the host's semantic layer, block 8 and
the section *The proposal file for the host's semantic layer*.

## The owner's run, in order

Every block can be pasted as it is after the two paths in block 1 are set. Blocks 1 to 5 need
no credentials and make no model call. Needed on the machine: `git`, `python3.12`, `uv`
0.12.21 or newer, Node 24.11.0.

**1. Both checkouts on the branch**

```bash
export LATENCE=$HOME/latence        # change to where the two checkouts are
export METIS=$HOME/metis
cd $LATENCE && git fetch origin && git checkout claude/blissful-maxwell-bpln7a && git pull --ff-only
cd $METIS   && git fetch origin && git checkout claude/blissful-maxwell-bpln7a && git pull --ff-only
```

**2. The inputs, written with the host's own tools**

```bash
cd $METIS && uv sync
uv run metis-synth estates load --data tools/synth/estates/nhtsa --duckdb /tmp/vehicles.duckdb
uv run metis-synth estates internal --duckdb /tmp/vehicles.duckdb
uv run metis-dev metric-views --estate vehicles plan --catalog demo > /tmp/metric_views.txt
uv run metis-dev usage export \
    --golden blueprints/talk-to-data/suites/estate/golden-en.jsonl \
    --golden blueprints/talk-to-data/suites/estate/golden-de.jsonl \
    --library libraries/estate-semantics/library.yaml --out /tmp/usage.jsonl
```

**3. The two environments and their tests**

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
python3.12 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
python3.12 -m venv ../spike/.venv && ../spike/.venv/bin/pip install -q -r ../spike/requirements-link.txt
PYTHONHASHSEED=0 .venv/bin/python -m pytest -q tests                      # 124 pass
```

**4. The seeded second source, the build, the export**

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
.venv/bin/python -m estate seed --config estates/vehicles.yaml --duckdb /tmp/vehicles.duckdb \
    --out-duckdb /tmp/vehicles_seeded.duckdb --out out/vehicles
PYTHONHASHSEED=0 .venv/bin/python -m estate build --config estates/vehicles.yaml \
    --duckdb /tmp/vehicles_seeded.duckdb --metric-views /tmp/metric_views.txt \
    --verified $METIS/tools/synth/estates/catalog.yaml --resolver ../spike \
    --class-answer estates/vehicles.classes_answer.gpt-6-luna.json \
    --usage /tmp/usage.jsonl --out out/vehicles
.venv/bin/python -m estate export --out out/vehicles --ui-dir $METIS/reference-ui/public/data/ontology
```

The build takes about a minute, most of it the record linker. Expected on the vehicle estate:
39 concepts, 87 relations, 38 measures; 2,514 entities; 938 of 1,274 accounts linked.

**5. The page**

```bash
cd $METIS/reference-ui && npm ci && npm run build
cd $METIS && uv run metis-dev ui up --static reference-ui/dist --world dashboards --estate vehicles
```

Then open `/ontology` (the entry is right under Dashboards). The last command is the usual
way to start the Dashboards world; add `--databricks` as usual. The page needs no backend: it
reads the exported file. The export has to be written **before** `npm run build`, because the
build copies it.

**6. The hand-checked list** (ticket 29; until this is done no score counts)

Open `estates/vehicles.gold.yaml`, correct what is wrong, decide the three entries under
`unsure` or leave them there, and put a name and a date under `checked_by`. Then:

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
.venv/bin/python -m estate score --out out/vehicles --gold estates/vehicles.gold.yaml
git add estates/vehicles.gold.yaml && git commit -m "estate: the hand-checked list for the vehicle estate, corrected"
git pull --rebase origin claude/blissful-maxwell-bpln7a && git push origin claude/blissful-maxwell-bpln7a
```

**7. One build against the real catalog** (ticket 33; the reader has only met a stand-in)

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
export DATABRICKS_HOST=https://<workspace>        # in the shell, never in a file
export DATABRICKS_WAREHOUSE_ID=<id>
export DATABRICKS_TOKEN=<token>                    # or DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET
.venv/bin/python -m estate build --config estates/vehicles.yaml --catalog <catalog> \
    --verified $METIS/tools/synth/estates/catalog.yaml \
    --class-answer estates/vehicles.classes_answer.gpt-6-luna.json --out out/vehicles-uc
.venv/bin/python -m estate compare --a out/vehicles --b out/vehicles-uc
```

Report back the output of `compare`, or the first error with the statement id it names. The
seeded second source is not in the catalog unless `out/vehicles/seeded_accounts.sql` was
loaded, so the catalog's build links no records; that difference is expected.

**8. The proposal file for the host's semantic layer** (ticket 40; after decisions were made)

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
.venv/bin/python -m estate propose --config estates/vehicles.yaml --out out/vehicles \
    --decided-on <YYYY-MM-DD>                # the day of the decisions that state none
cd $METIS
uv run metis-dev propose estate-semantics --apply --release <version> --receipt /tmp/receipt.json \
    --glossary $LATENCE/.scratch/semantic-sidecar/estate/out/vehicles/proposal.yaml
```

Read what `estate propose` prints before the second command: every entry written, and every
approved decision that was not, with the reason. The host's tool prints one line per entry:
placed, already there, or refused and why. Then rerun the build of block 4 with
`--receipt /tmp/receipt.json` added, and its export: the page says on those elements since
which version of the library they are in the semantic layer.

The host's change of its library is reviewed and released as the host's handbook says; this
side only writes the file and reads the receipt.

## What was verified, and where

| Step | On the machine that built it (Linux, no container, no Databricks) | Still to do on the owner's machine |
|---|---|---|
| Estate mirror, metric views, usage log from the host's fixtures | done | blocks 2 |
| Engine tests (`estate/tests`, 124) and matcher tests (`spike`, 155) | pass | block 3 (the engine's) |
| The matcher in a clean environment from `spike/requirements-link.txt` | the same resolution file byte for byte | block 3 |
| `estate build` on the mirror with the seeded source, the kept model answer and the usage log | done, twice, byte-identical | block 4 |
| One live call of the class voice on the small hosted model | done once (ticket 31), USD 0.0035 | not needed; the answer is kept as a file |
| Decisions of every kind applied by the next build; reviewer labels changing the link | done (tickets 28, 30) | by hand on the page, if wanted |
| Host UI: type check, 209 unit and route tests, build | pass on Node 24.11.0 | rerun |
| Host UI: browser walk of the page with accessibility checks, English and German | pass against the host's e2e world | rerun |
| Host checks `layout`, `secrets`, `domain-terms` | pass | rerun |
| Host checks `names`, `client-terms`, `assistant` | **not run: they need the guard key** | run |
| The host's landing tier and full gate | **not run** | run |
| The page next to TalkToData and TalkToDashboards on the Databricks world | **not run** | run |
| Reading the estate from Unity Catalog | **against a stand-in server only** | block 7 |
| The proposal file (ticket 40), from the two manufacturer pairs and the dealer list, with decisions written for the trial | placed by the host's tool in a throwaway checkout of the host's final state of 12 Nov: onto the library before the host's own change the result is byte for byte the host's release 1.1.0; onto 1.1.0 all three entries are already there; an entry on an object the layer does not read is refused by name | block 8, with decisions a person made |
| The receipt read back into the export (`in_layer`) | done on a fresh build of the vehicle estate **without the record linker** | block 8 |
| The seeded table in a catalog (`out/vehicles/seeded_accounts.sql`) | **written, never run against a catalog** | read it, then run it |
| The score against a list a person checked | **draft list only; its numbers do not count** | block 6 |

## Decisions made on the page

Approve or reject on the page (concepts, relations, hierarchy steps, part-of, classes and
placements, pairs of names), press *Download decisions*, and put the file at
`out/vehicles/decisions.jsonl` (append to it if it exists). Rerun block 4.

- An approved element shows as approved; a rejected one is gone from the page and the Turtle
  file and stays in `assertions.jsonl`.
- A rejected hierarchy step is taken out before the steps are reduced, so a step it hid
  returns. A rejected placement falls to the other voice's class, or to none.
- Each decision carries what the element stated when it was made. If the next build states
  something else under the same id, the decision is not applied and is listed as stale.

## The proposal file for the host's semantic layer

The one way anything a review approved reaches the host's semantic layer (ADR-0070).
`estate propose` writes `out/vehicles/proposal.yaml`; its shape is
`schemas/proposal-file.schema.json`, written from the host's description of what its tool reads.

- **What is in it.** For every approved pair of names of a value set: the value kept and its
  other spelling, stated on a catalog object and column (`spellings`). For every approved
  master list that the estate file names under `propose.names` (here the dealers): each
  record's key with its name (`names`). Every entry carries the element's id, its claim, and
  the decision: approved, by whom, on which day.
- **The dealers need a line written by hand.** The page offers no decision on a stated
  concept. Append to `out/vehicles/decisions.jsonl`:
  `{"subject": "concept:internal.dealers", "verdict": "approved", "by": "<person id>", "claim": "<claim>", "at": "<YYYY-MM-DD>"}`,
  with the claim the export states for it:
  `.venv/bin/python -c "import json; print(next(c['claim'] for c in json.load(open('out/vehicles/ontology.json'))['concepts'] if c['id'] == 'concept:internal.dealers'))"`.
- **The day.** A line the page downloads carries no day. Add `"at"` to the line, or give
  `--decided-on` for all the lines without one. Without a day an entry is not written.
- **What is never in it:** a rejected or undecided element; the seeded service accounts, any
  pair that holds one, and any pair of values of a value set the seeded table carries; a link
  between records, the machine's (certified here on simulated labels) and a person's alike;
  a decision made on another statement than the build now makes (printed under `stale`); a
  decision without a claim. An approved hierarchy step, relation or class is counted under
  `not_carried`: the file has no entry for it.
- **Which object a pair is stated on.** The one that holds the most values of the value set,
  the first by name among equals (for the manufacturers: `clean.recall_campaigns`). If the
  host refuses an entry because no dimension reads that column, name another object in the
  estate file (`propose.spellings_on: {manufacturer: clean.recalls}`) and write the file again.
- **What the host does with it.** `metis-dev propose <library> --glossary <file>` without
  `--apply` (it then also wants `--sql`) lists each entry with the node it would change;
  with `--apply` it writes the placed entries into the library, each with its origin, and
  `--receipt` writes what was placed, what was already there and what was refused. The host
  refuses on its own what is marked seeded or rests on simulated labels; this side never
  writes such an entry.
- **The receipt.** `estate build --receipt <file>` (repeat for more, oldest first) puts
  `in_layer` (the library and its version) on an approved element the receipt lists as
  *placed*, if the element still states what was placed. An element a receipt lists as
  already there gets nothing from that receipt: it does not say since when. The host's
  release 1.1.0 was placed from a file written by hand; its receipt names the same elements
  and claims, so `--receipt $METIS/tools/synth/estates/decided/ontology.receipt.json`, given
  first, is what says "since 1.1.0" for the two pairs and the dealers, once a person's
  approvals of them are in `decisions.jsonl`.

## The seeded second source and the labels

- `estate seed` writes the partner's service accounts from the dealers, and
  `out/vehicles/seed_truth.json`, which says which dealer each account belongs to. That file
  stays outside the catalog and is never exported.
- The matcher asks for labels in two rounds: first the pairs it learns from, then two test
  samples drawn from its own scores. On the page they are under *Possibly the same*, *Pairs to
  label*; a test pair is shown without the machine's score.
- A reviewer's label wins. Where the estate file allows it (`link.labels.truth_stands_in`, on unless set to false), the seed's
  truth fills the rest, and the page says per sample how many labels came from a person and
  how many were simulated. With open labels and no truth nothing is linked.
- Without `--resolver`, or on the mirror without the seed, the build is as before and the
  page shows no links.

## Classes, and the hosted model

The build always adds the structure the data proves and the class proposals of the pattern
voice. For the language-model voice add one of:

- `--class-answer estates/vehicles.classes_answer.gpt-6-luna.json`: the answer the small
  hosted model gave on 2026-11-11, kept as a file. Sends nothing anywhere and gives the same
  class tree every time. This is what block 4 uses.
- `--hosted-model [--max-usd 0.05]`: one call to the model the estate file names
  (`classes.model`), on the provider's official SDK (`pip install openai==3.49.0` in the
  estate's environment, `OPENAI_API_KEY` set in the shell). The estate file must allow it
  (`classes.hosted_model: true`) and state the prices. The call is made only when what would
  be sent has changed; the answer is kept in `<out>/class_answers/`, each call is logged in
  `<out>/model_calls.jsonl` with its cost, and a call whose worst case is above `--max-usd`
  is not made. Read `<out>/classes_prompt.json` first: it is exactly what is sent. The model
  sees names, types and comments, never a row or a value.

## Reading Unity Catalog instead of the mirror

`--catalog <name>` in place of `--duckdb` reads the estate from Unity Catalog and runs the
counting statements on a SQL warehouse (`estate/reader_databricks.py`). Host, warehouse and
credentials come from the environment (block 7). **Proven against a stand-in server only; not
yet run against a workspace.** The stand-in proves the protocol (pages, pending statements,
results in parts, a refused token, a failed statement); it does not prove the warehouse's SQL.
Lineage, row filters and column masks are not read.

## What to say aloud in a demonstration

- The page shows the **ontology** (concepts, relations, measures, a class tree, hierarchies)
  and the **entities** read from the data: values, master rows, and facts as counts between
  them. The entity view draws the graph around one entity; choosing a box walks on from there.
- **The second source is seeded.** The service accounts are generated from the dealers; the
  page marks them. They are there to show linking, not as a finding about the estate.
- **What the machine links, it links under a certified line:** 938 accounts, at least 95%
  right by the test, 916 right against the seed. In this build every label came from the
  seed's truth in place of a reviewer, and the page says so. A reviewer's labels are read
  when they exist; this was run once with 64 answers.
- **On an estate this small the labels are most of the work** (1,300 labels for 1,274
  accounts). The number does not grow with the estate: 5,000 were enough for 1.2 million
  real company records.
- **The class tree and the hierarchies are proposals** until a person approves them. Roll-ups
  are checked on the data and name the values that break them; what kind of thing a concept
  is was proposed by column patterns and by a language model that saw names only. A model's
  definition is marked as a model's until the placement is approved.
- **How right the proposals are is not measured yet.** The list they are scored against is a
  draft nobody has checked.
- **Only 95 of 211 dealers carry a dealer group;** the page says so. The other 116 are in no
  tree.
- **What people ask** is shown as a signal: 408 golden questions, 358 answered. They are
  synthetic and written from the same metric views the model is induced from, so the counts
  say what the host's test suite covers, not what people want. Use changes no status.
- **The Turtle file** (OWL 2 QL and SKOS) holds classes, properties and value hierarchies;
  what nobody approved is written as a marked proposal, not as a plain statement. Master
  rows, record links and measures are not in it.
- The ontology is built from the **local mirror**, not read live from Unity Catalog, until
  block 7 has run.
- **Stated** means a metric view or a verified join says so. **Proposed** means mined from the
  data and not yet reviewed by anyone. Nothing is approved automatically.
- Which tables are derived, and which columns are load columns, was **told** to the engine for
  this estate.
- The file is served **to a signed-in session only** (Metis 213 S-2: anything under the
  UI's `data/` needs a session, and the image build takes no export from the working tree).
  Sign in before opening `/ontology`; a signed-out request gets a 401. The file holds
  synthetic and public data, and no value of a personal column. An object with a personal column gives no rows at all, unless the estate
  file names it under `without_personal_columns` (here: the dealers, an organisation with a
  contact person); then its rows are exported without those columns. Counts of claims per
  dealer and part are exported; no claim row is.

## Credentials

Blocks 1 to 6 need none. Block 7 reads the workspace's host, the warehouse id and a token (or
a service principal's id and secret) from the shell's environment. `--hosted-model` reads
`OPENAI_API_KEY` from the shell. None of them belongs in a file.

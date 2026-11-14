# Build: every use case walked in a browser on the local backend

Type: task
Mode: AFK
Status: open: every browser spec passes, the acceptance run included (14 Nov); the host's Postgres-backed Python tests are not run to the end on this machine
Blocked by: 36

## Question

"So that all use cases work properly" (owner, 2026-11-11). On the local backend, with scripted
models, each use case is walked in a real browser, and what breaks is fixed.

## Goal

- Run the host's existing browser specs on the scripted world: workbench, samples,
  pre-screening, console (overview, kill switch, changes, why), shell, ontology, TalkToData;
  the acceptance run; the dashboards spec on its world.
- **One new spec**, the only path no spec covers today: TalkToData on the estate world, through
  DuckDB and pgvector: an answer that cites a document, and the notice a person sees when a row
  filter applies to them.
- The host's Python tests that need Postgres and pgvector.
- **Defects** (owner: fix everything, the host's core included): each with a test that failed
  first, in a pull request of its own in the host's repository, listed in the table below.

## Done check

Every spec above passes on this machine; the table of defects is complete; what was not walked
is named.

| Defect | Where | Test that was red | Fix |
|---|---|---|---|

## Tests that prove it

The specs themselves. No further tests are added beyond the one new spec and one test per
defect.

## Limit

Scripted models answer prepared questions only. What a real model does on this backend is
ticket 38 and the owner's hand rehearsal.

## State (2026-11-11, 23:20 UTC): walked on the worlds the launcher started

| Use case | Spec | Result |
|---|---|---|
| Console (overview, kill switch, changes, why) | `console.spec.ts` | pass, en and de |
| Screening experiment | `experiment.spec.ts` | pass, en and de |
| Ontology | `ontology.spec.ts` | pass, en and de, and the empty page |
| Pre-screening at scale | `prescreening.spec.ts` | pass, en and de |
| Sampled decisions | `samples.spec.ts` | pass, en and de |
| Shell | `shell.spec.ts` | pass, en and de |
| TalkToData on the scripted source | `talk-to-data.spec.ts` | 9 of 9 pass |
| Workbench | `workbench.spec.ts` | 4 of 4 pass |
| TalkToDashboards on its world | `dashboards.spec.ts` | pass, en and de |
| TalkToData through DuckDB and pgvector | `talk-to-data-estate.spec.ts` (new) | pass |
| The acceptance run | `acceptance.spec.ts` | **not yet run** |

28 specs pass; every screen with the accessibility check.

**The new spec is narrower than planned, with the reason.** A document-cited answer cannot come
from the scripted models: they hold the estate's golden questions, and none of those reads a
document. The spec proves what the local stores do without a model: a figure read under the
asker's row rights with the notice, and a document question that is searched in the documents
the person may read and is honestly not answered. A cited answer needs a real model (ticket 38
or the owner's hand walk).

| Defect | Where | Test that was red | Fix |
|---|---|---|---|
| A question the scripts do not hold, put after one they do, came back as the earlier question's figure and table under the new question | the scripted oracle of the host's dev tool (`ui_talk.py`), which looked for a known question anywhere in a call that carries the earlier turns | `test_the_scripted_world_answers_the_question_put_and_never_an_earlier_one`; seen first in a browser on the TalkToData world | the oracle reads the question the asker put, where the call states it (host commit `66caed57`) |

**Seen and not changed:** under a refusal the scripted composer still writes "The table below
holds the answer." although there is no table. It is the script's sentence, not the engine's;
a real model does not write it.

**Still to do here:** the acceptance run; the host's other Python tests that need Postgres.

## From the map (moved 12 Nov)

Route table row: every use case walked in a browser; defects fixed | AFK | 36

## State (14 Nov): the acceptance run passed; the host's Postgres tests, see below

- **The acceptance run** (`acceptance.spec.ts`, `npm run e2e:acceptance` with
  `playwright.acceptance.config.ts`), on the host's integration head `c883b5d2`, in a throwaway
  clone of the host's checkout (the checkout itself was only read), the acceptance world built
  on the scenarios' scripts on its own port and state folder: **2 of 2 pass** (en and de, 3.2 min
  with the world's build), every screen with axe's WCAG 2.2 AA rules: 0 violations on 10 screens.
  No defect, so nothing for the table above.
- **The host's Python tests that need Postgres: not run to the end here.** The host's suite
  runs every store test on both backends, so these are 55 files (the plug-ins for Postgres and
  pgvector, the Postgres catalog and store, the graph, DuckDB and Databricks tools, the dev and
  synth tools' Postgres tests, and the core's and service's tests parametrized over Postgres):
  2,790 tests. Two attempts on this machine, shared with other runs, ended early: the first
  when the disk filled (its errors are the disk's, not the tests'), the second stopped on purpose
  at about a third of the 953 tests outside the core and service (no failure so far) when the
  shared Postgres cluster had grown to 3.4 GB and under 1 GB of disk was left. No defect was
  seen. **Live request** (the owner's machine, Postgres 17 with pgvector from its package
  manager, as the handoff says):
  `cd $METIS && METIS_TEST_POSTGRES_URL=postgresql://<user>@:<port>/postgres?host=<socket dir> uv run pytest -n 4 <the 55 files> packages/metis-service/tests`,
  the file list being every test file under `packages/` and `tools/` that names
  `postgres_database`, `postgres_backend`, `METIS_TEST_POSTGRES_URL`, `METIS_ESTATE_POSTGRES`,
  `pgvector` or `PostgresBackend` (`grep -rlE` over those names). Report the summary line.

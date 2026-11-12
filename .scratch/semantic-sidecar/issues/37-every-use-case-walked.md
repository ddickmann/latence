# Build: every use case walked in a browser on the local backend

Type: task
Mode: AFK
Status: open
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

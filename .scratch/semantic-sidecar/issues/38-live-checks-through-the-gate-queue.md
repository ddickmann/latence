# Build: live checks on real models, through the owner's runner

Type: task
Mode: HITL
Status: open
Decided by the owner (12 Nov): Will you start the runner with both lanes; Start it; the checks
run at low reasoning, as the model file sets
Blocked by: 37

## Question

Real models are not reachable from the machine this work runs on: the guard models run on the
owner's computer. The host's project has a queue for this: a request file pushed to the
companion repository is picked up by a runner on that computer, which runs a fixed command and
pushes the result back.

**Owner, 2026-11-11:** use the queue. Request files, and nothing else, may be pushed to the
companion repository's main branch. Spend limit USD 25 for all hosted-model calls of this work.

## Goal

Requests on the host's working branch, after ticket 37 is green:

- the landing gate, with the three guard checks that need the owner's key;
- the user-interface lane;
- one stability run of TalkToData on the configured chat model, one run per question, with a
  cost limit inside the spend limit.

Results are read back from the companion repository and recorded here: the request, the
verdict, the cost.

## Done check

Each request has a result recorded here, green or with its failure read and either fixed
(ticket 37's rule) or named as open.

## Not covered by the queue

The queue has no kind that walks a live world in a browser. Dashboards and the Ontology page
on live models are the owner's hand walk from the runbook (ticket 36).

## Needs the owner

The runner started with both lanes and the model servers up, and confirmation that the
runner's clones read from the hosted repository.

## Open point

The owner named medium reasoning for the chat model; the host's file sets low, by the owner's
decision of 2 November after medium ran drafts into the timeout. The checks run with the file
as it is unless the owner says otherwise.

## From the map (moved 12 Nov)

Route table row: live checks on real models through the owner's runner | HITL | 37

## Decision brief for the owner (12 Nov)

- **Question:** Will you start the runner with both lanes, and do the checks run at low or
  medium reasoning?
- **Why his:** his machine and guard key; the effort is his 2 Nov decision.
- **Options:**
  - A. Start it; low.
  - B. Start it; medium.
  - C. Skip the queue and rely on the local agent's rehearsal.
- **Facts:**
  - The file sets low: "medium made drafts slower and ran 3 of 9 into the timeout"
    (`deploy/models/openai-talk-to-data-terra.yaml` (in the host's repository) line 34).
  - A local agent already rehearsed on real models on 12 Nov (`local-agent-handoff-4.md`).
  - What only the queue gives: the landing gate with the three name guards (not run anywhere
    without the key), the UI lane, one stability run.
  - 38 is blocked by 37, whose `acceptance.spec.ts` is "not yet run".
- **Recommendation:** A. Low is what every recorded measurement since 2 Nov ran on, and the
  name guards have not run on this branch at all.
- **Urgency:** before the demonstration. Blocks 39 and the merge.
- **Cost:** starting the runner; spend inside the USD 25 limit.
- **Decided by the owner (12 Nov):** Start it; the checks run at low reasoning, as the model
  file sets.

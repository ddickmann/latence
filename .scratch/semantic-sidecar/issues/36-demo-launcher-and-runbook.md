# Build: one command starts the demonstration, one page says how to give it

Type: task
Mode: AFK
Status: resolved
Blocked by: 35

## Question

The host starts one world per use case, each a server of its own. A demonstration that moves
from claims to TalkToData to Dashboards to the Ontology page needs three of them up at once on
ports that do not collide with each other or with the model servers, and a person who has not
built it needs to know the order, the sign-in people and the addresses.

## Goal

- In the host's dev tool: `ui demo up|down|status`. It starts the scripted world, the
  TalkToData world and the Dashboards world on three separate port ranges, each with its own
  state folder and the built user interface, and reports each as up or not. It passes through
  the choice of scripted or real models and the switch to the Databricks binding.
- The ontology export is built and copied into the user interface's data folder first.
- One runbook in the host's handbook: the commands, the people to sign in as, the order of the
  demonstration, what each step shows, and what is synthetic.

## Done check

`ui demo up` on this machine brings up all three worlds, and each use case's first screen
opens.

## Tests that prove it

- The three port ranges do not overlap each other or the model servers' range.
- The switch to the Databricks binding is refused for a world where the host's own `run`
  refuses it, with the same message.

## Not built

One server for several worlds. The demonstration moves between addresses.

## Answer

Built on 2026-11-11, in the host's repository (commit `6eba75d7`).

- **The command is `metis-dev demo up|down|status`**, not `ui demo`: the `ui` command takes
  one action word, and a second one would have made every `ui` call accept a stray argument.
- It starts the claims world (the scripted one), TalkToData and Dashboards, each as `ui up` in
  a process group of its own on ports 8400, 8410 and 8420, waits until each answers and prints
  the addresses and the people to sign in as. `down` stops them; `status` says which answer
  and whether the Ontology page has an export.
- Model options reach the two worlds that call a model; the Databricks binding reaches the
  worlds `ui up` binds there, by one rule both commands now share; Postgres goes to TalkToData
  alone. Without Postgres and without the Databricks binding it stops before starting
  anything.
- **It does not build the ontology export.** That is the sidecar's, and the host must not
  depend on it. `up` and `status` say whether the export is in the built user interface, and
  the runbook says where to put it.
- The runbook is a section of the host's handbook page on the reference UI.
- **Tests, 5:** the three worlds' ports overlap neither each other nor the model servers'; each
  world gets its options and no other; a world is started, waited for and stopped without a
  process left; one that stops while starting is reported at once; nothing is started without
  Postgres.
- **Done check:** `demo up --scripted-models` brought up all three worlds on this machine, and
  the browser walks of ticket 37 ran against them.

## From the map (moved 12 Nov)

Route table row: one command starts it, one page says how to give it | AFK | 35

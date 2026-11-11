# Build: one command starts the demonstration, one page says how to give it

Type: task
Mode: AFK
Status: open
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

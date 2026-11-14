# Build: an actor on every merge decision, and a person's links kept as must-link and cannot-link

Type: task
Mode: AFK
Status: open
Blocked by: nothing
Emerges from: 13 (decided by the owner, 14 Nov: "every human decision is kept as a must-link or
cannot-link with its actor on each merge decision")

## Question

`MergeDecision` in the core's contracts says which strategy proposed a merge, with what
confidence and on what evidence, but not who decided it. The estate's record linker keeps a
person's answer on a pair in the decisions file and replays it on every build, but a link the
machine made says nothing of who made it. Every merge decision is to carry its Actor (the
engine, a model with its name, or a person), and a person's answers are to stay must-link and
cannot-link constraints that every build replays.

## Goal

- The core: `MergeDecision.actor`, required, in one of three forms (`engine`, `model:<name>`,
  `human:<id>`); every producer states it; `SCHEMA_VERSION` bumped.
- The estate: every `same_as` link says who made it (the engine, under which line, or the
  person); a person's "same" is a must-link and "different" a cannot-link, replayed on every
  build whatever the linker proposes.

## Done check

- No `MergeDecision` validates without an actor; the core's and the linker package's producers
  state theirs, with tests.
- On the estate, a link the machine made and one a person made say so; a person's cannot-link
  removes the machine's link on every rebuild, and a must-link holds where the machine proposed
  nothing.

## Not in this build

The text path's Disambiguation takes no person's decisions yet (no review surface feeds it);
replaying constraints there, and under a Delta, follows with that surface.

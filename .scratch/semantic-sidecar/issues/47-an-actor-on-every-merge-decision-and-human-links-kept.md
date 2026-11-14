# Build: an actor on every merge decision, and a person's links kept as must-link and cannot-link

Type: task
Mode: AFK
Status: resolved (14 Nov): built; the text path's constraints follow with its review surface
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

## Result (14 Nov)

- **The core.** `MergeDecision.actor` is required and validated (`engine`, `model:<name>`,
  `human:<id>`); no value is assumed. `SCHEMA_VERSION` is 23 (the history comment in
  `contracts.py` says why it breaks: a v22 entity with merge decisions no longer validates and is
  re-derived from its Source). The engine's `MergeAudit` carries `actor` (`engine` for the
  cascade's rules), and the Stage maps it onto the contract; the two Disambiguator packages build
  their decisions through that mapping and needed no change. The contract reference was
  regenerated; `CONTEXT.md`'s *Actor* entry now says where it is in code and where not; the
  tutorial's example record and `CHANGELOG.md` say 23.
- **The estate.** Every `same_as` link says who made it: `actor: engine`, or `human:<id>` with
  `decided_by`. A person's "same" is a must-link (drawn even where the linker proposed nothing
  that time) and "different" a cannot-link (the machine's link is taken away), both from the
  decisions file and so replayed by every build. The export's schema states `actor`.
- **Tests, red first:** `test_a_merge_decision_says_who_made_it` and the actor added to
  `test_merge_decision_records_below_policy_as_not_applied` (red: the contract had no actor),
  `test_every_merge_decision_of_the_cascade_is_the_engines` (red: no actor on the decisions);
  in the estate, `test_every_link_says_who_made_it_and_a_persons_answers_are_constraints_every_build_replays`
  (red: `KeyError: 'actor'`).
- **Not built**, as the ticket says: the text path's Disambiguation takes no person's decisions
  (no review surface feeds it), so must-link and cannot-link there, and under a Delta, follow
  with that surface.

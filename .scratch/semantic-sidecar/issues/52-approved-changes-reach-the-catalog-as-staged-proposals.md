# Build: approved changes reach the catalog as staged proposals its owner applies

Type: task
Mode: AFK (the writer); HITL (applying it in a workspace)
Status: open
Blocked by: nothing
Emerges from: 16 (decided by the owner, 14 Nov: "approved changes reach the catalog as staged
proposals its owner applies, never as direct writes")

## Question

Nothing on this side writes to Unity Catalog, and nothing proposes a change to it. An approved
comment on an object, a declared key or a tag is to reach the catalog as a file of statements
its owner reviews and applies, never by a call from the engine.

## Goal

A command that writes, from approved decisions only, a staged proposal: one statement per
change (a comment, a key, a tag), each with the decision it rests on, in the warehouse's
dialect, never run by the engine.

## Done check

- Tests: only approved (or certified) elements give a statement; each statement names its
  origin; text is quoted so that no statement can be broken by a name; the reader has no call
  that writes.
- Applying it in a workspace is the owner's.

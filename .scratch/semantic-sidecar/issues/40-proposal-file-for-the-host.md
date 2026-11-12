# Build: the proposal file for the host, written from approved decisions

Type: task
Mode: HITL (the host's owner approves what enters its semantic layer)
Status: open, before the demonstration
Blocked by: nothing
Emerges from:
`docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md`
("Nothing here is built yet"), the host's ticket 220 (its design proposal and the owner's
decision of 12 Nov)

## Question

The decision record makes one seam to the host: a reviewable proposal file in the host's own
proposal shape. Nothing writes it. `estate/estate/` has no module for it: the build writes the
export, the Turtle file and the ledger, and reads a decisions file back.

What the host's ticket 220 asks of this side, as its owner decided on 12 Nov ("Yes, build it as
proposed", with a real manufacturer spelling pair as the example):

- From approved decisions only, a proposal file in the shape the host's `metis-dev propose`
  reads, extended by two kinds of entry: two spellings a reviewer decided are one value, and
  the names of a master list's records.
- Each entry carries the decision it rests on: who, when, and the element's claim.
- Anything marked seeded, or certified on simulated labels, is never written.
- The export says of an approved element since which version of the host's library it is in the
  semantic layer, learnt from the proposal file's receipt.

## Goal

`python -m estate propose` (or the name the build settles on) writes that file for the vehicle
estate from `decisions.jsonl`; the host's tool places each entry or refuses it by name.

## Done check

- The file for the vehicle estate holds the decided pair of manufacturer spellings and the
  dealers' names, and nothing seeded.
- Two runs on the same inputs give the same bytes.
- The host's ticket 220 names the commit here that it was tried against.

## Tests that prove it

- A rejected or undecided element is in no proposal.
- A seeded record and a link certified on simulated labels are in no proposal.
- An entry whose decision's claim no longer matches the build is listed as stale and not
  written.

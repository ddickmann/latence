# Build: drift becomes a new proposal, and the old entry is marked stale

Type: task
Mode: AFK
Status: open
Blocked by: nothing
Emerges from: 16 (decided by the owner, 14 Nov: "drift (a rename, split, retype or drop)
becomes a new proposal with the old entry marked stale")

## Question

When the catalog changes under an approved element (a column renamed, a table split, a type
changed, an object dropped), the next build states something else under the same id or no
longer has the element. Today a decision on it is listed as stale or unknown, and nothing says
what changed.

## Goal

The build compares the elements a person decided with what it states now and, for each that
changed, says what drifted (renamed, split, retyped, dropped), keeps the old decision on the
record as stale, and lists the element as it is now as a new proposal for review.

## Done check

Tests on the toy estate for each of the four kinds of drift.

# Build: a column that turns sensitive is erased from every export and the ledger

Type: task
Mode: AFK
Status: open
Blocked by: nothing
Emerges from: 16 (decided by the owner, 14 Nov: "a column that turns sensitive is erased from
every export and ledger, recorded as an erasure")

## Question

A column that was read as ordinary may later be tagged sensitive. The next build no longer reads
its values, but what earlier builds wrote stays: values in the ledger's assertions and in the
exports of earlier builds in the folder.

## Goal

When a build finds a column sensitive that an earlier build read, it removes every value of
that column from the ledger and from the files of the build folder, and appends an erasure
record: which column, when, by which build, how many lines changed; never the values.

## Done check

Tests: a value seen before is in no file of the folder afterwards; the erasure is recorded once;
a second build erases nothing more.

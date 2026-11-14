# Build: a column that turns sensitive is erased from every export and the ledger

Type: task
Mode: AFK
Status: resolved (14 Nov): built in the estate prototype
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

## Result (14 Nov)

- `estate/erasure.py`, run by every build before it reads the folder's decisions. A column
  turned sensitive when the ledger shows it read as ordinary (an attribute marked not
  sensitive, or an object a value set was read from) and the catalog now says it is sensitive
  (a tag or the estate file). Where the build no longer holds a value set over it, every value
  of it that the earlier export held (its value rows, their other spellings, the pairs of its
  values) is erased from the build folder: a line of a `.jsonl` file that holds one (the ledger,
  the decisions, the calls) is removed; any other file that holds one (a proposal file, a staged
  file) is removed; the export and the Turtle file are written again from what the build reads
  now.
- **The record:** `erasures.jsonl`, one line per column erased: the column, the objects, the
  build's number in the ledger, how many values, lines and files; never a value. The build's
  summary says the same (`erased`). A later build that finds nothing more records nothing.
- **The ledger** is append-only except here: a legal erasure outranks it (CONTEXT's *Purge*).
  A decision that held no value stays, and is applied as before.
- **Limits:** the values are known from the export the earlier build left in the folder; a
  folder whose export was already removed gives nothing to look for. A value set still read from
  an object where the column is not sensitive keeps its values (the host's guard refuses a value
  set over a sensitive column only where that object is named). Files outside the build folder
  (a host's copy of the export, a proposal file already handed over) are the host's to erase.
- **Test** (`estate/tests/test_erasure.py`; red on the tree before: the values stayed): the toy
  estate's markets read as ordinary, a decision on two of them and a file written from them;
  the column made sensitive in every object; no value left in any file of the folder, one
  erasure recorded without a value, the decision that held none still applied; a second build
  records nothing.

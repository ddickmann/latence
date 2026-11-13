# Build: the proposal file for the host, written from approved decisions

Type: task
Mode: HITL (the host's owner approves what enters its semantic layer)
Status: open (built and tried against the host's tool; the host's ticket 220 does not yet name the commit here)
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

## State (2026-11-12): built, tried against the host's final state of 12 Nov

Commit `24f82e7` here is the one the host's ticket 220 is to name.

- **The command.** `python -m estate propose --config <estate file> --out <build folder>
  [--file <path>] [--decided-on <day>]` writes `proposal.yaml` from the folder's
  `ontology.json` and `decisions.jsonl` and prints what it wrote and what it did not, each
  with the reason (`estate/proposal.py`). The file's shape is
  `estate/schemas/proposal-file.schema.json`, written from the host's description of what
  its tool reads; no code of the host is in this repository.
- **What is written.** A `spellings` entry for every approved pair of names of a value set
  (the value kept and its other spelling, on a catalog object and column), and a `names`
  entry for every approved master list the estate file names under `propose.names` (each
  record's key with its name). Each carries the element's id, its claim, and the decision:
  approved, by whom, on which day.
- **What is never written.** A rejected or undecided element; a seeded list, a record of one,
  a pair that holds one, and a pair of values of a value set a seeded object carries; a link
  between records; a decision whose claim is not the element's claim now (listed under
  `stale`); a decision without a claim, or without a day.
- **The receipt.** `estate build --receipt <file>` (repeatable, oldest first) reads the
  host's `layer-receipt/1`. An approved element it lists as placed, on the claim the element
  still states, carries `in_layer: {library, version}` in the export. Without a receipt the
  field is absent.
- **Docs.** `estate/README.md` (*The one seam to a host: the proposal file*) and
  `estate/HANDOFF.md` (block 8 and *The proposal file for the host's semantic layer*).
- **Tests: 10** in `estate/tests/test_proposal.py`, on an estate built for them (a dealer
  list, three pairs of maker names, a seeded second source that shares a value set). Each was
  seen failing first: against a first cut that wrote an entry for every decision, and one
  against an earlier rule for the value kept. The three of this ticket are among them.

**The done check.**

| Check | State |
|---|---|
| The vehicle estate's file holds the decided pairs of manufacturer spellings and the dealers' names, and nothing seeded | Holds for the file written in the trial: the Kia pair, the Chrysler pair and 211 dealers. **The decisions were written for the trial**, not made by a reviewer. An approved seeded list and an approved pair of a dealer and a seeded account were in the decisions file and are not in the proposal file |
| Two runs on the same inputs give the same bytes | Holds: twice on the kept export of 11 Nov, and again on a fresh build before and after it applied the decisions, all four byte for byte the same |
| The host's ticket 220 names the commit here | **Open.** It is the host's to write: `24f82e7` |

**Tried against the host's tool** (its final state of 12 Nov, in a throwaway checkout, offline):

| Run | Entries written | Placed | Already there | Refused |
|---|---|---|---|---|
| Listed without applying, on the library before the host's own change | 3 | 3 would be placed | 0 | 0 |
| Listed without applying, on the host's library as it is (1.1.0) | 3 | 0 | 3 | 0 |
| Applied onto the library before the host's own change, released as 1.1.0 | 3 | 3 | 0 | 0 |
| Applied onto the host's library as it is (1.1.0) | 3 | 0 | 3 | 0 |
| The two pairs stated on `clean.complaints` by the estate file, to see a refusal | 3 | 1 (the dealers) | 0 | 2, each by its entry and element: "no dimension of the layer reads clean.complaints.manufacturer" |

- The library that the second run wrote is **byte for byte the host's own release 1.1.0**:
  the same two pairs, the same 211 names, the same origins. The trial's decisions carried the
  same person id and day as the host's hand-written file, so that the two could be compared.
- **Where the file differs from the host's hand-written one,** with no effect on what was
  placed: the pairs are stated on `clean.recall_campaigns`, not `clean.recalls` (the rule
  takes the object that holds the most values of the set, the first by name among equals;
  both are read by the host's one manufacturer dimension); every text is quoted and the file
  is in block style; the spellings come first and the names after them, each in the order of
  their element ids; the receipt names the entries `proposal.yaml#n`.
- The receipt of the second run, given to a fresh build of the vehicle estate, put
  `in_layer` (`estate-semantics`, `1.1.0`) on the two pairs and the dealers. A receipt that
  lists all three as already there put it on none.

**Decided here, where the ticket was open** (each the cautious choice):

- **The day of a decision.** The decisions file carried no time. A line may now carry `at`;
  for lines without one the command takes `--decided-on`. Without either, the entry is not
  written.
- **A decision needs its claim** to be written, also on a concept or a pair, where the build
  itself accepts a line without one.
- **Names need the estate file's word.** A reviewer's approval of a concept does not by
  itself send its rows' names to a host: the list has to be named under `propose.names`.
- **The dealers are approved by a line written by hand.** The host's page offers no decision
  on a stated concept.
- **A value set that a seeded object carries gives no spellings at all,** because the export
  cannot say which object a value was read from.
- **No link between records is written,** not a person's either. The host's names entry keeps
  one origin per record, so a linked record's name decided on another element has no place
  in it.
- **The value kept rests only on the pairs that are written.** Where two approved pairs share
  a name and only one can be written, its value is one of its own two names.
- **`in_layer` comes from `placed` only.** A receipt that lists an element as already there
  does not say since when.
- **`graph-export/3` keeps its number.** `in_layer` is optional, the host's page already
  reads it under version 3, and the page reads one version at a time.
- **The object of a pair** is chosen by a rule, and the estate file may name another
  (`propose.spellings_on`).

**Not done.**

- No reviewer's decision has gone through it: the vehicle estate's `decisions.jsonl` for the
  two pairs and the dealers is the owner's to make (handoff, block 8).
- The trial's file came from the export kept on 11 Nov and from a fresh build **without the
  record linker** (its environment is not on the machine that built this). The machine's
  links were in the kept export only.
- The host's page writes no day into the decisions it downloads. That is the host's to add;
  until then `--decided-on` states it.
- The adapter is a module of the estate prototype, not yet the optional package the decision
  record names.

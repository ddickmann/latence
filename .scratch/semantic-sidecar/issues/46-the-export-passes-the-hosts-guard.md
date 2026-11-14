# Build: the export holds only what the host's guard lets every signed-in person read

Type: task
Mode: AFK, with one decision on the host's side
Status: open: the check is built and an undecided build passes it (14 Nov); a build that applied a decision on two spellings does not, and which side changes is open for the owner
Blocked by: nothing
Emerges from: the host's ticket 238 (a guard on what an ontology export may hold), built on
the host's side on 14 Nov

## Question

The host now refuses to serve an export that holds a row of an object about persons, a row keyed
by a personal column, a value set over a personal column, a row carrying more than its id,
concept, label, key and count, a reference to a row or a concept the export does not hold, or
anything it cannot check (another format, a concept without `instances`, an attribute without
`sensitive`). Does every export this side writes pass, and does every personal column say so?

## Done check

- A test on this side that every personal column carries `sensitive` and that a build passes the
  host's rules.
- The vehicle estate's export passes them.

## Result (14 Nov)

- **The check**, written here from the rules as the host states them (no code of the host is
  copied): `estate/tests/test_host_guard.py`, `what_a_host_refuses`. Two tests on the toy
  estate, which holds an object about persons, a key that is personal and personal columns in
  an object that is not about persons: every personal column the export names says
  `"sensitive": true` and no other does, every concept says what its rows may hold, and the
  build passes every rule; the check is shown to disagree with a row of the persons' object and
  with a field a row may not carry. **Both passed on the tree as it was:** no defect of that
  kind exists, so nothing was fixed and nothing was red.
- **The vehicle estate:** a fresh build of the mirror the host's tools write, with no decision
  applied, passes, by this check and by the host's own (run from a read-only copy of the host's
  commit `4f44ae84`, outside this repository): 0 refusals.
- **It fails once a reviewer's "same value" on two spellings is applied** (the trial's Kia and
  Chrysler pairs, as block 8 of the handoff asks of the owner): 4 refusals, by both checks:
  - the kept value's row carries `aliases` (its other spelling), which the host's page reads
    and shows as "Also written", and which the guard does not allow on a row;
  - the decided pair stays listed, with `merged_into`, and its second end names the spelling's
    row, which the merge took away: the guard reads that as a reference to a row the export
    does not hold.
  The page, the receipt (`in_layer` sits on the pair) and the proposal file (it is written from
  the pair as listed) all rely on both, so dropping either from the export would break them.

## Open for the owner (and the host's ticket 238)

Which side changes so that a decided pair of spellings can be shown?

- **A. The host's guard allows both (recommended):** `aliases` on a row of a value set (the
  names of the same value; the guard already refuses a value set over a personal column), and a
  pair end that names a row the pair's `merged_into` says it became, where the export holds
  that row. Nothing changes here; the two are what the page already reads.
- **B. The export changes:** each spelling keeps its own row, with its own count, and the merge
  is an edge between the two values. A contract change of `graph-export/3` that the page has to
  learn, the receipt and the proposal writer to follow; more work on both sides for no gain in
  what may be read.
- **C. Decided pairs leave the export:** the page no longer shows what was decided, the receipt
  cannot mark it, and the proposal file loses its spellings. Not recommended.

Until it is decided, the demonstration's Ontology page is refused as soon as the owner applies
the two manufacturer pairs and rebuilds (handoff block 8, then block 4 again).

Reproduction (this side): build the vehicle estate as in block 4, append two approved lines on
`same:manufacturer:Kia America, Inc.~Kia Motors America` and
`same:manufacturer:Chrysler (FCA US LLC)~Chrysler (FCA US, LLC)` with their claims to
`out/vehicles/decisions.jsonl`, build again, and hold `out/vehicles/ontology.json` to
`tests.test_host_guard.what_a_host_refuses`.

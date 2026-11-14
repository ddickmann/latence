# Build: the export holds only what the host's guard lets every signed-in person read

Type: task
Mode: AFK, with one decision on the host's side
Status: resolved on this side (14 Nov): the check is built and passes on a decided build under the host's corrected rule (option A, the host's ticket 238); open: one run against the host's guard once its fix is on the integration line
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

## Resolution (14 Nov): the host's guard is corrected (option A)

Decided on the host's side as a correction within the owner's rule, so no question for the owner:
the host's guard allows `aliases` on a row of a value set (not over a sensitive column) and
accepts a pair end whose pair's `merged_into` names a row the export holds. The host makes the
change in its ticket 238. **The export contract stays as it is.** The other two ways (each
spelling a row of its own with the merge as an edge; decided pairs left out of the export) were
not taken.

- This side's check (`what_a_host_refuses`) follows the corrected rule, and a third test,
  `test_a_build_that_applied_a_persons_same_value_passes_too`, builds the yard estate, applies a
  person's "same value" on two maker spellings and holds the export to it: it passes, and the
  check still refuses `aliases` on a row of a concept that is no value set and a pair whose
  `merged_into` names a row the export does not hold. Against the first rule it was red (the
  four refusals above).
- **Still to do:** run the vehicle estate's decided export against the host's own guard once its
  fix is on the integration line; this side's tests stay as the proof until then.

Reproduction (this side): build the vehicle estate as in block 4, append two approved lines on
`same:manufacturer:Kia America, Inc.~Kia Motors America` and
`same:manufacturer:Chrysler (FCA US LLC)~Chrysler (FCA US, LLC)` with their claims to
`out/vehicles/decisions.jsonl`, build again, and hold `out/vehicles/ontology.json` to
`tests.test_host_guard.what_a_host_refuses`.

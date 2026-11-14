# Build: the five-state lifecycle, a reviewer's approval and the data owner's certification

Type: task
Mode: AFK
Status: resolved (14 Nov): built in the estate prototype
Blocked by: nothing
Emerges from: 16 (decided by the owner, 14 Nov: "five states (draft, proposed, approved,
certified, retired); a named reviewer approves and the data owner certifies; nothing is approved
automatically until a calibration proves one class safe, and then only that class")

## Question

An element of the estate's layer is `declared` or `proposed`, and a decision on it `approved` or
`rejected`. The lifecycle the owner decided has five states, two people and one rule for what
may be approved without a person.

## Goal

- Every element carries one of five states: draft (the engine has it, below what is shown),
  proposed, approved (a named reviewer), certified (the data owner, on an approved element
  only), retired (withdrawn, kept for the record).
- The decisions file takes `certified` by a person the estate file names as the data owner,
  and refuses it from anyone else or on an element not approved.
- No element is approved by the engine, unless the estate file names a class that a recorded
  calibration proves safe; then only that class, and the approval names the calibration.

## Done check

Tests: each transition, each refusal, and an automatic approval allowed for a calibrated class
only.

## Result (14 Nov)

- `estate/lifecycle.py`, applied by every build; the export carries `state` on every element a
  person can decide (`proposed`, `approved`, `certified`, `retired`; a draft, an element below
  the support floor, is not shown and is counted), and the build's summary carries `lifecycle`:
  the count per state, the retired elements, the refused certifications with the reason, and
  the engine's approvals by kind.
- **Approval** stays the reviewer's line in `decisions.jsonl` (`by` names the reviewer).
  **Certification** is the data owner's line in `certifications.jsonl` (`subject`, `by`, `claim`,
  `at`), written by `python -m estate certify --out … --subject … --by … --at …` on the statement
  the build makes now. It stands only from the person `lifecycle.data_owner` names, on an
  element whose approval is in force, on the same statement; otherwise the build lists it as
  `not_the_data_owner`, `not_approved` or `stale`. A certified element carries
  `certification: {by, at}`.
- **Retired:** an element a reviewer rejected leaves the layer (as before) and is counted and
  listed as retired; nothing is deleted from the ledger.
- **The engine approves nothing** unless `lifecycle.auto_approve` names a kind (concept,
  relation or measure) with a calibration record whose `precision_lower_bound` reaches its
  `target` on stated labels; then only the proposed elements of that kind that no person
  decided, each approval saying `by: engine` and the calibration. A person's decision outranks
  it, a calibration below its target proves nothing and is listed, and an engine's approval
  never reaches a proposal file (that is written from people's decisions only).
- **Tests, red first** (`estate/tests/test_lifecycle.py`, 3; red: no `state`): approval,
  certification and each refusal; the engine's approval of one calibrated kind only, outranked
  by a person, and refused under a calibration below its target; the `certify` command.
- **Not done here:** the host's Ontology page shows no state yet (the host's to read, an
  optional field of `graph-export/3`); recording a calibration from the label loop's own
  numbers is a step of its own (today the record is written by whoever ran the calibration).

# Build: the five-state lifecycle, a reviewer's approval and the data owner's certification

Type: task
Mode: AFK
Status: open
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

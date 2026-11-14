# Build: a hosted model may decide pairs in the uncertain band, under a stated egress rule

Type: task
Mode: AFK (the build); HITL (a live run: spend and egress)
Status: open
Blocked by: nothing
Emerges from: 13 (decided by the owner, 14 Nov: "a hosted model may decide pairs in the
uncertain band only under the deployment's stated egress rule, off by default, each call
recorded")

## Question

The record linker leaves the pairs between its reject line and its accept line for a person.
The owner allows a hosted model to decide such pairs, but only where the deployment states an
egress rule that lets the pair's fields leave, off unless switched on, and with each call
recorded.

## Goal

- A switch in the estate file, off by default, and an egress rule beside it: which fields of a
  record may be sent, to which provider and model, with a spending cap.
- Without both, no call is made and the build says why.
- Each call recorded (what was sent, which model, the answer, the cost), and the decision on
  the pair recorded with the model as its actor; the model's verdict accepts nothing by itself
  (ADR-0069): it is a vote a person or a calibrated line confirms.

## Done check

- Tests with a stand-in client: off by default; refused without an egress rule; only the
  allowed fields leave; the cap stops the calls; each call is in the record.
- A live run on synthetic data is a live request.

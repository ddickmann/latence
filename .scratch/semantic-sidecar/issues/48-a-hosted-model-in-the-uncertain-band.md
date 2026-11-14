# Build: a hosted model may decide pairs in the uncertain band, under a stated egress rule

Type: task
Mode: AFK (the build); HITL (a live run: spend and egress)
Status: open: built and tested with a stand-in client (14 Nov); a live run is a live request
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

## Result (14 Nov)

- `estate/pair_voice.py`, called by the build after the linker's answer is merged. **Off
  unless** the estate file says `link.hosted_model.enabled: true` with an egress rule:
  `fields` (of `name`, `city`, `country`; a key, an id or any other field is refused by name),
  `model`, `prices` and `max_usd`. Switched on without a rule, or without prices or a cap, the
  build stops and says what is missing; nothing is guessed. The vehicle estate leaves it off.
- **What leaves:** for each pair of records left for review, the named fields of the two
  records and nothing else, without their ids. A pair of names of a value set is not the
  linker's and is not asked.
- **The cap:** each call's worst case (every character sent counted as a token, the whole output
  allowance used) is added to what the calls made could have cost; the first call that would
  pass `max_usd` is not made, nor any after it, and the build says how many were not asked.
- **The record:** `pair_model_calls.jsonl` in the build folder, one line per call: the pair,
  what was sent, the model that served it, the answer, the worst case and the cost at the
  listed prices. The build's summary and the export's `meta.resolution.pair_model` count the
  calls and the cost.
- **It decides nothing:** the verdict (same, different, unsure) is a `model_vote` on the review
  item with the model as its actor (`model:<served model>`); the pair stays for a person, and no
  link is made from it. Using the vote (a calibrated line on it, or a person who confirms) is a
  decision of its own, not made here.
- **Tests, red first** (`estate/tests/test_pair_voice.py`, 4; red: no module): off by default and
  refused without a rule; only the named fields leave, each call is recorded, the vote carries its
  actor; the cap; and a whole build of the yard estate with a stand-in client, whose export
  validates and carries the vote and no model link.

**Live request** (synthetic data only, inside the USD 25 route limit; the owner's machine or the
gate queue): in a copy of `estates/vehicles.yaml`, set under `link`:
`hosted_model: {enabled: true, model: gpt-6-luna, fields: [name, city], prices: {input_per_1m: <list price>, output_per_1m: <list price>}, max_usd: 0.50}`,
then run block 4 of `estate/HANDOFF.md` with `--config` set to that copy and with
`OPENAI_API_KEY` in the shell. Report `pair_model` from the build's summary and the first three
lines of `out/vehicles/pair_model_calls.jsonl`.

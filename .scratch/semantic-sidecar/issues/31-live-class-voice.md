# Build: the model's voice on classes runs live on a small hosted model

Type: task
Mode: HITL
Status: resolved
Blocked by: 26

## Question

The class tree's second voice has never spoken through an API: a separate model session
answered the prompt file and the answer was committed. The direct call that exists was tested
against a stand-in only and rests on request parameters nobody has seen accepted. v1 should
carry no code path that never ran.

**Owner, 2026-11-11:** an OpenAI key is available; use a small model, only where the builder
needs one; USD 25 for all calls from this work, of which this ticket may use USD 5.

## Goal

- `OpenAIClassModel` behind the existing `ClassModel` protocol, on the vendor's official SDK
  and its documented structured-output call, read from the current documentation before any
  code. Model: the small model the host already configures.
- The never-run voice is removed. `RecordedClassModel` stays, for builds that must repeat.
- A call is made only when needed: the prompt is written to `classes_prompt.json` first; the
  answer is cached under a digest of prompt and model, and a build with an unchanged prompt
  makes no call. `--max-usd` stops the build before a call that could pass it; prices are in
  the estate file with their date.
- Cost and token counts go to `model_calls.jsonl` and to `meta.classes.cost`.
- A definition written by the model is shown as "written by a model, not reviewed" until the
  placement is approved. Definitions are asked for in English and German.
- The key comes from the environment only. Nothing but names, types, comments, keys and
  relations leaves; no row, no value.

## Done check

One live run on the synthetic estate: the answer is recorded, compared with the committed one
(placements that differ are listed), and the cost is written here.

## Tests that prove it

- A second build with the same prompt makes no call (the stand-in client fails if called).
- An estate that does not allow a hosted model refuses, with or without a key.
- The request the stand-in receives holds no value of any column of the toy estate and asks
  for a strict schema.
- An answer that names a parent outside the offered classes is refused, as today.

## Needs the owner

The key, through the secure form, when this ticket starts.

## Answer

Built and run live on 2026-11-11.

- **`OpenAIClassModel`** (`estate/classes.py`) on the official SDK, version 3.49.0, over the
  Responses API: the instruction, the top classes and the concepts go in; a strict JSON
  schema names the form of the answer; definitions are asked for in English and German.
  The answer is read as a stream and the SDK's automatic retries are off (see below).
- **`CachedClassModel`** asks only when what would be sent has changed: the answer is kept
  under a digest of the whole request. Before a call that has to be made, its worst case
  (every character sent counted as a token, the whole output allowance used) is held against
  `--max-usd`; a call that could cost more is not made. Every call is appended to
  `model_calls.jsonl` with its tokens and cost, and the export carries the cost of the answer
  it holds (`classes.voices.cost`), the same on a rebuild that asks nothing.
- **The never-run voice is removed.** `RecordedClassModel` stays; the live answer is also kept
  as a recorded one (`estates/vehicles.classes_answer.gpt-6-luna.json`), so a build without a
  key gives the same class tree.
- **The estate file** names the model, the effort, the languages and the list prices with the
  day they were read.

**Where the request shape and the prices come from.** The provider's documentation and price
page could not be opened from where this was built (the hosts are not reachable there). The
request shape was read from the SDK's own types, and a stand-in client in the tests refuses
what those types refuse; the live run confirms it. The prices (USD 0.10 per million input
tokens, USD 0.50 per million output tokens for `gpt-6-luna`) are from third-party listings
read on 2026-11-11, and the estate file says so. **Not verified on the provider's page.**

**The live run, on the synthetic vehicle estate:**

- One call: 15,033 input tokens, 4,034 output tokens, USD 0.0035 at list price. The rebuild
  made no call and gave the same file byte for byte.
- A first attempt failed: without streaming the gateway answered 502 after about 30 seconds,
  and the SDK repeated the call twice. Those three attempts may have been billed (their
  prompt was served from the provider's cache on the fourth); at worst USD 0.011 together.
  **Spent on this ticket, at worst: USD 0.015 of the USD 5 it may use.**
- What left: the request, which holds names, types, comments, keys and relations of 39
  concepts and no row or value (`classes_prompt.json` in the build folder; a test holds that
  no value of the toy estate is in it).
- **Against the recorded answer:** the live model placed 26 concepts (recorded: 28), 5
  contested (6), and proposed one class of its own, "vehicle safety communication" under
  Document (recorded: six). The top class differs for two concepts: it left the two
  component value sets unplaced, which the recorded answer put under Product.
- **Against the draft list (not counting, ticket 29):** classes precision 0.88 and recall 0.92
  (recorded answer: 0.89 and 1.00).

**Tests: 99 pass (5 new, 1 replaced):** the voice sends names only, in a strict form, and
reads the answer as data, with definitions in two languages; an answer that was cut off or
refused proposes nothing; the model is asked once for the same request and every call is
logged with its cost; a call that could cost more than allowed is not made; a build asks the
hosted model once, only where the estate allows it and states prices, and a second build
makes no call and gives the same file.

**Host page:** a definition a model wrote names the model and says that no person has reviewed
it, until the placement is approved; the German page reads the German definition. 201 unit and
route tests, the browser walk in both languages.

## From the map (moved 12 Nov)

Route table row: the model's voice on classes runs live on a small hosted model | HITL | **done**

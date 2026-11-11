# Build: the model's voice on classes runs live on a small hosted model

Type: task
Mode: HITL
Status: open
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

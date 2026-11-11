# Build: the export says what it means in codes, and every proposal carries its claim

Type: task
Mode: AFK
Status: open
Blocked by:

## Question

The export (`graph-export/2`) carries English sentences the engine wrote from templates, so a
host cannot show them in another language. And a decision is keyed on an element's id alone:
`subclass:<concept>` keeps its id when the proposed parent changes, so an approval of "X under
A" would carry over to "X under B". Both are contract defects, and every later ticket of this
route builds on the contract, so this one comes first.

## Goal

`graph-export/3`:

- Every signal, evidence line and review reason is `{kind, code, params, ref}`. The engine
  writes no English sentence into the export. `estate/codes.py` lists each code with its
  parameter names and renders English for the Turtle file and the command line.
- Every element a person can decide carries `claim`: a short digest of what it states (kind,
  both ends, parent, columns), not of its support counts. The same claim on a rebuild gives the
  same digest; a changed claim gives another.
- Free text that is not ours (a catalog comment, a model's definition) is
  `{text, lang, origin}` with origin `catalog` or `model:<name>`.
- The matcher's evidence for a pair (`spike/link_records.py`, `evidence_of`) is codes too.

The host's page reads schema 3 and renders `ontology.code.<code>` from its own messages in
English and German; an unknown code is shown raw, never hidden.

## Done check

- A build on the vehicle estate validates against `schemas/graph-export-3.schema.json`.
- `schemas/codes.json` is generated from `codes.py` and committed; the host keeps a copy.
- The page in German shows no sentence the engine wrote.

## Tests that prove it

- The export of the toy estate validates against schema 3 and holds no sentence field (a walk
  over the whole file fails on any `detail` or `reason` string outside `{text, lang, origin}`).
- Every code the build can emit is in `codes.py` with exactly the parameters it is emitted with.
- The claim of an element is unchanged by a change in support counts and changes when an end
  or the parent changes.
- Host: a unit test fails when a code of `codes.json` has no English or no German message, or
  a message uses a parameter the code does not carry.

## Not tested, and why

The wording of each message. It is read by a person in the browser walk, in both languages.

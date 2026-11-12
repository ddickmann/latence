# Build: the export says what it means in codes, and every proposal carries its claim

Type: task
Mode: AFK
Status: resolved
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

## Answer

Built on 2026-11-11.

- **Engine.** `estate/codes.py` holds 59 codes with their parameters and English; `say()`
  refuses an unknown code or a wrong parameter where the statement is made. Signals, reasons
  on review items (`reasons`, a list) and the matcher's evidence are statements; the matcher
  writes `resolution/2` and the build refuses another format or a code the table does not
  hold. Definitions are lists of texts with language and origin; the six top classes carry the
  code of their label and definition. `claim` is on concepts, relations, measures, the four
  kinds of structure, proposed classes, placements and review pairs (`ledger.claim_of`).
- **Contract.** `schemas/graph-export-3.schema.json`; `schemas/codes.json` is generated and a
  test fails when it is stale. The Turtle file writes definitions with their language tag.
- **Vehicle estate.** The rebuild validates against schema 3: 4,096 statements over 41 codes,
  the same 39 concepts, 86 relations, 38 measures, 2,514 nodes and 3,857 triples as before.
- **Host.** The page reads schema 3 and says every code from its own English and German
  messages. Its tests: 196 unit and route tests (6 new), and the browser walk in both
  languages with the accessibility checks.
- **Tests here:** 58 pass (7 new): no sentence in the export; every statement in the export is
  in the table with its parameters; a wrong parameter fails where it is made; the committed
  table is the code's; a claim follows what is stated and not what was counted; every
  decidable element carries one; an answer of the matcher in another format or with an unknown
  code is refused.

**Left to later tickets, on purpose:**

- the page does not yet write the claim into `decisions.jsonl`, and the ledger does not read
  it (ticket 28);
- the labels of the six top classes are shown in English on the German page; their codes are
  in the file (`label_code`), the page's tree does not use them yet (ticket 28, page work);
- a model's definition exists in English only (ticket 31).

## From the map (moved 12 Nov)

Route table row: the export speaks in codes, every proposal carries its claim (`graph-export/3`) | AFK | **done**

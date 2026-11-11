# Build: a reviewer's decision on a hierarchy step or a class changes the layer

Type: task
Mode: AFK
Status: open
Blocked by: 26

## Question

The ledger covers concepts, relations and measures. Roll-ups, equivalences, part-of, describes,
proposed classes and class placements are shown with approve and reject, the decisions are
downloaded, and the next build does not read them. Done check 1 (every element links to an
assertion and an actor) and done check 2 (curation survives a rebuild) do not hold for them.
The Turtle file also states part-of and `skos:broader` as plain triples before anyone has
approved them.

## Goal

- `ledger.assertions_of` covers all nine kinds of element. A decision line carries the `claim`
  it was made on (ticket 26). `replay` applies a decision only when the claim still matches;
  otherwise the decision is reported under `meta.decisions.stale` and changes nothing.
- A rejection is applied where the structure is built, before the steps are reduced to direct
  ones:
  - a rejected step a→b lets an implied step a→c come back;
  - a rejected equivalence gives two levels again;
  - a rejected step writes no `broader` edges, and review reasons that leaned on it go.
- Classes: a rejected placement falls to the other voice's parent if there is one, else to
  unplaced with a code; a rejected proposed class hands its members to its parent.
- Turtle: an approved statement is plain; a proposed part-of and a proposed `broader` are
  written as marked axioms only.
- The page writes the claim into `decisions.jsonl` and lists stale decisions.

## Done check

A set of decisions on every kind, made on the page over the vehicle estate, is intact after a
rebuild; the export and the Turtle file differ from the undecided build exactly as decided.

## Tests that prove it

- On a fixture with a→b→c: rejecting a→b brings a→c back and removes the `broader` edges of
  a→b. Red before.
- An approval of "X under A" is not applied when the proposal now says "X under B"; it is
  listed as stale. Red before.
- A decision on each kind survives a second build with the ledger file kept.
- The Turtle file parsed back: an approved roll-up is a plain triple, a proposed one appears
  only inside an annotated axiom. Red before.
- A decision on a subject the model does not hold is reported, as today.

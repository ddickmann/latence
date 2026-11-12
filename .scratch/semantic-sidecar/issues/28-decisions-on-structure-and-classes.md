# Build: a reviewer's decision on a hierarchy step or a class changes the layer

Type: task
Mode: AFK
Status: resolved
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

## Answer

Built on 2026-11-11.

- **The ledger.** `assertions_of` writes one assertion per element of all nine kinds.
  `ledger.Decisions` answers for one element at a time while the layer is built and remembers
  what it answered, so the build can say at the end which decisions were applied, which are
  stale, which carry no claim and which name nothing.
- **Where a claim is needed.** A decision on a concept, relation, measure, roll-up,
  equivalence, describes or a pair may come without a claim: the id of those says all the claim
  says. A decision on a placement, a proposed class or a part-of without a claim is not applied
  and is listed under `meta.decisions.without_claim`.
- **Structure.** A rejected step is taken out before the steps are reduced to the direct ones,
  so the step it hid comes back; a rejected equivalence leaves two levels; no `broader` edge is
  written for a rejected step.
- **Classes.** A rejected class of the model's is gone and its members sit under its parent. A
  rejected placement falls to the class the other voice named if that is another one, else the
  concept is unplaced. **An approval of the class either voice named settles the placement**
  (not in the ticket, and needed: otherwise a person who agrees with the columns against the
  model could not say so).
- **Turtle.** A proposed part-of and a proposed step with all its `skos:broader` are written
  only as marked axioms; once a person approves the step, they are plain triples.

**On the vehicle estate:** nine decisions, one of each kind, written with their claims. Two
rebuilds on top of them give byte-identical export and Turtle files; the summary reads 3
approved, 6 rejected, none stale, none unknown. What changed is what was decided:

| Decision | In the next build |
|---|---|
| approve manufacturer → make | the step carries the decision; its 18 `broader` are plain triples, the other 283 stay marked |
| reject dealer group → market, and reject "market and distance unit are one level" | both gone. A new step appears, dealer group → distance unit: with the two levels apart the data supports it. A rejection can bring a new proposal for a person to decide |
| reject recall report "is part of" recall campaign | gone |
| reject the model's class "vehicle" | gone; its four members sit under Product |
| approve a placement | carries the decision, and is a plain `rdfs:subClassOf` |
| reject a contested placement (bulletin under the model's class) | the bulletin sits under Event, which the columns said; no longer contested |
| approve a concept, reject a relation | as before this ticket |

The ledger of that build holds 221 assertions over the nine kinds.

**Tests: 88 pass (8 new, each red first):** a rejected step leaves and the step it hid comes
back, with its edges; an approved step says so and a rejected equivalence gives two levels; a
decision is not applied to a proposal that now says something else, and is reported; an
approval of what the other voice said settles the placement; a decision on a placement needs
its claim; a rejected placement falls to the other voice and a rejected class hands on its
members; decisions on structure and classes survive two rebuilds and every element is an
assertion; the Turtle file states a step plainly only once a person approved it.

**Host page:** each downloaded decision carries the claim; a decision kept in the browser
whose claim no longer matches is not shown as applied, and the page names those and the ones
the build reported; the six top classes are said in the page's language. 198 unit and route
tests, the browser walk in both languages.

**Not done:** a value that breaks a step cannot yet be given its place by a decision; a
decision on a pair of names does not merge the two values (ticket 30).

## From the map (moved 12 Nov)

Route table row: a decision on a hierarchy step or a class changes the layer | AFK | **done**

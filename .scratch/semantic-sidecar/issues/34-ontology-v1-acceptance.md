# Decision: is the ontology capability at v1 quality?

Type: decision
Mode: HITL
Status: open
Blocked by: 26, 27, 28, 29, 30, 31, 32, 33

## Question

The owner wants the whole demonstration set up on the local backend, "but only once the
ontology capability is at v1 quality" (2026-11-11). This ticket is that gate: each line below
is checked, the evidence is written beside it, and the owner sees the page and decides.

| Check | State | Evidence (2026-11-11) |
|---|---|---|
| Decisions on every kind of element survive a rebuild and change export and Turtle; a decision whose claim changed is reported and not applied | **met** | Ticket 28: nine decisions, one per kind, on the vehicle estate; two rebuilds byte-identical; 3 approved, 6 rejected, none stale. Tests hold the stale case and each kind |
| The export validates as `graph-export/3`; the German page shows no untranslated engine text | **met** | Tickets 26, 27 and 32: 63 codes, each with an English and a German message with the same parameters (a test fails otherwise). The German browser walk passes; read by eye on its screenshots: what is left in English is catalog names, source names and a model's English-only definition in the test file |
| The six named defects are gone, each with its test | **met** | Ticket 27. Two causes were other than assumed and one more defect was found on the way; the ticket says which |
| Classes, hierarchies and part-of are scored against the list the owner corrected | **not met: waits for the owner** | Ticket 29: the scorer and a draft list exist. Against the draft, which does not count: classes 0.88 precision and 0.92 recall (live model), steps 1.00 and 0.89, part-of 1.00 and 0.33, links 1.00 and 1.00 |
| The label loop runs end to end on the seeded source, and the page says how many labels came from a person | **met** | Ticket 30: 64 answers written as a reviewer would give them change the next build (60 training and 2 test labels a person's, `certified_on: mixed`, 31 links a person's, one machine link taken away) |
| One live class call made, its cost recorded, no call on the cached rebuild | **met** | Ticket 31: one call to the small model, 15,033 tokens in and 4,034 out, USD 0.0035 at list price; the rebuild made no call and gave the same file. At worst USD 0.015 spent, counting three failed attempts |
| Usage signals sit on existing elements only, their source marked | **met** | Ticket 32: 408 golden questions, 358 answered; counts on measures, value sets, entities; nine things asked for and not in the model are listed; a test holds that no status changes. All of it synthetic, and marked so |
| The reader gives the same model through the stand-in; the run on the real catalog is listed as the owner's step | **met here; the owner's run is outstanding** | Ticket 33: the same model through the catalog reader as through the mirror, against a stand-in that refuses, pages, stalls and fails. The commands for the real catalog are in the ticket |
| Two builds of the same inputs give byte-identical files | **met** | Shown four times on the vehicle estate: on nine decisions, on the reviewer's answers, on the kept model answer, and before and after the dialect seam |

**Where it stands:** seven of nine checks are met. One waits for the owner's correction of
the hand-checked list; one is met as far as it can be here and waits for the owner's run
against the real catalog.

**Tests at this point:** estate 114, matcher 155, host page 202 unit and route tests, host
exporter 4, and the browser walk of the page in English and German with accessibility checks
on every screen.

**For the owner to look at** (the build of the vehicle estate that the page shows):

- the class tree with the live model's definitions in both languages;
- the hierarchies with the value that breaks manufacturer → make named, and "Subject" listed
  as no level;
- "Possibly the same": the summary of whose labels the line rests on, the pairs to label, and
  a pair opened without the machine's score;
- the measures view: what recorded questions used, and what they asked for that is not in the
  model;
- a decision made, downloaded and applied by the next build.

## What v1 does not hold, on purpose

Mentions from the estate's documents; classes from an outside vocabulary; SHACL and PROV-O
exports; a service route in the host; real packages in place of the prototype folder.

## Answer

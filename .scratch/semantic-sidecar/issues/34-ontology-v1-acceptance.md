# Decision: is the ontology capability at v1 quality?

Type: decision
Mode: HITL
Status: open
Decided by the owner (12 Nov): Is the ontology capability at v1; Yes once the list's score is
recorded; the run on the real catalog moves to the switch
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

**Tests at this point:** estate 114, matcher 155, host page 209 unit and route tests, host
exporter 4, and the browser walk of the page in English and German with accessibility checks
on every screen.

**After the owner's first look (2026-11-11):** "the graph visualization is missing in the
screen. Other than that they look fine." The page had a map of the concepts and a tree of the
hierarchies and no graph of the entities. Added the same evening: the entity view draws the
graph around one entity (what points at it, what it points at, by kind of link), and choosing
a box walks on from there. The owner also said the demonstration setup may start.

**The owner's run** is `estate/HANDOFF.md`, seven blocks to paste in order. Blocks 2 and 4 were
run from nothing on the machine that built this and gave the kept export and Turtle file byte
for byte; the matcher's environment is pinned in `spike/requirements-link.txt` and was checked
in a clean install.

**For the owner to look at** (the build of the vehicle estate that the page shows):

- the class tree with the live model's definitions in both languages;
- the entities: the graph around a dealer, and a walk from it to a part or a linked account;
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

## From the map (moved 12 Nov)

Route table row: **is the capability at v1 quality?** | HITL | seven of nine checks met; **the owner's decision is open**

Route note: **What this route needs from the owner:** the hand-checked list corrected (29); the key through the secure form (31); one build against the real catalog (33); the look at v1 (34); the runner started (38).

## Decision brief for the owner (12 Nov)

- **Question:** Do you declare ontology v1 reached?
- **Why his:** the ticket is his gate ("the owner sees the page and decides").
- **Options:**
  - A. Yes now, with 29 and 33 as dated conditions.
  - B. Yes once 29's score is recorded, with 33 moved to the switch.
  - C. Wait for both.
- **Facts:**
  - Seven of nine checks are met; the two open ones are your own steps (34, table).
  - On 11 Nov you said "they look fine" apart from the graph (added that evening) and that the
    demonstration setup may start.
  - The Answer section is empty.
  - Total spend so far is at most USD 0.015 of the USD 25 limit.
- **Recommendation:** B. The gate's purpose (unblocking the setup) is already served, and the
  one check that measures quality should not be waived.
- **Urgency:** before the demonstration. Blocks ticket 42 (the order of what v1 leaves out).
- **Cost:** a signature after the hand-checked list is corrected.
- **Decided by the owner (12 Nov):** Yes once the list's score is recorded; the run on the real
  catalog moves to the switch.

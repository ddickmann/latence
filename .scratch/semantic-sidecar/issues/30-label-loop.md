# Build: the matcher asks for the labels it needs and takes them from a reviewer

Type: task
Mode: AFK
Status: resolved
Blocked by: 26, 28

## Question

The linked records rest on 1,300 labels taken from the seed's truth. The matcher never asks a
person, and an answer given on the page reaches no build. The certified line is honest only if
its test labels are drawn before anyone looks at them, and the two test samples can only be
drawn once a scorer is fitted, so labelling has to run in two rounds.

## Goal

- `run_a.run` takes `labels`: pair → same, different or open. The truth column is used for the
  realised figures only.
- `link_records.py --labels FILE --requests-out FILE` writes `label-requests/1`: the digest of
  the records, the seed, the round, and per sample (`training`, `accept_test`, `reject_test`)
  the pairs as `{id: "pair:<a>~<b>", a, b}` with counts.
  1. Round 1 asks for the training pairs (`draw_training`, fixed by the digest and the seed).
  2. When they are all answered the scorer is fitted, the two test samples are drawn and
     frozen in `label_samples.json`. If a training label changes later, they are redrawn and
     the summary says so.
  3. A line is certified only when every test label exists.
  Test pairs carry no score, so the reviewer is not led.
- Labels come from the same `decisions.jsonl` as every other decision (subject `pair:…`;
  approved means the same, rejected means different). A reviewer's label wins. The seed's
  truth fills the rest only where the estate file allows it, and the export says per sample how
  many labels came from a person and how many were simulated (`certified_on`: reviewer, mixed
  or simulated). With open labels and no truth, no `same_as` edge is written.
- An answer on a "possibly the same" pair decides that pair and becomes a training label of
  the next build. It is never a test label: it was picked by its score.
- The page: a queue "Pairs to label" per sample with a progress count.

## Done check

On the seeded dealer source: a reviewer's answers given on the page change the next build, and
the page states how many labels of each sample came from a person.

## Tests that prove it

- Reviewer labels that contradict the truth change the fitted scorer and the accepted pairs.
- With open labels and no truth, nothing is accepted and the state says which sample is waited
  for.
- The same training labels give the same test samples, byte for byte.
- A review-band answer is used for training and never appears in a test sample.

## Limit to state on the page

Nobody labels 1,300 pairs in a demonstration. Most labels will read "simulated"; the loop is
shown on the pairs a person does answer.

## Answer

Built on 2026-11-11.

- **The matcher** (`spike/run_a.py`, `learned.py`, `link_records.py`). `run` takes a
  reviewer's `labels` and whether the truth column may stand in for the rest. Certification is
  split into drawing the two test samples from the scores (`draw_tests`, no label read) and
  testing the lines on them (`certify_drawn`). The answer (`resolution/3`) carries `state`
  (`awaiting_training`, `awaiting_tests`, `certified`, `not_certified`), per sample how many
  labels a person gave, how many were simulated and how many are open, `certified_on`
  (`reviewer`, `mixed`, `simulated`), and `requests`: every pair of the three samples with
  what it rests on, its label and who gave it, and no score.
- **The build.** A reviewer's answers on pairs of records are read from the decisions file
  (`pair_answers`) and handed to the matcher. Whether the seed's truth may stand in is the
  estate's to say (`link.labels.truth_stands_in`); without a truth nothing stands in.
- **An answer on a pair decides it,** whatever the matcher made of it: "the same" links the
  record to that master record and to no other, "different" takes a machine link away. Such
  an edge carries `decided_by`.
- **Names a reviewer calls the same become one value** (`instances.merge_values`): the name
  with more rows stays, the other is its alias, and what pointed at either points at it.

**Changed from the ticket, with the reason:**

- No file freezes the test samples. They follow from the training labels and the seed alone,
  so the same answers give the same samples (a test holds this), and a second file would be a
  second truth to keep in step.
- An answer on a "possibly the same" pair is not added to the pairs the scorer learns from.
  It would change the scorer, the scorer's scores draw the test samples, and a reviewer's
  test labels given so far would be asked for again. The answer decides its pair; the scorer
  learns from the pairs it asked for.

**On the vehicle estate:** the build asks for 1,300 pairs (400 to learn from, 600 and 300 to
test). With no answer and the seed's truth standing in, the result is the one before this
ticket: 938 of 1,274 accounts linked, `certified_on: simulated`. With 64 answers written as a
reviewer would give them (60 of the pairs to learn from, 3 pairs from the review list, and one
wrong machine link rejected): 60 training labels and 2 test labels are a person's,
`certified_on: mixed`, 939 linked of which 918 right (97.8%), 31 links are a person's and one
machine link was taken away. Two rebuilds on those answers give the same file.

**Tests.** Matcher, 4 new (155 in all): with no label and no truth to stand in, nothing is
accepted and the pairs to learn from are asked for, without a score; the tests are drawn once
those are answered, never from the pairs learned from, the same answers give the same
samples, and a line is certified once they are answered; the summary says how many labels a
person gave; a reviewer who says otherwise changes what the scorer learns. Build, 4 new (95
in all): the answer becomes links, review items, label requests and a summary; a reviewer's
answer decides a pair whatever the engine made of it; answers on pairs of records are read
from the decisions and pairs of names are not; two names a reviewer calls the same become one
value.

**Host page:** a queue "Pairs to label" per sample with how many a person has given, pairs
without a person's answer first; a pair opens without the machine's score; the summary says
whose labels a line rests on. 200 unit and route tests, the browser walk in both languages.

**Limits.** Nobody labels 1,300 pairs in a demonstration: the page shows most labels as
simulated and the loop on the pairs a person does answer. The label requests add about 0.7 MB
to the export. Merging two names does not recount the support of the roll-ups above them.

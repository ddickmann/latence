# Build: the matcher asks for the labels it needs and takes them from a reviewer

Type: task
Mode: AFK
Status: open
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

# Build: what people ask is a signal on the elements they use

Type: task
Mode: AFK
Status: open
Blocked by: 26

## Question

Cut C of the estate slice: read the questions people ask and the queries that answered them,
and show them as evidence. Two cautions from reading the host's files:

- the estate's golden questions were written from the same metric views the model is induced
  from, so they mostly confirm joins that are already stated;
- a term is evidence for a synonym only where a record states the mapping (a question that was
  asked back and answered). Guessing a mapping from the words would be our invention.

So usage is a signal that is shown with its source. It never changes a status.

## Goal

- Format `usage/1`, one JSON object per question: `source` (golden, walkthrough, turns),
  `synthetic`, `id`, `language`, `answered`, `measures` (`view.measure`), `joins`
  (`{from, columns, to}`), `filters` and `group_by` (`{object, column}`, no values), `terms`
  (`{text, element, how}`). The schema file is committed.
- Engine: `estate/usage.py` and `build --usage FILE` (repeatable). It matches by id and adds
  the signal `used_in_answers {n, sources}` to relations, measures and value sets that exist.
  A stated term becomes a proposed synonym of its concept. Records that match nothing are
  counted in `meta.usage.unmatched` and listed.
- Host: `metis-dev usage export` writes the file from the golden questions, the walkthrough
  list and the recorded turns of a local world's store, read-only. No value of a filter leaves.
- The page shows usage on an element's detail and a short list "asked for, not in the model".

## Done check

The vehicle estate's export carries usage from its golden questions, marked synthetic, and the
page shows it.

## Tests that prove it

- A planted log gives the planted counts per relation, measure and value set.
- A join the model does not hold attaches to nothing and is counted as unmatched.
- A stated term becomes a proposed synonym; the same words without a stated mapping do not.
- No status of any element differs between a build with and without `--usage`.
- Host: one known golden question exports its measure and its dimensions.

## Cut first if time runs short

Terms and synonyms. The counts stay.

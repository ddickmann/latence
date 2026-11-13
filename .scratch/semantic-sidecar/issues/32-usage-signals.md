# Build: what people ask is a signal on the elements they use

Type: task
Mode: AFK
Status: resolved
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

## Answer

Built on 2026-11-11.

- **The format `usage/1`** (one JSON object per question): `source`, `synthetic`, `id`,
  `language`, `answered`, `measures` (`{object, name}`), `dimensions`
  (`{name, columns: [object.column]}`), `joins` (`{from, columns, to}`) and `terms`
  (`{text, element, how}`). A record that carries a value of a dimension is refused.
- **The engine** (`estate/usage.py`, `build --usage FILE`, repeatable) counts answered
  questions on elements the model already holds and adds the signal `used_in_answers`:
  - a measure by its object and name;
  - a dimension on the value set of its column; else on the entity whose key it is; else, where
    it is a column of one object that the concept holds, on that concept;
  - a join on the relation with the same ends and columns;
  - a stated term as a proposed synonym of its concept, unless it is a synonym already.
  What matches nothing is counted and listed in `meta.usage.unmatched`. Nothing is created
  from use, and a test holds that no status differs with and without a log.
- **The host** writes the log with `metis-dev usage export` from the estate's golden questions:
  each question's measures and dimensions are read off the query form of its answer and
  resolved through the estate's semantic notation. No value of a filter is exported.

**Narrower than the ticket, with the reason:**

- Only the golden questions are exported. The walkthrough list holds each question's words
  and the figures a right answer shows, and no query form, so nothing can be read off it
  without guessing. Recorded turns of a running world are not read (the cut the plan named
  first).
- The exporter writes no joins and no terms: a query form names no join (the host's compiler
  derives it), and a golden question states no meaning of a term. Both are in the format and
  in the engine, with tests on planted records.

**On the vehicle estate:** 408 golden questions in two languages, 358 answered. 548 uses of a
measure and 604 of a dimension matched an element. The most asked for: the number of warranty
claims (112 answered questions) and of complaints (106); make (174), model (126), recall
campaign (78), market and component system (70 each). 5 of the 38 measures were never asked
for. **Asked for and not in the model:** the company's own component group (12 questions),
the internal campaign (10), model year and operation code (2 each), and five measures on
repeat repairs and claimed vehicles (30 questions together). Every one of these questions is
synthetic and written from the same metric views the model is induced from: the counts say
what the host's test suite covers, not what people want.

**Tests.** Engine, 5 new (104 in all): counts land on the measure, the value set and the
relation a question used, and an unanswered question counts for nothing; what is not in the
model is counted and attached to nothing; use changes no status and a term is a proposal only
where the record states what it means; a record in another format or with a value in it is
refused; a dimension that is an entity's key, or a column of one object, counts on that
concept. Host, 4 new: a question exports its measure and dimensions by name and none of its
values; a question that cannot be answered is recorded as asked and not answered; a dimension
a stage groups by counts like one it filters on; the command writes one line per question.

**Host page:** the measures view says how many recorded questions were read and lists what was
asked for and is not in the model; "Asked for" is one more signal on an element. 202 unit and
route tests, the browser walk in both languages.

## From the map (moved 12 Nov)

Route table row: what people ask is a signal on the elements they use | AFK | **done**

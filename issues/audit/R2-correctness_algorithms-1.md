# R2 · correctness/algorithms · 1 — BM25 rescorer accepts `df > doc_count` → NEGATIVE IDF, inverted ranking

**Severity:** major
**Status:** fixed
**Package:** `latence-retrieval`
**File:** `packages/latence-retrieval/src/latence_retrieval/bm25.py` — `Bm25Rescorer._parse_stats`

## What was wrong

`_parse_stats` validated each caller-supplied document frequency only as
`int >= 0` but never bounded it by `doc_count`. A `df > doc_count` is
arithmetically impossible (a term cannot occur in more documents than the
corpus holds), yet it constructed successfully. The bad `df` then flowed into
`_CorpusStats.idf`, where the Lucene/Robertson formula
`idf = ln(1 + (N − df + 0.5) / (df + 0.5))` drops its log-argument below `1`
and returns a **negative** IDF. In `process` the `if idf == 0.0` guard does not
catch a negative value, so the negative IDF is applied and a document that
**contains** the query term scores **below** one that does not — a fully
inverted, quietly-wrong ranking on the module's own "exact BM25" path. This
directly violated the module's stated anti-false-green promise to "reject
nonsensical global statistics loudly at construction rather than yielding
quietly-wrong scores at query time."

## Failure scenario

A caller wires their store's stats but passes collection/total-term frequency
instead of document frequency (the classic BM25 df-vs-cf mixup): a frequent
term gets `df=999` with `doc_count=5`. Constructing
`Bm25Rescorer({"stats": {"doc_count": 5, "avg_doc_len": 10.0, "doc_freq": {"apple": 999}}})`
succeeds (no rejection). Rescoring query `"apple"` over
`[Candidate("has", "apple apple apple"), Candidate("hasnot", "banana cherry")]`
yields `idf ≈ -5.115`, so the term-bearing doc scores `≈ -10.33` and ranks
**below** the non-bearing doc at `0.0`.

## Fix

In `_parse_stats`, after the `df >= 0` check, reject any `df > doc_count` with a
`ValueError` naming the term, its df, and `doc_count`. `doc_count` is already
validated (`int >= 1`) before the `doc_freq` loop, so the bound is well-defined.
The class docstring now states the `doc_freq` invariant as `int in [0, doc_count]`.

## Acceptance criteria

- Constructing a rescorer with any `doc_freq[term] > doc_count` raises
  `ValueError` (loud, at construction) — no query-time execution reached.
- `df == doc_count` remains accepted (smallest non-negative IDF; a term in every
  document is legitimate).
- On the corrected wiring (`df <= doc_count`), a term-bearing document never
  ranks below a non-bearing one; its score is positive.
- Feature tests + `ruff` + `mypy --strict` stay green; change is additive
  validation only, no contract or pipeline regression.

## Regression tests

`packages/latence-retrieval/tests/test_bm25_rescorer.py`:
- `test_bm25_rejects_a_doc_freq_above_doc_count` (fails on old code — construction
  did not raise)
- `test_bm25_accepts_a_doc_freq_equal_to_doc_count` (boundary stays open)
- `test_bm25_never_ranks_a_term_bearing_doc_below_a_non_bearing_one_under_supplied_stats`

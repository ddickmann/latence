# R5-wizard_profile-13 — the blessed multilingual enterprise stack ships an ASCII-only BM25 leg

**Severity:** major
**Stream:** feat/release-fleet-integration (#194 enterprise-sota wizard profile, #164/ADR-0049 BM25)
**Status:** FIXED

## What was reported

> `packages/latence-core/src/latence_core/setup/wizard.py:120`
> (`SOTA_BM25_TOKENIZER = "tokenizer.regex"`, emitted by `_sota_bm25_block()` into the
> `export_corpus` stage) blesses `tokenizer.regex` as the enterprise-sota BM25 tokenizer, but the
> tokenizer itself
> (`packages/latence-core/src/latence_core/bm25.py:55`, `_WORD = re.compile(r"[A-Za-z0-9]+")`)
> drops every non-ASCII letter — so the blessed *multilingual* enterprise stack ships a sparse leg
> that shatters accented words into junk terms and produces literally zero terms for
> CJK/Cyrillic documents.

## Verification against the tree — CONFIRMED

The profile is explicitly the multilingual stack: it emits
`ibm-granite/granite-embedding-311m-multilingual-r2` (dense) and
`fastino/GLiNER2-Guardrails-PII-Multi` (PII), and #194's own test corpus is German.

Observed by running the accumulator the emitted `export_corpus` block wires
(`{'bm25': {'provider': 'tokenizer.regex', 'config': {}}}`) over three chunks:

```
c1 doc_len= 8 ['ausfuhr', 'e', 'f', 'gmbh', 'gr', 'ller', 'm', 'r']
c2 doc_len= 0 []
c3 doc_len= 4 ['classification', 'code', 'control', 'export']
avgdl 4.0 total_tokens 12
--- whitespace ---
c1 ['ausfuhr', 'für', 'gmbh', 'größe', 'müller']
c2 ['关税', '出口', '分类', '管制', '编码']
c3 ['classification', 'code', 'control', 'export']
```

Three distinct harms, all from one root cause:

1. **Latin-script mutilation.** `Müller` → `m` + `ller`, `für` → `f` + `r`, `Größe` → `gr` + `e`.
   The corpus IDF is polluted with high-`df` single-letter junk terms, and the real terms are
   unfindable.
2. **A permanently dead sparse leg for unsegmented/non-Latin scripts.** The CJK chunk indexes as
   `doc_len: 0` with an empty postings row. Because the query side
   (`packages/latence-retrieval/src/latence_retrieval/bm25.py:62`, `_TOKEN = re.compile(r"[a-z0-9]+")`)
   uses the identical ASCII pattern, a CJK/Cyrillic query tokenises to `[]` too — every candidate
   scores exactly `0.0`, so the sparse leg of the hybrid fuser is a no-op for those documents.
3. **Cross-contamination of every other document.** A zero-length document still counts toward
   `doc_count`, so it drags `avgdl` down and changes the BM25 length-normalisation — and therefore
   the score — of every *Latin* document in the same corpus.

The in-tree alternative `tokenizer.whitespace` preserves all of these tokens, and no ADR, docstring
or wizard header discloses the ASCII-only limitation: the emitted header sells the artifact
unqualified as what "the query-side sparse rescorer runs on".

The same ASCII-only pattern is shared by the in-core dense/sparse hashing references
(`latence_core/embedding.py:25`, re-exported by `latence_core/signals.py`), which
`bm25.py`'s module docstring explicitly claims the BM25 tokenisation "lines up with" — so fixing
BM25 alone would either leave that alignment claim false or leave the same defect in the signals
the same corpus emits.

## What to build

Make the reference tokenisation **Unicode-aware** rather than ASCII-only, at every place the one
pattern is spelled, and state the tokenisation honestly where it is blessed.

`[^\W_]+` is the replacement: Unicode word characters minus the underscore. Over pure-ASCII text it
matches `[A-Za-z0-9]+` character-for-character, so every existing corpus, artifact and golden
fixture stays **byte-identical** (the ADR-0049/0051 additive guarantee); over non-ASCII text it
keeps letters, digits and combining marks as parts of a term.

The residual limitation that *remains* after the fix — scripts written without word spacing
(Chinese/Japanese/Thai) yield one term per whitespace run, because a reference tokenizer is not a
word segmenter — must be **disclosed** in the docstrings and at the wizard blessing site, with the
Tokenizer seam named as the place a corpus plugs in a real analyzer.

## Failure scenario (the regression tests)

`test_regex_tokenizer_preserves_non_ascii_scripts` and
`test_multilingual_corpus_does_not_produce_empty_or_shattered_documents`
(`packages/latence-core/tests/test_bm25_artifact.py`) fold the mixed German/Chinese/English corpus
above through the blessed `tokenizer.regex` and assert no document is empty, `Müller`/`für`/`Größe`
survive whole, and no single-letter junk term enters the vocabulary. On the pre-fix code the first
fails with `['ausfuhr', 'e', 'f', 'gmbh', 'gr', 'ller', 'm', 'r']`.

`test_non_ascii_query_is_not_a_dead_no_op` (`packages/latence-retrieval/tests/test_bm25_rescorer.py`)
rescores candidates with a Cyrillic query and asserts the matching candidate outranks the
non-matching one with a strictly positive score; pre-fix every candidate scores `0.0`.

`test_export_and_query_tokenisation_agree_on_non_ascii`
(`packages/latence-core/tests/test_bm25_artifact.py`) pins the index-side and query-side patterns to
the same behaviour on non-ASCII input, so the two packages (deliberately dependency-independent)
cannot drift apart again.

## Acceptance criteria

- [x] `tokenizer.regex` keeps non-ASCII letters and combining marks inside a term; `Müller`,
      `für`, `Größe`, `Zollgüter` tokenise whole.
- [x] A CJK/Cyrillic document produces a non-empty posting and contributes to `avgdl`.
- [x] The query-side `Bm25Rescorer` tokenises a non-ASCII query to a non-empty term list and
      produces a strictly positive score for a matching candidate.
- [x] The in-core dense (`embedding.hashing`) / sparse (`sparse.hashing`) references use the same
      pattern, so `bm25.py`'s documented alignment claim stays true.
- [x] ASCII behaviour is **byte-identical** — no existing artifact, vector or golden fixture moves.
- [x] The remaining unsegmented-script limitation is disclosed in the tokenizer docstring, the
      Tokenizer Capability docstring, and the wizard's blessing comment + `_sota_bm25_block()`.
- [x] Regression tests fail on the old code and pass on the fix.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.

## Out of scope (noted, not fixed here)

`packages/latence-retrieval/src/latence_retrieval/pack.py:79` carries the same `[a-z0-9]+` pattern
for its `TokenCounter` word-proxy. That is a context-budget estimator, not the BM25 leg, and it has
its own `_HEURISTIC_TOKEN` (`\w+|[^\w\s]`, already Unicode-aware) alongside it; the interaction
between the two deserves its own ticket rather than a drive-by change under this one.

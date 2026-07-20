# R6-cross_lane_integration-13 — `processor.pack` redundancy penalty is dead for non-Latin scripts

**Severity:** major
**Site:** `packages/latence-retrieval/src/latence_retrieval/pack.py:79` (`_TOKEN`), consumed by
`_lexical_similarity` (pack.py:508-519) and thence `Packer._similarity` (pack.py:403).

## What to build

The R5 fix that made the reference BM25 tokenizer Unicode-aware (commit `c1fd8fc`) was wired to
three of its four call sites — `latence_core/bm25.py:65`, `latence_core/embedding.py:29`,
`latence_retrieval/bm25.py:68` all became `[^\W_]+`. The fourth, `latence_retrieval/pack.py:79`,
was missed and still tokenizes with `[a-z0-9]+`.

Make the packer's default lexical-Jaccard redundancy tokenizer Unicode-aware (`[^\W_]+`, the same
term unit the blessed sparse leg uses), and align the in-repo `FakeBackend` lexical scorer
(`latence_retrieval/testing.py:41`) so the test double cannot mask the same class of bug. Both
changes are ASCII-identical (the text is lowercased first), so pre-existing Latin behaviour is
bit-for-bit unchanged.

## Failure scenario

`_lexical_similarity(t, t)` returns `0.0` for
`t = "Таможенный кодекс Европейского союза регулирует импорт товаров."` and for
`t = "欧洲联盟海关法典规定了货物进口。"`, but `1.0` for the equivalent English/German strings —
`[a-z0-9]+` matches nothing outside ASCII, both token sets are empty, and `_lexical_similarity`
short-circuits to `0.0`.

End-to-end through the public seam: with candidates `dup1` (score 1.0), `dup2` (score 0.98, an
EXACT duplicate of `dup1`'s text) and `diverse` (score 0.90), and `Packer({"budget": 20})` — the
default, i.e. neither `similarity_key` nor `cluster_key` configured — the Latin corpus correctly
selects `['dup1', 'diverse']`, while the byte-for-byte equivalent Cyrillic corpus selects
`['dup1', 'dup2', 'diverse']`: the LLM context budget is spent on a verbatim duplicate and genuine
new evidence is evicted at tighter budgets.

Impact: #194 blesses `enterprise-sota` as the explicitly MULTILINGUAL profile (multilingual
embedder + multilingual SPLADE + multilingual PII model), and `processor.pack` is a documented
first-class component of that retrieval pipeline (`docs/TUTORIAL-retrieval-tooling.md` §3.3), so the
blessed multilingual stack shipped a packer whose de-duplication silently does nothing on exactly
the corpora the profile targets. No test covered non-ASCII input to `pack.py`.

## Acceptance criteria

1. `_lexical_similarity(t, t) == 1.0` for Cyrillic and CJK `t`, as it already is for Latin `t`.
2. Through the public `Packer.process` seam with default config (no `similarity_key`, no
   `cluster_key`), a Cyrillic exact-duplicate corpus selects the same id set as the equivalent
   Latin corpus — the duplicate is dropped in favour of the diverse candidate.
3. ASCII behaviour is byte-identical: every pre-existing pack test stays green unchanged.
4. The `FakeBackend` lexical scorer uses the same Unicode-aware term unit, so the in-repo double
   cannot false-green the same defect.
5. Regression test fails on the pre-fix code and passes on the fix. Full fast suite does not
   regress.

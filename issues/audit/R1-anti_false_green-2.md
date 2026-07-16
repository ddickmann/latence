# R1 — SPLADE redaction gate uses a substring `in` check, re-leaking a masked PII surface

- **Severity:** major
- **Dimension:** anti-false-green / REDACTION-BY-DEFAULT (ADR-0053)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #162 (04-splade-composition), ADR-0053
- **Status:** fixed

## What is wrong

ADR-0053's SPLADE composition runs on the corpus **redaction policy** by default
(`redact=True`): an entity whose surface no longer survives the corpus redaction must **not** be
re-injected as a boosted term, so a masked PII name is never made searchable again
(REDACTION-BY-DEFAULT).

The redaction gate in `SpladeCompositionEmbedder._entity_terms`
(`packages/latence-splade/src/latence_splade/provider.py:259`) enforced this with a **substring**
membership test:

```python
if self._redact and surface.casefold() not in haystack:
    continue
```

`in` on strings is a raw substring test, not a token/word-boundary test. So a redacted PII
surface whose casefolded form is a **substring of a surviving, unrelated word** passes the gate
and its vocab id is injected — even though that standalone token appears nowhere in the redacted
text. The one redaction test (`test_splade_embedder.py:142`) only asserted the full-absence case
(surface `"john"` vs text `"the invoice total"`), so the substring re-leak path was uncovered and
the suite stayed green.

## Failure scenario

NER tags the person `"Ann"` and the corpus redaction masks that span. The redacted corpus text no
longer contains the standalone token `"ann"` but still contains the unrelated word `"annual"`
(e.g. `"the annual revenue report"`). The gate evaluates
`"ann" in "the annual revenue report"` → `True`, so the entity is **not** dropped and
`model.term_ids("Ann")` is injected. Run through the shipped `FakeSpladeModel` via
`encode_composed`, the emitted `SparseVector` carries weight `1.0` at the vocab id for `"ann"`
(id 46) — a token that appears nowhere in the redacted text and is distinct from `"annual"`
(id 60). A store's sparse index built from this column now matches a query for `"ann"` to the
document, making the redacted name searchable. Realistically triggerable by common short names
colliding as substrings: Ann/annual, Sam/same, Al/also, Ed/editor, Ian/median.

## What to build

Replace the substring gate with a **word-token** survival check, in the same lower-cased token
space ADR-0053 injects into:

1. Tokenise both the entity surface and the redacted text into word tokens (unicode `\w+`,
   casefolded — mirroring the model's lower-cased subword vocabulary).
2. An entity survives redaction iff its surface's token sequence occurs as a **contiguous
   sub-sequence** of the text's token sequence (so a masked span whose token is only a *substring*
   of a surviving word — `ann` inside `annual` — is dropped, and multi-word surfaces like
   `"John Smith"` are matched as a phrase).
3. `redact=False` (the explicit fully-internal opt-in) is unchanged: it injects every
   confidence-gated entity regardless of the redaction floor.

The gate stays inside the Provider (it needs the redacted text) and remains a small hidden
behaviour behind `_entity_terms` (deep-module standard); the composition algebra is untouched.

## Acceptance criteria

- [x] A redacted entity whose surface is only a **substring** of a surviving word (not a standalone
      token) is dropped and its vocab id appears nowhere in the emitted `SparseVector`
      (REDACTION-BY-DEFAULT preserved).
- [x] An entity whose surface **does** occur as a standalone word (or contiguous phrase) in the
      redacted text is still injected (no false-drop regression) — including multi-word surfaces.
- [x] `redact=False` still injects the entity even when absent from the text.
- [x] A non-vacuous regression test **fails on the old substring code** and **passes on the fix**,
      asserting the absence on the *observable emitted vector* (not a config flag); the same input
      with `redact=False` is shown to inject the id, proving the id resolves and only the redaction
      gate suppresses it.

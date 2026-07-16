# R1 — SPLADE redaction gate uses substring containment, re-injecting a masked PII surface

- **Severity:** major
- **Dimension:** deep-module quality / REDACTION-BY-DEFAULT (ADR-0053)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #162 (04-splade-composition), ADR-0053
- **Status:** resolved (already fixed on this branch — see below)

## What is wrong

ADR-0053's SPLADE composition runs on the corpus **redaction policy** by default (`redact=True`):
an entity whose surface no longer survives the corpus redaction must **not** be re-injected as a
boosted sparse term, so a masked PII name is never made searchable again (REDACTION-BY-DEFAULT).

The redaction gate in `SpladeCompositionEmbedder._entity_terms`
(`packages/latence-splade/src/latence_splade/provider.py`) enforced this with a raw **substring**
membership test:

```python
if self._redact and surface.casefold() not in haystack:
    continue
```

`in` on strings is a raw substring test, not a token/word-boundary test. A redacted PII surface
whose casefolded form is a **substring of a surviving, unrelated word** passes the gate, and its
distinct vocab id (a standalone token that appears nowhere in the redacted text) is injected into
the composed `SparseVector` — re-leaking the name the corpus redacted.

## Failure scenario

With `redact=True` (default),
`SpladeCompositionEmbedder(model=fake).encode_composed([SpladeInput(text="the annual total report", entities=[EntityTerm("Ann", 0.99)])])`
re-injects the redacted surface `"Ann"`: because `"ann" in "the annual total report"` is `True`
(substring of `"annual"`), the entity passes the redaction gate and `model.term_ids("Ann")`
(the token id for `"ann"`, distinct from `"annual"`) is added to the composed `SparseVector` with
weight `1.0`. A store's sparse index built from this column now matches a query for `"ann"` to the
document, making the redacted name searchable. Realistically triggerable by common short names
colliding as substrings: Ann/annual, Al/total, Kim/victim, Sam/same, Ed/editor, Ian/median.

## What to build

Replace the substring gate with a **word-token** survival check, in the same lower-cased token
space ADR-0053 injects into:

1. Tokenise both the entity surface and the redacted text into word tokens (unicode `\w+`,
   casefolded — mirroring the model's lower-cased subword vocabulary).
2. An entity survives redaction iff its surface's token sequence occurs as a **contiguous
   sub-sequence** of the text's token sequence (so a masked span whose token is only a *substring*
   of a surviving word — `ann` inside `annual` — is dropped, and multi-word surfaces like
   `"John Smith"` are matched as an adjacent phrase).
3. `redact=False` (the explicit fully-internal opt-in) is unchanged: it injects every
   confidence-gated entity regardless of the redaction floor.

The gate stays inside the Provider (it needs the redacted text) and remains a small hidden
behaviour behind `_entity_terms` (deep-module standard); the composition algebra is untouched.

## Acceptance criteria

- [x] A redacted entity whose surface is only a **substring** of a surviving word (not a standalone
      token) is dropped and its vocab id appears nowhere in the emitted `SparseVector`.
- [x] An entity whose surface **does** occur as a standalone word (or contiguous phrase) in the
      redacted text is still injected (no false-drop regression) — including multi-word surfaces.
- [x] `redact=False` still injects the entity even when absent from the text.
- [x] A non-vacuous regression test **fails on the old substring code** and **passes on the fix**,
      asserting the absence on the *observable emitted vector* (not a config flag); the same input
      with `redact=False` is shown to inject the id, proving the id resolves and only the redaction
      gate suppresses it.

## Resolution

Confirmed real and **already fixed on this branch** in commit `3d8c561`
("fix(audit-R1): SPLADE redaction gate uses a substring `in` check ..."), which was tracked under
the equivalent ticket `issues/audit/R1-anti_false_green-2.md`. This ticket is filed for
`deep_module_quality-4` traceability and records the same confirmed defect.

- **Fix:** `provider.py` — the substring gate `surface.casefold() not in haystack` was replaced by
  `_occurs_as_phrase(_word_tokens(surface), haystack)`, a contiguous word-token sub-sequence check
  over lower-cased unicode `\w+` tokens (helpers `_word_tokens` / `_occurs_as_phrase`).
- **Regression test:**
  `packages/latence-splade/tests/test_splade_embedder.py::test_redaction_drops_a_masked_surface_that_is_only_a_substring_of_a_surviving_word`
  — asserts `_tid("ann")` is absent from the emitted vector while `_tid("annual")` is present
  (proving composition ran, not an empty vector), and that the same input under `redact=False`
  DOES inject `"ann"` (non-vacuous). Independently re-verified during this audit: the test **fails
  on the reverted substring gate** (leaks vocab id 46 = `"ann"`) and **passes on the shipped fix**.
- **Companion test:** `test_redaction_keeps_a_multiword_surface_that_survives_as_an_adjacent_phrase`
  guards against a false-drop regression on surviving multi-word phrases.

No further code change required. Feature suite + ruff + mypy --strict remain green.

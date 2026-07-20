# R7-wizard_profile-11 — the tutorial-truth gate exempts the one `model` pin that is NOT operator-supplied

**Severity:** major
**Area:** wizard_profile / docs-truth gates
**Files:**
- `packages/latence-core/tests/test_tutorial_sota_stack_truth.py` (`_OPERATOR_SPECIFIC`, `_strip_operator_specific`)
- `docs/TUTORIAL-fresh-pod-walkthrough.md` §4
- `packages/latence-core/src/latence_core/setup/wizard.py` (`_sota_embedder_block()`)

## What is wrong

The R6 tutorial-truth gate neutralises the config key `model` **recursively at every nesting
depth** before diffing §4's hand-authored stack against the live wizard output. The wizard's
enterprise-sota embedder checkpoint (`_sota_embedder_block()`, hard-coded
`ibm-granite/granite-embedding-311m-multilingual-r2`, emitted into BOTH
`disambiguate.config.embedder` and `export_corpus.config.embedder`) is therefore exempt from the
"exact enterprise-sota spine" diff — the gate passes with exactly the drift class it was written
to catch (R6-wizard_profile-15: §4 silently diverging from the blessed wizard output on a
model/provider pin).

The exemption is also gratuitous. `_wizard_stages()` already supplies the tutorial's own
`ocr_model` and `llm_model` in its `SetupAnswers` fixture, so a raw *unstripped* diff shows
`parse.config` and `induce.config` matching byte-for-byte. The only stage-config values that
genuinely differ between the tutorial and the fixture are `source.config.path` (`{CORPUS_DIR}`)
and `graph/complete.config.corpus_id` (the stack name). Stripping `model` protects nothing and
blinds the one `model` key that is not operator-supplied.

## Failure scenario (proven)

Copy the tutorial, replace `ibm-granite/granite-embedding-311m-multilingual-r2` with
`sentence-transformers/all-MiniLM-L6-v2`, re-run the gate's own comparison logic verbatim:

```
ORIGINAL:                 gate drift = NONE
MUTATED-embedder-model:   gate drift = NONE  (test PASSES)
  tutorial embedder model = sentence-transformers/all-MiniLM-L6-v2
  wizard   embedder model = ibm-granite/granite-embedding-311m-multilingual-r2
```

All four gate tests stay green — including
`test_tutorial_section4_stage_wiring_equals_the_wizard_output`, whose docstring claims "any future
wizard change the tutorial does not track fails here".

Concretely: an operator hand-authoring §4 (the documented way to "understand every knob") would
wire a 384-dim English-only MiniLM behind the block's own `dimension: 768` (which IS compared and
would still read 768), producing a dimension-inconsistent, non-multilingual dense leg for both the
GLinker STS rung and `export_corpus`, and no test in the tree would fail.

## What to build

1. Drop `"model"` from `_OPERATOR_SPECIFIC` — the fixture supplies the tutorial's `ocr_model` and
   `llm_model`, so `parse.config.model` and `induce.config.model` compare byte-for-byte.
2. Stop neutralising placeholder keys at arbitrary nesting depth. The operator-supplied
   placeholders live at the TOP level of a stage's `config`; nested Provider blocks (embedder,
   graph_features, bm25, the universal parser's `ocr` sub-block) are pure wiring and must be
   compared exactly.
3. Extract the drift computation into a reusable helper so a regression test can drive it against
   a mutated tutorial fixture rather than re-implementing it.

## Acceptance criteria

- [ ] `"model"` is not exempt anywhere in the gate.
- [ ] Placeholder neutralisation is depth-limited to the top level of a stage's `config`.
- [ ] A regression test mutates the tutorial's nested embedder `model` (and a nested provider pin)
      and asserts the gate's own comparison logic reports drift; it FAILS on the pre-fix code.
- [ ] The unmutated tutorial still shows zero drift (the gate stays green on the real tree).
- [ ] `make test` stays at or above 3320 passed / 29 skipped.

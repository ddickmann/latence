# R1-benchmark_integrity-13 — Published "learned re-pin" Provider names are not registered under any entry point

**Severity:** major
**Stream:** feat/release-fleet-integration (#187 benchmark suites)
**Status:** FIXED

## What is wrong

Every "learned re-pin" Provider name published by the #187 RED^FM / redaction suites is wrong:

| published (wrong) | actually registered | registering package |
| --- | --- | --- |
| `ner.gliner` | `entity.gliner` | `packages/latence-ner-gliner/pyproject.toml:50` |
| `relation.gliner` | `relation.gliner_relex` | `packages/latence-relation-gliner/pyproject.toml:50` |
| `pii.gliner` | `redaction.gliner_pii` | `packages/latence-pii-gliner/pyproject.toml:52` |

These strings are not merely doc prose. They live in the `notes` tuples of `EntitySuite`,
`RelationSuite` and `RedactionSuite`, so `render_markdown` / `render_json` emit them verbatim
into **both** published artifacts (`docs/BENCHMARKS.md`, `docs/benchmarks/results.json`) on
every run.

Sites:
- `packages/latence-benchmark/src/latence_benchmark/redfm_suites.py:72, 92, 271, 376`
- `packages/latence-benchmark/src/latence_benchmark/redaction_suite.py:288`
- `docs/running-the-benchmarks.md:120, 126`
- `docs/BENCHMARKS.md:180`, `docs/benchmarks/results.json:371` (generated artifacts)

`parser.lighton`, `relation.llm` and `disambiguation.glinker` were checked and are correct, so
this is a per-name transcription defect, not a registry-wide naming change.

## Failure scenario

A reader follows the published results page and runs:

```
latence-benchmark run --suite entity_extraction --provider ner.gliner
```

Even with `latence-ner-gliner` correctly installed, this raises
`ProviderUnavailableError: ... is not installed. Install the package that registers it and
re-run` — which actively misdirects the reader into reinstalling a package that will never
register that name.

## What to build

1. Correct the three names at every source site so the notes (and therefore the published
   markdown + JSON) name the real entry points.
2. Regenerate / correct the published artifacts that already carry the wrong strings.
3. Add a **structural** regression guard so this class of defect cannot recur: any
   `--provider <name>` string emitted by the benchmark package must correspond to a Provider
   name actually declared under a `latence.providers` / `latence.retrieval` entry-point group
   in some `packages/*/pyproject.toml`.

## Acceptance criteria

- [x] `ner.gliner`, `relation.gliner`, `pii.gliner` appear nowhere in the repo.
- [x] `EntitySuite`, `RelationSuite`, `RedactionSuite` notes name `entity.gliner`,
      `relation.gliner_relex`, `redaction.gliner_pii`.
- [x] `docs/running-the-benchmarks.md`, `docs/BENCHMARKS.md`, `docs/benchmarks/results.json`
      carry the corrected names.
- [x] A regression test cross-checks every `--provider <name>` mentioned by the benchmark
      package against the real declared entry-point names, and FAILS on the pre-fix source.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.

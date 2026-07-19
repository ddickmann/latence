# R1 — the benchmark guide claims four suites are published; only Parse is on the results page

- **Severity:** major
- **Dimension:** docs-truth / published-artifact integrity
- **Branch:** feat/release-fleet-integration
- **Tickets:** #187 (Stage benchmark suites), #189 (flip public)
- **Status:** fixed

## What is wrong

`docs/running-the-benchmarks.md` opens with

> The published numbers in [Public benchmark results](BENCHMARKS.md) are not something you have
> to take on faith.

then tables **five** suites with what each one *reports* (per-language precision/recall/F1 for
the three RED^FM suites; redaction recall and precision per language and per PII type), and
twice instructs the reader to read those numbers off the page:

- `:84` — "That is why the per-language slices are published."
- `:106` — "Three criteria are published, and the headline is the loosest of them on purpose."

None of that is on the page. `docs/BENCHMARKS.md` contains exactly one section,
`## Parse — \`parse\` suite`, and `docs/benchmarks/results.json` carries a `measurements` array
of length **1** (dataset `olmocr-bench`). #187's acceptance criterion — *all suites publish to
the same results page as the Parse slice* — is unmet, and `README.md:174` sends every reader to
that page as "public benchmark results".

The harness itself is fine: all five suites are registered
(`disambiguation, entity_extraction, parse, redaction, relation_extraction`), and the
**redaction** suite needs no network at all — its synthetic corpus ships inside the package and
is pinned by content digest. The privacy number that ticket #187 exists to make externally
checkable was obtainable in seconds and simply never published, while the shipped guide asserts
that it was.

A second, structural cause: `latence-benchmark run --suite <one>` **overwrites** both published
artifacts with a page containing only that suite. So publishing redaction would have deleted the
Parse numbers, and the only way to have a complete page was a single `--suite all` run with every
dataset present at once — which is why the page never grew past the first suite anyone ran.

## Failure scenario

An evaluator follows `README.md:174` → `docs/BENCHMARKS.md` looking for the framework's central
privacy claim, having just been told by `docs/running-the-benchmarks.md` that redaction recall is
published per language and per PII type under three match criteria. They find a Parse pass rate
and nothing else: no extraction numbers, no disambiguation numbers, no redaction numbers, and —
unlike `PERF-RESULTS.md`, which keeps a rigorous *Not yet measured* section — no statement that
these are unmeasured. The published guide is false in the present tense on the one page an
outsider uses to decide whether the project's numbers can be trusted.

Reproduced on the pre-fix tree:

```
$ uv run --no-sync python -c "import json;print(len(json.load(open('docs/benchmarks/results.json'))['measurements']))"
1
$ grep -c '^## .* suite$' docs/BENCHMARKS.md
1
$ PYTHONHASHSEED=0 uv run --no-sync latence-benchmark run --suite redaction --markdown-out /tmp/r.md --json-out /tmp/r.json
redaction: precision 100.0%, recall 77.1%, F1 87.1% (micro over language; 35 gold items, 20 documents)
```

## What to build

1. **Make a partial run additive, not destructive.** A run that covers a subset of the registered
   suites republishes the previously published measurements for the suites it did not re-run,
   reconstructed from `docs/benchmarks/results.json` and validated against the registry: if the
   suite is no longer registered, or its dataset is now pinned to a different revision, the run is
   **refused** with a typed error rather than republishing a number against data that moved
   (ADR-0034: no silent errors). `--no-carry-forward` writes a clean single-suite page.
2. **Date every measurement.** Each published measurement carries the date it was measured
   (`measured_on`), rendered in its section, so a page assembled from several runs never implies
   that a carried-forward number was produced today.
3. **Publish the redaction suite.** It runs offline from a clean clone; there is no excuse for its
   absence. Its section must carry the per-language, per-PII-type and per-match-criterion tables
   the guide describes.
4. **Tell the truth about the rest.** `docs/running-the-benchmarks.md` states which suites are
   published today and carries an explicit **Not yet measured** section for the three RED^FM
   suites (they need the RED^FM corpus downloaded), in the same discipline `PERF-RESULTS.md`
   applies — including the command that produces them.

## Acceptance criteria

- [x] `docs/benchmarks/results.json` and `docs/BENCHMARKS.md` contain the **redaction** suite
      alongside Parse; the Parse numbers are unchanged (same pass rates, same pin fingerprint).
- [x] Every suite the guide describes as published has a section on the results page and a
      measurement in the sidecar; every suite that is not published is named in a *Not yet
      measured* section with the reason and the command that would produce it.
- [x] A partial run no longer destroys previously published suites, and refuses (typed error,
      non-zero exit) to carry a measurement whose dataset revision no longer matches the registry.
- [x] Each measurement records and renders the date it was measured.
- [x] A non-vacuous regression test derives the claim set from the guide and the published set
      from the artifacts, **fails on the pre-fix tree** (four suites claimed, one published, no
      *Not yet measured* section) and passes on the fix — asserting the published artifacts, not
      a flag recording intent.

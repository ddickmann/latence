# Run the public benchmarks

The published numbers in [Public benchmark results](BENCHMARKS.md) are not something you have to
take on faith. This page is how you reproduce them, and how you add a suite for another Stage.

## One command

```bash
uv sync
uv run latence-benchmark run --suite parse
```

That does the whole loop: check the dataset's license, fetch it at its pinned revision, run the
Parse Stage through the pinned Provider, score the dataset's unit tests, and rewrite
`docs/BENCHMARKS.md` + `docs/benchmarks/results.json`.

Remote acquisition needs the `[hub]` extra (`uv pip install 'latence-benchmark[hub]'`). If you
already have the dataset — or you are on an air-gapped host — skip acquisition entirely:

```bash
uv run latence-benchmark run --suite parse --dataset-root /path/to/olmOCR-bench
```

Useful flags:

| Flag | What it does |
| --- | --- |
| `--provider parser.lighton` | Re-pin the Stage's Provider. The new pin is fingerprinted into the results, so two runs are always distinguishable. |
| `--limit 25` | Cap documents per split for a fast smoke run. The result is labelled `SUBSET RUN` and must not be compared against full-benchmark numbers. |
| `--accept-non-commercial` | Acknowledge a research-only dataset license. Never lifts an `UNVERIFIED` license. |
| `--markdown-out` / `--json-out` | Write the artifacts somewhere other than `docs/`. |

## What the harness refuses to do

**Publish an unreproducible number.** A dataset is pinned to a full 40-hex commit SHA;
branches and tags are rejected at construction, because they move. A partially-downloaded
dataset is refused rather than partially scored — scoring the splits that happened to arrive
would publish a number that is not the benchmark.

**Use a dataset whose license was not checked.** Corpus and evaluation-toolkit licenses are
verified *separately* (ADR-0012) and the gate runs *before* acquisition. This is not
hypothetical: OmniDocBench's repository badge is Apache-2.0, but that is the toolkit — the
corpus copyright statement restricts it to research use. It is registered, gated behind
`--accept-non-commercial`, and never the default.

**Flatter itself.** A parse failure counts its document's tests as *failures*, so a Provider
cannot crash its way to a better score. Test classes the harness does not yet score are
reported as *unsupported* — excluded from the pass-rate denominator and shown in their own
column, never counted as passes. Numbers published by other projects live in their own section
labelled as claims, because we did not run them.

## Reading the Parse result honestly

The pinned Parse Provider, `parser.pdfplumber`, reads the PDF **text layer**. olmOCR-bench is
substantially scanned and photographed material with no text layer at all, so a large share of
its tests fail by construction rather than by defect. That is the point of publishing the
number with its pin attached: it measures the CPU-first default an adopter gets out of the box,
and the way to measure an OCR path is to re-pin (`--provider parser.lighton`, GPU) and publish
that alongside — not to quietly swap the Provider behind the same headline figure.

## Adding a Stage suite

The seam is `latence_benchmark.suite.BenchmarkSuite`. A suite supplies a name, the Stage it
measures, a pinned `DatasetSpec`, a `StackPin` and a `run(root)`; `register_suite` makes it
visible to the CLI and to the results page with no other change. Everything around it — the
license gate, acquisition, the CLI, both artifacts — is written against that protocol only.

Two rules for a new dataset: pin a full commit SHA, and verify the corpus and toolkit licenses
separately, recording both on the `DatasetSpec`. If you cannot verify one, record it as
`UNVERIFIED` — the gate will then refuse the dataset, which is the correct outcome.

## Not the gold-set harness

This is the framework-side **public** benchmark. The adopter gold-set harness (ADR-0028), which
scores *your* corpus against *your* annotations inside a pipeline run's Quality Report, is a
separate thing and stays separate.

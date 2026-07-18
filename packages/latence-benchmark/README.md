# latence-benchmark

The **public** benchmark harness for the Latence framework: one command runs a framework Stage
over a public dataset pinned to an exact commit, under a pinned stack config, and publishes both
a human-readable results page and a machine-readable sidecar.

```bash
uv run latence-benchmark run --suite parse
```

That fetches [`allenai/olmOCR-bench`](https://huggingface.co/datasets/allenai/olmOCR-bench) at
its pinned revision (needs the `[hub]` extra), runs the Parse Stage through
`parser.pdfplumber`, scores the dataset's unit tests, and writes `docs/BENCHMARKS.md` +
`docs/benchmarks/results.json`. Already have the dataset? Point at it and run fully offline:

```bash
uv run latence-benchmark run --suite parse --dataset-root /path/to/olmOCR-bench
```

## What this is not

This is **not** the adopter gold-set harness of ADR-0028. That one scores a *customer's* corpus
against *their* annotations and stays adopter-side. This one scores the framework against public
data so an outsider can reproduce our numbers — and it registers nothing under
`latence.providers`, because it is a consumer of the pipeline, not a part of it.

## The three things it refuses to do

**Publish an unreproducible number.** A `DatasetSpec` rejects anything but a full 40-hex commit
SHA — branches and tags move. Every result carries the dataset revision and a fingerprint of the
exact Provider + config that produced it.

**Use a dataset whose license we did not check.** Corpus and evaluation-toolkit licenses are
verified *separately* (ADR-0012) and the gate runs before acquisition, so restricted bytes never
land on disk. This is not hypothetical: OmniDocBench, the obvious pick for document parsing,
ships an Apache-2.0 toolkit next to a corpus that is research-only, so it is registered but
gated behind `--accept-non-commercial` and is never the default.

**Flatter itself.** `table` and `math` assertions are not scored yet, so they are reported as
*unsupported* — excluded from the pass-rate denominator and shown in their own column, rather
than counted as passes. A parse failure counts its document's tests as failures, so a parser
cannot crash its way to a better score. Vendor numbers are labelled claims and live in their own
section, because we did not run them.

## Adding a Stage suite (#187)

The seam is `latence_benchmark.suite.BenchmarkSuite`. Supply a name, the Stage you measure, a
pinned `DatasetSpec`, a `StackPin` and a `run(root)`, then `register_suite` it — the CLI and the
results page pick it up with no other change. See the module docstring in `suite.py`.

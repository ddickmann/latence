# Run the public benchmarks

The published numbers in [Public benchmark results](BENCHMARKS.md) are not something you have to
take on faith. This page is how you reproduce them, and how you add a suite for another Stage.

## One command

```bash
uv sync
uv run latence-benchmark run --suite all
```

That does the whole loop for **every** suite: check each dataset's license, fetch it at its
pinned revision, run the Stage through its pinned Provider, score it, and rewrite
`docs/BENCHMARKS.md` + `docs/benchmarks/results.json` — one page, every Stage.

| Suite | Stage | Dataset | What it reports |
| --- | --- | --- | --- |
| `parse` | Parse | olmOCR-bench | Unit-test pass rate per split |
| `entity_extraction` | Entity Extraction | RED^FM (7 languages) | Precision/recall/F1 per language, on two match criteria |
| `relation_extraction` | Relation Extraction | RED^FM (7 languages) | Precision/recall/F1 per language and per predicate, from GOLD mentions |
| `disambiguation` | Disambiguation | RED^FM (7 languages) | Pairwise precision/recall against gold Wikidata identity |
| `redaction` | Redaction | Shipped synthetic PII corpus (en + de) | Redaction recall (and precision) per language and per PII type |

Remote acquisition needs the `[hub]` extra (`uv pip install 'latence-benchmark[hub]'`). If you
already have the datasets — or you are on an air-gapped host — skip acquisition entirely. The
synthetic PII corpus never needs a path: it ships inside the package.

```bash
uv run latence-benchmark run --suite parse --dataset-root /path/to/olmOCR-bench
uv run latence-benchmark run --suite all \
  --dataset-root olmocr-bench=/path/to/olmOCR-bench \
  --dataset-root redfm=/path/to/REDFM
```

Useful flags:

| Flag | What it does |
| --- | --- |
| `--suite all` | Run every registered suite onto the same results page. |
| `--provider parser.lighton` | Re-pin the Stage's Provider. The new pin is fingerprinted into the results, so two runs are always distinguishable. Refused with `--suite all`, because one Provider name cannot be the right pin for five Stages. |
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

## Reading the extraction and disambiguation results honestly

All three RED^FM suites pin the framework's **dependency-free CPU built-ins**, for the same
reason Parse pins `parser.pdfplumber`: the published number must be the one you reproduce from a
clean clone. That has consequences you should read off the tables rather than be surprised by.

* **Entity Extraction** pins `entity.gazetteer` with a capitalisation + numeric recogniser. It is
  an entity *detection* baseline: it has no notion of RED^FM's label inventory beyond `NUMBER`,
  so `span+type` sits far below `span`, and Arabic and Chinese — which have no case distinction —
  score near zero **by construction**. That is why the per-language slices are published. The
  learned re-pin is `--provider ner.gliner`.
* **Relation Extraction** pins `relation.pattern` with five type-prior co-occurrence rules whose
  head/tail signatures were derived from RED^FM's **train** split (never the scored `test`
  split). A co-occurrence rule fires on every in-window type pair, so recall is respectable and
  precision is the honest weakness of a rule baseline. It is scored from **gold mentions**, so
  the number measures relation extraction rather than the entity extractor upstream of it. The
  learned re-pins are `--provider relation.gliner` / `relation.llm`.
* **Disambiguation** pins `disambiguation.cascade`, which resolves entities by *surface* and has
  no Wikidata index. It therefore cannot link two different surfaces of one entity, and it will
  merge two same-surfaced different entities — both of which the pairwise metric charges it for.
  The KB-linking re-pin is `--provider disambiguation.glinker`.

## Reading the redaction result honestly

The redaction suite exists to make the framework's central privacy claim externally checkable, so
its corpus is built to be **failable**: alongside the structured PII a rule engine can catch, it
plants person names and postal addresses, which a rule-based redactor structurally cannot. A gold
set containing only what the pinned Provider already detects would report 100% recall forever and
catch no regression in either direction.

Three criteria are published, and the headline is the loosest of them on purpose:

* **coverage** — is the gold value entirely inside some masked span? That is the question a
  privacy claim actually answers.
* **exact span** — do the boundaries agree?
* **exact span + type** — do the boundaries *and* the PII label agree? A date of birth caught by
  the phone recogniser is still masked, but a policy scoped to `types: ["date_of_birth"]` would
  have missed it — so the gap between this and coverage is a real operational risk, published
  rather than smoothed over.

No real personal data is in this repository. Every planted value is a reserved or published
example (RFC 2606 domains, RFC 5737 TEST-NET addresses, 555 fiction-reserved numbers, the
published test card and IBAN values, AWS's own documentation key, fictional names), the corpus is
generated by a committed generator whose output is byte-checked in CI, and it is pinned by
**content digest** — editing it without re-pinning fails the run rather than silently moving the
published number.

## Adding a Stage suite

The seam is `latence_benchmark.suite.BenchmarkSuite`. A suite supplies a name, the Stage it
measures, a pinned `DatasetSpec`, a `StackPin` and a `run(root)`; `register_suite` makes it
visible to the CLI and to the results page with no other change. Everything around it — the
license gate, acquisition, the CLI, both artifacts — is written against that protocol only.

A suite reports in one of two currencies and never invents a third: `SplitScore`
(pass/fail/unsupported) for a suite scoring unit tests, and `MetricScore` (TP/FP/FN, from which
precision, recall and F1 are derived) for a suite scoring a labelled set. A pass rate cannot
express a labelled set — a system that predicts nothing would score a perfect "no false
positives" — and an undefined ratio is reported as `n/a`, never as 0%.

Two rules for a new dataset: pin a full commit SHA, and verify the corpus and toolkit licenses
separately, recording both on the `DatasetSpec`. If you cannot verify one, record it as
`UNVERIFIED` — the gate will then refuse the dataset, which is the correct outcome. This is not
theoretical: WikiANN/PAN-X, the obvious multilingual NER set, carries `license: unknown` on its
dataset card and was therefore **excluded** rather than guessed at (see
`THIRD-PARTY-LICENSES.md`). A corpus with no upstream repository — the synthetic PII set — is
pinned by content digest instead (`revision_kind: content-digest`), which
`ShippedCorpusSource` verifies on every run.

## Not the gold-set harness

This is the framework-side **public** benchmark. The adopter gold-set harness (ADR-0028), which
scores *your* corpus against *your* annotations inside a pipeline run's Quality Report, is a
separate thing and stays separate.

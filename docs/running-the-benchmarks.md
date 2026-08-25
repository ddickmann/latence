# Run the public benchmarks

The published numbers in [Public benchmark results](BENCHMARKS.md) are not something you have to
take on faith. This page is how you reproduce them, how you produce the ones that are [not yet
measured](#not-yet-measured), and how you add a suite for another Stage.

## One command

```bash
uv sync
uv run latence-benchmark run --suite all
```

That does the whole loop for **every** suite: check each dataset's license, fetch it at its
pinned revision, run the Stage through its pinned Provider, score it, and rewrite
`docs/BENCHMARKS.md` + `docs/benchmarks/results.json` — one page, every Stage.

| Suite | Stage | Dataset | What it reports | Published today |
| --- | --- | --- | --- | --- |
| `parse` | Parse | olmOCR-bench | Unit-test pass rate per split | **yes** |
| `redaction` | Redaction | Shipped synthetic PII corpus (en + de) | Redaction recall (and precision) per language and per PII type, on three match criteria | **yes** |
| `entity_extraction` | Entity Extraction | RED^FM (7 languages) | Precision/recall/F1 per language, on two match criteria | no — [not yet measured](#not-yet-measured) |
| `relation_extraction` | Relation Extraction | RED^FM (7 languages) | Precision/recall/F1 per language and per predicate, from GOLD mentions | no — [not yet measured](#not-yet-measured) |
| `disambiguation` | Disambiguation | RED^FM (7 languages) | Pairwise precision/recall against gold Wikidata identity | no — [not yet measured](#not-yet-measured) |

"What it reports" is what the suite *computes when you run it*. Only the rows marked **yes**
have numbers on [Public benchmark results](BENCHMARKS.md) right now; the rest are declared under
[Not yet measured](#not-yet-measured) rather than implied. Every section of the results page
states the date its own numbers were measured, because covering every Stage takes more than one
run — a run that scores a subset republishes the other Stages' published numbers unchanged
rather than deleting them (`--no-carry-forward` opts out), and refuses to republish any number
whose dataset has since been re-pinned.

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
| `--no-carry-forward` | Write a page containing only the suites this run scored. By default a partial run republishes the other Stages' published numbers unchanged, each with the date it was measured — and refuses to republish one whose dataset has since been re-pinned. |

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

## Not yet measured

The three RED^FM suites — `entity_extraction`, `relation_extraction`, `disambiguation` — are
**implemented and runnable, but their numbers are not published yet**. They score the RED^FM
corpus, which has to be downloaded (it does not ship in this repository), so nobody has run them
on a host that had it. There is no partial result hiding behind that sentence: the results page
has no section for them, and the JSON sidecar has no measurement for them.

Produce them yourself — the run publishes onto the same page, next to Parse and Redaction, and
leaves those two untouched:

```bash
uv pip install 'latence-benchmark[hub]'
uv run latence-benchmark run --suite entity_extraction
uv run latence-benchmark run --suite relation_extraction
uv run latence-benchmark run --suite disambiguation
# or, with every dataset already on disk:
uv run latence-benchmark run --suite all \
  --dataset-root olmocr-bench=/path/to/olmOCR-bench \
  --dataset-root redfm=/path/to/REDFM
```

The section below says what those numbers will look like and why — read it *before* you read the
numbers, not after.

### GLiNER 2.5 has no numbers at all (ADR-0064)

`fused_entity_relation.gliner25` and `redaction.gliner25` shipped **unmeasured**, and the silence is
deliberate rather than an omission: the checkpoint has never been loaded in this repository. Their
whole test suite runs against a faithful offline fake of `gliner2` 2.0.0 which reproduces the real
API *and its real degradations* — that proves the Providers parse, ground and batch it correctly and
proves **nothing** about extraction or redaction quality.

Two measurements are outstanding, and this is where each lands:

| What | Where the number goes | What produces it |
| --- | --- | --- |
| PII recall/precision for `redaction.gliner25` | a Redaction slice on [the results page](BENCHMARKS.md), with its own stack pin and measured-on date | `latence-benchmark run --suite redaction --provider redaction.gliner25` — the existing suite, parameterised by Provider |
| Fused NER+RE quality for `fused_entity_relation.gliner25` | the RED^FM extraction sections above, once run with that Provider pinned | `latence-benchmark run --suite entity_extraction --provider fused_entity_relation.gliner25` (and `relation_extraction`), which needs the RED^FM download |
| Throughput | nowhere on this page | a GPU pod run. It is never estimated offline; see `PERF-RESULTS.md` for how the gliner2 1.x numbers were produced. |

#### What IS measured about GLiNER 2.5, and why it is not on the results page

One thing about the 2.5 checkpoint *has* been measured, and it is a property of the model rather
than a score, so it belongs here rather than on [the results page](BENCHMARKS.md). On a pod GPU
(RTX 2000 Ada 16 GB, driver 580.159.04, torch 2.4.1+cu124, transformers 4.57.6, gliner2 2.0.0,
`fastino/gliner2.5-multi-v1`, 2026-08-25), one call at the card's advertised `max_len=4096` over a
document seeded with a repeated PII sentence:

| tokens | pii_spans returned | how far into the document the spans reach |
| ---: | ---: | ---: |
| 674 | 24 | 100% |
| 1262 | 30 | 99% |
| 1850 | 27 | 90% |
| 2438 | 13 | 34% |
| 3026 | 25 | 72% |
| 4874 | 32 | 55% |
| 7562 | 32 | 35% |

At 7,562 tokens the document holds 90 SSNs and the model returns 32 spans covering the first 35% of
the text — no error, no warning, a well-formed result. At 12,612 tokens with a single label it
returned zero spans, silently; longer inputs OOM'd the card inside DeBERTa's quadratic attention.

The filler is synthetic repetitive prose, so **the percentages are directional, not a benchmark** —
which is exactly why they are not published as one. The failure MODE is what matters and is not in
doubt: overflow is silent, partial and non-monotonic (2,438 scored worse than 3,026), so no
threshold test catches it reliably, and useful recall ends far below 4096.

The framework's response is in ADR-0064: 1024 tokens is the decided chunk budget, the Providers'
`max_len` is derived from it, raising it past the safe window is warned about on extraction and
**refused** on redaction, and a chunk too long for the window is refused rather than silently
truncated. If you raise any of those knobs, this table is what you are trading against.

The redaction measurement is the one that matters, because **there is no GLiNER 2.5 PII
checkpoint**: `redaction.gliner25` asks the *general* 2.5 checkpoint for PII labels zero-shot. Until
that number exists, no parity with the task-trained `redaction.gliner2` checkpoint is claimed
anywhere — the Provider's own `ProviderProfile.perf_note` says so and a test asserts it does — and
the ADR-0044 deterministic financial-PII floor is what holds underneath either backend regardless.

## Reading the extraction and disambiguation results honestly

All three RED^FM suites pin the framework's **dependency-free CPU built-ins**, for the same
reason Parse pins `parser.pdfplumber`: a published number must be the one you reproduce from a
clean clone. That has consequences you should read off the tables the run writes rather than be
surprised by.

* **Entity Extraction** pins `entity.gazetteer` with a capitalisation + numeric recogniser. It is
  an entity *detection* baseline: it has no notion of RED^FM's label inventory beyond `NUMBER`,
  so `span+type` sits far below `span`, and Arabic and Chinese — which have no case distinction —
  score near zero **by construction**. That is why the suite reports a per-language slice rather
  than one aggregate. The learned re-pin is `--provider entity.gliner`.
* **Relation Extraction** pins `relation.pattern` with five type-prior co-occurrence rules whose
  head/tail signatures were derived from RED^FM's **train** split (never the scored `test`
  split). A co-occurrence rule fires on every in-window type pair, so recall is respectable and
  precision is the honest weakness of a rule baseline. It is scored from **gold mentions**, so
  the number measures relation extraction rather than the entity extractor upstream of it. The
  learned re-pin is `--provider relation.llm` — the only learned Provider that satisfies the
  standalone `RelationExtractor` seam this Stage resolves. The learned gliner relation
  extractors are **Fused** Providers (ADR-0036): they belong under a `fused_entity_relation`
  Stage and cannot be re-pinned into this one.
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

The mirror of that hazard is just as real, and a `MetricScore` only defends against it if the
suite counts both halves in the same unit. The redaction suite scores *containment* rather than
boundary equality, so a wider mask still counts as having masked the value; if it also credited
one true positive per value swallowed while charging false positives per emitted span, a Provider
that masked whole documents — destroying every ounce of corpus utility — would post perfect
recall for free and outrank a real redactor. `coverage` therefore aligns detections to values
**one-to-one** (a maximum matching under containment, the same multiset rule `confusion` applies
to exact spans): a mask is credited for at most one value and answers for every span it emits, so
predicting everything is as visible a failure as predicting nothing.

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

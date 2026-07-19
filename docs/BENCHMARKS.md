# Public benchmarks

Reproducible quality numbers for the Latence framework, produced by the harness in
`packages/latence-benchmark`. Every number in the **Our measurements** section was
produced by running the framework's real Stages over a public dataset pinned to an
exact commit; every number in **Vendor claims** was copied from someone else's
publication and is labelled as a claim, not a measurement.

_Generated 2026-07-19._ Covering every Stage takes more than one run — the RED^FM
suites need a download, Parse needs olmOCR-bench, the redaction corpus ships in this
package — so **each section states the date its own numbers were measured**. A Stage
with no section here has not been measured; it is never implied by silence.

## Parse — `parse` suite

_Measured 2026-07-18._

### Dataset (license verified before inclusion)

| Field | Value |
| --- | --- |
| Dataset | [`allenai/olmOCR-bench`](https://huggingface.co/datasets/allenai/olmOCR-bench) |
| Pinned revision (commit) | `54a96a6fb6a2bd3b297e59869491db4d3625b711` |
| Corpus license | **ODC-BY-1.0** |
| Evaluation-toolkit license | **Apache-2.0** |
| Commercial use permitted | yes |
| Attribution required | yes |
| Verified on / at | 2026-07-18 — <https://huggingface.co/datasets/allenai/olmOCR-bench> |

> Corpus license read from the dataset card front-matter (`license: odc-by`), which states: 'This dataset is licensed under ODC-BY-1.0. It is intended for research and educational use in accordance with AI2's Responsible Use Guidelines.' ODC-BY-1.0 is permissive-with-attribution and does not restrict commercial use. The evaluation toolkit (github.com/allenai/olmocr) is Apache-2.0, verified separately per ADR-0012.

Attribution: olmOCR-bench, Allen Institute for AI, ODC-BY-1.0 — https://huggingface.co/datasets/allenai/olmOCR-bench

### Stack pin

| Field | Value |
| --- | --- |
| Stage | `parse` |
| Provider | `parser.pdfplumber` |
| Config | `{}` |
| Pin fingerprint | `4c438beb9b912e86` |

### Our measurements

| Split | Pass rate | Passed | Failed | Scored | Unsupported |
| --- | --- | --- | --- | --- | --- |
| `arxiv_math` | n/a | 0 | 0 | 0 | 2927 |
| `headers_footers` | 44.5% | 338 | 422 | 760 | 0 |
| `long_tiny_text` | 7.5% | 33 | 409 | 442 | 0 |
| `multi_column` | 9.3% | 82 | 802 | 884 | 0 |
| `old_scans` | 13.3% | 70 | 456 | 526 | 0 |
| `old_scans_math` | n/a | 0 | 0 | 0 | 458 |
| `table_tests` | 50.0% | 1 | 1 | 2 | 1020 |
| **OVERALL** | **20.0%** | 524 | 2090 | 2614 | 4405 |

Documents parsed: 1403 (parse failures: 7).

**Scored test classes:** absent, baseline, order, present. `table` and `math` assertions are counted
in the *Unsupported* column and excluded from the pass-rate denominator — they
are neither passes nor failures here. Scoring them remains UNIMPLEMENTED (issue
#187 added the Entity/Relation, Disambiguation and Redaction suites through the
same seam, but did not implement the `table`/`math` assertion classes). This is
why the OVERALL figure is **not** directly comparable to the vendor claims,
which are computed over all five classes.

**Run notes:**

- Scored with Provider 'parser.pdfplumber' exactly as pinned. A text-layer PDF parser cannot recover text from a scanned page, and much of olmOCR-bench (old_scans, long_tiny_text) is photographed or scanned material with no text layer — those tests fail by construction, not by defect. Comparing an OCR Provider means re-pinning: `--provider parser.lighton` (GPU).

### Reproduce

```bash
latence-benchmark run --suite parse --provider parser.pdfplumber
```

Determinism: the pinned Provider is CPU-only and deterministic, documents are
visited in a fixed order and splits/language slices in a fixed order, and the
dataset is pinned to a full 40-hex revision — an upstream commit SHA, or a content
digest for a corpus shipped in this repository (branch/tag pins are rejected at
construction, and a shipped corpus whose bytes no longer match its digest is
refused). Two runs of the same revision and pin fingerprint produce identical
numbers.

### Vendor claims — NOT measured by us

The following are **published claims** transcribed from their sources. We did
not run these systems, cannot vouch for the configurations behind the numbers,
and they are computed over all five olmOCR-bench test classes. They are listed
for orientation only and must not be read as a head-to-head against the
measurements above.

| System | Claimed value | Metric | Source | Retrieved |
| --- | --- | --- | --- | --- |
| Chandra OCR 0.1.0 | 83.1 ± 0.9 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |
| olmOCR v0.4.0 | 82.4 ± 1.1 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |
| PaddleOCR-VL | 80.0 ± 1.0 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |
| Marker 1.10.1 | 76.1 ± 1.1 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |
| DeepSeek-OCR | 75.7 ± 1.0 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |
| MinerU 2.5.4 | 75.2 ± 1.1 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |
| Mistral OCR API | 72.0 ± 1.1 | Overall (all 5 test classes, olmOCR harness) | <https://github.com/allenai/olmocr> | 2026-07-18 |

## Redaction — `redaction` suite

_Measured 2026-07-19._

### Dataset (license verified before inclusion)

| Field | Value |
| --- | --- |
| Dataset | [`latence-framework/latence_benchmark.corpora`](https://github.com/ddickmann/latence-framework) |
| Pinned revision (content-digest) | `74c22ebeb6a98e7c8c057b065c1cb4fe31d69ef0` |
| Corpus license | **Apache-2.0** |
| Evaluation-toolkit license | **Apache-2.0** |
| Commercial use permitted | yes |
| Attribution required | yes |
| Verified on / at | 2026-07-18 — <https://github.com/ddickmann/latence-framework/blob/main/LICENSE> |

> Corpus and generator are both original work in this repository and carry this repository's own Apache-2.0 license — the one case where corpus and toolkit licenses coincide, verified rather than assumed (ADR-0012). The corpus contains NO real personal data: every value is a reserved or published example (RFC 2606 example domains, RFC 5737 TEST-NET addresses, 555 fiction-reserved phone numbers, the 4111... test card, AWS's AKIAIOSFODNN7EXAMPLE docs key, published example IBANs, and fictional names). It is inspectable in full in the repository.

Attribution: Latence synthetic PII corpus (Apache-2.0) — part of the latence-framework repo.

### Stack pin

| Field | Value |
| --- | --- |
| Stage | `redaction` |
| Provider | `redaction.hybrid_rule` |
| Config | `{}` |
| Pin fingerprint | `ea1d8b03b23d9a0d` |

### Our measurements

**Headline (micro over language):** precision 100.0% · recall 77.1% · F1 87.1% (27 TP / 0 FP / 8 FN).

#### By language

| Slice | Precision | Recall | F1 | TP | FP | FN | Gold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `en` | 100.0% | 77.8% | 87.5% | 14 | 0 | 4 | 18 |
| `de` | 100.0% | 76.5% | 86.7% | 13 | 0 | 4 | 17 |
| **MICRO** | **100.0%** | **77.1%** | **87.1%** | 27 | 0 | 8 | 35 |

#### By PII type

| Slice | Precision | Recall | F1 | TP | FP | FN | Gold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `api_key` | 100.0% | 100.0% | 100.0% | 2 | 0 | 0 | 2 |
| `aws_key` | 100.0% | 100.0% | 100.0% | 2 | 0 | 0 | 2 |
| `credit_card` | 100.0% | 100.0% | 100.0% | 2 | 0 | 0 | 2 |
| `date_of_birth` | n/a | 0.0% | n/a | 0 | 0 | 2 | 2 |
| `email` | 100.0% | 100.0% | 100.0% | 5 | 0 | 0 | 5 |
| `iban` | 100.0% | 100.0% | 100.0% | 3 | 0 | 0 | 3 |
| `ip` | 100.0% | 100.0% | 100.0% | 2 | 0 | 0 | 2 |
| `jwt` | 100.0% | 100.0% | 100.0% | 2 | 0 | 0 | 2 |
| `national_id` | n/a | 0.0% | n/a | 0 | 0 | 1 | 1 |
| `person_name` | n/a | 0.0% | n/a | 0 | 0 | 6 | 6 |
| `phone` | 50.0% | 100.0% | 66.7% | 3 | 3 | 0 | 3 |
| `postal_address` | n/a | 0.0% | n/a | 0 | 0 | 2 | 2 |
| `secret` | 100.0% | 100.0% | 100.0% | 2 | 0 | 0 | 2 |
| `ssn` | 100.0% | 100.0% | 100.0% | 1 | 0 | 0 | 1 |
| **MICRO** | **88.9%** | **68.6%** | **77.4%** | 24 | 3 | 11 | 35 |

#### By match criterion

| Slice | Precision | Recall | F1 | TP | FP | FN | Gold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `coverage (value fully masked)` | 100.0% | 77.1% | 87.1% | 27 | 0 | 8 | 35 |
| `exact span` | 92.6% | 71.4% | 80.6% | 25 | 2 | 10 | 35 |
| `exact span+type` | 81.5% | 62.9% | 71.0% | 22 | 5 | 13 | 35 |

These rows are **alternative scoring criteria over the same gold items**, not disjoint slices of the corpus, so there is deliberately no aggregate row: summing them would count every gold item once per criterion and report a denominator several times the size of the gold set. Read each criterion on its own.

Documents scored: 20.

A precision or recall with no denominator is reported as `n/a`, never as 0% — an undefined ratio is not a measured failure.

**Run notes:**

- Scored with Provider 'redaction.hybrid_rule' at its DEFAULT config: no `detectors` key, which selects the comprehensive built-in recogniser set. Narrowing that default would show up here as a recall drop.
- The corpus is SYNTHETIC — no real personal data is in this repository. Every value is a reserved or published example (see the DatasetSpec license notes).
- The gold set deliberately includes person names and postal addresses, which a rule-based redactor structurally cannot detect. The recall below is therefore NOT expected to be 1.0, and the per-type table shows exactly where the residual is. The learned re-pin for names is `--provider redaction.gliner_pii`.
- OVER-redaction is charged, not only under-redaction: `coverage` credits ONE detection per gold value (a maximum one-to-one alignment under containment), so a Provider cannot buy recall by widening its masks — every span it emits that is not matched to a value is a false positive. Masking everything and masking nothing are both visible failures here.
- Per-type slices count a detected span as a false positive for its own type when it covers no gold value; a value masked under a different label counts as a miss for its gold type (and appears under the `exact span+type` criterion), because a type-scoped redaction policy would not have covered it.

### Reproduce

```bash
latence-benchmark run --suite redaction --provider redaction.hybrid_rule
```

Determinism: the pinned Provider is CPU-only and deterministic, documents are
visited in a fixed order and splits/language slices in a fixed order, and the
dataset is pinned to a full 40-hex revision — an upstream commit SHA, or a content
digest for a corpus shipped in this repository (branch/tag pins are rejected at
construction, and a shipped corpus whose bytes no longer match its digest is
refused). Two runs of the same revision and pin fingerprint produce identical
numbers.

## Adding another Stage suite

The harness is built around one seam, `latence_benchmark.suite.BenchmarkSuite`. A new
suite supplies a name, the Stage it measures, a pinned `DatasetSpec` (full commit SHA
plus a separately verified corpus/toolkit license record, ADR-0012), a `StackPin`, and
a `run(root)`; registering it makes it available to the CLI and to this page with no
other change. See the module docstring of `latence_benchmark/suite.py`.

This is the framework-side *public* benchmark. The adopter-side gold-set harness
(ADR-0028), which scores a customer's own corpus, is a separate thing and stays so.


# Public benchmarks

Reproducible quality numbers for the Latence framework, produced by the harness in
`packages/latence-benchmark`. Every number in the **Our measurements** section was
produced by running the framework's real Stages over a public dataset pinned to an
exact commit; every number in **Vendor claims** was copied from someone else's
publication and is labelled as a claim, not a measurement.

_Generated 2026-07-18._

## Parse — `parse` suite

### Dataset (license verified before inclusion)

| Field | Value |
| --- | --- |
| Dataset | [`allenai/olmOCR-bench`](https://huggingface.co/datasets/allenai/olmOCR-bench) |
| Pinned revision | `54a96a6fb6a2bd3b297e59869491db4d3625b711` |
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
in the *Unsupported* column and excluded from the pass-rate denominator — they are
neither passes nor failures here. Issue #187 implements them through the same
suite seam. This is why the OVERALL figure below is **not** directly comparable to
the vendor claims, which are computed over all five classes.

### Reproduce

```bash
latence-benchmark run --suite parse --provider parser.pdfplumber
```

Determinism: the pinned Provider is CPU-only and deterministic, documents are
visited in sorted path order and splits in a fixed order, and the dataset is pinned
to a full commit SHA (branch/tag pins are rejected at construction). Two runs of
the same revision and pin fingerprint produce identical tallies.

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

## Adding another Stage suite

The harness is built around one seam, `latence_benchmark.suite.BenchmarkSuite`. A new
suite supplies a name, the Stage it measures, a pinned `DatasetSpec` (full commit SHA
plus a separately verified corpus/toolkit license record, ADR-0012), a `StackPin`, and
a `run(root)`; registering it makes it available to the CLI and to this page with no
other change. See the module docstring of `latence_benchmark/suite.py`.

This is the framework-side *public* benchmark. The adopter-side gold-set harness
(ADR-0028), which scores a customer's own corpus, is a separate thing and stays so.


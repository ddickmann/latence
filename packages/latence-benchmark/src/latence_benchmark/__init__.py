"""``latence-benchmark`` — the reproducible PUBLIC benchmark harness.

Converts the project's anti-false-green rigour into something an outsider can verify: one
command runs a framework Stage over a public dataset pinned to an exact commit, under a pinned
stack config, and publishes both a human-readable results page and a machine-readable sidecar.

This is *not* the adopter gold-set harness of ADR-0028 — that one scores a customer's own
corpus with their own annotations and stays adopter-side. This one scores the framework itself
against public data, so the numbers can be reproduced by anyone.

It registers nothing under ``latence.providers``: it is a consumer of the pipeline, not a part
of it. The Parse suite is the tracer slice; :mod:`latence_benchmark.suite` is the seam the
remaining Stage suites (#187) plug into.
"""

from __future__ import annotations

from latence_benchmark.datasets import (
    DATASETS,
    DatasetSpec,
    DatasetUnavailableError,
    HuggingFaceDatasetSource,
    LocalDatasetSource,
    get_dataset,
)
from latence_benchmark.licensing import UNVERIFIED, DatasetLicense, LicenseGateError, require_usable
from latence_benchmark.parse_suite import ParseSuite, parse_suite
from latence_benchmark.results import VendorClaim, render_json, render_markdown, write_results
from latence_benchmark.scoring import Outcome, TestResult, fuzzy_contains, fuzzy_find, score
from latence_benchmark.suite import (
    BenchmarkSuite,
    SplitScore,
    StackPin,
    SuiteResult,
    get_suite,
    register_suite,
    suite_names,
)
from latence_benchmark.unittests import BenchmarkDataError, UnitTest, UnitTestType

__all__ = [
    "DATASETS",
    "UNVERIFIED",
    "BenchmarkDataError",
    "BenchmarkSuite",
    "DatasetLicense",
    "DatasetSpec",
    "DatasetUnavailableError",
    "HuggingFaceDatasetSource",
    "LicenseGateError",
    "LocalDatasetSource",
    "Outcome",
    "ParseSuite",
    "SplitScore",
    "StackPin",
    "SuiteResult",
    "TestResult",
    "UnitTest",
    "UnitTestType",
    "VendorClaim",
    "fuzzy_contains",
    "fuzzy_find",
    "get_dataset",
    "get_suite",
    "parse_suite",
    "register_suite",
    "render_json",
    "render_markdown",
    "score",
    "suite_names",
    "write_results",
]

# The Parse tracer slice is registered at import time; #187's suites append here.
register_suite(ParseSuite())

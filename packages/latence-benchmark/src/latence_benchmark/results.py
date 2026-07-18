"""Rendering the published results page — and keeping measurements apart from claims.

The single rule this module enforces is that a number we **measured** and a number a vendor
**published** never share a table. Our rows carry the dataset revision, the stack fingerprint
and the command that reproduces them; vendor rows carry a source URL and a retrieval date and
are labelled ``CLAIM``, because we did not run them and cannot vouch for the configuration
behind them. :class:`VendorClaim` has no field for a pass rate we computed, and
:class:`~latence_benchmark.suite.SuiteResult` has no field for a citation — the separation is
in the types, not in the prose.

Both artifacts are written from the same data: a human-readable markdown page for ``docs/`` and
a machine-readable JSON sidecar so a later run can be diffed against a published one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_benchmark.scoring import SUPPORTED_TYPES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from latence_benchmark.suite import SplitScore, SuiteResult


@dataclass(frozen=True)
class VendorClaim:
    """A benchmark number **published by someone else**. Never measured here.

    :param system: The system the claim is about.
    :param dataset: Which benchmark the claim refers to (must match to be comparable at all).
    :param metric: The metric name exactly as the source states it.
    :param value: The value exactly as the source states it, as a string — including any
        uncertainty the source gives, which a float would silently discard.
    :param source_url: Where the claim was read.
    :param retrieved_on: ISO date the claim was read.
    """

    system: str
    dataset: str
    metric: str
    value: str
    source_url: str
    retrieved_on: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "dataset": self.dataset,
            "metric": self.metric,
            "value": self.value,
            "source_url": self.source_url,
            "retrieved_on": self.retrieved_on,
            "kind": "vendor-claim",
        }


_OLMOCR_SOURCE = "https://github.com/allenai/olmocr"

#: Published olmOCR-bench overall scores, transcribed verbatim from the olmocr README's
#: leaderboard table on the date below. NOT reproduced by us, NOT comparable to our rows
#: without matching the harness and the scored test classes.
VENDOR_CLAIMS: tuple[VendorClaim, ...] = tuple(
    VendorClaim(
        system=system,
        dataset="olmocr-bench",
        metric="Overall (all 5 test classes, olmOCR harness)",
        value=value,
        source_url=_OLMOCR_SOURCE,
        retrieved_on="2026-07-18",
    )
    for system, value in (
        ("Chandra OCR 0.1.0", "83.1 ± 0.9"),
        ("olmOCR v0.4.0", "82.4 ± 1.1"),
        ("PaddleOCR-VL", "80.0 ± 1.0"),
        ("Marker 1.10.1", "76.1 ± 1.1"),
        ("DeepSeek-OCR", "75.7 ± 1.0"),
        ("MinerU 2.5.4", "75.2 ± 1.1"),
        ("Mistral OCR API", "72.0 ± 1.1"),
    )
)


def _rate(score: SplitScore) -> str:
    rate = score.pass_rate
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _reproduce_command(result: SuiteResult) -> str:
    return (
        f"latence-benchmark run --suite {result.suite} "
        f"--provider {result.stack_pin.provider}"
    )


def render_markdown(results: Sequence[SuiteResult], *, generated_on: str) -> str:
    """Render the published results page for one or more suite results."""
    supported = ", ".join(sorted(t.value for t in SUPPORTED_TYPES))
    lines: list[str] = [
        "# Public benchmarks",
        "",
        "Reproducible quality numbers for the Latence framework, produced by the harness in",
        "`packages/latence-benchmark`. Every number in the **Our measurements** section was",
        "produced by running the framework's real Stages over a public dataset pinned to an",
        "exact commit; every number in **Vendor claims** was copied from someone else's",
        "publication and is labelled as a claim, not a measurement.",
        "",
        f"_Generated {generated_on}._",
        "",
    ]

    for result in results:
        licence = result.dataset.license
        lines += [
            f"## {result.stage.title()} — `{result.suite}` suite",
            "",
            "### Dataset (license verified before inclusion)",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Dataset | [`{result.dataset.repo_id}`]({result.dataset.homepage}) |",
            f"| Pinned revision | `{result.dataset.revision}` |",
            f"| Corpus license | **{licence.dataset_license}** |",
            f"| Evaluation-toolkit license | **{licence.code_license}** |",
            f"| Commercial use permitted | {'yes' if licence.commercial_use else 'NO'} |",
            f"| Attribution required | {'yes' if licence.attribution_required else 'no'} |",
            f"| Verified on / at | {licence.verified_on} — <{licence.source_url}> |",
            "",
            f"> {licence.notes}",
            "",
            f"Attribution: {result.dataset.citation}",
            "",
            "### Stack pin",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Stage | `{result.stack_pin.stage}` |",
            f"| Provider | `{result.stack_pin.provider}` |",
            f"| Config | `{json.dumps(dict(result.stack_pin.config), sort_keys=True)}` |",
            f"| Pin fingerprint | `{result.stack_pin.fingerprint}` |",
            "",
            "### Our measurements",
            "",
            "| Split | Pass rate | Passed | Failed | Scored | Unsupported |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for split in result.splits:
            lines.append(
                f"| `{split.split}` | {_rate(split)} | {split.passed} | {split.failed} | "
                f"{split.scored} | {split.unsupported} |"
            )
        overall = result.overall
        lines += [
            f"| **OVERALL** | **{_rate(overall)}** | {overall.passed} | {overall.failed} | "
            f"{overall.scored} | {overall.unsupported} |",
            "",
            f"Documents parsed: {result.documents_scored} "
            f"(parse failures: {result.document_failures}).",
            "",
            f"**Scored test classes:** {supported}. `table` and `math` assertions are counted",
            "in the *Unsupported* column and excluded from the pass-rate denominator — they are",
            "neither passes nor failures here. Issue #187 implements them through the same",
            "suite seam. This is why the OVERALL figure below is **not** directly comparable to",
            "the vendor claims, which are computed over all five classes.",
            "",
        ]
        if result.notes:
            lines.append("**Run notes:**")
            lines.append("")
            lines += [f"- {note}" for note in result.notes]
            lines.append("")
        lines += [
            "### Reproduce",
            "",
            "```bash",
            f"{_reproduce_command(result)}",
            "```",
            "",
            "Determinism: the pinned Provider is CPU-only and deterministic, documents are",
            "visited in sorted path order and splits in a fixed order, and the dataset is pinned",
            "to a full commit SHA (branch/tag pins are rejected at construction). Two runs of",
            "the same revision and pin fingerprint produce identical tallies.",
            "",
        ]

        claims = [c for c in VENDOR_CLAIMS if c.dataset == result.dataset.name]
        if claims:
            lines += [
                "### Vendor claims — NOT measured by us",
                "",
                "The following are **published claims** transcribed from their sources. We did",
                "not run these systems, cannot vouch for the configurations behind the numbers,",
                "and they are computed over all five olmOCR-bench test classes. They are listed",
                "for orientation only and must not be read as a head-to-head against the",
                "measurements above.",
                "",
                "| System | Claimed value | Metric | Source | Retrieved |",
                "| --- | --- | --- | --- | --- |",
            ]
            for claim in claims:
                lines.append(
                    f"| {claim.system} | {claim.value} | {claim.metric} | "
                    f"<{claim.source_url}> | {claim.retrieved_on} |"
                )
            lines.append("")

    lines += [
        "## Adding another Stage suite",
        "",
        "The harness is built around one seam, `latence_benchmark.suite.BenchmarkSuite`. A new",
        "suite supplies a name, the Stage it measures, a pinned `DatasetSpec` (full commit SHA",
        "plus a separately verified corpus/toolkit license record, ADR-0012), a `StackPin`, and",
        "a `run(root)`; registering it makes it available to the CLI and to this page with no",
        "other change. See the module docstring of `latence_benchmark/suite.py`.",
        "",
        "This is the framework-side *public* benchmark. The adopter-side gold-set harness",
        "(ADR-0028), which scores a customer's own corpus, is a separate thing and stays so.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_json(results: Sequence[SuiteResult], *, generated_on: str) -> str:
    """Render the machine-readable results sidecar."""
    payload = {
        "generated_on": generated_on,
        "measurements": [r.as_dict() for r in results],
        "vendor_claims": [c.as_dict() for c in VENDOR_CLAIMS],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_results(
    results: Sequence[SuiteResult],
    *,
    markdown_path: Path,
    json_path: Path,
    generated_on: str,
) -> None:
    """Write both published artifacts."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(results, generated_on=generated_on), encoding="utf-8")
    json_path.write_text(render_json(results, generated_on=generated_on), encoding="utf-8")

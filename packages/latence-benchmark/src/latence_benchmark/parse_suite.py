"""The Parse benchmark suite — the tracer slice through the whole harness.

Runs the framework's real Parse Stage, through the real ``latence.providers`` entry-point
registry, over every PDF the pinned dataset's unit tests refer to, and scores the emitted
markdown with :mod:`latence_benchmark.scoring`.

Three properties are deliberate:

* **The real Stage, not a re-implementation.** The Provider is resolved by name from the same
  registry the Runner uses and instantiated the same way (``provider_cls(config)``), so the
  numbers describe the Parse Stage an adopter actually runs. The pin is validated through
  core's own :class:`~latence_core.pipeline.Stage` model, so a typo'd Provider name or a
  capability mismatch fails here rather than producing a mystery zero.
* **A parse failure is a failure.** A ``PARSE_ERROR`` document's tests are scored as failed,
  not skipped. Skipping them would let a parser improve its score by crashing.
* **Determinism.** Documents are visited in sorted path order and splits in a fixed order, so
  two runs of the same revision + pin produce byte-identical results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_core.capability import Parser, ProviderRegistry
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    ParserInput,
    Provenance,
)
from latence_core.pipeline import CapabilityKind

from latence_benchmark.datasets import OLMOCR_BENCH, DatasetSpec
from latence_benchmark.providers import ProviderUnavailableError, load_provider
from latence_benchmark.scoring import Outcome, TestResult, score
from latence_benchmark.suite import SplitScore, StackPin, SuiteResult
from latence_benchmark.unittests import load_split

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from latence_benchmark.unittests import UnitTest

__all__ = [
    "OLMOCR_SPLITS",
    "PARSE_STACK_PIN",
    "ParseSuite",
    # Re-exported: the error moved to `latence_benchmark.providers` when the four #187 suites
    # started needing the same Provider-resolution rules, and callers still catch it from here.
    "ProviderUnavailableError",
    "parse_suite",
    "split_names",
]

#: Fixed split order — determinism, and it is also the order the results page reads best in.
OLMOCR_SPLITS: tuple[str, ...] = (
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
)

#: The pinned stack config the published Parse numbers are produced with. ``parser.pdfplumber``
#: is the CPU-first, offline, permissive (MIT) default an adopter gets out of the box — so the
#: published number is the number they will reproduce, not a number from a GPU stack they do
#: not have. A GPU OCR Provider is a one-line pin change (``--provider parser.lighton``), which
#: is exactly how the comparison should be run.
PARSE_STACK_PIN = StackPin(stage="parse", provider="parser.pdfplumber", config={})


def _parser_input(path: Path, root: Path) -> ParserInput:
    """Build the Parse Stage's input contract for one benchmark PDF."""
    content = path.read_bytes()
    return ParserInput(
        provenance=Provenance(
            source_uri=str(path),
            file_name=path.name,
            file_type=path.suffix.lstrip(".").lower() or "pdf",
            file_size=len(content),
            document_id=str(path.relative_to(root)),
            source_system="olmOCR-bench",
        ),
        classification=Classification(language="und", sensitivity="public"),
        content=content,
    )


@dataclass(frozen=True)
class ParseSuite:
    """Benchmarks the Parse Stage on a pinned public document-parsing benchmark."""

    name: str = "parse"
    stage: str = "parse"
    dataset: DatasetSpec = OLMOCR_BENCH
    stack_pin: StackPin = PARSE_STACK_PIN
    splits: tuple[str, ...] = OLMOCR_SPLITS
    registry: ProviderRegistry = field(default_factory=ProviderRegistry)

    def _load_parser(self) -> Parser:
        """Resolve and instantiate the pinned Parse Provider through the real registry."""
        return load_provider(
            self.stack_pin,
            capability=CapabilityKind.PARSE,
            protocol=Parser,
            registry=self.registry,
        )

    def _parse(self, parser: Parser, path: Path, root: Path) -> DocumentRecord:
        """Parse one document, turning any Provider exception into a PARSE_ERROR record."""
        parser_input = _parser_input(path, root)
        try:
            records = list(parser.parse([parser_input]))
        except Exception as exc:  # noqa: BLE001 - one bad file never aborts a benchmark run
            return DocumentRecord(
                record_id=parser_input.provenance.document_id,
                provenance=parser_input.provenance,
                classification=parser_input.classification,
                disposition=Disposition.PARSE_ERROR,
                error=f"{type(exc).__name__}",
            )
        if not records:
            return DocumentRecord(
                record_id=parser_input.provenance.document_id,
                provenance=parser_input.provenance,
                classification=parser_input.classification,
                disposition=Disposition.PARSE_ERROR,
                error="Provider emitted no DocumentRecord",
            )
        return records[0]

    def run(self, root: Path, *, limit: int | None = None) -> SuiteResult:
        """Score the Parse Stage over ``root``, a materialised dataset directory.

        :param limit: Optional cap on documents scored **per split**. A capped run is a
            different measurement, so it is recorded as a prominent note on the result and must
            never be compared against full-benchmark numbers.
        """
        parser = self._load_parser()
        pdf_root = root / "bench_data" / "pdfs"
        split_scores: list[SplitScore] = []
        document_failures = 0
        documents_scored = 0
        missing_pdfs = 0

        for split in self.splits:
            tests = load_split(root, split)
            by_pdf: dict[str, list[UnitTest]] = {}
            for test in tests:
                by_pdf.setdefault(test.pdf, []).append(test)
            pdfs = sorted(by_pdf)
            if limit is not None:
                pdfs = pdfs[:limit]

            results: list[TestResult] = []
            for pdf in pdfs:
                path = pdf_root / pdf
                split_tests = sorted(by_pdf[pdf], key=lambda t: t.id)
                if not path.is_file():
                    # The manifest guarantees the unit-test files, not every referenced PDF
                    # (the pdfs/ tree can be fetched selectively). A referenced-but-absent PDF
                    # is counted as a failure and surfaced — never silently dropped.
                    missing_pdfs += 1
                    results.extend(
                        TestResult(t.id, split, t.type, Outcome.FAILED, "referenced PDF missing")
                        for t in split_tests
                    )
                    continue
                record = self._parse(parser, path, pdf_root)
                documents_scored += 1
                if record.disposition is Disposition.PARSE_ERROR:
                    document_failures += 1
                    results.extend(
                        TestResult(
                            t.id, split, t.type, Outcome.FAILED, f"parse failed: {record.error}"
                        )
                        for t in split_tests
                    )
                    continue
                results.extend(score(t, record.content, split=split) for t in split_tests)
            split_scores.append(SplitScore.from_results(split, results))

        notes: list[str] = [
            f"Scored with Provider {self.stack_pin.provider!r} exactly as pinned. A text-layer "
            "PDF parser cannot recover text from a scanned page, and much of olmOCR-bench "
            "(old_scans, long_tiny_text) is photographed or scanned material with no text "
            "layer — those tests fail by construction, not by defect. Comparing an OCR "
            "Provider means re-pinning: `--provider parser.lighton` (GPU).",
        ]
        if limit is not None:
            notes.append(
                f"SUBSET RUN — capped at {limit} document(s) per split. These numbers are NOT "
                "comparable with full-benchmark results, including the vendor claims."
            )
        if missing_pdfs:
            notes.append(
                f"{missing_pdfs} referenced PDF(s) were absent from the dataset root and their "
                "tests are counted as FAILED. Fetch the full bench_data/pdfs tree for a "
                "comparable number."
            )
        return SuiteResult(
            suite=self.name,
            stage=self.stage,
            dataset=self.dataset,
            stack_pin=self.stack_pin,
            splits=tuple(split_scores),
            document_failures=document_failures,
            documents_scored=documents_scored,
            notes=tuple(notes),
        )


def parse_suite(provider: str | None = None, config: dict[str, Any] | None = None) -> ParseSuite:
    """Build a :class:`ParseSuite`, optionally re-pinning the Parse Provider.

    Re-pinning is how a comparison is run (``--provider parser.lighton``); the new pin's
    fingerprint lands in the result, so a reader can always tell two runs apart.
    """
    if provider is None and config is None:
        return ParseSuite()
    pin = StackPin(
        stage="parse",
        provider=provider or PARSE_STACK_PIN.provider,
        config=config if config is not None else dict(PARSE_STACK_PIN.config),
    )
    return ParseSuite(stack_pin=pin)


def split_names(splits: Sequence[str] | None = None) -> tuple[str, ...]:
    """Normalise a user-supplied split selection against the pinned split list."""
    if not splits:
        return OLMOCR_SPLITS
    unknown = [s for s in splits if s not in OLMOCR_SPLITS]
    if unknown:
        msg = f"Unknown split(s) {unknown}. Available: {', '.join(OLMOCR_SPLITS)}"
        raise ValueError(msg)
    return tuple(s for s in OLMOCR_SPLITS if s in set(splits))

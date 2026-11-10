"""What ``latence tune --sample N`` is, and what the tune report says its figures are.

``--sample`` was documented as "Records to time each candidate on (the small sample)", and the
report called its figures ``records/sec``. Neither is what the code does. The probe never reads
the number: every candidate is timed on one run of the whole stack over the corpus bundled for
stack validation. The tuner then divides ``N`` by the median wall time of those runs. So ``N`` is
the numerator of the reported figure, the figure compares settings on one device, and it is not a
count of records per second.

The first tests pin the wording, loosely: they require the texts to say what the number is and
forbid the two claims that were false. The last two pin the behaviour the wording describes, so
that a change to the behaviour (real sampling, say) turns a test red and the wording is revisited.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import typer.main
from latence_core.cli import app
from latence_core.tune.fingerprint import DeviceFingerprint
from latence_core.tune.probe import ProbeOutcome
from latence_core.tune.realprobe import RegistrySampleProbe
from latence_core.tune.report import KnobSetting, ProviderTuning, TuneReport
from latence_core.tune.tuner import Tuner, _throughput

_REPO_ROOT = Path(__file__).resolve().parents[3]

# The two claims that were false: that N records are timed, and that the run is cut down to N.
_CLAIMS_A_SAMPLE_IS_TAKEN = re.compile(
    r"records? to time|timed each candidate on|small (real )?sample|limit(s|ed)? (the run )?to",
    re.IGNORECASE,
)
_CLAIMS_RECORDS_PER_SECOND = re.compile(r"records ?/ ?sec|\br/s\b", re.IGNORECASE)


def _option_help(command: str, option: str) -> str:
    group = typer.main.get_command(app)
    params = group.commands[command].params  # type: ignore[attr-defined]
    (param,) = [p for p in params if option in p.opts]
    text = param.help
    assert isinstance(text, str) and text
    return " ".join(text.split())


def _one_line(text: str | None) -> str:
    assert text
    return " ".join(text.split())


def test_the_sample_help_says_what_the_number_is() -> None:
    text = _option_help("tune", "--sample")

    assert not _CLAIMS_A_SAMPLE_IS_TAKEN.search(text), text
    # What it is: the numerator of the figure, over the time of a whole-stack run.
    assert "divided by" in text
    assert "whole stack" in text
    assert "validation corpus" in text
    # What it is not.
    assert "does not" in text and "records" in text


def test_the_repeats_help_does_not_promise_records_per_second() -> None:
    text = _option_help("tune", "--repeats")
    assert not _CLAIMS_RECORDS_PER_SECOND.search(text), text
    assert "median" in text


def test_the_tune_command_help_does_not_claim_a_sample_is_taken() -> None:
    group = typer.main.get_command(app)
    text = _one_line(group.commands["tune"].help)  # type: ignore[attr-defined]
    assert not _CLAIMS_A_SAMPLE_IS_TAKEN.search(text), text
    assert "validation corpus" in text


def test_the_report_fields_say_what_the_figures_are() -> None:
    sample = _one_line(TuneReport.model_fields["sample_size"].description)
    assert not _CLAIMS_A_SAMPLE_IS_TAKEN.search(sample), sample
    assert "numerator" in sample

    for name in ("baseline_throughput", "winner_throughput"):
        text = _one_line(ProviderTuning.model_fields[name].description)
        assert not _CLAIMS_RECORDS_PER_SECOND.search(text), text
        assert "sample_size" in text and "median" in text


def test_the_rendered_report_says_how_to_read_its_figures() -> None:
    report = Tuner(_FixedProbe([2.0]), sample_size=8).tune_stack(
        ["parser.lighton"],
        stack_name="gpu-ocr.yaml",
        fingerprint=_fingerprint(),
    )
    rendered = report.render()

    assert not _CLAIMS_RECORDS_PER_SECOND.search(rendered), rendered
    one_line = _one_line(rendered)
    assert "sample_size divided by the median" in one_line
    assert "whole stack" in one_line and "validation corpus" in one_line
    assert "not records per second" in one_line


# -- the behaviour the wording describes -----------------------------------------------------


class _FixedProbe:
    """A probe whose every candidate takes the same time and gives the same output."""

    def __init__(self, timings: list[float]) -> None:
        self._timings = timings

    def probe(
        self, provider: str, setting: KnobSetting, *, sample_size: int, repeats: int
    ) -> ProbeOutcome:
        return ProbeOutcome(fingerprint="same", timings=list(self._timings))


def _fingerprint() -> DeviceFingerprint:
    return DeviceFingerprint(cuda=False, cpu_count=4)


def test_the_figure_is_the_sample_number_over_the_median_seconds() -> None:
    """Doubling ``--sample`` doubles the figure for the same timed runs: it is a numerator."""
    outcome = ProbeOutcome(fingerprint="same", timings=[4.0, 2.0, 2.0])
    assert _throughput(8, outcome) == pytest.approx(4.0)
    assert _throughput(16, outcome) == pytest.approx(8.0)


def test_the_real_probe_runs_the_same_corpus_whatever_the_sample_number(tmp_path: Path) -> None:
    """The run is not sized by ``sample_size``: the target Stage's output is byte-identical.

    Run on the CPU default stack, on a Stage with no knobs, so this needs no model and no GPU.
    Both probes use the same storage root, emptied in between, because a record's Provenance
    carries the URI of its source document.
    """
    stack = _REPO_ROOT / "stacks" / "default.yaml"
    baseline = KnobSetting()
    root = tmp_path / "tune"

    small = RegistrySampleProbe(stack, storage_root=f"file://{root}").probe(
        "chunk.markdown", baseline, sample_size=1, repeats=1
    )
    shutil.rmtree(root)
    large = RegistrySampleProbe(stack, storage_root=f"file://{root}").probe(
        "chunk.markdown", baseline, sample_size=10_000, repeats=1
    )

    assert small.error is None and large.error is None, (small.error, large.error)
    assert small.fingerprint is not None
    assert small.fingerprint == large.fingerprint

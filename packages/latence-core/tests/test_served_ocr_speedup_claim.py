"""No page states a served-against-in-process OCR speedup while the comparison is unmeasured.

Six places said that LightOn-OCR served by vLLM is "about 2.7x faster" than the in-process
``transformers`` path, some of them "pod-measured on the same GPU". ``PERF-RESULTS.md`` holds the
performance records, and its section on exactly this comparison says it is pending: the pod that
measured the in-process ladder could not start a vLLM that supports the model. The tree has no
record the figure could have come from.

Two assertions keep the pages and the record in step:

* ``PERF-RESULTS.md`` still lists the comparison as pending. When the measurement is recorded,
  this test fails and is replaced by one that pins the pages to the recorded figure.
* While it is pending, a paragraph that writes the figure must say, in the same paragraph, that
  it is not on record. Stating it as a result is what this gate refuses.

ADRs are not read: they are dated decision records and are not revised.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The figure, as it was written: ``2.7x``, ``2.7×``, ``2.7 x``, with or without ``~`` and ``**``.
_FIGURE = re.compile(r"(?<![\d.])2\.7\s*[x×](?![\w.])", re.IGNORECASE)

#: Wording that marks the figure as not on record.
_SAYS_UNRECORDED = re.compile(
    r"pending|no (measurement )?record|not (been )?measured|unmeasured|has no source", re.IGNORECASE
)

_TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".yaml", ".yml", ".sh", ".cff"}


def _pages() -> list[Path]:
    """Prose and source a reader can reach: root pages, the docs site, packages, scripts, deploy."""
    found = [p for p in _REPO_ROOT.iterdir() if p.is_file() and p.suffix in _TEXT_SUFFIXES]
    for root in ("docs", "scripts", "deploy", "stacks", "examples"):
        found += [p for p in (_REPO_ROOT / root).rglob("*") if p.is_file()]
    for package in sorted((_REPO_ROOT / "packages").iterdir()):
        found += [package / "README.md", package / "pyproject.toml"]
        found += list((package / "src").rglob("*.py"))
    adr = _REPO_ROOT / "docs" / "adr"
    return sorted(
        p
        for p in found
        if p.is_file()
        and p.suffix in _TEXT_SUFFIXES
        and adr not in p.parents
        and "__pycache__" not in p.parts
    )


def _paragraphs(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def test_the_scan_reaches_the_pages_that_carried_the_figure() -> None:
    scanned = {str(p.relative_to(_REPO_ROOT)) for p in _pages()}
    for expected in (
        "docs/serving-lighton-vllm.md",
        "docs/TUTORIAL-fresh-pod-walkthrough.md",
        "docs/deployment.md",
        "docs/anatomy/01-ingest-and-document-understanding.md",
        "scripts/validate-lighton-vllm-on-gpu.md",
        "packages/latence-parser-lighton-vllm/README.md",
        "packages/latence-parser-lighton-vllm/pyproject.toml",
        "packages/latence-parser-lighton-vllm/src/latence_parser_lighton_vllm/__init__.py",
        "packages/latence-parser-lighton-vllm/src/latence_parser_lighton_vllm/provider.py",
        "PERF-RESULTS.md",
        "CHANGELOG.md",
    ):
        assert expected in scanned, f"{expected} is not scanned"


def test_the_pattern_matches_the_figure_as_it_was_written() -> None:
    for text in ("~**2.7× faster**", "(~2.7x faster than", "2.7 x faster", "the ~2.7× claim"):
        assert _FIGURE.search(text), text
    for text in ("scale ~2.78", "apache-airflow>=2.7", "torch 2.7.1", "12.7 x", "2.7xl"):
        assert not _FIGURE.search(text), text


def test_perf_results_still_lists_the_comparison_as_pending() -> None:
    text = (_REPO_ROOT / "PERF-RESULTS.md").read_text(encoding="utf-8")
    section = re.search(
        r"^## transformers-vs-vLLM OCR comparison[^\n]*PENDING.*?(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert section, (
        "PERF-RESULTS.md no longer has the pending served-OCR comparison section. If the "
        "measurement is recorded, replace this gate with one that pins the pages to the figure."
    )
    assert "Not measured" in section.group(0)


def test_no_page_states_the_speedup_while_it_is_pending() -> None:
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}: {paragraph[:160]}"
        for path in _pages()
        for paragraph in _paragraphs(path.read_text(encoding="utf-8", errors="replace"))
        if _FIGURE.search(paragraph) and not _SAYS_UNRECORDED.search(paragraph)
    ]
    assert offenders == [], (
        "these paragraphs state a served-against-in-process OCR speedup that PERF-RESULTS.md "
        "lists as pending. Say what is recorded, as docs/deployment.md does:\n"
        + "\n".join(offenders)
    )

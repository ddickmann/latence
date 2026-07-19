"""The committed benchmark artifacts must be what today's renderer emits.

``docs/BENCHMARKS.md`` and ``docs/benchmarks/results.json`` are *generated* files that are also
*committed* files, and ``README.md`` points the outside world at them as this project's benchmark
evidence. That combination rots silently: the renderer is edited, the harness is not re-run, and
the published page keeps asserting whatever the previous renderer said. It happened — the page
shipped by #181 survived #187 untouched, so it carried ``| Pinned revision |`` where the renderer
had moved to ``| Pinned revision (commit) |``, claimed #187 "implements" the ``table``/``math``
assertion classes where the renderer now says their scoring remains UNIMPLEMENTED, carried the
pre-#187 determinism paragraph, and shipped a sidecar with none of the ``metrics``/``headline``
keys ``SuiteResult.as_dict`` had grown.

Nothing could see that: no test, no ``scripts/verify-local.sh`` step and no CI job read those two
files at all, so the gate was green throughout.

This module closes that hole without re-measuring anything. The sidecar is a complete record of
every published measurement, and :func:`~latence_benchmark.results.carry_forward` already rebuilds
:class:`~latence_benchmark.suite.SuiteResult` objects from it — validating each against the
dataset revision the registry pins *today*. So the artifacts are reproducible offline from bytes
that are already committed, with no dataset, no network and no Provider. Rebuilding them and
demanding a byte-identical render turns "someone forgot to re-run the harness" into a red test.

What this is *not*: it never asserts a particular pass rate, precision or recall. Those come from
runs against pinned data and are checked elsewhere. This asserts only that the committed bytes are
the bytes today's renderer produces for the numbers that were measured — freshness, not quality.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import pytest
from latence_benchmark.results import carry_forward, render_json, render_markdown
from latence_benchmark.suite import SuiteResult

# tests/  ->  latence-benchmark/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _REPO_ROOT / "docs" / "BENCHMARKS.md"
_SIDECAR = _REPO_ROOT / "docs" / "benchmarks" / "results.json"

#: ``## Redaction — `redaction` suite`` on the published page.
_PAGE_SECTION = re.compile(r"^## .*—\s*`([a-z_]+)`\s+suite\s*$", re.MULTILINE)

_REGENERATE = "uv run latence-benchmark run --suite all"


def _read(path: Path) -> str:
    assert path.is_file(), (
        f"{path.relative_to(_REPO_ROOT)} is missing. It is a committed, generated artifact — "
        f"regenerate it with `{_REGENERATE}`."
    )
    return path.read_text(encoding="utf-8")


def _published() -> tuple[tuple[SuiteResult, ...], str]:
    """Rebuild the published measurements from the committed sidecar, plus its stamp.

    ``carry_forward`` with no fresh results carries *every* measurement in the sidecar, and
    refuses (``PublishedResultsError``) any whose suite is no longer registered or whose dataset
    has since been re-pinned — so a repin that invalidates a published number surfaces here as a
    hard error rather than as a quietly wrong page.
    """
    payload = json.loads(_read(_SIDECAR))
    generated_on = payload.get("generated_on")
    assert generated_on, "the published sidecar carries no `generated_on` stamp"
    return carry_forward([], json_path=_SIDECAR), generated_on


def _diff(committed: str, rendered: str, label: str) -> str:
    body = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"{label} (committed)",
            tofile=f"{label} (what the renderer emits today)",
            n=2,
        )
    )
    return (
        f"\n{label} is STALE: it is not what `latence_benchmark.results` renders today, so the "
        "published page asserts things the harness no longer says.\n"
        f"Re-run the harness and commit both artifacts: `{_REGENERATE}`\n\n"
        f"{body}"
    )


def test_the_gate_is_not_vacuous() -> None:
    """An emptied sidecar must fail here, not silently make the freshness checks pass."""
    results, _ = _published()
    assert results, (
        "docs/benchmarks/results.json carries no measurements, which would make every freshness "
        "assertion below trivially true"
    )
    sectioned = set(_PAGE_SECTION.findall(_read(_PAGE)))
    assert sectioned == {r.suite for r in results}, (
        "the markdown page and the JSON sidecar disagree about which suites are published: "
        f"page={sorted(sectioned)} sidecar={sorted(r.suite for r in results)}"
    )


def test_the_published_markdown_page_is_what_the_renderer_emits_today() -> None:
    """`docs/BENCHMARKS.md` must be byte-identical to a re-render of the committed numbers."""
    results, generated_on = _published()
    committed = _read(_PAGE)
    rendered = render_markdown(results, generated_on=generated_on)
    assert rendered == committed, _diff(committed, rendered, "docs/BENCHMARKS.md")


def test_the_published_json_sidecar_is_what_the_renderer_emits_today() -> None:
    """The sidecar must round-trip through `as_dict`, so new schema keys cannot go unpublished."""
    results, generated_on = _published()
    committed = _read(_SIDECAR)
    rendered = render_json(results, generated_on=generated_on)
    assert rendered == committed, _diff(committed, rendered, "docs/benchmarks/results.json")


@pytest.mark.parametrize("artifact", [_PAGE, _SIDECAR])
def test_rendering_is_deterministic(artifact: Path) -> None:
    """Two renders of the same committed record are identical — no now(), no hash-order."""
    results, generated_on = _published()
    render = render_markdown if artifact == _PAGE else render_json
    assert render(results, generated_on=generated_on) == render(
        results, generated_on=generated_on
    ), f"rendering {artifact.name} twice produced different bytes"

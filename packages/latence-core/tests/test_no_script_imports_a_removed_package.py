"""No Python file in the tree imports a package ADR-0068 removed.

ADR-0068 took four distributions out of the workspace: ``latence-retrieval``,
``latence-sink-qdrant``, ``latence-splade`` and ``latence-muvera``. A script that still imports
one of their modules cannot run, and nothing said so: the benchmark folders are outside
``pytest packages`` and outside the lint scope, so scripts that imported the retrieval package
sat under ``benchmark/rfc/`` and ``benchmark/enterprise-gold/`` for as long as nobody ran them.

This gate reads every ``.py`` file under the directories that hold code and fails on an import
of a removed module, at module level or inside a function. One file is exempt, by name and with
its reason, and the exemption is checked too: it must still exist and still import the package,
so it cannot outlive the file it excuses.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The import names of the four removed distributions.
_REMOVED_MODULES = ("latence_retrieval", "latence_sink_qdrant", "latence_splade", "latence_muvera")

#: An ``import x`` or ``from x[.y] import`` statement for a removed module, at any indentation.
_IMPORTS_A_REMOVED_MODULE = re.compile(
    rf"^[ \t]*(?:from|import)[ \t]+(?:{'|'.join(_REMOVED_MODULES)})\b", re.MULTILINE
)

#: Every top-level directory that holds Python.
_ROOTS = ("benchmark", "demo", "deploy", "examples", "packages", "results", "scripts")

#: Files allowed to import a removed module, each with the reason. ``results/`` holds dated
#: records; this script is kept beside the result file it wrote, as the record of how the
#: numbers were produced. It is not run by anything.
_RECORDS = {
    "results/dogfood-rfc/report/rfc_bench.py": "the script of a dated result, kept as its record",
}


def _python_files() -> list[Path]:
    found: list[Path] = []
    for root in _ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            relative = path.relative_to(_REPO_ROOT)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            found.append(path)
    return found


def _imports_a_removed_module(path: Path) -> bool:
    return bool(_IMPORTS_A_REMOVED_MODULE.search(path.read_text(encoding="utf-8")))


def test_the_scan_reads_the_directories_that_hold_code() -> None:
    for root in _ROOTS:
        assert (_REPO_ROOT / root).is_dir(), f"{root}/ is gone; update _ROOTS"
    scanned = {p.relative_to(_REPO_ROOT).parts[0] for p in _python_files()}
    assert {"benchmark", "packages", "scripts"} <= scanned


def test_the_pattern_matches_the_ways_an_import_is_written() -> None:
    for line in (
        "import latence_retrieval",
        "from latence_retrieval.bm25 import Bm25Rescorer",
        "    from latence_retrieval import (",
        "\tfrom latence_sink_qdrant.testing import FakeQdrantClient",
        "import latence_splade.model as m",
        "from latence_muvera import MuveraFde",
    ):
        assert _IMPORTS_A_REMOVED_MODULE.search(line), line
    for line in (
        "# from latence_retrieval import Packer",
        'note = "the script used to import latence_retrieval"',
        "from latence_retrieval_notes import x",
        "from latence_core.pipeline import Pipeline",
    ):
        assert not _IMPORTS_A_REMOVED_MODULE.search(line), line


def test_no_python_file_imports_a_removed_package() -> None:
    offenders = sorted(
        str(path.relative_to(_REPO_ROOT))
        for path in _python_files()
        if _imports_a_removed_module(path)
        and str(path.relative_to(_REPO_ROOT)) not in _RECORDS
    )
    assert offenders == [], (
        "these files import a package ADR-0068 removed and cannot run; delete them and keep "
        f"their result files as dated records, as benchmark/official/README.md does: {offenders}"
    )


def test_every_exempt_record_still_exists_and_still_needs_its_exemption() -> None:
    for relative, reason in _RECORDS.items():
        path = _REPO_ROOT / relative
        assert path.is_file(), f"{relative} is gone; drop its exemption ({reason})"
        assert _imports_a_removed_module(path), f"{relative} no longer needs its exemption"

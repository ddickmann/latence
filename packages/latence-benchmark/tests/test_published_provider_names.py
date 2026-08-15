"""Every Provider name the benchmark package PUBLISHES must actually be registered.

The suites bake ``--provider <name>`` re-pin hints into their ``SuiteResult.notes``, which
``render_markdown`` / ``render_json`` emit verbatim into ``docs/BENCHMARKS.md`` and
``docs/benchmarks/results.json``. A name that no package registers is worse than a typo: the
reader who follows it hits ``ProviderUnavailableError`` and is told to install a package that
will never register that name.

The declared-name set is read from the real ``[project.entry-points."latence.*"]`` tables in
``packages/*/pyproject.toml`` — the same declarations the installed registry is built from — so
this guard holds offline and does not depend on which optional packages are installed here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"
# Derived from THIS checkout, not from ``latence_benchmark.__file__``. The installed module
# resolves to wherever the venv points — in a git worktree that is the *main* repo, so the
# scan silently audited a different tree than the one under test and ``relative_to`` blew up
# on it. A repo-hygiene gate has to read the repo it is running in.
_BENCHMARK_SRC = _PACKAGES / "latence-benchmark" / "src" / "latence_benchmark"

#: ``--provider foo.bar`` as written in prose/notes, optionally wrapped in backticks.
_PROVIDER_FLAG = re.compile(r"--provider\s+`?([a-z0-9_]+\.[a-z0-9_]+)`?")

_ENTRY_POINT_GROUPS = ("latence.providers", "latence.retrieval")


def _declared_provider_names() -> set[str]:
    names: set[str] = set()
    for pyproject in sorted(_PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        groups = data.get("project", {}).get("entry-points", {})
        for group in _ENTRY_POINT_GROUPS:
            names.update(groups.get(group, {}))
    return names


def _published_provider_mentions() -> dict[str, list[str]]:
    """``{provider name: [source locations]}`` for every ``--provider`` hint in the package."""
    mentions: dict[str, list[str]] = {}
    for source in sorted(_BENCHMARK_SRC.rglob("*.py")):
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            for name in _PROVIDER_FLAG.findall(line):
                where = f"{source.relative_to(_REPO_ROOT)}:{lineno}"
                mentions.setdefault(name, []).append(where)
    return mentions


def test_declared_provider_names_are_discoverable() -> None:
    """Guard the guard: the entry-point scan must actually find the real declarations."""
    declared = _declared_provider_names()
    assert {
        "entity.gazetteer",
        "entity.gliner",
        "relation.gliner_relex",
        "redaction.gliner_pii",
        "relation.llm",
    } <= declared, declared


def test_benchmark_package_mentions_provider_names() -> None:
    """Guard the guard: the source scan must actually find the published re-pin hints."""
    mentions = _published_provider_mentions()
    assert len(mentions) >= 3, mentions


def test_every_published_provider_name_is_registered() -> None:
    declared = _declared_provider_names()
    unknown = {
        name: where for name, where in _published_provider_mentions().items() if name not in declared
    }
    assert not unknown, (
        "The benchmark package publishes `--provider` names that no package registers under "
        f"{' / '.join(_ENTRY_POINT_GROUPS)}; a reader who follows them hits "
        f"ProviderUnavailableError: {unknown}"
    )


def test_suite_repin_hints_match_the_suites_own_stage() -> None:
    """A re-pin hint must be resolvable by the Stage it is printed next to.

    ``latence-benchmark run --provider X`` re-pins the suite's OWN ``StackPin.stage``, and a
    Provider is mapped to its Capability by its entry-point name PREFIX (docs/CONFORMANCE.md).
    A hint whose prefix does not match the suite's stage prefix cannot possibly load.
    """
    import inspect

    from latence_benchmark.redaction_suite import RedactionSuite
    from latence_benchmark.redfm_suites import DisambiguationSuite, EntitySuite, RelationSuite

    checked = 0
    wrong: list[str] = []
    for suite_cls in (EntitySuite, RelationSuite, DisambiguationSuite, RedactionSuite):
        stage_prefix = suite_cls.stack_pin.provider.split(".", 1)[0]
        # The class body only — the module may host several suites side by side.
        for name in _PROVIDER_FLAG.findall(inspect.getsource(suite_cls)):
            checked += 1
            if name.split(".", 1)[0] != stage_prefix:
                wrong.append(f"{suite_cls.__name__} (stage prefix {stage_prefix!r}) → {name}")
    assert checked >= 3, f"the class-body scan found no re-pin hints to check ({checked})"
    assert not wrong, (
        "Published re-pin hints name a Provider whose Capability prefix does not match the "
        f"Stage the suite re-pins, so `--provider` would refuse to load it: {wrong}"
    )


@pytest.mark.parametrize(
    "doc",
    [
        Path("docs/running-the-benchmarks.md"),
        Path("docs/BENCHMARKS.md"),
        Path("docs/benchmarks/results.json"),
    ],
)
def test_published_artifacts_name_registered_providers(doc: Path) -> None:
    path = _REPO_ROOT / doc
    if not path.exists():  # pragma: no cover - the artifacts are committed
        pytest.skip(f"{doc} is not committed")
    declared = _declared_provider_names()
    unknown = sorted(
        {name for name in _PROVIDER_FLAG.findall(path.read_text(encoding="utf-8"))}
        - declared
    )
    assert not unknown, f"{doc} names unregistered Providers: {unknown}"

"""Every ``examples/*.yaml`` parses, resolves its Providers, and is a real DAG.

``stacks/*.yaml`` have been globbed and validated end-to-end for a long time
(``test_stack_validate.py``, ``test_envspec.py``, and the ``stacks`` job in ``ci.yml``).
``examples/*.yaml`` were not: only two of them were touched by any test, and only for a
single PII property (``test_example_corpus_pii_default.py``). That is a front-door gap —
``README.md`` quotes ``diff examples/demo-pipeline.yaml examples/demo-pipeline.swap.yaml``
as the model-agnosticism proof, and the quickstart's one-command demo runs
``demo-pipeline.yaml``. An example that stopped parsing, or that named a Provider nobody
registers any more, would be discovered by a reader rather than by CI.

The rule this file enforces is docling's: **every example is classified, never silently
unrun.** ``_EXAMPLES_CLASSIFICATION`` must name every file in ``examples/``; adding a new
example without a line here fails, so "we forgot to cover it" is not reachable.

What is deliberately NOT here: executing the pipelines. These are Pipeline declarations
over corpora that mostly do not ship (``EXAMPLES_DIR`` is a placeholder the demo replaces),
and the one that does run end-to-end — ``demo-pipeline.yaml``, via ``latence-demo`` — is
already exercised by the demo's own tests and by the quickstart. Static validation is the
gap; execution is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.pipeline import Pipeline

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES = _REPO_ROOT / "examples"

# The placeholders the demo/tests substitute with absolute URIs before a run. They are
# documented in the example headers; substituting them here is what lets a declaration that
# is not directly runnable still be fully validated.
_PLACEHOLDERS = {
    "EXAMPLES_DIR": "/tmp/latence-examples",
    "STORAGE_URI": "file:///tmp/latence-examples-out",
}

# Every shipped example, with what it is for. A file in examples/ that is absent from this
# map fails `test_every_shipped_example_is_classified` — the point of the map is that
# forgetting to cover a new example is impossible, not that the notes are load-bearing.
_EXAMPLES_CLASSIFICATION: dict[str, str] = {
    "pipeline.yaml": "walking skeleton: Source -> Parse -> Export",
    "parse-slice.yaml": "Parse stage slice",
    "chunk-screening-slice.yaml": "Chunk + content Screening slice",
    "entity-slice.yaml": "Entity Extraction slice",
    "redaction-slice.yaml": "Redaction (PII) slice",
    "disambiguation-slice.yaml": "Disambiguation / entity-resolution slice",
    "graph-slice.yaml": "Graph Assembly + KG Export slice (README front-door artifact)",
    "profiling-slice.yaml": "Profiling slice",
    "quality-slice.yaml": "Quality report / gold-set slice",
    "delta-slice.yaml": "incremental-ingest (delta) slice",
    "performance-path-slice.yaml": "the performance-path configuration",
    "demo-pipeline.yaml": "the one-command demo pipeline (quickstart front door)",
    "demo-pipeline.swap.yaml": "the ONE-LINE Provider swap the README diffs against demo-pipeline.yaml",
}

# Examples whose Providers are NOT resolvable in the CPU-first dev env, each with the reason
# and the package that would supply them. Empty today, and that is the honest state: every
# shipped example resolves with `uv sync` plus the --no-deps heavy sources CI installs. The
# mechanism stays so that a future GPU/endpoint example is skipped WITH a stated reason
# rather than quietly weakening the assertion for everything.
_PROVIDERS_NOT_IN_THE_CPU_DEV_ENV: dict[str, str] = {}


def _example_files() -> list[Path]:
    return sorted(_EXAMPLES.glob("*.yaml"))


def _load(path: Path) -> Pipeline:
    text = path.read_text(encoding="utf-8")
    for token, value in _PLACEHOLDERS.items():
        text = text.replace(token, value)
    return Pipeline.from_yaml_text(text)


def test_examples_directory_is_not_empty() -> None:
    """A glob that matches nothing is a green test that checks nothing."""
    assert _example_files(), "examples/ holds no *.yaml — this whole file would vacuously pass"


def test_every_shipped_example_is_classified() -> None:
    """No example may exist without a line in ``_EXAMPLES_CLASSIFICATION``."""
    on_disk = {p.name for p in _example_files()}
    classified = set(_EXAMPLES_CLASSIFICATION)

    unclassified = sorted(on_disk - classified)
    assert not unclassified, (
        f"examples/ gained {unclassified} with no entry in _EXAMPLES_CLASSIFICATION. Add one "
        "(a short note on what the example is for) so it is covered by the checks below "
        "rather than silently unvalidated."
    )
    stale = sorted(classified - on_disk)
    assert not stale, (
        f"_EXAMPLES_CLASSIFICATION names {stale}, which is not in examples/ any more — "
        "delete the entry so the map keeps describing reality."
    )


@pytest.mark.parametrize("name", sorted(_EXAMPLES_CLASSIFICATION))
def test_shipped_example_parses_and_validates(name: str) -> None:
    """The example loads through the real ``Pipeline`` model, validators and all.

    ``Pipeline`` is ``extra="forbid"`` and runs ``_validate_dag`` / ``_validate_extract_on``
    on construction, so this covers unknown keys, duplicate Stage names, a ``depends_on``
    naming a Stage that does not exist, and a ``extract_on: masked`` pipeline whose
    extraction does not actually run after Redaction.
    """
    pipeline = _load(_EXAMPLES / name)
    assert pipeline.stages, f"{name}: declares no Stages"
    assert pipeline.name, f"{name}: declares no pipeline name"


@pytest.mark.parametrize("name", sorted(_EXAMPLES_CLASSIFICATION))
def test_shipped_example_names_only_registered_providers(name: str) -> None:
    """Every ``provider:`` in the example is a real registered ``latence.providers`` entry.

    This is the check that catches a renamed or deleted Provider: the example keeps parsing
    (the name is just a string to the model) and only fails at run time, in front of whoever
    followed the README.
    """
    if name in _PROVIDERS_NOT_IN_THE_CPU_DEV_ENV:
        pytest.skip(_PROVIDERS_NOT_IN_THE_CPU_DEV_ENV[name])

    registered = set(ProviderRegistry().names())
    assert registered, "no Providers discovered at all — the workspace is not installed"

    declared = {stage.provider for stage in _load(_EXAMPLES / name).stages}
    unknown = sorted(declared - registered)
    assert not unknown, (
        f"{name} names Provider(s) {unknown} that nothing registers under "
        "'latence.providers'. Either the Provider was renamed/removed and the example was "
        "not updated, or it ships in a package the CPU-first dev env does not install — in "
        "which case add the example to _PROVIDERS_NOT_IN_THE_CPU_DEV_ENV with the reason."
    )


@pytest.mark.parametrize("name", sorted(_EXAMPLES_CLASSIFICATION))
def test_shipped_example_dag_is_acyclic(name: str) -> None:
    """``depends_on`` must topologically sort.

    ``Pipeline._validate_dag`` checks that every dependency NAMES a real Stage and that no
    Stage depends on itself, but it does not reject a longer cycle (a -> b -> a). A cycle is
    not a config nuance; it is a pipeline that cannot be scheduled.
    """
    stages = _load(_EXAMPLES / name).stages
    pending = {s.name: set(s.depends_on) for s in stages}
    resolved: set[str] = set()

    while pending:
        ready = [n for n, deps in pending.items() if deps <= resolved]
        assert ready, (
            f"{name}: dependency cycle among {sorted(pending)} — no Stage has all of its "
            "depends_on satisfied, so the DAG cannot be scheduled."
        )
        resolved.update(ready)
        for n in ready:
            del pending[n]


def test_the_readme_provider_swap_pair_differs_only_in_providers() -> None:
    """The README's model-agnosticism proof is a Provider swap, not a different pipeline.

    ``README.md`` presents ``diff examples/demo-pipeline.yaml examples/demo-pipeline.swap.yaml``
    as "one line changes, the pipeline does not". If the two files ever drift into different
    Stage graphs, that claim silently becomes false while the diff still looks small.
    """
    base = _load(_EXAMPLES / "demo-pipeline.yaml")
    swap = _load(_EXAMPLES / "demo-pipeline.swap.yaml")

    def shape(p: Pipeline) -> list[tuple[str, str, tuple[str, ...]]]:
        return [(s.name, str(s.capability), tuple(s.depends_on)) for s in p.stages]

    assert shape(base) == shape(swap), (
        "demo-pipeline.yaml and demo-pipeline.swap.yaml no longer share a Stage graph, so the "
        "README's 'one line changes, the pipeline does not' proof no longer holds."
    )
    changed = {
        (b.name, b.provider, s.provider)
        for b, s in zip(base.stages, swap.stages, strict=True)
        if b.provider != s.provider
    }
    assert changed, "the swap file swaps no Provider at all — it is a copy, not a proof"

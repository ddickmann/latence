"""The compatibility-matrix regression guard (issue #199).

Two things are proved here, and — as in ``test_envspec.py`` — the second is the one that matters:

* the committed ``deploy/env/compatibility-matrix.yaml`` verifies clean and its published document
  is current; and
* **the verifier genuinely bites.** Every way a matrix can quietly stop being a record of
  measurements is injected into a copy of the real matrix and MUST be reported: a ``verified`` cell
  with no evidence, an ``UNVERIFIED`` cell with no reproduce command, a component the environments
  declare but the matrix forgot, a version that no longer matches the committed lock, a dropped
  live-run failure mode, an unlicensed served model, a full-stack record pointing at a script that
  does not exist.

Plus the artifact assertion the one-and-only full-stack run hangs on: ``check_run`` in
``deploy/env/full-stack/assert_full_stack.py`` is driven over synthesised run directories, both a
good one and every near-miss the framework's own history has produced (a skipped Stage, a Stage
that resolved to ``cpu``, an empty export). An assertion that guards a run nobody can re-run
cheaply must itself be tested.

Everything here is offline and import-free: YAML, the committed locks and the filesystem.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from latence_core.compat import (
    REQUIRED_FAILURE_MODES,
    UNVERIFIED,
    VERIFIED,
    Axis,
    Cell,
    CompatibilityMatrix,
    Evidence,
    load_matrix,
    render_matrix,
    required_components,
    verify_matrix,
)
from latence_core.envspec import load_environment, model_loading_providers, stack_providers
from latence_core.errors import ConfigError

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_PATH = _REPO_ROOT / "deploy" / "env" / "compatibility-matrix.yaml"
_FULL_STACK = _REPO_ROOT / "deploy" / "env" / "full-stack"


@pytest.fixture
def matrix() -> CompatibilityMatrix:
    return load_matrix(_MATRIX_PATH)


def _codes(matrix: CompatibilityMatrix, **kwargs: bool) -> set[str]:
    return {f.code for f in verify_matrix(matrix, _REPO_ROOT, **kwargs).findings}


def _with_cells(matrix: CompatibilityMatrix, cells: tuple[Cell, ...]) -> CompatibilityMatrix:
    return dataclasses.replace(matrix, cells=cells)


def _first(matrix: CompatibilityMatrix, kind: str) -> Cell:
    return next(cell for cell in matrix.cells if cell.kind == kind)


def _replace_cell(matrix: CompatibilityMatrix, target: Cell, new: Cell) -> CompatibilityMatrix:
    return _with_cells(matrix, tuple(new if c is target else c for c in matrix.cells))


# --------------------------------------------------------------------------------------------
# The committed matrix is green, complete and published.
# --------------------------------------------------------------------------------------------


def test_the_committed_matrix_verifies_clean(matrix: CompatibilityMatrix) -> None:
    report = verify_matrix(matrix, _REPO_ROOT)
    assert report.ok, report.describe()


def test_the_published_document_is_the_render_of_the_matrix(matrix: CompatibilityMatrix) -> None:
    """The Markdown is GENERATED. Hand-editing it, or editing only the YAML, is caught."""
    published = (_REPO_ROOT / matrix.doc_path).read_text(encoding="utf-8")
    assert published == render_matrix(matrix)


def test_rendering_is_deterministic(matrix: CompatibilityMatrix) -> None:
    assert render_matrix(matrix) == render_matrix(matrix)


def test_the_document_states_its_hardware_and_verification_date(
    matrix: CompatibilityMatrix,
) -> None:
    """AC: the matrix states the hardware it was verified on and when (it lives with deploy docs)."""
    published = (_REPO_ROOT / matrix.doc_path).read_text(encoding="utf-8")
    assert "Blackwell" in published
    assert matrix.verified_on in published
    assert "Reference hardware" in published


def test_every_environment_and_component_of_the_committed_specs_has_a_cell(
    matrix: CompatibilityMatrix,
) -> None:
    """The AC's 'covers every learned Provider and serving dependency', asserted from the specs."""
    assert not _codes(matrix) & {"E-MATRIX-CELL-UNCOVERED", "E-MATRIX-ENVIRONMENT-UNCOVERED"}
    covered = {cell.component for cell in matrix.cells}
    # A spot-check in plain sight, so a future refactor of the derivation cannot vacuously pass.
    for expected in ("vllm", "flashinfer-python", "torch", "transformers", "flash-attn"):
        assert expected in covered
    for provider in (
        "fused_entity_relation.gliner2",
        "redaction.gliner_pii",
        "redaction.gliner2",
        "disambiguation.glinker",
        "embedding.sentence_transformers",
    ):
        assert provider in covered


def test_a_model_loading_provider_is_covered_even_when_the_spec_declares_none(
    matrix: CompatibilityMatrix,
) -> None:
    """R3-env_reproducibility-1: coverage must not rest on an OPTIONAL hand-written field.

    ``deploy/env/cpu-text/environment.yaml`` declares no ``learned_providers`` at all (correctly —
    that field is the GPU-placement probe set and this is a CPU environment), yet it installs five
    Providers whose packages pull a model runtime. Before the fix the derivation read that field
    alone, so ``required_components`` returned zero learned-Provider cells for cpu-text and
    ``verify_matrix`` was silent about all five: the matrix's own "a new learned Provider turns
    this file red" promise was false for any spec that omits the field.
    """
    spec = load_environment(_REPO_ROOT / "deploy" / "env" / "cpu-text" / "environment.yaml")
    assert spec.learned_providers == frozenset()  # the silence this test exists to survive

    shipped = {
        "fused_entity_relation.gliner2",
        "redaction.gliner_pii",
        "disambiguation.glinker",
        "embedding.sentence_transformers",
        "sparse.splade",
    }
    derived = required_components(spec, _REPO_ROOT)
    assert {name for name, kind in derived.items() if kind == "learned_provider"} == shipped

    # A Provider that owns NO model runtime of its own is NOT demanded a cell: latence-disambig-
    # embedding declares only latence-core and takes its vectors from an injected embedder. Without
    # this the derivation would be "everything in providers", which is not coverage, it is noise.
    assert "disambiguation.embedding" in spec.providers
    assert "disambiguation.embedding" not in derived

    # And the committed matrix really carries them — asserted through the public verify path, one
    # cell at a time, so a future refactor cannot pass this vacuously.
    assert not _codes(matrix) & {"E-MATRIX-CELL-UNCOVERED"}
    for provider in sorted(shipped):
        target = next(c for c in matrix.cells if c.key == ("cpu-text", provider))
        remaining = tuple(c for c in matrix.cells if c is not target)
        assert "E-MATRIX-CELL-UNCOVERED" in _codes(_with_cells(matrix, remaining))


def test_the_model_runtime_markers_cover_the_committed_workspace() -> None:
    """The derivation's one residual, pinned: every model-loading package IS recognised.

    ``model_loading_providers`` matches declared dependencies against a curated set of runtime
    distributions, so a Provider package that loads weights through some *other* loader would slip
    through. This test names the committed model-loading packages explicitly, so adding one whose
    runtime the set does not know is a red test and a one-line fix — never a silently empty
    derivation.
    """
    loads_weights = {
        "cpu-text": {
            "fused_entity_relation.gliner2",
            "redaction.gliner_pii",
            "disambiguation.glinker",
            "embedding.sentence_transformers",
            "sparse.splade",
        },
        "gpu-learned": {
            "fused_entity_relation.gliner2",
            "redaction.gliner_pii",
            "redaction.gliner2",
            "disambiguation.glinker",
            "embedding.sentence_transformers",
        },
        "served-ocr-client": set(),
        "served-ocr-server": set(),
    }
    # Classified as owning NO model runtime of their own. Hard-coded on purpose: it is the half
    # that makes this test bite. A Provider added to an environment lands in this complement, so
    # if its loader distribution is not in `_MODEL_RUNTIME_DISTRIBUTIONS` the omission surfaces
    # HERE rather than as an empty derivation nobody notices.
    deterministic = {
        "source.local_folder",
        "screening.intake_signature",
        "screening.content_keyword",
        "parser.pdfplumber",
        "parser.plaintext",
        "parser.endpoint",
        # Both are CLIENTS of a served model, not loaders of one (ADR-0030): their packages depend
        # on `openai`, and the model they talk to is mapped as its own `served_model` cell. The
        # in-process `parser.lighton` — which does load a 1B VLM — is deliberately absent from the
        # served-ocr-client Provider set, so the derivation correctly never sees it.
        "parser.lighton_vllm",
        "chunk.markdown",
        "profiling.statistical",
        "disambiguation.embedding",
        "graph.canonical",
        "export.knowledge_graph",
        "export.jsonl_parquet",
    }
    seen = set()
    for path in sorted((_REPO_ROOT / "deploy" / "env").glob("*/environment.yaml")):
        spec = load_environment(path)
        seen.add(spec.name)
        derived = model_loading_providers(spec, _REPO_ROOT)
        assert derived == loads_weights[spec.name], spec.name
        assert set(spec.providers) - derived <= deterministic, spec.name
    assert seen == set(loads_weights)


def test_no_cell_is_left_without_a_verdict(matrix: CompatibilityMatrix) -> None:
    """The ticket's core rule, asserted directly: measured or UNVERIFIED, nothing in between."""
    for cell in matrix.cells:
        for axis in (cell.resolution, cell.runtime):
            assert axis.status in (VERIFIED, UNVERIFIED)
            if axis.verified:
                assert axis.evidence is not None and not axis.evidence.missing()
            else:
                assert axis.reproduce and axis.blocked_by


# --------------------------------------------------------------------------------------------
# The verifier bites — one injected drift per case.
# --------------------------------------------------------------------------------------------


def test_a_verified_cell_without_evidence_is_reported(matrix: CompatibilityMatrix) -> None:
    target = _first(matrix, "learned_provider")
    injected = dataclasses.replace(
        target, runtime=Axis(name="runtime", status=VERIFIED, evidence=None)
    )
    assert "E-MATRIX-EVIDENCE-MISSING" in _codes(_replace_cell(matrix, target, injected))


@pytest.mark.parametrize("blank", ["date", "hardware", "command", "observed"])
def test_a_verified_cell_with_an_incomplete_measurement_is_reported(
    matrix: CompatibilityMatrix, blank: str
) -> None:
    """Each of the four is load-bearing: without any one of them the claim is not reproducible."""
    complete = {
        "date": "2026-07-17",
        "hardware": "Blackwell sm_120",
        "command": "bash deploy/env/gpu-learned/smoke.sh",
        "observed": "all gates passed",
    }
    complete[blank] = ""
    target = _first(matrix, "learned_provider")
    injected = dataclasses.replace(
        target,
        runtime=Axis(name="runtime", status=VERIFIED, evidence=Evidence(**complete)),
    )
    assert "E-MATRIX-EVIDENCE-MISSING" in _codes(_replace_cell(matrix, target, injected))


@pytest.mark.parametrize(
    ("reproduce", "blocked_by"),
    [("", "no GPU here"), ("bash deploy/env/gpu-learned/smoke.sh", "")],
)
def test_an_unverified_cell_must_stay_actionable(
    matrix: CompatibilityMatrix, reproduce: str, blocked_by: str
) -> None:
    """UNVERIFIED is a legitimate state ONLY while it names the command and the blocker."""
    target = _first(matrix, "learned_provider")
    injected = dataclasses.replace(
        target,
        runtime=Axis(name="runtime", status=UNVERIFIED, reproduce=reproduce, blocked_by=blocked_by),
    )
    assert "E-MATRIX-REPRODUCE-MISSING" in _codes(_replace_cell(matrix, target, injected))


def test_a_reproduce_command_pointing_at_a_missing_script_is_reported(
    matrix: CompatibilityMatrix,
) -> None:
    """A runbook that names a script nobody shipped is worse than no runbook."""
    target = _first(matrix, "learned_provider")
    injected = dataclasses.replace(
        target,
        runtime=Axis(
            name="runtime",
            status=UNVERIFIED,
            reproduce="bash deploy/env/gpu-learned/does-not-exist.sh",
            blocked_by="no GPU here",
        ),
    )
    assert "E-MATRIX-REPRODUCE-UNRUNNABLE" in _codes(_replace_cell(matrix, target, injected))


def test_a_dropped_learned_provider_cell_is_reported(matrix: CompatibilityMatrix) -> None:
    target = _first(matrix, "learned_provider")
    remaining = tuple(c for c in matrix.cells if c is not target)
    assert "E-MATRIX-CELL-UNCOVERED" in _codes(_with_cells(matrix, remaining))


def test_a_dropped_serving_dependency_cell_is_reported(matrix: CompatibilityMatrix) -> None:
    target = next(c for c in matrix.cells if c.component == "vllm")
    remaining = tuple(c for c in matrix.cells if c is not target)
    assert "E-MATRIX-CELL-UNCOVERED" in _codes(_with_cells(matrix, remaining))


def test_a_whole_environment_dropped_from_the_matrix_is_reported(
    matrix: CompatibilityMatrix,
) -> None:
    remaining = tuple(c for c in matrix.cells if c.environment != "cpu-text")
    assert "E-MATRIX-ENVIRONMENT-UNCOVERED" in _codes(_with_cells(matrix, remaining))


def test_an_invented_cell_is_reported(matrix: CompatibilityMatrix) -> None:
    """A matrix that grows rows nothing declares stops describing the thing it claims to."""
    invented = dataclasses.replace(
        _first(matrix, "distribution"), component="jaxlib", version="0.4.0"
    )
    assert "E-MATRIX-CELL-UNKNOWN" in _codes(_with_cells(matrix, (*matrix.cells, invented)))


def test_a_cell_naming_an_unknown_environment_is_reported(matrix: CompatibilityMatrix) -> None:
    invented = dataclasses.replace(_first(matrix, "distribution"), environment="gpu-imaginary")
    assert "E-MATRIX-CELL-UNKNOWN" in _codes(_with_cells(matrix, (*matrix.cells, invented)))


def test_a_duplicated_cell_is_reported(matrix: CompatibilityMatrix) -> None:
    target = _first(matrix, "distribution")
    assert "E-MATRIX-CELL-DUPLICATE" in _codes(_with_cells(matrix, (*matrix.cells, target)))


def test_a_version_that_disagrees_with_the_committed_lock_is_reported(
    matrix: CompatibilityMatrix,
) -> None:
    """The matrix describes the PINNED environments; a stale version silently describes nothing."""
    target = next(c for c in matrix.cells if c.component == "vllm")
    injected = dataclasses.replace(target, version="0.25.1")
    assert "E-MATRIX-VERSION-DRIFT" in _codes(_replace_cell(matrix, target, injected))


def test_a_distribution_cell_with_no_version_is_reported(matrix: CompatibilityMatrix) -> None:
    target = _first(matrix, "distribution")
    injected = dataclasses.replace(target, version="")
    assert "E-MATRIX-VERSION-MISSING" in _codes(_replace_cell(matrix, target, injected))


def test_an_excluded_distribution_that_claims_a_version_is_reported(
    matrix: CompatibilityMatrix,
) -> None:
    """An exclusion has no version: it is the claim that NO version can be installed."""
    target = _first(matrix, "excluded_distribution")
    injected = dataclasses.replace(target, version="2.8.3")
    assert "E-MATRIX-VERSION-DRIFT" in _codes(_replace_cell(matrix, target, injected))


def test_a_component_filed_under_the_wrong_kind_is_reported(matrix: CompatibilityMatrix) -> None:
    target = _first(matrix, "excluded_distribution")
    injected = dataclasses.replace(target, kind="distribution", version="2.8.3")
    codes = _codes(_replace_cell(matrix, target, injected))
    assert "E-MATRIX-CELL-UNKNOWN" in codes


def test_a_served_model_without_both_licences_is_reported(matrix: CompatibilityMatrix) -> None:
    """ADR-0012: weights and code are verified SEPARATELY, so one of them is not enough."""
    target = _first(matrix, "served_model")
    for injected in (
        dataclasses.replace(target, weights_license=""),
        dataclasses.replace(target, code_license=""),
    ):
        assert "E-MATRIX-LICENCE-UNCITED" in _codes(_replace_cell(matrix, target, injected))


@pytest.mark.parametrize("dropped", REQUIRED_FAILURE_MODES)
def test_dropping_any_live_run_failure_mode_is_reported(
    matrix: CompatibilityMatrix, dropped: str
) -> None:
    """The five that cost a real operator real hours. None may quietly leave the matrix."""
    remaining = tuple(fm for fm in matrix.failure_modes if fm.id != dropped)
    injected = dataclasses.replace(matrix, failure_modes=remaining)
    assert "E-MATRIX-FAILURE-MODE-MISSING" in _codes(injected)


def test_every_required_failure_mode_is_present_and_resolved(matrix: CompatibilityMatrix) -> None:
    recorded = {fm.id: fm for fm in matrix.failure_modes}
    assert set(REQUIRED_FAILURE_MODES) <= set(recorded)
    for fm in recorded.values():
        assert fm.resolution and fm.enforced_by
        assert fm.observation.verified, "a failure mode nobody observed is a prediction"


def test_a_failure_mode_without_a_resolution_is_reported(matrix: CompatibilityMatrix) -> None:
    target = matrix.failure_modes[0]
    for injected_fm in (
        dataclasses.replace(target, resolution=""),
        dataclasses.replace(target, enforced_by=()),
    ):
        injected = dataclasses.replace(
            matrix,
            failure_modes=tuple(injected_fm if fm is target else fm for fm in matrix.failure_modes),
        )
        assert "E-MATRIX-FAILURE-MODE-UNRESOLVED" in _codes(injected)


def test_a_failure_mode_enforced_by_a_file_that_does_not_exist_is_reported(
    matrix: CompatibilityMatrix,
) -> None:
    """The resolution has to live in the repo. A fix that lives only in a table is folklore."""
    target = matrix.failure_modes[0]
    injected_fm = dataclasses.replace(target, enforced_by=("deploy/env/nowhere/serve.sh",))
    injected = dataclasses.replace(
        matrix,
        failure_modes=tuple(injected_fm if fm is target else fm for fm in matrix.failure_modes),
    )
    assert "E-MATRIX-ENFORCEMENT-MISSING" in _codes(injected)


def test_a_full_stack_record_pointing_at_a_missing_script_is_reported(
    matrix: CompatibilityMatrix,
) -> None:
    injected = dataclasses.replace(
        matrix,
        full_stack_run=dataclasses.replace(
            matrix.full_stack_run, script="deploy/env/full-stack/nope.sh"
        ),
    )
    assert "E-MATRIX-RUN-UNRECORDED" in _codes(injected)


def test_a_full_stack_record_with_no_artifacts_is_reported(matrix: CompatibilityMatrix) -> None:
    """'Quality Report + Exports out' is the acceptance criterion, so it is part of the record."""
    injected = dataclasses.replace(
        matrix, full_stack_run=dataclasses.replace(matrix.full_stack_run, artifacts=())
    )
    assert "E-MATRIX-RUN-UNRECORDED" in _codes(injected)


def test_a_stale_published_document_is_reported(matrix: CompatibilityMatrix) -> None:
    injected = dataclasses.replace(matrix, summary="something else entirely")
    assert "E-MATRIX-DOC-STALE" in _codes(injected)


def test_require_verified_refuses_to_call_an_unmeasured_map_complete(
    matrix: CompatibilityMatrix,
) -> None:
    """The switch a maintainer flips ON the Blackwell host; red here is the honest answer today."""
    codes = _codes(matrix, require_verified=True)
    assert {"E-MATRIX-CELL-UNVERIFIED", "E-MATRIX-RUN-UNVERIFIED"} <= codes


@pytest.mark.parametrize("status", ["probably", "partial", "ok", "VERIFIED?"])
def test_a_third_softer_status_is_refused_at_load(tmp_path: Path, status: str) -> None:
    """There is deliberately no 'mostly works' — that is how an inferred cell gets written down."""
    text = _MATRIX_PATH.read_text(encoding="utf-8").replace(
        "  cpu_text: &rt_cpu_text\n    status: UNVERIFIED",
        f"  cpu_text: &rt_cpu_text\n    status: {status}",
    )
    spoiled = tmp_path / "compatibility-matrix.yaml"
    spoiled.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="status must be exactly one of"):
        load_matrix(spoiled)


# --------------------------------------------------------------------------------------------
# The full-stack verification: its stack, and the artifact assertion that gates it.
# --------------------------------------------------------------------------------------------


class _AssertModule(Protocol):
    CUDA_STAGES: tuple[str, ...]
    PARSE_STAGE: str

    def check_run(self, root: Path, run_id: str = ...) -> list[str]: ...


def _load_assert_module() -> _AssertModule:
    path = _FULL_STACK / "assert_full_stack.py"
    spec = importlib.util.spec_from_file_location("latence_full_stack_assert", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast("_AssertModule", module)


_ASSERT = _load_assert_module()
_RUN_ID = "full-stack-0001"


def _stage(name: str, **overrides: Any) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "name": name,
        "capability": name,
        "provider": f"provider.{name}",
        "records_out": 3,
        "skipped": False,
        "device_selected": "cuda",
    }
    stage.update(overrides)
    return stage


def _write_run(root: Path, *, report: dict[str, Any], corpus_lines: int = 2) -> None:
    run_dir = root / "_latence" / "runs" / _RUN_ID
    export = run_dir / "export"
    export.mkdir(parents=True, exist_ok=True)
    (run_dir / "quality-report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "quality-report.md").write_text("# report\n", encoding="utf-8")
    (export / "corpus.jsonl").write_text(
        "".join(json.dumps({"id": i}) + "\n" for i in range(corpus_lines)), encoding="utf-8"
    )
    for artifact in ("graph-nodes.parquet", "graph-edges.parquet"):
        (export / artifact).write_bytes(b"PAR1")


def _good_report() -> dict[str, Any]:
    stages = [_stage(name) for name in (*_ASSERT.CUDA_STAGES, _ASSERT.PARSE_STAGE)]
    return {
        "document_count": 2,
        "contracts_complete": True,
        "stages": stages,
    }


def test_a_complete_full_stack_run_passes_the_artifact_assertion(tmp_path: Path) -> None:
    _write_run(tmp_path, report=_good_report())
    assert _ASSERT.check_run(tmp_path, _RUN_ID) == []


def test_a_missing_quality_report_fails(tmp_path: Path) -> None:
    assert _ASSERT.check_run(tmp_path, _RUN_ID)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.update({"document_count": 0}), "document_count=0"),
        (lambda r: r.update({"contracts_complete": False}), "contracts_complete is false"),
    ],
    ids=["no-documents", "incomplete-contracts"],
)
def test_a_hollow_run_fails(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], expected: str
) -> None:
    report = _good_report()
    mutate(report)
    _write_run(tmp_path, report=report)
    failures = _ASSERT.check_run(tmp_path, _RUN_ID)
    assert any(expected in failure for failure in failures), failures


def test_a_skipped_learned_stage_fails_the_gpu_verification(tmp_path: Path) -> None:
    """Skip-with-flag is CORRECT on a CPU host and a false green here — the run proves the GPU."""
    report = _good_report()
    report["stages"] = [
        _stage(s["name"], skipped=True, skip_reason="no cuda device", device_selected=None)
        if s["name"] == _ASSERT.CUDA_STAGES[0]
        else s
        for s in report["stages"]
    ]
    _write_run(tmp_path, report=report)
    failures = _ASSERT.check_run(tmp_path, _RUN_ID)
    assert any("SKIPPED" in failure for failure in failures), failures


def test_a_stage_that_silently_ran_on_the_cpu_fails(tmp_path: Path) -> None:
    """The #191 signature: configured cuda, executed cpu, output still plausible."""
    report = _good_report()
    report["stages"] = [
        _stage(s["name"], device_selected="cpu") if s["name"] == _ASSERT.CUDA_STAGES[1] else s
        for s in report["stages"]
    ]
    _write_run(tmp_path, report=report)
    failures = _ASSERT.check_run(tmp_path, _RUN_ID)
    assert any("device_selected='cpu'" in failure for failure in failures), failures


def test_a_parse_stage_that_returned_nothing_fails(tmp_path: Path) -> None:
    """No page came back from the served OCR endpoint — nothing downstream means anything."""
    report = _good_report()
    report["stages"] = [
        _stage(s["name"], records_out=0) if s["name"] == _ASSERT.PARSE_STAGE else s
        for s in report["stages"]
    ]
    _write_run(tmp_path, report=report)
    assert any("served OCR endpoint" in f for f in _ASSERT.check_run(tmp_path, _RUN_ID))


def test_an_empty_export_fails(tmp_path: Path) -> None:
    _write_run(tmp_path, report=_good_report(), corpus_lines=0)
    assert any("empty" in f for f in _ASSERT.check_run(tmp_path, _RUN_ID))


def test_a_missing_knowledge_graph_export_fails(tmp_path: Path) -> None:
    _write_run(tmp_path, report=_good_report())
    (tmp_path / "_latence" / "runs" / _RUN_ID / "export" / "graph-edges.parquet").unlink()
    assert any("knowledge-graph export" in f for f in _ASSERT.check_run(tmp_path, _RUN_ID))


def test_the_assertion_covers_every_cuda_stage_of_the_committed_stack() -> None:
    """The assertion list and the stack cannot drift: the stack file is the source of truth."""
    import yaml

    stack = yaml.safe_load((_FULL_STACK / "stack.yaml").read_text(encoding="utf-8"))
    cuda_stages = {
        stage["name"]
        for stage in stack["stages"]
        if (stage.get("config") or {}).get("device") == "cuda"
    }
    assert cuda_stages == set(_ASSERT.CUDA_STAGES)


def test_the_full_stack_stack_only_uses_providers_the_gpu_environment_installs() -> None:
    """The composed run must be runnable in the pinned gpu-learned environment, not a wish."""
    from latence_core.envspec import load_environment

    spec = load_environment(_REPO_ROOT / "deploy" / "env" / "gpu-learned" / "environment.yaml")
    used = stack_providers(_FULL_STACK / "stack.yaml")
    assert used <= set(spec.providers), used - set(spec.providers)


def test_the_full_stack_stack_crosses_the_served_ocr_boundary() -> None:
    """Its ONLY source of document text is the endpoint Parser — no text Provider can fake it."""
    import yaml

    stack = yaml.safe_load((_FULL_STACK / "stack.yaml").read_text(encoding="utf-8"))
    parse = next(s for s in stack["stages"] if s["capability"] == "parse")
    assert parse["provider"] == "parser.endpoint"
    source = next(s for s in stack["stages"] if s["capability"] == "source")
    assert "txt" not in source["config"]["extensions"]


# --------------------------------------------------------------------------------------------
# The licence register covers every environment, not just the one that was audited first.
#
# Issue #196 and issue #198 each shipped a pinned GPU environment; only #198's was disclosed in
# THIRD-PARTY-LICENSES.md, and nothing in the tree noticed, because no gate related a committed
# lock to the register. These tests inject exactly that omission — one wheel name, one lock path,
# one matrix distribution — into a repo whose register is doctored, and require the report to
# bite. Audit R2 / cross-lane #15.
# --------------------------------------------------------------------------------------------

_REGISTER = "THIRD-PARTY-LICENSES.md"


def _repo_with_register(tmp_path: Path, text: str | None) -> Path:
    """A repo root that IS the real one, except for a doctored (or deleted) licence register."""
    root = tmp_path / "repo"
    root.mkdir()
    for entry in _REPO_ROOT.iterdir():
        if entry.name == _REGISTER:
            continue
        (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    if text is not None:
        (root / _REGISTER).write_text(text, encoding="utf-8")
    return root


def _findings(matrix: CompatibilityMatrix, root: Path) -> list[tuple[str, str]]:
    return [(f.code, f.message) for f in verify_matrix(matrix, root).findings]


def test_the_committed_register_names_every_cuda_wheel_every_lock_pins() -> None:
    """The disclosure itself, asserted directly: no `nvidia-*` wheel is missing from the page."""
    from latence_core.envspec import parse_lock

    register = (_REPO_ROOT / _REGISTER).read_text(encoding="utf-8")
    for lock in sorted(_REPO_ROOT.glob("deploy/env/*/requirements.lock")):
        wheels = [
            n for n in parse_lock(lock.read_text(encoding="utf-8")) if n.startswith("nvidia-")
        ]
        if not wheels:
            continue
        rel = lock.relative_to(_REPO_ROOT).as_posix()
        assert rel in register, f"{rel} pins {len(wheels)} nvidia-* wheels and is not disclosed"
        assert not [w for w in wheels if w not in register], f"{rel}: undisclosed wheels"


def test_a_cuda_wheel_missing_from_the_register_is_reported(
    matrix: CompatibilityMatrix, tmp_path: Path
) -> None:
    """`nvidia-cuda-tileiras` is pinned only by the served-OCR lock — dropping it must be seen."""
    doctored = (
        (_REPO_ROOT / _REGISTER)
        .read_text(encoding="utf-8")
        .replace("nvidia-cuda-tileiras", "some-other-wheel")
    )
    reported = _findings(matrix, _repo_with_register(tmp_path, doctored))
    assert [m for code, m in reported if code == "E-LICENCE-UNDISCLOSED" and "tileiras" in m]


def test_a_lock_the_register_never_cites_is_reported(
    matrix: CompatibilityMatrix, tmp_path: Path
) -> None:
    """The #196-vs-#198 defect exactly: a carve-out scoped to one lock does not cover another."""
    doctored = (
        (_REPO_ROOT / _REGISTER)
        .read_text(encoding="utf-8")
        .replace(
            "deploy/env/served-ocr-server/requirements.lock",
            "deploy/env/gpu-learned/requirements.lock",
        )
    )
    reported = _findings(matrix, _repo_with_register(tmp_path, doctored))
    assert [
        m
        for code, m in reported
        if code == "E-LICENCE-UNDISCLOSED" and "served-ocr-server" in m and "never cites" in m
    ]


def test_a_matrix_distribution_absent_from_the_register_is_reported(
    matrix: CompatibilityMatrix, tmp_path: Path
) -> None:
    """A `kind: distribution` cell carries a licence whether or not the cell has a field for it."""
    import re as _re

    doctored = _re.sub(
        "vllm", "some-server", (_REPO_ROOT / _REGISTER).read_text(encoding="utf-8"), flags=_re.I
    )
    reported = _findings(matrix, _repo_with_register(tmp_path, doctored))
    assert [m for code, m in reported if code == "E-LICENCE-UNDISCLOSED" and "'vllm'" in m]


def test_a_missing_register_is_reported_once(matrix: CompatibilityMatrix, tmp_path: Path) -> None:
    reported = _findings(matrix, _repo_with_register(tmp_path, None))
    assert [code for code, _ in reported].count("E-LICENCE-REGISTER-MISSING") == 1

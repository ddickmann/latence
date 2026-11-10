"""The offline stack check (C8): ``check_stack`` + ``latence stack check`` + the nested-block module.

Three gaps this suite pins, each of which shipped:

1. **The static checks were unreachable.** ``latence stack validate`` staged a corpus and ran the
   stack twice without ever calling the capability guard; ``latence process`` ran straight into
   execution. A hand-edited stack could only be checked by running it.
2. **A nested ``{provider, config}`` block was nobody's job.** The blessed SOTA stack emits
   several of them and a block outside the one hard-coded key was neither
   capability-checked, device-skip-checked, nor seen by any static tool — so a typo in a nested
   Provider name reached run time.
3. **A block under the wrong Capability was silently ignored.** Nothing reads it; the operator gets
   no signal and no error.

The assertions here are behavioural: an exit code, whether a corpus was staged, whether a run
happened, what the operator is told. None of them asserts a flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from latence_core.cli import app
from latence_core.errors import ConfigError, ContractError
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.stacks.check import Severity, check_stack
from latence_core.stacks.validate import validate_stack
from latence_core.subprovider import (
    declared_slot_keys,
    slots_for,
    stage_sub_providers,
    sub_provider_ref,
    undeclared_sub_provider_keys,
)
from typer.testing import CliRunner

if TYPE_CHECKING:
    from latence_core.pipeline import Stage

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACKS = _REPO_ROOT / "stacks"
_DEFAULT = _STACKS / "default.yaml"

cli = CliRunner()


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the device seam to CPU-only so the skip-with-flag posture is deterministic."""
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _default_stack_dict() -> dict[str, Any]:
    """The blessed default stack as a mutable dict (placeholders intact)."""
    loaded: dict[str, Any] = yaml.safe_load(_DEFAULT.read_text(encoding="utf-8"))
    return loaded


def _stage_of(stack: dict[str, Any], name: str) -> dict[str, Any]:
    stage: dict[str, Any] = next(s for s in stack["stages"] if s["name"] == name)
    stage.setdefault("config", {})
    return stage


def _write(tmp_path: Path, stack: dict[str, Any], name: str = "stack.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(stack, sort_keys=False), encoding="utf-8")
    return path


def _runnable_stack(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A tiny but real source→chunk→graph→export stack over a one-file corpus on disk.

    Used by the ``latence process`` tests: small enough to be fast, complete enough that a
    pre-check-guard ``process`` would genuinely start executing it.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Alice works at Acme Corporation.\n", encoding="utf-8")
    stack = {
        "name": "tiny",
        "storage_uri": f"file://{tmp_path}/store",
        "stages": [
            {
                "name": "source",
                "capability": "source",
                "provider": "source.local_folder",
                "config": {"path": f"file://{corpus}", "extensions": ["txt"]},
            },
            {
                "name": "parse",
                "capability": "parse",
                "provider": "parser.plaintext",
                "depends_on": ["source"],
            },
            {
                "name": "chunk",
                "capability": "chunk",
                "provider": "chunk.markdown",
                "depends_on": ["parse"],
                "config": {"max_tokens": 64, "overlap_tokens": 8, "min_tokens": 4},
            },
            {
                "name": "entities",
                "capability": "entity_extraction",
                "provider": "entity.gazetteer",
                "depends_on": ["chunk"],
                "config": {"labels": {"ORG": {"terms": ["Acme Corporation"]}}},
            },
            {
                "name": "disambiguate",
                "capability": "disambiguation",
                "provider": "disambiguation.exact_surface",
                "depends_on": ["entities"],
            },
            {
                "name": "graph",
                "capability": "graph_assembly",
                "provider": "graph.canonical",
                "depends_on": ["disambiguate"],
                "config": {"corpus_id": "tiny"},
            },
            {
                "name": "export_corpus",
                "capability": "export",
                "provider": "export.jsonl_parquet",
                "depends_on": ["chunk"],
                "config": {"basename": "corpus"},
            },
        ],
    }
    return _write(tmp_path, stack), stack


# -- REGRESSION 1: a sub-Provider block under the wrong Capability is silently ignored --------


def test_stack_validate_bites_on_a_sub_provider_block_that_nothing_reads(tmp_path: Path) -> None:
    """An ``ocr`` block on an EXPORT stage is read by NOBODY — the harness must say so.

    Before C8 this stack validated GREEN: the Export never looks at ``ocr``, so the run
    succeeded, every check passed, and the operator was left believing the block did something.
    Silent, and undetectable without reading the Export's source.
    """
    stack = _default_stack_dict()
    _stage_of(stack, "export_corpus")["config"]["ocr"] = {
        "provider": "parser.pdfplumber",
        "config": {},
    }
    path = _write(tmp_path, stack)

    result = validate_stack(path, storage_root=f"file://{tmp_path}/store")

    assert not result.ok, result.render()
    static = next(c for c in result.checks if c.name == "static_check")
    assert not static.passed
    assert "'ocr'" in static.detail
    assert "IGNORED" in static.detail


# -- REGRESSION 2: a mistyped nested Provider reached run time ---------------------------------


def test_process_refuses_a_mistyped_nested_provider_before_running(tmp_path: Path) -> None:
    """``embedding.hashng`` must stop ``latence process`` at the door, with a suggestion.

    Before C8 nothing static looked inside a nested block, so this typo survived ``latence setup``
    and died mid-run — after the corpus had been sourced, parsed, chunked and extracted. The
    operator's evidence that it stopped at the door: no run state was written at all.
    """
    path, stack = _runnable_stack(tmp_path)
    _stage_of(stack, "export_corpus")["config"]["embedder"] = {
        "provider": "embedding.hashng",
        "config": {},
    }
    path = _write(tmp_path, stack)

    result = cli.invoke(app, ["process", "--config", str(path)])

    assert result.exit_code == 2, result.output
    assert "embedding.hashng" in result.output
    assert "did you mean 'embedding.hashing'" in result.output
    # Nothing ran: no run state, no export, no corpus read.
    assert not (tmp_path / "store").exists()


def test_process_still_runs_a_correctly_wired_stack(tmp_path: Path) -> None:
    """The guard is a gate, not a wall — a clean stack runs exactly as before (no false bite)."""
    path, _ = _runnable_stack(tmp_path)

    result = cli.invoke(app, ["process", "--config", str(path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "store").exists()


# -- REGRESSION 3: a Stage-Provider typo cost a full run to discover ---------------------------


def test_a_stage_provider_typo_is_named_with_a_suggestion_and_no_corpus_is_staged(
    tmp_path: Path,
) -> None:
    """A one-letter Stage Provider typo used to make the publish gate report **PASS**.

    Measured on the pre-change code: ``chunk.markdow`` is not in the registry, so the device
    pre-flight read it as "provider not installed on this host", skipped it with-flag, transitively
    dropped everything downstream, and returned a green ``device_honesty`` result. A typo was
    indistinguishable from an absent GPU package — a false green in the command that gates
    CI/publish. The near-miss check tells the two apart: a name one edit away from a registered one
    is a typo (error, everywhere), a name nothing resembles is an absent package (warning).
    """
    stack = _default_stack_dict()
    _stage_of(stack, "chunk")["provider"] = "chunk.markdow"
    path = _write(tmp_path, stack)

    result = validate_stack(path, storage_root=f"file://{tmp_path}/store")

    assert not result.ok
    static = next(c for c in result.checks if c.name == "static_check")
    assert not static.passed
    assert "did you mean 'chunk.markdown'" in static.detail
    assert not any(c.name == "run" for c in result.checks)
    assert not (tmp_path / "store" / "_validation-corpus").exists()


# -- REGRESSION 4: a nested GPU Provider was never device-skip-checked -------------------------


def test_a_gpu_only_nested_ocr_provider_is_skipped_with_flag() -> None:
    """An ``ocr`` block pinned to a GPU Provider must skip-with-flag on a CPU host.

    Only ``config['embedder']`` was ever device-checked. Every other nested slot went straight to
    the Runner, so a GPU-only nested OCR model on a CPU box ran and failed instead of being
    honestly reported as skipped. The slot declaration now drives the pre-flight, so a slot added
    tomorrow is checked without touching the harness.
    """
    from latence_core.pipeline import Stage
    from latence_core.providers.profile import ProviderProfile
    from latence_core.stacks.validate import _skip_reason_for_stage

    class _GpuOnlyRegistry:
        """Mirrors ProviderRegistry, and REJECTS the same things: an unregistered name raises."""

        def names(self) -> list[str]:
            return ["parser.render", "parser.gpu_ocr"]

        def profile(self, name: str) -> ProviderProfile:
            if name not in self.names():
                msg = f"No Provider registered as {name!r}"
                raise KeyError(msg)
            return ProviderProfile(
                compute="gpu" if name == "parser.gpu_ocr" else "cpu",
                memory_mb=512,
                license="Apache-2.0",
                license_verified=False,
                deterministic=True,
                batch=False,
            )

    stage = Stage(
        name="parse",
        capability=CapabilityKind.PARSE,
        provider="parser.render",
        config={
            "ocr": {
                "provider": "parser.gpu_ocr",
                "config": {"device": "cuda"},
            }
        },
    )
    reason = _skip_reason_for_stage(stage, _GpuOnlyRegistry())  # type: ignore[arg-type]
    assert reason is not None
    assert "ocr" in reason
    assert "parser.gpu_ocr" in reason


# -- The command itself ------------------------------------------------------------------------


def test_cli_stack_check_exits_zero_on_the_blessed_default_stack() -> None:
    result = cli.invoke(app, ["stack", "check", str(_DEFAULT)])
    assert result.exit_code == 0, result.output
    assert "stack check OK" in result.output


def test_cli_stack_check_exits_one_on_a_broken_stack(tmp_path: Path) -> None:
    stack = _default_stack_dict()
    _stage_of(stack, "chunk")["provider"] = "chunk.markdow"
    path = _write(tmp_path, stack)
    result = cli.invoke(app, ["stack", "check", str(path)])
    assert result.exit_code == 1, result.output
    assert "stack check FAILED" in result.output


def test_cli_stack_check_exits_two_on_a_missing_file(tmp_path: Path) -> None:
    result = cli.invoke(app, ["stack", "check", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


def test_every_committed_stack_passes_the_offline_check() -> None:
    """The repo's own fleet is lint-clean — the enterprise use case, dogfooded.

    Providers that are not installed on this CPU host are WARNINGs, so the same files pass here and
    on a GPU pod; anything that is an ERROR is wrong everywhere.
    """
    for stack in sorted(_STACKS.glob("*.yaml")):
        result = check_stack(stack)
        assert result.ok, f"{stack.name}:\n{result.render()}"


# -- The check's own behaviour -----------------------------------------------------------------


def test_a_structurally_invalid_stack_is_a_finding_not_an_exception() -> None:
    """A linter that raises on the input it exists to lint is not a linter."""
    result = check_stack("stages: [1, 2, 3]")
    assert not result.ok
    assert [f.check for f in result.findings] == ["structure"]


def test_a_phase_boundary_violation_is_reported() -> None:
    pipeline = Pipeline.from_yaml_text(
        """
name: inverted
storage_uri: file:///tmp/x
stages:
  - name: profile
    capability: profiling
    provider: profiling.statistical
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [profile]
"""
    )
    result = check_stack(pipeline)
    assert not result.ok
    assert any(f.check == "phase_boundary" for f in result.errors)


def test_a_miswired_stage_provider_fails_the_capability_check() -> None:
    """A Fused Provider under the standalone relation seam — the ADR-0036 §3 motivating bug."""
    pipeline = Pipeline.from_yaml_text(
        """
name: miswired
storage_uri: file:///tmp/x
stages:
  - name: relations
    capability: relation_extraction
    provider: fused_entity_relation.header_refs
"""
    )
    result = check_stack(pipeline)
    assert not result.ok
    assert any(f.check == "capability" for f in result.errors)


def test_a_miswired_nested_provider_fails_the_sub_provider_check() -> None:
    """A Chunker named as an Export's embedder — checked from the CLASS, never constructed."""
    pipeline = Pipeline.from_yaml_text(
        """
name: nested-miswired
storage_uri: file:///tmp/x
stages:
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    config:
      embedder: {provider: chunk.markdown}
"""
    )
    result = check_stack(pipeline)
    assert not result.ok
    errors = [f for f in result.errors if f.check == "sub_provider"]
    assert errors and "Embedder" in errors[0].message
    # Reported ONCE: the capability guard's narrower embedder copy must not double-report it.
    assert len(errors) == 1


def test_an_uninstalled_provider_is_a_warning_not_an_error() -> None:
    """A CPU CI box legitimately lacks the GPU packages (ADR-0016) — that cannot be an error."""
    pipeline = Pipeline.from_yaml_text(
        """
name: gpu
storage_uri: file:///tmp/x
stages:
  - name: parse
    capability: parse
    provider: parser.no_such_vendor_ocr
"""
    )
    result = check_stack(pipeline)
    assert result.ok
    assert [f.severity for f in result.findings] == [Severity.WARNING]


def test_the_check_is_deterministic() -> None:
    """Identical input ⇒ byte-identical output (the framework-wide invariant)."""
    stack = _DEFAULT.read_text(encoding="utf-8")
    assert check_stack(stack).render() == check_stack(stack).render()


# -- The sub-Provider module -------------------------------------------------------------------


def test_the_blessed_sota_stack_nested_blocks_are_enumerable(tmp_path: Path) -> None:
    """The nested blocks the wizard's SOTA stack emits are found by ONE walk over the slot declaration.

    Enumerating them is what lets the checker, the device pre-flight and the placement
    probe all see them. Read from the GENERATED config, not a hand-written fixture, so a wizard
    wiring change cannot make this test lie.
    """
    from latence_core.setup import SetupAnswers, build_stack_config
    from latence_core.setup.wizard import PROFILE_SOTA

    src = tmp_path / "corpus"
    src.mkdir()
    (src / "a.pdf").write_text("a", encoding="utf-8")
    text = build_stack_config(
        SetupAnswers(
            source=f"file://{src}",
            storage=f"file://{tmp_path}/out",
            parser="parser.render",
            device="auto",
            pii=True,
            extract_on="unmasked",
            profile=PROFILE_SOTA,
            extensions=("pdf", "txt"),
        )
    )
    pipeline = Pipeline.from_yaml_text(text)
    found = {
        (stage.name, ref.key, ref.provider)
        for stage in pipeline.stages
        for ref in stage_sub_providers(stage)
    }
    assert ("disambiguate", "embedder", "embedding.sentence_transformers") in found
    assert ("export_corpus", "embedder", "embedding.sentence_transformers") in found
    # And the check has an opinion about them — the thing they never had.
    assert check_stack(pipeline).ok, check_stack(pipeline).render()


def test_the_declared_slot_keys_are_exactly_the_frameworks_nested_keys() -> None:
    """EXACT, not a superset: a superset assertion cannot notice a walker that fell behind.

    The former version of this test asserted a SUBSET ``<= declared_slot_keys()`` and stayed
    green for the whole time ``envspec`` kept walking its own hard-coded key tuple
    (R3-seams_real_not_hypothetical-0).
    """
    assert declared_slot_keys() == frozenset({"embedder", "ocr"})


def test_a_newly_declared_slot_is_walked_without_editing_envspec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE seam: ``envspec``'s stack walker reads THIS declaration, live, at walk time.

    Declaring a new nested slot here must be enough — no second key list to update, and no
    module-level snapshot taken at import that freezes the walker at the value it saw first. That
    is the property ``declared_slot_keys`` was introduced for and did not have: the walker kept its
    own tuple, so four of the seven declared slots were invisible to the GPU placement probe and to
    the env-stack constructibility gate.
    """
    from latence_core import subprovider
    from latence_core.capability import Embedder
    from latence_core.envspec import stack_components

    stack = tmp_path / "stack.yaml"
    stack.write_text(
        "name: s\nstages:\n"
        "  - name: export_corpus\n"
        "    capability: export\n"
        "    provider: export.jsonl_parquet\n"
        "    config:\n"
        "      reranker_embedder:\n"
        "        provider: embedding.brand_new\n"
        "        config: {device: cuda}\n",
        encoding="utf-8",
    )

    assert "embedding.brand_new" not in {c.provider for c in stack_components(stack)}

    slot = subprovider.SubProviderSlot("reranker_embedder", Embedder)
    monkeypatch.setitem(
        subprovider._SLOTS,  # noqa: SLF001 - the declaration under test
        CapabilityKind.EXPORT,
        (*subprovider.slots_for(CapabilityKind.EXPORT), slot),
    )

    walked = {c.provider: c for c in stack_components(stack)}
    assert "embedding.brand_new" in walked, (
        "envspec's walker did not see a slot declared in subprovider._SLOTS — it is carrying its "
        "own key list again"
    )
    assert walked["embedding.brand_new"].capability == "reranker_embedder"
    assert walked["embedding.brand_new"].config["device"] == "cuda"


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        (["not", "a", "mapping"], "must be a mapping"),
        ({"config": {}}, "non-empty string 'provider'"),
        ({"provider": ""}, "non-empty string 'provider'"),
        ({"provider": "x", "config": []}, "must be a mapping if present"),
    ],
)
def test_a_malformed_block_is_one_typed_config_error_at_every_slot(
    spec: Any, fragment: str
) -> None:
    """Same mistake, same type, same wording — whichever Capability's slot it is written under."""
    for capability, key in (
        (CapabilityKind.EXPORT, "embedder"),
        (CapabilityKind.PARSE, "ocr"),
        (CapabilityKind.TYPE_CONSOLIDATION, "embedder"),
    ):
        with pytest.raises(ConfigError, match=fragment):
            sub_provider_ref(spec, capability=capability, key=key)


def test_an_absent_block_parses_to_none() -> None:
    assert sub_provider_ref(None, capability=CapabilityKind.EXPORT, key="embedder") is None


def test_asking_for_an_undeclared_slot_is_a_programming_error_not_a_config_error() -> None:
    """Operator input and caller bugs must not look alike (ADR-0034: failures are typed)."""
    with pytest.raises(LookupError):
        sub_provider_ref({}, capability=CapabilityKind.EXPORT, key="ocr")


def test_undeclared_keys_are_detected_per_capability() -> None:
    export: Stage = Pipeline.from_yaml_text(
        """
name: x
storage_uri: file:///tmp/x
stages:
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    config:
      ocr: {provider: parser.pdfplumber}
      embedder: {provider: embedding.hashing}
      basename: corpus
"""
    ).stages[0]
    assert undeclared_sub_provider_keys(export) == ("ocr",)


def test_the_type_consolidator_capability_mismatch_is_a_contract_error() -> None:
    """One mistake, one exception type: the TC slot now matches the Export/Graph slots.

    It previously raised ``ConfigError`` for the same mistake the other two raised
    ``ContractError`` for — a difference a caller had to learn per call site.
    """
    from latence_core.stages.type_consolidation import CascadeTypeConsolidator

    provider = CascadeTypeConsolidator(
        {"use_embedding_merge": True, "embedder": {"provider": "chunk.markdown"}}
    )
    with pytest.raises(ContractError, match="Embedder capability"):
        provider.consolidate([])


def test_every_declared_slot_names_a_capability_protocol() -> None:
    """A slot whose Protocol is not a Protocol would silently accept anything."""
    from latence_core.capability import protocol_members

    for kind in CapabilityKind:
        for slot in slots_for(kind):
            assert protocol_members(slot.protocol), f"{kind}.{slot.key} declares no members"

"""The GPU-placement guard: a silent CPU fallback can never pass a GPU gate again (issue #198).

#191 is the bug this module exists to make unshippable. ``redaction.gliner_pii`` was configured
``device: cuda``, on a pod with a GPU, and ran every forward pass on the CPU — ~11.5 s/chunk, ~94
minutes for one Stage, the GPU idle throughout. It did not crash. It did not warn. Its OUTPUT WAS
CORRECT. Every assertion a test would normally reach for — the Provider loaded, the records are
right, the config says cuda — was true while the bug was live.

So these tests never assert any of those things. They assert the one fact a CPU-resident model
cannot fake: **the CUDA allocator did work**. A model on the GPU allocates activation memory on
every forward pass; a model on the CPU allocates exactly zero bytes, whatever it says about itself.

Offline by construction (ADR-0016 — core does not depend on torch): the allocator and the Provider
registry are injected. The fake allocator MIRRORS the real ``torch.cuda`` contract *including its
refusals* — real ``reset_peak_memory_stats()`` raises without an initialised CUDA device, so a fake
that returned quietly would manufacture a green the real thing cannot produce. That fidelity is
itself asserted below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from latence_core.envspec import EnvironmentSpec, load_environment
from latence_core.placement import (
    CUDA_ALLOCATION_FLOOR_BYTES,
    DRIVERS,
    probe_placement,
)

# --------------------------------------------------------------------------------------------
# The fakes.
# --------------------------------------------------------------------------------------------


class FakeCuda:
    """A faithful stand-in for the slice of ``torch.cuda`` the probe reads.

    ``allocate_per_call`` is how many bytes a driven Provider is pretended to allocate *transiently*
    (activations): a positive number models a GPU-resident forward pass, ``0`` models the #191
    silent CPU fallback.

    **The fidelity that matters most is the reset.** Real ``reset_peak_memory_stats()`` does NOT
    zero the peak — the allocator's ``Stat::reset_peak`` sets ``peak = current``. A fake that zeroed
    it would be strictly stricter than the dependency and would prove a property torch does not
    deliver ("a CPU-resident model allocates exactly zero"), hiding the fact that a model whose
    weights sit on CUDA carries its whole weight size into the peak. So this fake tracks resident
    bytes (:meth:`allocate_resident`, the weights) separately from transient ones
    (:meth:`note_forward_pass`, the activations) and resets the peak to the resident total.
    """

    def __init__(self, *, available: bool = True, allocate_per_call: int = 4_194_304) -> None:
        self._available = available
        self._allocate_per_call = allocate_per_call
        self._current = 0
        self._peak = 0
        self.synchronized = 0

    # -- the real contract, refusals included ---------------------------------------------
    def is_available(self) -> bool:
        return self._available

    def _require_device(self) -> None:
        """Real torch raises here without an initialised CUDA device; so does this."""
        if not self._available:
            msg = "Torch not compiled with CUDA enabled"
            raise RuntimeError(msg)

    def reset_peak_memory_stats(self) -> None:
        self._require_device()
        self._peak = self._current  # torch: Stat::reset_peak sets peak = current, NOT zero.

    def memory_allocated(self) -> int:
        self._require_device()
        return self._current

    def max_memory_allocated(self) -> int:
        self._require_device()
        return self._peak

    def synchronize(self) -> None:
        self._require_device()
        self.synchronized += 1

    # -- what a driven model does to it ----------------------------------------------------
    def allocate_resident(self, n_bytes: int) -> None:
        """Weights land on the GPU and STAY there — e.g. ``model.to('cuda')``."""
        self._current += n_bytes
        self._peak = max(self._peak, self._current)

    def note_forward_pass(self) -> None:
        """Activations: allocated during the pass, freed after it, visible only in the peak."""
        self._peak = max(self._peak, self._current + self._allocate_per_call)


class FakeProvider:
    """A Provider that implements every learned Capability method and records what was called."""

    driven: list[str] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _run(self, method: str, out: Any) -> Any:
        type(self).driven.append(method)
        cuda = type(self).cuda  # type: ignore[attr-defined]
        cuda.note_forward_pass()
        return out

    def embed(self, texts: Any) -> Any:
        return self._run("embed", [[0.1, 0.2] for _ in texts])

    def redact(self, chunks: Any) -> Any:
        return self._run("redact", list(chunks))

    def extract_fused(self, chunks: Any) -> Any:
        return self._run("extract_fused", list(chunks))

    def extract(self, chunks: Any) -> Any:
        return self._run("extract", [])

    def disambiguate(self, mentions: Any, relations: Any, chunks: Any = ()) -> Any:
        del relations, chunks
        return self._run("disambiguate", list(mentions))


class FakeRegistry:
    """The slice of ``ProviderRegistry`` the probe needs."""

    def __init__(self, cls: type) -> None:
        self._cls = cls

    def load(self, name: str) -> type:
        del name
        return self._cls


# --------------------------------------------------------------------------------------------
# A minimal, real environment on disk (the probe reads the spec + stack exactly as the CLI does).
# --------------------------------------------------------------------------------------------


def _write_env(
    tmp_path: Path,
    *,
    stages: str,
    learned: tuple[str, ...] = ("redaction.gliner_pii",),
    providers: tuple[str, ...] = ("redaction.gliner_pii",),
) -> EnvironmentSpec:
    env_dir = tmp_path / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "stack.yaml").write_text(f"name: s\nstages:\n{stages}", encoding="utf-8")
    (env_dir / "environment.yaml").write_text(
        "name: probe-env\nsummary: s\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: [packages/latence-core]\n"
        "providers:\n" + "".join(f"  - {p}\n" for p in providers) + "learned_providers:\n"
        "" + "".join(f"  - {p}\n" for p in learned),
        encoding="utf-8",
    )
    return load_environment(env_dir / "environment.yaml")


_REDACT_STAGE = (
    "  - name: redact\n"
    "    capability: redaction\n"
    "    provider: redaction.gliner_pii\n"
    "    config: {device: cuda}\n"
)


@pytest.fixture(autouse=True)
def _reset_fake_provider() -> None:
    FakeProvider.driven = []


def _probe(spec: EnvironmentSpec, cuda: FakeCuda) -> Any:
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    return probe_placement(spec, cuda=cuda, registry=FakeRegistry(FakeProvider))


# --------------------------------------------------------------------------------------------
# THE regression: the #191 shape — correct output, configured cuda, zero CUDA bytes.
# --------------------------------------------------------------------------------------------


def test_a_provider_that_ran_on_the_cpu_fails_the_gate(tmp_path: Path) -> None:
    """The Provider loads, is driven, returns correct records — and allocates nothing.

    That is #191 exactly. Every conventional assertion passes; only the allocator dissents.
    """
    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    result = _probe(spec, FakeCuda(allocate_per_call=0))

    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-ON-CPU"]
    # The Provider really was exercised — this is not a load failure dressed up as a placement one.
    # Twice: an unmeasured warm-up (which absorbs lazy weight placement) and the measured drive.
    assert FakeProvider.driven == ["redact", "redact"]
    assert result.placements[0].cuda_bytes == 0
    assert not result.placements[0].on_gpu


def test_weights_resident_on_cuda_are_not_mistaken_for_gpu_work(tmp_path: Path) -> None:
    """THE R2 regression: ``model.to('cuda')`` in ``__init__``, forward pass entirely on the CPU.

    Real ``reset_peak_memory_stats()`` sets peak = CURRENT, so the 1.1 GB of weights placed at
    construction is already inside the peak before the probe drives anything. Reading the raw
    ``max_memory_allocated()`` therefore reported 1.1 GB of "GPU work" for a model that did every
    forward pass on the CPU — the #191 shape passing the gate built to make it unshippable. Only
    the delta above the resident baseline separates weights from work.
    """

    class WeightsOnCudaComputeOnCpu(FakeProvider):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)
            type(self).cuda.allocate_resident(1_100_000_000)  # type: ignore[attr-defined]

        def redact(self, chunks: Any) -> Any:
            type(self).driven.append("redact")
            return list(chunks)  # correct output, computed on the CPU: allocates nothing.

    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda()
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    result = probe_placement(spec, cuda=cuda, registry=FakeRegistry(WeightsOnCudaComputeOnCpu))

    assert not result.ok, result.describe()
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-ON-CPU"]
    assert result.placements[0].cuda_bytes == 0
    assert not result.placements[0].on_gpu


def test_weights_loaded_lazily_onto_cuda_are_not_mistaken_for_gpu_work(tmp_path: Path) -> None:
    """The same hole one call later: the model reaches CUDA on its FIRST inference, then computes
    on the CPU. An unmeasured warm-up drive is what puts that placement outside the measurement.
    """

    class LazyLoadThenCpu(FakeProvider):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)
            self._loaded = False

        def redact(self, chunks: Any) -> Any:
            type(self).driven.append("redact")
            if not self._loaded:
                type(self).cuda.allocate_resident(900_000_000)  # type: ignore[attr-defined]
                self._loaded = True
            return list(chunks)

    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda()
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    result = probe_placement(spec, cuda=cuda, registry=FakeRegistry(LazyLoadThenCpu))

    assert not result.ok, result.describe()
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-ON-CPU"]
    assert result.placements[0].cuda_bytes == 0


def test_a_gpu_resident_model_still_passes_when_its_weights_are_on_cuda(tmp_path: Path) -> None:
    """The delta must not turn the honest case red: weights on CUDA *and* activations per pass."""

    class WeightsAndActivations(FakeProvider):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)
            type(self).cuda.allocate_resident(1_100_000_000)  # type: ignore[attr-defined]

    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda(allocate_per_call=8_388_608)
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    result = probe_placement(spec, cuda=cuda, registry=FakeRegistry(WeightsAndActivations))

    assert result.ok, result.describe()
    # Exactly the activations, not the weights: the gate reports work, not residency.
    assert result.placements[0].cuda_bytes == 8_388_608


def test_a_provider_that_really_used_the_gpu_passes(tmp_path: Path) -> None:
    """The green case, so the failing case above is causal and not a permanently-red gate."""
    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    result = _probe(spec, FakeCuda(allocate_per_call=8_388_608))

    assert result.ok, result.describe()
    assert result.placements[0].on_gpu
    assert result.placements[0].cuda_bytes >= CUDA_ALLOCATION_FLOOR_BYTES
    assert "GPU" in result.describe()


def test_one_allocated_byte_is_the_whole_threshold(tmp_path: Path) -> None:
    """The gate answers a binary question (#191), not a throughput one (#180).

    A single byte separates 'the GPU did work' from 'it did not', so the floor cannot be tuned
    upward into a pass or downward into a meaningless one.
    """
    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    assert _probe(spec, FakeCuda(allocate_per_call=1)).ok
    assert not _probe(spec, FakeCuda(allocate_per_call=0)).ok


# --------------------------------------------------------------------------------------------
# It fails; it never skips.
# --------------------------------------------------------------------------------------------


def test_no_cuda_device_is_a_failure_not_a_skip(tmp_path: Path) -> None:
    """A GPU gate that reports success on a CPU host is the false green this ticket prevents."""
    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    result = _probe(spec, FakeCuda(available=False))

    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-NO-CUDA"]
    # Nothing was driven and nothing was measured — the gate refuses rather than inventing a number.
    assert FakeProvider.driven == []
    assert result.placements == ()


def test_a_learned_provider_with_no_driver_is_reported_not_ignored(tmp_path: Path) -> None:
    """Unexercised is not passed: an undriven Provider is where a CPU fallback would hide."""
    stage = (
        "  - name: mystery\n"
        "    capability: profiling\n"
        "    provider: redaction.gliner_pii\n"
        "    config: {device: cuda}\n"
    )
    spec = _write_env(tmp_path, stages=stage)
    result = _probe(spec, FakeCuda())

    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-UNDRIVABLE"]


def test_a_learned_provider_pinned_to_cpu_fails_the_gpu_gate(tmp_path: Path) -> None:
    """A cuda gate over a cpu-pinned Stage would prove nothing, so it is refused up front."""
    spec = _write_env(tmp_path, stages=_REDACT_STAGE.replace("{device: cuda}", "{device: cpu}"))
    result = _probe(spec, FakeCuda())

    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-DEVICE-NOT-CUDA"]
    assert FakeProvider.driven == []


def test_a_provider_that_raises_while_driven_is_a_finding(tmp_path: Path) -> None:
    """An inference fault must not be mistaken for a placement verdict."""

    class Exploding(FakeProvider):
        def redact(self, chunks: Any) -> Any:
            del chunks
            msg = "CUDA error: no kernel image is available for execution on the device"
            raise RuntimeError(msg)

    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda()
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    result = probe_placement(spec, cuda=cuda, registry=FakeRegistry(Exploding))

    assert not result.ok
    codes = [f.code for f in result.report.findings]
    assert codes == ["E-PLACEMENT-FAILED"]
    assert "no kernel image" in result.report.findings[0].message


def test_a_provider_that_cannot_be_constructed_is_a_finding(tmp_path: Path) -> None:
    """A construction fault is reported, never swallowed into an implicit pass."""

    class Unbuildable:
        def __init__(self, config: dict[str, Any]) -> None:
            del config
            msg = "pii_labels is required"
            raise ValueError(msg)

    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    result = probe_placement(spec, cuda=FakeCuda(), registry=FakeRegistry(Unbuildable))

    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-FAILED"]


# --------------------------------------------------------------------------------------------
# Scope + fidelity.
# --------------------------------------------------------------------------------------------


def test_only_declared_learned_providers_are_probed(tmp_path: Path) -> None:
    """A chunker owns no model, so demanding CUDA allocation of it would be noise, not a guard."""
    stages = _REDACT_STAGE + (
        "  - name: chunk\n"
        "    capability: chunk\n"
        "    provider: chunk.markdown\n"
        "    config: {max_tokens: 640}\n"
    )
    spec = _write_env(
        tmp_path,
        stages=stages,
        learned=("redaction.gliner_pii",),
        providers=("redaction.gliner_pii", "chunk.markdown"),
    )
    result = _probe(spec, FakeCuda())

    assert result.ok, result.describe()
    assert [p.provider for p in result.placements] == ["redaction.gliner_pii"]


@pytest.mark.parametrize(
    ("capability", "method"),
    [
        ("redaction", "redact"),
        ("fused_entity_relation", "extract_fused"),
        ("entity_extraction", "extract"),
        ("disambiguation", "disambiguate"),
        ("embedder", "embed"),
    ],
)
def test_each_capability_is_driven_through_its_real_protocol_method(
    tmp_path: Path, capability: str, method: str
) -> None:
    """The drivers call the PUBLIC Capability method, so the probe measures a real forward pass.

    A driver that touched a private loader instead would measure model loading, not inference —
    and a model can be loaded onto the GPU and still have its forward run somewhere else.
    """
    stage = (
        "  - name: s\n"
        f"    capability: {capability}\n"
        "    provider: p.fake\n"
        "    config: {device: cuda}\n"
    )
    spec = _write_env(tmp_path, stages=stage, learned=("p.fake",), providers=("p.fake",))
    result = _probe(spec, FakeCuda())

    assert result.ok, result.describe()
    assert FakeProvider.driven == [method, method]  # warm-up + measured drive


def test_generators_are_fully_consumed_so_lazy_providers_really_run(tmp_path: Path) -> None:
    """Most Capability methods return an Iterator; an unconsumed one does no work at all.

    A driver that merely *called* ``redact`` and dropped the generator would measure zero CUDA
    bytes for a perfectly GPU-resident Provider — a false RED — or, worse, zero for a CPU one and
    call it inconclusive. The drivers consume; this proves it.
    """

    class Lazy(FakeProvider):
        def redact(self, chunks: Any) -> Any:
            for chunk in chunks:
                type(self).driven.append("redact")
                type(self).cuda.note_forward_pass()  # type: ignore[attr-defined]
                yield chunk

    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda()
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    result = probe_placement(spec, cuda=cuda, registry=FakeRegistry(Lazy))

    assert result.ok, result.describe()
    assert FakeProvider.driven == ["redact", "redact"]  # warm-up + measured drive


def test_the_fake_allocator_refuses_what_real_torch_refuses() -> None:
    """A fake that accepts what the real API rejects manufactures false green — so this one does not.

    Real ``torch.cuda.reset_peak_memory_stats()`` / ``max_memory_allocated()`` raise without an
    initialised CUDA device. If :class:`FakeCuda` returned 0 quietly instead, every test above
    would still be green while the probe silently reported "ran on CPU" for a host that simply has
    no GPU — the exact conflation this gate must not make.
    """
    cuda = FakeCuda(available=False)
    for call in (
        cuda.reset_peak_memory_stats,
        cuda.memory_allocated,
        cuda.max_memory_allocated,
        cuda.synchronize,
    ):
        with pytest.raises(RuntimeError, match="CUDA"):
            call()


def test_the_fake_allocator_resets_the_peak_to_current_like_real_torch() -> None:
    """Real ``reset_peak_memory_stats()`` sets peak = CURRENT (``Stat::reset_peak``), not zero.

    This is the fidelity the whole gate rests on. A fake that zeroed the peak would be stricter
    than the dependency and would make every test below prove "a CPU-resident model allocates
    exactly zero" — which real torch does not deliver for a model whose weights sit on CUDA.
    """
    cuda = FakeCuda(allocate_per_call=0)
    cuda.allocate_resident(1_100_000_000)
    cuda.reset_peak_memory_stats()

    assert cuda.memory_allocated() == 1_100_000_000
    assert cuda.max_memory_allocated() == 1_100_000_000  # NOT 0 — the resident weights survive.


def test_the_measurement_is_read_after_a_synchronize(tmp_path: Path) -> None:
    """CUDA kernels are queued; a peak read before a sync can under-report a real GPU run."""
    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda()
    _probe(spec, cuda)
    assert cuda.synchronized == 2  # once after the warm-up, once after the measured drive


def test_every_learned_capability_of_the_shipped_gpu_stack_has_a_driver() -> None:
    """The committed GPU environment must be fully drivable, or its gate has a hole in it."""
    from latence_core.envspec import stack_components

    repo_root = Path(__file__).resolve().parents[3]
    spec = load_environment(repo_root / "deploy" / "env" / "gpu-learned" / "environment.yaml")
    for component in stack_components(spec.stack_path):
        if component.provider in spec.learned_providers:
            assert component.capability in DRIVERS, component.provider


# --------------------------------------------------------------------------------------------
# R5-env_reproducibility-3 — the probe set is DERIVED, so omitting a line of YAML cannot
# silently drop a model-loading Provider out of the #191 guard.
# --------------------------------------------------------------------------------------------

_EMBED_STAGE = (
    "  - name: embed\n"
    "    capability: embedding\n"
    "    provider: embedding.learned\n"
    "    config: {device: cuda}\n"
)


def _write_workspace(tmp_path: Path) -> Path:
    """A minimal repo root whose `latence-embedder-learned` package declares a MODEL RUNTIME dependency.

    Nothing here is a fake of the derivation: `model_loading_providers` reads exactly this
    metadata — the package's own `dependencies` and its `latence.providers` entry points — off a
    real `pyproject.toml`, which is what makes the probe set structural rather than declared.
    """
    repo_root = tmp_path / "repo"
    pkg = repo_root / "packages" / "latence-embedder-learned"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "pyproject.toml").write_text(
        "[project]\n"
        'name = "latence-embedder-learned"\n'
        'dependencies = ["latence-core", "transformers>=4.44"]\n'
        '[project.entry-points."latence.providers"]\n'
        '"embedding.learned" = "latence_embedder_learned:Learned"\n',
        encoding="utf-8",
    )
    return repo_root


def _write_derived_env(
    tmp_path: Path, *, learned: tuple[str, ...], stages: str = _EMBED_STAGE
) -> EnvironmentSpec:
    """An environment that INSTALLS `latence-embedder-learned` and drives it, with `learned` as declared."""
    env_dir = tmp_path / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "stack.yaml").write_text(f"name: s\nstages:\n{stages}", encoding="utf-8")
    (env_dir / "environment.yaml").write_text(
        "name: probe-env\nsummary: s\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: [packages/latence-embedder-learned]\n"
        "providers:\n  - embedding.learned\n"
        "learned_providers:\n" + "".join(f"  - {p}\n" for p in learned),
        encoding="utf-8",
    )
    return load_environment(env_dir / "environment.yaml")


def _probe_root(spec: EnvironmentSpec, repo_root: Path, cuda: FakeCuda) -> Any:
    FakeProvider.cuda = cuda  # type: ignore[attr-defined]
    return probe_placement(
        spec, repo_root=repo_root, cuda=cuda, registry=FakeRegistry(FakeProvider)
    )


def test_a_model_loading_provider_absent_from_learned_providers_is_still_probed(
    tmp_path: Path,
) -> None:
    """THE R5 regression: adding a CUDA Provider WITHOUT touching `learned_providers`.

    A routine change — new package in `workspace_packages`, new name under `providers`, new stage
    with `device: cuda` — and the one optional field left alone. Pre-fix the probe read that field
    alone, so `embedding.learned` was never loaded, never driven, never measured, and emitted NO
    finding: the gate that exists to make #191 unshippable reported green over an unmeasured CUDA
    Provider. The derivation reads the package's `transformers` dependency instead, so the omission
    can no longer buy silence.
    """
    repo_root = _write_workspace(tmp_path)
    spec = _write_derived_env(tmp_path, learned=())
    assert spec.learned_providers == frozenset()  # the silence this test exists to survive

    # A forward pass that allocated nothing: the #191 shape, which must now be REPORTED.
    result = _probe_root(spec, repo_root, FakeCuda(allocate_per_call=0))

    assert [p.provider for p in result.placements] == ["embedding.learned"]
    assert FakeProvider.driven == ["embed", "embed"]
    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-ON-CPU"]


def test_deleting_a_learned_providers_entry_cannot_drop_it_from_the_probe_set(
    tmp_path: Path,
) -> None:
    """Removal is the same hole as addition: the declaration may only ADD to the probe set.

    Pre-fix, deleting one line from `learned_providers` silently removed that Provider from the
    gate while `latence env verify` stayed OK — `E-LEARNED-UNKNOWN` and `E-LEARNED-UNDRIVEN` both
    key off the very field that was emptied.
    """
    repo_root = _write_workspace(tmp_path)
    declared = _write_derived_env(tmp_path, learned=("embedding.learned",))
    undeclared = _write_derived_env(tmp_path, learned=())

    with_declaration = _probe_root(declared, repo_root, FakeCuda())
    FakeProvider.driven = []
    without_declaration = _probe_root(undeclared, repo_root, FakeCuda())

    assert [p.provider for p in with_declaration.placements] == ["embedding.learned"]
    assert [p.provider for p in without_declaration.placements] == ["embedding.learned"]
    assert with_declaration.ok and without_declaration.ok


def test_an_empty_probe_set_is_a_finding_not_a_green(tmp_path: Path) -> None:
    """A placement gate that measured nothing must never report success.

    With `learned_providers: []` and no model-loading package, the probe used to return `ok=True`
    with zero placements and zero findings — green earned by measuring nothing, which is the exact
    shape of the bug the gate exists to catch.
    """
    repo_root = tmp_path / "repo"  # no packages at all: nothing loads a model here
    spec = _write_derived_env(tmp_path, learned=())
    result = _probe_root(spec, repo_root, FakeCuda())

    assert not result.ok
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-NONE"]
    assert result.placements == ()


def test_a_probe_set_no_stack_component_matches_is_a_finding_not_a_green(tmp_path: Path) -> None:
    """The OTHER route to the same false green: a probe set nothing in the stack instantiates.

    R6-env_reproducibility-5. The guard used to sit on `if not probe_set` — one CAUSE of measuring
    nothing rather than the fact of it — which left the measurement loop as an unguarded second
    route to the identical result. Here the derivation finds `embedding.learned` (the installed package
    depends on `transformers`), so the probe set is NON-empty and `E-PLACEMENT-NONE` did not fire;
    but the stack drives only `source.fs`, so every iteration hit `continue`, no finding was
    appended, and the gate returned `ok=True` with zero placements: green earned by measuring
    nothing, the exact shape `E-PLACEMENT-NONE` exists to forbid.

    This is `deploy/env/gpu-learned` with its learned Stages removed while its Providers stay
    installed and declared: gate 3 of `smoke.sh` would exit 0 printing `OK (no findings)` without a
    single measured forward pass.
    """
    repo_root = _write_workspace(tmp_path)  # -> a NON-empty derived probe set: {'embedding.learned'}
    spec = _write_derived_env(
        tmp_path,
        learned=(),
        stages="  - capability: source\n    provider: source.fs\n",  # nothing learned is driven
    )
    from latence_core.envspec import placement_probe_providers

    assert placement_probe_providers(spec, repo_root) == {"embedding.learned"}  # so NOT the empty case

    FakeProvider.driven = []
    result = _probe_root(spec, repo_root, FakeCuda())

    assert FakeProvider.driven == []  # not one forward pass was driven...
    assert result.placements == ()
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-NONE"]
    assert "embedding.learned" in result.report.findings[0].message  # ...and the skipped name is said
    assert not result.ok  # ...so the gate must NOT be green


def test_a_measured_failure_is_not_reported_as_having_measured_nothing(tmp_path: Path) -> None:
    """The `E-PLACEMENT-NONE` guard keys off what was PROBED, not off what was successfully placed.

    A component the probe REJECTED — pinned to `device: cpu`, so `E-PLACEMENT-DEVICE-NOT-CUDA` —
    yields no `Placement`, so `placements` is empty here exactly as it is when nothing was probed at
    all. Guarding on empty placements instead of on empty probing would bury that finding under a
    second, false one claiming the gate measured nothing, when it examined precisely the component
    it exists to examine.
    """
    repo_root = _write_workspace(tmp_path)
    spec = _write_derived_env(
        tmp_path,
        learned=(),
        stages=(
            "  - name: embed\n"
            "    capability: embedding\n"
            "    provider: embedding.learned\n"
            "    config: {device: cpu}\n"
        ),
    )

    result = _probe_root(spec, repo_root, FakeCuda())

    assert result.placements == ()  # the same empty placements an unprobed run has...
    assert [f.code for f in result.report.findings] == ["E-PLACEMENT-DEVICE-NOT-CUDA"]  # ...but
    assert not result.ok


def test_the_shipped_gpu_probe_set_is_unchanged_by_the_derivation() -> None:
    """The committed GPU environment's derived set EQUALS its declared set — today.

    The fix removes the ability to lose a Provider from the gate; it must not quietly change what
    the shipped gate measures. If these ever diverge, `deploy/env/gpu-learned` has grown a
    model-loading Provider its `learned_providers` never mentioned — which is precisely the case
    this ticket makes visible rather than silent.
    """
    from latence_core.envspec import placement_probe_providers

    repo_root = Path(__file__).resolve().parents[3]
    spec = load_environment(repo_root / "deploy" / "env" / "gpu-learned" / "environment.yaml")
    assert placement_probe_providers(spec, repo_root) == spec.learned_providers


# --------------------------------------------------------------------------------------------
# R3-seams_real_not_hypothetical-0 — the probe iterates `stack_components`, so a nested slot the
# walker cannot see is a GPU model the gate silently never drives.
# --------------------------------------------------------------------------------------------


def test_a_nested_gpu_model_cannot_be_skipped_into_a_green_report(tmp_path: Path) -> None:
    """The documented universal-parse shape: a Parse Stage whose OCR model is nested under ``ocr:``.

    Pre-fix, ``envspec.stack_components`` walked a hard-coded tuple of nested keys that did not list ``ocr``, so the ``ocr`` block — the ONLY GPU-gated node of that stack —
    was invisible here. The loop below ``continue``d past it without appending a finding, and the
    ``E-PLACEMENT-NONE`` guard did not fire either, because the sibling redaction Stage *was*
    probed. The report was a genuine ``ok=True`` earned by measuring the wrong set of models: #191
    with an extra level of nesting.

    Post-fix the nested model is enumerated. It has no driver (``DRIVERS`` knows text-in/vector-out
    Capabilities, not OCR), so the gate says ``E-PLACEMENT-UNDRIVABLE`` and names it — unexercised
    is not passed. What must never happen again is silence.
    """
    stages = _REDACT_STAGE + (
        "  - name: parse\n"
        "    capability: parse\n"
        "    provider: parser.render\n"
        "    config:\n"
        "      device: cuda\n"
        "      ocr:\n"
        "        provider: parser.lighton\n"
        "        config: {device: cuda}\n"
    )
    spec = _write_env(
        tmp_path,
        stages=stages,
        learned=("redaction.gliner_pii", "parser.lighton"),
        providers=("redaction.gliner_pii", "parser.render", "parser.lighton"),
    )

    result = _probe(spec, FakeCuda(allocate_per_call=8_388_608))

    codes = [f.code for f in result.report.findings]
    assert not result.ok, (
        "the nested OCR model was never driven, yet the gate reported success: "
        f"{result.describe()}"
    )
    assert codes == ["E-PLACEMENT-UNDRIVABLE"], codes
    assert "parser.lighton" in result.report.findings[0].message

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

    ``allocate_per_call`` is how many bytes a driven Provider is pretended to allocate: a positive
    number models a GPU-resident model, ``0`` models the #191 silent CPU fallback.
    """

    def __init__(self, *, available: bool = True, allocate_per_call: int = 4_194_304) -> None:
        self._available = available
        self._allocate_per_call = allocate_per_call
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
        self._peak = 0

    def max_memory_allocated(self) -> int:
        self._require_device()
        return self._peak

    def synchronize(self) -> None:
        self._require_device()
        self.synchronized += 1

    # -- what a driven model does to it ----------------------------------------------------
    def note_forward_pass(self) -> None:
        self._peak = max(self._peak, self._allocate_per_call)


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

    def encode_sparse(self, texts: Any) -> Any:
        return self._run("encode_sparse", [{} for _ in texts])

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
    assert FakeProvider.driven == ["redact"]
    assert result.placements[0].cuda_bytes == 0
    assert not result.placements[0].on_gpu


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
        ("sparse_embedder", "encode_sparse"),
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
    assert FakeProvider.driven == [method]


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
    assert FakeProvider.driven == ["redact"]


def test_the_fake_allocator_refuses_what_real_torch_refuses() -> None:
    """A fake that accepts what the real API rejects manufactures false green — so this one does not.

    Real ``torch.cuda.reset_peak_memory_stats()`` / ``max_memory_allocated()`` raise without an
    initialised CUDA device. If :class:`FakeCuda` returned 0 quietly instead, every test above
    would still be green while the probe silently reported "ran on CPU" for a host that simply has
    no GPU — the exact conflation this gate must not make.
    """
    cuda = FakeCuda(available=False)
    for call in (cuda.reset_peak_memory_stats, cuda.max_memory_allocated, cuda.synchronize):
        with pytest.raises(RuntimeError, match="CUDA"):
            call()


def test_the_measurement_is_read_after_a_synchronize(tmp_path: Path) -> None:
    """CUDA kernels are queued; a peak read before a sync can under-report a real GPU run."""
    spec = _write_env(tmp_path, stages=_REDACT_STAGE)
    cuda = FakeCuda()
    _probe(spec, cuda)
    assert cuda.synchronized == 1


def test_every_learned_capability_of_the_shipped_gpu_stack_has_a_driver() -> None:
    """The committed GPU environment must be fully drivable, or its gate has a hole in it."""
    from latence_core.envspec import stack_components

    repo_root = Path(__file__).resolve().parents[3]
    spec = load_environment(repo_root / "deploy" / "env" / "gpu-learned" / "environment.yaml")
    for component in stack_components(spec.stack_path):
        if component.provider in spec.learned_providers:
            assert component.capability in DRIVERS, component.provider

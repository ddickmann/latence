"""The **GPU-placement probe** — proof that a learned Provider really ran on the GPU (issue #198).

**The bug this exists to make unreproducible.** #191: ``redaction.gliner_pii`` was configured with
``device: cuda``, the pod had a GPU, the Provider loaded, the Stage produced *correct output* — and
every forward pass ran on the CPU at roughly 11.5 s/chunk with the GPU idle. Nothing failed.
Nothing warned. The only symptom was a 94-minute Stage. ``place_on_device`` (the #191 fix) closed
that hole for the one Provider whose model exposes a readable ``.device``; it cannot close it for
the ones that *don't*. ``disambiguation.glinker`` in particular degrades to a CPU fallback path on
any backend load failure **by design** — a correct-output, wrong-device run is its documented
graceful-degradation behaviour, which is precisely the shape #191 wore.

**So the probe does not ask the model where it is. It measures whether the GPU did any work.**
Nor does it ask the spec *which* components to measure: the probe set is derived from the workspace
packages' own dependencies and only then unioned with the optional ``learned_providers:``
declaration (:func:`~latence_core.envspec.placement_probe_providers`), so a model-loading Provider
cannot be dropped out of the gate by omitting a line of YAML (R5-env_reproducibility-3). For each
such component, it drives real inferences through the Provider's *public* Capability method and
reads torch's CUDA allocator around the call:

    drive once (warm-up, UNMEASURED)
    reset_peak_memory_stats()  →  baseline = memory_allocated()
    drive again  →  synchronize()  →  max_memory_allocated() - baseline

**Why a delta and not the raw peak.** Real ``torch.cuda.reset_peak_memory_stats()`` does not zero
the peak — the allocator's ``Stat::reset_peak`` sets ``peak = current``. A Provider that puts its
weights on CUDA (in ``__init__``, or lazily on its first call) therefore carries its whole weight
size into any raw ``max_memory_allocated()`` reading, and would pass this gate while every forward
pass ran on the CPU — the #191 bug wearing the gate's own green. The warm-up drive pushes weight
placement out of the measured window, and subtracting the resident baseline leaves only what the
measured forward pass itself allocated.

A model resident on the GPU allocates activation memory on every forward pass. A model that
silently stayed on the CPU allocates **zero bytes above its resident footprint**, no matter what it
reports about itself, what it logs, or how correct its output is. That number is behaviour, not
intent — it cannot be satisfied by a flag, a config echo, or a Provider that claims placement it
does not have.

**It fails; it never skips.** No CUDA device, an undrivable component, a Provider that raises —
each is a *finding*. A GPU smoke test that quietly narrows its own scope when the GPU is missing is
the false green this whole ticket exists to prevent.

Dependency posture (ADR-0016): ``latence-core`` does not depend on torch, so torch and the Provider
registry are **accepted, not created** — :func:`probe_placement` takes them as arguments and
defaults to the real ones only at the call boundary. That is also what makes the probe's own
regression tests offline: they inject a fake allocator and a fake Provider and assert the probe
reports the #191 shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    PageMap,
    PageSpan,
    Provenance,
)
from latence_core.envspec import EnvironmentReport, EnvironmentSpec, Finding, StackComponent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from pathlib import Path

# The device the probe demands. A learned component pinned to anything else in a GPU environment is
# a finding, not a preference: the environment exists to run these models on the GPU.
CUDA = "cuda"

# The smallest CUDA allocation that counts as "the GPU did work". One byte: the assertion is
# emphatically NOT a throughput threshold (that is #180's job) — it is the binary question #191
# answered wrong. A real forward pass allocates activations in the megabytes ABOVE the model's
# resident weights; a forward pass that ran on the CPU allocates exactly zero above them. The
# measurement is that delta (see the module docstring), so a floor of 1 cannot be tuned into a pass
# and resident weights cannot be mistaken for work.
CUDA_ALLOCATION_FLOOR_BYTES = 1


class CudaAllocator(Protocol):
    """The slice of ``torch.cuda`` the probe reads — the seam that keeps torch out of core.

    Mirrors the real module's contract exactly, including its refusals: real
    ``torch.cuda.reset_peak_memory_stats()`` raises when no CUDA device is initialised, so a fake
    that returns quietly would manufacture a green the real thing cannot produce. It also mirrors
    the semantics that make the delta necessary — ``reset_peak_memory_stats()`` sets the peak to the
    CURRENT allocation, not to zero — which is why ``memory_allocated()`` is part of this seam.
    """

    def is_available(self) -> bool: ...
    def reset_peak_memory_stats(self) -> None: ...
    def memory_allocated(self) -> int: ...
    def max_memory_allocated(self) -> int: ...
    def synchronize(self) -> None: ...


class ProviderLoader(Protocol):
    """The slice of :class:`~latence_core.capability.ProviderRegistry` the probe needs."""

    def load(self, name: str) -> type: ...


@dataclass(frozen=True)
class Placement:
    """One component's MEASURED placement: what the GPU actually did while it ran."""

    provider: str
    capability: str
    cuda_bytes: int
    """Peak CUDA bytes the driven inference allocated ABOVE the Provider's resident footprint.

    Weights already on the GPU do not count — only what the measured forward pass itself allocated.
    Zero means the CPU did the work, however much of the model happens to be resident on CUDA.
    """

    @property
    def on_gpu(self) -> bool:
        return self.cuda_bytes >= CUDA_ALLOCATION_FLOOR_BYTES


@dataclass(frozen=True)
class PlacementReport:
    """The probe's result: the findings, plus the measurements behind them."""

    report: EnvironmentReport
    placements: tuple[Placement, ...]

    @property
    def ok(self) -> bool:
        return self.report.ok

    def describe(self) -> str:
        lines = [self.report.describe()]
        lines.extend(
            f"  {'GPU' if p.on_gpu else 'CPU'}  {p.provider} ({p.capability}): "
            f"{p.cuda_bytes} peak CUDA bytes"
            for p in self.placements
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# The drivers: one real inference per learned Capability, through the PUBLIC Protocol method.
# --------------------------------------------------------------------------------------------

_PROBE_TEXT = (
    "Acme Corporation and International Business Machines announced a partnership. "
    "Contact Dr. Erika Mustermann at erika.mustermann@acme.example or +49 30 123456."
)


def _probe_chunk() -> ChunkRecord:
    """A single, self-contained chunk — enough text for every learned Stage to do real work."""
    return ChunkRecord(
        record_id="placement-probe-chunk-0",
        provenance=Provenance(
            source_uri="file://placement-probe/doc.txt",
            source_system="placement-probe",
            file_name="doc.txt",
            file_type="txt",
            file_size=len(_PROBE_TEXT),
            document_id="placement-probe-doc",
            char_start=0,
            char_end=len(_PROBE_TEXT),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="de", sensitivity="internal"),
        document_record_id="placement-probe-doc",
        index=0,
        content=_PROBE_TEXT,
        token_count=len(_PROBE_TEXT.split()),
        page_map=PageMap(
            pages=[PageSpan(page_number=1, char_start=0, char_end=len(_PROBE_TEXT))],
            total_chars=len(_PROBE_TEXT),
        ),
    )


def _probe_mentions(chunk: ChunkRecord) -> list[EntityMention]:
    """Two mentions of two organisations — the minimum a Disambiguator has to reason over."""
    surfaces = (("Acme Corporation", 0), ("International Business Machines", 21))
    return [
        EntityMention(
            record_id=f"placement-probe-mention-{index}",
            provenance=chunk.provenance,
            classification=chunk.classification,
            document_record_id=chunk.document_record_id,
            chunk_record_id=chunk.record_id,
            index=index,
            label="organization",
            text=surface,
            confidence=1.0,
        )
        for index, (surface, _offset) in enumerate(surfaces)
    ]


def _drive_embedding(provider: Any) -> None:
    list(provider.embed([_PROBE_TEXT]))


def _drive_sparse(provider: Any) -> None:
    list(provider.encode_sparse([_PROBE_TEXT]))


def _drive_redaction(provider: Any) -> None:
    list(provider.redact([_probe_chunk()]))


def _drive_fused(provider: Any) -> None:
    provider.extract_fused([_probe_chunk()])


def _drive_entity_extraction(provider: Any) -> None:
    list(provider.extract([_probe_chunk()]))


def _drive_disambiguation(provider: Any) -> None:
    chunk = _probe_chunk()
    list(provider.disambiguate(_probe_mentions(chunk), [], [chunk]))


# Capability (or nested-component key) → the one public call that makes the model do real work.
# A learned component whose capability is absent here is REPORTED, never skipped: an unexercised
# Provider in a GPU environment is exactly the blind spot #191 hid in.
DRIVERS: dict[str, Callable[[Any], None]] = {
    "embedder": _drive_embedding,
    "embedding": _drive_embedding,
    "sparse_embedder": _drive_sparse,
    "sparse": _drive_sparse,
    "redaction": _drive_redaction,
    "fused_entity_relation": _drive_fused,
    "entity_extraction": _drive_entity_extraction,
    "disambiguation": _drive_disambiguation,
}


def _cuda_module() -> CudaAllocator:
    import torch  # noqa: PLC0415 - optional heavy dep; core must import without it (ADR-0016)

    cuda: CudaAllocator = torch.cuda
    return cuda


def probe_placement(
    spec: EnvironmentSpec,
    *,
    repo_root: Path | None = None,
    cuda: CudaAllocator | None = None,
    registry: ProviderLoader | None = None,
) -> PlacementReport:
    """Drive every model-loading component of ``spec``'s stack and PROVE the GPU did the work.

    Only components that own a model are probed — a stack's ``source`` / ``chunk`` / ``export``
    Stages own none, so demanding CUDA allocation from them would be noise. Everything that *is*
    probed must produce a non-zero CUDA allocation or it is a finding.

    **Which components those are is DERIVED, not asked for.** The probe set is
    :func:`~latence_core.envspec.placement_probe_providers`: the optional, hand-written
    ``learned_providers:`` field UNIONED with the set derived from the declaring packages' own
    dependencies. Reading the field alone (as this did until R5-env_reproducibility-3) made the
    #191 guard ask the spec's author whether to guard — a model-loading Provider added to a GPU
    stack with ``device: cuda`` and not added to the field was silently skipped and emitted no
    finding in any of the four smoke gates. ``repo_root`` is where the derivation reads the
    workspace packages' metadata; it defaults to the same three-parents-up rule ``latence env
    verify`` uses.

    Findings:

    * ``E-PLACEMENT-NO-CUDA`` — no CUDA device. The probe FAILS rather than skipping; a GPU smoke
      that reports success on a CPU host is the false green this ticket exists to prevent.
    * ``E-PLACEMENT-NONE`` — the probe set is empty, so the gate would report green having measured
      nothing. Also a failure, for the same reason.
    * ``E-PLACEMENT-DEVICE-NOT-CUDA`` — a learned component is pinned to ``cpu`` in a GPU
      environment's stack, so the run would prove nothing.
    * ``E-PLACEMENT-UNDRIVABLE`` — no driver for the component's capability. Unexercised is not
      passed.
    * ``E-PLACEMENT-FAILED`` — the Provider raised while being driven.
    * ``E-PLACEMENT-ON-CPU`` — **the #191 signature**: the Provider ran, produced output, and
      allocated zero CUDA bytes above its resident footprint doing it. Weights parked on the GPU do
      not rescue it; only the measured forward pass's own allocations count.
    """
    from latence_core.capability import ProviderRegistry  # noqa: PLC0415 - avoids an import cycle
    from latence_core.envspec import (  # noqa: PLC0415 - avoids an import cycle
        placement_probe_providers,
        repo_root_for,
        stack_components,
    )

    allocator = cuda if cuda is not None else _cuda_module()
    loader: ProviderLoader = registry if registry is not None else ProviderRegistry()

    if not allocator.is_available():
        return PlacementReport(
            EnvironmentReport(
                spec.name,
                (
                    Finding(
                        "E-PLACEMENT-NO-CUDA",
                        "torch reports no CUDA device, so GPU placement cannot be proven. This is "
                        "a failure, not a skip: the whole point of this gate is that a "
                        "silent CPU fallback (#191) can never pass it.",
                    ),
                ),
            ),
            (),
        )

    root = repo_root if repo_root is not None else repo_root_for(spec.spec_path)
    probe_set = placement_probe_providers(spec, root)
    if not probe_set:
        return PlacementReport(
            EnvironmentReport(
                spec.name,
                (
                    Finding(
                        "E-PLACEMENT-NONE",
                        "this environment declares no learned Provider and none of the packages it "
                        "installs loads a model, so there is nothing for the GPU-placement gate to "
                        "measure. Reporting success here would be a green earned by measuring "
                        "nothing — the exact shape of #191 — so it is a failure instead.",
                    ),
                ),
            ),
            (),
        )

    findings: list[Finding] = []
    placements: list[Placement] = []
    for component in stack_components(spec.stack_path):
        if component.provider not in probe_set:
            continue
        placement = _probe_one(component, allocator, loader, findings)
        if placement is not None:
            placements.append(placement)

    return PlacementReport(EnvironmentReport(spec.name, tuple(findings)), tuple(placements))


def _probe_one(
    component: StackComponent,
    allocator: CudaAllocator,
    loader: ProviderLoader,
    findings: list[Finding],
) -> Placement | None:
    """Drive one component and measure it; append findings in place. Returns the measurement."""
    device = str(component.config.get("device", "auto")).strip().lower()
    if device not in {CUDA, "auto"}:
        findings.append(
            Finding(
                "E-PLACEMENT-DEVICE-NOT-CUDA",
                f"'{component.provider}' is pinned to device={device!r} in this GPU environment's "
                "stack; a run on that device proves nothing about GPU placement",
            )
        )
        return None

    driver = DRIVERS.get(component.capability)
    if driver is None:
        findings.append(
            Finding(
                "E-PLACEMENT-UNDRIVABLE",
                f"'{component.provider}' declares capability '{component.capability}', which this "
                "probe has no driver for — an unexercised learned Provider is where a silent CPU "
                "fallback hides (#191). Add a driver to latence_core.placement.DRIVERS.",
            )
        )
        return None

    try:
        provider_cls = loader.load(component.provider)
        provider = provider_cls(dict(component.config))
    except Exception as exc:  # noqa: BLE001 - any construction failure is the finding
        findings.append(
            Finding(
                "E-PLACEMENT-FAILED",
                f"'{component.provider}' could not be constructed: {type(exc).__name__}: {exc}",
            )
        )
        return None

    try:
        # Warm-up, DELIBERATELY UNMEASURED: a Provider may load its weights onto CUDA lazily on its
        # first call, and weight placement is not evidence that the forward pass ran on the GPU.
        driver(provider)
        allocator.synchronize()

        # Real reset_peak_memory_stats() sets peak = CURRENT, so the resident weights (placed in
        # __init__ or by the warm-up) are inside the peak from the first instant. Subtracting the
        # baseline is what leaves only the measured forward pass's own allocations.
        allocator.reset_peak_memory_stats()
        baseline = int(allocator.memory_allocated())

        driver(provider)
        # The allocator's peak is only trustworthy once the queued kernels have actually run.
        allocator.synchronize()
    except Exception as exc:  # noqa: BLE001 - any inference failure is the finding
        findings.append(
            Finding(
                "E-PLACEMENT-FAILED",
                f"'{component.provider}' raised while being driven: {type(exc).__name__}: {exc}",
            )
        )
        return None
    peak = max(int(allocator.max_memory_allocated()) - baseline, 0)

    placement = Placement(
        provider=component.provider, capability=component.capability, cuda_bytes=peak
    )
    if not placement.on_gpu:
        findings.append(
            Finding(
                "E-PLACEMENT-ON-CPU",
                f"'{component.provider}' completed a real inference and allocated {peak} CUDA "
                "bytes above its resident footprint doing it — the forward pass did NOT run on the "
                "GPU (weights sitting on cuda are not work). This is the #191 signature: "
                "correct output, configured cuda, and the whole forward pass on the CPU.",
            )
        )
    return placement

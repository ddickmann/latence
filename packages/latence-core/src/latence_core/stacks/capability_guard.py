"""Static capability-satisfaction guard for a stack's wiring (W2-gpu-sota, ADR-0036).

The gap this closes: on a CPU-only host ``latence stack validate`` *skips-with-flag* every
``device: cuda`` Stage (the P3-F1 device-honesty posture), so the Runner's dispatch-time
``isinstance(provider, <Capability Protocol>)`` check — the one that raises ``ContractError`` for a
mis-wired Stage — **never runs for a GPU Stage on CI**. That is exactly how the pod E2E's
fused/standalone miswiring hid: the blessed ``gpu-sota.yaml`` declared ``relation.gliner_relex``
(a FUSED Provider fulfilling :class:`~latence_core.capability.FusedEntityRelationExtractor`) under a
``relation_extraction`` Stage, so at run it failed the standalone
:class:`~latence_core.capability.RelationExtractor` seam — but no CPU gate saw it.

This module makes the wiring statically checkable **without a GPU and without instantiating the
Provider** (instantiation would load torch). It reads the Provider *class* from the registry
(module import is dependency-free for the heavy Providers — the heavy import is deferred to first
use, ADR-0016) and structurally checks it against the Protocol its Stage's :class:`CapabilityKind`
declares.

That mapping used to be a copy kept here, under a comment claiming it was "kept in ONE place so the
two cannot drift" — while the Runner held its own inline copy and three other modules held adjacent
facts about the same Capabilities. It now comes from the Capability's own
:class:`~latence_core.capability_descriptor.CapabilityDescriptor`, which is the one place, and which
refuses at import to describe a :class:`CapabilityKind` that has not declared its Protocol.

A ``runtime_checkable`` Protocol whose members are all methods supports ``issubclass``; the
:class:`~latence_core.capability.Embedder` Protocol has a ``dimension`` **property** (a non-method
member), so ``issubclass`` is unavailable for it — we fall back to a structural attribute check
(every named member is present on the class), which is what ``isinstance`` would verify on an
instance. Either way the check is a class-only, torch-free, CPU pre-flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from latence_core.capability import Embedder, ProviderRegistry
from latence_core.capability_descriptor import dag_protocol_for, descriptor_for

if TYPE_CHECKING:
    from latence_core.pipeline import CapabilityKind, Pipeline, Stage


def capability_protocol(kind: CapabilityKind) -> type:
    """Return the Capability Protocol a DAG Stage of ``kind`` requires its Provider to satisfy.

    A thin read of the Capability's own descriptor — the same declaration the Runner's dispatch
    enforces at run time, so this CPU pre-flight and that run-time check cannot answer differently.

    Raises ``KeyError`` for a kind that is not a top-level DAG node (``EMBEDDING`` — a nested Export
    sub-config, not a Stage), so a caller that mis-treats it as a Stage fails loudly rather than
    silently skipping the check.
    """
    return dag_protocol_for(kind)


def _stage_protocol(kind: CapabilityKind) -> type | None:
    """The Protocol to check a Stage's Provider against, ``None`` when the kind is not a Stage."""
    descriptor = descriptor_for(kind)
    return descriptor.protocol if descriptor.dag_node else None


def _class_satisfies(provider_cls: type, protocol: type) -> bool:
    """Does ``provider_cls`` structurally satisfy ``protocol`` — class-only, no instantiation.

    For a ``runtime_checkable`` Protocol whose members are all methods, ``issubclass`` is exactly
    the check ``isinstance`` would apply to an instance, so we use it directly. A Protocol with a
    non-method member (a property — :class:`Embedder`'s ``dimension``) does not support
    ``issubclass``; there we fall back to verifying every declared member name is an attribute of
    the class (the same shape ``isinstance`` checks), which needs no constructed instance.
    """
    try:
        return issubclass(provider_cls, protocol)
    except TypeError:
        # Protocol has non-method members (e.g. Embedder.dimension is a property). Structural check:
        # every member the Protocol names must be present on the class.
        members = getattr(protocol, "__protocol_attrs__", None)
        if members is None:  # pragma: no cover - all our Protocols expose this on 3.11+
            members = {n for n in dir(protocol) if not n.startswith("_")}
        return all(hasattr(provider_cls, name) for name in members)


@dataclass(frozen=True)
class SatisfactionFinding:
    """One mis-wired (stage, provider) → the Capability Protocol it fails to satisfy.

    ``embedder`` is True when the finding is a nested Export ``config['embedder']`` Provider rather
    than the Stage's own top-level Provider — so the message can say *which* seam is mis-wired.
    """

    stage_name: str
    provider: str
    capability: str
    protocol: str
    embedder: bool = False

    def describe(self) -> str:
        seam = "embedder" if self.embedder else "provider"
        return (
            f"stage {self.stage_name!r} {seam} {self.provider!r} does not satisfy the "
            f"{self.protocol} capability declared by capability={self.capability!r}"
        )


def embedder_provider_of(stage: Stage) -> str | None:
    """The nested ``config['embedder']`` Provider name of an Export Stage, or None.

    Mirrors the Runner/harness embedder resolution: an Export Stage may opt into an
    :class:`Embedder` via ``config['embedder'] = {"provider": ..., "config": {...}}`` (ADR-0017).
    Returns the provider name when present and well-formed, else None (a Stage with no embedder).
    """
    spec = stage.config.get("embedder")
    if not isinstance(spec, dict):
        return None
    provider = spec.get("provider")
    if isinstance(provider, str) and provider:
        return provider
    return None


def check_stack_capabilities(
    pipeline: Pipeline, registry: ProviderRegistry | None = None
) -> list[SatisfactionFinding]:
    """Assert every Provider named in ``pipeline`` satisfies the Capability its Stage declares.

    For each Stage: load its Provider class from the registry and structurally check it against the
    Protocol :func:`capability_protocol` maps its :class:`CapabilityKind` to. For an Export Stage
    that opts into a nested embedder, additionally check that embedder Provider against
    :class:`Embedder`. Returns the list of :class:`SatisfactionFinding`s — empty means the whole
    stack is correctly wired.

    A Provider that is **not installed** on this host (its heavy package absent, ADR-0016) is
    **skipped**, not a finding: the guard checks *wiring correctness of installed Providers*, which
    is what bites a fused-under-relation mismatch; a genuinely-absent Provider is the device-honesty
    story (the ``stack validate`` skip-with-flag), not a wiring bug. A Provider whose class fails to
    import is likewise skipped (it cannot be checked without importing) — the check is a pure,
    torch-free class inspection, never an instantiation.
    """
    registry = registry or ProviderRegistry()
    installed = set(registry.names())
    findings: list[SatisfactionFinding] = []

    for stage in pipeline.stages:
        protocol = _stage_protocol(stage.capability)
        if protocol is not None and stage.provider in installed:
            try:
                provider_cls = registry.load(stage.provider)
            except Exception:  # noqa: BLE001 - an un-importable provider cannot be class-checked
                provider_cls = None
            if provider_cls is not None and not _class_satisfies(provider_cls, protocol):
                findings.append(
                    SatisfactionFinding(
                        stage_name=stage.name,
                        provider=stage.provider,
                        capability=str(stage.capability),
                        protocol=protocol.__name__,
                    )
                )
        # Nested Export embedder (ADR-0017): a mis-typed embedder Provider is a wiring bug too.
        embedder_name = embedder_provider_of(stage)
        if embedder_name is not None and embedder_name in installed:
            try:
                embedder_cls = registry.load(embedder_name)
            except Exception:  # noqa: BLE001 - un-importable embedder cannot be class-checked
                embedder_cls = None
            if embedder_cls is not None and not _class_satisfies(embedder_cls, Embedder):
                findings.append(
                    SatisfactionFinding(
                        stage_name=stage.name,
                        provider=embedder_name,
                        capability=str(stage.capability),
                        protocol=Embedder.__name__,
                        embedder=True,
                    )
                )
    return findings

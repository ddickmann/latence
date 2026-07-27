"""The nested ``{provider, config}`` sub-Provider block — declared once, parsed once, checked once.

A Stage names its Provider at the top level (``provider:`` + ``config:``). Some Capabilities
additionally let a Stage name a **sub-Provider** *inside* its config — a separately-loaded model
behind its own Capability seam:

.. code-block:: yaml

    - name: export_corpus
      capability: export
      provider: export.jsonl_parquet
      config:
        embedder: {provider: embedding.hashing, config: {dimension: 64}}
        bm25:     {provider: tokenizer.regex,   config: {}}

That ``{provider, config}`` block is a real concept and it used to have no module. A grep for its
readers found **ten** independent implementations: the Export's five signal slots, the Graph
Assembly's ``graph_features``, the Type Consolidator's semantic ``embedder``, the universal
Parser's ``ocr``, both neural resolvers' ``embedder``, the capability guard, the ``stack validate``
device pre-flight, the envspec walker and the autotune overlay — each with its own hard-coded key
list, its own shape validation, and its own error strings. They disagreed: the same mistake raised
``ConfigError`` in one and ``ContractError`` in another; four of them knew only about ``embedder``.

The consequence was measurable. The blessed SOTA stack emits two of these blocks
(``graph.config.graph_features`` and ``export_corpus.config.bm25``) and NEITHER was
capability-checked, device-skip-checked, or seen by the placement probe, so a typo in
``graph_features.pagerannk`` sailed through ``latence setup`` and died at run time; and a block
written under a Capability that declares no such slot was read by nobody at all, silently.

This module is the one place that knows about the block:

* :data:`SubProviderSlot` — a Capability *declares* its nested slots (key → Capability Protocol,
  plus any companion slot it requires). :func:`slots_for` is that declaration, and it is the only
  list of nested keys in the codebase.
* :func:`sub_provider_ref` — parse one block into a typed :class:`SubProviderRef`, with the shape
  errors (:class:`~latence_core.errors.ConfigError`) raised in ONE place.
* :func:`build_sub_provider` / :func:`resolve_sub_provider` — load, construct and capability-check
  it, raising a :class:`~latence_core.errors.ContractError` for a Provider that does not satisfy
  the slot's declared Protocol.
* :func:`stage_sub_providers` — every nested ref a Stage declares, so a *static* consumer (the
  offline ``latence stack check``, the device-skip pre-flight, the envspec walker, the GPU
  placement probe) enumerates them from the SAME declaration the runtime resolver uses.
* :func:`undeclared_sub_provider_keys` — the inverse: a ``{provider, ...}`` block sitting under a
  Capability that declares no such slot. That block is silently IGNORED at run time today, which
  is exactly how a mistyped or misplaced slot key survives to production.
* :func:`class_satisfies_capability` — the class-only (never instantiated, therefore torch-free)
  form of the capability check, for the offline checker.

The capability check is :func:`~latence_core.capability.satisfies_capability`, NOT plain
``isinstance`` (audit R6): several of these Protocols declare a property, and a
``runtime_checkable`` ``isinstance`` calls ``hasattr`` — which *invokes* it and pulls a Provider's
heavy backend (``sparse.splade`` loads a Hugging Face checkpoint from ``vocab_size``) at wiring
time, breaking ADR-0016's defer-to-first-use contract and making config validation reach the
network.

Thin core (ADR-0016): this module imports only ``latence_core.capability`` /
``latence_core.pipeline`` / ``latence_core.errors`` — the Protocol objects and the Stage contract.
It never imports a Provider package and never instantiates anything the caller did not ask for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_core.capability import (
    Embedder,
    FdeConverter,
    GraphFeatureComputer,
    MultiVectorEmbedder,
    Parser,
    SparseEmbedder,
    Tokenizer,
    protocol_members,
    satisfies_capability,
)
from latence_core.errors import ConfigError, ContractError
from latence_core.pipeline import CapabilityKind

if TYPE_CHECKING:
    from latence_core.capability import ProviderRegistry
    from latence_core.pipeline import Stage

__all__ = [
    "SubProviderRef",
    "SubProviderSlot",
    "build_sub_provider",
    "class_satisfies_capability",
    "declared_slot_keys",
    "missing_required_slots",
    "owner_label",
    "resolve_sub_provider",
    "slot_for",
    "slots_for",
    "stage_sub_providers",
    "sub_provider_ref",
    "undeclared_sub_provider_keys",
]


@dataclass(frozen=True)
class SubProviderSlot:
    """One nested ``{provider, config}`` slot a Capability declares.

    ``key`` is the config key the block sits under; ``protocol`` is the Capability Protocol the
    named Provider must satisfy; ``requires`` names sibling slots the block is meaningless without
    (the Export's ``fde_converter`` reduces the multi-vector to one dense vector, so it needs a
    ``multivector_embedder`` — a wiring error that used to bite only at run time).
    """

    key: str
    protocol: type
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubProviderRef:
    """One PARSED nested block: which slot, which Provider, with which config.

    Construct it through :func:`sub_provider_ref` (which enforces the shape) rather than by hand,
    except where a Stage supplies its own default Provider for an absent block (the Type
    Consolidator's deterministic in-core embedder).
    """

    slot: SubProviderSlot
    provider: str
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """The config key this block sits under (``embedder``, ``bm25``, …)."""
        return self.slot.key

    @property
    def protocol(self) -> type:
        """The Capability Protocol the named Provider must satisfy."""
        return self.slot.protocol

    @property
    def device(self) -> str:
        """The device the block requests (``auto`` when unset) — the device-skip pre-flight's input.

        A nested Provider is as GPU-gated as a Stage-level one (the pod post-mortems were full of
        nested-embedder surprises), and the honest skip decision runs through the same
        ``select_device`` seam. Reading it here means every consumer reads it the same way.
        """
        return str(self.config.get("device", "auto"))


# -- the declaration -----------------------------------------------------------
#
# Every nested slot in the framework, keyed by the Capability that declares it. This is the ONLY
# list of nested keys: the runtime resolvers, the offline checker, the device-skip pre-flight, the
# envspec walker and the GPU placement probe all read it, so a new slot is declared once instead
# of being added to six hard-coded tuples (and forgotten in four of them).
_SLOTS: dict[CapabilityKind, tuple[SubProviderSlot, ...]] = {
    # W5-universal-parser: ``parser.render`` rasterises any file to page images and delegates the
    # page OCR to a nested Parser (``stacks/universal-parse.yaml`` wires ``parser.lighton`` there).
    # It is the GPU-gated node of that stack and, before this declaration, no static tool saw it.
    CapabilityKind.PARSE: (SubProviderSlot("ocr", Parser),),
    # ADR-0017 (dense embedder) + ADR-0049/0050 (the index-time retrieval signals) + #164 (BM25).
    CapabilityKind.EXPORT: (
        SubProviderSlot("embedder", Embedder),
        SubProviderSlot("sparse_embedder", SparseEmbedder),
        SubProviderSlot("multivector_embedder", MultiVectorEmbedder),
        SubProviderSlot("fde_converter", FdeConverter, requires=("multivector_embedder",)),
        SubProviderSlot("bm25", Tokenizer),
    ),
    # ADR-0051: the optional graph-feature computer the Assembler stamps onto every node.
    CapabilityKind.GRAPH_ASSEMBLY: (SubProviderSlot("graph_features", GraphFeatureComputer),),
    # ADR-0054: the Type Consolidator's optional semantic rung.
    CapabilityKind.TYPE_CONSOLIDATION: (SubProviderSlot("embedder", Embedder),),
    # The neural resolvers (latence-disambig-glinker / latence-disambig-embedding) resolve their
    # STS rung from the same block shape.
    CapabilityKind.DISAMBIGUATION: (SubProviderSlot("embedder", Embedder),),
}

# Boolean config flags that are meaningless without a companion slot. Same class of wiring error
# as :attr:`SubProviderSlot.requires` (the Export's ``keep_multivectors`` writes the multi-vector
# sidecar — there is nothing to persist without a ``multivector_embedder``), so it is declared
# here rather than being discoverable only by reading the Export's constructor.
_FLAG_REQUIRES: dict[CapabilityKind, dict[str, tuple[str, ...]]] = {
    CapabilityKind.EXPORT: {"keep_multivectors": ("multivector_embedder",)},
}

# The human label that prefixes a slot's error messages, per Capability. Kept beside the slot
# declaration so the message an operator sees names the Stage kind they wrote, not this module.
_OWNER_LABEL: dict[CapabilityKind, str] = {
    CapabilityKind.PARSE: "Parse config",
    CapabilityKind.EXPORT: "Export config",
    CapabilityKind.GRAPH_ASSEMBLY: "Graph Assembly config",
    CapabilityKind.TYPE_CONSOLIDATION: "Type Consolidation config",
    CapabilityKind.DISAMBIGUATION: "Disambiguation config",
}


def slots_for(capability: CapabilityKind) -> tuple[SubProviderSlot, ...]:
    """The nested slots ``capability`` declares, in declaration order (empty when it declares none).

    Declaration order is stable, so every consumer that iterates slots produces byte-identical
    output for the same stack (the determinism invariant).
    """
    return _SLOTS.get(capability, ())


def slot_for(capability: CapabilityKind, key: str) -> SubProviderSlot | None:
    """The slot ``capability`` declares under ``key``, or ``None`` when it declares none."""
    for slot in slots_for(capability):
        if slot.key == key:
            return slot
    return None


def declared_slot_keys() -> frozenset[str]:
    """Every nested key any Capability declares — the walker's key set (envspec, placement probe).

    Read by :func:`latence_core.envspec.stack_components`, the walker that feeds the GPU-placement
    probe and the env-stack constructibility gate, AT WALK TIME — so a newly declared slot cannot
    leave it behind. It kept its own three-key tuple until R3-seams_real_not_hypothetical-0, and
    four of the seven slots below (``ocr``, ``fde_converter``, ``bm25``, ``graph_features``) were
    invisible to both gates the whole time.

    A *Stage-scoped* consumer wants :func:`stage_sub_providers` instead: this set is
    capability-agnostic on purpose, because a structural YAML walk has no Stage to ask.
    """
    return frozenset(slot.key for slots in _SLOTS.values() for slot in slots)


def owner_label(capability: CapabilityKind) -> str:
    """The message prefix for ``capability``'s nested-block errors (e.g. ``"Export config"``)."""
    return _OWNER_LABEL.get(capability, f"{capability} config")


def sub_provider_ref(
    spec: Any,
    *,
    capability: CapabilityKind,
    key: str,
    owner: str | None = None,
) -> SubProviderRef | None:
    """Parse one nested ``{provider, config}`` block into a typed :class:`SubProviderRef`.

    Returns ``None`` when ``spec`` is ``None`` (the slot is simply not wired — every nested slot is
    opt-in and its absence leaves the Stage's output byte-identical). Every malformed shape is a
    :class:`~latence_core.errors.ConfigError` raised HERE, so the six call sites no longer each
    carry their own three shape checks and three message strings.

    ``owner`` overrides the message prefix (a Provider that prefers to name itself — the Type
    Consolidator prefixes its entry-point name).

    Raises ``LookupError`` when ``key`` is not a slot ``capability`` declares: that is a
    programming error in the caller, not operator input, and must not read as a config error.
    """
    slot = slot_for(capability, key)
    if slot is None:
        msg = (
            f"{capability} declares no nested sub-Provider slot {key!r}; "
            f"declared slots: {[s.key for s in slots_for(capability)]}"
        )
        raise LookupError(msg)
    label = owner if owner is not None else owner_label(capability)
    if spec is None:
        return None
    if not isinstance(spec, dict):
        msg = (
            f"{label} {key!r} must be a mapping "
            "{'provider': <name>, 'config': {...}} or absent; "
            f"got {type(spec).__name__}."
        )
        raise ConfigError(msg)
    provider = spec.get("provider")
    if not isinstance(provider, str) or not provider:
        msg = f"{label} {key!r} requires a non-empty string 'provider' entry-point name."
        raise ConfigError(msg)
    provider_config = spec.get("config", {})
    if not isinstance(provider_config, dict):
        msg = f"{label} '{key}.config' must be a mapping if present."
        raise ConfigError(msg)
    return SubProviderRef(slot=slot, provider=provider, config=dict(provider_config))


def build_sub_provider(
    ref: SubProviderRef,
    *,
    registry: ProviderRegistry,
    owner: str,
) -> Any:  # noqa: ANN401 - the concrete Provider type is the slot's Protocol, known to the caller
    """Load, construct and capability-check the Provider ``ref`` names.

    A mis-named Provider fails **loudly at wiring time** — a
    :class:`~latence_core.errors.ContractError` — rather than silently emitting no signal
    (ADR-0034: no silent errors). The check is :func:`satisfies_capability`, never ``isinstance``,
    so a Protocol property is not invoked (and a heavy backend not loaded) during validation.
    """
    provider_cls = registry.load(ref.provider)
    provider = provider_cls(ref.config)
    if not satisfies_capability(provider, ref.protocol):
        msg = (
            f"Provider {ref.provider!r} named in {owner} {ref.key!r} does not satisfy the "
            f"{ref.protocol.__name__} capability."
        )
        raise ContractError(msg)
    return provider


def resolve_sub_provider(
    spec: Any,
    *,
    capability: CapabilityKind,
    key: str,
    registry: ProviderRegistry,
    owner: str | None = None,
) -> Any | None:  # noqa: ANN401 - see build_sub_provider
    """Parse + build one nested block; ``None`` when the slot is not wired.

    The single call a Stage makes: ``resolve_sub_provider(config.get("embedder"),
    capability=CapabilityKind.EXPORT, key="embedder", registry=registry)``. Shape errors are
    :class:`~latence_core.errors.ConfigError`, capability mismatches
    :class:`~latence_core.errors.ContractError` — the same two typed failures at every slot.
    """
    label = owner if owner is not None else owner_label(capability)
    ref = sub_provider_ref(spec, capability=capability, key=key, owner=label)
    if ref is None:
        return None
    return build_sub_provider(ref, registry=registry, owner=label)


def stage_sub_providers(stage: Stage) -> tuple[SubProviderRef, ...]:
    """Every nested block ``stage`` declares, parsed, in slot-declaration order.

    The enumeration a *static* consumer needs — the offline checker, the device-skip pre-flight,
    the envspec walker, the GPU placement probe — so none of them carries its own key list. A slot
    the Stage did not wire is absent from the result; a malformed one raises the same
    :class:`~latence_core.errors.ConfigError` the runtime resolver would.
    """
    refs: list[SubProviderRef] = []
    for slot in slots_for(stage.capability):
        ref = sub_provider_ref(
            stage.config.get(slot.key), capability=stage.capability, key=slot.key
        )
        if ref is not None:
            refs.append(ref)
    return tuple(refs)


def _looks_like_a_block(value: Any) -> bool:
    """A config value shaped like a sub-Provider block: a mapping carrying a ``provider`` key."""
    return isinstance(value, dict) and "provider" in value


def undeclared_sub_provider_keys(stage: Stage) -> tuple[str, ...]:
    """Config keys that LOOK like a sub-Provider block but are not a slot this Capability declares.

    Sorted, so the result is deterministic. Such a block is silently IGNORED at run time — nothing
    reads it — which is exactly how a mistyped slot key (``embedders:``) or a block written under
    the wrong Stage (``graph_features`` on an Export) survives ``latence setup`` and quietly costs
    an operator the signal they thought they configured.
    """
    declared = {slot.key for slot in slots_for(stage.capability)}
    return tuple(
        sorted(
            key
            for key, value in stage.config.items()
            if key not in declared and _looks_like_a_block(value)
        )
    )


def missing_required_slots(stage: Stage) -> tuple[tuple[str, str], ...]:
    """``(wired_key, missing_companion)`` pairs this Stage's config is missing.

    Covers both declared kinds of companion requirement: a slot that requires a sibling slot
    (``fde_converter`` → ``multivector_embedder``) and a boolean flag that requires one
    (``keep_multivectors`` → ``multivector_embedder``). Both are wiring errors the Stage's own
    constructor raises at run time; surfacing them here makes them catchable offline.
    """
    missing: list[tuple[str, str]] = []
    for slot in slots_for(stage.capability):
        if stage.config.get(slot.key) is None:
            continue
        missing.extend(
            (slot.key, required)
            for required in slot.requires
            if stage.config.get(required) is None
        )
    for flag, requires in _FLAG_REQUIRES.get(stage.capability, {}).items():
        if not stage.config.get(flag):
            continue
        missing.extend(
            (flag, required) for required in requires if stage.config.get(required) is None
        )
    return tuple(missing)


def class_satisfies_capability(provider_cls: type, protocol: type) -> bool:
    """Does ``provider_cls`` structurally satisfy ``protocol`` — class-only, never instantiated.

    The offline form of :func:`satisfies_capability`: an offline checker must decide whether a
    named Provider fits a slot WITHOUT constructing it (constructing a learned Provider imports
    torch and may allocate a model — ADR-0016 keeps that deferred to first use). Every Protocol
    member must be present on the class, which is exactly what ``isinstance`` verifies on an
    instance, minus the instantiation and minus the property invocation.
    """
    members = protocol_members(protocol)
    return all(hasattr(provider_cls, name) for name in members)

"""CPU capability-satisfaction guard for the committed stacks (W2-gpu-sota, ADR-0036).

The gap this closes (the pod E2E post-mortem): ``latence stack validate`` on a CPU-only host
*skips-with-flag* every ``device: cuda`` Stage, so the Runner's dispatch-time
``isinstance(provider, <Protocol>)`` check — the one that raised ``ContractError`` for the fused
``relation.gliner_relex`` mis-declared under a ``relation_extraction`` Stage — NEVER runs in CI. All
three pod bugs (fused-under-relation, the ``pii_labels`` omission, the double-NER pairing) hid
behind that skip.

:func:`~latence_core.stacks.capability_guard.check_stack_capabilities` makes the wiring statically
checkable without a GPU and without instantiating the Provider (no torch load): it reads each
Provider *class* from the registry and structurally checks it against the Protocol its Stage's
:class:`CapabilityKind` declares — the SAME mapping the Runner enforces. These tests prove:

* the fixed ``gpu-sota.yaml`` + ``gpu-sota-pod.yaml`` have ZERO capability-satisfaction findings
  (every Provider — including the nested Export embedders — satisfies its declared Capability);
* the guard GENUINELY BITES — the exact pod miswiring (fused ``relation.gliner_relex`` under a
  standalone ``relation_extraction`` Stage) IS a finding, and so is a mis-typed nested embedder.

Requires the heavy Provider *sources* installed ``--no-deps`` (verify-local does this) so their
classes are registered; the import is torch-free (the heavy dep is deferred to first use), so the
class check runs on a CPU host. If a heavy Provider is not installed the guard skips it (not a
finding) — device-honesty — so the fixed-stack tests are robust to a bare workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.stacks.capability_guard import (
    capability_protocol,
    check_stack_capabilities,
)
from latence_core.stacks.validate import load_stack

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACKS = _REPO_ROOT / "stacks"
_GPU_SOTA = _STACKS / "gpu-sota.yaml"
_GPU_SOTA_POD = _STACKS / "gpu-sota-pod.yaml"
_DEFAULT = _STACKS / "default.yaml"
_GPU_OCR = _STACKS / "gpu-ocr.yaml"

_HEAVY_PROVIDERS = {
    "fused_entity_relation.gliner2",
    "redaction.gliner_pii",
    "embedding.sentence_transformers",
}


def _load(stack: Path) -> Pipeline:
    return load_stack(stack, storage_uri="file:///tmp/x", corpus_dir="file:///tmp/c")


def _heavy_installed() -> bool:
    """Are the gpu-sota heavy Provider sources registered (verify-local installs them --no-deps)?"""
    return set(ProviderRegistry().names()) >= _HEAVY_PROVIDERS


# -- The fixed stacks are correctly wired (zero findings) ---------------------


@pytest.mark.parametrize("stack", [_GPU_SOTA, _GPU_SOTA_POD, _DEFAULT, _GPU_OCR])
def test_committed_stack_has_no_capability_findings(stack: Path) -> None:
    """Every Provider a committed stack names satisfies the Capability its Stage declares.

    This is the check that would have caught the pod bug in CI: the fused Stage's
    ``fused_entity_relation.gliner2`` (W9, superseding ``relation.gliner_relex``) satisfies
    ``FusedEntityRelationExtractor`` (its declared ``fused_entity_relation`` capability), and no
    Provider is mis-declared. Runs on a CPU host — class inspection only, no GPU, no instantiation.
    """
    findings = check_stack_capabilities(_load(stack))
    assert findings == [], "\n".join(f.describe() for f in findings)


def test_gpu_sota_declares_the_fused_stage_not_separate_entity_relation() -> None:
    """The blessed SOTA path is ONE ``fused_entity_relation`` Stage — not separate entity+relation.

    Pins the architectural fix: the fused Provider must be wired under the fused capability (so the
    Runner routes its mixed carrier to both consumers), and there must be NO standalone
    ``relation_extraction`` Stage naming the fused Provider (the pod miswiring).
    """
    pipeline = _load(_GPU_SOTA)
    fused = [s for s in pipeline.stages if s.capability is CapabilityKind.FUSED_ENTITY_RELATION]
    assert len(fused) == 1
    # W9: the unified default is fused_entity_relation.gliner2 (relation.gliner_relex superseded).
    assert fused[0].provider == "fused_entity_relation.gliner2"
    # No separate Entity Extraction / Relation Extraction Stage in the SOTA path (no double-NER).
    assert not any(
        s.capability in (CapabilityKind.ENTITY_EXTRACTION, CapabilityKind.RELATION_EXTRACTION)
        for s in pipeline.stages
    )
    # Downstream depends_on rewired onto the single fused Stage.
    by_name = {s.name: s for s in pipeline.stages}
    assert set(by_name["profiling"].depends_on) == {"parse", "extract"}
    assert by_name["disambiguate"].depends_on == ["extract"]


def test_redact_stage_carries_nonempty_pii_labels() -> None:
    """The learned Redaction Stage carries the REQUIRED non-empty ``pii_labels`` (the pod ValueError).

    ``redaction.gliner_pii`` raises ``ValueError`` at load with an empty/missing ``pii_labels``; the
    fix adds the list. Asserted on both blessed variants so a future edit that drops it is caught.
    """
    for stack in (_GPU_SOTA, _GPU_SOTA_POD):
        redact = next(s for s in _load(stack).stages if s.name == "redact")
        labels = redact.config.get("pii_labels")
        assert isinstance(labels, list) and labels, f"{stack.name}: pii_labels must be non-empty"
        assert all(isinstance(x, str) and x for x in labels)
        # Lower-case zero-shot convention (matches the extract labels + case-sensitive KB linker).
        assert all(x == x.lower() for x in labels)


def test_gpu_sota_kb_entity_type_is_lowercase() -> None:
    """The Disambiguation KB ``entity_type`` stays lower-case (matches the fused labels; #129)."""
    disambig = next(s for s in _load(_GPU_SOTA).stages if s.name == "disambiguate")
    kb = disambig.config["kb"]
    assert all(entry["entity_type"] == entry["entity_type"].lower() for entry in kb)
    assert all(entry["entity_type"] == "organization" for entry in kb)


def test_pod_variant_uses_in_process_embedder_not_endpoint() -> None:
    """gpu-sota-pod's RAG export embedder is in-process ST (self-contained), not the served endpoint."""
    export = next(s for s in _load(_GPU_SOTA_POD).stages if s.name == "export_corpus")
    assert export.config["embedder"]["provider"] == "embedding.sentence_transformers"
    # The blessed gpu-sota keeps the served endpoint — the two differ ONLY here.
    sota_export = next(s for s in _load(_GPU_SOTA).stages if s.name == "export_corpus")
    assert sota_export.config["embedder"]["provider"] == "embedding.endpoint"


# -- The guard GENUINELY BITES ------------------------------------------------


@pytest.mark.skipif(not _heavy_installed(), reason="heavy Provider sources not installed")
def test_guard_bites_on_fused_provider_under_relation_stage() -> None:
    """A fused Provider mis-declared under a standalone ``relation_extraction`` Stage IS a finding.

    This is the exact pod miswiring. ``relation.gliner_relex`` satisfies
    ``FusedEntityRelationExtractor`` but NOT the standalone ``RelationExtractor.relate`` seam, so
    under a ``relation_extraction`` Stage it must be flagged — the bite that would have failed CI.
    """
    text = """
name: miswired-fused
storage_uri: file:///tmp/x
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: /tmp}
  - name: rel
    capability: relation_extraction
    provider: relation.gliner_relex
    depends_on: [source]
    config: {labels: [organization], relation_labels: [works for]}
"""
    findings = check_stack_capabilities(Pipeline.from_yaml_text(text))
    assert len(findings) == 1
    f = findings[0]
    assert f.stage_name == "rel"
    assert f.provider == "relation.gliner_relex"
    assert f.protocol == "RelationExtractor"
    assert "does not satisfy" in f.describe()


@pytest.mark.skipif(not _heavy_installed(), reason="heavy Provider sources not installed")
def test_guard_bites_on_mistyped_nested_export_embedder() -> None:
    """A non-Embedder Provider named as a nested Export ``config['embedder']`` IS a finding.

    The nested embedder seam (ADR-0017) is checked too: naming a Parser (say) where an Embedder is
    required is a wiring bug the Runner would hit only at export time on a GPU host.
    """
    text = """
name: bad-embedder
storage_uri: file:///tmp/x
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: /tmp}
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [source]
    config:
      basename: corpus
      embedder: {provider: source.local_folder, config: {}}
"""
    findings = check_stack_capabilities(Pipeline.from_yaml_text(text))
    embedder_findings = [f for f in findings if f.embedder]
    assert len(embedder_findings) == 1
    assert embedder_findings[0].provider == "source.local_folder"
    assert embedder_findings[0].protocol == "Embedder"


def test_guard_skips_uninstalled_providers_not_a_finding() -> None:
    """A Provider absent on this host is skipped (device-honesty), never a false-positive finding."""
    text = """
name: absent-provider
storage_uri: file:///tmp/x
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: /tmp}
  - name: parse
    capability: parse
    provider: parser.totally_not_installed_xyz
    depends_on: [source]
"""
    findings = check_stack_capabilities(Pipeline.from_yaml_text(text))
    assert findings == []  # the absent provider is skipped, not flagged


# -- The capability→Protocol map is complete for every DAG-node kind ----------


def test_every_dag_capability_kind_maps_to_a_protocol() -> None:
    """Each DAG-node CapabilityKind (all but the nested EMBEDDING) resolves to a Protocol.

    A new Stage kind added to :class:`CapabilityKind` without extending the guard's map would let a
    mis-wire of that kind slip the CPU gate — this asserts the map covers every DAG-node kind.
    """
    for kind in CapabilityKind:
        if kind is CapabilityKind.EMBEDDING:
            # EMBEDDING is a nested Export sub-config, not a DAG Stage — checked via the embedder path.
            with pytest.raises(KeyError):
                capability_protocol(kind)
            continue
        assert capability_protocol(kind) is not None

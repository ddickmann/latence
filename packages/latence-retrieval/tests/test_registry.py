"""The ``latence.retrieval`` registry + config resolution, tested through the public seam.

Proves the whole register → resolve → build path against the *real* installed entry point
(``processor.identity``), that resolution enforces the :class:`CandidateProcessor` seam, and that the
group is genuinely separate from the pipeline's ``latence.providers`` (ADR-0049) — a query-time
component is never discoverable as a Stage Provider and vice versa.
"""

from __future__ import annotations

import pytest
from latence_core.capability import ProviderRegistry
from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    Fuser,
    IdentityProcessor,
    RetrievalRegistry,
    RrfFuser,
    WeightedFuser,
    resolve_fuser,
    resolve_processor,
)
from latence_retrieval.registry import ENTRY_POINT_GROUP


def test_group_is_the_new_retrieval_seam() -> None:
    assert ENTRY_POINT_GROUP == "latence.retrieval"


def test_identity_processor_is_discovered() -> None:
    assert "processor.identity" in RetrievalRegistry().names()


def test_load_returns_the_processor_class() -> None:
    assert RetrievalRegistry().load("processor.identity") is IdentityProcessor


def test_resolve_builds_a_working_processor() -> None:
    processor = resolve_processor({"provider": "processor.identity", "config": {}})
    assert isinstance(processor, CandidateProcessor)
    candidates = [Candidate(id="a", score=1.0)]
    assert processor.process("q", candidates) == candidates


def test_resolve_defaults_config_to_empty() -> None:
    processor = resolve_processor({"provider": "processor.identity"})
    assert isinstance(processor, CandidateProcessor)


def test_load_unknown_name_raises_keyerror_listing_available() -> None:
    with pytest.raises(KeyError, match="processor.identity"):
        RetrievalRegistry().load("processor.nonexistent")


def test_resolve_requires_a_provider_name() -> None:
    with pytest.raises(ValueError, match="provider"):
        resolve_processor({"config": {}})


def test_resolve_rejects_non_mapping_config() -> None:
    with pytest.raises(TypeError, match="config"):
        resolve_processor({"provider": "processor.identity", "config": ["not", "a", "map"]})


def test_both_fusers_are_discovered() -> None:
    names = RetrievalRegistry().names()
    assert "fuser.rrf" in names
    assert "fuser.weighted" in names


def test_resolve_fuser_builds_the_selected_adapter() -> None:
    assert isinstance(resolve_fuser({"provider": "fuser.rrf"}), RrfFuser)
    assert isinstance(resolve_fuser({"provider": "fuser.weighted"}), WeightedFuser)
    assert isinstance(resolve_fuser({"provider": "fuser.rrf"}), Fuser)


def test_resolve_fuser_rejects_a_processor_component() -> None:
    # The identity processor is a different shape (no `fuse`); resolving it AS a fuser must fail
    # loudly, so the two seams cannot be crossed by config.
    with pytest.raises(TypeError, match="Fuser seam"):
        resolve_fuser({"provider": "processor.identity"})


def test_resolve_processor_rejects_a_fuser_component() -> None:
    with pytest.raises(TypeError, match="CandidateProcessor seam"):
        resolve_processor({"provider": "fuser.rrf"})


def test_resolve_fuser_requires_a_provider_name() -> None:
    with pytest.raises(ValueError, match="provider"):
        resolve_fuser({"config": {}})


def test_retrieval_group_is_separate_from_providers() -> None:
    # The pipeline's provider registry must NOT see the query-time processor, and the retrieval
    # registry must NOT see any pipeline Provider — the two seams are strictly separate (ADR-0049).
    provider_names = set(ProviderRegistry().names())
    retrieval_names = set(RetrievalRegistry().names())
    assert "processor.identity" not in provider_names
    assert provider_names.isdisjoint(retrieval_names)
    # Sanity: the pipeline registry is non-empty (its in-core reference Providers), so the
    # disjointness above is a real separation, not two empty sets.
    assert provider_names

"""S9 HashingEmbedder tests: deterministic, fixed-width, L2-normalised, opt-in reference."""

from __future__ import annotations

import math

import pytest
from latence_core.capability import Embedder
from latence_core.embedding import HashingEmbedder


def test_provider_satisfies_embedder_capability() -> None:
    assert isinstance(HashingEmbedder(), Embedder)


def test_dimension_default_and_configurable() -> None:
    assert HashingEmbedder().dimension == 64
    assert HashingEmbedder({"dimension": 16}).dimension == 16


def test_invalid_dimension_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashingEmbedder({"dimension": 0})


def test_vectors_are_fixed_width_and_normalised() -> None:
    emb = HashingEmbedder({"dimension": 32})
    vectors = list(emb.embed(["hello world", "the framework reads the folder"]))
    assert all(len(v) == 32 for v in vectors)
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_empty_text_is_zero_vector() -> None:
    emb = HashingEmbedder({"dimension": 8})
    (vector,) = list(emb.embed([""]))
    assert vector == [0.0] * 8


def test_deterministic_across_instances() -> None:
    a = list(HashingEmbedder({"dimension": 16}).embed(["acme corporation"]))
    b = list(HashingEmbedder({"dimension": 16}).embed(["acme corporation"]))
    assert a == b


def test_different_text_gives_different_vector() -> None:
    emb = HashingEmbedder({"dimension": 64})
    a, b = list(emb.embed(["alice works at acme", "bob joined globex"]))
    assert a != b

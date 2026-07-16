"""Reference index-time signal generators, tested at the Capability interface (#160).

The three in-core reference emitters (`sparse.hashing`, `multivector.hashing`, `fde.reference`)
exercised THROUGH their Capability seams — `encode_sparse` / `encode_multivector` / `convert` —
asserting observable behaviour (shapes, determinism, the faithful width-rejection an FDE converter
performs, empty-input handling), never internal state. These survive an internal refactor of the
hashing/partitioning math.
"""

from __future__ import annotations

import pytest
from latence_core.capability import (
    FdeConverter,
    MultiVectorEmbedder,
    SparseEmbedder,
    SparseVector,
)
from latence_core.embedding import HashingEmbedder
from latence_core.errors import ConfigError
from latence_core.signals import (
    HashingMultiVectorEmbedder,
    ReferenceFdeConverter,
    SparseHashingEmbedder,
)

# ---------------------------------------------------------------------------
# The references satisfy their declared Capability seams
# ---------------------------------------------------------------------------


def test_references_satisfy_their_protocols() -> None:
    assert isinstance(SparseHashingEmbedder({}), SparseEmbedder)
    assert isinstance(HashingMultiVectorEmbedder({}), MultiVectorEmbedder)
    assert isinstance(ReferenceFdeConverter({"input_dimension": 4}), FdeConverter)


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------


def test_sparse_is_canonical_and_frequency_weighted() -> None:
    """Term frequency is the weight; indices are canonical (ascending, unique, in-vocab)."""
    embedder = SparseHashingEmbedder({"vocab_size": 128})
    [vec] = list(embedder.encode_sparse(["alpha beta alpha"]))
    assert isinstance(vec, SparseVector)
    assert vec.indices == sorted(set(vec.indices))
    assert all(0 <= i < 128 for i in vec.indices)
    # "alpha" occurs twice, "beta" once → total weight mass is 3.0 across ≤2 buckets.
    assert sum(vec.values) == pytest.approx(3.0)


def test_sparse_empty_text_is_the_empty_vector() -> None:
    [vec] = list(SparseHashingEmbedder({}).encode_sparse([""]))
    assert vec.indices == [] and vec.values == []


def test_sparse_is_deterministic() -> None:
    a = list(SparseHashingEmbedder({}).encode_sparse(["gamma delta", "epsilon"]))
    b = list(SparseHashingEmbedder({}).encode_sparse(["gamma delta", "epsilon"]))
    assert [(v.indices, v.values) for v in a] == [(v.indices, v.values) for v in b]


def test_sparse_rejects_bad_vocab() -> None:
    with pytest.raises(ConfigError):
        SparseHashingEmbedder({"vocab_size": 0})


# ---------------------------------------------------------------------------
# Multi-vector
# ---------------------------------------------------------------------------


def test_multivector_is_one_vector_per_token_matching_the_dense_embedding() -> None:
    """Each token vector equals the token's own embedding.hashing vector (one source of truth)."""
    dim = 16
    [mv] = list(HashingMultiVectorEmbedder({"dimension": dim}).encode_multivector(["alpha beta"]))
    assert len(mv) == 2 and all(len(tok) == dim for tok in mv)
    expected = list(HashingEmbedder({"dimension": dim}).embed(["alpha", "beta"]))
    assert mv == expected


def test_multivector_empty_text_is_empty() -> None:
    [mv] = list(HashingMultiVectorEmbedder({}).encode_multivector([""]))
    assert mv == []


# ---------------------------------------------------------------------------
# FDE
# ---------------------------------------------------------------------------


def test_fde_output_width_is_buckets_times_input_dim() -> None:
    converter = ReferenceFdeConverter({"input_dimension": 8, "buckets": 4})
    assert converter.dimension == 32
    mv = [[float(j) for j in range(8)] for _ in range(3)]
    [dense] = list(converter.convert([mv]))
    assert len(dense) == 32


def test_fde_empty_multivector_is_the_zero_vector() -> None:
    converter = ReferenceFdeConverter({"input_dimension": 4, "buckets": 2})
    [dense] = list(converter.convert([[]]))
    assert dense == [0.0] * 8


def test_fde_rejects_wrong_width_token_vector() -> None:
    """A token vector of the wrong width is rejected (anti-false-green: no silent mis-encode)."""
    converter = ReferenceFdeConverter({"input_dimension": 4})
    with pytest.raises(ConfigError, match="token vector of width"):
        list(converter.convert([[[0.0, 0.0, 0.0]]]))  # width 3, expected 4


def test_fde_rejects_non_power_of_two_buckets() -> None:
    with pytest.raises(ConfigError, match="power of two"):
        ReferenceFdeConverter({"input_dimension": 4, "buckets": 3})


def test_fde_requires_input_dimension() -> None:
    with pytest.raises(ConfigError, match="input_dimension"):
        ReferenceFdeConverter({})


def test_fde_is_deterministic_across_instances() -> None:
    """Two independently-constructed converters produce identical output (fixed hyperplanes)."""
    mv = [[float((i * 2 + j) % 5) for j in range(6)] for i in range(4)]
    a = list(ReferenceFdeConverter({"input_dimension": 6, "buckets": 8}).convert([mv]))
    b = list(ReferenceFdeConverter({"input_dimension": 6, "buckets": 8}).convert([mv]))
    assert a == b


def test_fde_partitions_tokens_into_distinct_buckets() -> None:
    """A genuine space-partition: opposite token vectors land in different cells (not a passthrough).

    Two token vectors that are negatives of each other fall on opposite sides of every hyperplane,
    so their bucket ids differ — their mass lands in disjoint slots of the concatenated output. A
    passthrough/pooling that ignored the partition would sum them into the same slots (→ zero).
    """
    converter = ReferenceFdeConverter({"input_dimension": 4, "buckets": 2})
    v = [1.0, 2.0, -1.0, 0.5]
    neg = [-x for x in v]
    [dense] = list(converter.convert([[v, neg]]))
    # If both tokens shared a bucket the two halves would each be v+neg = 0; a real partition keeps
    # them apart, so the summed output is NOT the all-zero vector.
    assert any(abs(x) > 1e-9 for x in dense)

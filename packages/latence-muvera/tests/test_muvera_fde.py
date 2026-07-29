"""MuveraFdeConverter: the canonical MUVERA FDE, exercised through the ``FdeConverter`` seam.

Everything is asserted on the observable output of the interface (``dimension`` + ``convert`` +
``encode_query``) so the tests survive an internal refactor of the SimHash / aggregation machinery.
The headline test is a genuine **MaxSim-approximation** assertion on synthetic multi-vectors (ADR-0050
anti-false-green: the dense dot-product must *track* true MaxSim as a document's overlap with the
query grows — not a shape check), plus the faithful wrong-width rejection a real converter performs.
Pure-Python + deterministic, so it runs offline with no model and no weights.
"""

from __future__ import annotations

import math
import random

import pytest
from latence_core.capability import FdeConverter
from latence_core.errors import ConfigError
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    ProviderProfile,
    profile_of,
)
from latence_muvera import MuveraFdeConverter

# --------------------------------------------------------------------------- #
# Synthetic multi-vector helpers (deterministic; unit vectors ⇒ dots are cosines)
# --------------------------------------------------------------------------- #


def _unit(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _maxsim(query: list[list[float]], doc: list[list[float]]) -> float:
    """The late-interaction target: Σ over query tokens of its best dot with any doc token."""
    return sum(max((_dot(q, d) for d in doc), default=0.0) for q in query)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else 0.0


# --------------------------------------------------------------------------- #
# The seam: shape, dimension, protocol conformance
# --------------------------------------------------------------------------- #


def test_satisfies_the_fde_converter_protocol() -> None:
    assert isinstance(MuveraFdeConverter({"input_dimension": 4}), FdeConverter)


def test_dimension_is_repetitions_times_buckets_times_input_dimension() -> None:
    conv = MuveraFdeConverter({"input_dimension": 5, "buckets": 8, "repetitions": 3})
    assert conv.dimension == 3 * 8 * 5


def test_default_repetitions_one_matches_the_reference_output_width() -> None:
    # A drop-in for fde.reference: one repetition ⇒ width buckets × input_dimension.
    conv = MuveraFdeConverter({"input_dimension": 6, "buckets": 4})
    assert conv.dimension == 4 * 6


def test_convert_yields_one_dense_vector_per_input_of_the_fixed_width() -> None:
    conv = MuveraFdeConverter({"input_dimension": 3, "buckets": 4})
    rng = random.Random(1)
    mvs = [[_unit(rng, 3) for _ in range(5)], [_unit(rng, 3) for _ in range(2)]]
    out = list(conv.convert(mvs))
    assert len(out) == 2
    assert all(len(v) == conv.dimension for v in out)


def test_empty_multivector_yields_the_zero_vector() -> None:
    conv = MuveraFdeConverter({"input_dimension": 3, "buckets": 4, "repetitions": 2})
    (doc,) = list(conv.convert([[]]))
    assert doc == [0.0] * conv.dimension
    (query,) = list(conv.encode_query([[]]))
    assert query == [0.0] * conv.dimension


def test_empty_batch_yields_nothing() -> None:
    conv = MuveraFdeConverter({"input_dimension": 3})
    assert list(conv.convert([])) == []
    assert list(conv.encode_query([])) == []


# --------------------------------------------------------------------------- #
# The faithful contract: a wrong-width token vector is rejected (anti-false-green)
# --------------------------------------------------------------------------- #


def test_convert_rejects_a_token_vector_of_the_wrong_width() -> None:
    conv = MuveraFdeConverter({"input_dimension": 4})
    with pytest.raises(ConfigError, match="width"):
        list(conv.convert([[[0.0, 0.0, 0.0]]]))  # width 3 ≠ declared 4


def test_encode_query_rejects_a_token_vector_of_the_wrong_width() -> None:
    conv = MuveraFdeConverter({"input_dimension": 4})
    with pytest.raises(ConfigError, match="width"):
        list(conv.encode_query([[[0.0] * 5]]))  # width 5 ≠ declared 4


# --------------------------------------------------------------------------- #
# Determinism (byte-reproducible → the Baseline bar)
# --------------------------------------------------------------------------- #


def test_two_instances_with_the_same_seed_produce_identical_output() -> None:
    rng = random.Random(7)
    mv = [_unit(rng, 8) for _ in range(6)]
    a = MuveraFdeConverter({"input_dimension": 8, "buckets": 8, "repetitions": 3, "seed": 0})
    b = MuveraFdeConverter({"input_dimension": 8, "buckets": 8, "repetitions": 3, "seed": 0})
    assert list(a.convert([mv])) == list(b.convert([mv]))
    assert list(a.encode_query([mv])) == list(b.encode_query([mv]))


def test_a_different_seed_gives_a_different_partition() -> None:
    rng = random.Random(9)
    mv = [_unit(rng, 8) for _ in range(6)]
    a = MuveraFdeConverter({"input_dimension": 8, "buckets": 8, "seed": 0})
    b = MuveraFdeConverter({"input_dimension": 8, "buckets": 8, "seed": 1})
    assert list(a.convert([mv])) != list(b.convert([mv]))


# --------------------------------------------------------------------------- #
# The query/document asymmetry and fill-empty are REAL (not cosmetic)
# --------------------------------------------------------------------------- #


def test_query_sum_differs_from_document_mean_when_a_cell_holds_multiple_tokens() -> None:
    # buckets=1 ⇒ every token lands in the one cell: the query sums it, the document means it, so
    # for >1 token the two vectors differ by exactly the token count — the asymmetry made observable.
    conv = MuveraFdeConverter({"input_dimension": 3, "buckets": 1})
    mv = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    (q,) = list(conv.encode_query([mv]))
    (d,) = list(conv.convert([mv]))
    assert q[0] == pytest.approx(4.0)  # sum
    assert d[0] == pytest.approx(2.0)  # mean


def test_fill_empty_changes_a_document_with_empty_cells() -> None:
    rng = random.Random(3)
    # A single-token multi-vector: it occupies exactly one of the many cells, leaving the rest empty,
    # so fill_empty has something to fill — the output must differ from the no-fill encoding.
    mv = [_unit(rng, 8)]
    filled = MuveraFdeConverter({"input_dimension": 8, "buckets": 16, "fill_empty": True})
    unfilled = MuveraFdeConverter({"input_dimension": 8, "buckets": 16, "fill_empty": False})
    assert list(filled.convert([mv])) != list(unfilled.convert([mv]))


# --------------------------------------------------------------------------- #
# THE HEADLINE: the FDE dot-product approximates MaxSim (ADR-0050, AC1/AC4)
# --------------------------------------------------------------------------- #


def test_fde_dot_product_tracks_maxsim_as_document_overlap_grows() -> None:
    """A planted, deterministic family: doc_j shares the first j of the query's tokens (exact),
    padded with random tokens. True MaxSim rises with j (each shared token contributes ~1.0), so a
    genuine MUVERA FDE — the asymmetric sum/mean-fill construction — must have its dense dot-product
    rise with it too. A passthrough or a content-blind reduction would NOT. We assert both a strong
    Pearson correlation with true MaxSim AND that the full-overlap document ranks first.
    """
    dim, k = 32, 6
    rng = random.Random(2024)
    query = [_unit(rng, dim) for _ in range(k)]
    docs: list[list[list[float]]] = []
    for j in range(k + 1):
        shared = [query[i] for i in range(j)]  # exact copies of the first j query tokens
        noise = [_unit(rng, dim) for _ in range(k - j)]
        docs.append(shared + noise)

    conv = MuveraFdeConverter(
        {"input_dimension": dim, "buckets": 16, "repetitions": 24, "seed": 0}
    )
    (q_fde,) = list(conv.encode_query([query]))
    doc_fdes = list(conv.convert(docs))
    fde_scores = [_dot(q_fde, d) for d in doc_fdes]
    true_scores = [_maxsim(query, d) for d in docs]

    # The estimate correlates strongly with the exact late-interaction score.
    assert _pearson(fde_scores, true_scores) > 0.9
    # And the full-overlap document (max true MaxSim) is ranked first by the FDE dot-product.
    assert fde_scores.index(max(fde_scores)) == true_scores.index(max(true_scores)) == k


def test_fde_retrieval_recovers_the_planted_nearest_document() -> None:
    """Among many random documents, the one seeded with the query's own tokens must win the FDE
    dense-dot ranking — the retrieval property the ``fde_embedding`` column exists to provide."""
    dim, k = 24, 5
    rng = random.Random(77)
    query = [_unit(rng, dim) for _ in range(k)]
    planted = list(query)  # the document that IS the query's tokens
    distractors = [[_unit(rng, dim) for _ in range(k)] for _ in range(30)]
    corpus = [planted, *distractors]

    conv = MuveraFdeConverter(
        {"input_dimension": dim, "buckets": 16, "repetitions": 16, "seed": 0}
    )
    (q_fde,) = list(conv.encode_query([query]))
    scores = [_dot(q_fde, d) for d in conv.convert(corpus)]
    assert scores.index(max(scores)) == 0  # the planted document ranks #1


# --------------------------------------------------------------------------- #
# Config validation + license profile
# --------------------------------------------------------------------------- #


def test_missing_input_dimension_is_rejected() -> None:
    with pytest.raises(ConfigError, match="input_dimension"):
        MuveraFdeConverter({})


def test_buckets_must_be_a_power_of_two() -> None:
    with pytest.raises(ConfigError, match="power of two"):
        MuveraFdeConverter({"input_dimension": 4, "buckets": 6})


def test_repetitions_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="repetitions"):
        MuveraFdeConverter({"input_dimension": 4, "repetitions": 0})


def test_single_bucket_is_valid_and_holds_every_token() -> None:
    # buckets=1 ⇒ zero hyperplanes ⇒ one cell; a degenerate but valid FDE (no crash).
    conv = MuveraFdeConverter({"input_dimension": 3, "buckets": 1})
    (out,) = list(conv.convert([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]))
    assert out == [2.0, 2.0, 2.0]  # the mean of the two tokens, in the one cell


def test_profile_declares_a_verified_pure_code_repo_license_and_no_model() -> None:
    prof = profile_of(MuveraFdeConverter)
    assert isinstance(prof, ProviderProfile)
    # Training-free, data-oblivious ⇒ no weights to defer: an honest VERIFIED Apache-2.0 (not the
    # UNVERIFIED sentinel sparse.splade must use for its un-chosen model).
    assert prof.model_id is None
    assert prof.license == REPO_LICENSE_CODE
    assert prof.license_verified is True
    assert prof.license_weights is None
    assert prof.license_code == REPO_LICENSE_CODE
    assert prof.deterministic is True
    assert prof.compute == "cpu"

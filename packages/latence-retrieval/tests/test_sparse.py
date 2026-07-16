"""Query-side sparse generation (:class:`SpladeQueryEncoder`), tested through the ``encode`` seam.

The encoder's whole job is to **reuse** the index-time ADR-0053 SPLADE composition over the query
text, so the load-bearing test asserts exactly that: an encoded query equals what the index-time
:class:`~latence_splade.provider.SpladeCompositionEmbedder` produces for the same text — proving the
query and the corpus land in one sparse space, not two drifting re-implementations. Every assertion
is on the returned :class:`~latence_core.capability.SparseVector`, driven by an injected faithful
:class:`FakeSpladeModel` (the offline second adapter of the ``SpladeModel`` seam), so no
``transformers`` weights are touched.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest
from latence_core.capability import SparseVector
from latence_retrieval import (
    SparseQueryEncoder,
    SpladeQueryEncoder,
    resolve_processor,
    resolve_sparse_encoder,
)

# Query-side sparse reuses `latence-splade` (the `[sparse]` extra). It is a first-party workspace
# package installed `--no-deps` in CI/verify-local (exactly like latence-splade's own offline tests);
# skip the module cleanly if the optional extra is absent rather than erroring at collection.
pytest.importorskip("latence_splade")

from latence_splade.provider import SpladeCompositionEmbedder  # noqa: E402

# --------------------------------------------------------------------------- #
# A faithful fake SpladeModel — the offline second adapter of the model seam.
# (Mirrors the one in latence-splade's own tests: term weights = per-text term
#  frequencies over a fixed vocab; term_ids = the surface's unique subword ids.)
# --------------------------------------------------------------------------- #

_FAKE_VOCAB = 64
_TOKEN = re.compile(r"[a-z0-9]+")


def _tid(token: str) -> int:
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _FAKE_VOCAB


class FakeSpladeModel:
    """Deterministic, torch-free ``SpladeModel``: the honest double the composition is driven by."""

    def __init__(self, vocab_size: int = _FAKE_VOCAB) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def term_weights(self, texts: Any) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        for text in texts:
            terms: dict[int, float] = {}
            for token in _TOKEN.findall(text.lower()):
                idx = _tid(token)
                terms[idx] = terms.get(idx, 0.0) + 1.0
            out.append(terms)
        return out

    def term_ids(self, term: str) -> list[int]:
        seen: set[int] = set()
        ids: list[int] = []
        for token in _TOKEN.findall(term.lower()):
            idx = _tid(token)
            if idx not in seen:
                seen.add(idx)
                ids.append(idx)
        return ids


def _as_dict(vector: SparseVector) -> dict[int, float]:
    return dict(zip(vector.indices, vector.values, strict=True))


def _encoder(config: dict[str, Any] | None = None) -> SpladeQueryEncoder:
    return SpladeQueryEncoder(config or {}, model=FakeSpladeModel())


# --- seam identity ----------------------------------------------------------------------------


def test_encoder_satisfies_the_sparse_query_encoder_seam() -> None:
    assert isinstance(_encoder(), SparseQueryEncoder)


# --- the AC: the query vector mirrors the index-time composition (same class, same output) -----


def test_encoded_query_equals_the_index_time_composition_for_the_same_text() -> None:
    # The load-bearing reuse guarantee: query-side sparse is byte-identical to what the index-time
    # SparseEmbedder emits for that text — one composition, two clocks (ADR-0053).
    model = FakeSpladeModel()
    query = "acme corp quarterly filing"
    encoder = SpladeQueryEncoder({}, model=model)
    index_time = SpladeCompositionEmbedder({}, model=model)
    (index_vector,) = list(index_time.encode_sparse([query]))
    assert encoder.encode(query) == index_vector


def test_encode_emits_the_model_leg_over_the_query_text() -> None:
    assert _as_dict(_encoder().encode("acme acme corp")) == {_tid("acme"): 2.0, _tid("corp"): 1.0}


def test_encoded_query_is_a_canonical_ascending_sparse_vector() -> None:
    vector = _encoder().encode("gamma delta epsilon")
    assert list(vector.indices) == sorted(vector.indices)
    assert len(vector.indices) == len(vector.values)


def test_empty_query_yields_the_empty_sparse_vector() -> None:
    assert _encoder().encode("") == SparseVector()


# --- query-time structured composition (entities + metadata, same knobs as index time) ---------


def test_confidence_gated_query_entities_are_injected_over_the_model_leg() -> None:
    encoder = _encoder({"confidence_threshold": 0.5, "entity_boost": 1.0})
    vector = encoder.encode("acme corp", entities=[("acme", 0.9), ("corp", 0.2)])
    terms = _as_dict(vector)
    assert terms[_tid("acme")] == 2.0  # 0.9 >= gate → model 1.0 + boost 1.0
    assert terms[_tid("corp")] == 1.0  # 0.2 < gate → model weight only, no boost


def test_redaction_default_drops_a_query_entity_absent_from_the_query_text() -> None:
    # redact=True (default): an entity whose surface is not in the query text is not injected.
    vector = _encoder().encode("the invoice total", entities=[("john", 0.99)])
    assert _tid("john") not in _as_dict(vector)


def test_redact_false_injects_an_expansion_term_not_present_in_the_query() -> None:
    # redact=False is the query-expansion opt-in: inject a confident term even if not in the query.
    vector = _encoder({"redact": False}).encode("the invoice total", entities=[("john", 0.99)])
    assert _as_dict(vector)[_tid("john")] == 1.0


def test_selected_metadata_fields_are_injected_at_query_time() -> None:
    encoder = _encoder({"metadata_fields": ["classification"], "metadata_boost": 3.0})
    vector = encoder.encode("quarterly report", metadata={"classification": "confidential"})
    assert _as_dict(vector)[_tid("confidential")] == 3.0


def test_weighted_mode_scales_the_model_leg_like_index_time() -> None:
    vector = _encoder({"mode": "weighted", "model_weight": 0.5}).encode("acme acme corp")
    assert _as_dict(vector)[_tid("acme")] == 1.0  # 2.0 * 0.5


# --- config selection (the AC: sparse model chosen by config, same seam as index time) ---------


def test_resolve_sparse_encoder_selects_splade_by_name() -> None:
    # Construction is lazy (no model built), so a model id in config is not touched here.
    encoder = resolve_sparse_encoder(
        {"provider": "query_sparse.splade", "config": {"model": "x/y"}}
    )
    assert isinstance(encoder, SpladeQueryEncoder)
    assert isinstance(encoder, SparseQueryEncoder)


def test_resolve_sparse_encoder_rejects_a_non_encoder_component() -> None:
    # A CandidateProcessor is a different shape; it must not satisfy the sparse-encoder seam.
    with pytest.raises(TypeError, match="SparseQueryEncoder seam"):
        resolve_sparse_encoder({"provider": "processor.identity", "config": {}})


def test_resolve_processor_rejects_the_sparse_encoder() -> None:
    # Symmetry: the sparse encoder is not a CandidateProcessor and must be rejected as one.
    with pytest.raises(TypeError, match="CandidateProcessor seam"):
        resolve_processor({"provider": "query_sparse.splade", "config": {"model": "x/y"}})


# --- purity -----------------------------------------------------------------------------------


def test_encoding_is_deterministic_across_repeated_calls() -> None:
    encoder = _encoder({"metadata_fields": ["classification"]})
    first = encoder.encode("acme corp", entities=[("acme", 0.9)], metadata={"classification": "x"})
    second = encoder.encode("acme corp", entities=[("acme", 0.9)], metadata={"classification": "x"})
    assert first == second

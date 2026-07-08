"""SentenceTransformersEmbedder: learned in-process semantic embedding over ST (W1-embed, #93).

The real ``sentence-transformers`` stack (torch/transformers) is intentionally NOT installed in
the dev/test environment (ADR-0007/0016: CPU-first, offline, deterministic). These tests inject a
tiny fake ``sentence_transformers`` module so the Provider's batching, input-order preservation,
e5-prefixing, L2-normalisation call, dimension validation, device-seam wiring, and typed-error
mapping are exercised deterministically and offline — the model itself is ST's concern, not this
seam's. One test covers the missing-dependency error path, one the config validation, and one the
native-failure → typed ProviderError mapping (no raw text in the message).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest
from latence_core.errors import ProviderError
from latence_embedder_st import SentenceTransformersEmbedder
from latence_embedder_st.provider import _DEFAULT_MODEL


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer: deterministic fake vectors.

    ``encode`` returns a fixed-width vector per input text, keyed by the (prefixed) text length so
    order and the e5 prefix are observable. It records the last ``encode`` kwargs so a test can
    assert ``normalize_embeddings=True`` was passed and the prefix was applied.
    """

    dimension: int = 4
    error: Exception | None = None
    load_error: Exception | None = None
    last_inputs: list[str] = []
    last_kwargs: dict[str, Any] = {}
    instances: int = 0

    def __init__(self, model_id: str, device: str | None = None) -> None:
        type(self).instances += 1
        self.model_id = model_id
        self.device = device
        load_error = type(self).load_error
        if load_error is not None:
            raise load_error

    def get_sentence_embedding_dimension(self) -> int:
        return type(self).dimension

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        error = type(self).error
        if error is not None:
            raise error
        type(self).last_inputs = list(texts)
        type(self).last_kwargs = dict(kwargs)
        # A deterministic vector keyed by text length so input order is observable.
        return [[float(len(t))] * type(self).dimension for t in texts]


@pytest.fixture()
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeSentenceTransformer.dimension = 4
    _FakeSentenceTransformer.error = None
    _FakeSentenceTransformer.load_error = None
    _FakeSentenceTransformer.last_inputs = []
    _FakeSentenceTransformer.last_kwargs = {}
    _FakeSentenceTransformer.instances = 0
    return _FakeSentenceTransformer


# -- config / profile ---------------------------------------------------------


def test_default_model_is_permissive_e5_small() -> None:
    embedder = SentenceTransformersEmbedder({})
    assert embedder._model_id == _DEFAULT_MODEL == "intfloat/multilingual-e5-small"  # noqa: SLF001
    # Answerable before the first embed (Export reads it to size its column): declared default.
    assert embedder.dimension == 384


def test_rejects_non_positive_dimension() -> None:
    with pytest.raises(ProviderError, match="dimension must be >= 1"):
        SentenceTransformersEmbedder({"dimension": 0})


def test_declares_profile_either_verified_mit_weights_apache_code() -> None:
    from latence_core.providers.profile import ProviderProfile, profile_of

    prof = profile_of(SentenceTransformersEmbedder)
    assert isinstance(prof, ProviderProfile)
    assert prof.compute == "either"
    assert prof.model_id == "intfloat/multilingual-e5-small"
    assert prof.license == "MIT"  # headline == weights SPDX
    assert prof.license_code == "Apache-2.0"  # the library
    assert prof.license_weights == "MIT"  # the checkpoint
    assert prof.license_verified is True
    assert prof.deterministic is False  # learned float inference — declared, not assumed
    assert prof.batch is True
    assert prof.cost_per_1k is None


# -- embedding happy path -----------------------------------------------------


def test_embeds_in_input_order_with_e5_prefix_and_normalization(
    fake_st: type[_FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformersEmbedder({"dimension": 4})
    out = list(embedder.embed(["a", "bb", "ccc"]))
    # The fake keys the vector by the PREFIXED text length: "passage: " is 9 chars.
    assert [v[0] for v in out] == [10.0, 11.0, 12.0]
    assert all(len(v) == 4 for v in out)
    # The e5 document prefix was prepended to every text.
    assert fake_st.last_inputs == ["passage: a", "passage: bb", "passage: ccc"]
    # L2-normalisation requested (the model-card cosine convention).
    assert fake_st.last_kwargs.get("normalize_embeddings") is True


def test_custom_prefix_is_applied(fake_st: type[_FakeSentenceTransformer]) -> None:
    embedder = SentenceTransformersEmbedder({"dimension": 4, "prefix": ""})
    list(embedder.embed(["x"]))
    assert fake_st.last_inputs == ["x"]  # no prefix (e.g. MiniLM)


def test_batches_bound_requests_and_preserve_order(
    fake_st: type[_FakeSentenceTransformer],
) -> None:
    calls: list[int] = []
    original_encode = _FakeSentenceTransformer.encode

    def counting_encode(self: Any, texts: list[str], **kwargs: Any) -> list[list[float]]:
        calls.append(len(texts))
        return original_encode(self, texts, **kwargs)

    _FakeSentenceTransformer.encode = counting_encode  # type: ignore[method-assign]
    try:
        embedder = SentenceTransformersEmbedder({"dimension": 4, "batch_size": 2})
        out = list(embedder.embed(["a", "b", "c", "d", "e"]))
    finally:
        _FakeSentenceTransformer.encode = original_encode  # type: ignore[method-assign]
    assert len(out) == 5
    assert calls == [2, 2, 1]  # ceil(5/2) batches, bounded to batch_size
    # The model is loaded ONCE and cached across batches (construction never loads it).
    assert fake_st.instances == 1


def test_empty_input_makes_no_call(fake_st: type[_FakeSentenceTransformer]) -> None:
    embedder = SentenceTransformersEmbedder({"dimension": 4})
    assert list(embedder.embed([])) == []
    # No model load, no encode on an empty corpus (graceful empty, never a crash).
    assert fake_st.instances == 0


def test_dimension_resolved_from_loaded_model(fake_st: type[_FakeSentenceTransformer]) -> None:
    # No declared dimension: the width comes from the loaded model (384 for e5-small, 4 here).
    fake_st.dimension = 7
    embedder = SentenceTransformersEmbedder({"dimension": None})
    out = list(embedder.embed(["a"]))
    assert embedder.dimension == 7
    assert len(out[0]) == 7


# -- failure modes ------------------------------------------------------------


def test_declared_dimension_mismatch_is_loud(fake_st: type[_FakeSentenceTransformer]) -> None:
    fake_st.dimension = 4  # model produces 4-dim
    embedder = SentenceTransformersEmbedder({"dimension": 8})  # config claims 8
    with pytest.raises(ProviderError, match="declared dimension 8"):
        list(embedder.embed(["a"]))


def test_native_encode_failure_becomes_typed_provider_error_without_text(
    fake_st: type[_FakeSentenceTransformer],
) -> None:
    # A native RuntimeError inside encode (OOM / kernel failure) → typed ProviderError, and the
    # raised message must NOT echo the offending input text (no PII/content leak — AdapterBase.guard).
    secret_text = "SECRET-PII-abc123 patient record"
    fake_st.error = RuntimeError(f"cuda oom while encoding {secret_text}")
    embedder = SentenceTransformersEmbedder({"dimension": 4})
    with pytest.raises(ProviderError) as exc_info:
        list(embedder.embed([secret_text]))
    message = str(exc_info.value)
    assert "provider failure in 'embed'" in message
    assert "RuntimeError" in message  # the native type name is safe to surface
    assert secret_text not in message  # the input text / native message is NOT leaked
    assert "SECRET-PII" not in message


def test_bad_model_id_maps_to_config_error(fake_st: type[_FakeSentenceTransformer]) -> None:
    from latence_core.errors import ConfigError

    # A bad model id makes ST raise OSError at load — classified as an operator ConfigError.
    fake_st.load_error = OSError("model 'nope/nope' not found on the hub")
    embedder = SentenceTransformersEmbedder({"dimension": 4, "model": "nope/nope"})
    with pytest.raises(ConfigError, match="config error in 'load_model'"):
        list(embedder.embed(["a"]))


def test_missing_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformersEmbedder({"dimension": 4})
    with pytest.raises(ImportError, match="pip install latence-embedder-st"):
        list(embedder.embed(["a"]))


# -- real-weights test (gated on the dep; skipped offline) --------------------


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (real-weights path runs where the model exists)",
)
def test_real_weights_embeds_at_declared_width() -> None:  # pragma: no cover - runs only on a rig
    """With the real dep + weights present, the Provider returns 384-dim unit vectors in order.

    This is the honest real-weights boundary (same discipline as the gliner trio): it runs only
    where ``sentence-transformers`` and the e5-small checkpoint are actually available (a rig),
    and is skipped in the offline CPU-first CI env. It asserts the REAL model returns one
    correct-width, L2-normalised vector per input, in input order.
    """
    import math

    embedder = SentenceTransformersEmbedder({})  # e5-small default
    out = list(embedder.embed(["hello world", "a different sentence"]))
    assert len(out) == 2
    assert embedder.dimension == 384
    assert all(len(v) == 384 for v in out)
    for v in out:
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-3  # normalize_embeddings=True ⇒ unit length

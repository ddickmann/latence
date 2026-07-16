"""The ``rerank`` seam (:class:`Reranker` over a :class:`CrossEncoderModel`), tested at the boundary.

Two boundaries, both offline (no ``sentence-transformers``/torch installed — ADR-0016):

* the **:class:`Reranker` processor** driven through an injected faithful
  :class:`~latence_retrieval.testing.FakeCrossEncoder` — every observable behaviour (re-score,
  reorder, tie-break, ``top_k`` truncation, no-mutation/no-state, one batch call) asserted on the
  returned candidate list, never the internal scoring maths, so the tests survive a refactor of *how*
  the reranking is computed;
* the **real** :class:`~latence_retrieval.rerank.SentenceTransformersCrossEncoder` against a fake
  ``sentence_transformers`` module — the pair-build → ``predict`` path and its typed error mapping —
  the same honest fake-module boundary the splade/gliner packages use.

A real-weights test is present and gated on the dep (skipped offline).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest
from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    CrossEncoderModel,
    Reranker,
    RerankModelError,
    SentenceTransformersCrossEncoder,
    resolve_processor,
)
from latence_retrieval.testing import FakeCrossEncoder


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


def _reranker(config: dict[str, Any] | None = None) -> Reranker:
    return Reranker(config or {}, model=FakeCrossEncoder())


# --- seam identity ----------------------------------------------------------------------------


def test_reranker_satisfies_the_candidate_processor_seam() -> None:
    assert isinstance(_reranker(), CandidateProcessor)


def test_fake_cross_encoder_satisfies_the_cross_encoder_seam() -> None:
    assert isinstance(FakeCrossEncoder(), CrossEncoderModel)


def test_real_adapter_satisfies_the_cross_encoder_seam() -> None:
    assert isinstance(SentenceTransformersCrossEncoder("some/model"), CrossEncoderModel)


# --- rerank behaviour (via the faithful lexical-overlap fake) ---------------------------------


def test_rerank_reorders_by_cross_encoder_relevance_not_first_stage_score() -> None:
    # First-stage order puts the least query-relevant candidate on top; rerank must fix that.
    query = "annual revenue growth"
    candidates = [
        Candidate(id="a", score=9.9, text="a poem about the sea"),
        Candidate(id="b", score=0.1, text="annual revenue growth accelerated"),
        Candidate(id="c", score=5.0, text="revenue growth was flat"),
    ]
    reranked = _reranker().process(query, candidates)
    # 'b' shares 3 query terms, 'c' shares 2, 'a' shares 0 → strict rerank order.
    assert _ids(reranked) == ["b", "c", "a"]


def test_rerank_replaces_the_score_with_the_cross_encoder_score() -> None:
    candidates = [
        Candidate(id="a", score=0.42, text="alpha beta"),
        Candidate(id="b", score=0.99, text="gamma"),
    ]
    reranked = _reranker().process("alpha beta gamma", candidates)
    by_id = {c.id: c for c in reranked}
    # Overlap scores: 'a' shares {alpha, beta}=2, 'b' shares {gamma}=1 — the new scores, not 0.42/0.99.
    assert by_id["a"].score == pytest.approx(2.0)
    assert by_id["b"].score == pytest.approx(1.0)


def test_rerank_keeps_text_and_metadata_untouched() -> None:
    candidates = [Candidate(id="a", score=0.1, text="alpha", metadata={"src": "s3", "page": 4})]
    (only,) = _reranker().process("alpha", candidates)
    assert only.text == "alpha"
    assert only.metadata == {"src": "s3", "page": 4}


def test_rerank_breaks_ties_by_id_ascending() -> None:
    # Three candidates with identical overlap (each shares exactly one query term) → deterministic id
    # order, mirroring the fuse seam's tie-break.
    candidates = [
        Candidate(id="c", score=1.0, text="alpha"),
        Candidate(id="a", score=1.0, text="beta"),
        Candidate(id="b", score=1.0, text="gamma"),
    ]
    reranked = _reranker().process("alpha beta gamma", candidates)
    assert _ids(reranked) == ["a", "b", "c"]


def test_rerank_top_k_truncates_to_the_best_k() -> None:
    query = "one two three"
    candidates = [
        Candidate(id="a", score=0.0, text="one two three"),  # 3
        Candidate(id="b", score=0.0, text="one two"),  # 2
        Candidate(id="c", score=0.0, text="one"),  # 1
        Candidate(id="d", score=0.0, text="nothing here"),  # 0
    ]
    reranked = _reranker({"top_k": 2}).process(query, candidates)
    assert _ids(reranked) == ["a", "b"]


def test_rerank_without_top_k_keeps_every_candidate() -> None:
    candidates = [
        Candidate(id="a", score=0.0, text="one"),
        Candidate(id="b", score=0.0, text="two"),
    ]
    assert len(_reranker().process("one two", candidates)) == 2


# --- purity: no mutation, no state, batch-efficient -------------------------------------------


def test_rerank_does_not_mutate_the_input_list_or_candidates() -> None:
    candidates = [
        Candidate(id="a", score=0.9, text="alpha"),
        Candidate(id="b", score=0.8, text="beta"),
    ]
    snapshot = list(candidates)
    _reranker().process("alpha", candidates)
    assert candidates == snapshot
    assert candidates[0].score == 0.9  # frozen candidate untouched


def test_rerank_is_deterministic_across_repeated_calls() -> None:
    candidates = [
        Candidate(id="a", score=0.1, text="alpha beta"),
        Candidate(id="b", score=0.2, text="beta"),
    ]
    reranker = _reranker()
    assert reranker.process("alpha beta", candidates) == reranker.process("alpha beta", candidates)


def test_rerank_empty_candidates_returns_empty_without_touching_the_model() -> None:
    # A model that would explode if called proves the empty short-circuit never invokes it.
    class _Exploding:
        def score(self, query: str, texts: Any) -> list[float]:
            raise AssertionError("model must not be called for an empty candidate list")

    assert Reranker({}, model=_Exploding()).process("q", []) == []


def test_rerank_scores_the_whole_list_in_a_single_batch_call() -> None:
    # Batch-efficiency is the contract: one model.score call carrying every candidate's text, not a
    # per-candidate loop.
    class _SpyModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def score(self, query: str, texts: Any) -> list[float]:
            self.calls.append(list(texts))
            return [0.0] * len(list(texts))

    spy = _SpyModel()
    candidates = [Candidate(id=str(i), score=0.0, text=f"t{i}") for i in range(5)]
    Reranker({}, model=spy).process("q", candidates)
    assert len(spy.calls) == 1
    assert spy.calls[0] == ["t0", "t1", "t2", "t3", "t4"]


# --- config selection (the AC: reranker chosen by config) -------------------------------------


def test_resolve_processor_selects_rerank_by_name() -> None:
    # `processor.rerank` resolves through the same registry as every query-time component; a missing
    # model id is not touched at construction (weights load lazily), so this succeeds offline.
    processor = resolve_processor({"provider": "processor.rerank", "config": {"model": "x/y"}})
    assert isinstance(processor, Reranker)
    assert isinstance(processor, CandidateProcessor)


def test_config_selected_rerank_reorders_end_to_end() -> None:
    # Prove the config-selected component actually reranks (with an injected fake so no weights).
    reranker = Reranker({"model": "x/y"}, model=FakeCrossEncoder())
    candidates = [
        Candidate(id="a", score=1.0, text="unrelated"),
        Candidate(id="b", score=0.0, text="the query terms"),
    ]
    assert _ids(reranker.process("the query terms", candidates)) == ["b", "a"]


# --- config validation ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_rerank_rejects_a_bad_top_k(bad: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        Reranker({"top_k": bad})


@pytest.mark.parametrize("bad", [0, -5, True, "512"])
def test_rerank_rejects_a_bad_max_length(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_length"):
        Reranker({"max_length": bad})


def test_rerank_model_returning_wrong_score_count_is_a_typed_error() -> None:
    class _ShortModel:
        def score(self, query: str, texts: Any) -> list[float]:
            return [1.0]  # one score for two candidates — a broken seam contract

    candidates = [Candidate(id="a", score=0.0), Candidate(id="b", score=0.0)]
    with pytest.raises(RerankModelError, match="one score per candidate"):
        Reranker({}, model=_ShortModel()).process("q", candidates)


# --- the faithful FakeCrossEncoder (anti-false-green: mirror/reject what the real API does) ----


def test_fake_returns_exactly_one_score_per_text_in_order() -> None:
    scores = FakeCrossEncoder().score("alpha beta", ["beta", "alpha beta", "nope"])
    assert scores == [1.0, 2.0, 0.0]


def test_fake_empty_batch_returns_empty() -> None:
    assert FakeCrossEncoder().score("anything", []) == []


def test_fake_rejects_a_non_string_query() -> None:
    # A real cross-encoder's tokenizer raises on a non-string; the fake must too (no silent coerce).
    with pytest.raises(TypeError, match="query must be a string"):
        FakeCrossEncoder().score(123, ["a"])  # type: ignore[arg-type]


def test_fake_rejects_a_non_string_text() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        FakeCrossEncoder().score("q", ["ok", None])  # type: ignore[list-item]


# --- the real adapter against a fake sentence_transformers module -----------------------------


class _FakeArr:
    """A minimal numpy-array stand-in: what ``CrossEncoder.predict`` returns (has ``tolist``)."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakePredictModel:
    """A faithful fake ``sentence_transformers.CrossEncoder``: scores pairs by lexical overlap."""

    def __init__(self, model_id: str, max_length: int = 512, **_kw: Any) -> None:
        if model_id == "bad/model":
            raise OSError("no such checkpoint")
        self.model_id = model_id
        self.max_length = max_length

    def predict(self, pairs: list[list[str]], **_kw: Any) -> _FakeArr:
        scores: list[float] = []
        for pair in pairs:
            query, text = pair  # a real predict wants exactly [text_a, text_b]
            shared = set(query.lower().split()) & set(text.lower().split())
            scores.append(float(len(shared)))
        return _FakeArr(scores)


class _RaisingPredictModel:
    def __init__(self, model_id: str, max_length: int = 512, **_kw: Any) -> None:
        pass

    def predict(self, pairs: list[list[str]], **_kw: Any) -> _FakeArr:
        raise RuntimeError("native torch blew up")


def _install_fake_st(monkeypatch: pytest.MonkeyPatch, cross_encoder: type) -> None:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = cross_encoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_real_adapter_scores_pairs_through_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_st(monkeypatch, _FakePredictModel)
    model = SentenceTransformersCrossEncoder("stub/cross-encoder")
    scores = model.score("annual revenue", ["annual revenue growth", "a poem"])
    assert scores == [2.0, 0.0]


def test_real_adapter_reranks_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # No injected model: the Reranker builds the real adapter from config and runs through predict.
    _install_fake_st(monkeypatch, _FakePredictModel)
    reranker = Reranker({"model": "stub/cross-encoder"})
    candidates = [
        Candidate(id="a", score=9.0, text="unrelated text"),
        Candidate(id="b", score=0.0, text="annual revenue growth"),
    ]
    assert _ids(reranker.process("annual revenue growth", candidates)) == ["b", "a"]


def test_real_adapter_empty_texts_returns_empty_without_loading() -> None:
    # No fake module installed: an empty batch must not import sentence-transformers.
    assert SentenceTransformersCrossEncoder("stub/cross-encoder").score("q", []) == []


def test_real_adapter_missing_model_id_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="requires a 'model'"):
        SentenceTransformersCrossEncoder("")


def test_reranker_without_a_model_fails_loudly_on_use() -> None:
    # No injected model, no 'model' id in config → the lazy real adapter rejects at first process.
    reranker = Reranker({})
    with pytest.raises(ValueError, match="requires a 'model'"):
        reranker.process("q", [Candidate(id="a", score=0.0, text="x")])


def test_real_adapter_bad_model_id_maps_to_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_st(monkeypatch, _FakePredictModel)
    model = SentenceTransformersCrossEncoder("bad/model")
    with pytest.raises(ValueError, match="could not load cross-encoder"):
        model.score("q", ["anything"])


def test_real_adapter_runtime_failure_maps_to_rerank_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_st(monkeypatch, _RaisingPredictModel)
    model = SentenceTransformersCrossEncoder("stub/cross-encoder")
    with pytest.raises(RerankModelError, match="failure during scoring"):
        model.score("q", ["anything"])


def test_real_adapter_missing_dependency_raises_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    model = SentenceTransformersCrossEncoder("stub/cross-encoder")
    with pytest.raises(ImportError, match=r"latence-retrieval\[rerank\]"):
        model.score("q", ["anything"])


# --- real-weights boundary (gated on the dep; skipped offline) --------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (real-weights path runs where a cleared model exists)",
)
def test_real_weights_rerank_smoke() -> None:  # pragma: no cover - runs only on a rig
    """With the real dep + a *cleared* checkpoint, the reranker returns a reordered list.

    Skipped offline. A real run pins a license-cleared cross-encoder via ``LATENCE_RERANK_MODEL``
    (never a hardcoded default — ADR-0012) and asserts the transform returns one score per candidate.
    """
    import os

    model_id = os.environ.get("LATENCE_RERANK_MODEL")
    if not model_id:
        pytest.skip("set LATENCE_RERANK_MODEL to a license-cleared cross-encoder to run this")
    reranker = Reranker({"model": model_id})
    candidates = [
        Candidate(id="a", score=0.0, text="the mitochondria is the powerhouse of the cell"),
        Candidate(id="b", score=0.0, text="quarterly revenue rose twelve percent"),
    ]
    reranked = reranker.process("what were the company earnings", candidates)
    assert {c.id for c in reranked} == {"a", "b"}

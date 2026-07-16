"""The ``pack`` seam (:class:`Packer` over a :class:`TokenCounter`), tested at the boundary.

Every assertion is on the *observable output* of the transform — which candidates are selected, the
total token cost respected, redundancy penalised, signals honoured — never the internal QKP maths, so
the tests survive a refactor of *how* the packing is computed. Two boundaries, both offline:

* the **:class:`Packer` processor** driven through the pure-Python day-one
  :class:`~latence_retrieval.pack.HeuristicTokenCounter` (and injected spy/exact counters), asserting
  budget, redundancy penalty, optional-signal use + graceful absence, determinism, no-mutation;
* the **real** :class:`~latence_retrieval.pack.TiktokenCounter` against a fake ``tiktoken`` module —
  the ``get_encoding → encode`` path and its typed error mapping — the same honest fake-module
  boundary the rerank/splade packages use.
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
    HeuristicTokenCounter,
    Packer,
    PackTokenizerError,
    TiktokenCounter,
    TokenCounter,
    resolve_processor,
)


def _ids(candidates: list[Candidate]) -> set[str]:
    return {c.id for c in candidates}


def _ordered_ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


# --- seam identity ----------------------------------------------------------------------------


def test_packer_satisfies_the_candidate_processor_seam() -> None:
    assert isinstance(Packer({"budget": 100}), CandidateProcessor)


def test_both_counters_satisfy_the_token_counter_seam() -> None:
    assert isinstance(HeuristicTokenCounter(), TokenCounter)
    assert isinstance(TiktokenCounter("cl100k_base"), TokenCounter)


def test_resolve_processor_selects_pack_by_name() -> None:
    processor = resolve_processor({"provider": "processor.pack", "config": {"budget": 50}})
    assert isinstance(processor, Packer)
    assert isinstance(processor, CandidateProcessor)


# --- budget respected (the core AC) -----------------------------------------------------------


def test_pack_respects_the_token_budget_using_exact_metadata_costs() -> None:
    # token_key gives exact per-candidate costs; the selected subset must fit the budget.
    candidates = [
        Candidate(id="a", score=1.0, text="alpha", metadata={"tok": 10}),
        Candidate(id="b", score=0.9, text="bravo", metadata={"tok": 10}),
        Candidate(id="c", score=0.8, text="charlie", metadata={"tok": 10}),
    ]
    packed = Packer({"budget": 25, "token_key": "tok", "redundancy_weight": 0.0}).process(
        "q", candidates
    )
    assert sum(c.metadata["tok"] for c in packed) <= 25
    assert len(packed) == 2  # two 10-token items fit in 25, a third does not


def test_pack_selects_the_highest_value_items_within_budget() -> None:
    candidates = [
        Candidate(id="low", score=0.1, text="one", metadata={"tok": 10}),
        Candidate(id="hi", score=0.9, text="two", metadata={"tok": 10}),
        Candidate(id="mid", score=0.5, text="three", metadata={"tok": 10}),
    ]
    packed = Packer({"budget": 20, "token_key": "tok", "redundancy_weight": 0.0}).process(
        "q", candidates
    )
    assert _ids(packed) == {"hi", "mid"}  # the two most relevant, not "low"


def test_pack_budget_smaller_than_every_item_returns_empty() -> None:
    candidates = [Candidate(id="a", score=1.0, text="x", metadata={"tok": 100})]
    assert Packer({"budget": 5, "token_key": "tok"}).process("q", candidates) == []


def test_pack_empty_candidates_returns_empty_without_touching_the_counter() -> None:
    class _Exploding:
        def count(self, text: str) -> int:
            raise AssertionError("counter must not be called for an empty candidate list")

    assert Packer({"budget": 10}, tokens=_Exploding()).process("q", []) == []


def test_pack_clamps_zero_cost_items_to_at_least_one_token() -> None:
    # An empty text counts 0 heuristically; the packer treats it as costing >=1 so a budget of 1
    # admits exactly one such item (not unboundedly many "free" ones).
    candidates = [Candidate(id=str(i), score=1.0, text="") for i in range(3)]
    packed = Packer({"budget": 1, "redundancy_weight": 0.0}).process("q", candidates)
    assert len(packed) == 1


# --- redundancy penalised (the core AC) -------------------------------------------------------


def test_pack_penalises_redundancy_dropping_a_near_duplicate_for_a_diverse_item() -> None:
    # dupA/dupB are near-identical (high similarity); "diverse" is unlike either. With the budget
    # fitting two of three, a redundancy-aware packer keeps dupA + diverse, NOT dupA + dupB.
    sims = {
        "dupA": {"dupB": 1.0, "diverse": 0.0},
        "dupB": {"dupA": 1.0, "diverse": 0.0},
        "diverse": {"dupA": 0.0, "dupB": 0.0},
    }
    candidates = [
        Candidate(id="dupA", score=1.0, text="x", metadata={"tok": 10, "s": sims["dupA"]}),
        Candidate(id="dupB", score=0.95, text="x", metadata={"tok": 10, "s": sims["dupB"]}),
        Candidate(id="diverse", score=0.6, text="y", metadata={"tok": 10, "s": sims["diverse"]}),
    ]
    packed = Packer(
        {"budget": 20, "token_key": "tok", "similarity_key": "s", "redundancy_weight": 0.5}
    ).process("q", candidates)
    assert _ids(packed) == {"dupA", "diverse"}


def test_pack_without_redundancy_penalty_keeps_the_near_duplicate() -> None:
    # Same inputs, redundancy_weight=0 → the packer is a plain top-value knapsack and keeps both
    # high-score duplicates over the diverse one. Proves the penalty (not some other effect) drove
    # the previous test.
    sims = {"dupA": {"dupB": 1.0}, "dupB": {"dupA": 1.0}}
    candidates = [
        Candidate(id="dupA", score=1.0, text="x", metadata={"tok": 10, "s": sims["dupA"]}),
        Candidate(id="dupB", score=0.95, text="x", metadata={"tok": 10, "s": sims["dupB"]}),
        Candidate(id="diverse", score=0.6, text="y", metadata={"tok": 10}),
    ]
    packed = Packer(
        {"budget": 20, "token_key": "tok", "similarity_key": "s", "redundancy_weight": 0.0}
    ).process("q", candidates)
    assert _ids(packed) == {"dupA", "dupB"}


def test_pack_default_redundancy_uses_lexical_overlap() -> None:
    # No similarity signal supplied: redundancy falls back to lexical token overlap over text.
    # "dupA"/"dupB" share every word; "diverse" shares none → dupB is the redundant one dropped.
    candidates = [
        Candidate(id="dupA", score=1.0, text="quarterly revenue rose sharply", metadata={"tok": 10}),
        Candidate(id="dupB", score=0.95, text="quarterly revenue rose sharply", metadata={"tok": 10}),
        Candidate(id="diverse", score=0.6, text="feline biology and habitats", metadata={"tok": 10}),
    ]
    packed = Packer({"budget": 20, "token_key": "tok", "redundancy_weight": 0.5}).process(
        "q", candidates
    )
    assert _ids(packed) == {"dupA", "diverse"}


def test_pack_treats_same_cluster_as_redundant() -> None:
    # cluster_key: two items in cluster "c1" are redundant even with disjoint text; the packer keeps
    # one c1 item plus the c2 item rather than both c1 items.
    candidates = [
        Candidate(id="a", score=1.0, text="alpha", metadata={"tok": 10, "cl": "c1"}),
        Candidate(id="b", score=0.95, text="bravo", metadata={"tok": 10, "cl": "c1"}),
        Candidate(id="c", score=0.6, text="charlie", metadata={"tok": 10, "cl": "c2"}),
    ]
    packed = Packer(
        {"budget": 20, "token_key": "tok", "cluster_key": "cl", "redundancy_weight": 0.5}
    ).process("q", candidates)
    assert _ids(packed) == {"a", "c"}


# --- optional signals used + graceful absence (the core AC) -----------------------------------


def test_pack_runs_on_defaults_with_no_signals_and_no_metadata() -> None:
    # The day-one path: budget only, no token_key, no signals, plain text — still returns a budgeted
    # subset via the heuristic counter.
    candidates = [Candidate(id=str(i), score=1.0 - i / 10, text=f"word{i} " * 5) for i in range(6)]
    # Each text is 5 heuristic tokens; a 12-token budget admits 2, not all 6.
    packed = Packer({"budget": 12}).process("q", candidates)
    assert 0 < len(packed) < 6  # some fit, not all
    assert _ids(packed).issubset({str(i) for i in range(6)})


def test_pack_uses_a_value_signal_to_promote_a_lower_scored_candidate() -> None:
    # Base scores favour "a"; a strong centrality signal on "b" flips the selection when only one
    # fits. Proves the optional signal is actually used.
    candidates = [
        Candidate(id="a", score=0.9, text="x", metadata={"tok": 10, "centrality": 0.0}),
        Candidate(id="b", score=0.1, text="y", metadata={"tok": 10, "centrality": 5.0}),
    ]
    packed = Packer(
        {"budget": 10, "token_key": "tok", "value_signals": {"centrality": 1.0}}
    ).process("q", candidates)
    assert _ids(packed) == {"b"}


def test_pack_ignores_a_signal_key_absent_on_a_candidate() -> None:
    # value_signals names "recency"; one candidate has it, one does not. The absent one must not
    # crash and simply gets no bonus — graceful degradation per candidate.
    candidates = [
        Candidate(id="a", score=0.5, text="x", metadata={"tok": 10, "recency": 3.0}),
        Candidate(id="b", score=0.5, text="y", metadata={"tok": 10}),  # no recency
    ]
    packed = Packer(
        {"budget": 10, "token_key": "tok", "value_signals": {"recency": 1.0}}
    ).process("q", candidates)
    assert _ids(packed) == {"a"}  # the one with the recency bonus wins the single slot


def test_pack_rejects_a_non_numeric_signal_value_where_present() -> None:
    candidates = [Candidate(id="a", score=0.5, text="x", metadata={"tok": 10, "c": "high"})]
    with pytest.raises(ValueError, match="value_signal 'c'"):
        Packer({"budget": 10, "token_key": "tok", "value_signals": {"c": 1.0}}).process(
            "q", candidates
        )


def test_pack_consumes_the_frameworks_per_entity_centrality_list_by_its_peak() -> None:
    # R5 regression: the framework emits kg_node_centrality as a per-entity list<double>
    # (export._context_columns_for_row), and both export.py and graph_features.py document it as
    # pack's centrality value-signal. The OLD scalar-only _signal_bonus raised ValueError on the
    # emitted list; the fix reduces it to its PEAK, exactly like expand._peak_centrality, so the
    # documented graph-augmented-pack wiring runs on the framework's own corpus.
    #
    # Base scores favour "a" (max-scaled relevance 1.0); "b" carries the emitted per-entity
    # centrality list whose PEAK (5.0) supplies a signal bonus that flips the single-slot selection.
    # On the pre-fix code this call raised ValueError on the list instead of packing.
    candidates = [
        Candidate(id="a", score=0.9, text="x", metadata={"tok": 10, "kg_node_centrality": [0.0, 0.0]}),
        Candidate(id="b", score=0.1, text="y", metadata={"tok": 10, "kg_node_centrality": [1.0, 5.0]}),
    ]
    packed = Packer(
        {"budget": 10, "token_key": "tok", "value_signals": {"kg_node_centrality": 1.0}}
    ).process("q", candidates)
    assert _ids(packed) == {"b"}


def test_pack_centrality_list_peak_matches_the_equivalent_scalar_signal() -> None:
    # The list<double> peak must promote a candidate EXACTLY as the equivalent scalar would, so the
    # two intake shapes are interchangeable (the peak IS the scalar). "b"'s peak is 5.0; the run
    # driven by the list and the run driven by the scalar 5.0 must select identically.
    config = {"budget": 10, "token_key": "tok", "value_signals": {"kg_node_centrality": 1.0}}
    list_candidates = [
        Candidate(id="a", score=0.9, text="x", metadata={"tok": 10, "kg_node_centrality": [0.0]}),
        Candidate(id="b", score=0.1, text="y", metadata={"tok": 10, "kg_node_centrality": [1.0, 5.0]}),
    ]
    scalar_candidates = [
        Candidate(id="a", score=0.9, text="x", metadata={"tok": 10, "kg_node_centrality": 0.0}),
        Candidate(id="b", score=0.1, text="y", metadata={"tok": 10, "kg_node_centrality": 5.0}),
    ]
    packed_list = Packer(config).process("q", list_candidates)
    packed_scalar = Packer(config).process("q", scalar_candidates)
    assert _ids(packed_list) == _ids(packed_scalar) == {"b"}


def test_pack_empty_centrality_list_contributes_no_bonus() -> None:
    # An unenriched chunk emits kg_node_centrality = [] (export's graph-features-OFF path). Like
    # expand, an empty list is no signal (0.0) — never a crash and never a spurious bonus, so the
    # higher base-scored candidate wins the single slot.
    candidates = [
        Candidate(id="a", score=0.9, text="x", metadata={"tok": 10, "kg_node_centrality": []}),
        Candidate(id="b", score=0.1, text="y", metadata={"tok": 10, "kg_node_centrality": []}),
    ]
    packed = Packer(
        {"budget": 10, "token_key": "tok", "value_signals": {"kg_node_centrality": 1.0}}
    ).process("q", candidates)
    assert _ids(packed) == {"a"}


def test_pack_rejects_a_non_numeric_element_in_a_signal_list() -> None:
    candidates = [
        Candidate(id="a", score=0.5, text="x", metadata={"tok": 10, "kg_node_centrality": [0.5, "hi"]})
    ]
    with pytest.raises(ValueError, match="value_signal 'kg_node_centrality'"):
        Packer(
            {"budget": 10, "token_key": "tok", "value_signals": {"kg_node_centrality": 1.0}}
        ).process("q", candidates)


def test_pack_rejects_a_non_finite_element_in_a_signal_list() -> None:
    candidates = [
        Candidate(
            id="a",
            score=0.5,
            text="x",
            metadata={"tok": 10, "kg_node_centrality": [0.5, float("inf")]},
        )
    ]
    with pytest.raises(ValueError, match="value_signal 'kg_node_centrality' must be finite"):
        Packer(
            {"budget": 10, "token_key": "tok", "value_signals": {"kg_node_centrality": 1.0}}
        ).process("q", candidates)


# --- determinism + purity ---------------------------------------------------------------------


def test_pack_is_deterministic_across_repeated_calls() -> None:
    candidates = [
        Candidate(id="a", score=0.5, text="alpha beta", metadata={"tok": 10}),
        Candidate(id="b", score=0.5, text="beta gamma", metadata={"tok": 10}),
        Candidate(id="c", score=0.5, text="delta", metadata={"tok": 10}),
    ]
    packer = Packer({"budget": 20, "token_key": "tok"})
    assert packer.process("q", candidates) == packer.process("q", candidates)


def test_pack_breaks_ties_by_id_ascending() -> None:
    # Identical value + cost + zero redundancy → order determined by the id tiebreak, and the
    # returned subset is sorted by (-score, id).
    candidates = [
        Candidate(id="c", score=1.0, text="x", metadata={"tok": 10}),
        Candidate(id="a", score=1.0, text="y", metadata={"tok": 10}),
        Candidate(id="b", score=1.0, text="z", metadata={"tok": 10}),
    ]
    packed = Packer({"budget": 20, "token_key": "tok", "redundancy_weight": 0.0}).process(
        "q", candidates
    )
    assert _ordered_ids(packed) == ["a", "b"]  # two of three fit; smallest ids, sorted


def test_pack_does_not_mutate_the_input_list_or_candidates() -> None:
    candidates = [
        Candidate(id="a", score=0.9, text="alpha", metadata={"tok": 10}),
        Candidate(id="b", score=0.8, text="beta", metadata={"tok": 10}),
    ]
    snapshot = list(candidates)
    Packer({"budget": 20, "token_key": "tok"}).process("q", candidates)
    assert candidates == snapshot
    assert candidates[0].score == 0.9  # frozen candidate untouched


def test_pack_returns_the_originals_unchanged_selecting_not_rescoring() -> None:
    candidates = [Candidate(id="a", score=0.42, text="alpha", metadata={"tok": 5, "k": "v"})]
    (only,) = Packer({"budget": 10, "token_key": "tok"}).process("q", candidates)
    assert only.score == 0.42  # pack selects; it does not replace the score (unlike rerank)
    assert only.metadata == {"tok": 5, "k": "v"}


# --- config validation ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 0, -1, True, 1.5, "10"])
def test_pack_rejects_a_bad_budget(bad: Any) -> None:
    with pytest.raises(ValueError, match="budget"):
        Packer({"budget": bad})


@pytest.mark.parametrize("bad", [-1, -0.5, True, "0.5"])
def test_pack_rejects_a_bad_redundancy_weight(bad: Any) -> None:
    with pytest.raises(ValueError, match="redundancy_weight"):
        Packer({"budget": 10, "redundancy_weight": bad})


def test_pack_rejects_non_mapping_value_signals() -> None:
    with pytest.raises(TypeError, match="value_signals"):
        Packer({"budget": 10, "value_signals": ["not", "a", "map"]})


def test_pack_rejects_a_non_numeric_signal_weight() -> None:
    with pytest.raises(ValueError, match="value_signals"):
        Packer({"budget": 10, "value_signals": {"c": "heavy"}})


@pytest.mark.parametrize("key", ["token_key", "cluster_key", "similarity_key"])
def test_pack_rejects_a_non_string_metadata_key(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        Packer({"budget": 10, key: 123})


def test_pack_rejects_a_non_numeric_token_key_value() -> None:
    candidates = [Candidate(id="a", score=1.0, text="x", metadata={"tok": "ten"})]
    with pytest.raises(ValueError, match="token_key"):
        Packer({"budget": 10, "token_key": "tok"}).process("q", candidates)


def test_pack_rejects_a_negative_token_key_value() -> None:
    candidates = [Candidate(id="a", score=1.0, text="x", metadata={"tok": -3})]
    with pytest.raises(ValueError, match="non-negative"):
        Packer({"budget": 10, "token_key": "tok"}).process("q", candidates)


def test_pack_falls_back_to_the_counter_when_a_candidate_lacks_the_token_key() -> None:
    # token_key set, but one candidate has no such metadata → count via the injected counter, no
    # crash. The counter returns a big cost so that candidate is excluded under a tight budget.
    class _Big:
        def count(self, text: str) -> int:
            return 1000

    candidates = [
        Candidate(id="a", score=1.0, text="x", metadata={"tok": 5}),
        Candidate(id="b", score=0.9, text="y"),  # no "tok" → counter → 1000
    ]
    packed = Packer({"budget": 10, "token_key": "tok"}, tokens=_Big()).process("q", candidates)
    assert _ids(packed) == {"a"}


def test_pack_rejects_a_bad_tokenizer_spec() -> None:
    with pytest.raises(ValueError, match="encoding"):
        Packer({"budget": 10, "tokenizer": {"encoding": ""}})
    with pytest.raises(TypeError, match="tokenizer"):
        Packer({"budget": 10, "tokenizer": "cl100k_base"})


# --- the HeuristicTokenCounter (pure-Python day-one default) ----------------------------------


def test_heuristic_counter_counts_words_and_punctuation_deterministically() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("hello, world!") == 4  # hello , world !
    assert counter.count("") == 0
    assert counter.count("one two three") == 3


def test_heuristic_counter_rejects_a_non_string() -> None:
    with pytest.raises(TypeError, match="string"):
        HeuristicTokenCounter().count(123)  # type: ignore[arg-type]


# --- the real TiktokenCounter against a fake tiktoken module ----------------------------------


class _FakeEncoding:
    """A faithful fake ``tiktoken.Encoding``: encode splits on whitespace (deterministic count)."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


class _RaisingEncoding:
    def encode(self, text: str) -> list[int]:
        raise RuntimeError("native tiktoken blew up")


def _install_fake_tiktoken(monkeypatch: pytest.MonkeyPatch, encoding: object) -> None:
    module = types.ModuleType("tiktoken")

    def get_encoding(name: str) -> object:
        if name == "unknown/enc":
            raise ValueError("no such encoding")
        return encoding

    module.get_encoding = get_encoding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", module)


def test_real_counter_counts_via_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tiktoken(monkeypatch, _FakeEncoding())
    assert TiktokenCounter("cl100k_base").count("one two three four") == 4


def test_packer_builds_the_real_counter_from_config_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    # Construction must NOT import tiktoken (only process does). Build with no fake installed…
    packer = Packer({"budget": 5, "tokenizer": {"encoding": "cl100k_base"}})
    # …then install the fake and run: the packer resolves the real counter on first process.
    _install_fake_tiktoken(monkeypatch, _FakeEncoding())
    candidates = [
        Candidate(id="a", score=1.0, text="one two"),  # 2 tokens
        Candidate(id="b", score=0.9, text="a b c d e f"),  # 6 tokens → excluded by budget 5
    ]
    packed = packer.process("q", candidates)
    assert _ids(packed) == {"a"}


def test_real_counter_unknown_encoding_is_a_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tiktoken(monkeypatch, _FakeEncoding())
    with pytest.raises(ValueError, match="could not load tiktoken encoding"):
        TiktokenCounter("unknown/enc").count("x")


def test_real_counter_missing_encoding_name_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="requires an 'encoding'"):
        TiktokenCounter("")


def test_real_counter_runtime_failure_maps_to_pack_tokenizer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tiktoken(monkeypatch, _RaisingEncoding())
    with pytest.raises(PackTokenizerError, match="failure during encode"):
        TiktokenCounter("cl100k_base").count("x")


def test_real_counter_rejects_a_non_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tiktoken(monkeypatch, _FakeEncoding())
    with pytest.raises(TypeError, match="string"):
        TiktokenCounter("cl100k_base").count(123)  # type: ignore[arg-type]


def test_real_counter_missing_dependency_raises_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ImportError, match=r"latence-retrieval\[pack\]"):
        TiktokenCounter("cl100k_base").count("x")


# --- real-tokenizer boundary (gated on the dep; skipped offline) ------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("tiktoken") is None,
    reason="tiktoken not installed (the exact-count path runs where the [pack] extra is present)",
)
def test_real_tiktoken_pack_smoke() -> None:  # pragma: no cover - runs only where tiktoken exists
    """With the real ``tiktoken`` dep, the packer returns a budget-respecting subset."""
    candidates = [
        Candidate(id="a", score=1.0, text="the quarterly revenue rose twelve percent"),
        Candidate(id="b", score=0.5, text="the mitochondria is the powerhouse of the cell"),
    ]
    packed = Packer(
        {"budget": 8, "tokenizer": {"encoding": "cl100k_base"}, "redundancy_weight": 0.0}
    ).process("what were the earnings", candidates)
    assert _ids(packed).issubset({"a", "b"})

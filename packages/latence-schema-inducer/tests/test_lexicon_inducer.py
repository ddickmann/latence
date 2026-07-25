"""T5: the inducer seam has a REAL non-frontier adapter — `label_inducer.lexicon`.

The claim under test is not "a class exists" but "the swap is real": the same Capability, the same
:class:`InducedLabels` contract, the same Quality-Report columns, the same fail-open posture — with
zero network, zero tokens, and byte-reproducible output. The tests therefore assert the OBSERVABLE
behaviour a stack sees when it swaps ``induce.provider``, not the tagger's internals.

The decisive one for T5 is :func:`test_no_openai_client_is_ever_constructed`: the fake ``openai``
module the LLM tests install raises on import here, so a Provider that reached for an endpoint would
FAIL rather than quietly work. That is what makes "zero LLM calls" a measured fact.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest
from latence_core.capability import LabelInducer
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    Provenance,
)
from latence_schema_inducer.lexicon import LexiconLabelInducer

if TYPE_CHECKING:
    from collections.abc import Iterator

_CANDIDATES = {
    "entity_types": {
        "tariff heading": ["heading", "HS code"],
        "commodity": ["commodity", "goods"],
        "vessel": ["vessel"],
    },
    "relation_types": ["classified under"],
    "pii_types": {"iban": ["iban"]},
}


def _chunk(doc_id: str = "doc-1", index: int = 0, content: str = "") -> ChunkRecord:
    prov = Provenance(
        source_uri="mem://t/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc_id}",
        char_start=0,
        char_end=len(content),
    )
    cls = Classification(language="en", category="memo", sensitivity="internal")
    return ChunkRecord(
        record_id=f"{doc_id}#chunk-{index}",
        document_record_id=doc_id,
        index=index,
        content=content,
        token_count=max(1, len(content) // 4),
        provenance=prov,
        classification=cls,
    )


def _provider(**overrides: object) -> LexiconLabelInducer:
    config: dict[str, object] = {"candidates": _CANDIDATES}
    config.update(overrides)
    return LexiconLabelInducer(config)


# -- the seam is real ---------------------------------------------------------


def test_it_satisfies_the_label_inducer_capability() -> None:
    assert isinstance(_provider(), LabelInducer)


def test_no_openai_client_is_ever_constructed() -> None:
    """REGRESSION (T5): the cheap path makes NO endpoint call — proved by making one impossible.

    An ``openai`` module that raises on attribute access is installed for the duration: any attempt
    to build a client (the only way this Provider could reach an endpoint) raises rather than
    silently succeeding against a real environment that happens to have the package.
    """

    class _Exploding(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            msg = f"the lexicon inducer must never touch the openai client (asked for {name!r})"
            raise AssertionError(msg)

    saved = sys.modules.get("openai")
    sys.modules["openai"] = _Exploding("openai")
    try:
        provider = _provider()
        out = list(provider.induce([_chunk(content="The vessel carried the goods.")]))
    finally:
        if saved is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = saved
    assert out[0].induced_labels is not None
    assert provider.induction_calls == 0
    assert provider.induction_prompt_tokens == 0
    assert provider.induction_completion_tokens == 0
    assert provider.induction_budget_exhausted is False


def test_the_profile_declares_a_deterministic_zero_cost_cpu_provider() -> None:
    profile = LexiconLabelInducer.profile()
    assert profile.compute == "cpu"
    assert profile.deterministic is True
    assert profile.cost_per_1k == 0.0
    assert profile.model_id is None  # bundles no weights


# -- what it induces ----------------------------------------------------------


def test_each_chunk_keeps_only_the_types_evidenced_in_it() -> None:
    """The whole point of the per-chunk pass: a chunk carries ITS types, not the corpus's."""
    chunks = [
        _chunk(index=0, content="The commodity was classified under heading 8471."),
        _chunk(index=1, content="Wire the balance to the IBAN on file."),
        _chunk(index=2, content="Nothing relevant here at all."),
    ]
    out = list(_provider().induce(chunks))
    first = out[0].induced_labels
    assert first is not None
    assert first.entity_types == ["tariff heading", "commodity"]  # declaration order
    assert first.relation_types == ["classified under"]
    assert first.pii_types == []
    second = out[1].induced_labels
    assert second is not None
    assert second.entity_types == []
    assert second.pii_types == ["iban"]
    # Fail-open, exactly like the LLM path: nothing matched ⇒ the extractor uses its config labels.
    assert out[2].induced_labels is None


def test_cue_matching_is_word_bounded_by_default() -> None:
    """REGRESSION: a short cue must not fire inside a longer word (``iban`` inside ``Ibanez``)."""
    out = list(_provider().induce([_chunk(content="He plays Ibanez guitars.")]))
    assert out[0].induced_labels is None
    # ...and an operator can opt out of boundaries for a compounding language (German
    # `Rechnungsnummernkreis` contains `Rechnungsnummer`), accepting the false positives.
    loose = list(_provider(whole_word=False).induce([_chunk(content="He plays Ibanez.")]))
    labels = loose[0].induced_labels
    assert labels is not None
    assert labels.pii_types == ["iban"]


def test_min_hits_requires_more_than_one_distinct_cue() -> None:
    strict = _provider(min_hits=2)
    weak = list(strict.induce([_chunk(content="The commodity arrived.")]))
    assert weak[0].induced_labels is None
    strong = list(strict.induce([_chunk(content="The commodity, i.e. the goods, arrived.")]))
    labels = strong[0].induced_labels
    assert labels is not None
    assert labels.entity_types == ["commodity"]


def test_repeating_one_cue_is_still_one_piece_of_evidence() -> None:
    """``min_hits`` counts DISTINCT cues, so a keyword-stuffed chunk cannot fake corroboration."""
    out = list(_provider(min_hits=2).induce([_chunk(content="commodity commodity commodity")]))
    assert out[0].induced_labels is None


def test_type_names_are_canonicalized_like_the_llm_path() -> None:
    """``Payment_Product`` and ``payment product`` must be ONE type across both Providers."""
    provider = LexiconLabelInducer({"candidates": {"entity_types": ["Payment_Product"]}})
    out = list(provider.induce([_chunk(content="The Payment Product was renewed.")]))
    labels = out[0].induced_labels
    assert labels is not None
    assert labels.entity_types == ["payment product"]


def test_output_is_byte_identical_across_two_runs() -> None:
    chunks = [
        _chunk(index=0, content="The commodity was classified under heading 8471."),
        _chunk(index=1, content="Send the IBAN."),
    ]
    first = [c.model_dump_json() for c in _provider().induce(chunks)]
    second = [c.model_dump_json() for c in _provider().induce(chunks)]
    assert first == second


def test_the_chunk_stream_is_threaded_through_one_to_one_in_order() -> None:
    chunks = [_chunk(index=i, content=f"chunk {i} commodity") for i in range(5)]
    out = list(_provider().induce(chunks))
    assert [c.record_id for c in out] == [c.record_id for c in chunks]


# -- the cost columns ---------------------------------------------------------


def test_duplicate_chunk_texts_are_tagged_once_and_reported_as_cache_hits() -> None:
    boiler = "Confidential. The goods remain the property of the seller."
    chunks = [
        _chunk(doc_id="d1", index=0, content=boiler),
        _chunk(doc_id="d2", index=0, content=boiler),
        _chunk(doc_id="d3", index=0, content="Send the IBAN."),
    ]
    provider = _provider()
    out = list(provider.induce(chunks))
    assert out[0].induced_labels == out[1].induced_labels
    assert provider.induction_documents == 3
    assert provider.induction_cache_hits == 1
    assert provider.induction_calls == 0


def test_unmatched_chunks_are_tallied_as_fallbacks() -> None:
    provider = _provider()
    list(provider.induce([_chunk(content="Nothing relevant."), _chunk(index=1, content="Also no.")]))
    assert provider.induction_fallbacks == 2
    assert provider.induction_empty_after_sanitization == 0  # no re-prompt concept here


# -- doc granularity + typed config errors ------------------------------------


def test_doc_granularity_broadcasts_one_documents_types_to_its_chunks() -> None:
    chunks = [
        _chunk(doc_id="d1", index=0, content="The commodity arrived."),
        _chunk(doc_id="d1", index=1, content="Send the IBAN."),
    ]
    out = list(_provider(granularity="doc").induce(chunks))
    a, b = out[0].induced_labels, out[1].induced_labels
    assert a is not None and a == b
    assert a.entity_types == ["commodity"]
    assert a.pii_types == ["iban"]


def test_corpus_granularity_is_refused_by_construction() -> None:
    """There is no corpus mode: a corpus-merged schema is what the redesign removed."""
    with pytest.raises(ValueError, match="corpus"):
        _provider(granularity="corpus")


@pytest.mark.parametrize(
    "candidates",
    [
        None,  # absent
        {},  # empty
        [],  # not a mapping
        {"entities": ["a"]},  # typo'd slot
        {"entity_types": 5},  # not a list/mapping
        {"entity_types": {"a": 5}},  # cue list is not a list
        {"entity_types": ["   "]},  # sanitizes away to nothing
    ],
)
def test_an_unusable_candidate_space_is_a_typed_config_error(candidates: object) -> None:
    """A config mistake fails LOUDLY at construction — never a silent label-free run."""
    with pytest.raises(ValueError, match="Lexicon"):
        LexiconLabelInducer({"candidates": candidates})


def test_the_label_ceiling_bounds_a_degenerate_candidate_space() -> None:
    many = [f"type {i:03d}" for i in range(500)]
    provider = LexiconLabelInducer({"candidates": {"entity_types": many}, "label_ceiling": 5})
    out = list(provider.induce([_chunk(content=" ".join(many))]))
    labels = out[0].induced_labels
    assert labels is not None
    assert labels.entity_types == many[:5]


def test_empty_and_too_small_chunks_flow_through_untouched() -> None:
    chunks: list[ChunkRecord] = [_chunk(index=0, content=" ")]
    out: Iterator[ChunkRecord] = _provider(min_chars=5).induce(chunks)
    assert [c.induced_labels for c in out] == [None]

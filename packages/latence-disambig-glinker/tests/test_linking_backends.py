"""KB builder + EntityLinker: corpus-internal KB, neural linking, direct-KB fallback, provisional.

Uses the faithful :class:`FakeGLinkerBackend` so the linking ALGORITHM (unique-text grouping,
neural link, direct-KB fallback, canonical clustering) is exercised offline + deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from conftest import FakeGLinkerBackend
from latence_disambig_glinker.backends import (
    GLinkerBackend,
    InferenceBackend,
    LinkedEntity,
    _collect_modules,
    _forward_names_input_ids,
    _strip_unsupported_forward_kwargs,
)
from latence_disambig_glinker.kb_builder import build_entity_kb, kb_to_glinker_format
from latence_disambig_glinker.linker import EntityLinker
from latence_disambig_glinker.models import MentionRecord


def _m(mid: str, doc: str, text: str, label: str = "organization", score: float = 0.9) -> MentionRecord:
    return MentionRecord(
        mention_id=mid, dataset_id="ds", document_id=doc, text=text, label=label, score=score
    )


def test_build_entity_kb_dominant_label_and_aliases() -> None:
    mentions = [
        _m("m1", "d1", "Acme", "organization"),
        _m("m2", "d2", "Acme", "organization"),
        _m("m3", "d1", "Acme Corp", "organization"),
        _m("m4", "d1", "Acme", "product", score=0.2),  # weak product vote loses to org
    ]
    kb = build_entity_kb(mentions, "ds", min_mentions=1, min_score=0.1)
    by_label = {e["label"]: e for e in kb}
    assert "Acme" in by_label
    assert by_label["Acme"]["entity_type"] == "organization"  # dominant weighted vote
    glinker_kb = kb_to_glinker_format(kb)
    assert all({"entity_id", "label", "entity_type", "aliases"} <= set(e) for e in glinker_kb)


def test_min_score_filters_weak_entries() -> None:
    mentions = [_m("m1", "d1", "Weak", score=0.1)]
    kb = build_entity_kb(mentions, "ds", min_mentions=1, min_score=0.5)
    assert kb == []


def test_linker_neural_link_clusters_acronym_with_expansion() -> None:
    """The fake biencoder context-links 'BAFA' and its full name to ONE KB entry -> one cluster."""
    FakeGLinkerBackend.link_map = {
        "bafa": ("grp_bafa", "organization"),
        "bundesamt für wirtschaft und ausfuhrkontrolle": ("grp_bafa", "organization"),
    }
    mentions = [
        _m("m1", "d1", "BAFA"),
        _m("m2", "d2", "Bundesamt für Wirtschaft und Ausfuhrkontrolle"),
    ]
    linker = EntityLinker(backend=FakeGLinkerBackend())
    updated, entities = linker.link_dataset(mentions, "ds")
    assert len({m.canonical_entity_id for m in updated}) == 1
    assert len(entities) == 1
    assert entities[0].evidence_count == 2


def test_linker_direct_kb_fallback_for_unmapped_surface() -> None:
    """An unmapped surface (neural returns None) resolves via the corpus-internal direct-KB match."""
    FakeGLinkerBackend.link_map = {}  # neural resolves nothing
    mentions = [_m("m1", "d1", "Acme"), _m("m2", "d2", "Acme")]
    linker = EntityLinker(backend=FakeGLinkerBackend())
    updated, entities = linker.link_dataset(mentions, "ds")
    # Both 'Acme' mentions fell back to the same KB entry (direct-KB), forming one cluster.
    assert len(entities) == 1
    assert all(m.canonical_entity_id for m in updated)


def test_linker_provisional_entity_for_truly_unresolved() -> None:
    """A singleton the direct-KB filters out (below min_score) still gets a provisional entity."""
    FakeGLinkerBackend.link_map = {}
    mentions = [_m("m1", "d1", "Ghost", score=0.05)]  # below default threshold 0.3 -> not in KB
    linker = EntityLinker(backend=FakeGLinkerBackend(), threshold=0.3)
    updated, entities = linker.link_dataset(mentions, "ds")
    assert len(entities) == 1  # a provisional entity, never dropped
    assert updated[0].canonical_entity_id is None  # unresolved keeps None on the mention


def test_linker_graceful_when_backend_raises() -> None:
    """A backend that raises on warmup/link degrades to direct-KB (never crashes) — the S8 robustness."""

    class _BrokenBackend(InferenceBackend):
        @property
        def name(self) -> str:
            return "broken"

        def warmup(self, entity_kb: list[dict[str, object]]) -> None:
            raise RuntimeError("no GPU / model load failed")

        def link_mentions(
            self,
            mentions: list[dict[str, object]],
            entity_kb: list[dict[str, object]],
            threshold: float = 0.3,
            batch_size: int = 32,
        ) -> list[LinkedEntity | None]:
            raise RuntimeError("should not be reached")

    mentions = [_m("m1", "d1", "Acme"), _m("m2", "d2", "Acme")]
    linker = EntityLinker(backend=_BrokenBackend())
    updated, entities = linker.link_dataset(mentions, "ds")  # must not raise
    assert len(entities) == 1  # direct-KB fallback still clusters
    assert all(m.canonical_entity_id for m in updated)


# --------------------------------------------------------------------------- #
# The REAL neural-glue result parser (_parse_glinker_result) against the faithful
# L0Output shape it consumes from glinker 0.1.x (C-F2: the parsing glue is tested,
# not just the fake backend that stands in for the model).
# --------------------------------------------------------------------------- #


@dataclass
class _FakeLinkedEntity:
    """Mirrors glinker's L0 ``linked_entity`` (entity_id / label / confidence)."""

    entity_id: str
    label: str
    confidence: float


@dataclass
class _FakeL0Entity:
    """Mirrors glinker's L0Entity: a ``mention_text`` + optional ``linked_entity`` / ``candidates``."""

    mention_text: str
    linked_entity: _FakeLinkedEntity | None = None
    candidates: list[Any] = field(default_factory=list)


@dataclass
class _FakeL0Result:
    """Mirrors L0Output: ``entities`` is list[list[L0Entity]] (outer per text, inner per mention)."""

    entities: list[list[_FakeL0Entity]]


def test_parse_glinker_result_maps_l0output_to_linked_entities() -> None:
    """The real parser lifts the glinker L0Output shape into LinkedEntity, enriched from the KB."""
    backend = GLinkerBackend(device="cpu")  # __init__ is lazy — pulls no heavy dep
    entity_kb = [
        {
            "entity_id": "grp_ibm",
            "label": "IBM",
            "description": "the computer company",
            "entity_type": "organization",
            "aliases": ["Big Blue"],
        }
    ]
    result = {
        "l0_result": _FakeL0Result(
            entities=[[_FakeL0Entity("IBM", _FakeLinkedEntity("grp_ibm", "IBM", 0.91))]]
        )
    }
    parsed = backend._parse_glinker_result(result, [{"text": "IBM"}], entity_kb, threshold=0.3)
    assert len(parsed) == 1
    le = parsed[0]
    assert le is not None
    assert le.entity_id == "grp_ibm"
    assert le.label == "IBM"
    assert le.description == "the computer company"  # enriched from the KB entry
    assert le.entity_type == "organization"
    assert le.aliases == ["Big Blue"]
    assert le.score == 0.91
    assert le.source == "glinker"


def test_parse_glinker_result_filters_below_threshold_and_handles_missing_l0() -> None:
    """Below-threshold links drop to None; a missing l0_result yields all-unresolved (never a crash)."""
    backend = GLinkerBackend(device="cpu")
    kb = [{"entity_id": "grp_x", "label": "X", "entity_type": "organization", "aliases": []}]

    low = {
        "l0_result": _FakeL0Result(
            entities=[[_FakeL0Entity("X", _FakeLinkedEntity("grp_x", "X", 0.10))]]
        )
    }
    assert backend._parse_glinker_result(low, [{"text": "X"}], kb, threshold=0.3) == [None]

    assert backend._parse_glinker_result({}, [{"text": "X"}], kb, threshold=0.3) == [None]


# --------------------------------------------------------------------------- #
# FIX-2: the token_lengths encoder-forward patch (the DEAD-neural-path guard).
#
# gliner 0.2.x injects token_lengths (+ word_lengths/packing_config/pair_attention_mask) and forwards
# them into the encoder, but a plain transformers 4.57 DeBERTa forward() REJECTS token_lengths -> every
# neural batch raised -> silent fallback to direct-KB (entities_linked: 0). These tests use a fake
# encoder whose forward signature rejects token_lengths EXACTLY like real transformers, so a green here
# is the test that would have caught the original dead neural path (not a fake-backend false green).
# --------------------------------------------------------------------------- #


class _PlainDebertaLike:
    """A plain-transformers-style encoder: forward declares input_ids but REJECTS token_lengths."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def forward(
        self, input_ids: object, attention_mask: object = None, token_type_ids: object = None
    ) -> dict[str, object]:
        # No **kwargs and no token_lengths — real DeBERTa raises TypeError on an unknown kwarg.
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask})
        return {"ok": True, "input_ids": input_ids}


def test_glinker_encoder_forward_patch_strips_token_lengths() -> None:
    """The provider's patch strips token_lengths so the plain-DeBERTa forward succeeds; unpatched raises."""
    enc = _PlainDebertaLike()

    # The gliner collator kwargs the real forward does NOT declare (dict-unpacked so this test also
    # type-checks — a direct kwarg would be a static call-arg error, which is the whole runtime bug).
    collator_kwargs: dict[str, object] = {
        "token_lengths": [3],
        "word_lengths": [3],
        "packing_config": None,
        "pair_attention_mask": None,
    }

    # UNPATCHED: forwarding token_lengths raises exactly like real transformers 4.57 (the dead path).
    with pytest.raises(TypeError):
        enc.forward(input_ids=[1, 2, 3], **{"token_lengths": [3]})

    # The encoder gate recognises this as a transformer-encoder forward (declares input_ids).
    assert _forward_names_input_ids(enc) is True

    installed = _strip_unsupported_forward_kwargs(enc)
    assert installed is True  # a wrapper WAS installed (signature lacks the collator kwargs)

    # PATCHED: every gliner collator kwarg is dropped before the plain forward — the call succeeds.
    out = enc.forward(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], **collator_kwargs)
    assert out == {"ok": True, "input_ids": [1, 2, 3]}
    # The real forward still received the arguments it DOES accept (nothing legitimate was stripped).
    assert enc.calls[-1] == {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    # Idempotent: a second patch is a no-op (already wrapped).
    assert _strip_unsupported_forward_kwargs(enc) is False


def test_encoder_forward_patch_is_forward_compatible_noop_when_signature_accepts_kwargs() -> None:
    """A future backbone whose forward accepts **kwargs (or names token_lengths) is NOT wrapped."""

    class _FutureEncoderVarKw:
        def forward(self, input_ids: object, **kwargs: object) -> object:
            return kwargs

    class _FutureEncoderNamed:
        def forward(
            self,
            input_ids: object,
            token_lengths: object = None,
            word_lengths: object = None,
            packing_config: object = None,
            pair_attention_mask: object = None,
        ) -> object:
            return token_lengths

    # **kwargs → nothing to strip; a forward that explicitly declares every collator kwarg → nothing
    # to strip either (the real forward handles them). Both are left un-wrapped (forward-compat).
    assert _strip_unsupported_forward_kwargs(_FutureEncoderVarKw()) is False
    assert _strip_unsupported_forward_kwargs(_FutureEncoderNamed()) is False


def test_encoder_gate_skips_non_encoder_forwards() -> None:
    """The input_ids gate keeps the patch surgical: a Linear/pooler-style forward is not a candidate."""

    class _Pooler:
        def forward(self, hidden_states: object) -> object:  # no input_ids
            return hidden_states

    assert _forward_names_input_ids(_Pooler()) is False


def test_collect_modules_finds_encoder_in_a_nested_executor_graph() -> None:
    """The bounded traversal locates encoder modules held deep in the executor's attribute graph.

    A fake 'module type' + a fake executor holding the encoder inside nested dict/list/object layers
    (the shape ProcessorFactory builds) — the walk must reach it without unbounded recursion.
    """

    class _Module:  # stands in for torch.nn.Module (the traversal takes the type as a param)
        pass

    encoder = _Module()

    class _L3Processor:
        def __init__(self) -> None:
            self.model = {"backbone": encoder}

    class _Executor:
        def __init__(self) -> None:
            self.nodes = [{"id": "l3", "processor": _L3Processor()}]
            self.self_ref = self  # a cycle: the visited-set must tolerate it

    found = _collect_modules(_Executor(), _Module)
    assert encoder in found

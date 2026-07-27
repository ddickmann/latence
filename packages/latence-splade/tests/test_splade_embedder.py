"""SpladeCompositionEmbedder: the ADR-0053 structured composition, exercised through the seam.

Two boundaries are tested here, both offline (the heavy transformers/torch stack is not installed —
ADR-0007/0016):

* the **composition Provider** against an injected faithful :class:`FakeSpladeModel` — every ADR-0053
  behaviour (model leg, confidence-gated entity injection, redaction-aware injection, metadata
  injection, inject vs weighted, ordering, empty) asserted on the emitted ``SparseVector``;
* the **real** :class:`~latence_splade.model.TransformersSpladeModel` against a fake ``transformers``
  module — the tokenize → MLM → SPLADE-pool path and its typed error mapping — the same honest fake
  boundary the gliner / embedder-st packages use.

A real-weights test is present and gated on the dep (skipped offline).
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import types
from typing import Any

import pytest
from latence_core.capability import SparseSignals, SparseVector
from latence_core.errors import ConfigError
from latence_core.providers.profile import UNVERIFIED_LICENSE, ProviderProfile, profile_of
from latence_splade import SpladeCompositionEmbedder
from latence_splade.composition import EntityTerm, SpladeInput
from latence_splade.model import TransformersSpladeModel

# --------------------------------------------------------------------------- #
# A faithful fake SpladeModel (the second adapter of the model seam).
# --------------------------------------------------------------------------- #

_FAKE_VOCAB = 64
_TOKEN = re.compile(r"[a-z0-9]+")


def _tid(token: str) -> int:
    """The stable vocab id a token folds to — byte-stable (sha256), NOT the salted builtin hash."""
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _FAKE_VOCAB


class FakeSpladeModel:
    """Deterministic, torch-free :class:`~latence_splade.model.SpladeModel` for offline tests.

    Faithful to the real contract it stands in for: term weights are per-text term frequencies over a
    fixed vocabulary (positive weights only, empty text → empty map), and ``term_ids`` maps a surface
    to the unique vocab ids of its subword-ish tokens (empty/punctuation-only → ``[]``, exactly as the
    real tokenizer yields no injectable ids). Both use the SAME token→id folding, so an entity surface
    injects onto the same id its token weighted in the model leg — the real union behaviour.
    """

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


def _embedder(config: dict[str, Any] | None = None) -> SpladeCompositionEmbedder:
    return SpladeCompositionEmbedder(config or {}, model=FakeSpladeModel())


# --------------------------------------------------------------------------- #
# encode_sparse — the SparseEmbedder Protocol method (#160 Export seam)
# --------------------------------------------------------------------------- #


def test_encode_sparse_emits_the_model_leg_over_text_in_input_order() -> None:
    embedder = _embedder()
    out = list(embedder.encode_sparse(["acme corp", "acme acme"]))
    assert len(out) == 2
    # Model leg only (no structured signals cross the text-only seam): term frequencies.
    assert _as_dict(out[0]) == {_tid("acme"): 1.0, _tid("corp"): 1.0}
    assert _as_dict(out[1]) == {_tid("acme"): 2.0}


def test_encode_sparse_yields_canonical_ascending_vectors() -> None:
    (vector,) = list(_embedder().encode_sparse(["gamma delta epsilon"]))
    assert list(vector.indices) == sorted(vector.indices)
    assert len(vector.indices) == len(vector.values)


def test_empty_text_yields_the_empty_sparse_vector() -> None:
    (vector,) = list(_embedder().encode_sparse([""]))
    assert vector == SparseVector()


def test_empty_batch_yields_nothing() -> None:
    assert list(_embedder().encode_sparse([])) == []


# --------------------------------------------------------------------------- #
# encode_composed — the full ADR-0053 composition
# --------------------------------------------------------------------------- #


def test_confidence_gated_entity_terms_are_injected_over_the_model_leg() -> None:
    embedder = _embedder({"confidence_threshold": 0.5, "entity_boost": 1.0})
    item = SpladeInput(
        text="acme corp invoice",
        entities=[EntityTerm("acme", 0.9), EntityTerm("corp", 0.2)],
    )
    (vector,) = list(embedder.encode_composed([item]))
    terms = _as_dict(vector)
    # 'acme' (0.9 >= 0.5) is injected: its model weight 1.0 + boost 1.0 = 2.0.
    assert terms[_tid("acme")] == 2.0
    # 'corp' (0.2 < 0.5) is gated out: it keeps only its model weight 1.0 (no boost).
    assert terms[_tid("corp")] == 1.0
    assert terms[_tid("invoice")] == 1.0


def test_redaction_default_drops_an_entity_whose_surface_did_not_survive_redaction() -> None:
    # 'john' was masked out of the corpus text; with redact=True (default) it must NOT be re-injected.
    item = SpladeInput(text="the invoice total", entities=[EntityTerm("john", 0.99)])
    (redacted,) = list(_embedder().encode_composed([item]))
    assert _tid("john") not in _as_dict(redacted)


def test_redact_false_injects_the_entity_even_if_absent_from_text() -> None:
    item = SpladeInput(text="the invoice total", entities=[EntityTerm("john", 0.99)])
    embedder = _embedder({"redact": False})
    (vector,) = list(embedder.encode_composed([item]))
    assert _as_dict(vector)[_tid("john")] == 1.0


def test_redaction_drops_a_masked_surface_that_is_only_a_substring_of_a_surviving_word() -> None:
    """REDACTION-BY-DEFAULT: the gate is a WORD-boundary check, not a substring `in` test.

    The person "Ann" was masked out of the corpus; the redacted text no longer holds the standalone
    token "ann" but still holds the unrelated word "annual". The old substring gate
    (``"ann" in "the annual revenue report"`` -> True) re-injected ``term_ids("Ann")`` -> the vocab
    id for "ann" (46), a token that appears NOWHERE in the redacted text and is distinct from
    "annual" (60) -- making the redacted name searchable. This fails on the old code and passes on
    the word-boundary fix.
    """
    text = "the annual revenue report"  # 'Ann' was redacted; only the substring survives in 'annual'
    item = SpladeInput(text=text, entities=[EntityTerm("Ann", 0.99)])
    (redacted,) = list(_embedder().encode_composed([item]))
    terms = _as_dict(redacted)
    # The masked surface's own token id is NOT re-leaked...
    assert _tid("ann") not in terms
    # ...while the surviving word 'annual' (a different id) IS present, proving the text was composed
    # (the absence is the redaction fix, not an empty vector).
    assert _tid("annual") in terms

    # Non-vacuous: the SAME input with redact=False DOES inject 'ann' -- the id resolves and the
    # ONLY thing suppressing it above is the redaction gate.
    (unredacted,) = list(_embedder({"redact": False}).encode_composed([item]))
    assert _tid("ann") in _as_dict(unredacted)


def test_redaction_keeps_a_multiword_surface_that_survives_as_an_adjacent_phrase() -> None:
    """A whole-phrase surface that survives redaction is still injected (no false-drop regression)."""
    text = "memo from john smith re: the annual budget"
    item = SpladeInput(text=text, entities=[EntityTerm("John Smith", 0.99)])
    (vector,) = list(_embedder({"entity_boost": 3.0}).encode_composed([item]))
    terms = _as_dict(vector)
    # 'john'/'smith' survive as an adjacent phrase -> injected (model leg 1.0 + boost 3.0).
    assert terms[_tid("john")] == 4.0
    assert terms[_tid("smith")] == 4.0


# --------------------------------------------------------------------------- #
# encode_sparse_composed — the index-time ComposingSparseEmbedder seam (ADR-0053, R1)
# --------------------------------------------------------------------------- #


def test_encode_sparse_composed_mirrors_the_query_side_composition() -> None:
    """R1: the index-time composing seam yields the SAME vector the query side composes.

    One composition, two clocks (ADR-0053): ``encode_sparse_composed`` (the seam Export drives at
    index time, over the core ``SparseSignals``) must produce the identical ``SparseVector`` the
    query-side ``encode_composed`` produces for the same ``(text, entities, metadata)`` — otherwise
    the corpus and query land in different sparse spaces (the audit defect).
    """
    embedder = _embedder(
        {
            "confidence_threshold": 0.5,
            "entity_boost": 5.0,
            "metadata_boost": 7.0,
            "metadata_fields": ["category"],
        }
    )
    text = "acme corp invoice"
    signals = SparseSignals(
        entities=(("acme", 0.9), ("corp", 0.2)),
        metadata={"category": "litigation"},
    )
    (index_vec,) = list(embedder.encode_sparse_composed([text], [signals]))

    item = SpladeInput(
        text=text,
        entities=[EntityTerm("acme", 0.9), EntityTerm("corp", 0.2)],
        metadata={"category": "litigation"},
    )
    (query_vec,) = list(embedder.encode_composed([item]))

    assert index_vec == query_vec  # symmetric index/query sparse space
    terms = _as_dict(index_vec)
    assert terms[_tid("acme")] == 6.0  # model leg 1.0 + entity_boost 5.0
    assert terms[_tid("corp")] == 1.0  # confidence 0.2 < 0.5 → gated, model leg only
    assert terms[_tid("litigation")] == 7.0  # selected metadata injected


def test_encode_sparse_composed_empty_signals_equal_the_model_leg() -> None:
    """A record with empty SparseSignals composes to exactly the text-only model leg."""
    embedder = _embedder({"entity_boost": 5.0, "metadata_fields": ["category"]})
    text = "acme corp"
    (composed,) = list(embedder.encode_sparse_composed([text], [SparseSignals()]))
    (text_only,) = list(embedder.encode_sparse([text]))
    assert composed == text_only


def test_encode_sparse_composed_rejects_nonparallel_signals() -> None:
    """The signals must be parallel to the texts (one SparseSignals per text)."""
    with pytest.raises(ConfigError, match="parallel"):
        list(_embedder().encode_sparse_composed(["a", "b"], [SparseSignals()]))


def test_selected_metadata_fields_are_injected() -> None:
    embedder = _embedder({"metadata_fields": ["classification"], "metadata_boost": 3.0})
    item = SpladeInput(
        text="quarterly report",
        metadata={"classification": "confidential", "provenance": "ignored"},
    )
    (vector,) = list(embedder.encode_composed([item]))
    terms = _as_dict(vector)
    assert terms[_tid("confidential")] == 3.0
    # A field NOT in metadata_fields is not injected.
    assert _tid("ignored") not in terms


def test_inject_mode_ignores_model_weight_but_weighted_mode_scales_the_model_leg() -> None:
    text = "acme acme corp"  # 'acme' has model weight 2.0
    item = SpladeInput(text=text)
    inject = _embedder({"mode": "inject", "model_weight": 0.5})
    (inj_vec,) = list(inject.encode_composed([item]))
    assert _as_dict(inj_vec)[_tid("acme")] == 2.0  # model_weight ignored in inject mode
    weighted = _embedder({"mode": "weighted", "model_weight": 0.5})
    (w_vec,) = list(weighted.encode_composed([item]))
    assert _as_dict(w_vec)[_tid("acme")] == 1.0  # 2.0 * 0.5


def test_encode_composed_preserves_input_order() -> None:
    items = [SpladeInput(text="alpha"), SpladeInput(text="beta beta")]
    out = list(_embedder().encode_composed(items))
    assert _as_dict(out[0]) == {_tid("alpha"): 1.0}
    assert _as_dict(out[1]) == {_tid("beta"): 2.0}


def test_vocab_size_comes_from_the_model() -> None:
    assert _embedder().vocab_size == _FAKE_VOCAB


# --------------------------------------------------------------------------- #
# config validation + license profile
# --------------------------------------------------------------------------- #


def test_bad_mode_is_rejected() -> None:
    with pytest.raises(ConfigError, match="'mode' must be one of"):
        SpladeCompositionEmbedder({"mode": "nonsense"}, model=FakeSpladeModel())


def test_bad_metadata_fields_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match="metadata_fields"):
        SpladeCompositionEmbedder({"metadata_fields": "classification"}, model=FakeSpladeModel())


def test_profile_declares_no_default_model_and_an_honest_unverified_license() -> None:
    prof = profile_of(SpladeCompositionEmbedder)
    assert isinstance(prof, ProviderProfile)
    # ADR-0053: no checkpoint is blessed until its license is verified — model_id is None and the
    # weights license is the honest UNVERIFIED sentinel (never a fabricated permissive claim).
    assert prof.model_id is None
    assert prof.license == UNVERIFIED_LICENSE
    assert prof.license_verified is False
    assert prof.license_weights is None
    assert prof.license_code == "Apache-2.0"  # the package + transformers library code, truthfully
    assert prof.deterministic is False
    assert prof.compute == "either"


# --------------------------------------------------------------------------- #
# TransformersSpladeModel — the real adapter, against a fake `transformers` module
# --------------------------------------------------------------------------- #

_TRANSFORMERS_VOCAB = 48


class _FakeArr:
    def __init__(self, data: Any, device: str = "cpu") -> None:
        self._data = data
        self.device = device

    def tolist(self) -> Any:
        return self._data

    def to(self, target: Any) -> _FakeArr:
        """A real torch tensor moves; the fake tracks where it was moved to (see _FakeEncoding)."""
        return _FakeArr(self._data, str(target))


class _FakeEncoding(dict[str, _FakeArr]):
    """A ``BatchEncoding`` stand-in.

    Faithful in the way the adapter now depends on: ``AutoTokenizer(..., return_tensors="pt")``
    returns a ``BatchEncoding``, which is a mapping (so ``**encoding`` spreads) AND carries a
    ``.to(device)`` that moves its tensors. The encoding always comes back on CPU no matter where
    the model was placed — which is exactly why the adapter has to move it.
    """

    def to(self, target: Any) -> _FakeEncoding:
        return _FakeEncoding({key: value.to(target) for key, value in self.items()})


def _fake_tok_ids(text: str) -> list[int]:
    return [_hash_tid(tok) for tok in _TOKEN.findall(text.lower())]


def _hash_tid(token: str) -> int:
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _TRANSFORMERS_VOCAB


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_id: str, **_kw: Any) -> _FakeTokenizer:
        if model_id == "bad/model":
            raise OSError("model 'bad/model' not found on the hub")
        return cls()

    def __call__(self, texts: Any, **_kw: Any) -> dict[str, _FakeArr]:
        if isinstance(texts, str):
            ids = _fake_tok_ids(texts)
            return {"input_ids": _FakeArr(ids), "attention_mask": _FakeArr([1] * len(ids))}
        seqs = [_fake_tok_ids(t) for t in texts]
        width = max((len(s) for s in seqs), default=0)
        padded = [s + [0] * (width - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return _FakeEncoding({"input_ids": _FakeArr(padded), "attention_mask": _FakeArr(mask)})


class _FakeConfig:
    vocab_size = _TRANSFORMERS_VOCAB


class _FakeOutput:
    def __init__(self, logits: _FakeArr) -> None:
        self.logits = logits


class _FakeMaskedLM:
    """A faithful stand-in for a transformers ``AutoModelForMaskedLM``.

    Faithful in the two ways the load choreography now depends on: a real ``nn.Module`` HAS a
    ``.to(device)`` and reports a ``.device``, so the fake does too — otherwise the seam's "prove it
    landed or raise" step would fail here against a fake that is simply less capable than the real
    dependency, which is a fake bug, not a seam bug. ``device`` starts at ``"cpu"`` exactly like a
    freshly ``from_pretrained``-ed model (which is the whole reason placement has to happen at all).
    """

    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kw: Any) -> _FakeMaskedLM:
        return cls()

    def to(self, device: str) -> _FakeMaskedLM:
        self.device = device
        return self

    def eval(self) -> _FakeMaskedLM:
        return self

    def __call__(self, input_ids: _FakeArr, attention_mask: _FakeArr | None = None, **_kw: Any) -> _FakeOutput:  # noqa: E501
        logits: list[list[list[float]]] = []
        for seq in input_ids.tolist():
            seq_logits: list[list[float]] = []
            for tid in seq:
                row = [0.0] * _TRANSFORMERS_VOCAB
                row[tid] = 3.0  # a positive logit at the token's own id → log1p after relu
                seq_logits.append(row)
            logits.append(seq_logits)
        return _FakeOutput(_FakeArr(logits))


@pytest.fixture()
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    module.AutoModelForMaskedLM = _FakeMaskedLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_transformers_adapter_reduces_logits_to_positive_splade_weights(
    fake_transformers: None,
) -> None:
    import math

    model = TransformersSpladeModel("stub/splade")
    assert model.vocab_size == _TRANSFORMERS_VOCAB
    (weights,) = model.term_weights(["acme corp"])
    # Each token's id carries a positive weight = log1p(3.0); no other ids are present (relu floor).
    expected = pytest.approx(math.log1p(3.0))
    assert weights[_hash_tid("acme")] == expected
    assert weights[_hash_tid("corp")] == expected
    assert all(v > 0.0 for v in weights.values())


def test_transformers_adapter_term_ids_are_unique_in_vocab(fake_transformers: None) -> None:
    model = TransformersSpladeModel("stub/splade")
    ids = model.term_ids("acme acme corp")  # 'acme' repeated must not duplicate
    assert ids == [_hash_tid("acme"), _hash_tid("corp")]
    assert all(0 <= i < _TRANSFORMERS_VOCAB for i in ids)


def test_transformers_adapter_empty_text_returns_empty() -> None:
    # No model touch needed for an empty batch (and no transformers import): graceful empty.
    assert TransformersSpladeModel("stub/splade").term_weights([]) == []


def test_missing_model_id_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="requires a 'model'"):
        TransformersSpladeModel("")


def test_bad_model_id_maps_to_config_error(fake_transformers: None) -> None:
    model = TransformersSpladeModel("bad/model")
    with pytest.raises(ConfigError, match="could not load model"):
        model.term_weights(["anything"])


def test_missing_transformers_dependency_raises_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    model = TransformersSpladeModel("stub/splade")
    with pytest.raises(ImportError, match="pip install latence-splade"):
        model.term_weights(["anything"])


def test_provider_default_path_builds_the_transformers_adapter_and_composes(
    fake_transformers: None,
) -> None:
    # No injected model: the Provider builds the real adapter from config and runs end to end.
    embedder = SpladeCompositionEmbedder({"model": "stub/splade"})
    (vector,) = list(embedder.encode_sparse(["acme corp"]))
    terms = _as_dict(vector)
    assert _hash_tid("acme") in terms
    assert _hash_tid("corp") in terms


def test_provider_without_a_model_config_fails_loudly_on_use(fake_transformers: None) -> None:
    embedder = SpladeCompositionEmbedder({})  # no injected model, no 'model' id
    with pytest.raises(ConfigError, match="requires a 'model'"):
        list(embedder.encode_sparse(["x"]))


# --------------------------------------------------------------------------- #
# Real-weights boundary (gated on the dep; skipped offline)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed (real-weights path runs where a cleared model exists)",
)
def test_real_weights_encode_smoke() -> None:  # pragma: no cover - runs only on a rig
    """With the real dep + a *cleared* checkpoint present, the adapter returns positive term maps.

    Skipped offline. A real run pins a license-cleared multilingual SPLADE checkpoint via the
    ``LATENCE_SPLADE_MODEL`` env (never a hardcoded default — ADR-0053) and asserts the composition
    emits a non-empty, positive, canonical sparse vector.
    """
    import os

    model_id = os.environ.get("LATENCE_SPLADE_MODEL")
    if not model_id:
        pytest.skip("set LATENCE_SPLADE_MODEL to a license-cleared checkpoint to run this")
    embedder = SpladeCompositionEmbedder({"model": model_id})
    (vector,) = list(embedder.encode_sparse(["a multilingual sentence"]))
    assert list(vector.indices) == sorted(vector.indices)
    assert all(v > 0.0 for v in vector.values)

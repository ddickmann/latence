"""IncrementalLinker: link a delta batch against an existing KB (ported from dataset-intelligence)."""

from __future__ import annotations

from conftest import FakeGLinkerBackend
from latence_disambig_glinker.incremental_linker import IncrementalLinker
from latence_disambig_glinker.models import CanonicalEntityRecord, MentionRecord


def _existing(eid: str, name: str) -> CanonicalEntityRecord:
    return CanonicalEntityRecord(
        entity_id=eid, dataset_id="ds", canonical_name=name, label="organization",
        mention_ids=[f"{eid}-m0"], document_ids=["d0"], evidence_count=1, confidence=0.9,
    )


def _m(mid: str, doc: str, text: str) -> MentionRecord:
    return MentionRecord(
        mention_id=mid, dataset_id="ds", document_id=doc, text=text, label="organization", score=0.9
    )


def test_empty_delta_returns_existing() -> None:
    existing = [_existing("e1", "Acme")]
    linker = IncrementalLinker(existing, backend=FakeGLinkerBackend())
    linked, new, merged = linker.link_delta([], "ds")
    assert linked == [] and new == [] and merged == existing


def test_delta_produces_new_canonicals() -> None:
    FakeGLinkerBackend.link_map = {"globex": ("grp_globex", "organization")}
    existing = [_existing("e1", "Acme")]
    linker = IncrementalLinker(existing, backend=FakeGLinkerBackend())
    _linked, new, merged = linker.link_delta([_m("m1", "d1", "Globex")], "ds")
    new_names = {e.canonical_name for e in new}
    assert "Globex" in new_names
    # Merged = existing + the new canonicals (existing 'Acme' retained).
    assert any(e.canonical_name == "Acme" for e in merged)
    assert any(e.canonical_name == "Globex" for e in merged)

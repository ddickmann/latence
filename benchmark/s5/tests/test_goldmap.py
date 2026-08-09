# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The gold-resolution rules, driven by a five-chunk hand-written index (no corpus, no model)."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.s5.goldmap import (
    TRUST_THRESHOLD,
    GoldQuery,
    GoldRef,
    RefKind,
    report,
    resolve,
    resolve_ref,
)


@dataclass
class TinyIndex:
    """A hand-written :class:`~benchmark.s5.goldmap.ChunkIndex` — the whole point of the seam."""

    files: dict[str, list[str]] = field(default_factory=dict)
    pages: dict[tuple[str, int], list[str]] = field(default_factory=dict)
    texts: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def chunks_in_file(self, file_name: str) -> list[str]:
        return list(self.files.get(file_name, []))

    def chunks_on_page(self, file_name: str, page: int) -> list[str]:
        return list(self.pages.get((file_name, page), []))

    def chunks_containing(self, file_name: str, text: str) -> list[str]:
        return list(self.texts.get(file_name, {}).get(" ".join(text.lower().split()), []))


def _index() -> TinyIndex:
    return TinyIndex(
        files={"a.md": ["a#0", "a#1"], "b.md": ["b#0"]},
        pages={("a.md", 1): ["a#0"], ("a.md", 2): ["a#1"]},
        texts={"b.md": {"the fact": ["b#0"]}},
    )


def test_file_ref_resolves_to_every_chunk_of_the_document() -> None:
    ref = GoldRef(kind=RefKind.FILE, file_name="a.md")
    assert resolve_ref(ref, _index()) == ("a#0", "a#1")


def test_page_ref_resolves_only_that_page() -> None:
    ref = GoldRef(kind=RefKind.PAGE, file_name="a.md", page=2)
    assert resolve_ref(ref, _index()) == ("a#1",)


def test_text_ref_is_whitespace_normalised() -> None:
    ref = GoldRef(kind=RefKind.TEXT, file_name="b.md", text="  The\n FACT ")
    assert resolve_ref(ref, _index()) == ("b#0",)


def test_unresolvable_ref_yields_nothing_and_is_counted() -> None:
    query = GoldQuery(
        query_id="q1",
        question="?",
        refs=(
            GoldRef(kind=RefKind.FILE, file_name="a.md"),
            GoldRef(kind=RefKind.FILE, file_name="ghost.md"),
        ),
    )
    resolved = resolve(query, _index())
    assert resolved.chunk_ids == frozenset({"a#0", "a#1"})
    assert (resolved.resolved_refs, resolved.total_refs) == (1, 2)
    assert resolved.is_usable


def test_page_ref_without_a_page_number_does_not_widen_to_the_file() -> None:
    """The failure mode this module exists to prevent: a page miss becoming document-level gold."""
    ref = GoldRef(kind=RefKind.PAGE, file_name="a.md", page=None)
    assert resolve_ref(ref, _index()) == ()


def test_chain_slots_group_by_slot_and_need_two_occupied_slots() -> None:
    two_slots = GoldQuery(
        query_id="q2",
        question="?",
        refs=(
            GoldRef(kind=RefKind.FILE, file_name="a.md", slot=0),
            GoldRef(kind=RefKind.FILE, file_name="b.md", slot=1),
        ),
    )
    resolved = resolve(two_slots, _index())
    assert resolved.chain == (frozenset({"a#0", "a#1"}), frozenset({"b#0"}))

    one_slot = GoldQuery(
        query_id="q3",
        question="?",
        refs=(GoldRef(kind=RefKind.FILE, file_name="a.md", slot=0),),
    )
    assert resolve(one_slot, _index()).chain == ()


def test_a_slot_that_did_not_resolve_is_not_a_phantom_chain_link() -> None:
    query = GoldQuery(
        query_id="q4",
        question="?",
        refs=(
            GoldRef(kind=RefKind.FILE, file_name="a.md", slot=0),
            GoldRef(kind=RefKind.FILE, file_name="ghost.md", slot=1),
        ),
    )
    assert resolve(query, _index()).chain == ()


def test_report_flags_a_dataset_below_the_trust_threshold() -> None:
    index = _index()
    good = [
        resolve(
            GoldQuery(query_id=f"g{i}", question="?", refs=(GoldRef(RefKind.FILE, "a.md"),)),
            index,
        )
        for i in range(9)
    ]
    bad = resolve(
        GoldQuery(query_id="b", question="?", refs=(GoldRef(RefKind.FILE, "ghost.md"),)), index
    )
    assert report("ds", good).trustworthy
    mixed = report("ds", [*good, bad])
    assert mixed.rate == 0.9
    assert mixed.trustworthy is (TRUST_THRESHOLD <= 0.9)
    assert report("ds", [*good[:5], bad, bad, bad]).trustworthy is False

# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The gold-resolution rules, driven by a hand-written index (no corpus, no model)."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.s5.goldmap import (
    TRUST_THRESHOLD,
    ExclusionReason,
    GoldQuery,
    GoldRef,
    Granularity,
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
    paginated: frozenset[str] = frozenset()

    def chunks_in_file(self, file_name: str) -> list[str]:
        return list(self.files.get(file_name, []))

    def chunks_on_page(self, file_name: str, page: int) -> list[str]:
        return list(self.pages.get((file_name, page), []))

    def has_page_map(self, file_name: str) -> bool:
        return file_name in self.paginated


def _index() -> TinyIndex:
    return TinyIndex(
        files={"a.md": ["a#0", "a#1"], "b.md": ["b#0"]},
        pages={("a.md", 1): ["a#0"], ("a.md", 2): ["a#1"]},
        paginated=frozenset({"a.md"}),
    )


def test_file_ref_resolves_to_every_chunk_of_the_document() -> None:
    ref = GoldRef(kind=RefKind.FILE, file_name="a.md")
    outcome = resolve_ref(ref, _index())
    assert outcome.chunk_ids == ("a#0", "a#1")
    assert outcome.granularity is Granularity.DOCUMENT


def test_page_ref_resolves_only_that_page_and_says_so() -> None:
    outcome = resolve_ref(GoldRef(kind=RefKind.PAGE, file_name="a.md", page=2), _index())
    assert outcome.chunk_ids == ("a#1",)
    assert outcome.granularity is Granularity.PAGE


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
    assert resolve_ref(ref, _index()).chunk_ids == ()


def test_page_ref_falls_back_to_the_document_only_when_no_page_map_exists() -> None:
    """UDA's 74 unreconstructable reports / OHR-Bench's — coarser gold, labelled coarser."""
    index = _index()
    unpaginated = GoldRef(
        kind=RefKind.PAGE, file_name="b.md", page=7, allow_document_fallback=True
    )
    outcome = resolve_ref(unpaginated, index)
    assert outcome.chunk_ids == ("b#0",)
    assert outcome.granularity is Granularity.DOCUMENT


def test_a_wrong_page_on_a_paginated_document_stays_a_miss() -> None:
    """The fallback is for *unavailable* page spans, never for a page reference that is wrong."""
    ref = GoldRef(kind=RefKind.PAGE, file_name="a.md", page=99, allow_document_fallback=True)
    outcome = resolve_ref(ref, _index())
    assert outcome.chunk_ids == ()
    assert outcome.granularity is None


def test_resolve_counts_the_granularity_mix_per_query() -> None:
    query = GoldQuery(
        query_id="q5",
        question="?",
        refs=(
            GoldRef(kind=RefKind.PAGE, file_name="a.md", page=1, slot=0),
            GoldRef(
                kind=RefKind.PAGE,
                file_name="b.md",
                page=1,
                slot=1,
                allow_document_fallback=True,
            ),
        ),
    )
    resolved = resolve(query, _index())
    assert (resolved.page_refs, resolved.document_refs) == (1, 1)


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


def test_an_excluded_query_leaves_the_denominator_carrying_its_reason() -> None:
    """A domain this corpus never acquired is a coverage fact, not a retrieval miss."""
    query = GoldQuery(
        query_id="notes-1",
        question="?",
        refs=(GoldRef(kind=RefKind.PAGE, file_name="notes__x.md", page=1),),
        exclusion=ExclusionReason.CORPUS_MISSING_DOMAIN,
    )
    resolved = resolve(query, _index())
    assert resolved.exclusion is ExclusionReason.CORPUS_MISSING_DOMAIN
    assert (resolved.total_refs, resolved.resolved_refs, resolved.excluded_refs) == (0, 0, 1)
    assert not resolved.is_usable

    index = _index()
    good = [
        resolve(
            GoldQuery(query_id=f"g{i}", question="?", refs=(GoldRef(RefKind.FILE, "a.md"),)),
            index,
        )
        for i in range(4)
    ]
    aggregate = report("ohr_bench", [*good, resolved])
    assert aggregate.rate == 1.0  # the excluded query did NOT drag the rate down
    assert aggregate.queries_excluded == 1
    assert aggregate.refs_excluded == 1
    assert aggregate.exclusions == {"corpus-missing-domain": 1}
    assert aggregate.to_dict()["exclusions"] == {"corpus-missing-domain": 1}


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


def test_report_records_the_granularity_mix_and_page_share() -> None:
    index = _index()
    page = resolve(
        GoldQuery(
            query_id="p", question="?", refs=(GoldRef(RefKind.PAGE, "a.md", page=1),)
        ),
        index,
    )
    document = resolve(
        GoldQuery(query_id="d", question="?", refs=(GoldRef(RefKind.FILE, "b.md"),)), index
    )
    aggregate = report("ds", [page, page, document])
    assert (aggregate.refs_page, aggregate.refs_document) == (2, 1)
    assert aggregate.page_share == 2 / 3
    assert aggregate.to_dict()["granularity"] == {"page": 2, "document": 1}

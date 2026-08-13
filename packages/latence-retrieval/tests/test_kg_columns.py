# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The KG-column read contract (:mod:`latence_retrieval.kg_columns`), tested at its boundary.

This module is the ONE implementation of the graceful-absent / loud-malformed column reader and
the per-entity-list → peak reduction that ``expand`` / ``multihop`` / ``rescue`` / ``pack``
consume, so its contract is tested directly — and the consumer-facing guarantees (each call
site's message prefix, the identical peak on both centrality consumers) are asserted here rather
than re-proved per consumer.
"""

from __future__ import annotations

from typing import Any

import pytest
from latence_retrieval.contracts import Candidate
from latence_retrieval.kg_columns import (
    DEFAULT_KEYS,
    KgColumnKeys,
    KgColumns,
    parse_centrality_list,
    parse_id_list,
    peak,
)


def _cand(cid: str = "c1", **meta: Any) -> Candidate:
    return Candidate(id=cid, score=1.0, text=f"text {cid}", metadata=dict(meta))


# -- parse_id_list ------------------------------------------------------------------------------


class TestParseIdList:
    def test_an_absent_column_is_gracefully_nothing(self) -> None:
        assert parse_id_list(_cand(), "kg_node_ids", context="expand") == ()

    def test_a_present_column_is_returned_verbatim_order_and_duplicates(self) -> None:
        candidate = _cand(kg_node_ids=["n2", "n1", "n2"])
        assert parse_id_list(candidate, "kg_node_ids", context="expand") == ("n2", "n1", "n2")

    def test_a_tuple_column_is_accepted_like_a_list(self) -> None:
        candidate = _cand(kg_node_ids=("n1",))
        assert parse_id_list(candidate, "kg_node_ids", context="expand") == ("n1",)

    def test_a_non_list_value_is_a_loud_type_error(self) -> None:
        candidate = _cand(kg_node_ids={"n1": 1})
        with pytest.raises(TypeError, match="must be a list of node-id strings"):
            parse_id_list(candidate, "kg_node_ids", context="expand")

    def test_a_string_value_is_rejected_not_iterated_characterwise(self) -> None:
        candidate = _cand(kg_node_ids="n1")
        with pytest.raises(TypeError, match="must be a list of node-id strings"):
            parse_id_list(candidate, "kg_node_ids", context="expand")

    def test_a_non_string_element_is_a_loud_value_error(self) -> None:
        candidate = _cand(kg_node_ids=["n1", 7])
        with pytest.raises(ValueError, match="must contain non-empty strings"):
            parse_id_list(candidate, "kg_node_ids", context="expand")

    def test_an_empty_string_element_is_rejected(self) -> None:
        candidate = _cand(kg_node_ids=[""])
        with pytest.raises(ValueError, match="must contain non-empty strings"):
            parse_id_list(candidate, "kg_node_ids", context="expand")

    @pytest.mark.parametrize("context", ["expand", "multihop_expand", "graph-rescue"])
    def test_the_message_carries_each_call_sites_own_prefix(self, context: str) -> None:
        # The prefixes the three consumers pinned when each owned its own copy of this reader.
        candidate = _cand(kg_node_ids=3)
        with pytest.raises(TypeError) as excinfo:
            parse_id_list(candidate, "kg_node_ids", context=context)
        assert str(excinfo.value) == (
            f"{context} metadata 'kg_node_ids' must be a list of node-id strings, "
            "got 3 for candidate 'c1'."
        )


# -- parse_centrality_list ----------------------------------------------------------------------


class TestParseCentralityList:
    def test_an_absent_column_is_gracefully_nothing(self) -> None:
        assert parse_centrality_list(_cand(), "kg_node_centrality", context="expand") == ()

    def test_values_come_back_as_floats_in_order(self) -> None:
        candidate = _cand(kg_node_centrality=[1, 0.25])
        parsed = parse_centrality_list(candidate, "kg_node_centrality", context="expand")
        assert parsed == (1.0, 0.25)
        assert all(isinstance(value, float) for value in parsed)

    def test_a_non_list_value_is_a_loud_type_error(self) -> None:
        candidate = _cand(kg_node_centrality=0.5)
        with pytest.raises(TypeError, match="must be a list of centrality floats"):
            parse_centrality_list(candidate, "kg_node_centrality", context="expand")

    def test_a_bool_element_is_not_a_centrality(self) -> None:
        candidate = _cand(kg_node_centrality=[True])
        with pytest.raises(ValueError, match="must contain numbers"):
            parse_centrality_list(candidate, "kg_node_centrality", context="expand")

    def test_a_non_finite_element_is_rejected(self) -> None:
        candidate = _cand(kg_node_centrality=[float("nan")])
        with pytest.raises(ValueError, match="must be finite"):
            parse_centrality_list(candidate, "kg_node_centrality", context="expand")


# -- peak (the one lockstep reduction) ----------------------------------------------------------


class TestPeak:
    def test_the_peak_is_the_max(self) -> None:
        assert peak((0.2, 0.9, 0.5)) == 0.9

    def test_no_entities_is_no_signal(self) -> None:
        assert peak(()) == 0.0

    def test_the_true_max_is_never_clamped(self) -> None:
        assert peak((3.5,)) == 3.5


# -- KgColumns (parse-once per candidate) -------------------------------------------------------


class TestKgColumns:
    def test_all_four_columns_parse_into_the_value_view(self) -> None:
        candidate = _cand(
            kg_node_ids=["n1", "n2"],
            neighbor_node_ids=["n3"],
            kg_node_community=["c1"],
            kg_node_centrality=[0.1, 0.8],
        )
        columns = KgColumns.parse(candidate, context="expand")
        assert columns == KgColumns(
            nodes=frozenset({"n1", "n2"}),
            reach=frozenset({"n3"}),
            communities=frozenset({"c1"}),
            peak_centrality=0.8,
        )

    def test_a_candidate_with_no_kg_columns_parses_to_the_empty_view(self) -> None:
        assert KgColumns.parse(_cand(), context="expand") == KgColumns(
            nodes=frozenset(), reach=frozenset(), communities=frozenset(), peak_centrality=0.0
        )

    def test_custom_keys_redirect_every_read(self) -> None:
        keys = KgColumnKeys(
            node_ids="my_nodes",
            neighbor_ids="my_reach",
            community="my_comm",
            centrality="my_cent",
        )
        candidate = _cand(
            my_nodes=["a"], my_reach=["b"], my_comm=["c"], my_cent=[0.4],
            kg_node_ids=["ignored"],
        )
        columns = KgColumns.parse(candidate, keys=keys, context="expand")
        assert columns.nodes == frozenset({"a"})
        assert columns.reach == frozenset({"b"})
        assert columns.communities == frozenset({"c"})
        assert columns.peak_centrality == 0.4

    def test_the_default_keys_are_the_emitted_column_names(self) -> None:
        expected = KgColumnKeys(
            node_ids="kg_node_ids",
            neighbor_ids="neighbor_node_ids",
            community="kg_node_community",
            centrality="kg_node_centrality",
        )
        assert expected == DEFAULT_KEYS

    def test_a_malformed_column_stays_loud_through_the_value_view(self) -> None:
        with pytest.raises(TypeError, match="expand metadata 'neighbor_node_ids'"):
            KgColumns.parse(
                _cand(kg_node_ids=["n1"], neighbor_node_ids="n2"), context="expand"
            )

# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The denormalised KG-column read contract — one home (:class:`KgColumns`).

The pipeline projects each chunk's 1-hop KG neighbourhood *onto the chunk itself* (ADR-0051):
its own entity node ids (``kg_node_ids``), the node ids its triples reach
(``neighbor_node_ids``), and the optional per-entity graph features (``kg_node_community``,
``kg_node_centrality``). Those columns travel as :class:`~latence_retrieval.contracts.Candidate`
``metadata``, and every query-time consumer reads them under the same discipline:

* **graceful when absent** — a candidate lacking a column simply contributes nothing, so every
  transform runs on candidates that carry no KG columns at all;
* **loud when malformed** — a *present* column must be well-formed (a list of non-empty node-id
  strings; a list of finite centrality floats) or the read raises :class:`TypeError` /
  :class:`ValueError` at the boundary rather than silently scoring a candidate wrong (the
  anti-false-green contract).

Before this module, that reader existed three times (``expand``, ``multihop``, ``rescue``) as
near-identical privates differing only in their error-message prefix, and the per-entity-list →
**peak** reduction the centrality column takes existed twice (``expand``'s hub bonus and
``pack``'s ``value_signals``) held identical by a prose comment. This module is the single
implementation; the consumers pass their ``context`` prefix so every pinned error message is
unchanged.

:func:`parse_id_list` / :func:`parse_centrality_list` are the column readers; :func:`peak` is the
one per-entity-list reduction; :class:`KgColumns` is the parse-once-per-candidate value view
(``nodes`` / ``reach`` / ``communities`` / ``peak_centrality``) the 1-hop
:class:`~latence_retrieval.expand.Expander` scores from, with :class:`KgColumnKeys` naming which
metadata keys carry the columns (defaulting to the emitted names).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_retrieval.contracts import Candidate

#: The emitted column names (``export._context_columns_for_row``): the chunk's own entity node
#: ids, the node ids its triples reach, and the optional per-entity graph features.
KG_NODE_IDS_KEY = "kg_node_ids"
NEIGHBOR_NODE_IDS_KEY = "neighbor_node_ids"
KG_NODE_COMMUNITY_KEY = "kg_node_community"
KG_NODE_CENTRALITY_KEY = "kg_node_centrality"


def parse_id_list(candidate: Candidate, key: str, *, context: str) -> tuple[str, ...]:
    """Read ``metadata[key]`` as a list of non-empty id strings, verbatim order; ``()`` if absent.

    Absent → no signal (graceful). Present-but-malformed → loud: the value must be a list/tuple
    of non-empty strings (what the denormalised KG columns are), so a dict, a scalar, or a
    non-string element is a boundary error, not a silently-dropped signal (anti-false-green).
    ``context`` prefixes the error message with the consumer's name, so each call site's pinned
    diagnostics are exactly what they were when it owned its own copy of this reader.
    """
    if key not in candidate.metadata:
        return ()
    raw = candidate.metadata[key]
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        msg = (
            f"{context} metadata {key!r} must be a list of node-id strings, got {raw!r} for "
            f"candidate {candidate.id!r}."
        )
        raise TypeError(msg)
    for element in raw:
        if not isinstance(element, str) or not element:
            msg = (
                f"{context} metadata {key!r} must contain non-empty strings, got {element!r} for "
                f"candidate {candidate.id!r}."
            )
            raise ValueError(msg)
    return tuple(raw)


def parse_centrality_list(candidate: Candidate, key: str, *, context: str) -> tuple[float, ...]:
    """Read ``metadata[key]`` as a list of finite floats (per-entity values); ``()`` if absent.

    The same graceful-absent / loud-malformed discipline as :func:`parse_id_list`, for the
    numeric graph-feature column: a present value must be a list/tuple of finite numbers
    (``bool`` is rejected — it is not a centrality), aligned per-entity with ``kg_node_ids``
    upstream.
    """
    if key not in candidate.metadata:
        return ()
    raw = candidate.metadata[key]
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        msg = (
            f"{context} metadata {key!r} must be a list of centrality floats, got {raw!r} for "
            f"candidate {candidate.id!r}."
        )
        raise TypeError(msg)
    values: list[float] = []
    for element in raw:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            msg = (
                f"{context} metadata {key!r} must contain numbers, got {element!r} for candidate "
                f"{candidate.id!r}."
            )
            raise ValueError(msg)
        if not math.isfinite(element):
            msg = (
                f"{context} metadata {key!r} must be finite, got {element!r} for candidate "
                f"{candidate.id!r}."
            )
            raise ValueError(msg)
        values.append(float(element))
    return tuple(values)


def peak(values: Sequence[float]) -> float:
    """The per-entity-list → **peak** reduction: ``max(values)``, or ``0.0`` for no entities.

    THE one way a per-entity graph-feature list becomes a scalar. Both query-time consumers of
    ``kg_node_centrality`` — the :class:`~latence_retrieval.expand.Expander`'s hub bonus and the
    :class:`~latence_retrieval.pack.Packer`'s ``value_signals`` — call this, so "both reduce the
    list identically" is code, not a prose invariant kept true by review.
    """
    return max(values) if values else 0.0


@dataclass(frozen=True)
class KgColumnKeys:
    """Which metadata keys carry the four KG columns (defaults: the emitted names)."""

    node_ids: str = KG_NODE_IDS_KEY
    neighbor_ids: str = NEIGHBOR_NODE_IDS_KEY
    community: str = KG_NODE_COMMUNITY_KEY
    centrality: str = KG_NODE_CENTRALITY_KEY


#: The default keys — the emitted column names.
DEFAULT_KEYS = KgColumnKeys()


@dataclass(frozen=True)
class KgColumns:
    """One candidate's parsed KG columns — read once, consumed as sets and a peak.

    ``nodes`` are the candidate's own entity node ids, ``reach`` the node ids its triples reach
    (its 1-hop expansion keys), ``communities`` its entity community ids, and
    ``peak_centrality`` its most-central entity's value (``0.0`` when the column is absent or
    empty — no hub signal). Frozen: a value object, never a cache.
    """

    nodes: frozenset[str]
    reach: frozenset[str]
    communities: frozenset[str]
    peak_centrality: float

    @classmethod
    def parse(
        cls, candidate: Candidate, *, keys: KgColumnKeys = DEFAULT_KEYS, context: str
    ) -> KgColumns:
        """Parse all four columns off one candidate (graceful-absent / loud-malformed).

        Columns are read in the emitted order (nodes, reach, communities, centrality), so a
        candidate carrying several malformed columns raises on the same one it always did.
        """
        return cls(
            nodes=frozenset(parse_id_list(candidate, keys.node_ids, context=context)),
            reach=frozenset(parse_id_list(candidate, keys.neighbor_ids, context=context)),
            communities=frozenset(parse_id_list(candidate, keys.community, context=context)),
            peak_centrality=peak(
                parse_centrality_list(candidate, keys.centrality, context=context)
            ),
        )


__all__ = [
    "DEFAULT_KEYS",
    "KG_NODE_CENTRALITY_KEY",
    "KG_NODE_COMMUNITY_KEY",
    "KG_NODE_IDS_KEY",
    "NEIGHBOR_NODE_IDS_KEY",
    "KgColumnKeys",
    "KgColumns",
    "parse_centrality_list",
    "parse_id_list",
    "peak",
]

"""The names of the capabilities ADR-0068 removed, in ONE place.

Two gates read this list. ``test_public_release_gate.py`` uses it on the open release gates: a
gate on a removed capability can never be met. ``test_pages_name_no_removed_capability.py`` uses
it on prose pages: a page that describes a removed capability as part of the product is wrong.

The first four names are the list the release gate started with. The others were added when the
list began to police prose, where the removed tooling is named in more ways: the MUVERA encoder,
the BM25 export signal, the Context Enrichment Stage, the graph-feature computers and the MCP
server. Each name was searched for in ``packages/*/src`` before it was added. A name stays here
only while the tree has no such capability; if one returns, its name leaves this list in the
same change.

Not a pytest plugin and not in ``conftest.py``: a plain module the two gates import.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["REMOVED_CAPABILITY", "lines_naming_a_removed_capability"]

#: One alternative per removed capability. Case is ignored. ``MCP`` is matched as a whole word
#: because the three letters occur inside other identifiers.
REMOVED_CAPABILITY = re.compile(
    r"splade|qdrant|graph leg|hyperedge"
    r"|muvera|bm25|context enrichment|graph[_ ]features|\bmcp\b",
    re.IGNORECASE,
)


def lines_naming_a_removed_capability(lines: Iterable[str]) -> list[str]:
    """The lines, stripped, that name a capability ADR-0068 removed."""
    return [line.strip() for line in lines if REMOVED_CAPABILITY.search(line)]

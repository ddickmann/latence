"""A page may not describe a removed capability as part of the product.

ADR-0068 removed the query-time retrieval tooling. The code went in one slice; some pages kept
describing it. ``ROADMAP.md`` listed the retrieval library, the hyperedge sidecar and the MCP
server under "What is built" for as long as nobody rewrote it, and no gate read the page.

The rule checked here is about paragraphs. A paragraph of a listed page that names a removed
capability must also say that it is gone: it has to carry one of the phrases in ``_SAYS_GONE``.
That lets a page keep a dated record ("measured on the earlier pipeline") or state a removal
("was removed by ADR-0068"), and fails a page that presents the capability as current. A fenced
block or a table counts as one paragraph, so a YAML example with a removed config key fails
unless the block itself is marked.

The names come from ``_removed_capabilities.py``, the list the release gate also reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _removed_capabilities import lines_naming_a_removed_capability

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The pages held to the rule. A page joins this list when it has been rewritten against the
#: tree, in the same change, so the entry is red before the rewrite and green after it.
_PAGES = (
    "ROADMAP.md",
    "docs/anatomy/02-extraction-and-privacy.md",
)

#: What a paragraph must say when it names a removed capability.
_SAYS_GONE = re.compile(r"\bremoved\b|no longer|earlier pipeline|ADR-0068", re.IGNORECASE)

_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def stale_paragraphs(text: str) -> list[str]:
    """Every line that names a removed capability inside a paragraph that does not say it is gone."""
    stale: list[str] = []
    for paragraph in _BLANK_LINE.split(text):
        named = lines_naming_a_removed_capability(paragraph.splitlines())
        if named and not _SAYS_GONE.search(paragraph):
            stale.extend(named)
    return stale


@pytest.mark.parametrize("page", _PAGES)
def test_the_page_names_no_capability_the_tree_no_longer_has(page: str) -> None:
    path = _REPO_ROOT / page
    assert path.is_file(), f"{page} is missing; remove it from _PAGES in the same change"
    stale = stale_paragraphs(path.read_text(encoding="utf-8"))
    assert not stale, (
        f"{page} names capabilities removed by ADR-0068 without saying they are gone:\n"
        + "\n".join(stale)
    )


def test_the_rule_fails_a_current_claim_and_passes_a_dated_one() -> None:
    """The check has to be able to fail: the same name, once as a claim and once as a record."""
    claim = "## What is built\n\nAn MCP server exposes the hyperedge sidecar to agents.\n"
    assert stale_paragraphs(claim) == ["An MCP server exposes the hyperedge sidecar to agents."]

    record = "The run produced 677k hyperedges. That count was measured on the earlier pipeline.\n"
    assert stale_paragraphs(record) == []

    # The marker has to be in the same paragraph as the name; one elsewhere on the page is not enough.
    split = "Hyperedges were removed.\n\nThe graph Stage emits hyperedges.\n"
    assert stale_paragraphs(split) == ["The graph Stage emits hyperedges."]

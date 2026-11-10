"""The front door says what the code does today first, and marks the direction as not built.

ADR-0067 re-scoped the project, and ADR-0069 to ADR-0073 fixed parts of the design and its
vocabulary. None of it is implemented. Two ways of getting the front door wrong follow from that,
and both were open: a page that leads with the direction reads as if the engine existed, and a
page that never mentions it describes a product the decisions have already left behind.

So the two front-door pages, ``README.md`` and the docs-site home page, are held to an order and
to one sentence: what the code does, then a section headed *Where this is going* that opens by
saying nothing in it is built. The glossary carries the decided vocabulary in a section of its
own, each term marked *not yet in code*.

The last test is what keeps those markers true. It reads the source tree for the record types
the decisions name. While none exists the markers are right; the day the first one lands this
test fails, and the pages are updated in the same change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_README = _REPO_ROOT / "README.md"
_DOCS_HOME = _REPO_ROOT / "docs" / "index.md"
_GLOSSARY = _REPO_ROOT / "CONTEXT.md"

_DIRECTION_HEADING = "Where this is going"
_NOT_BUILT = "Nothing in this section is built"
_GLOSSARY_HEADING = "Decided, not yet in code"

#: The terms ADR-0069 fixed that the glossary must carry, each marked as not yet in code.
_DECIDED_TERMS = (
    "Ledger",
    "Assertion",
    "Locator",
    "Decision",
    "Actor",
    "Concept",
    "Entity",
    "Binding",
    "Accept band",
)

#: The record types those terms would become. ``Entity`` is left out: the tree already has
#: ``EntityMention`` and canonical entities, which are the existing meanings of the word.
_RECORD_TYPES = ("Assertion", "Locator", "Decision", "Actor", "Concept", "Binding", "Ledger")


def _headings(text: str) -> list[str]:
    """Every ``##`` heading of a page, in order, outside fenced code."""
    found: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            found.append(line[3:].strip())
    return found


def _section(text: str, heading: str) -> str:
    """The body of the ``##`` section called ``heading``, up to the next ``##`` heading."""
    match = re.search(
        rf"^## {re.escape(heading)}[ \t]*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert match, f"no '## {heading}' section"
    return match.group(1)


@pytest.mark.parametrize(
    ("page", "order"),
    [
        (_README, ("What it is", "Quick start", "What is built", "Status", _DIRECTION_HEADING)),
        (_DOCS_HOME, ("Quick start", "What is built", "Status", _DIRECTION_HEADING)),
    ],
    ids=["README.md", "docs/index.md"],
)
def test_the_page_leads_with_today_and_ends_with_the_direction(
    page: Path, order: tuple[str, ...]
) -> None:
    headings = _headings(page.read_text(encoding="utf-8"))
    missing = [h for h in order if h not in headings]
    assert not missing, f"{page.name} has no section(s) {missing}; it has {headings}"
    positions = [headings.index(h) for h in order]
    assert positions == sorted(positions), (
        f"{page.name} must describe what the code does before the direction: wanted the order "
        f"{list(order)}, found {[h for h in headings if h in order]}"
    )


@pytest.mark.parametrize("page", [_README, _DOCS_HOME], ids=["README.md", "docs/index.md"])
def test_the_direction_section_opens_by_saying_it_is_not_built(page: Path) -> None:
    section = _section(page.read_text(encoding="utf-8"), _DIRECTION_HEADING)
    first_paragraph = section.strip().split("\n\n", 1)[0]

    assert _NOT_BUILT in first_paragraph, (
        f"the '{_DIRECTION_HEADING}' section of {page.name} must open with '{_NOT_BUILT}'"
    )
    assert "ADR-0067" in section, f"{page.name} does not cite the re-scope decision"
    assert "ROADMAP.md" in section, f"{page.name} does not send the reader to the roadmap"


def test_the_readme_opening_points_at_the_direction_section() -> None:
    """A reader who stops after the first screen still learns that the scope changed."""
    opening = _section(_README.read_text(encoding="utf-8"), "What it is")
    assert "#where-this-is-going" in opening
    assert "not built" in opening


def _glossary_entries() -> dict[str, str]:
    """``{term: entry text}`` for the glossary's decided-vocabulary section."""
    section = _section(_GLOSSARY.read_text(encoding="utf-8"), _GLOSSARY_HEADING)
    entries: dict[str, str] = {}
    for block in re.split(r"\n\s*\n", section):
        match = re.match(r"\*\*([^*]+)\*\*", block.strip())
        if match:
            entries[match.group(1)] = block
    return entries


def test_the_glossary_carries_the_decided_vocabulary_marked_as_not_in_code() -> None:
    entries = _glossary_entries()

    missing = [term for term in _DECIDED_TERMS if term not in entries]
    assert not missing, f"CONTEXT.md '{_GLOSSARY_HEADING}' has no entry for {missing}"

    unmarked = [term for term in _DECIDED_TERMS if "not yet in code" not in entries[term]]
    assert not unmarked, f"these entries do not say 'not yet in code': {unmarked}"

    uncited = [term for term in _DECIDED_TERMS if "ADR-0069" not in entries[term]]
    assert not uncited, f"these entries do not cite the decision that fixed them: {uncited}"


def test_the_glossary_keeps_the_existing_meanings_of_entity() -> None:
    """The ADR-0069 Entity is added beside the existing uses of the word, not in place of them."""
    text = _GLOSSARY.read_text(encoding="utf-8")
    assert "**Entity Extraction**" in text
    assert "**Disambiguation**" in text

    entry = _glossary_entries()["Entity"]
    assert "entity mention" in entry
    assert "canonical entity" in entry


def test_no_source_module_defines_a_record_type_of_the_re_scope_yet() -> None:
    """The fact behind every 'not yet in code' marker above.

    When this fails, the first slice of the re-scope has landed. Move its term out of the
    glossary's decided-vocabulary section, and correct the 'Where this is going' sections of
    ``README.md`` and ``docs/index.md`` and the roadmap, in the same change.
    """
    defines = re.compile(rf"^class ({'|'.join(_RECORD_TYPES)})\b", re.MULTILINE)
    sources = sorted((_REPO_ROOT / "packages").glob("*/src/**/*.py"))
    assert len(sources) > 100, "the source scan found too few files to mean anything"

    found = sorted(
        f"{path.relative_to(_REPO_ROOT)}: class {name}"
        for path in sources
        for name in defines.findall(path.read_text(encoding="utf-8"))
    )
    assert found == [], (
        "a record type of the re-scope now exists, so the pages that call it unbuilt are "
        f"wrong: {found}"
    )

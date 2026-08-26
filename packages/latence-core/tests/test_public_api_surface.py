"""``latence_core.__all__`` is the published API surface, ENFORCED rather than hand-maintained.

``packages/latence-core/src/latence_core/__init__.py`` is the one door adopters import through:
it re-exports two hundred-odd symbols and then restates them in ``__all__``. The restatement is
the published contract — it is what ``from latence_core import *`` yields, what a documentation
generator enumerates, and what a reader takes as "this is the framework's surface". And it was
maintained entirely by hand: nothing anywhere failed when someone added the next re-export and
did not add the matching ``__all__`` entry.

The failure that produces is silent in both directions. A re-export missing from ``__all__`` is a
symbol that works but is not published — importable, undocumented, and easy to delete later
because no list claims it. An ``__all__`` entry with no matching import is worse: it is a name the
module advertises and does not have, so ``import *`` raises ``AttributeError`` at the adopter's
call site rather than in this repo's suite.

This workspace already gates its dependency caps (ADR-0065), its entry-point names, its ADR
currency, its aggregator coverage and its release integrity. The *public API surface* — the one
artefact every adopter touches first — was the conspicuous omission.

The rule enforced here: **the set of module-level names imported by ``latence_core/__init__.py``
equals the set declared in ``__all__``, exactly**, with no duplicates on either side. Read with
:mod:`ast` from the source file rather than off the imported module, because an imported module
cannot tell a name it re-exported from a name some submodule happened to bind — the source is
where the intent is written down.

``annotations`` (from ``from __future__ import annotations``) is the single deliberate exclusion:
it is a compiler directive, not an export.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _workspace_providers import workspace_root

#: ``from __future__ import annotations`` binds a module-level name that is a compiler directive,
#: never part of the surface. It is the ONLY name excluded, and it is excluded by exact match on
#: the ``__future__`` module rather than by name, so a real export called ``annotations`` would
#: still be gated.
_FUTURE_MODULE = "__future__"


def _init_source(root: Path) -> Path:
    path = root / "packages" / "latence-core" / "src" / "latence_core" / "__init__.py"
    if not path.is_file():
        msg = (
            f"no latence_core package __init__ at {path} — refusing to report an empty API "
            f"surface, which would let this gate pass vacuously."
        )
        raise RuntimeError(msg)
    return path


def imported_names(source: str) -> list[str]:
    """Every module-level name bound by a ``from … import …`` in ``source``, in file order.

    Returned as a LIST, not a set, so the duplicate check has something to see. Only top-level
    statements are read: an import inside a function or an ``if TYPE_CHECKING:`` block is not a
    re-export, and must not be counted as one.
    """
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == _FUTURE_MODULE:
            continue
        names.extend(alias.asname or alias.name for alias in node.names)
    return names


def declared_all(source: str) -> list[str]:
    """The literal string entries of the module-level ``__all__`` assignment, in file order.

    Raises when there is no ``__all__``, or when it is not a plain list of string literals: this
    gate reads the source, so an ``__all__`` built by computation would silently read as empty,
    and an empty expectation is exactly the vacuous green the module exists to prevent.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List) or not all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in node.value.elts
        ):
            msg = (
                "latence_core.__init__ declares __all__ as something other than a list of string "
                "literals; this gate reads it statically and cannot evaluate it."
            )
            raise RuntimeError(msg)
        return [element.value for element in node.value.elts]  # type: ignore[attr-defined]
    msg = "latence_core.__init__ declares no module-level __all__ — the published surface is gone."
    raise RuntimeError(msg)


def _duplicates(names: list[str]) -> list[str]:
    """The names appearing more than once, sorted — empty is the healthy answer."""
    return sorted({name for name in names if names.count(name) > 1})


@pytest.fixture(scope="module")
def source() -> str:
    return _init_source(workspace_root()).read_text(encoding="utf-8")


def test_every_re_export_is_declared_in_all(source: str) -> None:
    """A symbol imported into ``latence_core`` and not listed is importable but unpublished."""
    missing = sorted(set(imported_names(source)) - set(declared_all(source)))
    assert not missing, (
        "symbols re-exported by latence_core/__init__.py that are absent from __all__ — they are "
        "importable but not published, so `from latence_core import *` and the API docs both miss "
        "them. Add them to __all__ (or stop importing them):\n  " + "\n  ".join(missing)
    )


def test_every_all_entry_is_actually_imported(source: str) -> None:
    """An ``__all__`` entry with no import is a name the module advertises and does not have.

    This is the direction that breaks at the ADOPTER's call site: ``from latence_core import *``
    raises ``AttributeError`` for the phantom name, and nothing in this repo would have said so.
    """
    phantom = sorted(set(declared_all(source)) - set(imported_names(source)))
    assert not phantom, (
        "__all__ entries in latence_core/__init__.py with no matching import — `from latence_core "
        "import *` would raise AttributeError for each. Remove them (or import what they name):\n  "
        + "\n  ".join(phantom)
    )


def test_the_surface_has_no_duplicates(source: str) -> None:
    """The same name twice on either side hides one of the two edits that produced it."""
    assert not _duplicates(imported_names(source)), (
        "latence_core/__init__.py imports the same name more than once: "
        f"{_duplicates(imported_names(source))}"
    )
    assert not _duplicates(declared_all(source)), (
        f"latence_core.__all__ lists the same name more than once: {_duplicates(declared_all(source))}"
    )


def test_the_surface_is_not_empty(source: str) -> None:
    """A parse that finds nothing is a broken scan, not a clean tree."""
    assert len(declared_all(source)) > 100, (
        "latence_core.__all__ parsed to fewer than 100 entries — this gate reads the source "
        "statically, and a near-empty read means the parse broke, not that the surface shrank."
    )


def test_the_gate_bites_in_both_directions() -> None:
    """The mechanism ITSELF, proved against injected drift rather than against the real tree.

    Asserted on synthetic sources so this stays a proof on the day the real ``__init__`` is clean
    — which is every day after this one.
    """
    unpublished = "from __future__ import annotations\nfrom x import a, b\n__all__ = ['a']\n"
    assert set(imported_names(unpublished)) - set(declared_all(unpublished)) == {"b"}

    phantom = "from x import a\n__all__ = ['a', 'ghost']\n"
    assert set(declared_all(phantom)) - set(imported_names(phantom)) == {"ghost"}

    aliased = "from x import a as b\n__all__ = ['b']\n"
    assert imported_names(aliased) == ["b"], "an `as` alias binds the ALIAS, not the source name"

    nested = "from x import a\nif True:\n    from y import c\n__all__ = ['a']\n"
    assert imported_names(nested) == ["a"], "only module-level imports are re-exports"

    assert _duplicates(["a", "b", "a"]) == ["a"]

"""Anti-false-green anchor, guard #1 — the vendored ULTRA API surface (torch-FREE, merge gate).

The issue's non-negotiable false-green trap: an un-faithful provider, or a silent drift of the
vendored ULTRA subset, MUST fail an anchor test *in the merge gate*. torch / torch-geometric /
torch-scatter are deliberately NOT installed in the CPU/offline dev+CI env (they would break the
deterministic test env — ADR-0007/0016), so an anchor that *imports* the vendored modules only ever
runs on the pod and never guards a merge.

This guard therefore AST-parses the vendored ``models.py`` / ``tasks.py`` **source** — it never
imports them, never touches torch — and asserts the exact API the spec pins is defined:
``Ultra(rel_model_cfg, entity_model_cfg)`` with ``forward(self, data, batch)``;
``tasks.build_relation_graph(graph)`` / ``tasks.all_negative(data, batch)`` /
``tasks.compute_ranking(pred, target, mask=None)``. Renaming, removing, or reshaping any of these
in the vendored subset turns this test RED on the CPU host that gates merges — so a stub pretending
an API ULTRA doesn't have, or a vendored-subset drift, cannot slip through the green gate.

Guard #2 (``test_ultra_scorer_drive.py``) then drives ``UltraScorer`` end-to-end against a
shape-faithful fake wherever torch is installed (the pod). The two together are the trap.
"""

from __future__ import annotations

import ast
from pathlib import Path

_VENDOR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "latence_linkpred_ultra"
    / "_vendor"
    / "ultra"
)


def _func_params(tree: ast.Module, name: str) -> list[str] | None:
    """Return the positional param names of a top-level ``def name`` (minus self), or None."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args if a.arg != "self"]
    return None


def _method_params(tree: ast.Module, cls: str, method: str) -> list[str] | None:
    """Return the positional param names of ``cls.method`` (minus self), or None if absent."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method:
                    return [a.arg for a in sub.args.args if a.arg != "self"]
    return None


def _parse(name: str) -> ast.Module:
    return ast.parse((_VENDOR / name).read_text())


def test_vendored_ultra_api_surface_matches_the_pinned_spec() -> None:
    """The vendored ULTRA subset defines the exact API the spec pins — the torch-free drift anchor.

    AST-parses the vendored source (no import, no torch), so this fires in the offline CPU merge
    gate: renaming/removing/reshaping ``Ultra.__init__`` / ``Ultra.forward`` / ``all_negative`` /
    ``build_relation_graph`` / ``compute_ranking`` turns this RED without any heavy dep.
    """
    models = _parse("models.py")
    tasks = _parse("tasks.py")

    # Ultra(rel_model_cfg, entity_model_cfg) — the two-config constructor.
    assert _method_params(models, "Ultra", "__init__") == ["rel_model_cfg", "entity_model_cfg"]
    # model(data, batch): forward takes (data, batch).
    assert _method_params(models, "Ultra", "forward") == ["data", "batch"]

    # The three tasks helpers the recipe uses exist and take the pinned arguments.
    assert _func_params(tasks, "build_relation_graph") == ["graph"]
    assert _func_params(tasks, "all_negative") == ["data", "batch"]
    cr = _func_params(tasks, "compute_ranking")
    assert cr is not None and cr[:2] == ["pred", "target"]  # (pred, target, mask=None)

"""#190 (audit R2) — the docs must record the IMPORT-ORDERING fix, not the defeated seam-internal one.

The working #190 fix is an **ordering** property: :func:`neutralize_torch_jit` must run *before* the
Provider evaluates its backbone import (``from gliner import GLiNER`` / ``from gliner2 import
GLiNER2`` / ``from glinker import ProcessorFactory``). That import transitively pulls in
transformers' ``models.deberta_v2.modeling_deberta_v2``, whose relative-position helpers are
``@torch.jit.script``-decorated and are compiled into real ``ScriptFunction``s **at module import
time** — after which neither seam lever can undo it (``PYTORCH_JIT`` is read by torch at *torch*
import; ``jit._state.disable()`` affects only *future* ``script()`` calls).

The code carries that fix and ``test_perf_fp32_regression_guard.py`` enforces it structurally. The
DOCS carried the refuted arrangement: ADR-0036 (W9-jit) mandated the call be made "inside
``load_gliner2_optimized`` and ``load_then_cast`` … **not** copied into five Providers", the
compatibility matrix repeated it as the resolution of a *verified* production kill, and the
``gpu-learned`` stack comment told operators the load seam handles it. A maintainer following that
record deletes the five pre-import calls as banned duplication and silently re-opens the
Blackwell/CUDA-13 ``libnvrtc-builtins.so.13.0`` crash.

Anti-tautology: the property the docs are checked against is **derived from the source** (AST), not
hard-coded here — :func:`_pre_import_backbone_roots` reads which backbone modules are actually
imported after a ``neutralize_torch_jit`` call, and the doc assertions are built from that set. A
Provider that changed its backbone would change what the docs are required to say.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import TypeGuard

import pytest

# The sources that must neutralize the JIT BEFORE importing their DeBERTa-backed backbone: the five
# mdeberta Providers (mirrors ``_MDEBERTA_PROVIDER_SOURCES`` in the perf guard) plus the glinker
# disambiguation backend, which loads through the ``glinker`` library rather than the perf loaders.
_JIT_ORDERED_SOURCES = (
    "latence-ner-gliner/src/latence_ner_gliner/provider.py",
    "latence-relation-gliner/src/latence_relation_gliner/provider.py",
    "latence-pii-gliner/src/latence_pii_gliner/provider.py",
    "latence-pii-gliner2/src/latence_pii_gliner2/provider.py",
    "latence-extract-gliner2/src/latence_extract_gliner2/provider.py",
    "latence-disambig-glinker/src/latence_disambig_glinker/backends.py",
)

_DEBERTA_IMPORT_ROOTS = ("gliner", "gliner2", "glinker", "transformers")

#: Claims the field DISPROVED. Any of these in the #190 record sends a maintainer at the defeated
#: loader-internal mechanism — and at deleting the per-Provider pre-import calls that are the fix.
_REFUTED_CLAIMS = (
    "not** copied into five Providers",
    "not copied into five Providers",
    "Applied at the shared load seam",
    "Applied inside load_gliner2_optimized and load_then_cast",
    "so every mdeberta-backed Provider inherits it",
)

_ORDERING_CLAIM = re.compile(r"\bbefore\b[^.]{0,240}?\bimport", re.IGNORECASE | re.DOTALL)


def _repo_root() -> pathlib.Path:
    """Resolve the repo root from this test file's location (in-tree, not installed)."""
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "packages" / "latence-core").is_dir() and (parent / "docs").is_dir():
            return parent
    msg = "could not locate the repo root from the test file location"
    raise AssertionError(msg)


def _imports_backbone(node: ast.AST) -> str | None:
    """The backbone root module this import statement pulls in, if any."""
    if isinstance(node, ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        return root if root in _DEBERTA_IMPORT_ROOTS else None
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _DEBERTA_IMPORT_ROOTS:
                return root
    return None


def _calls_neutralize(node: ast.AST) -> TypeGuard[ast.Call]:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "neutralize_torch_jit"
    return isinstance(func, ast.Attribute) and func.attr == "neutralize_torch_jit"


def _pre_import_backbone_roots() -> set[str]:
    """The backbone modules the CODE imports *after* neutralizing the JIT, read out of the sources.

    This is the fact the documentation is then held to. It is derived, never asserted literally: if
    a Provider swapped its backbone the required doc vocabulary would move with it.
    """
    packages = _repo_root() / "packages"
    roots: set[str] = set()
    for rel in _JIT_ORDERED_SOURCES:
        path = packages / rel
        assert path.is_file(), f"expected source at {path}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            neutralizations = [n.lineno for n in ast.walk(func) if _calls_neutralize(n)]
            if not neutralizations:
                continue
            first_neutralization = min(neutralizations)
            for node in ast.walk(func):
                if not isinstance(node, ast.Import | ast.ImportFrom):
                    continue
                root = _imports_backbone(node)
                if root is not None and node.lineno > first_neutralization:
                    roots.add(root)
    return roots


def _adr_jit_section() -> str:
    """The W9-jit section of ADR-0036 — the authoritative record of the #190 fix."""
    text = (
        _repo_root() / "docs" / "adr" / "0036-provider-ecosystem-and-bakeoff.md"
    ).read_text(encoding="utf-8")
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("### W9-jit")]
    assert starts, "ADR-0036 no longer has a '### W9-jit' section — the #190 record must live there"
    start = starts[0]
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith("## ") or lines[i].startswith("### ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _matrix_failure_mode_text() -> str:
    """The FM-NVRTC-JIT block of the compatibility-matrix YAML (source of the published document)."""
    text = (
        _repo_root() / "deploy" / "env" / "compatibility-matrix.yaml"
    ).read_text(encoding="utf-8")
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "- id: FM-NVRTC-JIT"]
    assert starts, "the compatibility matrix no longer records the FM-NVRTC-JIT failure mode (#190)"
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].strip()
            and (len(lines[i]) - len(lines[i].lstrip())) <= indent
            and lines[i].lstrip().startswith("- ")
        ),
        len(lines),
    )
    # The YAML folds its prose across lines, so a claim is only detectable once re-flowed.
    return " ".join("\n".join(lines[start:end]).split())


def test_the_code_neutralizes_the_jit_before_every_backbone_import() -> None:
    """The code fact the docs are held to: a backbone is imported only after the JIT is off.

    Without this the doc assertions below could be held to a property the code does not have.
    """
    roots = _pre_import_backbone_roots()
    assert roots, (
        "no source neutralizes the JIT before importing a DeBERTa backbone — the #190 "
        "import-ordering fix has been removed (see test_perf_fp32_regression_guard.py)"
    )


def test_the_adr_records_the_import_ordering_requirement() -> None:
    """ADR-0036 W9-jit names the backbone import the neutralization must precede."""
    section = _adr_jit_section()
    for root in sorted(_pre_import_backbone_roots()):
        assert f"from {root} import" in section, (
            f"ADR-0036 W9-jit never mentions the `from {root} import` statement the "
            "neutralization must precede — the import-ordering requirement is unrecorded, so a "
            "maintainer cannot know the seam-internal call alone is defeated (#190)"
        )
    assert _ORDERING_CLAIM.search(section), (
        "ADR-0036 W9-jit states no ordering ('before … import') requirement for the #190 fix"
    )


@pytest.mark.parametrize("claim", _REFUTED_CLAIMS)
def test_the_adr_no_longer_mandates_the_defeated_seam_internal_mechanism(claim: str) -> None:
    """The refuted arrangement must not survive in the authoritative record.

    ADR-0036 mandated the call be made *inside* the shared loaders and explicitly banned the
    per-Provider copy — which is the actual fix. A maintainer following that record deletes the
    pre-import calls and silently re-opens the Blackwell/CUDA-13 nvrtc crash.
    """
    assert claim not in _adr_jit_section(), (
        f"ADR-0036 W9-jit still asserts {claim!r}: the loader-internal call is DEFEATED (the "
        "backbone import scripts the deberta helpers one statement earlier) and the per-Provider "
        "pre-import call it bans is the working fix (#190)"
    )


@pytest.mark.parametrize("claim", _REFUTED_CLAIMS)
def test_the_compatibility_matrix_no_longer_records_the_defeated_resolution(claim: str) -> None:
    """FM-NVRTC-JIT is the record of a VERIFIED production kill — its resolution must be the real one."""
    assert claim not in _matrix_failure_mode_text(), (
        f"compatibility-matrix.yaml FM-NVRTC-JIT still gives {claim!r} as the resolution of a "
        "verified 519-page production failure — an operator debugging that crash is pointed at a "
        "mechanism the field disproved (#190)"
    )


def test_the_compatibility_matrix_records_the_import_ordering_requirement() -> None:
    """The matrix resolution states the ordering, and the published document is in step with it."""
    block = _matrix_failure_mode_text()
    assert _ORDERING_CLAIM.search(block), (
        "compatibility-matrix.yaml FM-NVRTC-JIT states no ordering ('before … import') "
        "requirement — the property that makes the #190 fix work is unrecorded"
    )
    published = (_repo_root() / "docs" / "compatibility-matrix.md").read_text(encoding="utf-8")
    for claim in _REFUTED_CLAIMS:
        assert claim not in published, (
            f"docs/compatibility-matrix.md still publishes {claim!r} — re-render it from the YAML "
            "(`latence env matrix deploy/env/compatibility-matrix.yaml --render`)"
        )


def test_the_gpu_learned_stack_does_not_credit_the_load_seam_for_the_jit_fix() -> None:
    """The operator-facing stack comment must not say the load seam handles #190 on its own."""
    stack = (
        _repo_root() / "deploy" / "env" / "gpu-learned" / "stack.yaml"
    ).read_text(encoding="utf-8")
    jit_comment = "\n".join(
        line for line in stack.splitlines() if line.lstrip().startswith("#")
    )
    assert "neutralizes the\n  # TorchScript JIT (PYTORCH_JIT=0) before the load" not in stack, (
        "deploy/env/gpu-learned/stack.yaml still tells operators the shared perf module neutralizes "
        "the JIT 'before the load' — the load is too late; the Provider does it before its backbone "
        "import (#190)"
    )
    assert _ORDERING_CLAIM.search(jit_comment), (
        "deploy/env/gpu-learned/stack.yaml's #190 comment states no ordering ('before … import')"
    )

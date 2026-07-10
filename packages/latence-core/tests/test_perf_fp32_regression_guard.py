"""The fp32-miss regression guard (W8) — AST scan: NO learned model loads bare fp32.

The single teeth of the W8 "no bare fp32 load" contract: an AST scan over every learned Provider's
source that FAILS if a MODEL ``from_pretrained`` is called WITHOUT the perf kwargs (dtype +
attn_implementation). Concretely, a learned model load must go through the shared
``load_with_fa2_fallback(Model.from_pretrained, model_id, perf)`` seam (which spreads the
``load_perf_kwargs`` dict) OR pass ``torch_dtype`` / ``attn_implementation`` directly — a bare
``Model.from_pretrained(model_id)`` is the fp32/eager regression this guard bites.

It is deliberately robust against false negatives (a bare fp32 load it would miss) AND false
positives (a processor/tokenizer load, which takes no perf kwargs):

* it scans the REAL provider sources in-tree (resolved from this test's location), so a new learned
  provider that loads a model bare is caught the moment its source lands;
* it targets MODEL loads only — a callee whose class name matches a model pattern
  (``GLiNER``/``GLiNER2``/``*ForConditionalGeneration``/``AutoModel*``) — so a bare
  ``AutoProcessor.from_pretrained`` / ``LightOnOcrProcessor.from_pretrained`` (no perf kwargs by
  design) is correctly ignored;
* a model ``from_pretrained`` is OK iff it is the first argument to a ``load_with_fa2_fallback``
  call (the perf seam), or it passes ``torch_dtype`` / ``attn_implementation`` as a direct kwarg.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# The learned model-loading Providers this guard covers. Each loads a real checkpoint that MUST be
# loaded through the perf seam (bf16-on-GPU + sdpa/FA2), never bare fp32. Pure-code / endpoint / stub
# providers (pdfplumber, plaintext, presidio-spaCy, the *-endpoint HTTP providers) load no local
# model checkpoint and are out of scope.
_LEARNED_PROVIDER_SOURCES = (
    "latence-ner-gliner/src/latence_ner_gliner/provider.py",
    "latence-relation-gliner/src/latence_relation_gliner/provider.py",
    "latence-pii-gliner/src/latence_pii_gliner/provider.py",
    "latence-pii-gliner2/src/latence_pii_gliner2/provider.py",
    "latence-parser-lighton/src/latence_parser_lighton/provider.py",
    "latence-parser-glm/src/latence_parser_glm/provider.py",
    "latence-embedder-st/src/latence_embedder_st/provider.py",
)

# A callee whose (attribute) name matches one of these is a MODEL load that MUST go through the perf
# seam. Processors/tokenizers/feature-extractors are deliberately NOT here (they load bare, no perf).
_MODEL_CALLEE_HINTS = ("GLiNER", "ForConditionalGeneration", "AutoModel")

# The TWO valid perf-seam load helpers a model ``from_pretrained`` may be handed to as its first arg:
#   * ``load_with_fa2_fallback`` — spreads the perf kwargs (torch_dtype + attn_implementation) into
#     from_pretrained; for libs that ACCEPT them (gliner2, the OCR VLMs, ST via model_kwargs).
#   * ``load_then_cast`` — loads PLAIN then ``model.to(dtype)``; for libs that REJECT the perf kwargs
#     (the plain ``gliner`` 0.2.27 library — pod-proven to TypeError on torch_dtype). This is the
#     W8-corrected path for the 3 gliner-library providers.
_PERF_SEAM_FUNCS = ("load_with_fa2_fallback", "load_then_cast")

# ...or the call itself passes one of these kwargs directly.
_PERF_KWARGS = ("torch_dtype", "attn_implementation")

# The plain-``gliner``-library providers: their model MUST load via ``load_then_cast`` (the .to()
# path), NEVER via perf kwargs — which crash on gliner 0.2.27's real from_pretrained (pod-proven).
_PLAIN_GLINER_SOURCES = frozenset(
    {
        "latence-ner-gliner/src/latence_ner_gliner/provider.py",
        "latence-relation-gliner/src/latence_relation_gliner/provider.py",
        "latence-pii-gliner/src/latence_pii_gliner/provider.py",
    }
)


def _packages_root() -> pathlib.Path:
    """Resolve the repo ``packages/`` dir from this test file's location (in-tree, not installed)."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "packages"
        if candidate.is_dir() and (candidate / "latence-core").is_dir():
            return candidate
    msg = "could not locate the repo 'packages/' directory from the test file location"
    raise AssertionError(msg)


def _callee_attr_name(node: ast.Call) -> str | None:
    """The attribute name of a ``X.attr(...)`` call, e.g. ``from_pretrained`` for ``M.from_pretrained``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _callee_root_name(node: ast.Call) -> str:
    """A best-effort string of the callee's attribute chain, e.g. ``GLiNER.from_pretrained``."""
    return ast.unparse(node.func)


def _is_model_from_pretrained(node: ast.Call) -> bool:
    """Whether this call is a MODEL-class ``from_pretrained`` (not a processor/tokenizer)."""
    if _callee_attr_name(node) != "from_pretrained":
        return False
    root = _callee_root_name(node)
    return any(hint in root for hint in _MODEL_CALLEE_HINTS)


def _model_from_pretrained_passed_to_perf_seam(tree: ast.Module) -> set[int]:
    """The id()s of model ``from_pretrained`` *attribute* exprs handed to the perf seam as its func arg.

    ``load_with_fa2_fallback(Model.from_pretrained, model_id, perf)`` passes the bound
    ``Model.from_pretrained`` (an ``ast.Attribute``, NOT a call) as its first argument — so the model
    load never appears as a bare ``ast.Call`` at all. Collect those attribute nodes so the scan does
    not flag them (they ARE going through the seam).
    """
    ok: set[int] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if name not in _PERF_SEAM_FUNCS or not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Attribute) and first.attr == "from_pretrained":
            ok.add(id(first))
    return ok


@pytest.mark.parametrize("rel_source", _LEARNED_PROVIDER_SOURCES)
def test_no_learned_model_loads_bare_fp32(rel_source: str) -> None:
    """Every learned Provider loads its model through the perf seam — no bare fp32 ``from_pretrained``."""
    path = _packages_root() / rel_source
    assert path.is_file(), f"expected learned provider source at {path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    seam_routed = _model_from_pretrained_passed_to_perf_seam(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if id(node.func) in seam_routed:
            # This ``Model.from_pretrained`` is the func arg of the perf seam — perf kwargs applied.
            continue
        if not _is_model_from_pretrained(node):
            continue
        # A direct model ``from_pretrained(...)`` call: OK only if it passes the perf kwargs itself.
        kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if kwarg_names & set(_PERF_KWARGS):
            continue
        offenders.append(f"line {node.lineno}: {_callee_root_name(node)}(...) without perf kwargs")

    assert not offenders, (
        f"{rel_source}: learned MODEL load(s) bypass the W8 perf seam (bare fp32 regression):\n"
        + "\n".join(offenders)
    )


def test_sentence_transformers_construction_threads_model_kwargs() -> None:
    """The ST embedder constructs ``SentenceTransformer`` through the FA2-fallback loader (model_kwargs)."""
    path = _packages_root() / "latence-embedder-st/src/latence_embedder_st/provider.py"
    src = path.read_text(encoding="utf-8")
    # The ST load is routed through the shared perf-seam loader, which threads model_kwargs (perf) into
    # the SentenceTransformer(...) construction — assert the seam is used AND model_kwargs is threaded.
    assert "load_with_fa2_fallback" in src, "ST embedder must load through the shared perf seam"
    assert "model_kwargs" in src, "ST embedder must thread perf kwargs via model_kwargs"


@pytest.mark.parametrize("rel_source", sorted(_PLAIN_GLINER_SOURCES))
def test_plain_gliner_uses_load_then_cast_never_perf_kwargs(rel_source: str) -> None:
    """The 3 plain-``gliner`` providers load via ``load_then_cast`` (the ``.to(dtype)`` path) and
    pass NO perf kwargs to ``from_pretrained`` — which crash on the real gliner 0.2.27 API (pod-proven).

    This is the W8 correctness fix: gliner 0.2.27's ``from_pretrained`` rejects torch_dtype /
    attn_implementation, so the bf16 win must come from ``model.to(bf16)`` inside ``load_then_cast``,
    never from spreading the perf kwargs. Passing them (via ``load_with_fa2_fallback`` /
    ``load_perf_kwargs``) would re-introduce the crash — this guard forbids exactly that.
    """
    path = _packages_root() / rel_source
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    # It MUST route the model load through load_then_cast (the .to(dtype) path).
    calls = [
        c
        for c in ast.walk(tree)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "load_then_cast"
        and c.args
        and isinstance(c.args[0], ast.Attribute)
        and c.args[0].attr == "from_pretrained"
    ]
    assert calls, f"{rel_source}: plain-gliner provider must load its model via load_then_cast(...)"

    # It must NOT use the perf-kwargs seam or load_perf_kwargs (those crash gliner 0.2.27).
    assert "load_with_fa2_fallback" not in src, (
        f"{rel_source}: plain gliner must NOT use load_with_fa2_fallback "
        "(gliner 0.2.27 from_pretrained rejects torch_dtype/attn_implementation)"
    )
    assert "load_perf_kwargs" not in src, (
        f"{rel_source}: plain gliner must NOT spread load_perf_kwargs into from_pretrained"
    )


def test_guard_would_catch_a_bare_model_load() -> None:
    """Meta-test: the AST scan FAILS on a synthetic bare fp32 model load (no false-negative)."""
    bad = "m = GLiNER.from_pretrained(model_id)\n"
    tree = ast.parse(bad)
    seam_routed = _model_from_pretrained_passed_to_perf_seam(tree)
    flagged = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and id(n.func) not in seam_routed
        and _is_model_from_pretrained(n)
        and not ({kw.arg for kw in n.keywords} & set(_PERF_KWARGS))
    ]
    assert flagged, "the guard must flag a bare Model.from_pretrained(model_id) load"


def test_guard_accepts_both_seam_routed_loads() -> None:
    """Meta-test: the AST scan does NOT flag EITHER perf-seam helper (fa2-fallback OR load_then_cast)."""
    for good in (
        "m = load_with_fa2_fallback(GLiNER.from_pretrained, model_id, perf)\n",
        "m = load_then_cast(GLiNER.from_pretrained, model_id, config, None)\n",
    ):
        tree = ast.parse(good)
        seam_routed = _model_from_pretrained_passed_to_perf_seam(tree)
        flagged = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and id(n.func) not in seam_routed
            and _is_model_from_pretrained(n)
        ]
        assert not flagged, f"a perf-seam-routed model load must NOT be flagged: {good!r}"

"""The fp32-miss regression guard (W8) — AST scan: NO learned model loads bare fp32.

The single teeth of the W8 "no bare fp32 load" contract: an AST scan over every learned Provider's
source that FAILS if a MODEL ``from_pretrained`` is called WITHOUT the perf kwargs (dtype +
attn_implementation). Concretely, a learned model load must go through the shared
``load_with_fa2_fallback(Model.from_pretrained, model_id, perf)`` seam (which spreads the
``load_perf_kwargs`` dict) OR pass ``torch_dtype`` / ``attn_implementation`` directly — a bare
``Model.from_pretrained(model_id)`` is the fp32/eager regression this guard bites.

It is deliberately robust against false negatives (a bare fp32 load it would miss) AND false
positives (a processor/tokenizer load, which takes no perf kwargs):

* it scans the REAL provider sources in-tree (resolved from this test's location), never installed
  copies. Note the two scopes differ in how their file set is obtained: the W8 fp32 scan below
  parametrizes over the curated ``_LEARNED_PROVIDER_SOURCES`` list (a new learned provider must be
  added to it), whereas the #190 JIT guards parametrize over ``_DEBERTA_BACKED_SOURCES``, which is
  DISCOVERED by globbing ``packages/*/src`` — so for #190 a new backbone importer really is caught
  the moment its source lands, with no list to edit (R2/#190-5; before that fix the #190 guards used
  a hardcoded 5-tuple and missed ``disambiguation.glinker`` entirely);
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
from typing import TypeGuard

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
    "latence-extract-gliner2/src/latence_extract_gliner2/provider.py",
    "latence-parser-lighton/src/latence_parser_lighton/provider.py",
    "latence-parser-glm/src/latence_parser_glm/provider.py",
    "latence-embedder-st/src/latence_embedder_st/provider.py",
)

# A callee whose (attribute) name matches one of these is a MODEL load that MUST go through the perf
# seam. Processors/tokenizers/feature-extractors are deliberately NOT here (they load bare, no perf).
_MODEL_CALLEE_HINTS = ("GLiNER", "ForConditionalGeneration", "AutoModel")

# The valid perf-seam load helpers a model ``from_pretrained`` may be handed to as its first arg:
#   * ``load_with_fa2_fallback`` — spreads the perf kwargs (torch_dtype + attn_implementation) into
#     from_pretrained; for libs that ACCEPT them (the OCR VLMs, ST via model_kwargs).
#   * ``load_then_cast`` — loads PLAIN then ``model.to(dtype)``; for libs that REJECT the perf kwargs
#     (the plain ``gliner`` 0.2.27 library — pod-proven to TypeError on torch_dtype). This is the
#     W8-corrected path for the 3 gliner-library providers.
#   * ``load_gliner2_optimized`` — passes the gliner2 NATIVE ``quantize``/``compile`` kwargs (W9),
#     guarded so a fault degrades to a plain load; for the two gliner2-framework providers
#     (``redaction.gliner2`` + ``fused_entity_relation.gliner2``) whose from_pretrained accepts them.
_PERF_SEAM_FUNCS = ("load_with_fa2_fallback", "load_then_cast", "load_gliner2_optimized")

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


def _model_from_pretrained_passed_to_perf_seam(
    tree: ast.Module, seam_funcs: tuple[str, ...] = _PERF_SEAM_FUNCS
) -> set[int]:
    """The id()s of model ``from_pretrained`` *attribute* exprs handed to ``seam_funcs`` as func arg.

    ``load_with_fa2_fallback(Model.from_pretrained, model_id, perf)`` passes the bound
    ``Model.from_pretrained`` (an ``ast.Attribute``, NOT a call) as its first argument — so the model
    load never appears as a bare ``ast.Call`` at all. Collect those attribute nodes so the scan does
    not flag them (they ARE going through the seam).

    ``seam_funcs`` is a parameter, not a constant, because the accept-set differs per guard: the W8
    fp32 guard accepts ALL of :data:`_PERF_SEAM_FUNCS` (all three spread/cast a dtype), while the
    #190 JIT guard accepts only :data:`_JIT_NEUTRALIZING_LOADERS` (``load_with_fa2_fallback`` does
    NOT call ``neutralize_torch_jit``, so routing an mdeberta load through it is an offence).
    """
    ok: set[int] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if name not in seam_funcs or not call.args:
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


@pytest.mark.parametrize("rel_source", sorted(_PLAIN_GLINER_SOURCES))
def test_plain_gliner_places_its_model_on_the_resolved_device(rel_source: str) -> None:
    """EVERY plain-``gliner`` provider must PLACE its model (#191), not just cast its dtype.

    #191 proved the plain ``gliner`` library never auto-places — ``GLiNER.from_pretrained``
    map-locations to CPU — and encoded the fix in ``place_on_device``. But the fix was originally
    wired to only ``redaction.gliner_pii``; ``entity.gliner`` and ``relation.gliner_relex`` kept
    passing ``device=None`` into ``load_then_cast``, so they cast to bf16 off a bare
    ``cuda_available()`` probe while the weights stayed resident on CPU — bf16-eager-on-CPU, slower
    AND numerically different from the fp32 CPU path the seam intends, with no warning.

    Structural teeth so a FOURTH plain-gliner provider cannot land without the fix:

    1. the source calls ``place_on_device(...)``, and
    2. its ``load_then_cast`` call does NOT hand a literal ``None`` as the device argument — the
       dtype must key off the RESOLVED device, never a bare availability probe.
    """
    path = _packages_root() / rel_source
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    assert "place_on_device" in src, (
        f"{rel_source}: plain gliner does NOT auto-place its model (#191) — the provider must "
        "call place_on_device(model, device, ...) after load_then_cast, or it silently runs on "
        "CPU while claiming the resolved device"
    )

    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "load_then_cast"
            and len(call.args) >= 4
        ):
            device_arg = call.args[3]
            assert not (isinstance(device_arg, ast.Constant) and device_arg.value is None), (
                f"{rel_source}: load_then_cast(..., None) makes the bf16 cast key off a bare "
                "CUDA-availability probe while the model stays on CPU (#191) — pass the RESOLVED "
                "device instead"
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
    """Meta-test: the AST scan does NOT flag any perf-seam helper (fa2-fallback / then_cast / W9)."""
    for good in (
        "m = load_with_fa2_fallback(GLiNER.from_pretrained, model_id, perf)\n",
        "m = load_then_cast(GLiNER.from_pretrained, model_id, config, None)\n",
        "m = load_gliner2_optimized(GLiNER2.from_pretrained, model_id, dev, cfg)\n",
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


# --- #190: the mdeberta JIT-neutralization is a SHARED seam, never a per-Provider copy ------------

# The DeBERTa backbone roots. Importing ANY of them transitively evaluates transformers'
# ``models.deberta_v2.modeling_deberta_v2``, whose six relative-position helpers carry
# ``@torch.jit.script`` and are therefore compiled into real ``ScriptFunction``s AT IMPORT — the
# instant that decides the Blackwell/CUDA-13 nvrtc crash (``libnvrtc-builtins.so.13.0``, #190).
_DEBERTA_BACKBONE_ROOTS = ("gliner", "gliner2", "glinker")


def _imports_a_deberta_backbone_root(tree: ast.Module) -> bool:
    """Whether ``tree`` contains an import of a DeBERTa backbone library (at any nesting depth)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _DEBERTA_BACKBONE_ROOTS:
                return True
        elif isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] in _DEBERTA_BACKBONE_ROOTS for alias in node.names
        ):
            return True
    return False


def _discover_deberta_backed_sources() -> tuple[str, ...]:
    """Every in-tree package source that imports a DeBERTa backbone — DISCOVERED, never listed.

    R2/#190-5: this used to be a hardcoded 5-path tuple, and the sixth in-tree DeBERTa-backed
    Provider (``disambiguation.glinker``, whose ``GLinkerBackend._init_executor`` imports
    ``glinker``) was simply absent — so it had ZERO #190 coverage while this file's docstring
    claimed "a new learned provider … is caught the moment its source lands". Discovery makes that
    claim true: a seventh backbone importer is scanned by the guards below without editing a list.
    """
    root = _packages_root()
    found = [
        path.relative_to(root).as_posix()
        for path in sorted(root.glob("*/src/**/*.py"))
        if _imports_a_deberta_backbone_root(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
    ]
    return tuple(found)


#: Every in-tree source that imports a DeBERTa backbone. The ordering guard (#190-4) and the
#: no-hand-rolled-``PYTORCH_JIT`` guard apply to ALL of them: each must kill the TorchScript JIT via
#: the SHARED seam (``latence_core.providers.perf.neutralize_torch_jit``) before that import runs.
_DEBERTA_BACKED_SOURCES = _discover_deberta_backed_sources()

#: The DeBERTa-backed sources known to exist when this guard was written. Discovery must never
#: return LESS than this — a shrinking scan is how a Provider silently falls out of coverage.
_KNOWN_DEBERTA_BACKED_SOURCES = frozenset(
    {
        "latence-ner-gliner/src/latence_ner_gliner/provider.py",
        "latence-relation-gliner/src/latence_relation_gliner/provider.py",
        "latence-pii-gliner/src/latence_pii_gliner/provider.py",
        "latence-pii-gliner2/src/latence_pii_gliner2/provider.py",
        "latence-extract-gliner2/src/latence_extract_gliner2/provider.py",
        "latence-disambig-glinker/src/latence_disambig_glinker/backends.py",
    }
)

# The shared perf-seam loaders that neutralize the TorchScript JIT before calling the native loader.
# NOT the whole of ``_PERF_SEAM_FUNCS``: ``load_with_fa2_fallback`` spreads the perf kwargs but does
# NOT call ``neutralize_torch_jit`` — an mdeberta load routed through it stays exposed to the
# Blackwell/CUDA-13 nvrtc crash. This constant is the accept-set the #190 guard actually asserts on.
_JIT_NEUTRALIZING_LOADERS = ("load_then_cast", "load_gliner2_optimized")

# Hints matched case-INSENSITIVELY against the attribute chain of a ``*.from_pretrained`` reference
# in an mdeberta Provider, so an indirected class handle (``gliner2_cls.from_pretrained``,
# ``self._gliner2_cls().from_pretrained``) is covered too. Tokenizers/processors do not match.
_MDEBERTA_MODEL_HINTS = ("gliner", "forconditionalgeneration", "automodel")


def _is_mdeberta_model_from_pretrained_ref(node: ast.AST) -> TypeGuard[ast.Attribute]:
    """Whether ``node`` is a ``<mdeberta model>.from_pretrained`` *reference* (attribute expr)."""
    if not isinstance(node, ast.Attribute) or node.attr != "from_pretrained":
        return False
    root = ast.unparse(node).lower()
    return any(hint in root for hint in _MDEBERTA_MODEL_HINTS)


def _loads_an_mdeberta_model(tree: ast.Module) -> bool:
    """Whether ``tree`` carries at least one ``<mdeberta model>.from_pretrained`` reference."""
    return any(_is_mdeberta_model_from_pretrained_ref(node) for node in ast.walk(tree))


#: The subset of :data:`_DEBERTA_BACKED_SOURCES` that actually LOADS an mdeberta checkpoint via
#: ``<Model>.from_pretrained`` — DERIVED, not listed, so it cannot drift either. Only these can be
#: held to the routing contract ("the load goes through a JIT-neutralizing shared loader"): a
#: backbone importer that owns no ``from_pretrained`` (``disambiguation.glinker``, which hands the
#: checkpoint to the glinker library's own DAG and calls the shared seam directly) has no load to
#: route, and is covered by the ORDERING guard instead.
_MDEBERTA_PROVIDER_SOURCES = tuple(
    rel
    for rel in _DEBERTA_BACKED_SOURCES
    if _loads_an_mdeberta_model(
        ast.parse((_packages_root() / rel).read_text(encoding="utf-8"), filename=rel)
    )
)


def jit_seam_offenders(tree: ast.Module) -> list[str]:
    """Every mdeberta model ``from_pretrained`` in ``tree`` NOT routed through a JIT-neutralizing loader.

    Scans **attribute references**, not calls, because both offending shapes reduce to one: a direct
    ``GLiNER2.from_pretrained(model_id)`` call carries the attribute as its ``func``, and a load
    handed to a non-neutralizing loader (``load_with_fa2_fallback(GLiNER.from_pretrained, ...)``)
    passes the very same attribute node as an argument. Only the references collected as arguments
    of :data:`_JIT_NEUTRALIZING_LOADERS` are exempt.
    """
    jit_routed = _model_from_pretrained_passed_to_perf_seam(tree, _JIT_NEUTRALIZING_LOADERS)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not _is_mdeberta_model_from_pretrained_ref(node) or id(node) in jit_routed:
            continue
        offenders.append(f"line {node.lineno}: {ast.unparse(node)}")
    return offenders


@pytest.mark.parametrize("rel_source", _MDEBERTA_PROVIDER_SOURCES)
def test_mdeberta_providers_load_through_a_jit_neutralizing_seam(rel_source: str) -> None:
    """Every mdeberta-backed Provider's model load routes through the shared JIT-neutralizing seam.

    The teeth for #190's "shared seam, not a per-Provider copy": a further mdeberta Provider that
    calls ``GLiNER2.from_pretrained`` directly — **or** routes it through ``load_with_fa2_fallback``,
    which spreads the perf kwargs but does NOT neutralize the JIT — re-opens the Blackwell nvrtc
    crash. Its parametrization is DERIVED from :data:`_DEBERTA_BACKED_SOURCES` (R2/#190-5), so that
    claim is true of the code: the scan bites the moment such a source lands, with no list to edit.
    """
    path = _packages_root() / rel_source
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    jit_routed = _model_from_pretrained_passed_to_perf_seam(tree, _JIT_NEUTRALIZING_LOADERS)
    assert jit_routed, (
        f"{rel_source}: no model load routed through the shared JIT-neutralizing loaders "
        f"({', '.join(_JIT_NEUTRALIZING_LOADERS)})"
    )

    offenders = jit_seam_offenders(tree)
    assert not offenders, (
        f"{rel_source}: mdeberta model load(s) bypass the shared JIT-neutralizing loaders "
        f"({', '.join(_JIT_NEUTRALIZING_LOADERS)}) — re-opens the Blackwell/CUDA-13 nvrtc crash:\n"
        + "\n".join(offenders)
    )


def test_jit_guard_flags_a_load_routed_through_the_non_neutralizing_fa2_loader() -> None:
    """Meta-test: ``load_with_fa2_fallback`` is NOT an acceptable route for an mdeberta model.

    The pre-fix guard reused the W8 accept-set (``_PERF_SEAM_FUNCS``), which includes
    ``load_with_fa2_fallback`` — so this source produced ZERO offenders and the #190 AC was
    unenforced. The guard must flag it.
    """
    tree = ast.parse("m = load_with_fa2_fallback(GLiNER.from_pretrained, model_id, perf)\n")
    assert jit_seam_offenders(tree), (
        "an mdeberta model load handed to load_with_fa2_fallback (which never calls "
        "neutralize_torch_jit) must be flagged by the #190 guard"
    )


def test_jit_guard_flags_a_bare_and_an_indirected_mdeberta_load() -> None:
    """Meta-test: a bare load and an indirected class handle are both flagged (no false negative)."""
    for bad in (
        "m = GLiNER2.from_pretrained(model_id)\n",
        "m = gliner2_cls.from_pretrained(model_id, quantize=True)\n",
        "m = self._gliner2_cls().from_pretrained(model_id)\n",
    ):
        assert jit_seam_offenders(ast.parse(bad)), f"must be flagged: {bad!r}"


def test_jit_guard_accepts_the_neutralizing_loaders() -> None:
    """Meta-test: the two real JIT-neutralizing routes are NOT flagged (no false positive)."""
    for good in (
        "m = load_then_cast(GLiNER.from_pretrained, model_id, config, device)\n",
        "m = load_gliner2_optimized(gliner2_cls.from_pretrained, model_id, dev, cfg)\n",
        "m = load_gliner2_optimized(self._gliner2_cls().from_pretrained, model_id, dev, cfg)\n",
        # a processor/tokenizer load takes no perf kwargs and is not an mdeberta model
        "p = AutoProcessor.from_pretrained(model_id)\n",
    ):
        assert not jit_seam_offenders(ast.parse(good)), f"must NOT be flagged: {good!r}"


# --- #190-4: the JIT must be neutralized BEFORE the gliner import, not merely at the loader -------

#: Import roots whose evaluation transitively pulls in transformers'
#: ``models.deberta_v2.modeling_deberta_v2`` — the module whose six relative-position helpers carry
#: ``@torch.jit.script`` and are therefore compiled into real ``ScriptFunction``s **at import**.
_DEBERTA_IMPORT_ROOTS = ("gliner", "gliner2", "glinker", "transformers")


def _imports_deberta_backbone(node: ast.AST) -> TypeGuard[ast.Import | ast.ImportFrom]:
    """Whether ``node`` is an import statement that pulls in the deberta_v2 modeling module."""
    if isinstance(node, ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        return root in _DEBERTA_IMPORT_ROOTS
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in _DEBERTA_IMPORT_ROOTS for alias in node.names)
    return False


def _calls_neutralize_torch_jit(node: ast.AST) -> TypeGuard[ast.Call]:
    """Whether ``node`` is a ``neutralize_torch_jit(...)`` call (plain or attribute-qualified)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "neutralize_torch_jit"
    return isinstance(func, ast.Attribute) and func.attr == "neutralize_torch_jit"


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Whether ``test`` is the ``TYPE_CHECKING`` / ``typing.TYPE_CHECKING`` constant."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _collect_module_scope_backbone_imports(
    node: ast.AST, found: list[ast.Import | ast.ImportFrom]
) -> None:
    """Gather backbone imports **evaluated at module import** (see :func:`_module_scope_...`)."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        return  # a function body runs on call, not at import — the ordering scan owns it
    if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
        for stmt in node.orelse:  # the ELSE branch is the one that actually runs at import
            _collect_module_scope_backbone_imports(stmt, found)
        return
    if _imports_deberta_backbone(node):
        found.append(node)
        return
    for child in ast.iter_child_nodes(node):
        _collect_module_scope_backbone_imports(child, found)


def _module_scope_backbone_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Backbone imports that run at MODULE import: outside every function, outside TYPE_CHECKING.

    Class bodies count — they are executed at module import like any other top-level statement.
    ``if TYPE_CHECKING:`` bodies do NOT: they are never evaluated at runtime (and every in-tree
    gliner-family Provider annotates against its backbone that way), so exempting them is what keeps
    this guard from being a false positive that forces the annotations out.
    """
    found: list[ast.Import | ast.ImportFrom] = []
    _collect_module_scope_backbone_imports(tree, found)
    return found


def jit_import_ordering_offenders(tree: ast.Module) -> list[str]:
    """Every function that imports the deberta backbone **without** first neutralizing the JIT.

    The property the routing guard cannot see. ``load_then_cast`` / ``load_gliner2_optimized`` call
    ``neutralize_torch_jit`` *inside* themselves, i.e. only once the Provider has already evaluated
    ``from gliner2 import GLiNER2`` and transformers has already scripted its deberta_v2 helpers.
    Neither seam lever can undo that (``PYTORCH_JIT`` is read by torch at *torch* import;
    ``jit._state.disable()`` affects only *future* ``script()`` calls), so a Provider that imports
    first is still fully exposed to the Blackwell/CUDA-13 nvrtc crash while passing every
    routing-shaped assertion. This scans each function body for the ordering directly: the earliest
    ``neutralize_torch_jit(...)`` call must precede the earliest backbone import.

    **R3/#190-0 — module scope is scanned too, and admits no ordering repair.** This used to walk
    ``FunctionDef`` bodies ONLY, so the WORST shape was invisible: a backbone import at module level
    lives in no function, produced zero offenders, and passed every #190 guard clean. That import is
    evaluated at **provider-module / entry-point import**, i.e. before any Provider code — hence
    before any ``neutralize_torch_jit()`` call — can run, so the deberta_v2 helpers are already
    ``ScriptFunction``s and neither lever can un-script them. There is no "neutralize first" fix for
    it; the import must be DEFERRED into the loading function (which is also the ADR-0016
    lazy-heavy-dep discipline every in-tree Provider already follows), so it is flagged
    unconditionally.
    """
    offenders: list[str] = [
        f"<module>: imports the deberta backbone at line {node.lineno} at MODULE level "
        f"({ast.unparse(node)}) — evaluated at provider-module/entry-point import, before any "
        "Provider code and therefore before any neutralize_torch_jit() call can run; defer it into "
        "the loading function that neutralizes the JIT first (ADR-0016)"
        for node in _module_scope_backbone_imports(tree)
    ]
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        imports = [n.lineno for n in ast.walk(func) if _imports_deberta_backbone(n)]
        if not imports:
            continue
        first_import = min(imports)
        neutralizations = [
            n.lineno for n in ast.walk(func) if _calls_neutralize_torch_jit(n)
        ]
        if not neutralizations:
            offenders.append(
                f"{func.name}(): imports the deberta backbone at line {first_import} and never "
                "calls neutralize_torch_jit"
            )
        elif min(neutralizations) > first_import:
            offenders.append(
                f"{func.name}(): imports the deberta backbone at line {first_import} but only "
                f"neutralizes the JIT at line {min(neutralizations)} — too late to un-script it"
            )
    return offenders


#: What the ordering guard + the no-hand-rolled-``PYTORCH_JIT`` guard are parametrized over.
#: PRE-FIX this was ``_MDEBERTA_PROVIDER_SOURCES`` (a hardcoded 5-tuple); R2/#190-5 widened it to
#: every DISCOVERED backbone importer, which is what makes the docstrings' "caught the moment its
#: source lands" true rather than aspirational.
_JIT_ORDERING_SCANNED_SOURCES = _DEBERTA_BACKED_SOURCES


def test_every_in_tree_deberta_backed_source_is_scanned_by_the_jit_ordering_guard() -> None:
    """REGRESSION (R2/#190-5): no in-tree DeBERTa backbone importer escapes the #190 ordering scan.

    The defect this bites: ``_MDEBERTA_PROVIDER_SOURCES`` was a literal 5-entry tuple with no
    discovery and no completeness check, so ``latence-disambig-glinker``'s ``backends.py`` — a
    DeBERTa-backed Provider by its own code comment, which imports ``glinker`` inside
    ``_init_executor`` — had ZERO #190 coverage. ``jit_import_ordering_offenders()`` DOES catch a
    mutated (imports-first) version of that file; nothing ever handed it the file. Reordering the
    imports there would have shipped fully green and died with ``libnvrtc-builtins.so.13.0`` on the
    first ``link_mentions`` forward on an sm_120 pod.

    Completeness here is over the SOURCE SET only. The scan's other completeness axis — the SCOPES
    within a source — was the R3/#190-0 defect: module-level imports were not examined at all. That
    is fixed in :func:`jit_import_ordering_offenders` and covered by
    ``test_jit_ordering_guard_flags_a_module_level_backbone_import``.
    """
    discovered = set(_DEBERTA_BACKED_SOURCES)
    assert discovered >= _KNOWN_DEBERTA_BACKED_SOURCES, (
        "the in-tree scan no longer finds DeBERTa-backed sources it used to — a Provider has "
        "silently fallen out of #190 coverage: "
        f"{sorted(_KNOWN_DEBERTA_BACKED_SOURCES - discovered)}"
    )
    unscanned = discovered - set(_JIT_ORDERING_SCANNED_SOURCES)
    assert not unscanned, (
        "these in-tree sources import a DeBERTa backbone but are NOT scanned by the #190 "
        "import-ordering guard, so a neutralize-after-import regression in them ships green and "
        "re-opens the Blackwell/CUDA-13 nvrtc crash:\n" + "\n".join(sorted(unscanned))
    )


def test_deberta_backed_discovery_picks_up_a_newly_landed_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Meta-test: discovery is NON-VACUOUS — a fresh backbone importer is found with no list edit.

    Anti-false-green for the completeness test above: without this, ``_discover_...`` could return a
    frozen/empty set and the assertions would still pass.
    """
    fake_root = tmp_path / "packages"
    src = fake_root / "latence-new-thing/src/latence_new_thing"
    src.mkdir(parents=True)
    (src / "provider.py").write_text(
        "def _load(self):\n    from gliner2 import GLiNER2\n    return GLiNER2\n", encoding="utf-8"
    )
    (src / "unrelated.py").write_text("import json\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "_packages_root", lambda: fake_root)

    assert _discover_deberta_backed_sources() == (
        "latence-new-thing/src/latence_new_thing/provider.py",
    )


@pytest.mark.parametrize("rel_source", _JIT_ORDERING_SCANNED_SOURCES)
def test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone(
    rel_source: str,
) -> None:
    """REGRESSION (#190-4): every mdeberta Provider kills the JIT BEFORE it imports gliner.

    The teeth the routing guard above lacks. It accepts any load handed to a JIT-neutralizing
    loader — but the loader runs *after* ``from gliner2 import GLiNER2`` has already let
    transformers script the deberta_v2 relative-position helpers, which is the instant that
    matters. ``redaction.gliner_pii2`` passed the routing guard while never neutralizing the JIT at
    all; this scan bites on that and on any future Provider that imports the backbone first.

    R3/#190-0: it also bites a MODULE-level backbone import — the shape that used to escape the
    scan entirely because it lives in no function body, and the one shape no ordering can repair.
    """
    path = _packages_root() / rel_source
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    offenders = jit_import_ordering_offenders(tree)
    assert not offenders, (
        f"{rel_source}: the deberta backbone is imported while the TorchScript JIT is still live "
        "— transformers compiles its @torch.jit.script relative-position helpers AT IMPORT, and no "
        "later neutralization can un-script them, so the Blackwell/CUDA-13 nvrtc crash "
        "(libnvrtc-builtins.so.13.0) stays open:\n" + "\n".join(offenders)
    )


def test_jit_ordering_guard_flags_a_seam_routed_provider_that_imports_first() -> None:
    """Meta-test: routing through the neutralizing loader does NOT excuse importing first.

    This is exactly the shape ``latence-pii-gliner2`` had — and exactly what the routing-only guard
    reports as clean.
    """
    bad = (
        "def _load(self):\n"
        "    from gliner2 import GLiNER2\n"
        "    return load_gliner2_optimized(GLiNER2.from_pretrained, mid, dev, cfg)\n"
    )
    assert not jit_seam_offenders(ast.parse(bad)), (
        "precondition: the routing guard considers this source clean — that is the blind spot"
    )
    assert jit_import_ordering_offenders(ast.parse(bad)), (
        "a Provider that imports gliner2 before neutralizing the JIT must be flagged"
    )


def test_jit_ordering_guard_flags_a_module_level_backbone_import() -> None:
    """REGRESSION (R3/#190-0): a MODULE-level backbone import is an offence, seam-routed or not.

    The blind spot this bites: the ordering scan only walked ``FunctionDef`` bodies, so an import
    evaluated at **provider-module / entry-point import** — the one ordering
    ``neutralize_torch_jit()`` can never repair, because transformers has already compiled the six
    ``deberta_v2`` relative-position helpers into real ``ScriptFunction``s before any Provider code
    runs — lived in no function and was never examined. Such a source produced ZERO offenders and
    passed every #190 guard clean while dying with ``libnvrtc-builtins.so.13.0`` on its first
    forward on a Blackwell/CUDA-13 pod.
    """
    bad = (
        "from gliner2 import GLiNER2\n"
        "\n"
        "class NewProvider:\n"
        "    def _load_model(self):\n"
        "        neutralize_torch_jit(provider_name='x.new')\n"
        "        return load_gliner2_optimized(GLiNER2.from_pretrained, mid, dev, cfg)\n"
    )
    assert not jit_seam_offenders(ast.parse(bad)), (
        "precondition: the routing guard considers this source clean (the load IS handed to a "
        "JIT-neutralizing loader) — the module-level import is the blind spot"
    )
    offenders = jit_import_ordering_offenders(ast.parse(bad))
    assert offenders, (
        "a Provider that imports its DeBERTa backbone at MODULE level must be flagged: the import "
        "runs at entry-point discovery, before any neutralize_torch_jit() call can execute"
    )
    assert any("module" in o for o in offenders), (
        f"the offender message must name module scope so the fix is obvious: {offenders}"
    )


def test_jit_ordering_guard_flags_a_class_body_backbone_import() -> None:
    """A class body is module-scope code too — an import there is evaluated at module import."""
    bad = "class P:\n    from gliner import GLiNER\n"
    assert jit_import_ordering_offenders(ast.parse(bad))


def test_jit_ordering_guard_does_not_flag_a_type_checking_only_import() -> None:
    """No false positive: ``if TYPE_CHECKING:`` bodies are never evaluated at runtime.

    Every in-tree gliner-family Provider annotates against its backbone this way; flagging it would
    force the guard to be weakened or the annotations dropped.
    """
    for good in (
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from gliner import GLiNER\n",
        "import typing\n\nif typing.TYPE_CHECKING:\n    from gliner2 import GLiNER2\n",
    ):
        assert not jit_import_ordering_offenders(ast.parse(good)), f"must NOT be flagged: {good!r}"

    # …but the else-branch of that same guard DOES run at import, so it is still scanned.
    bad = (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from gliner import GLiNER\n"
        "else:\n"
        "    from gliner import GLiNER\n"
    )
    assert jit_import_ordering_offenders(ast.parse(bad))


def test_jit_ordering_guard_flags_a_provider_that_never_neutralizes() -> None:
    """Meta-test: no ``neutralize_torch_jit`` call at all in the importing function is an offence."""
    bad = "def _cls(self):\n    import gliner2\n    return gliner2.GLiNER2\n"
    assert jit_import_ordering_offenders(ast.parse(bad))


def test_jit_ordering_guard_accepts_a_correctly_ordered_provider() -> None:
    """Meta-test: neutralize-then-import is clean, and a gliner-free function is not scanned."""
    for good in (
        "def _cls(self):\n"
        "    neutralize_torch_jit(provider_name='p')\n"
        "    from gliner2 import GLiNER2\n"
        "    return GLiNER2\n",
        "def _cls(self):\n"
        "    perf.neutralize_torch_jit()\n"
        "    from gliner import GLiNER\n"
        "    return GLiNER\n",
        "def _helper(self):\n    return self._model.config.max_len\n",
    ):
        assert not jit_import_ordering_offenders(ast.parse(good)), f"must NOT be flagged: {good!r}"


def test_jit_neutralizing_loaders_constant_matches_the_real_perf_seam() -> None:
    """The accept-set is TRUE of ``perf.py``: each named loader calls ``neutralize_torch_jit``…

    …and a perf-seam loader that is NOT in the accept-set does not. This keeps the constant from
    drifting into a lie (the pre-fix state, where it was named in a message but asserted nowhere).
    """
    perf_src = (
        _packages_root() / "latence-core/src/latence_core/providers/perf.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(perf_src)
    bodies = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    def _calls_neutralize(fn_name: str) -> bool:
        fn = bodies.get(fn_name)
        assert fn is not None, f"perf.py has no function {fn_name!r}"
        return any(
            isinstance(c, ast.Call)
            and (
                (isinstance(c.func, ast.Name) and c.func.id == "neutralize_torch_jit")
                or (isinstance(c.func, ast.Attribute) and c.func.attr == "neutralize_torch_jit")
            )
            for c in ast.walk(fn)
        )

    for loader in _JIT_NEUTRALIZING_LOADERS:
        assert _calls_neutralize(loader), (
            f"perf.py:{loader} is listed in _JIT_NEUTRALIZING_LOADERS but does not call "
            "neutralize_torch_jit"
        )
    for loader in set(_PERF_SEAM_FUNCS) - set(_JIT_NEUTRALIZING_LOADERS):
        assert not _calls_neutralize(loader), (
            f"perf.py:{loader} now neutralizes the JIT — add it to _JIT_NEUTRALIZING_LOADERS so "
            "the #190 guard accepts it"
        )


@pytest.mark.parametrize("rel_source", _JIT_ORDERING_SCANNED_SOURCES)
def test_no_provider_hand_rolls_the_pytorch_jit_workaround(rel_source: str) -> None:
    """No Provider sets ``PYTORCH_JIT`` itself — the neutralization lives ONCE, in the perf seam."""
    src = (_packages_root() / rel_source).read_text(encoding="utf-8")
    assert "PYTORCH_JIT" not in src, (
        f"{rel_source}: the PYTORCH_JIT workaround must live in latence_core.providers.perf "
        "(neutralize_torch_jit), not be copied per Provider"
    )

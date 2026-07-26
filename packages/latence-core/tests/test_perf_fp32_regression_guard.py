"""The residue of the fp32/ordering AST linter — one structural rule, DISCOVERED, not listed.

**What this file used to be.** 829 lines of AST scanning that parsed every learned Provider's
SOURCE to assert that seven perf-seam calls were present and correctly sequenced: neutralize the JIT
*before* the backbone import, check the DeviceDecision, ``load_then_cast``, ``place_on_device``,
``apply_gliner_max_len`` *before* ``maybe_compile``. Five real "before" constraints, none of them
expressible in a signature, all of them enforced by linting the callers.

**Why it is gone.** When the only way to hold an invariant is to lint the callers, the invariant
belongs inside the module. It now does: :func:`~latence_core.providers.perf.load_gliner_model`,
:func:`~latence_core.providers.perf.load_gliner2_model` and
:func:`~latence_core.providers.perf.load_transformers_model` each own the WHOLE sequence, and the
backbone import is a factory they call — so "neutralize before import" is a straight line of code a
caller cannot reorder because a caller never sees it. The sequence is tested where it now lives, by
behaviour, through the interface: ``test_perf_load_choreography.py`` drives it with a recording fake
that asserts the observed ORDER of the real effects (and a fake that REJECTS the wrong order, so the
assertions cannot pass vacuously). The end-to-end proof that the JIT is dead at the moment the real
backbone module is imported stays in ``test_perf_jit_import_ordering.py``, which was always
behavioural.

**What is left here, and why it is worth keeping.** Exactly one rule, because it is the one thing a
NEW Provider can still get wrong: *a learned model's* ``from_pretrained`` *may only be referenced
inside the perf seam*. A Provider that writes ``GLiNER.from_pretrained(model_id)`` — or hands it to
some other helper — has hand-rolled a load again, and with it the choreography. The rule needs no
ordering analysis, no accept-list of loaders, and no list of Providers: it is DISCOVERED by globbing
``packages/*/src``, which is the fix the #190 guards already had and this scan did not. That
asymmetry is what let ``sparse.splade`` — a registered learned Provider doing real transformer
forwards, loading bare fp32 with no placement — sit outside the guard's curated
``_LEARNED_PROVIDER_SOURCES`` tuple indefinitely.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# A ``<X>.from_pretrained`` whose attribute chain matches one of these is a MODEL load, which must
# go through the perf seam. Processors / tokenizers / feature extractors deliberately do NOT match
# (``AutoProcessor``, ``AutoTokenizer``, ``LightOnOcrProcessor``): they take no perf kwargs, are
# never placed on a device, and loading one in a Provider is correct.
_MODEL_CALLEE_HINTS = ("gliner", "forconditionalgeneration", "automodel")

# The one module allowed to reference a model ``from_pretrained``: the perf seam itself, where the
# whole ordered sequence lives. (The path is repo-relative under ``packages/``.)
_PERF_SEAM_SOURCE = "latence-core/src/latence_core/providers/perf.py"

# The three deep entry points. A source that loads a learned model must call one of them.
_LOAD_ENTRY_POINTS = ("load_gliner_model", "load_gliner2_model", "load_transformers_model")

# Sources known to load a learned model when this guard was written. Discovery must never return
# LESS than this — a shrinking scan is how a Provider silently falls out of coverage. ``splade`` is
# in the list precisely because its absence from the OLD curated list is the defect this rewrite
# closes.
_KNOWN_LEARNED_LOAD_SOURCES = frozenset(
    {
        "latence-ner-gliner/src/latence_ner_gliner/provider.py",
        "latence-relation-gliner/src/latence_relation_gliner/provider.py",
        "latence-pii-gliner/src/latence_pii_gliner/provider.py",
        "latence-pii-gliner2/src/latence_pii_gliner2/provider.py",
        "latence-extract-gliner2/src/latence_extract_gliner2/provider.py",
        "latence-parser-lighton/src/latence_parser_lighton/provider.py",
        "latence-parser-glm/src/latence_parser_glm/provider.py",
        "latence-splade/src/latence_splade/model.py",
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


def _in_tree_sources() -> list[tuple[str, ast.Module]]:
    """Every in-tree package source, parsed — the discovery the old curated tuple lacked."""
    root = _packages_root()
    out: list[tuple[str, ast.Module]] = []
    for path in sorted(root.glob("*/src/**/*.py")):
        text = path.read_text(encoding="utf-8")
        out.append((path.relative_to(root).as_posix(), ast.parse(text, filename=str(path))))
    return out


def model_from_pretrained_refs(tree: ast.Module) -> list[str]:
    """Every ``<learned model>.from_pretrained`` REFERENCE in ``tree``, as ``line: expr`` strings.

    References, not calls, because both offending shapes reduce to one: a direct
    ``GLiNER.from_pretrained(model_id)`` carries the attribute as its ``func``, and a load handed to
    some helper (``some_loader(GLiNER.from_pretrained, ...)``) passes the very same attribute node
    as an argument. Matching the attribute expression catches both, plus the indirected class handle
    (``gliner2_cls.from_pretrained``, ``self._gliner2_cls().from_pretrained``) that a Provider used
    to reach for.
    """
    refs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "from_pretrained":
            continue
        expr = ast.unparse(node)
        if any(hint in expr.lower() for hint in _MODEL_CALLEE_HINTS):
            refs.append(f"line {node.lineno}: {expr}")
    return refs


def _calls_a_load_entry_point(tree: ast.Module) -> bool:
    """Whether ``tree`` calls one of the three deep load entry points."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else (func.attr if isinstance(func, ast.Attribute) else "")
        )
        if name in _LOAD_ENTRY_POINTS:
            return True
    return False


def _discover_learned_load_sources() -> tuple[str, ...]:
    """Every in-tree source that loads a learned model — DISCOVERED by glob, never listed.

    A source qualifies if it calls one of the three deep entry points OR still references a model
    ``from_pretrained`` itself (the offending shape). Both are counted so a Provider cannot leave
    coverage by regressing: dropping the entry-point call while keeping a hand-rolled load keeps it
    firmly in the scan below.
    """
    return tuple(
        rel
        for rel, tree in _in_tree_sources()
        if rel != _PERF_SEAM_SOURCE
        and (_calls_a_load_entry_point(tree) or model_from_pretrained_refs(tree))
    )


_LEARNED_LOAD_SOURCES = _discover_learned_load_sources()


def test_discovery_covers_every_known_learned_load_source() -> None:
    """REGRESSION: the scan is DISCOVERED, and it never shrinks below what we know exists.

    The defect this bites is the shape of the old file: ``_LEARNED_PROVIDER_SOURCES`` was a curated
    8-entry tuple, so ``sparse.splade`` — a registered learned Provider running real
    ``AutoModelForMaskedLM`` forwards — was never scanned at all, and shipped a bare fp32
    ``from_pretrained`` with no device placement while every guard in this file was green.
    """
    discovered = set(_LEARNED_LOAD_SOURCES)
    missing = _KNOWN_LEARNED_LOAD_SOURCES - discovered
    assert not missing, (
        "these sources load a learned model but the in-tree scan no longer finds them — a Provider "
        f"has silently fallen out of coverage: {sorted(missing)}"
    )


def test_discovery_picks_up_a_newly_landed_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-false-green: discovery is NON-VACUOUS — a fresh learned load is found, no list edit.

    Without this, ``_discover_learned_load_sources`` could return a frozen or empty tuple and the
    assertions above and below would still pass.
    """
    fake_root = tmp_path / "packages"
    src = fake_root / "latence-new-thing/src/latence_new_thing"
    src.mkdir(parents=True)
    (src / "provider.py").write_text(
        "def _load(self):\n    return GLiNER2.from_pretrained(self._model_id)\n", encoding="utf-8"
    )
    (src / "unrelated.py").write_text("import json\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "_packages_root", lambda: fake_root)

    assert _discover_learned_load_sources() == (
        "latence-new-thing/src/latence_new_thing/provider.py",
    )


@pytest.mark.parametrize("rel_source", _LEARNED_LOAD_SOURCES)
def test_no_provider_references_a_model_from_pretrained(rel_source: str) -> None:
    """A learned model's ``from_pretrained`` lives ONLY in the perf seam — nowhere else in-tree.

    The single structural rule that survives the linter's deletion, and the only one still worth
    lint-shaped teeth: a Provider that names ``<Model>.from_pretrained`` has hand-rolled a load, and
    a hand-rolled load is a hand-rolled choreography — which is what produced #190 (JIT killed after
    the backbone import), #191 (placement wired to one Provider of three) and the OCR parsers' bare
    ``model.to(device)``. Handing the seam an import FACTORY instead is what makes those unreachable.

    Note what this rule does NOT need: no accept-list of loaders, no ordering analysis, no list of
    Providers. That is the difference between an invariant that lives in the type system's blind
    spot and one that lives in a single module.
    """
    path = _packages_root() / rel_source
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = model_from_pretrained_refs(tree)
    assert not offenders, (
        f"{rel_source}: this source loads a learned model itself instead of through the perf seam "
        f"({', '.join(_LOAD_ENTRY_POINTS)}). Hand the entry point a zero-arg factory that returns "
        "the model class; the seam then kills the TorchScript JIT, caps the intra-op threads, "
        "imports the backbone, loads, PLACES + verifies, bounds the window and compiles, in that "
        "order:\n" + "\n".join(offenders)
    )


def test_the_perf_seam_itself_is_where_model_loads_live() -> None:
    """Anti-false-green for the rule above: it is a RELOCATION, not an eradication.

    If ``perf.py`` referenced no model ``from_pretrained`` either, the parametrized assertion could
    be trivially satisfied by a codebase that loads nothing at all. It does load: the three entry
    points call ``model_cls.from_pretrained`` through the family-appropriate helper.
    """
    perf = _packages_root() / _PERF_SEAM_SOURCE
    tree = ast.parse(perf.read_text(encoding="utf-8"), filename=str(perf))
    calls = {
        ast.unparse(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and node.args
    }
    assert any("from_pretrained" in expr for expr in calls), (
        "perf.py must be the place a model from_pretrained is actually called — otherwise the "
        "no-from_pretrained-in-Providers rule is vacuous"
    )


def test_guard_flags_a_hand_rolled_load_and_accepts_a_processor_load() -> None:
    """Meta-test: the rule bites every offending shape and no legitimate one."""
    for bad in (
        "m = GLiNER.from_pretrained(model_id)\n",
        "m = GLiNER2.from_pretrained(model_id, quantize=True)\n",
        "m = gliner2_cls.from_pretrained(model_id)\n",
        "m = self._gliner2_cls().from_pretrained(model_id)\n",
        "m = AutoModelForMaskedLM.from_pretrained(model_id)\n",
        "m = some_loader(LightOnOcrForConditionalGeneration.from_pretrained, mid, perf)\n",
    ):
        assert model_from_pretrained_refs(ast.parse(bad)), f"must be flagged: {bad!r}"
    for good in (
        # Processors / tokenizers take no perf kwargs and are never placed — a Provider loads them.
        "p = AutoProcessor.from_pretrained(model_id)\n",
        "t = AutoTokenizer.from_pretrained(model_id)\n",
        "p = LightOnOcrProcessor.from_pretrained(model_id)\n",
        # The seam-routed shape: the Provider hands over a factory, never the load.
        "m = load_gliner_model(_import_gliner, mid, cfg, decision, provider_name='p')\n",
    ):
        assert not model_from_pretrained_refs(ast.parse(good)), f"must NOT be flagged: {good!r}"


def test_no_provider_hand_rolls_the_pytorch_jit_workaround() -> None:
    """No Provider sets ``PYTORCH_JIT`` itself — the neutralization lives ONCE, in the perf seam."""
    offenders = [
        rel
        for rel, _tree in _in_tree_sources()
        if rel != _PERF_SEAM_SOURCE
        and "PYTORCH_JIT" in (_packages_root() / rel).read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "the PYTORCH_JIT workaround must live in latence_core.providers.perf "
        f"(neutralize_torch_jit), not be copied per Provider: {offenders}"
    )


def test_no_provider_hand_rolls_the_cuda_cache_reclaim() -> None:
    """REGRESSION (d): nobody reimplements ``empty_cuda_cache``'s gc-then-empty_cache ordering.

    ``disambiguation.glinker`` carried a line-for-line copy of
    :func:`~latence_core.providers.perf.empty_cuda_cache`, down to the ``gc.collect()``-first
    ordering — and that ordering is the load-bearing part (a torch ``nn.Module`` is held alive by a
    reference cycle, so ``empty_cache()`` alone reclaims nothing). A second copy of a subtle
    ordering is a second place for it to drift out of step with the one that was reasoned about.
    """
    offenders: list[str] = []
    for rel, tree in _in_tree_sources():
        if rel == _PERF_SEAM_SOURCE:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(
                "torch.cuda.empty_cache"
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "call latence_core.providers.perf.empty_cuda_cache instead of re-deriving the "
        f"gc.collect()-then-torch.cuda.empty_cache() ordering: {offenders}"
    )


def test_sentence_transformers_construction_threads_model_kwargs() -> None:
    """The ST embedder constructs ``SentenceTransformer`` through the FA2-fallback loader.

    ``sentence-transformers`` is neither a ``from_pretrained`` family nor a device-placing one (the
    library places its own model from a ``device=`` argument), so it stays on the step-level
    ``load_with_fa2_fallback`` seam rather than a deep entry point — asserted here so that
    exemption is a recorded decision rather than an omission.
    """
    path = _packages_root() / "latence-embedder-st/src/latence_embedder_st/provider.py"
    src = path.read_text(encoding="utf-8")
    assert "load_with_fa2_fallback" in src, "ST embedder must load through the shared perf seam"
    assert "model_kwargs" in src, "ST embedder must thread perf kwargs via model_kwargs"

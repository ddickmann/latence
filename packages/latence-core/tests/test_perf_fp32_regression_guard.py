"""The residue of the fp32/ordering AST linter — two structural rules, DISCOVERED, not listed.

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

**What is left here, and why it is worth keeping.** Exactly two rules, because they are the two
things a NEW Provider can still get wrong — and *neither is expressible inside the seam*, because
both are properties of the CALLER's source text, which the seam by construction never sees.

1. *A learned model's* ``from_pretrained`` *may only be referenced inside the perf seam.* A Provider
   that writes ``GLiNER.from_pretrained(model_id)`` — or hands it to some other helper — has
   hand-rolled a load again, and with it the choreography.

2. *A DeBERTa backbone may only be imported after the JIT is dead* (#190). The seam owns the
   ordering of the effects it performs; it does NOT own **where the caller wrote the import**. The
   factory contract makes ``_import_backbone`` the natural place for that import — but a Provider
   can still write ``from gliner import GLiNER`` at MODULE scope and hand the seam
   ``lambda: GLiNER``, which type-checks, satisfies rule 1, drives the whole recorded choreography
   in ``test_perf_load_choreography.py``, and yet has already scripted transformers' six
   ``deberta_v2`` relative-position helpers at provider-module import — before any Provider code,
   therefore before ``neutralize_torch_jit()`` can run, and neither of its levers can un-script them
   (``PYTORCH_JIT`` is read by torch at *torch* import; ``jit._state.disable()`` affects only
   *future* ``script()`` calls). The Blackwell/CUDA-13 ``libnvrtc-builtins.so.13.0`` crash, shipped
   green. This rule is the caller-side half the seam cannot absorb, so it stays lint-shaped.

Neither rule needs an ordering analysis of the seam's internals, an accept-list of loaders, or a
list of Providers: both are DISCOVERED by globbing ``packages/*/src``. That is load-bearing, and it
is what a curated tuple has twice failed to deliver here — ``_LEARNED_PROVIDER_SOURCES`` left
``sparse.splade`` (a registered learned Provider doing real transformer forwards, loading bare fp32
with no placement) unscanned indefinitely, and the hand-written 5-tuple that preceded
``_DEBERTA_BACKED_SOURCES`` left ``disambiguation.glinker`` with ZERO #190 coverage (R2/#190-5).
The behavioural end-to-end proof in ``test_perf_jit_import_ordering.py`` is a complement, not a
substitute: it needs a hand-written load driver per Provider, so its ``_PROVIDERS`` list is
necessarily curated and by construction cannot see a source that no one added to it.
"""

from __future__ import annotations

import ast
import pathlib
from typing import TypeGuard

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


# --- Rule 2 (#190): the DeBERTa backbone is imported only AFTER the JIT is dead -------------------

#: Import roots whose evaluation transitively pulls in transformers'
#: ``models.deberta_v2.modeling_deberta_v2`` — the module whose six relative-position helpers carry
#: ``@torch.jit.script`` and are therefore compiled into real ``ScriptFunction``s **at import**.
#: ``transformers`` is here and not in :data:`_DEBERTA_BACKBONE_ROOTS` on purpose: importing it is
#: an offence *inside a source already known to be DeBERTa-backed*, but is not by itself evidence
#: that a source is (every tokenizer-only Provider imports transformers).
_DEBERTA_IMPORT_ROOTS = ("gliner", "gliner2", "glinker", "transformers")

#: The roots whose import makes a source DeBERTa-backed, i.e. in scope for the ordering rule.
_DEBERTA_BACKBONE_ROOTS = ("gliner", "gliner2", "glinker")


def _import_roots(node: ast.AST, roots: tuple[str, ...]) -> TypeGuard[ast.Import | ast.ImportFrom]:
    """Whether ``node`` is an import statement whose top-level module is one of ``roots``."""
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] in roots
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in roots for alias in node.names)
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
        return  # a function body runs on call, not at import — the per-function scan owns it
    if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
        for stmt in node.orelse:  # the ELSE branch is the one that actually runs at import
            _collect_module_scope_backbone_imports(stmt, found)
        return
    if _import_roots(node, _DEBERTA_IMPORT_ROOTS):
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


def _seam_routed_factory_names(tree: ast.Module) -> frozenset[str]:
    """Names of the backbone-import factories ``tree`` HANDS to a deep load entry point.

    ``load_gliner_model(_import_gliner, ...)`` / ``load_gliner2_model(self._gliner2_cls, ...)``.
    Such a function is reached ONLY through the seam, which neutralizes the JIT before calling it —
    so requiring it to call ``neutralize_torch_jit`` itself would be demanding back exactly the
    per-Provider copy the seam was built to delete. Its ordering is owned by
    :mod:`latence_core.providers.perf` and proven there behaviourally
    (``test_perf_load_choreography.py``, ``test_perf_jit_import_ordering.py``), so the lexical rule
    below steps aside for it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        callee = (
            func.id
            if isinstance(func, ast.Name)
            else (func.attr if isinstance(func, ast.Attribute) else "")
        )
        if callee not in _LOAD_ENTRY_POINTS:
            continue
        factory = node.args[0]
        if isinstance(factory, ast.Name):
            names.add(factory.id)
        elif isinstance(factory, ast.Attribute):
            names.add(factory.attr)
    return frozenset(names)


def jit_import_ordering_offenders(tree: ast.Module) -> list[str]:
    """Every place in ``tree`` that imports the DeBERTa backbone without first killing the JIT.

    Two shapes, because there are two ways to be too late — and the rule is deliberately NARROWER
    than the linter it is restored from, because the seam now genuinely owns part of what that
    linter checked:

    * **Module scope (incl. class bodies)** — the import is evaluated at provider-module /
      entry-point import, i.e. before any Provider code and therefore before any
      ``neutralize_torch_jit()`` call can run. No runtime seam, however deep, can precede it, and
      there is no "neutralize first" repair: the import must be DEFERRED into a function (which is
      also the ADR-0016 lazy-heavy-dep discipline every in-tree Provider already follows). Flagged
      unconditionally. This is the shape the factory contract *invites* — hoist the import, hand the
      seam ``lambda: GLiNER`` — and the shape no behavioural, signature-level or choreography-level
      check can see.
    * **Inside a NON-seam-routed function** — for a function the seam does not call (the factories
      in :func:`_seam_routed_factory_names` are exempt), the earliest ``neutralize_torch_jit(...)``
      must precede the earliest backbone import. This is what still covers
      ``disambiguation.glinker``, the one in-tree DeBERTa-backed Provider that loads no
      ``from_pretrained`` model and therefore calls the shared seam directly from
      ``_init_executor`` rather than routing a factory through it.
    """
    offenders: list[str] = [
        f"<module>: imports the deberta backbone at line {node.lineno} at MODULE level "
        f"({ast.unparse(node)}) — evaluated at provider-module/entry-point import, before any "
        "Provider code and therefore before any neutralize_torch_jit() call can run; defer it into "
        "the loading function (or the zero-arg factory the perf seam calls), which neutralizes the "
        "JIT first (ADR-0016)"
        for node in _module_scope_backbone_imports(tree)
    ]
    seam_routed = _seam_routed_factory_names(tree)
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if func.name in seam_routed:
            continue  # the seam calls it, and the seam neutralizes first
        imports = [n.lineno for n in ast.walk(func) if _import_roots(n, _DEBERTA_IMPORT_ROOTS)]
        if not imports:
            continue
        first_import = min(imports)
        neutralizations = [n.lineno for n in ast.walk(func) if _calls_neutralize_torch_jit(n)]
        if not neutralizations:
            offenders.append(
                f"{func.name}(): imports the deberta backbone at line {first_import}, never calls "
                "neutralize_torch_jit, and is not handed to a perf-seam load entry point "
                f"({', '.join(_LOAD_ENTRY_POINTS)}) as an import factory"
            )
        elif min(neutralizations) > first_import:
            offenders.append(
                f"{func.name}(): imports the deberta backbone at line {first_import} but only "
                f"neutralizes the JIT at line {min(neutralizations)} — too late to un-script it"
            )
    return offenders


def _discover_deberta_backed_sources() -> tuple[str, ...]:
    """Every in-tree source that imports a DeBERTa backbone — DISCOVERED by glob, never listed.

    The scan the perf seam cannot replace. A source qualifies on the mere presence of a
    ``gliner`` / ``gliner2`` / ``glinker`` import at any nesting depth, so a new backbone importer
    is in scope the moment its source lands — no list to edit, and no way to be forgotten.
    """
    return tuple(
        rel
        for rel, tree in _in_tree_sources()
        if rel != _PERF_SEAM_SOURCE
        and any(_import_roots(node, _DEBERTA_BACKBONE_ROOTS) for node in ast.walk(tree))
    )


_DEBERTA_BACKED_SOURCES = _discover_deberta_backed_sources()

#: The DeBERTa-backed sources known to exist when this guard was written. Discovery must never
#: return LESS than this. ``disambiguation.glinker`` is listed for the same reason ``splade`` is
#: listed above: it is the source a hand-written list left uncovered (R2/#190-5), and it is the one
#: in-tree DeBERTa-backed Provider that loads NO ``from_pretrained`` model at all — so rule 1 alone
#: can never see it, and only this discovery keeps it inside a structural scan.
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


def test_deberta_backed_discovery_covers_every_known_source() -> None:
    """REGRESSION (R2/#190-5): no in-tree DeBERTa backbone importer escapes the ordering scan.

    The defect this bites is a curated list: the 5-tuple that preceded discovery simply omitted
    ``latence-disambig-glinker`` — whose ``GLinkerBackend._init_executor`` imports ``glinker`` — so
    the sixth DeBERTa-backed Provider had ZERO structural #190 coverage. It is also the Provider
    that rule 1 structurally cannot reach (it references no model ``from_pretrained``).
    """
    discovered = set(_DEBERTA_BACKED_SOURCES)
    missing = _KNOWN_DEBERTA_BACKED_SOURCES - discovered
    assert not missing, (
        "these sources import a DeBERTa backbone but the in-tree scan no longer finds them — a "
        f"Provider has silently fallen out of #190 coverage: {sorted(missing)}"
    )


def test_deberta_backed_discovery_picks_up_a_newly_landed_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-false-green: discovery is NON-VACUOUS — a fresh backbone importer is found, no list edit.

    Without this, ``_discover_deberta_backed_sources`` could return a frozen or empty tuple and the
    parametrized ordering assertion below would pass over nothing at all.
    """
    fake_root = tmp_path / "packages"
    src = fake_root / "latence-new-backbone/src/latence_new_backbone"
    src.mkdir(parents=True)
    (src / "provider.py").write_text(
        "def _load(self):\n"
        "    neutralize_torch_jit()\n"
        "    from gliner2 import GLiNER2\n"
        "    return GLiNER2\n",
        encoding="utf-8",
    )
    (src / "unrelated.py").write_text("import json\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "_packages_root", lambda: fake_root)

    assert _discover_deberta_backed_sources() == (
        "latence-new-backbone/src/latence_new_backbone/provider.py",
    )


@pytest.mark.parametrize("rel_source", _DEBERTA_BACKED_SOURCES)
def test_deberta_backed_sources_neutralize_the_jit_before_importing_the_backbone(
    rel_source: str,
) -> None:
    """REGRESSION (#190): the backbone import never runs while the TorchScript JIT is still live.

    The caller-side half of #190, which the perf seam cannot own however deep it gets: the seam
    sequences the effects it performs, but the Provider decides *where the import statement lives*.
    Hoisting it to module scope and handing the seam ``lambda: GLiNER`` satisfies every other guard
    in the repo — rule 1 (no ``from_pretrained`` reference), the recorded-order choreography suite,
    and the behavioural ``test_perf_jit_import_ordering.py`` (whose ``_PROVIDERS`` list is curated
    and cannot contain a source nobody added) — while transformers' ``deberta_v2`` helpers are
    already ``ScriptFunction``s and die in nvrtc on the first forward on Blackwell/CUDA-13.
    """
    path = _packages_root() / rel_source
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = jit_import_ordering_offenders(tree)
    assert not offenders, (
        f"{rel_source}: the DeBERTa backbone is imported while the TorchScript JIT is still live "
        "— transformers' six deberta_v2 relative-position helpers are compiled by @torch.jit.script "
        "AT IMPORT and neutralize_torch_jit() can never un-script them (#190, the Blackwell/CUDA-13 "
        "libnvrtc-builtins.so.13.0 crash):\n" + "\n".join(offenders)
    )


def test_jit_ordering_guard_flags_the_factory_shaped_module_level_hoist() -> None:
    """Meta-test: the exact refactor the factory seam invites is caught.

    ``load_gliner_model`` takes a zero-arg factory, so ``lambda: GLiNER`` over a module-level import
    is the shortest thing that compiles — and it is the #190 bug. There is no in-function
    neutralization that repairs it, hence the unconditional flag.
    """
    bad = (
        "from gliner import GLiNER\n"
        "class P:\n"
        "    def _load_model(self):\n"
        "        return load_gliner_model(lambda: GLiNER, mid, cfg, decision, provider_name='p')\n"
    )
    offenders = jit_import_ordering_offenders(ast.parse(bad))
    assert offenders, "a module-level backbone import must be flagged"
    assert "MODULE level" in offenders[0]
    # ...and rule 1 is blind to it, which is precisely why rule 2 has to exist.
    assert not model_from_pretrained_refs(ast.parse(bad))


def test_jit_ordering_guard_flags_a_class_body_and_a_transformers_hoist() -> None:
    """Class bodies and a bare ``transformers`` hoist run at import too."""
    for bad in (
        "class P:\n    from gliner import GLiNER\n",
        "import transformers\n\ndef _load(self):\n    from gliner import GLiNER\n",
    ):
        assert jit_import_ordering_offenders(ast.parse(bad)), f"must be flagged: {bad!r}"


def test_jit_ordering_guard_flags_an_in_function_ordering_inversion() -> None:
    """Import-then-neutralize, and never-neutralize, are both too late."""
    for bad in (
        "def _cls(self):\n    from gliner2 import GLiNER2\n    neutralize_torch_jit()\n    return GLiNER2\n",
        "def _cls(self):\n    import gliner2\n    return gliner2.GLiNER2\n",
    ):
        assert jit_import_ordering_offenders(ast.parse(bad)), f"must be flagged: {bad!r}"


def test_the_seam_routed_exemption_is_narrow_and_still_flags_the_inversion() -> None:
    """The factory exemption is not a blanket escape hatch.

    It applies to the function actually HANDED to a deep entry point, and it never rescues the two
    shapes the seam cannot fix: a module-level hoist, or a same-function import-then-neutralize.
    A namesake that is never passed to the seam is not exempt either.
    """
    routed = (
        "def _import_gliner():\n    from gliner import GLiNER\n    return GLiNER\n"
        "def _load(self):\n"
        "    return load_gliner_model(_import_gliner, mid, cfg, d, provider_name='p')\n"
    )
    assert not jit_import_ordering_offenders(ast.parse(routed))
    assert _seam_routed_factory_names(ast.parse(routed)) == frozenset({"_import_gliner"})

    # Same factory body, but nothing hands it to the seam: no exemption.
    unrouted = "def _import_gliner():\n    from gliner import GLiNER\n    return GLiNER\n"
    assert jit_import_ordering_offenders(ast.parse(unrouted))

    # Routed through the NON-neutralizing step-level loader: not an entry point, no exemption.
    fa2 = (
        "def _import_gliner():\n    from gliner import GLiNER\n    return GLiNER\n"
        "def _load(self):\n    return load_with_fa2_fallback(_import_gliner, mid)\n"
    )
    assert jit_import_ordering_offenders(ast.parse(fa2))

    # The exemption never reaches module scope, even with a routed factory present.
    hoisted = "from gliner import GLiNER\n" + routed
    assert any("MODULE level" in o for o in jit_import_ordering_offenders(ast.parse(hoisted)))


def test_jit_ordering_guard_accepts_correct_orderings_and_type_checking_imports() -> None:
    """Anti-false-positive: the rule is not simply "never name gliner"."""
    for good in (
        "def _cls(self):\n    neutralize_torch_jit()\n    from gliner import GLiNER\n    return GLiNER\n",
        "def _cls(self):\n    perf.neutralize_torch_jit(provider_name='p')\n    import glinker\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from gliner import GLiNER\n",
        "def _helper(self):\n    return self._model.config.max_len\n",
    ):
        assert not jit_import_ordering_offenders(ast.parse(good)), f"must NOT be flagged: {good!r}"


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


def test_sentence_transformers_loads_through_the_st_composite() -> None:
    """The ST embedder loads through the deep ST entry point, never piecemeal steps (Wave 3-H).

    This REVERSES the pre-3-H recorded exemption ("ST stays on the step-level
    ``load_with_fa2_fallback`` seam"): assembling the steps by hand in the Provider was re-opening
    the C4 ordering by hand, and ST was the ONE learned family without a composite. The Provider
    now makes exactly one perf-seam call — :func:`load_sentence_transformer` — and the FA2
    fallback + ``model_kwargs`` threading live inside that composite in ``perf.py``.
    """
    provider_path = _packages_root() / "latence-embedder-st/src/latence_embedder_st/provider.py"
    provider_src = provider_path.read_text(encoding="utf-8")
    assert "load_sentence_transformer" in provider_src, (
        "ST embedder must load through the deep ST composite"
    )
    assert "load_with_fa2_fallback" not in provider_src, (
        "no piecemeal step assembly in the Provider — the composite owns the sequence"
    )
    perf_path = _packages_root() / "latence-core/src/latence_core/providers/perf.py"
    perf_src = perf_path.read_text(encoding="utf-8")
    composite = perf_src[perf_src.index("def load_sentence_transformer") :]
    assert "load_with_fa2_fallback" in composite, "the composite owns the FA2→sdpa fallback"
    assert "model_kwargs=mk" in composite, "the composite threads perf kwargs via ST model_kwargs"

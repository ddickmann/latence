# R3 — C4 deleted the discovered #190 JIT import-ordering guard and replaced it with nothing

- **Severity:** major
- **Dimension:** test quality / false green; deepening that removed a real proof
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/tests/test_perf_fp32_regression_guard.py`
  (the C4 rewrite: 736 lines deleted, 208 added — `jit_import_ordering_offenders()`,
  `_discover_deberta_backed_sources()` and `_DEBERTA_BACKED_SOURCES` all gone) and
  `packages/latence-core/tests/test_perf_jit_import_ordering.py:_PROVIDERS`
  (the hand-written 6-entry behavioural list left as the only remaining proof)
- **Claim contradicted:** the rewritten module docstring —
  *"it is DISCOVERED by globbing `packages/*/src`, which is the fix the #190 guards already had and
  this scan did not"*. C4 kept the #190 guards' discovery **idea** and deleted the #190 guards.
- **Invariants at stake:** #190 (the Blackwell / CUDA-13 `libnvrtc-builtins.so.13.0` production
  kill), ADR-0016 (heavy deps stay lazy — a module-scope backbone import violates it directly),
  R2/#190-5 (a curated list of Providers is not coverage)
- **Status:** FIXED (this commit)

## What was wrong

C4 replaced the fp32/ordering AST linter with a single structural rule —
`model_from_pretrained_refs()`: *a learned model's `from_pretrained` may only be referenced inside
the perf seam*. That rule is right, and the deepening behind it (the three deep entry points
`load_gliner_model` / `load_gliner2_model` / `load_transformers_model` owning the whole ordered
sequence, with the backbone import handed in as a zero-arg **factory**) is right too. Both stay.

But #190 has two halves, and only one of them moved into the seam.

The seam owns the ordering of the effects **it performs**. It does not own — and by construction
cannot see — **where the caller wrote the import statement**. Once the entry point takes a factory,
the shortest thing that compiles is:

```python
from gliner import GLiNER            # module scope

def _load_model(self):
    return load_gliner_model(lambda: GLiNER, mid, cfg, decision, provider_name="entity.new")
```

That is evaluated at **provider-module import** — before any Provider code, therefore before
`neutralize_torch_jit()` can run. Neither of its levers can repair it afterwards (`PYTORCH_JIT` is
read by torch at *torch* import; `jit._state.disable()` affects only *future* `script()` calls), so
transformers' six `deberta_v2` relative-position helpers stay real `ScriptFunction`s and die in
nvrtc on the first forward.

Main caught it. `jit_import_ordering_offenders()` was parametrized over `_DEBERTA_BACKED_SOURCES`,
a glob-discovered set — the R2/#190-5 fix that existed precisely because the hand-written 5-tuple
before it had left `disambiguation.glinker` with zero #190 coverage. C4 deleted both, leaving the
curated `_PROVIDERS` list in `test_perf_jit_import_ordering.py` as the only remaining proof: six
entries, each needing a hand-written load driver, so it can only ever contain a Provider somebody
remembered to add. That is the R2/#190-5 blind spot, re-opened.

Second-order consequence: `latence-disambig-glinker/.../backends.py` fell out of **every**
structural scan. It is the one in-tree DeBERTa-backed Provider that loads no `from_pretrained`
model at all (it drives `glinker`'s `ProcessorFactory` and calls the shared seam directly from
`_init_executor`), so C4's surviving rule is structurally incapable of seeing it, and C4's discovery
predicate — "calls a load entry point OR references a model `from_pretrained`" — does not select it.

## Failure scenario (reproduced)

`packages/latence-ner-gliner/src/latence_ner_gliner/_probe_new_provider.py`, the exact shape above.
Pre-fix:

```
$ make test
3868 passed, 29 skipped, 3 warnings in 560.38s (0:09:20)
```

Full suite **green**, with a module-scope DeBERTa backbone import shipping. The three #190-adjacent
suites specifically (`test_perf_fp32_regression_guard.py`, `test_perf_jit_import_ordering.py`,
`test_perf_load_choreography.py`): `40 passed`. Against the same source, main's deleted rule
returns:

```
['<module>: imports the deberta backbone at line 2 at MODULE level (from gliner import GLiNER)
  — evaluated at provider-module/entry-point import, before any Provider code and therefore before
  any neutralize_torch_jit() call can run; ...']
```

while C4's surviving rule returns `[]` (there is no `.from_pretrained` reference to find), and
`_LEARNED_LOAD_SOURCES` = 8 sources, none of them the disambig-glinker backend.

## What was built

The caller-side half of #190 is restored as a structural rule in
`test_perf_fp32_regression_guard.py` — **discovered by glob, never listed** — and it is
deliberately **narrower** than the linter it comes from, because part of what that linter checked
genuinely did move into the seam:

- `_discover_deberta_backed_sources()` globs `packages/*/src/**/*.py` for any `gliner` / `gliner2` /
  `glinker` import at any nesting depth. Six sources today, including the disambig-glinker backend.
  `_KNOWN_DEBERTA_BACKED_SOURCES` is the floor the scan may never fall below.
- `jit_import_ordering_offenders()` flags two shapes:
  - **module scope / class body**, unconditionally — no runtime seam can precede it, so there is no
    "neutralize first" repair; the import must be deferred into a function (ADR-0016). `TYPE_CHECKING`
    bodies are exempt, since every in-tree Provider annotates against its backbone that way.
  - **inside a NON-seam-routed function** — neutralize-before-import ordering, which is what covers
    `disambiguation.glinker`'s `_init_executor`.
- `_seam_routed_factory_names()` is what keeps the fix from undoing the deepening. A function handed
  to a deep entry point as the import factory (`load_gliner2_model(self._gliner2_cls, ...)`) is
  **exempt** from the in-function ordering rule: the seam calls it and neutralizes first, so
  demanding a `neutralize_torch_jit()` inside it would be demanding back exactly the per-Provider
  copy the seam was built to delete. Porting main's rule unchanged false-positived on all five such
  factories — that part of the linter is genuinely obsolete and is not restored.

`test_perf_jit_import_ordering.py:_PROVIDERS` gains a comment stating what it is and is not: a
curated behavioural complement, not the completeness guard.

## Acceptance criteria

- [x] A new DeBERTa-backed source with a module-scope backbone import + `lambda: GLiNER` factory
      FAILS the suite, with no list edited anywhere.
- [x] `latence-disambig-glinker/src/latence_disambig_glinker/backends.py` is inside a structural
      scan again, and removing its `neutralize_torch_jit()` call is flagged.
- [x] Discovery is non-vacuous (`test_deberta_backed_discovery_picks_up_a_newly_landed_source`) and
      may not shrink (`test_deberta_backed_discovery_covers_every_known_source`).
- [x] The seam-routed exemption is narrow: proven not to rescue a module hoist, an
      import-then-neutralize inversion, a namesake never passed to the seam, or a factory routed
      through the non-neutralizing `load_with_fa2_fallback`.
- [x] The C4 deepening is untouched — no Provider gets a `neutralize_torch_jit()` call back, and the
      factory contract is unchanged. No test weakened or deleted.
- [x] `make test` green.

# conformance-cases-per-package — move the Provider fakes out of `latence-core`

**Severity:** major (ADR-0016 violation, already shipped)
**Area:** `packages/latence-core/src/latence_core/conformance/stub.py` (1,160 lines),
`conformance/cases.py` (`_LAZY_PROVIDER_CASES`, ~line 2021)
**Invariants touched:** ADR-0016 (thin core — heavy deps live in per-Provider packages),
anti-silent-skip (a registered case that fails to import must FAIL, never skip).

## What is wrong

`latence_core.conformance.stub` is 1,160 lines of hand-written fakes for `openai`, `gliner`,
`gliner2`, `sentence_transformers`, `transformers`, `presidio`, and `pypdfium2`/`PIL` — **none of
which are dependencies of `latence-core`**. ADR-0016 put the heavy dependencies in per-Provider
packages; their fakes did not follow. The consequence is a real coupling, not a tidiness
complaint: shipping `latence-pii-presidio` requires editing `latence-core`, and every new
Provider package grows the core by another fake.

The other half of the same shape was fixed in this lane: the *roster* is no longer restated in a
core test — it is derived from each package's own
`[project.entry-points."latence.providers"]` block (`packages/latence-core/tests/_workspace_providers.py`).
The fakes are the remaining restatement.

## The right shape is already proven in-tree

* `latence_source_sharepoint.testing` ships the Microsoft Graph fake **in its own `src`** — "a fake
  that lives next to the code it fakes".
* `conformance/cases.py::_LAZY_PROVIDER_CASES` already builds a case only when the owning package
  is installed.

Generalise that into a **second entry-point group**, `latence.conformance_cases`: each Provider
package declares the callable that yields its own `ConformanceCase` (fake included). `latence-core`
keeps `harness.py` (the C1–C6 checks — genuinely deep, and the whole reason the gate has value) and
`fixtures.py` (the canonical corpus every case shares). `stub.py` dissolves into the packages whose
dependencies it fakes.

## Acceptance criteria

1. `latence-core` contains no fake for a dependency it does not declare; `stub.py` is gone or holds
   only fakes for `latence-core`'s own deps.
2. Adding a Provider package adds its conformance case with **no edit to `latence-core`** (matching
   the roster derivation, which already needs none).
3. **The anti-silent-skip posture is preserved.** An entry-point-registered case that fails to
   IMPORT must FAIL the suite, never skip it — that is the entire value of the gate. A case whose
   package is *not installed* is not registered at all and is caught by the derived roster check
   (`test_registry_discovers_every_provider_the_workspace_declares`), which fails naming the
   declaring package.
4. Each package's fake must **reject what the real dependency rejects** (the no-fake-green bar),
   and the `stub_scoped` flag must keep marking those results as "live numbers need a rig".

## Why it was deferred

Split out of the Storage/conformance deepening lane, which delivered `Storage.atomic_writer` and
the derived roster. Dissolving `stub.py` touches ~10 Provider packages at once and is a
self-contained piece of work with its own review surface.

# Strict mypy is enforced on tests too, and the whole-tree conftest module collision is resolved

The Baseline DoD says "strict type-checking passes on the touched packages," but CI ran
`mypy --strict` on `src/` only. The `tests/` trees were never strict-checked, so the claim
was aspirational, not enforced: `packages/latence-core/tests/{test_contracts.py,test_e2e.py}`
carried strict-mode findings pre-existing from S1 (commit `077920e`) — a mislabeled
`# type: ignore[call-arg]` on a *statically valid* `Provenance(...)` call (the test asserts a
runtime min_length failure, not a missing-arg type error), an unused `[misc]` ignore on a
frozen-model assignment, and an untyped `pyarrow.read_table` call. A whole-tree
`mypy --strict packages` did not even reach these: it aborted immediately with
*"Duplicate module named 'conftest'"* (issue #24).

## Two root problems, two decisions

**1. `tests/` was outside the type gate.** Extending the gate surfaced 31 real findings
across 17 test modules — all legitimate test-side typing gaps, none a contract change:
`pyarrow` untyped calls (the same `# type: ignore[no-untyped-call]` already sanctioned in
`src`), base-`Record` values needing `isinstance(..., DisambiguationRecord)` narrowing
before `.entity`, `PageMap | None` / `char_start: int | None` narrowing in chunk tests,
`raise type(self).error` in the endpoint fakes (bind to a local so the `is not None` narrow
survives), and stale/mislabeled ignores. They are fixed at the call site, so
`mypy --strict packages` now checks `src` **and** `tests` in one pass.

**Sanctioned pattern for "prove Pydantic raises" tests.** A construction that is
*statically* invalid — a missing required field or an `extra="forbid"` field — is a genuine
type error, so we suppress *exactly that error* with a targeted `# type: ignore[call-arg]`
and assert `ValidationError` at runtime. A construction that is statically *valid* but
violates a *runtime* constraint (empty `min_length` string, assignment to a `frozen` model)
carries **no** ignore — `--warn-unused-ignores` would flag a stray one. This keeps the ignore
list an honest inventory of the static/runtime boundary rather than noise. Documented in the
`test_contracts.py` module docstring and the DoD "Typed" bullet.

## Grill: how to make one mypy invocation collision-free

Every package ships `tests/conftest.py`, and all of them map to the top-level module
`conftest`. Three resolutions were weighed:

- **(b) Add `__init__.py` to each `tests/` dir.** *Rejected.* It only moves the collision:
  the `tests` package itself then collides (`Duplicate module named "tests"`), and with
  hyphenated distribution dirs (`latence-core`, not a valid identifier) no directory layout
  yields unique dotted names (`tests.conftest` collides across every base) without renaming
  the distributions — a far larger, contract-touching change. It would also flip pytest from
  its working `prepend` import mode.
- **Rename the distributions / the `tests` packages.** *Rejected — out of all proportion to
  a fixture-file name clash, and it churns the published package boundary (ADR-0016).*
- **(a) `explicit_package_bases` + `namespace_packages`, and exclude the three `conftest.py`
  from the whole-tree run, type-checking them in a dedicated per-file step.** *Chosen.* The
  conftests hold only pytest fixtures, are already fully annotated, and each type-checks
  clean on its own; a `for c in packages/*/tests/conftest.py; do mypy --strict "$c"; done`
  CI step keeps them a strict gate with a clear failure signal, while the single
  `mypy --strict packages` invocation over all `src` + `test_*.py` modules stays
  collision-free. The `exclude` and the two `*_package_bases` settings live in
  `[tool.mypy]`, so even a bare `mypy packages` is clean.

pytest is unaffected: all `test_*.py` basenames are already unique across the tree, so its
default `prepend` import mode never hit the collision; only mypy's global module-name dedup
did. No `__init__.py` files are added anywhere.

## Consequence

`mypy --strict packages` (src + every package's test modules) is green in one invocation; a
second step checks the conftests. Two subprocess regression tests
(`test_strict_typing_gate.py`) lock both properties: one fails if the collision returns (the
"Duplicate module named" marker reappears / the run exits non-zero, and the whole-tree
checked-file count drops to the `src`-only count), the other fails if `--strict` ever stops
rejecting a mistyped `tests/`-shaped module. Heavy deps stay out of `latence-core`
(ADR-0016): the gate adds no runtime dependency — mypy is already a dev-only tool.

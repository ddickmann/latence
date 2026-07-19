# R4-docs_truth-6 — Two ADR-0048 links point at a filename that does not exist, in the two published pages no gate reads

**Severity:** major
**Stream:** feat/release-fleet-integration (#178–#199; introduced by #183 / #189)
**Sites:** `CHANGELOG.md:162`, `packages/latence-sink-qdrant/README.md:26`

## What is wrong

Both locations link to `docs/adr/0048-retrieval-tooling-is-stateless.md`. That file has never
existed. The real ADR is `docs/adr/0048-retrieval-tooling-is-strictly-stateless.md`.

```
$ ls docs/adr/ | grep 0048
0048-retrieval-tooling-is-strictly-stateless.md
$ git show main:CHANGELOG.md | grep -c '0048-retrieval-tooling-is-stateless'
0
```

Both were introduced by this stream — neither is inherited from `main`.

## Failure scenario

Both links render as live links and 404 on GitHub. Nothing catches them:

* `packages/latence-core/tests/test_front_door_truthfulness.py` scopes `_FRONT_DOOR` to
  `README.md` / `CONTEXT.md` / `CONTRIBUTING.md` / `SECURITY.md` plus tracked `docs/**/*.md`.
  `CHANGELOG.md` and `packages/*/README.md` are in **neither** set.
* `mkdocs build --strict` only walks `docs/`.

Worse: `packages/latence-sink-qdrant/README.md` is that distribution's `readme` in its
`pyproject.toml`, i.e. its **PyPI long_description**. The moment #189's flip-public + publish
steps run, the Qdrant sink's PyPI project page ships a broken link to the very ADR that
constrains the sink's behaviour — "it holds no index, caches nothing, and runs no search" —
so the one reader who tries to check that claim cannot reach it.

This is a direct miss of #189's AC *"Gap-analysis inconsistencies fixed: … no dangling
cross-references"*.

## What to build

1. Correct both links to `0048-retrieval-tooling-is-strictly-stateless.md`.
2. Close the gate hole: extend the cross-reference guard past the docs site, over the two
   *published-but-unwalked* surfaces — the root `CHANGELOG.md` and every tracked
   `packages/*/README.md` (each one a PyPI long_description). Markdown links only: a link is
   a navigational promise, whereas a backticked path may legitimately name a dead file when
   the prose is *about* that dead file (`CHANGELOG.md:254` narrates the historic
   `docs/INTEROP.md` fix and must keep passing).

## Acceptance criteria

* `grep -rn '0048-retrieval-tooling-is-stateless' .` returns nothing.
* Every relative markdown link in `CHANGELOG.md` and in every tracked
  `packages/*/README.md` resolves to a file that exists.
* A regression test fails on the pre-fix tree naming both sites, and passes after.
* The guard is non-vacuous: it asserts it actually inspected links, so deleting the
  cross-references cannot make it pass by finding nothing.
* `make test` stays green at or above 2985 passed / 28 skipped.

## Noted, not fixed (out of scope)

`packages/latence-sink-qdrant/README.md` is the only package README that links docs by
*relative* path (`../../docs/adr/…`). Those resolve on GitHub but not on PyPI, where the
long_description has no repository around it. Every other package README names ADRs as plain
text. Making package-README doc links absolute is a separate, repo-wide editorial decision.

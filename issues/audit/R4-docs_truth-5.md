# R4-docs_truth-5 — README's only pointer to incremental corpus updates links to a section anchor that does not exist, and the stream's dangling-cross-reference guard is structurally blind to fragments

**Severity:** major
**Round:** R4 · lane `docs_truth`
**Site:** `README.md:175` (target `docs/TUTORIAL-fresh-pod-walkthrough.md#7-incremental-updates-add--update--retract--purge`)
**Guard that should have caught it:** `packages/latence-core/tests/test_front_door_truthfulness.py::test_no_dangling_cross_references_in_the_front_door`
**Relates to:** #189 AC ("no dangling cross-references"; "a dangling documentation cross-reference" is one of the three gap-analysis inconsistencies the ticket must fix), ADR-0012

## Failure scenario

`docs/TUTORIAL-fresh-pod-walkthrough.md` has:

```
855:## 7. Discover the final output
942:## 8. Incremental updates: add / update / retract / purge
```

`README.md:175` — the README's *only* pointer to `latence delta` / `retract` / `purge` — reads:

```
- [`docs/TUTORIAL-fresh-pod-walkthrough.md#7-incremental-updates-add--update--retract--purge`](./docs/TUTORIAL-fresh-pod-walkthrough.md#7-incremental-updates-add--update--retract--purge) — incremental corpus updates (`latence delta` / `retract` / `purge`).
```

The anchor names §7 while the material is §8. A reader who clicks it on the public GitHub page —
the highest-traffic surface in the repository — lands at the top of an 1100-line tutorial with no
indication where the delta/retract/purge material lives. The link text is *also* wrong, so the
error is visible in the rendered prose, not just in the href.

No gate catches it:

* The stream's own guard does
  `resolved = (page.parent / target.split("#", 1)[0]).resolve()` — it **discards the fragment**
  before resolving, so it passes on *any* wrong anchor. It also skips same-page targets entirely
  (`not target.startswith(...("#",))`), leaving 18 further fragment links unchecked.
* `mkdocs build --strict` never sees `README.md` (it is not in the docs tree / nav), so its link
  checker cannot reach this link at all.

A GitHub-accurate slugifier run over README / CONTEXT / CONTRIBUTING / SECURITY / CHANGELOG /
`docs/**` / `packages/*/README.md` confirms this is the only real dangling anchor in the tree —
i.e. fixing it costs one character, and the *guard* is the actual deliverable.

## What to build

1. Correct `README.md:175` — both the link text and the href — to `#8-...`.
2. Extend `test_front_door_truthfulness.py` so the dangling-cross-reference guard **resolves the
   fragment**, not just the file: for every relative markdown link in the front-door set
   (including same-page `#anchor` links), the fragment must match a heading anchor that the target
   page actually defines.
   * Slugs must be computed the way **GitHub** computes them, since these pages are read on
     github.com; `mkdocs.yml` already pins `pymdownx.slugs.slugify` for the same reason, so both
     renderers agree.
   * Explicit anchors count too: `attr_list` `{#custom-id}` and inline HTML `id=`/`name=`.
   * Pages whose anchors are *generated* (mkdocstrings `:::` blocks, e.g. `docs/reference/*.md`)
     cannot be resolved statically and are skipped for fragment checking — recorded explicitly so
     the exemption is a decision, not an accident.

## Acceptance criteria

- **AC-1** — Every relative markdown link with a `#fragment` on a front-door page resolves to an
  anchor the target page really defines. The guard fails on the pre-fix `README.md:175` and passes
  after the one-character correction.
- **AC-2** — The guard covers same-page (`#anchor`) links as well as cross-file ones; the existing
  file-existence check is kept, not replaced.
- **AC-3** — The guard is non-vacuous: it asserts that it actually inspected fragment links
  (count > 0), and a unit-level test proves the resolver rejects a wrong anchor and accepts the
  right one on a synthetic page, so the guard cannot rot into a no-op if the README link is later
  removed.
- **AC-4** — The slugifier is GitHub-accurate: it reproduces `--` for a removed `+`
  (`gpu-requirements--the-blackwell-note`), drops `.` from numbered headings, strips inline code /
  emphasis / links, and keeps `_`.
- **AC-5** — `make test` stays green at or above 2985 passed, 28 skipped. No test weakened or
  deleted.

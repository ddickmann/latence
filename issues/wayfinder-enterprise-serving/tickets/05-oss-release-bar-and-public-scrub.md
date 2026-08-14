# 05 — OSS release bar (SOTA-repo comparison) + public-repo scrub

Type: task
Status: open
Blocked by: the current path landing fully green — Wave 5-J close, Wave 5-K2, the pod sweep
(21/21), the S5 re-run at scale, and the final campaign report. Do not start before.

## Why (Dennis, 2026-08-14)

"Once all of this is done, compare to SOTA open source repos like Qdrant, docling, scrapling,
hermes agent, the new deepseek harness etc whether release, contribution, CI, README, docs etc
are fully prepared. Also eliminate everything from the public repo (add gitignore) which is
private and not public. I do not want explicit claude references for examples, internal
documents and markdowns. but those valuable for users can stay."

## Part A — the SOTA-OSS comparison

Study the named reference repos (Qdrant, docling, scrapling, the Hermes agent, the new
DeepSeek harness — resolve each to its canonical GitHub repo, note the revision studied) and
extract their actual practice per axis; then gap-analyse this repo against the union bar and
fix what falls short:

1. **README**: structure, badges (CI/coverage/license/PyPI), the fold-line content (what a
   visitor sees in 10 seconds), benchmark presentation, quickstart honesty.
2. **Contribution surface**: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, issue/PR
   templates, good-first-issue signalling, DCO/CLA stance (flag for Dennis, don't decide).
3. **CI**: matrix breadth (OS × Python), caching, release automation, artifact publishing
   pipelines (prepared, NOT fired), badge truthfulness.
4. **Docs**: site structure, versioned docs, API reference generation, examples that run.
5. **Release discipline**: changelog format, tag/version conventions, release notes quality.
6. **Community affordances**: discussions, roadmap visibility, benchmark reproducibility
   instructions.

Deliverable: a comparison table (repo × axis, with what each does), the gap list, and the
fixes landed. Every "prepared" claim backed by an executed command or an existing file path.

## Part B — the public-repo scrub

1. **Inventory** every tracked file for private/internal content: campaign operating internals,
   session/handoff notes, personal paths, pod credentials/IPs, cost figures, internal issue
   machinery not meant for consumers. The `issues/` tree, wave reports, FINDINGS, and campaign
   STATE are the primary suspects — classify each: (a) keep public (user value: ADRs, guides,
   benchmark methodology), (b) remove from tracking + .gitignore (private operating material).
2. **Claude references**: remove explicit Claude/assistant references from examples, internal
   documents, and markdowns. KEEP what is valuable to users (e.g. genuinely useful agent-
   integration docs). Git-history trailers (Co-Authored-By) are history — flag the option of
   a history rewrite to Dennis but NEVER rewrite history unilaterally; the scrub is about the
   working tree.
3. **Secrets/identifiers audit**: pod IPs/ports, SSH details, API-key names, personal
   usernames/emails in tracked files — remove or genericize; verify with a scanner pass.
4. **.gitignore** covers everything classified private so it cannot return; `git rm --cached`
   for currently-tracked private files (content stays local).
5. **Gates stay green**: the publish-readiness, doc-truthfulness, and license-pin tests must
   pass after the scrub (some scanners walk the tree — verify none depended on removed files);
   full suite + ruff + mypy + mkdocs strict re-run and recorded.

## Non-negotiables

- Nothing outward-facing fired (no publishing, no tags, no history rewrite) without Dennis.
- The scrub must not delete local files — untrack + ignore, so operating material survives on
  the machine.
- Removal decisions with any user-value doubt get listed for Dennis's review instead of
  silently dropped.

## Gate artifact

The comparison table + gap fixes, the scrub inventory (kept/untracked lists), the scanner
results, and the re-run gate tails. Dennis inspects.

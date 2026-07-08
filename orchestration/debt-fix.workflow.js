export const meta = {
  name: 'debt-fix',
  description: 'Fix a batch of tracked before-G1 debt issues in one isolated worktree, adversarially verify, then open a PR closing them (never auto-merge).',
  phases: [
    { title: 'Fix', detail: 'implement all issues in the batch in a dedicated worktree + branch, with tests' },
    { title: 'Verify', detail: 'parallel panel: correctness, security, DoD, test-adequacy' },
    { title: 'Synthesize', detail: 'triage, fix blocking defects on-branch, open the PR' },
  ],
}

let a = args
if (typeof a === 'string') { try { a = JSON.parse(a) } catch { a = {} } }
if (!a || typeof a !== 'object') a = {}

// Portable default repo root: prefer an explicit arg, then the LATENCE_REPO env var, then the
// repo this script lives in (orchestration/ is one level below the toplevel), and finally the
// current working directory. No maintainer-specific absolute path is baked in, so this runs on a
// fresh clone in any location while still honoring an explicit `repo` arg when driven by args.
// No import.meta (invalid in the non-module Workflow runtime); guard process (may be absent).
const _env = (typeof process !== 'undefined' && process.env) ? process.env : {}
const _cwd = (typeof process !== 'undefined' && typeof process.cwd === 'function') ? process.cwd() : '.'
const defaultRepo = _env.LATENCE_REPO || _cwd

const title = String(a.title || 'debt batch')
const issues = Array.isArray(a.issues) ? a.issues : []
const branch = String(a.branch || 'debt-fix')
const repo = String(a.repo || defaultRepo)
// Default the worktree as a sibling of the repo root (repo's parent + repo basename + branch),
// so it too is portable rather than an absolute maintainer path.
const worktree = String(a.worktree || `${repo.replace(/\/$/, '')}-${branch}`)
const issueList = issues.join(', ')
const closes = issues.map(n => `Closes #${n}`).join('\n')

// ---- Phase 1: implement the whole batch ----
phase('Fix')
const impl = await agent(
  `Fix the tracked before-G1 debt batch "${title}" (GitHub issues ${issueList}) for latence-framework, to the project's standards. Repo: ${repo} (remote origin = ddickmann/latence-framework, gh user ddickmann).\n\n` +
  `1. Create an ISOLATED worktree so you never touch main:\n   git -C ${repo} worktree add ${worktree} -b ${branch}\n   Work ONLY inside ${worktree}.\n` +
  `2. Read EACH issue's full body for the exact fix required: ${issues.map(n => `gh issue view ${n}`).join(' ; ')}. Also read CONTEXT.md, issues/DEFINITION-OF-DONE.md (Gate bar for security/perf items, Baseline bar otherwise), and the referenced source files + the relevant docs/adr/*.md.\n` +
  `3. Implement the fix for EVERY issue in the batch. Preserve all existing contracts and correctness properties (offsets/page-provenance/Evidence/idempotency). Where an issue overclaims a docstring/ADR, correct the wording to match reality. Keep heavy deps out of latence-core (ADR-0016).\n` +
  `4. Tests: for each fix add a deterministic regression test that would FAIL if the fix were reverted (adversarial for the security/DoS/ReDoS items — a pathological input; for perf items, at least an assertion of the new bound or the wired path). Run ruff, strict mypy, and pytest LOCALLY until GREEN.\n` +
  `5. Commit in logical per-issue chunks (author ddickmann <ddickmann81@gmail.com>; end messages with 'Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>'). Push: git -C ${worktree} push -u origin ${branch}. Do NOT open a PR.\n` +
  `Return: per-issue, what you changed (files) and the regression test name; plus the local ruff/mypy/pytest results (counts). If an issue cannot be fully fixed within scope, say so explicitly and what remains.`,
  { label: `fix:${branch}`, phase: 'Fix' }
)

// ---- Phase 2: adversarial verification ----
phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  properties: {
    lens: { type: 'string' }, pass: { type: 'boolean' },
    blocking: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { summary: { type: 'string' }, file: { type: 'string' }, why: { type: 'string' } },
      required: ['summary', 'why'] } },
    notes: { type: 'string' },
  }, required: ['lens', 'pass', 'blocking'],
}
const LENSES = [
  { key: 'correctness', focus: 'Does each issue in the batch get genuinely fixed, end to end, with no regression to existing behavior (offsets, page provenance, Evidence, idempotency, determinism)? Is each regression test HONEST — would it actually fail if the fix were reverted?' },
  { key: 'security', focus: 'For the security/DoS/ReDoS/secret items: is the guard actually effective and not bypassable? Try to defeat it (pathological regex, oversized input, a config path that skips the guard, a secret key not on the scrub denylist). No new unsafe surface.' },
  { key: 'dod', focus: 'Strict typing, versioned contracts intact (no inappropriate SCHEMA_VERSION churn), tests present and deterministic, CI-clean, docs/ADR corrected where they overclaimed. Every issue in the batch actually addressed (cross-check each issue number).' },
  { key: 'tests', focus: 'Are the regression tests adequate and non-tautological? Does each map to a specific issue and prove the fix? Any issue closed without a test that locks it in?' },
]
const verdicts = (await parallel(LENSES.map(l => () =>
  agent(
    `You are the ${l.key.toUpperCase()} reviewer for the latence-framework debt batch "${title}" (issues ${issueList}) on branch '${branch}' at worktree ${worktree}. Diff: git -C ${worktree} diff main...${branch}. Read the changed files, each issue body (gh issue view N), and issues/DEFINITION-OF-DONE.md.\n\nFocus: ${l.focus}\n\nBe adversarial and specific; a finding is 'blocking' only if it genuinely leaves an issue unfixed, regresses existing behavior, or violates the DoD. Return your verdict.`,
    { label: `verify:${l.key}`, phase: 'Verify', schema: { ...VERDICT } }
  )
))).filter(Boolean)
const blocking = verdicts.flatMap(v => (v.blocking || []).map(b => ({ ...b, lens: v.lens })))
log(`${branch}: ${verdicts.filter(v => v.pass).length}/${verdicts.length} lenses pass; ${blocking.length} blocking`)

// ---- Phase 3: synthesize + PR ----
phase('Synthesize')
const summary = await agent(
  `Finalize the latence-framework debt batch "${title}" (issues ${issueList}). Branch '${branch}' at ${worktree}; repo ${repo}.\n\n` +
  `Verdicts: ${JSON.stringify(verdicts)}\nBlocking: ${JSON.stringify(blocking)}\n\n` +
  `1. If there are blocking findings, FIX them on ${branch} in ${worktree}, re-run ruff + strict mypy + pytest until green, and push. One focused pass; if something can't be safely fixed, leave it and flag it prominently.\n` +
  `2. Ensure a PR exists (gh pr create/edit) against main for ${branch}. Body: a per-issue summary (what changed + regression test), a verification-report table (one row per lens), local test/type/lint results, and a 'Reviewer should check' list. Include these lines so the issues auto-close on merge:\n${closes}\n` +
  `3. NEVER merge the PR.\n` +
  `Return: the PR URL, final ruff/mypy/pytest status, any unresolved blocking finding, and the top things to scrutinize.`,
  { label: `synthesize:${branch}`, phase: 'Synthesize' }
)

return { title, issues, branch, worktree, blockingCount: blocking.length, verdicts, synthesis: summary }

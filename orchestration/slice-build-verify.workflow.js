export const meta = {
  name: 'slice-build-verify',
  description: 'Implement one latence-framework slice in an isolated worktree, adversarially verify it against the Definition of Done, then open a PR for human review (never auto-merge).',
  phases: [
    { title: 'Implement', detail: 'build the slice in a dedicated git worktree + branch, with tests' },
    { title: 'Verify', detail: 'parallel review panel: correctness, security, DoD-compliance, test-adequacy, port-fidelity' },
    { title: 'Synthesize', detail: 'triage verdicts, fix blocking defects on-branch, open/update the PR with a verification report' },
  ],
}

// ---- parameters (pass via Workflow args) ----
// Normalize: args may arrive as a parsed object OR a JSON string. Coerce so a
// stringified args can't turn `args.slice` into String.prototype.slice (a function).
let a = args
if (typeof a === 'string') { try { a = JSON.parse(a) } catch { a = {} } }
if (!a || typeof a !== 'object') a = {}

// Portable default repo root WITHOUT import.meta: this script is executed by the Workflow runtime as
// a non-module (import.meta is a syntax error there) and Node's fs/env/cwd may be absent, so derive
// defensively — prefer the explicit `repo` arg (callers pass it), then a guarded LATENCE_REPO env /
// cwd, else '.'. No maintainer-specific absolute path is baked in.
const _env = (typeof process !== 'undefined' && process.env) ? process.env : {}
const _cwd = (typeof process !== 'undefined' && typeof process.cwd === 'function') ? process.cwd() : '.'
const defaultRepo = _env.LATENCE_REPO || _cwd

const slice = String(a.slice || 'slice')                   // e.g. "S2"
const title = String(a.title || slice)                     // e.g. "S2 — Parse"
const issueFile = String(a.issueFile || '')                // repo-relative, e.g. "issues/phase-1-machinery/S2-parse.md"
const branch = String(a.branch || `${slice.toLowerCase()}-slice`)
const repo = String(a.repo || defaultRepo)
// Default the worktree as a sibling of the repo root (repo path + slice), so it is portable rather
// than an absolute maintainer path.
const worktree = String(a.worktree || `${repo.replace(/\/$/, '')}-${slice.toLowerCase()}`)
const ghIssue = a.ghIssue != null ? a.ghIssue : null       // GitHub issue number to "Closes #", optional

const DESIGN = [
  'CONTEXT.md (glossary — use these exact terms)',
  'issues/DEFINITION-OF-DONE.md (meet the Baseline bar exactly)',
  'docs/ASSET-INVENTORY.md (port the named algorithmic assets; do not reinvent weaker versions)',
  'the relevant docs/adr/*.md',
  `the issue spec: ${issueFile}`,
].join('; ')

// ---- Phase 1: implement ----
phase('Implement')
const impl = await agent(
  `Implement latence-framework slice ${title} to the project's Baseline bar. Repo: ${repo} (remote origin = ddickmann/latence-framework, gh user ddickmann).\n\n` +
  `1. Create an ISOLATED worktree so you never touch main:\n` +
  `   git -C ${repo} worktree add ${worktree} -b ${branch}\n` +
  `   Work ONLY inside ${worktree}.\n` +
  `2. Read the design first (all committed): ${DESIGN}.\n` +
  `3. Implement the slice: typed Pydantic contracts with Provenance+Classification enforced at every boundary you touch; the Capability protocol + a reference Provider registered via the 'latence.providers' entry-point group; wire it into the local Runner and Pipeline; emit metrics into the Quality Report. Reuse existing latence-core seams — do not fork them. Where the issue names an algorithmic asset to PORT, port it faithfully.\n` +
  `4. Tests (deterministic, seeded): unit for happy-path + key failures, and at least one end-to-end through the local Runner. Run ruff, the strict type-checker, and pytest LOCALLY and make them pass.\n` +
  `5. Commit in logical chunks (author ddickmann <ddickmann81@gmail.com>; end messages with 'Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>'). Push: git -C ${worktree} push -u origin ${branch}.\n` +
  `Return: a concise summary of what you built (packages/modules), and the exact local results of ruff / type-check / pytest (pass/fail with counts). Do NOT open a PR yet.`,
  { label: `implement:${slice}`, phase: 'Implement' }
)

// ---- Phase 2: adversarial verification panel ----
phase('Verify')
const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    lens: { type: 'string' },
    pass: { type: 'boolean' },
    blocking: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          summary: { type: 'string' },
          file: { type: 'string' },
          why: { type: 'string' },
        },
        required: ['summary', 'why'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['lens', 'pass', 'blocking'],
}

const LENSES = [
  { key: 'correctness', focus: 'Does the code actually do what the issue says, end to end? Look for logic bugs, unhandled edge cases, contract violations, broken resume/idempotency, wrong offsets/provenance. Default to skeptical: prove a failing input if you can.' },
  { key: 'security', focus: 'Threat-model this slice: malicious/oversized/spoofed input, prompt injection, PII/secret leakage into logs or outputs, unsafe deserialization, SSRF via Source URIs, dependency risk. Is Screening respected where relevant?' },
  { key: 'dod', focus: 'Check the Baseline bar in issues/DEFINITION-OF-DONE.md line by line: strict typing, versioned contracts with Provenance+Classification enforced, tests present (unit + e2e, deterministic), CI wiring, package/dependency isolation, license-verified models, observability into the Quality Report, docs/ADR. List every unmet box as blocking.' },
  { key: 'tests', focus: 'Are the tests adequate and honest? Do they actually exercise the claimed behavior and the failure modes, or are they shallow/tautological? Would they catch a regression? Is anything untested that matters?' },
  { key: 'port-fidelity', focus: 'If the issue names an algorithmic asset to PORT (see docs/ASSET-INVENTORY.md), was it ported faithfully with its correctness properties intact (e.g. drift recovery, no-silent-over-merge, WAL atomicity), or was a weaker version reinvented? If nothing to port, pass with a note.' },
]

const verdicts = (await parallel(LENSES.map(l => () =>
  agent(
    `You are the ${l.key.toUpperCase()} reviewer for latence-framework slice ${title}. The branch '${branch}' is checked out at ${worktree}. Inspect the diff vs main (git -C ${worktree} diff main...${branch}) and read the changed files and the issue spec ${issueFile} and issues/DEFINITION-OF-DONE.md in ${repo}.\n\n` +
    `Focus: ${l.focus}\n\n` +
    `Be adversarial and specific — a finding is only 'blocking' if it genuinely violates the issue or the Baseline bar. Return your verdict.`,
    { label: `verify:${l.key}`, phase: 'Verify', schema: { ...VERDICT_SCHEMA } }
  )
))).filter(Boolean)

const blocking = verdicts.flatMap(v => (v.blocking || []).map(b => ({ ...b, lens: v.lens })))
log(`${slice}: ${verdicts.filter(v => v.pass).length}/${verdicts.length} lenses pass; ${blocking.length} blocking finding(s)`)

// ---- Phase 3: synthesize, fix blocking defects, open/update PR ----
phase('Synthesize')
const closesLine = ghIssue ? `Include 'Closes #${ghIssue}' in the PR body.` : `Find the GitHub issue via 'gh issue list --search "${title.replace(/"/g, '')}" --json number,title,state' and include 'Closes #<n>' if found; otherwise note the link is pending.`
const summary = await agent(
  `You are finalizing latence-framework slice ${title}. Branch '${branch}' at worktree ${worktree}; repo ${repo}.\n\n` +
  `Verification verdicts (JSON): ${JSON.stringify(verdicts)}\n` +
  `Consolidated blocking findings (JSON): ${JSON.stringify(blocking)}\n\n` +
  `1. If there are blocking findings, FIX them on branch ${branch} in ${worktree}, re-run ruff + strict type-check + pytest until green, and push. Do at most one focused fix pass; if something can't be safely fixed, leave it and flag it prominently for the human reviewer.\n` +
  `2. Ensure a PR exists against main for ${branch} (gh pr create, or gh pr edit if it exists). The PR body MUST contain: a short summary of what was built; a verification report table (one row per lens: pass/fail + key findings); the local test/type/lint results; and a clear 'Reviewer should check:' list. ${closesLine}\n` +
  `3. NEVER merge the PR — it is for human review.\n` +
  `Return: the PR URL, the final ruff/type/pytest status, whether any blocking finding remains unresolved, and the top 3 things the human reviewer should scrutinize.`,
  { label: `synthesize:${slice}`, phase: 'Synthesize' }
)

return { slice, title, branch, worktree, blockingCount: blocking.length, verdicts, synthesis: summary }

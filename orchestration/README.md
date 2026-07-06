# Orchestration — how slices get built

The execution cadence is **hybrid** (not an autopilot): foundation slices are sequential and human-review-gated; verification is fanned out.

## Per-slice: `slice-build-verify.workflow.js`

Each Phase-1 slice (S2+) runs through this Claude Workflow:

1. **Implement** — one agent builds the slice in an isolated git worktree + branch, with tests, and runs lint/type/pytest locally.
2. **Verify** — a parallel adversarial panel with schema'd verdicts: `correctness`, `security`, `dod` (Baseline-bar line-by-line), `tests` (adequacy/honesty), `port-fidelity` (was the named algorithm from `../docs/ASSET-INVENTORY.md` ported faithfully, or reinvented weaker?).
3. **Synthesize** — triages verdicts, fixes blocking defects on-branch, opens/updates a PR with a verification-report table. **Never merges** — the maintainer's PR review is the gate.

### Run it

```
Workflow({
  scriptPath: "orchestration/slice-build-verify.workflow.js",
  args: {
    slice: "S2",
    title: "S2 — Parse",
    issueFile: "issues/phase-1-machinery/S2-parse.md",
    branch: "s2-parse",
    ghIssue: 2
  }
})
```

## Gate & hardening

- **G1** (the profound-testing gate) runs as its own multi-lane verification workflow before any Phase-2 work starts.
- **Phase-2 hardening** issues each run their own verification workflow against the Hardening bar.

## Why not one big workflow

S1→S13 is a strict dependency chain with two deliberate human gates (per-slice PR review, and the G1 hard stop). Wrapping the whole spine in one autonomous workflow would remove those gates and run up large unattended cost. Workflows are used where they add value — independent verification and fan-out — not as an end-to-end autopilot.

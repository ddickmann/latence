export const meta = {
  name: 'g1-gate',
  description: 'Run the G1 profound-testing gate: parallel verification lanes (E2E, reliability/chaos, determinism, security+CVE, gold-set quality, perf baseline, provider-swap+license) → a signed G1-RESULTS.md with an overall gate verdict. Verification only; opens no PR.',
  phases: [
    { title: 'Verify', detail: 'parallel G1 lanes each run real checks and return structured evidence' },
    { title: 'Sign', detail: 'synthesize G1-RESULTS.md + overall gate verdict, honest about infra-dependent gaps' },
  ],
}

// Portable repo root: prefer the LATENCE_REPO env var, then the repo this script lives in
// (orchestration/ is one level below the toplevel), and finally the current working directory. No
// maintainer-specific absolute path is baked in, so this runs on a fresh clone in any location.
const scriptRepoRoot = new URL('..', import.meta.url).pathname.replace(/\/$/, '')
const repo = process.env.LATENCE_REPO || scriptRepoRoot || process.cwd()
const RESULT = {
  type: 'object', additionalProperties: false,
  properties: {
    lane: { type: 'string' },
    status: { type: 'string', enum: ['PASS', 'PASS-WITH-GAPS', 'FAIL'] },
    evidence: { type: 'string' },       // what was actually run + concrete numbers/outputs
    gaps: { type: 'string' },           // honest: what needs a real infra/data environment, or was not covered
  },
  required: ['lane', 'status', 'evidence'],
}

phase('Verify')
const preamble = `You are running ONE lane of the G1 profound-testing gate for latence-framework at ${repo} (main is green, 12 packages, uv workspace). Do REAL work in a scratch checkout — clone/worktree or just operate read+run in ${repo} WITHOUT committing (create temp dirs under /tmp for data/outputs). Read issues/gate/G1-profound-testing-gate.md and issues/DEFINITION-OF-DONE.md (Gate bar) for the exact criteria. Use \`uv run\` to execute. Be HONEST: run what you can, report concrete evidence (real numbers/paths), and clearly state any gap that needs a real perf/GPU/enterprise-data environment — do NOT fabricate results. `

const LANES = [
  { key: 'e2e', task: `LANE: E2E on multiple messy corpora. Build 2–3 DIVERSE synthetic-but-messy corpora (mixed .txt/.md + a generated PDF if feasible + deliberately dangerous fixtures: type-spoof, zip-bomb, oversized, a prompt-injection doc), each in a /tmp dir. Run the FULL pipeline end-to-end via the local Runner (Source→Screening→Parse→Chunk→NER→RE→Redaction→Profiling→Disambiguation→Graph Assembly→Export) using the CPU reference providers (the latence-demo build_pipeline is the fastest path; or compose a YAML). For EACH corpus assert: dangerous fixtures Quarantined/flagged, a KG (nodes/edges/Evidence) is produced, a PII-handled RAG corpus is produced, the Quality Report emits, and contract-completeness holds (Provenance/Classification present, offsets aligned). Report doc counts, entity/edge counts, quarantine counts per corpus.` },
  { key: 'reliability', task: `LANE: Reliability & chaos. Prove: (1) crash-mid-run resume — kill a pipeline after some stages, re-run same run_id, assert it resumes from checkpoints and the final output equals an uninterrupted run; (2) delta idempotency — re-run the same delta, assert no new Corpus Version; (3) Purge — physically erase a doc, assert no trace in prior versions/checkpoints/source AND co-resident docs survive; (4) Retraction durable + edit-then-revert restores. Run the existing delta/runner e2e suites AND add at least one genuine mid-commit chaos scenario if feasible. Report which guarantees are proven and how.` },
  { key: 'determinism', task: `LANE: Determinism. Run the same pipeline twice on the same input+config and assert byte-identical exports AND identical Corpus Version fingerprint. Vary PYTHONHASHSEED across ≥3 values and confirm stability. Report the fingerprints and any nondeterminism found.` },
  { key: 'security', task: `LANE: Security review + CVE scan. (1) Run a dependency vulnerability scan — try \`uv run pip-audit\` (install if needed via \`uv pip install pip-audit\`), or \`uv export | pip-audit -r -\`; report any CVEs. (2) Consolidate the threat model: enumerate the security posture across the pipeline (malicious files→Screening, prompt-injection→flag, PII→Redaction+salted report fingerprints, ReDoS guards, secret-scrub of manifest, storage scheme allowlist + symlink guard, no unsafe deserialization). (3) Confirm zero open Sev-1/Sev-2 by scanning the tracked security follow-ups. Report CVE findings (closed/accepted) and the consolidated threat-model table.` },
  { key: 'quality', task: `LANE: Gold-set quality baseline. Create a small hand-annotated gold set (~15–25 short docs with known entities and relations) in /tmp, wire it via the pipeline's gold_set_uri, run the gold-set harness, and record per-Stage precision/recall/F1 as the BASELINE Phase-2 must beat. Report the actual P/R/F1 numbers per stage and the gold-set size. Note this is a small synthetic gold set (methodology proof), not a domain benchmark.` },
  { key: 'perf', task: `LANE: Performance baseline. Generate synthetic corpora at increasing sizes (e.g. 100, 500, 1000, and as high as feasible in a few minutes) and measure wall-clock throughput (docs/sec) and PEAK memory (RSS, e.g. via resource.getrusage or tracemalloc) through the full CPU pipeline. Report a table of size→time→peak-RSS and the throughput trend. HONEST GAP: true 10k/100k-doc numbers and GPU-provider throughput need a dedicated perf environment — validate the methodology and trend at reduced scale, extrapolate cautiously, and state clearly what a real perf rig would confirm.` },
  { key: 'swap-license', task: `LANE: Provider-swap + license audit. (1) Provider-swap: for each Capability with ≥2 registered providers, verify the swap mechanism yields the SAME typed contracts (do it for the CPU/in-core providers that install without heavy deps: e.g. entity.gazetteer, relation.pattern/relation.llm, redaction.hybrid_rule, parse providers; the gliner/endpoint variants are heavy-dep/GPU — verify they REGISTER and share the seam, note that a live swap needs those deps/GPU). (2) License audit: consolidate the weights+code license verification for every default provider model from the ADRs (ADR-0012 etc.) into one table; confirm all defaults are permissive (Apache-2.0/MIT/BSD) and any restricted model (EmbeddingGemma) is opt-in. Report the swap results + the license table.` },
]

const results = (await parallel(LANES.map(l => () =>
  agent(preamble + '\n\n' + l.task, { label: `g1:${l.key}`, phase: 'Verify', schema: { ...RESULT } })
))).filter(Boolean)

phase('Sign')
const summary = await agent(
  `You are signing off the G1 profound-testing gate for latence-framework. Here are the lane results (JSON):\n${JSON.stringify(results, null, 2)}\n\n` +
  `Write \`${repo}/issues/gate/G1-RESULTS.md\` (create it) as the signed gate record: a header with date 2026-07-07 and the commit, then one section per lane (status + concrete evidence + honest gaps), then an OVERALL VERDICT: PASS (all criteria met with real evidence) / PASS-WITH-DOCUMENTED-GAPS (executable criteria pass; perf-at-scale and GPU-provider-swap validated only at reduced scale / need a real environment — enumerate exactly what) / FAIL (a real defect blocks the gate — enumerate). Then a 'Baselines to beat in Phase-2' subsection capturing the gold-set P/R/F1 and perf numbers. Be honest and precise; this is the artifact that authorizes Phase-2. Commit it: git -C ${repo} add issues/gate/G1-RESULTS.md && git -C ${repo} -c user.name=ddickmann -c user.email=ddickmann81@gmail.com commit -m 'gate: G1 profound-testing results + verdict' && git -C ${repo} push origin main (only if main is currently green and no other push is racing; if push fails, pull --ff-only and retry once).\n` +
  `Return: the OVERALL VERDICT, the per-lane status line, and the path.`,
  { label: 'g1:sign', phase: 'Sign' }
)

return { results, verdict: summary }

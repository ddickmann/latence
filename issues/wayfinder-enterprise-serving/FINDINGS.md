# Findings — wayfinder-enterprise-serving (out-of-scope, logged not chased)

Per the campaign pattern: things noticed during ticket work that are real but out of the
ticket's scope. Each entry names where it lives and what would close it. Nothing here blocks
the enterprise-readiness verdict; everything here was deliberately NOT fixed in passing.

## F1 — Graph sources re-read `graph-edges.parquet` per call

`NetworkxGraphSource` / the DuckDB graph sources in `latence-retrieval` open and read the
edges parquet on every `search` call; the S5 harness bypasses them with a run-lifetime source.
Out of ticket 01's profile scope (the harness never pays it), but a production caller using the
shipped sources directly pays a per-query file read. Candidate for the architecture pass —
the same file-identity-keyed cache discipline as `HypergraphIndex` (ADR-0056 amendment) applies.
(Found: ticket 01. Recorded there as a residual; carried here so it survives the ticket close.)

**CLOSED (2026-08-13, architecture pass Wave 1-A).** The shipped adapters
(`DuckDBGraphSource` / `NetworkxGraphSource` / `DuckDBGraphDocumentSource`) now default to a
process-shared decoded read structure keyed by file identity
(`latence_retrieval.read_cache.FileIdentityCache` — the generalized ADR-0056-amendment pattern,
now with an explicit lifetime policy: superseded-version eviction + bounded LRU;
`hypergraph._INDEX_CACHE` migrated onto it). `cache=False` preserves each original per-call path
verbatim as the parity oracle. Byte-identity pinned by the ticket-01 `kg_shipped`/`kg_shipped_nx`
parity lanes on wiki2multihop + multihop_rag (0 diffs) and package tests; per-query median
`GraphRetriever.search` over the shipped adapters drops 60.0→0.34 ms (wiki, DuckDB) and
270.6→0.34 ms (wiki, networkx). See the second ADR-0056 amendment (2026-08-13).

## F2 — `test_goldset_property.py` dense-cluster stress costs ~288 s CPU

`packages/latence-core/tests/test_goldset_property.py` is green but its dense-cluster stress
test alone burns ~5 min of CPU, which is why full-suite wall time is dominated by a single
file. Flagged in ticket 03's CI leg; a complexity fix (smaller cluster budget with the same
adversarial property) would cut suite time materially. Out of scope here — a test-performance
change should not ride a readiness pass.

## F3 — `wayfinder-e2e-sota-run` ticket statuses lag their stages

Tickets s1–s3 still read `Status: claimed` although their campaign stages completed
(S1/S2 approved in STATE.json, S3 completed 2026-08-06 per its metrics.json). s5/s5b are
genuinely in flight (the pod rescue sweep). Left untouched: those statuses belong to that
campaign's approve-flow, and flipping them from a different campaign would falsify who
closed them.

## F4 — local `benchmark/sota-campaign/STATE.json` is a stale snapshot

The tracked STATE.json in this working tree shows s3 `blocked` (a 2026-08-05 pod incident that
was subsequently resumed and completed) and sits modified-uncommitted, touched by parallel
campaign processes. The pod's own state is authoritative while the sweep runs. Left exactly as
found (the parallel-session discipline: never sweep another session's dirty files); reconcile
after the sweep lands.

## F5 — stale branch inventory (for Dennis — nothing deleted)

31 non-archive local branches besides `hypergraph-m1`/`main`, most long-merged or superseded
(`oss-*`, `pr-*`, `lane-*`, `docs*`, `_pod*`, `sota-final`, …), plus `archive/*` tags carrying
the same names. `git branch --merged main` is the safe triage start. Per the operating
contract, no branch was deleted or will be without an explicit go.

## F6 — `chunk-validation/` untracked WIP carries 2 lint errors

`benchmark/sota-campaign/stages/s3-pipeline/chunk-validation/{chunk_stats.py,
induction_sample.py}` are another session's in-flight, untracked scripts (2 ruff errors in
chunk_stats.py). Deliberately neither fixed, ignored, nor committed here — the owning session
decides whether they become evidence or disappear.

## F7 — one order-dependent flake on roster hosts: `test_real_weights_embeds_at_declared_width`

In the single-process full-suite run on a roster host (torch + sentence-transformers + gliner2
+ ULTRA all installed; 5,000+ tests in one interpreter), the embedder-st real-weights test
failed once with `RuntimeError: prims.iota only supports integer dtypes` inside the Granite
model load — global torch state polluted by some earlier test. It passes standalone, passes
with its own package suite, and passes in every targeted combination tried
(extract-gliner2 + pii-gliner2 + splade + embedder-st together: green). The CI-locked env
(no torch) is unaffected and fully green. Polluter not yet identified; candidates are the
torch.compile-using suites. A dynamo/state-reset fixture at the real-weights boundary would
close it — out of scope for the readiness pass, logged here with the evidence.

## F8 — demo builder minted its run dir without the runner's `rstrip('/')` (closed by Wave 2-E)

`latence_demo/builder.py` built `f"{self._storage_uri}/_latence/runs/{run_id}"` while the
runner mints `f"{storage_uri.rstrip('/')}/_latence/runs/{run_id}"` — a `--storage-uri` with a
trailing slash would have had the demo read its graph checkpoint / write its site page at a
`//_latence` path the runner never wrote (default URI unaffected, so no test caught it).
Structurally closed, not spot-fixed: every consumer now mints paths through
`latence_core.run_store.RunLayout`, so the divergence class cannot recur.

## F9 — full latence-core suite segfaults at interpreter shutdown (after 2286 passed / 7 skipped / 0 failed)

The Wave 2-E full-suite gate (single process, `PYTHONHASHSEED=0`, 2,293 tests) ran to
`[100%]` with zero `F`/`E` progress markers and zero `FAILED`/`ERROR` lines (2,286 `.` +
7 `s` counted from the log), printed the warnings summary — the last warning being the torch
JIT deprecation from `test_perf_fp32_regression_guard` — and then died SIGSEGV (exit 139)
before the final count line. A re-run of that torch-adjacent file plus the new
`test_run_store.py` exits 0, so the crash needs the whole 2,293-test session's teardown
(torch + everything in one interpreter) and is not attributable to any test or to the pure-
Python Wave 2-E change. Same family as F7's single-process torch-state fragility on roster
hosts; logged, not chased.

## F7 addendum — ROOT-CAUSED and closed structurally (Wave 3-H)

The `prims.iota only supports integer dtypes` flake reproduced DETERMINISTICALLY on the dev
Mac (torch + sentence-transformers installed) with `test_perf_jit_capability_gate.py` +
`test_real_weights_embeds_at_declared_width` in one interpreter, and bisected to a PAIR of
gate tests (`test_unreadable_nvidia_smi_output_is_neutralized` +
`test_nvidia_smi_missing_entirely_is_neutralized`). Mechanism, two compounding leaks:

1. **Franken re-import of torch.** Both tests `monkeypatch.delitem(sys.modules, "torch")`
   and then call `neutralize_torch_jit(config={})`, whose guarded `import torch`
   RE-EXECUTES `torch/__init__.py` against the stale `torch.*` submodules still in
   `sys.modules` (first such test imports torch cleanly; the second re-inits it). The
   re-init corrupts prims/decomp registration process-wide; the next real-weights
   `from_pretrained` (Granite 311m) dies in `prims.iota`. Fixed by pinning
   `sys.modules["torch"] = None` (import raises → the seam's guarded torch-free path),
   which is the established discipline everywhere else. The other `delitem` sites
   (`test_perf_hardware_derived_defaults.py`, two probe tests in the gate file) never
   trigger a torch import and are left as-is (their `"torch" not in sys.modules`
   assertions need the delitem).
2. **`PYTORCH_JIT=0` leaked past the file.** `monkeypatch.delenv(..., raising=False)` on an
   already-unset var registers no undo, so the env switch the seam sets inside those tests
   escaped into the remainder of any single-interpreter run. Closed by the Wave 3-H owner:
   the file's autouse fixture now wraps every test in `scoped_torch_runtime()`, which
   records and restores the env switch, the in-process JIT state, thread caps and dynamo
   state. The embedder-st real-weights test scopes its own load the same way.

Evidence: the failing pair is green after the fix, and the F7 configuration (embedder-st +
extract-gliner2 + pii-gliner2 + splade + all 8 perf test files, single process,
`PYTHONHASHSEED=0`) ran **3×: 388 passed / 1 skipped / 0 failed, exit 0, every run**.
Claim calibrated: the roster-host polluter was never caught in the act, but the local
reproduction shares the fingerprint (same error, same victim, order-dependent, single
interpreter), and the class — unowned global torch state — is now owned either way.

## F10 — gliner2/splade test suites swallow pytest's final summary line (pre-existing)

`pytest packages/latence-extract-gliner2/tests packages/latence-pii-gliner2/tests
packages/latence-splade/tests -q` exits 0 with all progress markers green but the log ends
at `[100%]` — no warnings summary, no `N passed` line (with embedder-st + perf files added,
the warnings summary appears but the count line is still missing). Reproduced on the
UNTOUCHED suites, so it predates Wave 3-H and is not attributable to it; something in those
suites' teardown swallows the terminal reporter's tail (F9's shutdown fragility family).
Logged, not chased — gate verdicts on these suites should read the exit code + marker
counts, not the summary line.

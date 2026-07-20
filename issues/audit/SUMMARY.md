# Hostile Audit — Summary (conduct-audit, branch `feat/release-fleet-integration`)

**Scope:** the release-fleet integration stream — the diff vs `main`
(`git diff main...feat/release-fleet-integration`, tickets **#178–#199**). That covers release
signing + SBOM (#178/#179), throughput (#180), the benchmark harness and its suites (#181/#187),
the run console (#186/#188), the front-door docs flip (#189), the Blackwell/CUDA-13 JIT work
(#190/#191), the derived sinks — Qdrant (#183) and Iceberg (#184) — the SharePoint connector
(#182), the setup wizard/profiles (#192–#194/#197), and the reproducible-environment +
compatibility-matrix lane (#195/#196/#198/#199). Pre-existing `main` code was in scope only where
this stream's changes touch it.

> The previous stream's audit record (`feat/retrieval-tooling`, tickets #159–#175) has been
> preserved unchanged at [`SUMMARY-feat-retrieval-tooling.md`](SUMMARY-feat-retrieval-tooling.md).
> This file is the record for **this** stream only.

---

## ⚠️ How this audit ended — read this first

**The loop was STOPPED by the maintainer at round 7. It did not reach a clean round.**

The `conduct-audit` protocol terminates when a round surfaces **zero** confirmed critical/major
findings. That never happened. Round 7 — the last round run — confirmed **14** findings and fixed
all of them, and those fixes have **not been re-audited**. This document is therefore a record of
*where the audit was stopped*, not a certificate of convergence.

The honest summary of the trend: the per-round finding count fell from 20 to 12 across rounds 1–4,
then **rose again** (14 → 16 → 14) across rounds 5–7. It was not converging when it was stopped.
The reason is documented below under *The fixes-of-fixes problem* — a substantial fraction of each
round's findings were defects **in the previous round's fixes**.

Nothing here should be read as "the branch is clean". It says: 107 real defects were found and
fixed, the gate is green, the protected invariants were re-checked and hold, and a named,
unquantified residual risk remains.

## Rounds run

**7 rounds** of parallel adversarial auditing, each followed by REFUTE-verification and fixes.

| Round | Tickets filed (confirmed) | Fix commits | Of which were defects in an EARLIER round's fix |
|-------|---------------------------|-------------|--------------------------------------------------|
| R1 | 20 | 20 | — |
| R2 | 16 | 16 | not separately tallied |
| R3 | 15 | 15 | not separately tallied |
| R4 | 12 | 12 | not separately tallied |
| R5 | 14 | 14 | not separately tallied |
| R6 | 16 | 16 | **6** |
| R7 | 14 | 14 | **4** |
| **Total** | **107** | **107** | — |

Counts verified from git in the closing pass, not taken on trust:

```
git log --oneline main..HEAD --grep='fix(audit-R<n>)'                # per-round fix commits
git diff --diff-filter=A --name-only main...HEAD -- issues/audit/    # 107 ticket files + this summary
```

Ticket-to-fix correspondence is exactly **1:1** per round. Fix commits restate the defect in the
subject rather than citing the ticket filename, so traceability is by round + defect statement.

> **Discrepancy noted honestly:** the closing brief quoted confirmed counts of `22/16/15/12/14/16/15`
> (total 110) and "94 fix commits". Neither total reconciles with the tree. The numbers above are
> the ones **git actually contains** (107 tickets, 107 fix commits, matching per round). If three
> additional R1/R7 findings were confirmed but never filed as tickets, they are not recoverable
> from this branch and are not accounted for here.

## The fixes-of-fixes problem (a finding about the process, not the code)

**Round 6 confirmed 16 findings; 6 of them were defects introduced by round 5's own fixes.**
Round 7 confirmed 14; **4** named an earlier round's fix as the source. Measured in the closing
pass by:

```
grep -l -iE "audit[- ]R5|R5-|the R5 fix|round 5" issues/audit/R6-*.md    # -> 6 of 16
grep -l -iE "audit[- ]R[1-6]|R[1-6] fix|the R6 fix" issues/audit/R7-*.md # -> 4 of 14
```

The mechanism is structural, not carelessness: each round dispatches many auditors and fixers **in
parallel**, and a fixer sees the defect it was assigned but not the other fixes landing in the same
round. The redaction span-merge logic is the extreme case — `_merge_floor` and
`_compose_prior_spans` were rewritten by **six sequential agents in round 6 alone** (commits
`0718105`, `fade96a`, `a6fa423`, `3815daa`, `6a41124`, `0f285db`), each closing one facet, none
seeing the others' work.

The practical consequence: **the marginal value of a round was partly offset by the defects that
round introduced**, and the last round's fixes carry that same unquantified risk (residual #1).

## Final gate — REAL measured numbers (closing pass)

Command: `bash scripts/verify-local.sh` → **exit 0**, all ten gates green.

```
✓ secret-scan   ✓ workspace sync   ✓ ruff   ✓ mypy strict   ✓ pytest
✓ conformance   ✓ stack validate   ✓ environment specs
✓ compatibility matrix   ✓ bake-off
```

- **Suite** (`make test`, the stated baseline lane) — the literal final line, exit 0:

  ```
  3417 passed, 29 skipped, 1 warning in 324.29s (0:05:24)
  ```

  Floor that must not regress: **3320 passed / 29 skipped**. **Zero** failures; skip count
  unchanged. No test was weakened, skipped or deleted to reach green.
- **Lint:** `ruff check packages` → All checks passed. **Types:** `mypy --strict` over src + tests
  of all 32 packages → Success in every package.

Not run here, and deliberately not claimed green (the script says so itself): the MinIO `s3://`
e2e lane, the Airflow DAG lane, real GPU/Blackwell runs on the CUDA-13 pod, and
`deploy/env/*/smoke.sh` gates 2+.

## Residual breakage found and fixed by THIS closing pass

The seven audit rounds left the merge gate **red**. Both failures were real and are fixed here.

1. **Secret-scan gate red — three findings not accounted for in the audited baseline**, all
   introduced after the baseline was last refreshed (R7 commit `de5324b`). Each was read
   line-by-line and verified a false positive before being blessed:
   - `docs/benchmarks/results.json` — a `scorer_pin.algorithm_digest` hex value. The *stale* digest
     it replaces was already blessed; the R7 ScorerPin fix changed the fingerprint, so the audited
     entry needed to follow it.
   - `issues/audit/R7-acceptance_criteria-12.md` — the `Secret Keyword` heuristic firing on the
     English word "secret" in prose about repository variables.
   - `issues/audit/R7-security_secrets-2.md` — a synthetic PEM-shaped placeholder (the standard
     armour header wrapped around a run of `A` characters, no key material) inside the ticket's own
     failure-scenario transcript. The literal string is deliberately not reproduced in this summary,
     so this file does not itself trip the scanner.

   Blessing was **scoped to those three verified pairs**; a check confirmed **zero** other
   un-audited findings were swept in, so the gate stays red on a genuinely new secret.

2. **`mypy --strict` red — 4 `unused-ignore` errors in 2 test files**, both introduced by the R7
   `scrub_sensitive` fixes (`test_runner.py`, `test_sharepoint_source.py`). `scrub_sensitive`
   returns `object`; the first `type: ignore[index]` laundered the value to `Any`, making every
   subsequent ignore on the same value redundant. Fixed by narrowing with `isinstance` asserts
   instead — which removes the `Any` laundering rather than merely deleting the ignores.

## Targeted re-verification of the redaction span-merge logic (closing pass)

Because `_merge_floor` / `_compose_prior_spans` were rewritten by six agents in one round and
touched again in round 7, the closing pass re-verified them **empirically** rather than by reading
the diffs. Throwaway probes (not committed) drove the two ADR-0044 invariants directly:

- **Coverage preservation under partial overlap, both directions** — 20 000 randomized
  `_merge_floor` cases (Provider spans vs floor spans at every overlap topology) asserting the
  output covers **exactly** the union of the inputs: no character lost, none invented, output
  ordered and non-overlapping. **Zero violations.**
- **Chained Redactors with overlapping labels and disagreeing boundaries** — 20 000 randomized
  `_compose_prior_spans` cases mixing MASK/REPLACE/HASH/TAG, asserting mask coverage is exactly the
  union, that no previously-masked character is re-exposed, and that a TAG span never overlaps (and
  so never shadows) a mask. **Zero violations.**
- **End-to-end through `finalize_redacted_chunk`** — partial overlaps in both directions, nested
  spans, a 3-deep `TAG → MASK → MASK` chain, chain-order reversal, and re-running the same
  Redactor. In every case the synthetic card/IBAN/SSN stayed masked and no upstream mask was lost.

The known-bad pattern (greedy earliest-start-wins that *dropped* the losing span instead of
clipping it to the gap) is **not present**: `_resolve_clipping` clips to the running maximum end,
and its correctness argument holds for start-sorted input. **Verdict: the final state is coherent;
no defect found.** One sub-critical observation is recorded as residual #4.

## Confirmed defects fixed, grouped by area

All 107 are REFUTE-verified confirmations with a regression test that fails on the pre-fix code.
Each line is one representative failure scenario, with the ticket prefix under `issues/audit/`.

### Redaction floor + span composition (7) — `R5-12`, `R6-invariants-*`, `R6-security_secrets-6/7`, `R6-wiring_dead_code-2/3`, `R7-invariants-0`
- **CRITICAL** `R6-invariants-0` — the R5-12 composition fix **swallowed masks**: a downstream
  Redactor's span dropped a partly-overlapping upstream span wholesale, re-exposing 12 digits of an
  already-masked card in the exported corpus.
- `R6-invariants-1` / `R6-wiring_dead_code-3` — `_merge_floor`'s greedy first-wins resolution
  defeated the ADR-0044 floor on the **single-Redactor** path: a mis-boundaried Provider span
  discarded the floor's card/IBAN/SSN span entirely.
- `R6-wiring_dead_code-2`, `R6-security_secrets-6/7` — the same hole re-reported from three
  dimensions; two closed as WONTFIX-already-fixed, with coverage hardened.
- `R6-cross_lane_integration-14` — chained Redactors double-counted in the Quality Report and could
  name a PII type the corpus never carries.
- `R7-invariants-0` — duplicate documents **halved** the Quality Report's PII tally.

### Blackwell / CUDA-13 JIT + device placement (5) — `*-blackwell_jit_190-*`, `*-invariants-*`
- **CRITICAL** `R1-blackwell_jit_190-3` — `neutralize_torch_jit()` ran **after** the provider's
  `from gliner import GLiNER`, which compiles mdeberta's relative-position helpers into real
  `ScriptFunction`s at import time; neither `PYTORCH_JIT` nor `torch.jit._state.disable()` can undo
  that, so the `libnvrtc-builtins.so.13.0` crash that killed a real 519-page run stayed open.
- `R1-blackwell_jit_190-4`, `R2-blackwell_jit_190-1`, `R3-blackwell_jit_190-0` — the guards for
  that fix were hollow or blind: one accepted a loader that never neutralizes; one scanned a
  hardcoded 5-path tuple, leaving the 6th in-tree DeBERTa provider unguarded; one walked only
  `FunctionDef` bodies, so a **module-level** backbone import — the exact shape of the bug — was
  invisible.
- `R2-blackwell_jit_190-0` — ADR-0036 and every derived doc still *mandated* the defeated mechanism
  and *forbade* the per-Provider pre-import call that actually works.
- `R1-invariants-0-gliner-device` + R5 — `redaction.gliner2` never placed its model on `cuda` (a
  silent CPU fallback); the #198 placement gate excluded the very Provider it exists to protect.

### Derived sinks: Qdrant #183 + Iceberg #184 (12) — `*-data_contracts_sinks-*`
- **CRITICAL** `R1-data_contracts_sinks-5` — a configured `load_id` silently disabled the
  reconciling sweep, so retracted/purged records kept serving while the load reported success.
- `R2` — two overlapping `QdrantSink.load` runs against one collection left it **completely empty**
  while both reported success; `verify_absent` (the documented GDPR erasure proof) returned an
  affirmative "absent" for a *missing collection*.
- `R3` — the sweep guard was a check-then-act with an open window; the convergence check counted
  Export **rows** against distinct **points**, so a corpus with two byte-identical documents never
  converged.
- `R4`/`R5` (Iceberg) — schema drift was caught only in the widening direction; a `uuid4` random
  filename put hidden nondeterminism on the export path; a failed commit left a corpus-sized
  **orphan Parquet** in the deliverable directory.
- `R6-data_contracts_sinks-9` — the Iceberg data file was published **inside the run-artifact tree
  that Purge deletes**.
- `R7-data_contracts_sinks-3/4` — a failed *publication* stranded a full-corpus temp Parquet,
  unnamed; and cleanup deleted a data file a **concurrent** run's snapshot references
  (check-then-act race).
- `R7-invariants-1` — `latence purge` never reached the derived Iceberg table.

### SharePoint connector #182 (9) — `*-connectors_ingestion-*`
- `R2-connectors_ingestion-9` / `R2` — the pre-authenticated download URL, which carries a **live
  access token in its query string**, was interpolated into error messages and into the transport.
- `R1` — the headline "an unchanged document is skipped without transferring a byte" was false
  under the real Delta semantics; the redirect leg bypassed retry/backoff entirely.
- `R3` — on SharePoint Online (quickXorHash) every delta run downloaded the **entire library
  twice**; the exported corpus was not byte-reproducible.
- `R4` — after a single 401, `refreshed` was never cleared, forcing a fresh token on every retry.
- `R6-connectors_ingestion-10` — `source_uri` built without percent-encoding, destroying the Graph
  identity.
- `R6-security_secrets-8` — `@odata.nextLink` was followed **credentialed to an arbitrary host**
  (cross-host bearer-token replay).
- `R7-connectors_ingestion-6` — `GraphClient.list_files` recursed into folders with **no cycle
  guard**.
- `R7-security_secrets-2` — an inlined `private_key` / `certificate` / `passphrase` was persisted
  **cleartext to `manifest.json`**, written before the guard that would reject it could run.

### Run console #186/#188 (12) — `*-console_readonly-*`
- `R2` — no `Host` header validation on the loopback console → **DNS-rebinding** exposure; the chunk
  inspector's PII "source document" span and page number were silently wrong for chunks carrying a
  context header.
- `R3` — a malformed `Origin` raised an unhandled `ValueError` inside the host guard; the Redaction
  disposition told the reader the PII control was **off** on a chunk the floor had actually redacted.
- `R4` — browse views materialised the entire run's checkpoints in memory per request (an
  O(corpus) blow-up in a read-only viewer).
- `R5` — past the 2000-record scan cap the KG views reported genuinely-checkpointed Evidence
  mentions as "not in this run's checkpoints" (a **false provenance claim**); the entity page
  rendered "Relations (0)" for a node that has edges.
- `R6-console_readonly-11` — the quarantine view still claimed **absence of risk markers** on a
  capped read.
- `R6-console_readonly-12` — `MAX_LINE_BYTES` was a **phantom guard**.
- `R7-console_readonly-7` — the console rendered an **unreadable checkpoint as an empty run**.
- `R7-console_readonly-8` — the quarantine + screening sidecar path was unbounded in the run's size.

### Benchmark integrity #181/#187 (10) — `*-benchmark_integrity-*`
- `R1` — every published "learned re-pin" Provider name was wrong; published artifacts predated
  #187; four new suites shipped with **zero** published results; the publisher **overwrote**
  `docs/BENCHMARKS.md` with only the suites of that one invocation.
- `R2` — the redaction `coverage` metric mixed measurement bases; a bold MICRO row was appended to a
  non-disjoint scoring dimension.
- `R3` — carried-forward measurements were re-rendered with **today's** scorer constants; the "By
  PII type" table silently dropped false positives whose predicted type is absent from gold.
- `R4`/`R5` — the one benchmark number that reproduces offline in seconds was wrong; a
  `PARSE_ERROR` document converted UNSUPPORTED units into FAILED, making the published pass rate
  depend on which documents the pinned Provider happened to parse.
- `R7-benchmark_integrity-9` — the **ScorerPin fingerprint**, whose entire purpose is to refuse
  republishing a pass rate whose scoring rules changed, was **blind to `score_document_failure`** —
  the exact rule an R5 audit fix had changed.
- `R7-benchmark_integrity-10` — `fuzzy_find`'s exact short-circuit broke its leftmost contract, so
  **adding** an OCR error to a parse could flip a FAILED reading-order test to PASSED, contradicting
  the harness's own published anti-gaming claim.

### Docs truth / front door #189 (12) — `*-docs_truth-*`
Self-contradictory release-signing claims; "44 architecture decisions" against 52 on disk;
present-tense claims for results never produced; a "last verified dry run" evidence block
describing a tree that never existed; every deployment doc telling a Blackwell operator to add
`--enforce-eager` against this stream's own machine-checked matrix; README summarising
`THIRD-PARTY-LICENSES.md` as "all permissive" — a claim **this stream itself falsified** by adding
~40 proprietary NVIDIA CUDA wheels; and a `#189` "front door may not lie" gate enforced only
against `README.md`, so the docs-site landing page could lie freely. `R7-docs_truth-5` — the GPU
half of `PERF-RESULTS.md` had no measurement date and its guard accepted an ADR id instead.

### Security / secrets / supply chain #178 (9) — `*-security_secrets-*`
- `R2` — `HttpLineageTransport` sent the OpenLineage collector API key through
  `urllib.request.urlopen`, whose default redirect handler replays headers to the redirect target.
- `R3` — the Sigstore identity from the `workflow_dispatch` path did not match the
  `--cert-identity` in the shipped `VERIFY.md`; the "PyPI auto-publish is disabled" guard matched a
  literal `if: false` that exists only inside a YAML comment.
- `R4` — the signed release SBOM omitted the dependencies of **18 of 32** shipped distributions.
- `R5` — the verification procedure could not detect an **added** release asset while `VERIFY.md`
  asserted full asset coverage.

### Acceptance criteria / release runbook (5) — `*-acceptance_criteria-*`
- **CRITICAL** `R7-acceptance_criteria-12` — #189's go-live runbook Step 4 was a **deadlock**:
  enabling the PyPI publish on `main` can never reach the tag ref the release-integrity guard
  forces the publishing run to use, so `pypi-publish` silently SKIPPED instead of publishing.
- `R7-acceptance_criteria-13` — the release rehearsal dispatched a tag that does not exist yet.

### Wizard / profiles (5) + env reproducibility (4)
- `R1`/`R4` — the go-live tutorial's install list omitted the newly-blessed `redaction.gliner2`;
  that Provider silently turned `torch.compile` **on** in the enterprise-sota config.
- `R5-wizard_profile-13` — `tokenizer.regex`, blessed as the enterprise BM25 tokenizer, uses
  `[A-Za-z0-9]+`: it shatters accented words and produces **literally zero terms** for
  CJK/Cyrillic, so the "multilingual enterprise stack" shipped a broken sparse leg.
- `R6-wizard_profile-15` / `R7-wizard_profile-11` — the tutorial's "exact enterprise-sota spine"
  was not that spine; then the R6 gate written to catch that drift **exempted the one `model` pin
  that is not operator-supplied**, passing with exactly the drift class it was written to catch.
- `R2` — the #198 GPU placement gate measured *resident CUDA memory*, not that the GPU did the
  forward pass. `R6-env_reproducibility-5` — `probe_placement` reported green **having measured
  nothing** when the probe set is non-empty but no stack component belongs to it.

### Cross-lane integration + wiring/dead code (13)
- `R2` — `document_id` was minted in **two incompatible namespaces both labelled `sha256:`**.
- `R1` — #183's sink wrote chunk text under payload key `content` while #173's `QdrantBackend` (the
  reader) expected another, so the two halves of the shipped stack did not interoperate.
- `R4` — the sink's Export reader interpolated the **entire raw Parquet row, including `content`**
  (chunk text) into an error message.
- `R5-wiring_dead_code-1/2` — `redact_pii` was **wired-but-dead** in both GPU deploy stacks, so the
  learned PII redactor's masking never reached the exported corpus while the Stage reported
  success; and `redact_financial` wired `redaction.gliner2` with no `pii_labels`, which the
  constructor hard-rejects.
- `R6-cross_lane_integration-13` — `processor.pack`'s redundancy penalty was **dead for non-Latin
  scripts**.

### Test quality / false green (3) — `*-test_quality_false_green-*`
- `R4` — the named regression test for the R3 SharePoint determinism fix was **hollow**: its
  "unordered" Graph double merely reversed a base fake, so the property could not fail.
- `R3-cross_lane_integration-14` — a fake made the duplicate-id case **structurally impossible**, so
  the guard never ran the input it claimed to guard.
- `R6-test_quality_false_green-4` — `test_env_stack_constructible` reported **PASS** for `cpu-text`
  while constructing nothing of its Export.

## Refuted / duplicate / pre-fixed — and why it matters

The harness only writes a ticket for a finding that survives REFUTE-verification, so refuted
candidates were killed in-flight and are not files here. What **is** recorded, and matters:

- `R1-acceptance_criteria-19`, `R1-cross_lane_integration-16`, `R1-test_quality_false_green-2`,
  `R5-cross_lane_integration-11`, `R6-security_secrets-6`, `R6-security_secrets-7` —
  **DUPLICATE / WONTFIX**: each re-surfaced a defect already fixed by an earlier commit. Recorded
  rather than re-fixed, so no second "fix" competes with the first. Two of these are in round 6 and
  concern the span-merge logic — three dimensions independently found the *same* hole, which is a
  sign the parallel-auditor design does produce genuine redundancy on high-risk code.
- `R2-connectors_ingestion-9`, `R1-acceptance_criteria-18`, `R4-acceptance_criteria-11` —
  **PARTIALLY PRE-FIXED**: the reported symptom was already closed, but a residual structural hole
  remained and *was* closed. This is the category that most often hides a false green — "already
  fixed" is not the same as "cannot recur".
- `R2-blackwell_jit_190-0` is the most instructive refutation on the branch: the *code* was right
  and the **authoritative record was wrong**. ADR-0036 mandated the mechanism the field had
  disproved. Nothing was failing; the defect was that the next maintainer would have deleted the
  working fix on the ADR's authority. Docs-as-defect is a real severity class here — which is why
  `docs_truth` is the largest single dimension.

## Protected-invariant verification (closing pass)

Re-checked in this pass. **How** each was checked is stated, so a reader can judge the strength of
the evidence rather than trusting a green cell.

| Invariant | How checked in THIS pass | Result |
|---|---|---|
| **ADR-0016** thin core | **Empirically, in an isolated venv**: created a fresh `uv venv`, installed only `packages/latence-core`, confirmed `torch`/`transformers`/`pyiceberg`/`qdrant-client`/`sentence-transformers`/`gliner`/`presidio`/`numpy`/`pandas` are **all absent** (`find_spec` → None), then `import latence_core` plus `pipeline`/`runner`/`redaction_policy` — all OK, with **zero** heavy modules in `sys.modules` afterwards | **GREEN (empirical)** |
| **ADR-0044/0042** redaction floor intact, chunk-level, no real PII | 40 000 randomized probes + end-to-end chains against the live seam (see the span-merge section above) — no mask lost, no PII re-exposed. Separately, scanned the whole tracked tree for card/IBAN/SSN patterns: every hit is a **published test value** (the Visa test PAN, the canonical DE/AT example IBANs, the reserved-range SSN) or a float in a BM25 score assertion. `synthetic_pii.jsonl` is template-generated by `build_synthetic_pii.py` | **GREEN (empirical)** |
| **ADR-0017** files are the source of truth; sinks/tables are DERIVED | The R6/R7 sink findings were all resolved in the direction of this invariant (the Iceberg data file moved *out* of the Purge-deleted tree; purge extended to reach the derived table). Guard tests naming ADR-0017 across `test_export_iceberg.py`, `test_bm25_export.py`, `test_export_embedding_seam.py`, `test_console_scale.py`, `test_unreadable_artifacts.py` are green in the 3417-test suite | **GREEN (guarded)** |
| **ADR-0048** retrieval tooling strictly stateless | `packages/latence-retrieval/tests/test_stateless_seam.py` green — five named guards including `test_processor_does_not_retain_candidates_across_calls` and `test_backend_holds_only_the_corpus_it_was_handed`. `qdrant.py` holds a lazily-opened **connection handle**, not an index or durable state | **GREEN (guarded)** |
| **ADR-0034** no silent errors; counts-only telemetry; no content/PII/secrets in any facet | The R4 "raw Parquet row incl. `content` in an error message" and R7 "`private_key` cleartext in `manifest.json`" findings were both of exactly this class and are fixed with regression tests. Secret-scan gate green with **zero** un-audited findings. `test_observability.py`, `test_openlineage.py`, `test_adapter_base.py` green | **GREEN (guarded)** |
| **ADR-0033** streaming / memory-bounded | The R4 console fix removed whole-run materialisation; R6 closed the **phantom** `MAX_LINE_BYTES` guard; R7 closed the unbounded quarantine/screening sidecar path. `test_export_memory_scaling.py`, `test_runner_executor_streaming.py`, `test_export_incremental.py`, `test_runner_streaming_checkpoint.py` green | **GREEN (guarded)** |
| **OFF-BY-DEFAULT ⇒ BYTE-IDENTICAL** | Named guards assert *observable* byte-identity, not a config flag: `test_export_iceberg.py::test_disabled_leaves_the_export_byte_identical`, `test_bm25_export.py::test_no_bm25_block_is_byte_identical_and_writes_no_files`, `test_context_columns_absent_by_default_schema_byte_identical`, `test_optional_stage_a_stack_without_it_is_byte_identical` — all green | **GREEN (guarded)** |

**Caveat on the "guarded" rows.** For those, the evidence is *the branch's own guard tests passing*.
Several of those guards were themselves written or repaired during this audit — and rounds 4, 6 and
7 each caught a guard that reported PASS while measuring nothing (`R4` hollow determinism test,
`R6-test_quality_false_green-4`, `R6-env_reproducibility-5`). A green guard is meaningful evidence,
but on this branch it is not the same strength of evidence as the two **empirical** rows above.

## RESIDUAL / KNOWN-OPEN items

Named explicitly. None is silently downgraded.

1. **Round 7's fixes were never re-audited — the risk is real and UNQUANTIFIED.** This is the most
   important open item. R6 found that **6 of its 16** findings were defects in R5's fixes, and R7
   found **4 of 14** in earlier rounds' fixes. R7's own 14 fixes have had no such round run against
   them. By the branch's own observed base rate that would suggest **roughly 3–5 latent defects in
   R7's fixes**, but this is an extrapolation from two data points, not a measurement. The affected
   R7 surfaces are: the Iceberg concurrency/cleanup race, the ScorerPin fingerprint, `fuzzy_find`,
   the Graph cycle guard, the manifest scrubber, `purge`, and the release runbook.
2. **GPU / Blackwell / CUDA-13 behaviour is verified only structurally and offline.** Several
   fixes — the JIT neutralisation ordering, gliner2 `cuda` placement, the #198 placement probe, the
   `pii_labels` construction abort, the vLLM environment pins — address code paths that have
   **never executed on real hardware** during this audit. The compatibility matrix records those
   cells as `UNVERIFIED` with runnable reproduce commands rather than claiming them green. The
   maintainer will validate on the pod separately. **"Green" here does not mean "proven on the
   GPU".**
3. **Lanes the local gate cannot cover, and does not claim to:** the MinIO `s3://` e2e lane, the
   Airflow DAG lane, and `deploy/env/*/smoke.sh` gates 2+.
4. **Overlapping TAG spans can survive composition** (found by the closing pass's fuzz; judged
   **below the critical/major bar**, left unfixed). When two chained Stages both emit a `TAG` span
   over the same region with different types, `_compose_prior_spans` can emit TAG fragments that
   overlap each other — occasionally two identical fragments. Assessed as benign: `apply_masking`
   skips TAG entirely so `masked_content` is unaffected; mask coverage is exactly correct; no
   validator on `ChunkRecord.pii_spans` requires non-overlap (only the deprecated
   `RedactionRecord` has one); and neither consumer (`browse.py`, `redaction_suite.py`) depends on
   it. Worst case is a duplicated audit record slightly inflating a TAG-type count in the Quality
   Report. It was deliberately **not** fixed at audit close — this is exactly the kind of
   low-value change to high-risk code that produced the fixes-of-fixes problem.
5. **`uuid.uuid4()` in `latence-sink-qdrant/.../sink.py` (`_load_marker`)** is *deliberate*
   nondeterminism and correctness-**required**: a marker reused across runs matches the previous
   run's points and the sweep silently becomes a no-op — the exact CRITICAL
   `R1-data_contracts_sinks-5` failure. It sits in a **derived** sink, not on the corpus-export
   path. Named so a future determinism sweep does not "fix" it back into the bug.
6. **The merge gate's pass/skip counts are not printed.** `addopts = "-q"` plus `verify-local.sh`'s
   own `-q` yields `-qq`, which suppresses pytest's summary line. The gate proves *green* but never
   states how many tests ran, so this pass had to run `make test` separately to obtain a quotable
   count. Dropping one `-q` would fix it.
7. **A slow block in the suite.** `packages/latence-core/tests/test_goldset_property.py` runs
   CPU-bound for minutes at ~33% progress and dominates wall-clock. Verified **untouched by this
   branch** — pre-existing, not an audit regression — but it makes the gate slow enough to
   discourage running it.
8. **`DeprecationWarning: invalid escape sequence` in `latence_core/observability.py:1`** (module
   docstring). Harmless today, an error in a future Python. One-character fix, left alone to keep
   this pass's diff limited to gate-breaking issues.

## Bottom line

107 confirmed defects were found and fixed across 7 rounds, including several that would have put
real PII into an exported corpus. The gate is green on measured numbers (3417 passed / 29 skipped,
`verify-local.sh` exit 0), the redaction span-merge logic was re-verified empirically and is
coherent, and the protected invariants hold.

**But the audit was stopped, not completed.** It was not converging, round 7's fixes are
un-re-audited, and the GPU paths are unproven on hardware. A reader deciding whether to merge
should weigh residuals #1 and #2 explicitly rather than reading the green gate as sufficient.

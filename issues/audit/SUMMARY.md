# Hostile Audit — Summary (conduct-audit, branch `feat/release-fleet-integration`)

**Scope:** the release-fleet integration stream — the diff vs `main`
(`git diff main...feat/release-fleet-integration`, 262 files, tickets **#178–#199**). That
covers release signing + SBOM (#178/#179), throughput (#180), the benchmark harness and its
suites (#181/#187), the run console (#186/#188), the front-door docs flip (#189), the
Blackwell/CUDA-13 JIT work (#190/#191), the derived sinks — Qdrant (#183) and Iceberg (#184) —
the SharePoint connector (#182), the setup wizard/profiles (#192–#194/#197), and the
reproducible-environment + compatibility-matrix lane (#195/#196/#198/#199). Pre-existing `main`
code was in scope only where this stream's changes touch it.

> The previous stream's audit record (`feat/retrieval-tooling`, tickets #159–#175) has been
> preserved unchanged at [`SUMMARY-feat-retrieval-tooling.md`](SUMMARY-feat-retrieval-tooling.md).
> This file is the record for **this** stream only. Ticket files from both streams share this
> directory; the 77 belonging to this stream are the ones added by this branch
> (`git diff --diff-filter=A main...HEAD -- issues/audit/`).

## Rounds run

**5 rounds** of parallel adversarial auditing, each followed by REFUTE-verification and fixes,
then this **final integration pass** (round 6 in effect: no new dimension sweep, but a full
green-gate reconciliation of everything the five rounds changed).

| Round | Confirmed critical/major tickets | Fix commits |
|-------|----------------------------------|-------------|
| R1 | 20 | 20 |
| R2 | 16 | 16 |
| R3 | 15 | 15 |
| R4 | 12 | 12 |
| R5 | 14 | 14 |
| **Total** | **77** | **77** |

Severity split across the 77: **3 critical, 48 major** by explicit `Severity:` header; the
remainder are tickets whose header records a duplicate/pre-fixed disposition instead of a fresh
severity (enumerated under *Refuted / duplicate / pre-fixed* below). Ticket-to-fix correspondence
is exactly 1:1 per round. Fix commits restate the defect in the subject rather than citing the
ticket filename, so traceability is by round + defect statement, not by filename grep.

## Final gate (this pass) — REAL numbers

Command: `bash scripts/verify-local.sh` → **exit 0, all ten gates green.**

```
✓ secret-scan   ✓ workspace sync   ✓ ruff   ✓ mypy strict   ✓ pytest
✓ conformance   ✓ stack validate   ✓ environment specs
✓ compatibility matrix   ✓ bake-off
LOCAL VERIFY GREEN — all mirrored CI gates pass.
```

- **Suite** (`make test`, the stated baseline lane) — the literal final line, exit 0:

  ```
  3320 passed, 29 skipped, 1 warning in 321.53s (0:05:21)
  ```

  Baseline that must never regress: **2985 passed, 28 skipped**. This pass is **+335 passed**
  over that baseline — the audit's own regression tests plus this stream's feature tests — with
  **zero** failures. The skip count moves 28 → 29; the skips are optional-dependency / heavy-model
  guards, not silenced failures.
- **The merge gate's own pytest step runs the same set** and is green, but its count line is
  **not printed**: the repo's `addopts = "-q"` plus `verify-local.sh`'s own `-q` make `-qq`, which
  suppresses the summary. Tallying its progress stream gives 3347 records, consistent with the
  3349 (3320 + 29) that `make test` reports directly. The count above is therefore the **measured**
  `make test` line, not an inferred one; see residual #2.
- **Lint:** `ruff check packages` → All checks passed.
- **Types:** `mypy --strict` over src + tests of every package → Success in every package.

Not run here, and deliberately not claimed green (the script says so itself): the MinIO `s3://`
e2e lane, the Airflow DAG lane, real GPU/Blackwell runs on the CUDA-13 pod, and
`deploy/env/*/smoke.sh` gates 2+ (which need the installed heavy venv and real hardware).

## Residual integration breakage found and fixed by THIS final pass

The five audit rounds left the merge gate **red**. Both failures were real and are fixed here.

1. **Secret-scan gate red — the audited baseline was never updated for this stream's new files.**
   `scripts/secret_scan_gate.py` failed with 9 un-audited `(file, hashed_secret)` pairs across 5
   files, *all of which are new on this branch*: `docs/benchmarks/results.json`,
   `packages/latence-core/tests/test_redaction_composition.py`, and three files of the #183 Qdrant
   sink. Every finding was read line-by-line and verified a false positive before being blessed:
   a pinned **git commit SHA** and two **determinism digests** (`algorithm_digest`,
   `fingerprint`) in the benchmark results; the constant *name* `API_KEY_ENV_KEY = "api_key_env"`;
   the env-var **name** `QDRANT_API_KEY`; dummy test values (`s3cr3t`, `from-flag`, `ab$cd12$xy`);
   and two **synthetic** PII fixtures — the example IBAN paired with *Erika Mustermann*
   at the reserved `@acme.example` TLD, and `Jane Doe / 123-45-6789 / 4111111111111111` (the
   published Visa test number, a fixture this audit itself introduced in `R4-invariants-0.md`).
   Blessing was **scoped to those five verified files**; the merge script leaves anything else
   un-audited so the gate stays red on a genuinely new secret (it reported
   `kept-prior-audit=64 newly-blessed=9 left-UNAUDITED=0`).

2. **`mypy --strict` red — 12 errors in 2 files, both introduced by audit fixes.**
   - `packages/latence-console/src/latence_console/artifacts.py` (10 errors): the R5 fix added the
     generic helper `_from_scan(select: Callable[[_Scan], _Kept[_R]], key) -> tuple[bool, _R | None]`.
     `_Kept` is **invariant** (it has `put`, which consumes an `_R`), and mypy 2.1.0 solves `_R`
     from the enclosing return context, widening it to `ChunkRecord | None` — so a selector
     returning `_Kept[ChunkRecord]` no longer satisfied the parameter. Fixed by introducing a
     read-only `_KeptView` Protocol (covariant `truncated` + `get`, no `put`), which both
     type-checks and states the real contract: `_from_scan` only ever *reads* the scan's maps. A
     `type[_R]` witness was tried first and empirically did **not** fix it; the Protocol did.
   - `packages/latence-core/tests/test_redaction_composition.py` (2 errors): bare `dict` in two
     helper signatures; annotated to `dict[str, Any]`.

## Confirmed defects fixed, grouped by theme

All 77 are REFUTE-verified confirmations with a regression test that fails on the pre-fix code.
Grouped by audit dimension (ticket counts in parentheses); each line is one representative
failure scenario, with the ticket prefix under `issues/audit/`.

### Blackwell / CUDA-13 JIT + device placement (5) — `*-blackwell_jit_190-*`, `*-invariants-*`
- **CRITICAL** `R1-blackwell_jit_190-3` — `neutralize_torch_jit()` ran **after** the provider's
  `from gliner import GLiNER`, which compiles mdeberta's relative-position helpers into real
  `ScriptFunction`s at import time; neither `PYTORCH_JIT` nor `torch.jit._state.disable()` can undo
  that, so the `libnvrtc-builtins.so.13.0` crash that killed a real 519-page run stayed open.
- `R1-blackwell_jit_190-4`, `R2-blackwell_jit_190-1`, `R3-blackwell_jit_190-0` — the guards for
  that fix were hollow or blind: one accepted a loader that never neutralizes; one scanned a
  hardcoded 5-path tuple, leaving the 6th in-tree DeBERTa provider (`disambiguation.glinker`)
  unguarded; one walked only `FunctionDef` bodies, so a **module-level** backbone import — the
  exact shape of the bug — was invisible.
- `R2-blackwell_jit_190-0` — ADR-0036 and every derived doc still *mandated* the defeated
  loader-internal mechanism and *forbade* the per-Provider pre-import call that actually works,
  inviting a maintainer to delete the real fix.
- `R1-invariants-0-gliner-device`, `R5` gliner2 placement — `redaction.gliner2` never placed its
  model on `cuda` (the `ensure_gliner2_on_cuda` seam was not reached), i.e. a silent CPU fallback.
- `R5` — the #198 GPU-placement gate derived its probe set in a way that excluded the very
  Provider it exists to protect.

### Derived sinks: Qdrant #183 + Iceberg #184 (9) — `*-data_contracts_sinks-*`
- **CRITICAL** `R1-data_contracts_sinks-5` — a configured `load_id` silently disabled the
  reconciling sweep, so retracted/purged records kept serving while the load reported success.
- `R2` — two overlapping `QdrantSink.load` runs against one collection left it **completely
  empty** while both reported success; and `verify_absent` (the documented GDPR erasure proof)
  returned an affirmative "absent" for a *missing collection*.
- `R3` — the sweep guard was a check-then-act with an open window; the convergence check counted
  Export **rows** against distinct **points**, so any corpus with two byte-identical documents
  never converged.
- `R4`/`R5` (Iceberg) — schema drift was caught only in the widening direction (a **dropped**
  optional signal column passed); a `uuid4` random filename put hidden nondeterminism on the
  export path; and every failed commit left a corpus-sized **orphan Parquet** file in the
  deliverable directory.
- `R1` — Iceberg catalog faults escaped as raw `pyiceberg` exceptions instead of typed errors.

### SharePoint connector #182 (7) — `*-connectors_ingestion-*`
- **CRITICAL-adjacent** `R2-connectors_ingestion-9` / `R2` — the pre-authenticated download URL,
  which carries a **live access token in its query string**, was interpolated into error messages
  and into the transport; the assigned leak was already fixed, the residual was closed here.
- `R1` — the headline "an unchanged document is skipped without transferring a byte" was false
  under the real Delta semantics; the redirect leg bypassed retry/backoff entirely.
- `R3` — on SharePoint Online (quickXorHash) every delta run downloaded the **entire library
  twice**; the documented determinism mechanism was not implemented, so the exported corpus was
  not byte-reproducible.
- `R4` — after a single 401, `refreshed` was never cleared, forcing a fresh token on every retry.

### Run console #186/#188 (8) — `*-console_readonly-*`
- `R2` — no `Host` header validation on the loopback console → **DNS-rebinding** exposure; and the
  chunk inspector's PII "source document" span and page number were silently wrong for chunks
  carrying a context header.
- `R3` — a malformed `Origin` or absolute-form request target raised an unhandled `ValueError`
  inside the host guard; the Redaction disposition told the reader the PII control was **off** on a
  chunk the ADR-0044 financial floor had actually redacted.
- `R4` — browse views materialised the entire run's checkpoints in memory per request and rendered
  every row unpaginated (an O(corpus) blow-up in a read-only viewer).
- `R5` — past the 2000-record scan cap the KG views reported genuinely-checkpointed Evidence
  mentions as "not in this run's checkpoints" (a **false provenance claim**, ADR-0021/0031); and
  the entity page rendered the affirmative falsehood "Relations (0) — No relation … touches this
  entity" for a node that has edges, with no truncation notice.

### Benchmark integrity #181/#187 (8) — `*-benchmark_integrity-*`
- `R1` — every published "learned re-pin" Provider name was wrong; published artifacts predated
  #187 and were never regenerated; four new suites shipped with **zero** published results; and
  the publisher **overwrote** `docs/BENCHMARKS.md` with only the suites of that one invocation.
- `R2` — the redaction `coverage` metric mixed measurement bases (gold-counted TPs vs
  detection-counted FPs); a bold MICRO row was appended to a non-disjoint scoring dimension.
- `R3` — carried-forward measurements were re-rendered with **today's** scorer constants; the
  "By PII type" table silently dropped every false positive whose predicted type is absent from gold.
- `R4`/`R5` — the one benchmark number that reproduces offline in seconds was wrong; and a
  `PARSE_ERROR`/missing-PDF document converted its `table`/`math` units from UNSUPPORTED
  (outside the denominator) into FAILED (inside it), making the published pass-rate depend on
  which documents the pinned Provider happened to parse.

### Docs truth / front door #189 (11) — `*-docs_truth-*`
Self-contradictory release-signing claims in `CONTINUITY.md`; "44 architecture decisions" against
52 on disk; present-tense claims for results that were never produced; `RELEASING.md`'s "last
verified dry run" evidence block describing a tree that never existed; every deployment doc telling
a Blackwell operator to add `--enforce-eager` against this stream's own machine-checked matrix;
README summarising `THIRD-PARTY-LICENSES.md` as "all permissive" — a claim **this stream itself
falsified** by adding ~40 proprietary NVIDIA CUDA wheels; a `CONTINUITY.md` SHA-pinning assertion
that the workflows do not satisfy; and dangling ADR/section anchors. The `#189` "front door may not
lie" gate was itself enforced only against `README.md`, so the docs-site landing page could lie freely.

### Security / secrets / supply chain #178 (5) — `*-security_secrets-*`
- `R2` — `HttpLineageTransport` sent the OpenLineage collector API key through
  `urllib.request.urlopen`, whose default redirect handler replays headers to the redirect target.
- `R3` — the Sigstore identity from the `workflow_dispatch` path did not match the `--cert-identity`
  in the shipped `VERIFY.md`; the "PyPI auto-publish is disabled" guard matched a literal `if: false`
  that exists only inside a YAML comment.
- `R4` — the signed release SBOM omitted the dependencies of **18 of 32** shipped distributions
  (the CRA "component inventory" evidence).
- `R5` — the verification procedure could not detect an **added** release asset while `VERIFY.md`
  asserted full asset coverage.

### Wizard / profiles #192–#194/#197 (3) + env reproducibility #195/#196/#198/#199 (3)
- `R1`/`R4` — the go-live tutorial's install list omitted the newly-blessed `redaction.gliner2`;
  that Provider silently turned `torch.compile` **on** in the enterprise-sota config.
- `R5-wizard_profile-13` — `tokenizer.regex`, blessed as the enterprise BM25 tokenizer, uses
  `[A-Za-z0-9]+`: it shatters accented words and produces **literally zero terms** for CJK/Cyrillic,
  so the "multilingual enterprise stack" shipped a broken sparse leg.
- `R2` — the #198 GPU placement gate measured *resident CUDA memory*, not that the GPU did the
  forward pass. `R3` — #199's derived coverage was silenced by an under-declared spec.
  `R2` — #196's pinned vLLM environment pulled 25 proprietary NVIDIA wheels none of which were recorded.

### Cross-lane integration + wiring/dead code (6 + 3)
- `R2` — `document_id` was minted in **two incompatible namespaces both labelled `sha256:`**.
- `R1` — #183's sink wrote chunk text under payload key `content` while #173's `QdrantBackend`
  (the reader) expected another, so the two halves of the shipped stack did not interoperate.
- `R4` — the sink's Export reader interpolated the **entire raw Parquet row, including `content`**
  (i.e. chunk text) into an error message.
- `R5-wiring_dead_code-1/2` — `redact_pii` was **wired-but-dead** in both GPU deploy stacks
  (`export_corpus` depended only on `redact_financial`, its parallel sibling), so the learned PII
  redactor's masking never reached the exported corpus while the Stage reported success; and
  `redact_financial` wired `redaction.gliner2` with no `pii_labels`, which the constructor
  hard-rejects — so #198's placement gate and #199's full-stack verification both aborted at Stage
  construction with a raw `ValueError` traceback.
- `R5` — the `latence-sink-qdrant` CLI caught only `SinkError`, so the most common first-run
  failure (the optional `qdrant-client` extra missing) surfaced as a raw traceback.

### Test quality / false green (2) — `*-test_quality_false_green-*`
- `R4` — the named regression test for the R3 SharePoint determinism fix was **hollow**: its
  "unordered" Graph double merely reversed a base fake, so the property could not fail.
- `R3-cross_lane_integration-14` — a fake made the duplicate-id case **structurally impossible**
  (`touched == 2` while the collection can hold only 1 marked point), so the guard never ran the
  input it claimed to guard.

## Refuted / duplicate / pre-fixed — and why it matters

The harness only writes a ticket for a finding that survives REFUTE-verification, so refuted
candidates were killed in-flight and are not files here. What **is** recorded, and matters:

- `R1-acceptance_criteria-19`, `R1-cross_lane_integration-16`, `R1-test_quality_false_green-2`,
  `R5-cross_lane_integration-11` — **DUPLICATE / WONTFIX**: each re-surfaced a defect already
  fixed on this branch by an earlier commit. Recorded rather than re-fixed, so no redundant code
  churn and no second "fix" competing with the first.
- `R2-connectors_ingestion-9`, `R1-acceptance_criteria-18`, `R4-acceptance_criteria-11` —
  **PARTIALLY PRE-FIXED**: the reported symptom was already closed, but a residual structural hole
  remained and *was* closed. This is the category that most often hides a false green — "already
  fixed" is not the same as "cannot recur".
- `R2-blackwell_jit_190-0` is the most instructive refutation on the branch: the *code* was right
  and the **authoritative record was wrong**. ADR-0036 mandated the mechanism the field had
  disproved. Nothing was failing; the defect was that the next maintainer would have deleted the
  working fix on the ADR's authority. Docs-as-defect is a real severity class here, not paperwork —
  which is why `docs_truth` is the largest single dimension (11 tickets).

## Protected-invariant verification

Each was checked in this final pass — by direct inspection of the code and by the live gate, not
by assuming the rounds left them intact.

| Invariant | How checked | Result |
|---|---|---|
| **ADR-0017** files are the source of truth; sinks are DERIVED | Read `iceberg.py` + `sink.py`: both document derived-from-the-Parquet-deliverable semantics; the Iceberg schema is derived from *the exact Arrow schema of the deliverable*; the R5 orphan-file and random-filename fixes exist precisely to keep the deliverable authoritative | **GREEN** |
| **ADR-0048** retrieval tooling strictly stateless | `packages/latence-retrieval/tests/test_stateless_seam.py` green in the suite. `latence_retrieval/qdrant.py` holds `self._client` — a **lazily-opened connection handle**, not an index or durable state; the module is the `RetrievalBackend` *adapter* to a store the operator owns, and its import is deferred to first `search()` | **GREEN** |
| **ADR-0016** thin core | Proved **empirically**, not by grep: built `latence-core` into an isolated venv with only pydantic/fsspec/pyarrow/pyyaml/typer and ran `import latence_core` — OK, with **zero** heavy modules pulled in (torch, transformers, qdrant-client, vLLM, sentence-transformers, gliner/gliner2, presidio, duckdb, neo4j all absent). Declared deps are exactly those five; qdrant/iceberg/observability are **optional extras**. Every heavy import in core is function-local behind a `# noqa: PLC0415` justification | **GREEN** |
| **ADR-0004/0012** core never names a model; licences verified weights-vs-code separately | Providers are entry-point plugins. Licence records are explicit and separate the two axes — e.g. `qdrant.py` states "code only; a vector store ships **no** model weights", Apache-2.0 for both client and engine, dated. The R2/R3 fixes *added* the ~40 proprietary NVIDIA CUDA wheels to `THIRD-PARTY-LICENSES.md` and corrected the README's false "all permissive" summary — i.e. this stream moved licence truth in the honest direction | **GREEN** |
| **ADR-0044/0042** universal redaction floor; chunk-level; no real PII | `synthetic_pii.jsonl` is **template-generated** (`build_synthetic_pii.py`) and unmistakably synthetic: `John Q. Public`, `@example.com`, `+1 (415) 555-0142` (reserved fictional range), `1600 Example Road, Springfield`, `123-45-6789`, `4111 1111 1111 1111`. `redaction_suite.py` states "**No real PII, ever.**" The R3 console fix restored the truthful financial-floor disposition; `test_redaction_floor.py` / `test_redaction_noop_floor.py` / `test_redaction_composition.py` green | **GREEN** |
| **ADR-0034** no silent errors; counts-only observability | AST-scanned every branch-changed `src/*.py` for `except: pass`. Six hits, all inspected and all legitimate: narrow *control-flow* exceptions in `iceberg.py` (`NamespaceAlreadyExists`, `NoSuchTable`, lost create race) each with a broad sibling handler that normalises to typed `ProviderError`; `KeyboardInterrupt` shutdown; a loopback-parse `ValueError` that falls through to an explicit refusal; and `runner.py:1254`, a best-effort failure-report write wrapped so it "never masks the real Stage failure" — the real error is re-raised on the next line. That last one **is** the ADR-0034 posture (telemetry failure must never fail a run), not a violation | **GREEN** |
| **ADR-0033** streaming / memory-bounded | The R4 console fix removed exactly the violation (whole-run materialisation per request) and added pagination + the 2000-record scan cap; `_read_jsonl` is line-streamed with a `MAX_LINE_BYTES` bound; `test_export_memory_scaling.py`, `test_streamed_jsonl_is_byte_identical_to_buffered` green | **GREEN** |
| **ADR-0021/0031** provenance is true | The R5 console fixes exist *because* a truncated scan was making false provenance claims; both the mention-resolution and the entity-page edge-truncation notice are fixed and guarded. Benchmark offsets are computed from the templates, so fixture offsets are exact by construction | **GREEN** |
| **OFF-BY-DEFAULT ⇒ BYTE-IDENTICAL** | Named guards assert *observable* byte-identity, not a config flag: `test_export_iceberg.py::test_disabled_leaves_the_export_byte_identical` and `::test_the_export_tree_is_byte_identical_across_two_fresh_runs`, `test_bm25_export.py::test_no_bm25_block_is_byte_identical_and_writes_no_files`, `test_context_columns_absent_by_default_schema_byte_identical`, `test_optional_stage_a_stack_without_it_is_byte_identical` — all green | **GREEN** |
| **DETERMINISM** — seed pinned, no hidden `now()`/`random` on pipeline/export paths | `PYTHONHASHSEED := 0` exported in the `Makefile` *before the interpreter starts*, and in `verify-local.sh`; the root conftest fails fast if unpinned. Grepped `uuid4/random/time.time/datetime.now/utcnow` across `iceberg.py`, the stages, and the sink: **no** nondeterminism on the export path (the R5 fix removed the `uuid4` filename). One deliberate exception, named rather than hidden — see residuals | **GREEN** |

No audit fix regressed any protected invariant.

## Residual issues, named explicitly (all judged below the critical/major bar)

1. **`uuid.uuid4()` in `latence-sink-qdrant/src/latence_sink_qdrant/sink.py:802`
   (`_load_marker`).** This is a *deliberate* nondeterminism and correctness-**required**: the
   sweep deletes `must_not(_latence_load_id == marker)`, so a marker reused across runs matches
   the previous run's points and the sweep silently becomes a no-op — the exact CRITICAL
   `R1-data_contracts_sinks-5` failure. It sits in a **derived** sink, not on the corpus-export
   path, so it does not touch the determinism or byte-identity invariants. Named here so a future
   determinism sweep does not "fix" it back into the bug.
2. **The merge gate's pass/skip counts are not printed.** `addopts = "-q"` in `pyproject.toml`
   combined with `verify-local.sh`'s own `-q` yields `-qq`, which suppresses pytest's summary
   line, so `verify-local.sh`'s log proves *green* but never states how many tests ran. Cosmetic,
   but it means the merge gate's headline number is not readable from its own output — this pass
   had to run `make test` separately to obtain a real, quotable count. Dropping the `-q` from the
   script (or from `addopts`) would fix it.
3. **A slow block in the suite.** `packages/latence-core/tests/test_goldset_property.py` runs
   CPU-bound for several minutes at ~34% progress and dominates wall-clock for both `make test`
   and the merge gate. Verified **untouched by this branch** (`git diff main...HEAD` is empty for
   it) — pre-existing, not an audit regression — but it makes the gate slow enough to discourage
   running it, which is its own risk.
4. **`DeprecationWarning: invalid escape sequence '\ '`** in
   `packages/latence-core/src/latence_core/observability.py:1` (module docstring). Emitted on every
   run; harmless today, an error in a future Python. One-character fix (raw docstring), left alone
   here to keep this pass's diff limited to gate-breaking issues.
5. **Lanes the local gate cannot cover, and does not claim to:** the MinIO `s3://` e2e lane, the
   Airflow DAG lane, real GPU/Blackwell runs on the CUDA-13 pod, and `deploy/env/*/smoke.sh`
   gates 2+. Several R2/R5 fixes (GPU placement actually measuring the forward pass, the gliner2
   `cuda` placement, the `pii_labels` Stage-construction abort) are **verified only structurally
   and offline**; their real proof needs the Blackwell host. The compatibility matrix records those
   cells as `UNVERIFIED` with runnable reproduce commands rather than claiming them green — which
   is the honest posture, but it means "green" here does not mean "proven on the GPU".

No open critical or major finding remains on this branch.

# P3-F3 — Provider Conformance suite: the enterprise-readiness gate

**Phase-3 Wave-0 foundation. GitHub issue #88. ADR-0036 §5 (goal C).** Builds on P3-F1
(`ProviderProfile`, registry `.profiles()`) and P3-F2 (`AdapterBase`, typed-error mapping).

Goal **C — "enterprise readiness across the board"**: a reusable, **parametrized-over-every-
registered-provider** pytest suite that asserts a fixed contract every provider must satisfy to
be called production-ready. "Passes conformance" is the definition of enterprise-ready. This is
what makes the model-agnostic claim *auditable* instead of asserted, and it is the gate every
Wave-1 adapter must pass to land.

## The registered provider landscape (what the suite parametrizes over)
29 providers today, named `<capability>.<variant>` (the Capability is derivable from the name
prefix): source, parse/parser, chunk, screening.intake/content, entity, relation, redaction,
profiling, disambiguation, graph, export, embedding, delta. Most are in-core CPU references;
a few are heavier/endpoint (`entity.gliner`, `relation.gliner_relex`, `redaction.gliner_pii`,
`*.endpoint`, `relation.llm`, `embedding.endpoint`). The suite must cover **all of them**,
running CPU providers for real and conformance-checking GPU/endpoint providers against a
**mock/stub** (real quality/throughput needs a rig, #64 — flagged, never fabricated).

## Scope (this slice)

### 1. Capability conformance cases — the per-Capability fixture registry
Providers span heterogeneous Capabilities, so a single canonical input won't do. Build a
registry `conformance/cases.py` mapping **each Capability** → a `ConformanceCase` with:
- a **canonical valid input** (a tiny, in-repo, deterministic fixture appropriate to the
  Capability — e.g. a short doc for Parse, a ChunkRecord for NER, an EntityMention set for RE,
  a text with a known PII span for Redaction);
- an **output-validity predicate** asserting the Capability's typed-record contract: every
  produced record carries **Provenance + Classification**, offsets are present and in-range
  (reuse the S10/completeness check + the ADR-0031 OffsetMap invariants — do not reinvent), and
  the record validates against its Pydantic contract;
- an **adversarial input** (malformed / oversized / injection-bearing / empty) and the
  **expected graceful outcome**: a typed `LatenceError` (or a Screening quarantine / PARSE_ERROR
  disposition where that Capability defines one) — **never** a bare crash, hang, or silent-wrong
  output. Reuse the doc-level/corpus-level fuzz discipline from #74/#75.
Map provider-name-prefix → Capability so a new provider is auto-covered once its Capability has
a case. A provider whose Capability has **no** case must FAIL the suite loudly (forces the case
to be written) rather than silently skipping — no silent coverage gaps.

### 2. The parametrized conformance checks (applied to every provider)
`conformance/test_conformance.py`, parametrized over `ProviderRegistry` providers × their
Capability case. For each provider assert:
- **C1 valid typed records.** On the canonical input it emits contract-valid records with
  Provenance/Classification/in-range offsets (the validity predicate). An empty/no-op result is
  allowed **only** where the Capability legitimately produces none (documented per case).
- **C2 graceful failure.** On the adversarial input it raises a **typed** error (`category_of`
  classifies it) or takes the Capability's defined quarantine/PARSE_ERROR path — never an
  untyped crash. Assert **no record content / PII in the error message** (P3-F2 discipline),
  proven with a PII-bearing adversarial input.
- **C3 license recorded + permissive-or-opt-in (ADR-0012).** The provider declares a
  `ProviderProfile` (missing profile = FAIL — every provider must declare one now); its
  `license` is a real SPDX id or the `UNVERIFIED` sentinel; a **restricted** license (non-OSI /
  the Gemma family) must be flagged opt-in, and `UNVERIFIED` must never appear with
  `license_verified=True` (the invariant, re-asserted at the provider level).
- **C4 determinism-or-documented.** If `profile.deterministic`, running the canonical input
  **twice** yields byte-identical typed output; if not, the suite records the declared
  non-determinism (no assertion) — but a provider that declares `deterministic=True` and is not
  FAILs. This is the signal the bake-off (P3-F5) trusts.
- **C5 resource/device honored.** `select_device` (P3-F1) is respected: a `compute="gpu"`
  provider on this CPU host is **skipped-with-flag** (recorded, not run, not faked); a CPU
  provider runs. No provider imports a GPU stack at module import on a CPU host (dep-light).
- **C6 no secret/PII leak.** Beyond C2's error path: a provider that is **not** the Redactor
  must not emit raw PII into logs or non-payload fields; assert the tracing spans + any provider
  logging stay counts-only (the repo's no-PII-in-report discipline).

### 3. Endpoint/GPU providers — conformance via stub, honestly bounded
Endpoint providers (`*.endpoint`, `relation.llm`, `embedding.endpoint`) and GPU model providers
(`*.gliner`) run the **contract/graceful-failure/license/determinism-declaration** checks against
a deterministic **mock/stub** (a fake endpoint returning a canned typed response; the existing
endpoint test doubles). Their real quality/throughput is explicitly **out of scope** (rig #64) and
the suite emits a clear "conformance-verified against stub; live numbers need a rig" flag for each.

### 4. Wire into CI
Run the conformance suite in `ci.yml` for CPU providers on py3.11 + py3.12 (deterministic,
`LATENCE_CUDA=0`). GPU providers run the suite where a GPU exists (skipped-with-flag otherwise).
Keep it fast (tiny fixtures); it is a gate, not a benchmark.

## Out of scope (later)
Stacks + validation harness (P3-F4), the bake-off matrix (P3-F5), and any Wave-1 adapter package.
The conformance suite is the *gate* those adapters pass; it does not itself add an adapter.

## Definition of Done (Baseline bar + these)
- A reusable conformance harness (`conformance/` package or module in `latence-core` tests, or a
  small installable `latence-conformance` if that fits ADR-0016 better) parametrized over **every**
  registered provider; a provider with no Capability case FAILs (no silent gap).
- C1–C6 implemented and green for **all in-core CPU providers**; GPU/endpoint providers pass the
  stub-scoped subset with the "needs a rig" flag surfaced (never a fabricated number).
- `mypy --strict` clean over src+tests; `ruff` clean; deterministic + seeded (`LATENCE_CUDA=0`).
- Wired into CI (both py versions) and demonstrably catching a violation (add a deliberately-broken
  throwaway provider in a test to prove C2/C4 bite, then remove — or an xfail-marked in-test fake).
- No heavy dep pulled into core by the harness; endpoint/GPU covered by stub only.
- ADR-0036 §5 status note ("realized, P3-F3") + Phase-3 plan entry + a short `docs/` note: "how a
  new provider passes conformance" (the Wave-1 adapter author's checklist).

## Reviewer should check
- Coverage is **real**: every registered provider is actually parametrized (assert the provider
  count), and a Capability with no case FAILs rather than skipping.
- The checks **bite**: C2 (graceful failure) and C4 (determinism) would actually fail a bad
  provider — proven, not asserted (the throwaway-broken-provider check).
- No PII leaks through C2's error assertions or C6; the no-PII discipline holds.
- GPU/endpoint honesty: stub-scoped, flagged, no fabricated quality/throughput; CPU host skips
  GPU providers with a flag (never a fake device or a silent skip).
- License gate is genuine: restricted licenses flagged opt-in, `UNVERIFIED`+verified impossible.

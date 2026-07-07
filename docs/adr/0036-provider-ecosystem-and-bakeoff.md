# Provider ecosystem: adapters, device profiles, conformance, and the bake-off harness (proposed)

**Status: proposed** — design for the Phase-3 workstream (many real providers per stage, config-driven, CPU/GPU, with an enterprise-readiness gate). Builds on ADR-0004 (Capability protocols + Provider plugins), ADR-0007 (CPU-first, GPU as upgrade), ADR-0012 (permissive defaults, restricted opt-in), ADR-0016 (thin core + per-Provider packages), ADR-0033 (streaming Runner).

## Goal
Turn "model-agnostic" from a seam into a proven, testable ecosystem: many real providers per Stage (PaddleOCR-VL, LightOnOCR, docling, kreuzberg, lightparse for Parse; the Rust chunking engine for Chunk; GLiNER variants for NER/RE; multiple embedders; multiple PII models), each a config-selectable plugin, with a harness that (A) validates a **default stack** end-to-end, (B) runs a **per-Stage bake-off** across many providers, and (C) enforces **enterprise readiness across the board**.

## Decisions

### 1. Provider = Adapter over a native library (no new seam)
The Capability protocol (ADR-0004) IS the stable input/output interface. A provider is an **adapter**: it maps a native library's I/O to the Capability's typed contract (bytes/ParserInput → DocumentRecord+PageMap for Parse; text → EntityMention for NER; etc.). Each adapter is its own isolated package with its heavy dep (ADR-0016). We add a thin optional `AdapterBase` in latence-core carrying the cross-cutting concerns every adapter repeats — device selection, batching, mapping native exceptions to the typed error taxonomy (ADR-0034), and offset/Provenance preservation helpers — so a new adapter is small and consistent. No change to the Capability protocols themselves.

### 2. `ProviderProfile` — a declared descriptor (new, small, additive)
Every provider declares a `ProviderProfile`: `compute` (cpu | gpu | either), approx `memory_mb`, `model_id` + `license` (SPDX + weights/code verified per ADR-0012), `deterministic` (bool), `batch` support, and `cost_per_1k` (for endpoint/API providers). The registry surfaces it. This drives: skipping GPU providers on a CPU-only host (flag, don't fail — the G1 posture), device routing, license/cost columns in the bake-off, and honest "needs a GPU/perf rig" gaps. Device handling stays the **adapter's** job (it moves its own model to `config.device` = auto|cpu|cuda); the framework does not orchestrate serving (ADR-0003) — it only passes config + reads the profile.

### 3. Stacks — the default stack + named alternatives, config-driven
A `stacks/` directory holds full Pipeline configs: `default.yaml` (the blessed CPU-first stack, ADR-0007), `gpu-sota.yaml` (the GPU/high-accuracy stack), and room for client stacks. (A) A **stack-validation** harness runs a named stack end-to-end on a corpus and asserts contracts/KG/RAG/report + the G1-style reliability checks — so "test the default stack" is one command.

### 4. Bake-off / Provider Matrix harness (the core new tool)
Config-driven `matrix/<stage>.yaml` lists candidate providers for one Stage; the harness holds every other Stage at the default and runs the SAME corpus + gold-set through each candidate, producing a comparison report — **quality** (the S10 gold-set P/R/F1 or a stage-specific metric), **latency/throughput**, **peak memory**, **cost**, **license**, **determinism**, **error rate**. Generalizes the S10 gold-set harness + the G1 perf lane into a per-Stage bake-off. Output: a signed `MATRIX-RESULTS.md` per stage (like G1-RESULTS). GPU candidates are gated on device availability (run where present, else skip+flag — no fabricated numbers).

### 5. Provider Conformance suite (enterprise readiness, C)
A reusable, parametrized-over-every-registered-provider pytest suite that is the "across the board" gate. For each provider it asserts: emits VALID typed records (Provenance/Classification/offsets present, contract-complete); **graceful failure** (malformed/oversized/adversarial input → typed error or PARSE_ERROR, never a crash — reusing the Screening/fuzz discipline); resource bounds respected (declared memory/device honored); **license recorded + permissive-or-opt-in** (ADR-0012); determinism **or** documented non-determinism; no secret/PII leak. "Passes conformance" = enterprise-ready. Runs in CI for CPU providers; GPU providers run the conformance suite where a GPU exists (flagged otherwise).

### 6. License + cost are first-class
Per ADR-0012 and the research-diligence directive, every adapter's weights+code license is verified and recorded in its `ProviderProfile` and surfaced in the bake-off — an enterprise decision input, not an afterthought. Restricted-license models (EmbeddingGemma) remain opt-in with flow-down.

## Honest boundary
The harness runs fully for CPU providers here; GPU-model adapters (PaddleOCR-VL, LightOnOCR, GLiNER-large, ColBERT) and endpoint providers can be conformance/seam-verified with mocks, but their real quality/throughput numbers need a GPU/perf environment (#64) — the bake-off produces those where the environment exists and flags them otherwise, never fabricated.

## Rejected
- A new "adapter" protocol distinct from the Capability protocol (the Capability already IS the I/O contract; a second layer is ceremony).
- The Runner orchestrating GPU placement/serving (ADR-0003: providers/endpoints own serving; the Runner is substrate-agnostic).
- A monolithic "all providers installed" env (defeats ADR-0016 thin-core; the matrix installs only the candidates under test).

## Status note — §2 realized (P3-F1, #86)

**§2 (`ProviderProfile` + device handling) is BUILT** (Phase-3 Wave-0, slice P3-F1). Delivered in `latence-core`:

- `latence_core.providers.profile.ProviderProfile` — a frozen, strict Pydantic v2 descriptor (`compute` cpu|gpu|either, `memory_mb`, `model_id`, SPDX `license` + `license_verified`, `deterministic`, `batch`, `cost_per_1k`), with the **UNVERIFIED invariant** enforced (`license == "UNVERIFIED"` ⇒ `license_verified` must be `False`, per ADR-0012). A provider declares it via a `profile` classmethod (preferred) or a `PROFILE` ClassVar; the `HasProfile` `runtime_checkable` Protocol + `profile_of()` read either without importing heavy deps.
- `latence_core.providers.device` — the **dep-light, torch-free** device seam (ADR-0016): `cuda_available()` is override-driven under `LATENCE_CUDA=0|1` (deterministic for CI) and defaults `False` when unknown; `select_device(requested, profile)` returns a typed `DeviceDecision` and **skips-with-flag (never raises)** a GPU-only provider on a CPU-only host — the G1 posture. Device *placement* stays the adapter's job.
- `ProviderRegistry.profile(name)` / `.profiles()` surface declared profiles (missing = `None`, never fabricated; an import-erroring provider is skipped-and-flagged, not fatal).
- The `LocalRunner` routes each Stage's `device` config value against its provider's profile before running; a skipped GPU stage produces an empty output + records `skipped=True` + the profile columns in the Quality Report's `StageMetrics` (QUALITY_SCHEMA_VERSION 16). The in-core reference providers (gazetteer, pattern, markdown chunker, intake screener) and `entity.gliner` declare honest profiles; `license_verified=True` appears only where genuinely verified.

§1 (`AdapterBase`), §3 (stacks), §4 (bake-off), §5 (conformance) remain **proposed** — the later Wave-0/Wave-1 slices (P3-F2…F5).

## Status note — §1 realized (P3-F2, #87)

**§1 (`AdapterBase`) is BUILT** (Phase-3 Wave-0, slice P3-F2). Delivered in `latence-core`, dep-light (no torch/model libs), as `latence_core.providers.adapter.AdapterBase` (re-exported from `latence_core`, `__all__`):

- **Optional, not a new seam.** A thin base carrying the cross-cutting glue every adapter repeats; a Provider still satisfies a Capability Protocol structurally **without** it (the in-core reference Providers are unchanged and untested-migrated). It names no Capability method — no second protocol layer.
- **Device resolution** — `super().__init__(config)` resolves `config.device` against the subclass's declared `ProviderProfile` via P3-F1's `select_device`; `self.device` exposes the `DeviceDecision` (skip-with-flag on a CPU-only host, never a fabricated device). Placement stays the adapter's job (ADR-0003).
- **Batching** — `batched(items, run_batch)` groups into a validated `batch_size` (≥1) when `profile.batch`, degrades to one-item calls otherwise; deterministic input-order output, one-batch-at-a-time (no whole-corpus buffer — the ADR-0033 streaming posture).
- **Native-exception → typed taxonomy (ADR-0034)** — `guard("<op>")` maps a native exception to `ProviderError` (runtime fault) or `ConfigError` (opted-in via `config_exceptions`), passes a framework `LatenceError` through unwrapped, and carries **no record content/PII** (provider + op + native *type name* only, never the native message).
- **Provenance/offset carry** — `carry_provenance` / `rebase_offsets` / `resolve_pages` reuse the existing `OffsetIndex` / `PageOffsetIndex` seam (ADR-0031), not a weaker copy.

**Also in P3-F2 (§2 malformed-profile fix, the deferred P3-F1 defect):** the `LocalRunner` now **FAILs a stage loud** (typed `ConfigError` + FAIL Quality Report + `config` span) when a Provider declares a **malformed** profile, instead of silently reading it as "no profile" and running. An **absent** profile still means follow-request (`None`); an import/load error still defers to the normal execution path. `profile_of` was hardened to recognise a profile **declaration** only as a genuine class-bound `@classmethod` (or `PROFILE` ClassVar / bare `ProviderProfile`), so the `Profiler` Capability's own `profile(self, …)` **instance** method is correctly read as "no declaration" rather than mislabelled malformed (a latent P3-F1 collision this slice surfaced and fixed).

§3 (stacks), §4 (bake-off), §5 (conformance) remain **proposed** — P3-F3…F5.

## Status note — §5 realized (P3-F3, #88)

**§5 (Provider Conformance suite) is BUILT** (Phase-3 Wave-0, slice P3-F3). Delivered in `latence-core`, dep-light (no torch/model/GPU libs), as the `latence_core.conformance` package + a parametrized pytest suite. "Passes conformance" *is* the definition of enterprise-ready, and the suite is the gate every Wave-1 adapter must pass.

- **Parametrized over EVERY registered Provider.** The suite discovers all 29 Providers via `ProviderRegistry.names()` and runs each against its Capability's `ConformanceCase`. A Provider is mapped to its Capability by name prefix, so a new Provider is auto-covered once its Capability has a case; a Provider whose Capability has **no** case (or an unknown prefix) FAILs **loudly** (`case_for_provider`/`capability_of` raise) — no silent coverage gap. A test asserts the exact 29-Provider roster + count so an accidentally-uncovered Provider is caught.
- **The C1–C6 checks** (all green for the in-core CPU Providers; the stub-subset green for endpoint/GPU): **C1** valid typed records (Provenance/Classification + in-range offsets, reusing the S10 completeness + ADR-0031 `PageOffsetIndex` seams — not a weaker copy); **C2** graceful failure (adversarial input → a typed `LatenceError` or a defined quarantine/PARSE_ERROR/empty path, **never a bare crash**, with **no PII** in the surfaced message — proven against a PII-bearing payload); **C3** license recorded + permissive-or-opt-in (every Provider now declares a `ProviderProfile`; `UNVERIFIED`+`verified` is impossible; a restricted license must be verified/opt-in-flagged); **C4** determinism-or-documented (a `deterministic` Provider is byte-identical over two runs; a non-deterministic one records its declaration); **C5** resource/device honored (a `compute="gpu"` Provider on a CPU-only host is **skipped-with-flag** via `select_device`, recorded, not run, not faked); **C6** no secret/PII leak (a non-Redactor does not surface raw PII in non-content record fields).
- **Every Provider now declares a `ProviderProfile`.** P3-F1 attached profiles to only a handful; P3-F3 completes the set — all 29 Providers declare honest, verified-where-verifiable profiles (the CPU references Apache-2.0/verified/deterministic; the GLiNER/endpoint Providers `compute="either"`, non-deterministic, `model_id=None` for the endpoint clients that bundle no weights, `cost_per_1k=None` where deployment-specific). The two `Profiler` Providers declare via a `PROFILE` ClassVar (not a `profile` classmethod) because the Profiler Capability's own I/O method is named `profile` (the P3-F2 collision).
- **Endpoint/GPU Providers are stub-scoped + flagged.** `*.endpoint`, `relation.llm`, `embedding.endpoint`, and `*.gliner` run the contract/graceful-failure/license/determinism-declaration subset against a deterministic openai/gliner stub (`latence_core.conformance.stub`, reusing the existing endpoint/gliner test-double pattern); each is flagged `stub_scoped` ("conformance-verified against stub; live numbers need a rig", #64) — never a fabricated quality/throughput number.
- **The checks BITE (proven, not asserted).** A throwaway broken Provider (declares `deterministic=True` but is not, and raises a bare `RuntimeError` on the adversarial input) FAILs C4 and C2; a Provider that leaks PII in its quarantine reason FAILs C2's no-PII gate; a synthetic `compute="gpu"` Provider is skipped-with-flag by C5 (its `chunk` is asserted never called). All in `test_conformance.py`.
- **Quality Report integration.** The report gains an optional `conformance` roll-up (`ConformanceReport`, `QUALITY_SCHEMA_VERSION` 17 — counts only, no record content, safe to persist/share). The report models live in the pure, stage-free `latence_core.conformance.report` so the lean Quality contract imports no harness/fixtures; the harness/cases are lazy-loaded from the package `__init__` so a bare `import latence_core` stays thin (no stage load).
- **CI.** The suite runs in `ci.yml` on py3.11 + py3.12 with `LATENCE_CUDA=0` (both inside the normal `pytest` step and as a dedicated visible gate). An E2E test runs a real pipeline through the `LocalRunner` and folds the conformance roll-up into its Quality Report.

A new adapter author's checklist is in `docs/CONFORMANCE.md`. §3 (stacks) and §4 (bake-off) remain **proposed** — P3-F4, P3-F5.

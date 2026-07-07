# P3-F1 — `ProviderProfile` descriptor + CPU/GPU device handling + registry surfacing

**Phase-3 Wave-0 foundation. GitHub issue #86. ADR-0036 §2 (+ §1 device note).**
This is the first foundation slice of the provider ecosystem: it gives every
provider a **declared, typed descriptor** and gives the framework a **dep-light
way to route device selection and skip GPU providers on a CPU-only host** — the
contract every Wave-1 adapter is written against. No adapters are built here.

## Why
"Model-agnostic" must be provable: the bake-off (P3-F5) needs license/cost/compute
columns; the conformance suite (P3-F3) needs to know a provider's declared bounds;
the stack harness (P3-F4) needs to skip-not-fail GPU providers where there is no
GPU (the G1 posture). All of that reads one small declared object. Build it first,
stable, so adapters and harnesses never churn.

## Scope (this slice only)

### 1. `ProviderProfile` — typed descriptor in `latence-core`
A frozen, strict-typed model (Pydantic v2, matching the existing contracts style)
in a new module `latence_core/providers/profile.py` (or `latence_core/profile.py`
if that fits the layout better — keep it dependency-light, **no torch/heavy deps**).
Fields, per ADR-0036 §2:
- `compute: Literal["cpu", "gpu", "either"]`
- `memory_mb: int` — approximate peak resident footprint (declared, not measured here)
- `model_id: str | None` — HF/model identifier, or `None` for pure-code providers (gazetteer, pattern)
- `license: str` — SPDX identifier (e.g. `"Apache-2.0"`, `"MIT"`) or the sentinel `"UNVERIFIED"`
- `license_verified: bool` — weights **and** code checked per ADR-0012 (research-diligence directive)
- `deterministic: bool`
- `batch: bool` — whether the adapter supports batched calls
- `cost_per_1k: float | None` — for endpoint/API providers; `None` for local
Validation: `memory_mb >= 0`; `cost_per_1k` if present `>= 0`; if `license == "UNVERIFIED"`
then `license_verified` MUST be `False` (no silently-verified unknowns). Frozen/immutable.

### 2. How a provider declares its profile
A provider class surfaces its profile via a **`profile` classmethod or a `PROFILE`
ClassVar** returning a `ProviderProfile`. Pick one and document it; a classmethod
(`@classmethod def profile(cls) -> ProviderProfile`) is preferred so a provider can
compute it from config. Add a small `runtime_checkable` `HasProfile` Protocol so the
registry can detect declaration structurally without importing the provider's heavy deps
beyond the class object it already loads.

### 3. Registry surfaces it
Extend `ProviderRegistry` (in `capability.py`) with:
- `profile(name: str) -> ProviderProfile | None` — load the class, return its declared
  profile, or `None` if it declares none (do **not** fabricate one). Cache alongside `_loaded`.
- `profiles() -> dict[str, ProviderProfile | None]` — map over `names()` **without** forcing
  a load of providers that error on import? No — loading is required to read the profile, so
  document that `profiles()` loads each provider (acceptable for tooling/bake-off; the lazy
  single-name path stays lazy). Skip-and-flag any provider whose import raises, never crash the map.

### 4. Device handling (dep-light)
- A `device` config value `auto | cpu | cuda` is threaded to adapters. **Device placement
  itself stays the adapter's job** (ADR-0003/0036 §2) — this slice does NOT move models.
- A tiny detection seam `latence_core/providers/device.py`:
  - `cuda_available() -> bool` — detect **without importing torch** (torch is not a core dep,
    ADR-0016). Probe order: honor an explicit `LATENCE_CUDA=0|1` env override first (for tests
    and CPU-only CI); otherwise best-effort detect (e.g. presence of `nvidia-smi` / a CUDA device
    file) and default to `False` when unknown. Must be deterministic under the env override.
  - `select_device(requested: str, profile: ProviderProfile | None) -> DeviceDecision` returning
    either the effective device (`"cpu"`/`"cuda"`) or a **skip** decision with a reason string
    when a `compute == "gpu"` provider is requested on a host with no CUDA. **Skip-with-flag,
    never raise** — the G1 posture. `either` → follow `requested`/availability; `cpu` → always cpu;
    `requested == "cuda"` on no-CUDA host → skip-with-flag (or explicit error only if the caller
    passed `cuda` explicitly — document which; default to skip-flag to match the stack harness).
- Represent the outcome as a small typed result (`DeviceDecision` dataclass:
  `device: str | None`, `skipped: bool`, `reason: str`), not a bare string, so harnesses can
  record the skip in a report.

### 5. Declare profiles for the existing in-core reference providers
So the seam is exercised end-to-end (not just unit-tested in the abstract), attach a
`ProviderProfile` to the **existing** in-core reference providers that are cheap to touch
(e.g. `entity.gazetteer`, `entity.gliner`, `relation.pattern`, a chunker, a screener):
`compute="cpu"`, honest `memory_mb`, correct `deterministic`, and a **verified** license for the
code-only ones (Apache-2.0, `license_verified=True`); for `entity.gliner` set `model_id` to the
GLiNER checkpoint and `license` per its verified terms — if not verifiable in this slice, set
`license="UNVERIFIED"`, `license_verified=False` (do NOT guess — that is the directive). Keep this
minimal: enough to prove `registry.profiles()` returns real data, not a sweep of all providers.

## Out of scope (later Wave-0 slices)
AdapterBase (P3-F2), the conformance suite (P3-F3), stacks + validation harness (P3-F4),
the bake-off matrix (P3-F5), and any new adapter package (Wave-1). Do not build these here.

## Definition of Done (Baseline bar + these)
- `ProviderProfile` + `DeviceDecision` are strict-typed, frozen, and exported from
  `latence_core` (`__init__` re-export + `__all__`); `mypy --strict` clean over src+tests.
- `ProviderRegistry.profile()` / `.profiles()` return declared profiles; missing = `None`,
  never fabricated; a provider that errors on import is skipped-and-flagged, not fatal.
- `cuda_available()` is override-driven under `LATENCE_CUDA` and defaults False when unknown;
  `select_device()` returns a **skip-with-flag** `DeviceDecision` for a GPU provider on a
  CPU-only host and **never raises** for that case. No torch import anywhere in core.
- Existing in-core reference providers declare honest profiles; `license_verified` is only
  `True` where actually verified (else `UNVERIFIED`/`False`).
- Deterministic seeded tests: profile validation (incl. the UNVERIFIED invariant), registry
  surfacing (declared / `None` / import-error-skipped), device routing (cpu / either / gpu-on-cpu
  skip-flag / `LATENCE_CUDA=1` cuda path), and an end-to-end assertion that a GPU-declared provider
  is skipped-with-flag by the routing path on this CPU host. No fabricated GPU numbers.
- ADR-0036 status note updated (§2 realized) and a short entry in the Phase-3 plan; `ruff` clean.

## Reviewer should check
- No heavy dep (torch/CUDA libs) pulled into `latence-core` (ADR-0016).
- The skip-with-flag posture is real (a GPU provider on this host is flagged, not crashed, not faked).
- `license_verified=True` never appears without genuine verification; `UNVERIFIED` invariant enforced.
- Determinism: `cuda_available()` fully controllable via env for CI reproducibility.

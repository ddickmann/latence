# ProviderProfile & device handling (P3-F1)

The Phase-3 Wave-0 foundation: every Provider declares a small, typed descriptor, and
the framework routes device selection from it — dependency-light, with **no torch/CUDA
in `latence-core`** (ADR-0016). Architecture: ADR-0036 §2.

## Declaring a profile

A Provider surfaces its `ProviderProfile` via a `profile` classmethod (preferred — it can
compute the descriptor from config) or a `PROFILE` ClassVar:

```python
from latence_core import ProviderProfile

class MyEntityExtractor:
    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="either",          # "cpu" | "gpu" | "either"
            memory_mb=2048,            # approx peak resident MiB (declared, not measured)
            model_id="vendor/model",   # or None for a pure-code provider
            license="Apache-2.0",      # SPDX id, or "UNVERIFIED"
            license_verified=True,     # True only when weights AND code checked (ADR-0012)
            deterministic=False,       # identical input+config -> identical output?
            batch=True,                # supports batched calls?
            cost_per_1k=None,          # per-1k cost for an endpoint provider; None for local
        )

    def __init__(self, config: dict | None = None) -> None: ...
    def extract(self, chunks): ...
```

`ProviderProfile` is frozen (immutable) and strict (`extra="forbid"`). The **UNVERIFIED
invariant** (ADR-0012 / the research-diligence directive) is enforced: if
`license == "UNVERIFIED"`, then `license_verified` MUST be `False` — an unverifiable
license is never silently declared verified, and never guessed.

## Reading profiles from the registry

```python
from latence_core import ProviderRegistry

reg = ProviderRegistry()
reg.profile("entity.gazetteer")   # -> ProviderProfile, or None if the provider declares none
reg.profiles()                    # -> {name: ProviderProfile | None} over every registered provider
```

`profile()` loads the provider class to read its declaration and caches the result (a
cached `None` means "declares none", never a fabricated default). `profiles()` maps over
every registered provider; a provider whose **import raises** (a missing heavy/optional
dep, ADR-0016) is **skipped-and-flagged** — omitted from the map — so one uninstalled
heavy provider never crashes the sweep.

## Device handling (dep-light, torch-free)

Device *placement* stays the adapter's job (it moves its own model to the resolved
device, ADR-0003/0036 §2). The framework only **routes** and **records**:

```python
from latence_core import cuda_available, select_device

cuda_available()                       # torch-free probe; override with LATENCE_CUDA=0|1
decision = select_device("auto", profile)   # -> DeviceDecision(device, skipped, reason)
```

- `cuda_available()` honors an explicit `LATENCE_CUDA=0|1` env override first (deterministic
  for CI and seeded tests), else a best-effort probe (`nvidia-smi` on PATH / an `/dev/nvidia*`
  node), defaulting to `False` when unknown. No torch import anywhere in core.
- `select_device(requested, profile)` returns a typed `DeviceDecision`. A `compute == "gpu"`
  provider requested on a **CPU-only host is skipped-with-flag** (`skipped=True` + a reason),
  **never raised** — the G1 posture, so the stack harness records the skip rather than
  crashing or fabricating a GPU number. `cpu` clamps to CPU; `either` (or no profile)
  follows the request against availability.

### In the pipeline

Set the requested device per Stage in the Pipeline YAML:

```yaml
  - name: entities
    capability: entity_extraction
    provider: entity.gliner
    depends_on: [chunk]
    config:
      device: auto        # auto | cpu | cuda
      labels: [person, organization]
```

The `LocalRunner` routes each Stage's `device` against the provider's profile before
running it. A skipped GPU-only Stage on a CPU-only host produces an empty output and
records `skipped=True` + the profile/device columns (`compute`, `model_id`, `license`,
`device`, `device_selected`, `skip_reason`) in the Quality Report's `StageMetrics`
(QUALITY_SCHEMA_VERSION 16). The skip is sticky across resume (an empty checkpoint is
written), so a re-run does not re-attempt the GPU stage; the profile/device columns and
the `skipped=True` flag are re-derived on the resume path too, so a resumed run still
reports the stage as skipped rather than as a plain non-skipped stage.

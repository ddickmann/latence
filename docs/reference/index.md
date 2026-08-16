# API Reference

Auto-generated from the source of `latence-core` (the framework spine) and a representative
Provider package. The core carries only the contract / serialization / IO / CLI spine and
near-zero dependencies ([ADR-0016](../adr/0016-thin-core-per-provider-packages.md)), so this
is the stable surface a Provider author and an embedding integrator program against.

| Module | What it defines |
| --- | --- |
| [Stage contracts (generated)](stage-contracts.md) | The generated per-Capability replacement contract — what every Stage receives and must emit, plus the exact steps to swap any Provider without breaking the flow. Machine-readable twin: [stage-contracts.json](stage-contracts.json). |
| [Data contracts](contracts.md) | The Pydantic record schema shared by every Stage — Provenance, Classification, the per-Stage Record types. |
| [Capability protocols](capability.md) | The typed `Protocol`s a Provider implements, plus the `ProviderRegistry`. |
| [Pipeline](pipeline.md) | The declarative DAG of Stages. |
| [Local Runner](runner.md) | The built-in files-on-Storage execution adapter. |
| [Storage seam](storage.md) | The fsspec-backed cloud-agnostic IO layer. |
| [Provider base & profile](providers.md) | `AdapterBase` and `ProviderProfile` — the optional adapter helpers and the enterprise-readiness descriptor. |
| [Errors & tracing](errors.md) | The typed error taxonomy and the thin tracing seam. |
| [Example provider](example-provider.md) | `parser.pdfplumber` — the reference Wave-1 adapter template. |

!!! tip "Docstring style"
    Source docstrings use Sphinx/RST cross-references (`` :class:`...` ``). Names render as
    written; follow the module links above for the parsed, structured API.

# Data contracts

The single Pydantic schema every inter-Stage Record shares, serialized to both JSONL and
Parquet/Arrow ([ADR-0006](../adr/0006-one-schema-dual-serialization.md)). `Provenance` and
`Classification` are the cross-cutting carriers every record holds.

::: latence_core.contracts

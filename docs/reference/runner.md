# Local Runner

The built-in Runner: it schedules Stages, keeping its state as files on Storage — no database
([ADR-0010](../adr/0010-local-runner-state-is-files-on-storage.md)) — and streams records so
memory stays bounded ([ADR-0033](../adr/0033-streaming-memory-bounded-runner.md)).

::: latence_core.runner

# S12 — Performance path + Runner adapter

**Parent**: S9 · **Phase 1 · Baseline bar**

## What to build
Two proofs of the seams' generality. (1) **Performance Provider path**: for the heavy Capabilities (Parse, NER, RE, Embedder), a Provider that targets any OpenAI-compatible / vLLM / Triton endpoint, selectable by one config/URL change — same Pipeline, no code edit. (2) **Runner adapter**: run the identical Stages under a second Runner — Databricks Workflows *or* Airflow — proving execution is pluggable (ADR-0003) and the adopter keeps their orchestrator.

## Ports (algorithmic assets)
- Endpoint lifecycle / ref-counted lease (`pipeline-worker/endpoint_lifecycle.py`) → PORT-WITH-HARDENING, generalized off RunPod.

## Acceptance criteria
- [ ] Each heavy Capability has an endpoint-backed performance Provider; switching CPU↔endpoint is one config change with identical outputs (modulo model).
- [ ] The same Pipeline runs end-to-end under the second Runner adapter against the sample corpus.
- [ ] Runner adapter uses the platform's own state/scheduling, not the local files-on-Storage state (ADR-0010).
- [ ] Endpoint failures are retried/surfaced without corrupting the Corpus Version.
- [ ] Baseline bar met.

## Blocked by
S9.

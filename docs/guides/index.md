# Guides

Task-focused walkthroughs for operating and extending a Latence stack. New to the
framework? Start with [Getting Started](../getting-started/index.md) and
[Concepts & Architecture](../concepts/index.md) first.

## Operate a stack

<div class="grid cards" markdown>

- :material-check-decagram: **[Run & validate a stack](../running-a-stack.md)**
  Assemble a full-stack Pipeline config and validate it end-to-end with
  `latence stack validate`.

- :material-tune: **[Tune a stack per device](../tuning.md)**
  Auto-tune batch sizes and device knobs per host with `latence tune`.

- :material-trophy: **[Run a bake-off](../running-a-bakeoff.md)**
  Compare Providers on a Capability — quality, throughput, license, and cost in one table.

</div>

## Deployment & serving

<div class="grid cards" markdown>

- :material-server-network: **[Serving LightOn-OCR with vLLM](../serving-lighton-vllm.md)**
  The production OCR path: serve `parser.lighton_vllm` behind an OpenAI-compatible endpoint.

- :material-expansion-card-variant: **[GPU testing guide](../GPU-TESTING-GUIDE.md)**
  Run the whole learned/GPU system yourself, validated on a CUDA pod.

</div>

## Integrate

<div class="grid cards" markdown>

- :material-database-import: **[Retrieval-engine input specification](../RETRIEVAL-ENGINE-INPUT-SPEC.md)**
  The exact contract a downstream retrieval engine consumes from a Latence run — the shape
  of the RAG-ready corpus and knowledge graph.

- :material-graph: **[Graph-rescue retrieval](graph-rescue-retrieval.md)**
  The production graph-at-query-time architecture: base dense+BM25, a policy-gated graph
  lane that appends (never fuses), and a token-budget selector — measured defaults, zero
  LLM calls online.

- :material-vector-combine: **[Embedders](../embedders.md)**
  The opt-in RAG-corpus vector augmentation and how it plugs into Export.

</div>

## Enterprise readiness

"Enterprise-ready" is not a marketing claim in Latence — it is the **Provider Conformance**
gate ([Authoring a Provider §5](../providers/README.md)) plus the declared, license-verified
[`ProviderProfile`](../reference/providers.md) on every Provider
([ADR-0036](../adr/0036-provider-ecosystem-and-bakeoff.md)). Together they mean: every
Provider fails gracefully with typed errors, leaks no PII, honors its device declaration,
and carries a *verified* weights-and-code license
([ADR-0012](../adr/0012-permissive-only-reference-providers-gemma-opt-in.md)). Each run's
[Quality Report](../adr/0008-quality-report-as-first-class-artifact.md) folds the
conformance result in, so the "AI-ready" claim is substantiated per run, not asserted.

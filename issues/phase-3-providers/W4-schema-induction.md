# W4-schema-induction — unsupervised label generation (LabelInducer) via an OpenAI-compatible LLM

**Restores the "point it at a folder and it discovers its own schema" property from the superpod repo.**
A NEW optional Stage `schema_induction`, inserted right after Content Screening and before the
extraction stages, that uses a **small OpenAI-compatible LLM with structured JSON output** to induce,
per document, the **entity types / relation types / PII types** present in the data — then feeds them to
Entity Extraction, Relation Extraction, and Redaction. The extraction stages are NOT rewritten: they
gain one additive, backward-compatible line that reads induced labels off the chunk (like `risk_markers`)
and **extends/overrides** their config labels. Off by default; existing stacks are byte-identical.

## Model-agnostic LLM seam (HARD REQUIREMENT — OpenAI ⇄ OpenRouter ⇄ on-prem vLLM)
The inducer LLM call MUST go through the **official `openai` SDK** with config `model` / `base_url` /
`api_key` — the SAME pattern the shipped `relation.llm` uses — so a client switches OpenAI → OpenRouter
→ local vLLM by config alone (all speak the OpenAI chat-completions API). Reuse
`latence_core.endpoint.call_with_retry` + `RetryPolicy` for fault tolerance (as `relation.llm` does).
- **Default `model` = `gpt-4.1-mini`** (NOT gpt-4o-mini). `base_url=None` ⇒ OpenAI; set to
  `https://openrouter.ai/api/v1` (OpenRouter) or `http://<vllm-host>:8000/v1` (on-prem, air-gapped).
- `api_key` from config or `$OPENAI_API_KEY`. Nothing hard-coded; lazy `import openai` (discovery stays
  import-light). License: `openai` client Apache-2.0; the model is a service (no weights license).

## The carrier (additive contract — no extractor rewrite)
- New contract **`InducedLabels`** (Pydantic, schema-versioned): `entity_types: list[str]`,
  `relation_types: list[str]`, `pii_types: list[str]`, plus `model_id: str` + per-type `confidence`
  (auditable). Add optional **`induced_labels: InducedLabels | None`** to `ChunkRecord` (additive field,
  SCHEMA_VERSION bump — mirrors the existing `risk_markers` annotation precedent). Default `None` ⇒
  every existing stack + record is unchanged.

## The Stage + Capability
- New Capability **`LabelInducer`**: `induce(chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]` —
  consumes chunks, returns the SAME chunks with `induced_labels` populated (a chunk→chunk transform, the
  Content-Screening shape). Corpus/doc-level: groups chunks by `document_record_id`, one LLM call per
  document (default), broadcasting the doc's induced labels to all its chunks. `granularity: doc|chunk`
  config (default `doc` — cheaper, plenty data-aware; `chunk` = one call per chunk).
- Runner wiring: `schema_induction` is an ordinary Stage routed by `depends_on` (it emits ChunkRecords;
  `entities`/`redact` depend_on it instead of `content_screen`). OPTIONAL: a stack without it is
  byte-identical to today (tested). Deterministic + seeded (temperature 0; stable label order).

## The extractor seam (additive, backward-compatible — the ONLY change to committed stages)
`entity.gliner`, `relation.gliner_relex`, `relation.llm`, `redaction.gliner_pii` (and
`redaction.gliner2` when merged) gain ONE label-source line: the effective label set is
**`config_labels ∪ chunk.induced_labels.<slot>`** when the chunk carries induced labels, else
`config_labels` (100% backward-compatible). Config labels are the **mandatory floor** ("presets"),
induced labels **extend** them (union, deduped, order-stable). A stage with NO config labels + induced
present becomes fully unsupervised; with config labels it's supervised-plus-augmented. The extraction
LOGIC is untouched — only where the per-chunk label list comes from. `auto_generate_labels: true` on a
stack turns the whole thing on; with it off (or no `schema_induction` stage) nothing changes.

## Structured output + SAFETY NETS (the hard requirements — non-negotiable)
The LLM is UNTRUSTED output. The inducer MUST be robust to any response:
1. **Structured output:** request `response_format={"type":"json_object"}` (portable across
   OpenAI/OpenRouter/vLLM; strict `json_schema` is NOT universally supported). Prompt pins the exact
   shape `{"entity_types":[...],"relation_types":[...],"pii_types":[...]}`.
2. **Generous token budget (no cut-off):** `max_tokens` default **2048** (configurable, e.g. up to
   4096); DETECT truncation via `finish_reason == "length"` → ONE retry at doubled budget → if still
   truncated, fall back to config labels + log (never a truncated-JSON crash).
3. **Parse-and-repair:** `json.loads` in a guard; on `JSONDecodeError` do ONE corrective retry ("return
   ONLY valid JSON of this exact shape"); on a second failure fall back to config labels. NEVER `eval`,
   NEVER execute model text.
4. **Schema validation:** the parsed object MUST be `{entity_types,relation_types,pii_types}` each a
   list of strings; anything else → reject → fall back. Extra keys ignored.
5. **Label sanitization (anti-injection / anti-bloat):** per label — coerce to `str`, strip control
   chars + surrounding whitespace, cap length (**≤64 chars**), drop empty/whitespace-only, dedupe
   case-insensitively, lower/title-normalize for GLiNER; cap COUNT per type (**≤40**), dropped items
   logged (no silent truncation). A label is data for a downstream zero-shot prompt, never executed —
   but bound it so a malicious model can't bloat the extractor prompt or smuggle megabytes.
6. **Fail-open to config, never crash (G1 posture):** ANY inducer failure (network, auth, malformed,
   truncated-twice, schema-invalid, empty) → the chunk keeps its config labels (or no labels) and the
   run continues; the failure is a typed, no-content log + a Quality-Report note. A missing/unreachable
   endpoint is a skip-with-flag, not a crash.

## Prompts
Ship a strong default prompt (system + user template) engineered for the exact JSON shape, few-shot
anchored, instructing: infer ONLY types actually evidenced in the text, lower/title-case, no explanations,
JSON only. Make it **overridable** via config (`system_prompt` / `user_prompt_template` with
`{text}` / `{max_labels}` placeholders) so the operator's own prompt engineering drops in. (Maintainer
note: default prompts written fresh here — swap in the superpod prompts when provided.)

## Anti-false-green + tests (verify THOROUGHLY)
- Offline: a **faithful mock `openai` client** returning the real chat-completions response shape
  (`choices[0].message.content` = JSON string, `finish_reason`). Tests cover EVERY safety net with
  fixtures: valid JSON; malformed JSON → repair-retry → success; malformed twice → fall back; truncated
  (`finish_reason=length`) → budget-bump retry; schema-invalid (list of ints / string not list / missing
  key) → fall back; injection/oversized/control-char labels → sanitized; empty → fall back; network/auth
  error → fail-open. Plus: the extractor seam (induced ∪ config; absent → config unchanged — the
  backward-compat proof); determinism (temperature 0, stable order, byte-identical two runs); an e2e
  through the runner with the mock (chunks → induced labels → extractors use them). A provider that
  skips a safety net (e.g. no truncation check, or `eval`s the output) MUST fail a test.
- `bash scripts/verify-local.sh` green (roster +1; new `label_induction` capability + `label_inducer.llm`
  provider; conformance C1–C6; stack-validate). ADR (new Capability + induced-labels contract +
  extractor-seam decision + the OpenAI-compatible/on-prem requirement). `matrix/` + a `stacks/` opt-in
  example (`auto_generate_labels: true`). `scripts/validate-schema-induction-on-gpu.md`.
- **Real-model validation (maintainer, needs `OPENAI_API_KEY`):** run `label_inducer.llm` with
  `model=gpt-4.1-mini` over a real document; confirm the induced `{entity_types,relation_types,pii_types}`
  are sensible + the extractors then run with them. (Deferred to a real key; the mock proves the
  machinery + every safety net offline.)

## Definition of Done
- `label_induction` Capability + `schema_induction` Stage + `label_inducer.llm` provider (OpenAI SDK,
  `model`/`base_url`/`api_key`, default gpt-4.1-mini, `call_with_retry`); `InducedLabels` contract +
  `ChunkRecord.induced_labels` additive field; extractor label-source seam (induced ∪ config,
  backward-compatible); runner wiring; OPTIONAL (stack without it byte-identical — tested).
- ALL safety nets implemented + tested against fixtures (structured output, token budget + truncation
  retry, parse-repair, schema validation, label sanitization, fail-open-never-crash). Faithful mock;
  un-safe provider FAILS a test. Determinism byte-identical.
- `bash scripts/verify-local.sh` green; ADR; stack opt-in example; pod-validation script. Real
  gpt-4.1-mini numbers come from the maintainer's keyed run — never fabricated.

## Reviewer should check
- The LLM call is OpenAI-SDK + `base_url`/`api_key` configurable (OpenAI/OpenRouter/vLLM switch) —
  the on-prem requirement — and default model is gpt-4.1-mini.
- EVERY safety net bites (a fixture per: malformed, truncated, schema-invalid, injection-label,
  empty, network-error) and the run NEVER crashes on bad LLM output (fail-open to config).
- The extractor seam is additive + backward-compatible (existing stacks byte-identical; induced ∪
  config; no induced ⇒ config unchanged) — the "don't rewrite verified stages" invariant.
- Structured output is `json_object` + validate/repair (portable), max_tokens generous + truncation
  detected, determinism (temp 0). Prompts overridable.

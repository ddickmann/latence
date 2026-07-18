# Guided setup is a deterministic wizard that emits ONE opinionated, quality-baked stack config, run by `latence process`

**Status: accepted** — W15-setup-wizard; **amended (profiles)** — the wizard now offers a
`--profile` choice: `lite` (the original CPU-first stack, unchanged and default) or
`enterprise-sota` (the pod-validated composed stack of ADR-0046 / `stacks/gpu-sota-glinker.yaml`),
see §7. Adds two CLI commands on the existing `latence` app —
`latence setup` (a guided, deterministic wizard that converges to a clean, validated, quality-baked
stack config) and `latence process` (load + run a saved config) — plus a pure config generator
(`latence_core.setup`). Builds on ADR-0015 (the declarative Pipeline contract this generates),
ADR-0036 §3 (the `stack validate` machinery the dry validation reuses), ADR-0038 (schema induction —
the mandatory quality lever), ADR-0039 (context enrichment — the enriched RAG chunk stream the
corpus is exported from), ADR-0016 (thin core — the generator is pure-Python, zero extra deps),
ADR-0007 (CPU-first — setup is offline; no model/GPU/key touched at generate time).

## Context

The framework already produces a pod-validated GOOD stack (see
`docs/TUTORIAL-fresh-pod-walkthrough.md` §3), but reaching it means hand-authoring ~150 lines of YAML
with the exact quality-critical settings (chunk 640/80, induction granularity `corpus`, the GLiNER2
threshold/window, financial-PII labels, the context-enrichment wiring, `context_columns`) correct.
That is a wall for a new adopter and a source of the exact miswirings the pod post-mortem found. The
one-command story the tutorial promises needs a first-class command.

The constraints:

1. **Opinionated, not a blank canvas.** The wizard must converge to THE quality-baked stack, not
   expose every knob. The spine is fixed; only a small number of real choices are surfaced.
2. **One honest safety toggle.** PII/guardrail handling is the single optional axis a
   non-specialist actually reasons about. It must add/omit exactly the right stages and nothing else.
3. **Deterministic + scriptable.** A `--non-interactive` mode must produce a **byte-identical**
   config for identical inputs, so CI/repro can pin it.
4. **Validated before saved, offline.** The generated config must be checked with the SAME machinery
   `stack validate` builds on — but WITHOUT running the pipeline (no corpus, no models, CPU-only).
5. **No secrets in the file.** The induction API key is read from the environment at run, never
   written into the config.

## Decisions

### 1. Two commands on the existing app; a pure generator behind them

`latence setup` prompts (typer) with a sensible default + a one-line "why it matters" per question,
resolves them into a frozen `SetupAnswers`, and `latence_core.setup.build_stack_config` turns that
struct into stack YAML. `latence process [--config ./latence.stack.yaml]` loads the saved config and
runs it through the SAME execution path as `latence run` (a shared `_execute_pipeline_file` helper).
The generator is a **pure function of `SetupAnswers`** — it reads no env, clock, or randomness — so
the interactive and `--non-interactive` paths share one code path and identical answers yield
identical bytes.

### 2. A fixed mandatory spine + ONE optional toggle

The generated stack always carries the mandatory spine: `source → intake_screen → parse →
chunk[640/80] → induce[label_inducer.llm, granularity=corpus] → extract[fused_entity_relation.gliner2,
thr 0.5, max_len 768] → profiling → disambiguate[disambiguation.embedding] → graph[graph.canonical]
→ enrich[context.kg_header] → export_kg[export.knowledge_graph] + export_corpus[export.jsonl_parquet,
context_columns:true]`. The **one optional toggle** is PII redaction + guardrail screening: ON adds
exactly `content_screen` (`screening.content_keyword`) + `redact` (`redaction.gliner2`, financial-PII
label floor); OFF omits exactly those two (and reroutes `induce` from `content_screen` to `chunk`,
and `enrich` from `redact` to `induce` — see §6). No other stage moves. The parser is the one
per-corpus provider choice (`parser.render` universal / `parser.pdfplumber` PDF-text /
`parser.lighton` OCR), recommended from the auto-detected file types.

### 3. Determinism via ordered-dict → `yaml.safe_dump(sort_keys=False)`

The generator builds an ordered Pipeline mapping and serializes with `sort_keys=False`, so field/stage
order is exactly the dict's insertion order. Byte-identical output for identical answers is a tested
contract (`--non-interactive` twice ⇒ equal bytes), which is what makes the config CI-pinnable.

### 4. Dry validation reuses the `stack validate` machinery — no pipeline run

`validate_config_text` (1) parses into a `Pipeline` (Pydantic + the DAG validator: unknown-dep /
cycle / duplicate-name), then (2) runs `check_stack_capabilities` (ADR-0036 §3's capability guard) —
a torch-free, no-instantiation class-vs-Protocol check. An un-installed heavy Provider is skipped
(device-honesty), never a spurious failure, so validation passes on a CPU-only host while still
biting a real miswiring. Setup fails loudly (no broken config saved) if either step fails.

### 5. The key is read from the environment, never stored; unset is a clear warning

The induction stage reads `$OPENROUTER_API_KEY` (OpenRouter base URL) or `$OPENAI_API_KEY` at run.
`setup` confirms the appropriate var is set and prints a clear, non-crashing WARNING if not — the
config is still generated. The generated file's header names the var and states the key is never
stored.

### 6. PII posture — the RAG chunks ARE masked (W16, ADR-0042)

With PII ON, `redact` is a **chunk-level transform** (W16, ADR-0042): it consumes the induced
**chunk** stream and sets each chunk's `masked_content` (each chunk ≤ the model window, so no PII is
truncated), and `enrich` reads THAT redacted chunk stream (`redact → enrich → export_corpus`). The
corpus Export materializes `masked_content`, so the exported RAG corpus is **both masked (PII-safe)
AND context-enriched** — the per-chunk context headers and the masking compose, with no trade-off.
This closes the pre-W16 gap where the RAG chunks were enriched but un-masked (and the doc-level
redactor could truncate PII past the model window). `latence setup --pii` now produces a genuinely
PII-safe, context-enriched corpus; the file header and the CLI summary state this accurately.

### 7. Profiles — `lite` (default) vs `enterprise-sota` (amendment)

The generator branches **once** on `answers.profile`. `lite` is §2's stack, unchanged and
byte-identical to before (the default when `--profile` is unset). `enterprise-sota` emits the
pod-validated composed stack (ADR-0046, `stacks/gpu-sota-glinker.yaml`) so `latence setup` →
`latence process` produces the SOTA stack out of the box rather than the lite path. Every seam swaps
to its enterprise Provider while the spine shape, the ONE PII toggle, the W18 mask-before-extract
option, determinism, and the dry-validation contract all carry over unchanged:

* **parse → `parser.lighton_vllm`** (served-vLLM OCR). The wizard prompts for the endpoint
  `base_url` (default `http://localhost:8000/v1`) and `model` (default `lightonocr`); `api_key`
  defaults to `EMPTY`, `max_concurrency` to `8`. This is the one new per-deployment input the SOTA
  profile adds (it replaces lite's parser-recommendation prompt).
* **extract → `fused_entity_relation.gliner2`** (`threshold 0.3`, labels floor
  `[person, organization]`, `compile` left to the provider default `False`); **redact →
  `redaction.gliner2`** (§7a — the same Provider `lite` emits, and the same W16 chunk-level
  `redact → enrich → export_corpus` dataflow).
* **disambiguate → `disambiguation.glinker`** — the neural GLinker linker + audited resolver
  (`context_window 200`, `threshold 0.3`, reranker + FlashDeBERTa on) with the IBM Granite r2
  embedder (ADR-0045). `use_embedding_merge` / `min_similarity` are **not emitted** — the provider
  defaults (embedding-merge OFF, fuzzy + distinguishing-token guard ON at 0.84) are precise by
  default, so the config stays minimal and the ER is precise out of the box (ER-audit posture).
* **complete → `graph_completion.ultra`** (`skip_if_unavailable: true` — a missing, un-vendored
  checkpoint skips-with-flag, never fails the run), feeding `export.knowledge_graph`; the corpus
  Export keeps the Granite r2 vectors.

### 7a. The blessed redact Provider — one decision, both profiles (amendment, #197)

**Blessed redact Provider (both profiles): `redaction.gliner2`.** (Pinned in code as
`latence_core.setup.wizard.BLESSED_REDACT_PROVIDER`, and asserted against this line by
`packages/latence-core/tests/test_setup_wizard_redactor.py` — the decision is a contract between
this ADR and the generator, not a comment either side can quietly outgrow.)

The generator had drifted from its own design: §2 above, the `wizard.py` module docstring, and the
`FINANCIAL_PII_LABELS` constant all named `redaction.gliner2`, while `_build_sota_stages` emitted
`redaction.gliner_pii` — a different package, a different model library, a different checkpoint for
the same Stage. Under §1's "the wizard's ONE opinionated config must BE the documented design" that
is a defect, not a preference. The decision, and why:

* **One engine, not two (ADR-0036 W9).** Both profiles already run `fused_entity_relation.gliner2`
  for `extract`. Blessing the GLiNER2 redactor means the stack resolves **one** library, **one**
  mdeberta-v3 encoder family, and **one** version pin for extraction and redaction, instead of
  additionally pulling the separate `gliner` library solely for `redact`. W9 named gliner2 the
  unified extraction+PII engine; this makes the guided setup actually reflect that.
* **The choice is only now a real one (#192).** `redaction.gliner2` previously failed at model load
  against gliner2 ≥ 1.3 with a native `AttributeError`, so "gliner_pii in the SOTA profile" was a
  forced pick, not a decision. #192 restored it and moved the pin into the Provider package
  (`gliner2[local]>=1.3,<2` + a typed `ProviderError` for a hand-installed environment the resolver
  never saw, ADR-0016/0034), so drift now fails at dependency resolution.
* **The rejected candidate is not deprecated.** `redaction.gliner_pii` remains a fully supported
  Provider behind the same `PIIDetector` seam, and #191 fixed its device placement (it honours the
  resolved `cuda` device at load instead of silently running on CPU). Hand-written stacks that wire
  it — `stacks/gpu-sota.yaml`, `stacks/gpu-sota-pod.yaml`, `stacks/gpu-sota-glinker.yaml`,
  `stacks/schema-induction.yaml` — are unchanged and still valid. This is a decision about what the
  **guided setup blesses**, so the enterprise-sota profile now differs from `gpu-sota-glinker.yaml`
  on exactly this one Stage (a deliberate, recorded delta, noted in `_build_sota_stages`).
* **Licensing verified separately (ADR-0012).** `gliner2` framework **code** is Apache-2.0
  (github.com/fastino-ai/GLiNER2 LICENSE + the PyPI OSI classifier); the default checkpoint
  `fastino/GLiNER2-Guardrails-PII-Multi` **weights** are Apache-2.0 and ungated (HF model card).
  Both verified in ADR-0036 W3 and recorded in `THIRD-PARTY-LICENSES.md` — no new license surface,
  since the profile already ships this framework for `extract`.
* **The ADR-0044 financial-PII floor is untouched by the choice.** The floor lives at the single
  shared `finalize_redacted_chunk` seam every Redaction Provider funnels through, so it is
  non-bypassable by backend — a blessed model that misses an IBAN/card/SSN still emits it masked.
  The regression suite proves that through *this* Provider with the config the wizard *actually
  emits* (a nothing-detecting fake model, an IBAN still masked as `floor:iban`), rather than
  inheriting the guarantee by assumption.

The emitted file header now names the redactor it wires, so the generated config is self-consistent
for the operator reading it too.

Every learned Stage runs on **CUDA** (the profile pins `device: cuda` — it is a GPU-pod stack); on a
CPU-only host those Stages are skip-with-flag at `latence stack validate` (device honesty, P3-F1),
and the generated config still passes the wizard's dry validation + `check_stack_capabilities`
(un-installed heavy Providers are skipped, installed ones are class-vs-Protocol checked). The
determinism contract holds per profile: `--non-interactive --profile enterprise-sota` with the same
flags is byte-identical across runs.

## Consequences

* A new adopter reaches the pod-validated GOOD stack in two commands, or scripts it deterministically
  in CI. The opinionated spine removes the miswiring class the pod post-mortem found.
* The generator stays pure and offline; the only I/O (prompts, env-key check, file write, run) lives
  in the CLI. Tests cover determinism, the PII-toggle delta, the always-present spine, `process`
  load+run, and the missing-source / unset-key paths.
* With W16 (ADR-0042) the chunk-level redactor composes with enrichment, so "PII ON" now means the
  RAG chunks ARE masked. The file header, CLI summary, this ADR, and the tutorial all state the
  masked-AND-enriched posture accurately — the pre-W16 "chunks NOT masked" caveat is retired.

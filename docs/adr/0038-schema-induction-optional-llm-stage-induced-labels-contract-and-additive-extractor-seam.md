# Schema Induction is an OPTIONAL LLM Stage; induced labels are an additive record contract UNIONed onto the extractors' config-label floor

**Status: accepted** — W4-schema-induction. Adds the `LabelInducer` Capability + a `schema_induction`
Stage (a reference Provider `label_inducer.llm`, a small OpenAI-compatible LLM with structured JSON
output), the additive `InducedLabels` contract + `ChunkRecord`/`DocumentRecord.induced_labels` field,
and the one additive line each extractor gains to UNION induced labels onto its config labels. Builds
on ADR-0004 (Capability protocols + Provider plugins), ADR-0007 (CPU-first — the endpoint is the
performance path, nothing GPU forced into first-run), ADR-0013 (the LLM extractor pattern — same
`openai`-SDK-over-`base_url` seam `relation.llm` uses), ADR-0016 (thin core, heavy deps isolated
per-package), ADR-0022/0023/0024 (the extractor record contracts this seam feeds), ADR-0030 (ported
endpoint `call_with_retry`), ADR-0036 (Provider ecosystem: profile, conformance, bake-off).

## Context

The superpod repo had a property this framework had lost: **point it at a folder and it discovers its
own schema**. Every shipped extractor is zero-shot but **supervised by config** — the operator must
hand-write the entity/relation/PII label sets per Pipeline. On a messy, unknown corpus the operator
does not yet know the schema; forcing them to guess it defeats the "messy → AI-ready" promise.

The restoration must not compromise two invariants the framework is built on:

1. **Don't rewrite verified Stages.** The extraction Providers (entity/relation/PII) are tested and
   hardened; re-plumbing their logic to be "unsupervised" would re-open closed risk. The change must
   be *additive* — a single label-source line — and existing stacks must stay **byte-identical**.
2. **The LLM is untrusted output.** An LLM inducing the schema is a new, adversarial surface: a
   malformed / truncated / injection-bearing / schema-invalid reply must NEVER crash the run or
   smuggle unbounded data into a downstream prompt. The G1 posture is fail-open-to-config.

## Decisions

### 1. A new `LabelInducer` Capability — an OPTIONAL chunk→chunk Stage after Content Screening

Schema induction is its own Capability (ADR-0004), shaped like Content Screening: `Iterable[
ChunkRecord]` in / `Iterator[ChunkRecord]` out, deterministic. It consumes the run's chunks and
returns the **same** chunks with `induced_labels` populated (a doc-level Provider groups chunks by
`document_record_id`, makes one LLM call per document by default, and broadcasts the doc's induced
labels to all its chunks). It is inserted **after Content Screening and before the extraction
Stages** — so induction reads clean, screened text and its output feeds Entity/Relation/Redaction. It
is **genuinely OPTIONAL**: a stack with no `schema_induction` Stage is byte-identical to today (the
`induced_labels` field defaults `None`, and every extractor with a `None` field uses its config
labels exactly as before). Proven by an e2e test that runs the same pipeline with and without the
Stage and asserts the extractor is asked for identical labels.

### 2. The carrier — an additive `InducedLabels` contract + a defaulted record field (SCHEMA_VERSION 14)

A new versioned Pydantic `InducedLabels` (`entity_types` / `relation_types` / `pii_types` +
`model_id` + optional per-label `confidences`) is attached to a record via a new optional
`induced_labels: InducedLabels | None` field on **both** `ChunkRecord` (the entity/relation
extractors read it) and `DocumentRecord` (the per-document Redaction Stage reads it). This mirrors the
`risk_markers` annotation precedent exactly: a field on the record contract (not an out-of-band note),
so it survives serialize/deserialize + into the corpus and a downstream Stage reads it off the record
it already holds. Default `None` ⇒ every existing record + stack is unchanged; `SCHEMA_VERSION` bumps
14 (additive, defaulted — a v13 record JSON still validates).

The `schema_induction` Stage is chunk→chunk, so it populates `ChunkRecord.induced_labels` directly
(the entity/relation extractors read it off the chunk they already hold). Redaction, however, reads a
per-document `DocumentRecord` (the whole-document markdown + page map its PII spans key to), not the
chunk stream. The **chunk→document bridge** (`latence_core.induced.project_induced_onto_documents`,
applied by the Runner in the Redaction dispatch) stamps each document's chunk-carried induced labels
onto its `DocumentRecord` when the `redact` Stage `depends_on` BOTH `parse` (the documents) AND the
`schema_induction` Stage (the induced chunks) — first-populated chunk per document wins, mirroring
`relation.llm`'s doc-level read. So `DocumentRecord.induced_labels` is populated exactly when a stack
wires that edge (`stacks/schema-induction.yaml`: `redact depends_on: [parse, induce]`); dropping the
edge (or the whole Stage) leaves it `None` and Redaction uses its config PII floor — byte-identical.

### 3. The extractor seam — `config_labels ∪ induced`, one additive backward-compatible line

The four extraction Providers (`entity.gliner`, `relation.gliner_relex`, `relation.llm`,
`redaction.gliner_pii`) gain ONE line: their **effective** label set is `config_labels ∪
record.induced_labels.<slot>` when the record carries induced labels, else `config_labels`. This is a
pure function in core (`latence_core.induced.effective_labels`) — one seam, never a forked per-package
union — that config labels are the **mandatory floor** (order-first) and induced labels **extend**
them (deduped, order-stable). A record with no induced labels returns *exactly* its config labels
(the byte-identical backward-compat proof). A Stage with an empty config floor + induced labels is
fully **unsupervised**; with a config floor it is supervised-plus-augmented. The extraction LOGIC is
untouched — only where the per-record label list comes from.

### 4. The model-agnostic seam — `openai` SDK over `base_url`/`api_key`, default gpt-4.1-mini

The reference Provider `label_inducer.llm` calls the **official `openai` SDK** with config `model` /
`base_url` / `api_key` — the SAME pattern `relation.llm` uses (ADR-0013) — so a client switches
OpenAI → OpenRouter → on-prem/air-gapped vLLM by config alone (all speak the OpenAI chat-completions
API). Default `model` is **`gpt-4.1-mini`** (a served-model id, not a bundled checkpoint);
`base_url=None` ⇒ OpenAI, set it for OpenRouter/vLLM. Fault tolerance is the ported
`call_with_retry` + `RetryPolicy` (ADR-0030), identical to the other endpoint Providers; the
transient-retry count surfaces on `endpoint_retries`. The `openai` client is Apache-2.0 (code); the
Provider bundles **no weights** (the model is a service). The import is deferred to first use so
discovery stays import-light (ADR-0016), and the heavy dep ships in the SEPARATE
`latence-schema-inducer` package.

### 5. Structured output + the mandatory safety nets (the LLM is untrusted)

The Provider is robust to ANY response, each net independently tested against a faithful `openai`
mock (a provider skipping a net FAILS a test):

- **Structured output** — `response_format={"type":"json_object"}` (portable across
  OpenAI/OpenRouter/vLLM; strict `json_schema` is not universal); the prompt pins the exact
  `{entity_types,relation_types,pii_types}` shape.
- **Generous budget + truncation detection** — `max_tokens` default 2048; a `finish_reason ==
  "length"` reply retries ONCE at doubled budget, then falls back (never a truncated-JSON crash).
- **Parse-and-repair** — `json.loads` in a guard; one corrective retry on malformed JSON, then fall
  back. The parser is also **wrapper-tolerant**: a JSON object the model buried in a Markdown fence
  (```` ```json … ``` ````) or in surrounding prose is peeled out and re-parsed with the SAME strict
  `json.loads` (it never loosens parsing — hallucinated noise still fails), so good labels wrapped in
  chatter are recovered instead of silently discarded. NEVER `eval`, NEVER execute model text.
- **Schema validation** — the parsed object must be `{entity_types,relation_types,pii_types}`, each a
  list of strings; anything else rejects → fall back. Extra keys ignored.
- **Label sanitization** — per label: strip control/format chars, collapse whitespace, cap length ≤
  64, drop empties, dedupe case-insensitively; cap count per type ≤ 40, dropped items logged (no
  silent truncation). Non-ASCII label text (German/Unicode, e.g. `Rechnungsnummer`) is PRESERVED —
  only control/format chars are stripped — so a valid non-ASCII schema is never over-stripped to
  empty. A label is inert data for a downstream prompt, bounded so a malicious model can't bloat it
  or smuggle megabytes.
- **Empty-induction recovery (robustness fix)** — a reply that parses + validates but sanitizes to an
  EMPTY / below-`min_labels` label set is NOT silently accepted. That quiet degrade to the bare
  config floor (`[person, organization]`) is precisely the quality cliff induce exists to fix — the
  whole point of induce is the domain schema, so an empty induction is the noisy-output failure, not
  a valid answer. The Provider **re-prompts ONCE** (the `EMPTY_RECOVERY_INSTRUCTION`, at a doubled
  budget — the same "double the budget on retry" precedent as the truncation net) before falling
  back, and only a still-empty result falls back — **LOUDLY**
  (`StageMetrics.induction_empty_after_sanitization` + a WARNING), never a silent revert. `min_labels`
  (default 1) is configurable: raise it to demand a richer schema before an induction is accepted.
- **Fail-open, never crash (G1)** — ANY failure (network, auth, malformed, truncated-twice,
  schema-invalid, empty-after-recovery, unreachable endpoint) leaves the record's `induced_labels`
  `None` so the extractors use their config labels and the run continues; the failure is a typed,
  no-content log + a Quality-Report note (`StageMetrics.induction_fallbacks` / `induction_documents`,
  plus `induction_empty_after_sanitization` for the empty case, folded via the same defensive
  attribute seam as `endpoint_retries` — ADR-0004, no base class).

### 6. Deterministic + overridable prompts

`temperature` 0 and stable label order (sanitization preserves first-seen order) so a seeded/mocked
run is byte-reproducible (Baseline bar). A strong default system + user prompt (engineered for the
exact JSON shape, JSON-only, evidence-only) ships fresh here and is **overridable** via config
(`system_prompt` / `user_prompt_template` with `{text}` / `{max_labels}` placeholders) so an
operator's own prompt engineering (or the superpod prompts, when provided) drops in.

## Consequences

- The "point it at a folder and it discovers its own schema" property is restored, opt-in and
  additive: `stacks/schema-induction.yaml` (`auto_generate_labels: true`) turns it on; every other
  stack is byte-identical. The roster is now **38** Providers across 16 Capabilities.
- The extractors gain zero risk: their logic is untouched, and with no induced labels they behave
  exactly as before (the union returns the config labels unchanged).
- Real `gpt-4.1-mini` numbers are the maintainer's keyed run
  (`scripts/validate-schema-induction-on-gpu.md`) — the offline mock proves the machinery + every
  safety net, and no quality number is fabricated for a keyed/skipped candidate.
- Open follow-ups: a second inducer Provider (a different LLM/prompt) drops in as a second bake-off
  row with no harness change; the induced `confidences` map is carried but not yet consumed
  downstream; the superpod prompts swap into the default when provided.

## Status note — canonical induced labels + corpus schema + redaction.gliner2 wiring (W12-label-canon)

Real-GPU-E2E found induced labels **fragmenting** corpus-wide (case + format + synonym), plus two
smaller gaps. Wave-12 closes all four, additive and backward-compatible:

1. **Canonical-form induced labels (the case/format fragmentation fix).** `sanitize_label`
   (`latence_schema_inducer.sanitize`) now emits ONE **canonical normal form** per label: strip
   controls, treat `_`/`-`/whitespace as equivalent separators collapsing to a single space, then
   `casefold()`. So `Date`==`date`, `Website`==`website`, `payment_product`==`payment-product`==
   `payment product` — every variant maps to the same string. It is a PURE per-string transform (no
   cross-call state), so it holds identically under per-doc / streaming / corpus induction. Dedup is
   on that canonical form. Defense-in-depth: `latence_core.induced.effective_labels` now dedupes the
   config∪induced union **case-insensitively** (config surface form kept, since config comes first),
   and its previously-misleading docstring (which claimed the inducer already case-normalized) is
   corrected.
2. **`granularity: "corpus"` (the synonym fragmentation fix — the structural one).** `LLMLabelInducer`
   gains a third granularity beside `doc`/`chunk`: a clean **collect→broadcast** two-phase over the
   existing async fan-out — induce per document (all W4 safety-nets intact: fail-open per doc,
   temperature-0, index-keyed reassembly), MERGE every document's induced entity/relation/pii sets
   into ONE canonical, **sorted**, deduped, count-capped schema, then STAMP that single shared schema
   on **every** chunk (including chunks of a document whose own induction failed open). One coherent
   vocabulary for the whole corpus → entity types stay consistent across the KG, no per-chunk/per-doc
   drift. The sort makes the corpus schema a byte-stable function of the label SET (seed-stable). It
   needs a full-corpus pass (documented; acceptable for the small-LLM-call induction Stage).
3. **`redaction.gliner2` honors induced `pii_types`.** The GLiNER2 Redactor resolved detection labels
   from the config `pii_labels` floor only, ignoring induced `pii_types` — unlike its twin
   `redaction.gliner_pii`. It now routes through the SAME `effective_labels(config, induced, PII)`
   seam (per-document, resolved in `_prepare`), and the batched flush **groups detect-needing docs by
   their effective label set** (one label set per `batch_extract_entities` call). The config floor
   stays mandatory (`pii_labels` validated non-empty); induction only EXTENDS it; masking / offsets /
   counts-only are unchanged. Absent induction every doc shares the config set → one group →
   byte-identical to before.

Tests: `sanitize` canonical-collapse (`Date`/`date`/`payment_product`/`payment product`→2);
`effective_labels` case-insensitive union (config `Bank` + induced `bank`→one); corpus granularity
(two docs' differing sets → one shared sorted-canonical schema on all chunks; a failed-LLM doc still
gets the corpus schema; byte-stable two runs); `redaction.gliner2` induced `pii_types` reach the
effective set (offline-fake equivalence + a no-induction control that proves the seam is load-
bearing). `induced_labels` `SCHEMA_VERSION` is unchanged (a pure per-string normal form + a new
granularity mode, no record-shape change). The clean-corpus KG-vocabulary numbers are the pod
maintainer's re-run (`granularity: corpus`), never fabricated here.

## Status note — empty-after-sanitization robustness (induce-robustness)

A live run on a real corpus logged `schema_induction fell back to config labels: empty after
sanitization` — the LLM returned output that sanitized to an EMPTY label set, so induce **silently**
reverted to the bare config floor (`[person, organization]`). On the same run's 2-document case
induce produced a rich domain schema, so this is an **intermittent robustness gap**, not a total
failure — but each silent revert is a quality cliff (a thin, generic KG — exactly the noisy-output
problem induce exists to fix), hidden behind a single log line. **Root cause:** an all-empty (or
below-`min_labels`) sanitized result was treated as a terminal outcome and quietly accepted as a
fall-back. Three additive fixes, backward-compatible:

1. **Empty-induction recovery, not silent accept.** When sanitization yields an empty / below-min
   label set the Provider now **re-prompts once** (`EMPTY_RECOVERY_INSTRUCTION`, at a doubled budget
   — the same precedent as the truncation retry) before falling back. A recovered induction is used;
   only a still-empty result falls back. The re-prompt does not coerce the model into fabricating
   types — a genuinely typeless document survives still-empty and falls back honestly.
2. **Wrapper-tolerant parse.** The parser now recovers a JSON object the model wrapped in a Markdown
   fence or in surrounding prose (peel the wrapper, re-parse with the SAME strict `json.loads`), so
   valid labels buried in chatter are recovered instead of discarded. This never loosens parsing —
   genuine garbage (single-quoted pseudo-JSON, prose with no braces) still fails. (Investigation also
   confirmed the sanitizer does NOT over-strip valid non-ASCII/German labels — only control/format
   chars are removed — so a German-only schema like `Rechnungsnummer` was never the cause; a
   regression test now locks that in.)
3. **Loud telemetry.** The empty case surfaces on its own Quality-Report field
   (`StageMetrics.induction_empty_after_sanitization`, the loud subset of `induction_fallbacks`) plus
   a WARNING, so an operator SEES "N documents induced no domain schema" directly rather than a
   silent degrade.

Tests (`test_label_inducer_provider.py`): `test_empty_induction_recovers_via_reprompt` (empty → real
schema on the recovery re-prompt, no fallback); `test_empty_induction_persists_falls_back_loudly`
(empty twice → fall back BUT with the telemetry flag, bounded to one re-prompt);
`test_prose_wrapped_json_recovers_without_fallback` + `test_fenced_code_block_json_recovers_without_fallback`
(wrapper-tolerant parse recovers on the first call); `test_german_unicode_labels_are_not_dropped_to_empty`
(non-ASCII preserved); `test_below_min_labels_triggers_recovery` (`min_labels` bar);
`test_empty_recovery_is_byte_identical_two_runs` (determinism preserved). `induced_labels`
`SCHEMA_VERSION` is unchanged (no record-shape change — a new retry path, a new `StageMetrics` field,
and a more tolerant parser).

### Tokenizer-mismatch caveat (W12-gliner-window)

The `schema-induction.yaml` stack feeds the gliner-family extractors, which run on mdeberta-v3 (a
768-token window). `chunk.markdown`'s `max_tokens` is counted by the CHUNKER's tokenizer, which
differs from mdeberta by **~1.2×** — a chunk that fits the chunker cap can still overflow the model
window and be SILENTLY truncated (tail entities/relations/PII lost, offsets past the window
unreliable). Keep the chunk cap ≤ ~640 chunker-tokens (≈ ≤768 mdeberta-tokens); the per-Provider
`max_len` guard (default 768, ADR-0036 W12-gliner-window note) is the hard backstop. The induced
label schema itself is unaffected (it is derived by the LLM inducer, not the mdeberta extractors),
but the CONSUMERS of the induced labels honor the same window as their config-label path.

# A universal financial-PII redaction floor, and an optional mask-before-extract mode

**Status: accepted** — W18-redaction-polish. Two small, focused improvements on top of W16's
chunk-level redaction (ADR-0042): (1) a UNIVERSAL financial-PII floor enforced at the single shared
redaction seam so it holds regardless of which detection backend a stack configures, and (2) an
OPTIONAL `extract_on: masked` mode that feeds extraction the redacted chunk stream so PII never
reaches the knowledge graph. Builds on ADR-0004 (Capability protocols + Provider plugins), ADR-0007
(CPU-first, zero-dep core), ADR-0024 (the S6 policy/masking machinery + the H-C1 §4 sensitive-doc
no-op floor), ADR-0038 (`config ∪ induced` labels), ADR-0041 (guided setup wizard), and ADR-0042
(redaction is a chunk→chunk transform through `plan_chunk_redaction` / `finalize_redacted_chunk`).

## Context

**1. The floor was not universal.** W16 moved all four `PIIDetector` Providers
(`redaction.hybrid_rule`, `.gliner_pii`, `.gliner2`, `.presidio`) onto the shared chunk seam, but the
financial-PII *coverage* was still per-Provider: only the in-core `redaction.hybrid_rule` recognises
`credit_card` / `iban` / `ssn` via its built-in regex library. The three learned Providers inherit
only their model's coverage, and the gliner-family were deliberately kept without the H-C1 §4 no-op
floor wiring (they pass `sensitive_set=set()`). The consequence: a learned redactor whose model
*misses* an IBAN/SSN/card — because the checkpoint is weak on that type, or the config `pii_labels`
/ policy `types` narrowed it away, or the induced schema simply did not surface it — emits a
`masked_content` that OMITS a high-harm financial identifier. It leaks into the exported RAG corpus.
This is a real PII-safety gap, and it is exactly the kind of guarantee that must not depend on which
model an operator happened to configure.

**2. Extraction always saw unmasked text.** W16's blessed dataflow has extraction read the UNMASKED
chunk `content` (so the KG keeps the real entities) while redaction masks only the exported corpus.
That is the right default — but some enterprises require the stricter posture where PII never reaches
extraction or the KG at all. There was no way to ask for it.

## Decisions

### 1. A universal financial-PII floor at the single shared seam

`latence_core.redaction_policy` gains a deterministic, zero-dep floor: a fixed
`FINANCIAL_FLOOR_PATTERNS` map for the high-harm subset (`ssn` incl. the digits-only form,
`credit_card`, `iban`) whose regexes are **byte-identical to the `redaction.hybrid_rule` built-ins**
(a drift-guard test pins them equal). `finalize_redacted_chunk` — the ONE tail every Provider already
calls — scans the chunk `content` for these patterns and UNIONs the matches into the Provider's
`accepted` spans before masking. Because all four Providers funnel through this seam, the floor is:

* **Non-bypassable by model/backend choice** — a Provider that detected nothing still emits
  `masked_content` with the floor types masked. Proven by a fake gliner/gliner2/presidio redactor
  that detects NOTHING yet whose output has the IBAN masked (`detector == "floor:iban"`), driven
  through each of the four Providers' chunk path.
* **Impossible to forget per-Provider again** — it lives at the choke point, not in four detection
  halves. A future fifth Provider inherits it for free.

Overlap resolution matches the hybrid_rule policy: a longer span wins first (max coverage), and on an
identical span the **Provider wins the tie** so its richer detector/label provenance is kept — a
floor span survives only where the Provider left a gap. When the two sets coincide (e.g.
`redaction.hybrid_rule`, whose built-ins ARE these patterns) the Provider wins every span, so a
hybrid_rule stack is **byte-identical**.

**Scope (deliberately narrow + honest, H-C1).** The floor guarantees *detection* of the financial
subset regardless of backend; it does NOT override an operator's explicit policy `action` or `skip`.
A floor span is handled with the chunk's resolved policy action (mask/replace/hash/tag), and the
floor runs ONLY when redaction actually runs on the chunk (`plan.needs_detection` — an explicit
`skip:true` or a near-empty chunk still skips). This mirrors the H-C1 §4 sensitive-doc floor
philosophy: the disposition is guaranteed-visible, operator intent is preserved. It is a floor for the
*financial* subset, not a mandate that all PII is always masked.

### 2. An optional `extract_on: masked | unmasked` mode (default `unmasked`)

The `Pipeline` gains an `ExtractOn` field, default `unmasked`:

* `unmasked` (default) — extraction reads the raw chunk `content`; the KG keeps the real entities.
  **Byte-identical to pre-W18** (a golden/determinism test proves a stack with no `extract_on` field
  and one with `extract_on: unmasked` produce identical artifacts).
* `masked` — the Runner feeds the extraction Stages (`entity_extraction`, `relation_extraction`,
  `fused_entity_relation`) the REDACTED chunk stream: each chunk's `content` is substituted with its
  `masked_content` before the extractor reads it, so PII never reaches extraction or the KG.

The mode is a genuine behaviour switch, so the input source of extraction becomes mode-dependent: the
extraction Stages' `depends_on` must run them AFTER a redaction Stage (so `masked_content` exists). A
`Pipeline` validator enforces this — `extract_on: masked` with no redaction Stage, or an extraction
Stage that does not transitively depend on one, is a loud structural error, never a silent fallback to
unmasked text (which would be a privacy hole). The substitution is the ONLY behavioural change; the
default path is untouched.

### 3. Surfaced in the setup wizard, off by default

`latence setup` (W15) gains an advanced `--extract-on` flag + interactive prompt (shown only when PII
is on), defaulting to `unmasked`. When `masked` is chosen the generator wires the `extract` Stage's
`depends_on` to `redact` and emits `extract_on: masked` in the stack; when `unmasked` (default) the
generated config is byte-identical to pre-W18 (the field is omitted entirely). `masked` requires PII
redaction ON (there is no redact Stage otherwise) — a clear error, not a broken config.

## Consequences

* **A financial identifier can no longer leak because a model missed it** — the floor is a
  deterministic safety net at the seam, independent of the configured backend.
* **Byte-identical where it matters** — a hybrid_rule stack (floor overlaps every Provider span) and
  every `extract_on: unmasked` stack (the default) are unchanged; only a backend that MISSED a
  financial type now masks it, and only an opt-in `masked` stack changes extraction's input.
* **The stricter privacy posture is one toggle away** — enterprises that require PII to never touch
  the KG set `extract_on: masked` (with the wizard or by hand) and the wiring/validator do the rest.
* **A small per-chunk cost** — four linear-time regexes over each chunk's `content` at the seam,
  negligible against the model forward pass; the floor short-circuits to the unchanged path when a
  chunk carries no financial PII (the common case).
* **Documented trade-off (masked mode):** mentions extracted from masked text carry offsets into the
  masked variant, so their source-page resolution is approximate (placeholders differ in length from
  the original). This is acceptable for the opt-in strict-privacy posture and is why `unmasked`
  remains the default.
* **Follow-ups:** the floor set could grow (e.g. `tax id`, `bank account number`) behind the same
  drift guard; and a masked-mode-aware offset remap could restore exact page resolution if a consumer
  needs it.

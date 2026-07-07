# Redaction is one RedactionRecord per document (masked text + auditable PII spans), keyed off Classification.sensitivity, with policy+masking shared across Providers

S6 adds **Redaction** (CONTEXT): the Stage that detects PII and produces
masked/replaced variants of the text. Three design questions had non-obvious answers;
they mirror how ADR-0022/0023 handled S4/S5, and extend ADR-0004/0007 (the
Capability + CPU-first-reference pattern) with the concrete Redaction contract, the
metadata-aware policy seam, and where the shared machinery lives.

## A RedactionRecord is a per-document inter-Stage Record: masked text + PII spans

Redaction could annotate each `ChunkRecord` in place (a masked-content field) or emit
one span record per detected PII item. It does neither. `RedactionRecord` is its own
versioned core contract (`SCHEMA_VERSION` → 6), a subclass of `Record` — so it carries
Provenance and Classification like every other inter-Stage carrier, threads through the
DAG as a Stage output, is checkpointed/resumed uniformly, and exports to the corpus
directly — and it is **one record per document**, carrying both the masked/replaced
text variant (`masked_content`) and the structured list of every detected `PIISpan`.

**Why per-document, not per-chunk or per-span.** The deliverable of Redaction is the
*PII-handled RAG-corpus variant of the document* (ADR-0017): a consumer wants the whole
masked text, not a bag of chunk fragments it must re-stitch, nor a span list with no
text. Emitting the masked document once also lets a PII value that spans a chunk
boundary be masked coherently (the Provider reconstructs the document text from its
chunks' document-level spans, S3, exactly as Relation Extraction does). The per-item
detail is preserved as the `pii_spans` list *inside* that record — so the audit is
co-located with the artifact it describes. Rejected: masked-content-as-a-field-on-the-
chunk (couples every consumer to the chunk shape, cannot express a cross-chunk PII
span, and scatters one document's redaction across many records) and a record-per-span
(loses the masked text, the actual deliverable).

The `PIISpan` records the PII **type**, its **document-offset** char span (the same
coordinate system every other Stage uses, so it resolves back through the parent page
map to its source page — the S6 round-trip), the **action** taken, the **placeholder**
written into the masked variant, and the **detector** that fired (`regex:email`,
`gazetteer:name`, `gliner-pii`) so a hybrid Provider's rule-vs-model provenance is
auditable. Critically, a `PIISpan` has **no field for the raw PII value** — the
surface string is used only transiently to derive a deterministic surrogate and is
never persisted. The contract *enforces* ordered, non-overlapping spans and a
placeholder on every non-`TAG` span, so a masked variant is always reconstructable and
the audit is unambiguous.

## Redaction is metadata-aware: policy keys off Classification.sensitivity

The S6 acceptance criterion requires the policy to key off `Classification.sensitivity`.
Concretely, a Provider takes a `policies` map: per sensitivity value (`public`,
`internal`, …, or a `default` fallback) it sets the `RedactionAction` to take (mask /
replace / hash / tag), an optional PII-type allow-list, and a `skip` flag. So the same
Pipeline masks a `public` document hard, only tags an `internal` one for review, and
disables redaction for `confidential` — driven purely by the document's Classification,
with no code change. The four actions are a deliberate spread: `mask` (lossy,
irreversible `[EMAIL]`), `replace` (a deterministic salted pseudonym so text stays
readable), `hash` (a stable salted digest, joinable across the corpus without exposing
the value), and `tag` (detect-only, surface text intact) — covering the real
enterprise policies (hard-strip vs pseudonymise vs flag-for-review) rather than only
one. The `RedactionRecord` records which policy name was applied, so the disposition is
auditable end to end.

## Policy + masking are framework-owned (shared) machinery; a Provider owns only detection

A Redaction Provider varies in exactly one thing: *where is the PII*. Everything else —
compiling the sensitivity-keyed policy, deriving the deterministic placeholder,
rewriting the text right-to-left, stamping an auditable span with its resolved page — is
identical for a rule Provider and a learned one. So that machinery lives once, in the
thin core (`latence_core.redaction_policy`: `RedactionPolicy`, `placeholder_for`,
`apply_masking`, `build_span`), and both Providers call it. This keeps the two from
drifting (a masked span is byte-identical whether it came from a regex or the model) and
keeps each Provider small enough that its only responsibility — detection — is obvious.
It is the same "framework owns the seam, the Provider owns the model" split as the rest
of the spine (ADR-0004).

## Redaction scans the parsed document, not the chunk stream

Redaction depends on **Parse** (the `DocumentRecord`), not **Chunk**. The masked
variant, every `PIISpan` offset, and the page resolution all live in the document's
*original-markdown* coordinate system — the one the parent page map and the S6
round-trip use. Chunk text is markup-stripped (`chunking.strip_markup`, 18 patterns)
with its `[char_start, char_end)` remapped back to the original document, so a chunk's
`content` length no longer equals its span width. Reconstructing document text by
writing stripped chunk content at original offsets therefore fabricates a *different*
string in a broken coordinate system: PII detected at a reconstructed offset would be
recorded at the wrong `char_start` and resolve to the wrong page. Scanning
`DocumentRecord.content` directly is exact and faithful — the masked variant IS the
document text Export writes to the corpus, and each span round-trips through the page map
to its true source page. (A `PARSE_ERROR` document has no usable content and is skipped,
mirroring the Chunk Stage.)

## Providers shipped (CPU-first reference, ADR-0007; licenses verified 2026-07-06)

* **`redaction.hybrid_rule`** ships in `latence-core`: a pure-Python, dependency-free,
  deterministic **hybrid-rule** PII detector — a library of well-known-PII and secret
  regexes (email / ssn / credit_card / phone / ip / iban / aws_key / api_key / jwt /
  secret — the secret/API-key recognisers and comprehensive-by-default coverage added
  for #61, and the compound-secret + structural-JWT recall extension for #66) plus a
  config gazetteer and custom patterns. It is the "rule half" the S6 issue names; shipping it in core keeps first
  `git clone` → working redaction free of torch and hash-stable for the Baseline bar.
  It is not a weaker reinvention of a learned model — the learned Provider is the
  drop-in below.
* **`redaction.gliner_pii`** (`latence-pii-gliner`): the learned "model half" — a
  zero-shot GLiNER-PII detector over `urchade/gliner_multi_pii-v1` (weights Apache-2.0,
  base `urchade/gliner_multi-v2.1` Apache-2.0, `gliner` code Apache-2.0), behind the
  *same* `PIIDetector` seam, so a Pipeline swaps `provider: redaction.hybrid_rule` for
  `provider: redaction.gliner_pii` with no code change. Like the other gliner/endpoint
  Providers it keeps its heavy dep out of the thin core (ADR-0016): a workspace member,
  type-checked from source, exercised with `gliner` monkeypatched by a fake in tests,
  never installed into the deterministic dev/test env.

The PII detection counts land in the Quality Report as `RedactionQuality` (per-type +
per-action counts, documents-with-PII) — counts and type/action names only, **never a
raw PII value**, satisfying the S6 "counts by type in the Quality Report; no raw PII in
logs" criterion.
